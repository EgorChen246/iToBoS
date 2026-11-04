"""Restore the three archived architectures. Imports happen only on demand."""
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {'yolo26s':ROOT/'models'/'yolo26s.pt',
               'yolov8s':ROOT/'models'/'yolov8s.pt',
               'dfine_s':ROOT/'models'/'dfine_s',
               'fasterrcnn':ROOT/'models'/'fasterrcnn.pth',
               'deformable_detr':ROOT/'models'/'deformable_detr'/'best_model.pt'}

def load_model(name, device='cpu'):
    path = CHECKPOINTS[name]
    if not path.exists(): raise FileNotFoundError(path)
    if name.startswith('yolo'):
        from ultralytics import YOLO
        return YOLO(str(path))
    if name == 'fasterrcnn':
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        model = fasterrcnn_resnet50_fpn(weights=None,weights_backbone=None)
        classes = checkpoint['model']['roi_heads.box_predictor.cls_score.weight'].shape[0]
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features,classes)
        model.load_state_dict(checkpoint['model'],strict=True)
        model.roi_heads.score_thresh = 0.01
        return model.to(device).eval()
    if name == 'deformable_detr':
        import torch
        from transformers import AutoImageProcessor,DeformableDetrConfig,DeformableDetrForObjectDetection
        folder = path.parent
        config = DeformableDetrConfig.from_pretrained(folder,local_files_only=True)
        config.use_pretrained_backbone = False
        config.disable_custom_kernels = True
        processor = AutoImageProcessor.from_pretrained(folder,use_fast=False,local_files_only=True)
        model = DeformableDetrForObjectDetection(config)
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        state = checkpoint['model']
        # Naming compatibility only: all weights must still match exactly.
        rules = [('model.backbone.model.','model.backbone.conv_encoder.model.'),
                 ('.mlp.fc1.','.fc1.'),('.mlp.fc2.','.fc2.'),
                 ('.self_attn.o_proj.','.self_attn.out_proj.')]
        converted = {}
        for key,value in state.items():
            for old,new in rules: key = key.replace(old,new)
            if key in converted: raise ValueError('Checkpoint key collision')
            converted[key] = value
        model.load_state_dict(converted,strict=True)
        return processor, model.to(device).eval()
    from transformers import AutoImageProcessor, DFineForObjectDetection
    processor = AutoImageProcessor.from_pretrained(path, use_fast=False, local_files_only=True)
    model = DFineForObjectDetection.from_pretrained(path, local_files_only=True).to(device).eval()
    return processor, model

def predict(name, paths, device='cpu', imgsz=896, confidence=0.01, nms_iou=0.5, timing=None):
    load_start = perf_counter()
    model = load_model(name,device)
    if timing is not None:
        timing.update(load_seconds=perf_counter()-load_start,per_image_seconds=[],device=str(device),imgsz=imgsz)
    predictions = {}
    if name.startswith('yolo'):
        for path in paths:
            start = perf_counter()
            r = model.predict(str(path),device=device,imgsz=imgsz,conf=confidence,
                              iou=nms_iou,max_det=500,verbose=False)[0]
            predictions[path.name] = [dict(box=b,score=s) for b,s in
                zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist())]
            if timing is not None: timing['per_image_seconds'].append(perf_counter()-start)
    elif name == 'fasterrcnn':
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        with torch.inference_mode():
            for path in paths:
                start = perf_counter()
                with Image.open(path) as source: img = source.convert('RGB')
                r = model([to_tensor(img).to(device)])[0]
                predictions[path.name] = [dict(box=b,score=s) for b,s,label in
                    zip(r['boxes'].cpu().tolist(),r['scores'].cpu().tolist(),r['labels'].cpu().tolist())
                    if label == 1 and s >= confidence]
                if timing is not None: timing['per_image_seconds'].append(perf_counter()-start)
    else:
        import torch
        from PIL import Image
        processor, detector = model
        with torch.inference_mode():
            for path in paths:
                start = perf_counter()
                with Image.open(path) as source: img = source.convert('RGB')
                batch = {k:v.to(device) for k,v in processor(images=img,return_tensors='pt').items()}
                output = detector(**batch)
                r = processor.post_process_object_detection(output,
                    target_sizes=[(img.height,img.width)],threshold=confidence)[0]
                predictions[path.name] = [dict(box=b,score=s) for b,s in
                    zip(r['boxes'].cpu().tolist(),r['scores'].cpu().tolist())]
                if timing is not None: timing['per_image_seconds'].append(perf_counter()-start)
    if timing is not None and timing['per_image_seconds']:
        import statistics
        times=timing['per_image_seconds']
        timing.update(first_request_seconds=times[0],mean_request_seconds=statistics.mean(times),
                      warm_mean_request_seconds=statistics.mean(times[1:]) if len(times)>1 else None,
                      warm_median_request_seconds=statistics.median(times[1:]) if len(times)>1 else None,
                      scope='File reading + preprocessing + forward + postprocessing; no network or model loading')
    return predictions
