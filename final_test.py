import sys,json,time,hashlib,statistics,platform
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from data import image_paths,load_ground_truth
from models import predict,CHECKPOINTS
from evaluate import evaluate,count_details
out=root/'results'/'final_test'
out.mkdir(exist_ok=True)
paths=image_paths('test');gt=load_ground_truth('test',paths)
thresholds={'yolo26s':.11,'fasterrcnn':.74,'deformable_detr':.43}
summary=[]
for name,threshold in thresholds.items():
    print('START',name,flush=True)
    timing={}
    preds=predict(name,paths,device='cpu',confidence=threshold,timing=timing)
    stats=evaluate(preds,gt,threshold=threshold,iou_match=.5)
    row=dict(model=name,threshold=threshold,**stats,timing=timing)
    summary.append(row)
    (out/(name+'.json')).write_text(json.dumps(dict(predictions=preds,metrics=row,counts=count_details(preds,gt,threshold)),indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('DONE',json.dumps(row),flush=True)
(out/'provenance.json').write_text(json.dumps(dict(split='test',images=len(paths),gt_boxes=sum(map(len,gt.values())),threshold_source='Validation F1 grid 0.01 to 0.99; fixed before this test',device='cpu',python=platform.python_version(),data={p.name:dict(image=hashlib.sha256(p.read_bytes()).hexdigest(),label=hashlib.sha256((root/'data/test/labels'/(p.stem+'.txt')).read_bytes()).hexdigest()) for p in paths}),indent=2),encoding='utf-8')
