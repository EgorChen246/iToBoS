"""CLI: inspect the project, run inference, or evaluate saved detections."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from data import ROOT, image_paths, inventory, load_ground_truth
from models import CHECKPOINTS

def main():
    parser=argparse.ArgumentParser(description='iToBoS Skin Lesion Detection')
    parser.add_argument('command',choices=['check','predict','evaluate'],nargs='?',default='check')
    parser.add_argument('--model',choices=list(CHECKPOINTS),default='yolo26s')
    parser.add_argument('--split',choices=['train','val','test'],default='val')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--limit',type=int,default=None)
    parser.add_argument('--threshold',type=float,default=0.03)
    parser.add_argument('--predictions',type=Path)
    args=parser.parse_args()
    if args.limit is not None and args.limit < 1: parser.error('--limit must be positive')
    if not 0 <= args.threshold <= 1: parser.error('--threshold must be between 0 and 1')
    if args.command=='check':
        print(json.dumps({'data':inventory(),'models':{k:p.exists() for k,p in CHECKPOINTS.items()}},indent=2))
        return
    paths=image_paths(args.split)
    if args.limit: paths=paths[:args.limit]
    if not paths: raise ValueError('No images in selected split')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.command=='predict':
        from models import predict
        # AP needs low-confidence detections; threshold affects only operating-point metrics.
        timing={}
        predictions=predict(args.model,paths,device=args.device,confidence=min(0.01,args.threshold),timing=timing)
        output=ROOT/'predictions'/f'{args.model}_{args.split}_{stamp}.json'
        payload={'model':args.model,'split':args.split,'images':[p.name for p in paths],
                 'confidence_floor':min(0.01,args.threshold),'predictions':predictions,'timing':timing}
        output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        print(output)
        return
    if not args.predictions: parser.error('evaluate requires --predictions FILE')
    payload=json.loads(args.predictions.read_text(encoding='utf-8'))
    if payload['split'] != args.split: parser.error('Prediction split does not match --split')
    if args.threshold < payload.get('confidence_floor',0):
        parser.error('Threshold is below saved prediction confidence floor')
    names=set(payload['images'])
    # Evaluate exactly the saved image subset, including images without detections.
    paths=[p for p in image_paths(args.split) if p.name in names]
    if {p.name for p in paths} != names: raise ValueError('Saved images not found in dataset')
    from evaluate import evaluate, count_details
    row={'model':payload['model'],'split':args.split,'images':len(paths),'threshold':args.threshold,
         **evaluate(payload['predictions'],load_ground_truth(args.split,paths),threshold=args.threshold)}
    output=ROOT/'results'/f'metrics_{payload["model"]}_{args.split}_{stamp}.csv'
    with output.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    details=output.with_name(output.stem+'_per_image.csv')
    with details.open('w',newline='',encoding='utf-8-sig') as f:
        rows=count_details(payload['predictions'],load_ground_truth(args.split,paths),args.threshold)
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(row,indent=2)); print(output)

if __name__=='__main__': main()
