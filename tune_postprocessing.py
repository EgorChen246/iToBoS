"""Select confidence + optional NMS from saved VALIDATION predictions only."""
import argparse
import json
from pathlib import Path
from data import image_paths,load_ground_truth
from evaluate import evaluate
from postprocess import suppress_duplicates


def tune(predictions, ground_truth):
    rows=[]
    for nms_iou in (None,.3,.5,.7,.9):
        processed=predictions if nms_iou is None else suppress_duplicates(predictions,nms_iou)
        for step in range(1,100):
            confidence=step/100
            metrics=evaluate(processed,ground_truth,threshold=confidence,iou_match=.5)
            rows.append({'threshold':confidence,'nms_iou':nms_iou,**metrics})
    # Preserve the original pipeline on exact F1 ties.
    best=max(rows,key=lambda x:(x['F1'],x['nms_iou'] is None,x['Precision'],x['threshold']))
    return best, rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/postprocessing_selection.json'))
    args=parser.parse_args()
    payload=json.loads(args.predictions.read_text(encoding='utf-8-sig'))
    if payload.get('split')!='val':parser.error('Only validation predictions are allowed; never tune on test')
    if payload.get('confidence_floor',1)>.01:parser.error('Predictions must be saved with confidence_floor <= 0.01')
    names=payload.get('images',[])
    if len(names)!=len(set(names)):parser.error('Duplicate image names in predictions')
    paths=image_paths('val')
    if set(names)!={p.name for p in paths}:parser.error('Predictions must cover the full validation split')
    if set(payload['predictions'])!=set(names):parser.error('Missing or extra prediction image keys')
    gt=load_ground_truth('val',paths)
    best,_=tune(payload['predictions'],gt)
    result={'model':payload['model'],'split':'val','images':len(paths),
            'objective':'F1','matching_iou':.5,'test_used':False,'best':best,
            'note':'Selection score is validation only. Does not replace baseline test metrics.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
