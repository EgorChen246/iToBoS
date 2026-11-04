"""Original single-class AP50 (101-point), precision/recall and F1/F2 evaluator."""
import numpy as np

def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def evaluate(predictions, gt, threshold=0.03, iou_match=0.5):
    total_gt = sum(map(len, gt.values()))
    if not gt:
        raise ValueError('No evaluated images')
    if not 0 <= threshold <= 1 or not 0 < iou_match <= 1:
        raise ValueError('Invalid threshold or IoU')

    scored = []
    for key in gt:
        for item in predictions.get(key, []):
            box = np.asarray(item['box'], dtype=float)
            conf = float(item['score'])
            if box.shape == (4,) and np.isfinite(box).all() and np.isfinite(conf):
                scored.append((conf, key, box))
    scored.sort(key=lambda x: -x[0])

    matched = {key: set() for key in gt}
    tps, fps, confidences, overlaps_tp = [], [], [], []
    for confidence, key, box in scored:
        best_id, best_iou = -1, iou_match
        for i, true_box in enumerate(gt[key]):
            if i in matched[key]:
                continue
            overlap = box_iou(box, true_box)
            if overlap >= best_iou:
                best_id, best_iou = i, overlap
        is_tp = best_id >= 0
        if is_tp:
            matched[key].add(best_id)
        tps.append(int(is_tp))
        fps.append(int(not is_tp))
        confidences.append(confidence)
        overlaps_tp.append(best_iou if is_tp else 0.0)

    cumulative_tp = np.cumsum(tps)
    cumulative_fp = np.cumsum(fps)
    recall_curve = cumulative_tp / max(total_gt, 1)
    precision_curve = cumulative_tp / np.maximum(1, cumulative_tp + cumulative_fp)
    ap = float(np.mean([max(precision_curve[recall_curve >= r], default=0.0) for r in np.linspace(0, 1, 101)]))

    above = np.asarray(confidences) >= threshold
    tp = int(np.asarray(tps)[above].sum())
    fp = int(np.asarray(fps)[above].sum())
    fn = total_gt - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(total_gt, 1)
    f1 = 2*p*r/(p+r) if p+r else 0.
    f2 = 5*p*r/(4*p+r) if 4*p+r else 0.
    ious = np.asarray(overlaps_tp)[above]
    tp_flags = np.asarray(tps)[above].astype(bool)
    mean_tp_iou = float(ious[tp_flags].mean()) if tp_flags.any() else None
    counts = {key: 0 for key in gt}
    for conf, key, box in scored:
        if conf >= threshold: counts[key] += 1
    per_image = [{'image':key,'gt_count':len(gt[key]),'pred_count':counts[key],
                  'absolute_count_error':abs(counts[key]-len(gt[key])),
                  'count_exact':int(counts[key]==len(gt[key]))} for key in gt]
    errors = [x['absolute_count_error'] for x in per_image]
    exact = sum(x['count_exact'] for x in per_image)
    return {'mAP50':ap if total_gt else None, 'Precision':p, 'Recall':r, 'F1':f1,
            'F2':f2, 'TP':tp, 'FP':fp, 'FN':fn,
            'MeanIoU_TP':mean_tp_iou,
            'IoU_TP_sum_per_GT':float(ious.sum()/total_gt) if total_gt else None,
            'CountMAE':float(np.mean(errors)),
            'CountExactAccuracy':exact/len(gt),'CountExactImages':exact,
            'EvaluatedImages':len(gt),'CountBias':float(np.mean([x['pred_count']-x['gt_count'] for x in per_image]))}

def count_details(predictions, gt, threshold=0.03):
    rows=[]
    for key, boxes in gt.items():
        count=sum(1 for item in predictions.get(key,[]) if float(item['score'])>=threshold
                  and np.asarray(item['box']).shape==(4,)
                  and np.isfinite(np.asarray(item['box'],dtype=float)).all()
                  and np.isfinite(float(item['score'])))
        rows.append({'image':key,'gt_count':len(boxes),'pred_count':count,
                     'absolute_count_error':abs(count-len(boxes)),
                     'count_exact':int(count==len(boxes))})
    return rows
