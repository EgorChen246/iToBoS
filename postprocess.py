"""Optional single-class NMS. Choose its parameters on validation data only."""
import math
from evaluate import box_iou


def suppress_duplicates(predictions, iou_threshold=0.5):
    """Keep highest-scoring boxes; suppress only IoU strictly above the limit.

    Close but separate objects may overlap, so NMS is optional and must be
    evaluated against the original detections. Inputs are never modified.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError('NMS IoU must be between 0 and 1')
    output = {}
    for image, items in predictions.items():
        valid = []
        for item in items:
            box = item['box']
            if (len(box) == 4 and all(math.isfinite(float(x)) for x in box)
                    and math.isfinite(float(item['score']))
                    and box[2] > box[0] and box[3] > box[1]):
                valid.append(item)
        kept = []
        for item in sorted(valid, key=lambda x: -float(x['score'])):
            if all(box_iou(item['box'], other['box']) <= iou_threshold for other in kept):
                kept.append(dict(box=list(item['box']), score=float(item['score'])))
        output[image] = kept
    return output
