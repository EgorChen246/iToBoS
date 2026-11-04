"""Local image and YOLO ground-truth loading; paths are independent of cwd."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def image_paths(split='val'):
    if split not in {'train','val','test'}:
        raise ValueError('split must be train, val or test')
    folder = DATA / split / 'images'
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)

def load_ground_truth(split='val', paths=None):
    """Return {image filename: xyxy boxes in pixels}. Missing labels are errors."""
    from PIL import Image
    result = {}
    for path in image_paths(split) if paths is None else paths:
        label = DATA / split / 'labels' / (path.stem + '.txt')
        if not label.is_file():
            raise FileNotFoundError(f'Missing ground truth: {label}')
        with Image.open(path) as img:
            width, height = img.size
        boxes = []
        for line in label.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            values = list(map(float, line.split()))
            if len(values) != 5 or values[0] != 0:
                raise ValueError(f'Expected single-class YOLO detection labels: {label}')
            _, x, y, w, h = values
            if not all(0 <= v <= 1 for v in values[1:]) or w <= 0 or h <= 0:
                raise ValueError(f'Invalid normalized box: {label}')
            boxes.append([(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])
        result[path.name] = boxes
    return result

def inventory():
    report = {}
    for split in ('train','val','test'):
        paths = image_paths(split)
        missing = [p.name for p in paths if not (DATA/split/'labels'/(p.stem+'.txt')).is_file()]
        report[split] = {'images':len(paths),'missing_labels':missing}
    return report
