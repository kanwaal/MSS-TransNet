"""Collect only image paths for optional SSL; no labels or masks are exported."""
from pathlib import Path
import argparse
import csv

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labeled-manifest', type=Path, action='append', default=[])
    p.add_argument('--unlabeled-root', type=Path, action='append', default=[])
    p.add_argument('--split', choices=['train','all'], default='train')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    paths = set()
    for manifest in args.labeled_manifest:
        with manifest.open(newline='') as f:
            for row in csv.DictReader(f):
                if args.split == 'all' or row['split'] == 'train':
                    paths.add(Path(row['image_path']).resolve())
    for root in args.unlabeled_root:
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths.update(p.resolve() for p in root.glob('case*.png')
                     if len(p.stem) == 7 and p.stem[4:].isdigit())
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('No valid images found, or an image path is missing')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path'])
        writer.writeheader()
        writer.writerows({'image_path':str(p)} for p in sorted(paths))
    print(f'Wrote {len(paths)} unlabeled image paths to {args.output}')
    if args.split == 'all':
        print('All splits enter SSL. Evaluation is transductive if test images are included.')

if __name__ == '__main__':
    main()
