"""Evaluate restoration quality using paper metrics.

Usage:
    # Without ground truth (kurtosis, brightness, entropy only):
    python scripts/evaluate.py --restored results/ --output metrics.csv

    # With ground truth (adds PSNR, SSIM):
    python scripts/evaluate.py --restored results/ --reference gt/ --output metrics.csv
"""

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)

from src.metrics import compute_metrics


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img)


def load_mask(path: str) -> np.ndarray | None:
    """Load binary mask. 0=text (black), 255=background (white)."""
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('L')
    arr = np.array(img)
    return (arr > 127).astype(np.uint8)


def find_image_pairs(restored_dir: str, reference_dir: str = None):
    """Yield (name, restored_path, ref_path_or_None) tuples."""
    exts = {'.png', '.jpg', '.jpeg'}
    for fname in sorted(os.listdir(restored_dir)):
        if os.path.splitext(fname)[1].lower() not in exts:
            continue
        restored_path = os.path.join(restored_dir, fname)
        ref_path = None
        if reference_dir:
            candidate = os.path.join(reference_dir, fname)
            if os.path.exists(candidate):
                ref_path = candidate
        yield fname, restored_path, ref_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate stain reduction metrics')
    parser.add_argument('--restored', required=True, help='Directory of restored images')
    parser.add_argument('--reference', default=None, help='Directory of ground truth images (optional)')
    parser.add_argument('--mask_dir', default=None, help='Directory of background masks (optional)')
    parser.add_argument('--output', default='metrics.csv', help='Output CSV file path')
    args = parser.parse_args()

    rows = []
    fieldnames = None

    for fname, restored_path, ref_path in find_image_pairs(args.restored, args.reference):
        restored = load_image(restored_path)
        reference = load_image(ref_path) if ref_path else None

        mask = None
        if args.mask_dir:
            mask = load_mask(os.path.join(args.mask_dir, fname))

        metrics = compute_metrics(restored, reference, mask)
        row = {'file': fname, **metrics}
        rows.append(row)

        summary = ', '.join(f'{k}={v:.4f}' for k, v in metrics.items())
        print(f'{fname}: {summary}')

        if fieldnames is None:
            fieldnames = list(row.keys())

    if not rows:
        print('No images found.')
        return

    # Compute means
    mean_row = {'file': 'MEAN'}
    for k in fieldnames[1:]:
        vals = [r[k] for r in rows if k in r]
        mean_row[k] = sum(vals) / len(vals) if vals else float('nan')
    rows.append(mean_row)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved metrics to {args.output}')


if __name__ == '__main__':
    main()
