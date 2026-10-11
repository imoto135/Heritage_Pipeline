"""Record a raw-pixel local npj control; never overwrite historical outputs.

This is a current-code crop control, not a replay of the undocumented old run.
The existing full-page CSC v1 masks are cropped without reclassification.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/npj_dirt_control_20261002'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert not OUT.exists(), 'Use a new output name, never overwrite a control'
    OUT.mkdir(parents=True)
    source = ROOT / 'modules/ddrm_npj_v2'
    snapshot = OUT / 'source_snapshot/ddrm_npj_v2'
    dependencies = {}
    for p in sorted(source.rglob('*.py')):
        if '__pycache__' in p.parts:
            continue
        q = snapshot / p.relative_to(source)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        dependencies[str(p.relative_to(ROOT))] = sha(p)
    original = ROOT / 'data/split_dataset/train/hnsd00000/hnsd001_011.jpg'
    masks = ROOT / 'data/split_dataset_csc_mask/train/hnsd00000/hnsd001_011'
    ckpt = ROOT / 'modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'
    background = ROOT / 'modules/ddrm_npj_v2/background/isemonogatari.png'
    inputs = OUT / 'input'
    inputs.mkdir()
    maskout = OUT / 'masks/hnsd_raw_control'
    maskout.mkdir(parents=True)
    crop = cv2.imread(str(original))[2280:2900, 80:700]
    assert crop.shape == (620, 620, 3)
    assert cv2.imwrite(str(inputs / 'hnsd_raw_control.png'), crop)
    assert np.array_equal(cv2.imread(str(inputs / 'hnsd_raw_control.png')), crop)
    for name in ['02_text_black.png', '03_text_red.png']:
        p = masks / name
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)[2280:2900, 80:700]
        old = ROOT / 'tmp_work/mask_ablation_hnsd/mask_v1/hnsd001_011_crop' / name
        assert np.array_equal(a, cv2.imread(str(old), cv2.IMREAD_GRAYSCALE))
        assert cv2.imwrite(str(maskout / name), a)
        dependencies[str(p.relative_to(ROOT))] = sha(p)
        dependencies[str(old.relative_to(ROOT))] = sha(old)
    for p in [original, ckpt, background, Path(__file__)]:
        dependencies[str(p.relative_to(ROOT))] = sha(p)
    command = [sys.executable, '-B', str(snapshot / 'scripts/pipeline_ddrm.py'),
               '--input_dir', str(inputs), '--output_dir', str(OUT / 'output'),
               '--mask_dir', str(OUT / 'masks'), '--ckpt', str(ckpt),
               '--back_ground', str(background), '--device', 'cuda:0',
               '--patch_size', '128', '--overlap', '32', '--no_wandb']
    manifest = dict(description='Current local npj v2 code with existing CSC v1 masks, unmodified original crop and external paper background; NOT historical replay or whole-page result',
                    original_xywh=[80, 2280, 620, 620], raw_crop_exact=True,
                    masks_exact_to_historical_v1=True, shared_noise=False,
                    seed=999, timesteps=20, masking=True, mask_timesteps=50,
                    dependencies=dependencies, command=command)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    shutil.copyfile(__file__, OUT / 'source_snapshot/run_npj_dirt_control.py')
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with open(OUT / 'run.log', 'w') as f:
        result = subprocess.run(command, cwd=snapshot, env=env, stdout=f, stderr=subprocess.STDOUT)
    assert result.returncode == 0, 'Inspect run.log'
    output = OUT / 'output/hnsd_raw_control/result/hnsd_raw_control.png'
    a = cv2.imread(str(output))
    assert a is not None and a.shape == crop.shape, 'CLI may log errors and return zero; require actual image'
    assert all(sha(ROOT / p) == digest for p, digest in dependencies.items())
    (OUT / 'inference_verified.json').write_text(json.dumps(dict(output=str(output.relative_to(ROOT)), sha256=sha(output), shape=list(a.shape), dependencies_unchanged=True), indent=2))
    print('Raw-input control saved:', output, flush=True)


if __name__ == '__main__':
    main()
