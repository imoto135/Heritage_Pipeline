"""Prepare new-page inputs with the frozen pilot's local-Sauvola settings.

Run compare_training_free_color_shape.py with this output's pilot/input_review.
This does not train a model or generate manual ground truth.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import cv2
import numpy as np
import skimage
from skimage.filters import threshold_sauvola

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/training_free_fixed_check_20260930'
PRIOR=ROOT/'tmp_work/training_free_preprocess_20260930'
COLOR=ROOT/'tmp_work/training_free_color_shape_20260930'
cv2.setNumThreads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def foreground_inputs(original):
    # Identical operations and constants to the frozen preprocessing pilot.
    h,w=original.shape[:2]
    gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
    small=cv2.resize(gray,((w+3)//4,(h+3)//4),interpolation=cv2.INTER_AREA)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
    closed=cv2.morphologyEx(small,cv2.MORPH_CLOSE,kernel,borderType=cv2.BORDER_REFLECT)
    smooth=cv2.GaussianBlur(closed.astype(np.float32),(0,0),2.,borderType=cv2.BORDER_REFLECT)
    background=cv2.resize(smooth,(w,h),interpolation=cv2.INTER_LINEAR)
    background=np.maximum(background,gray.astype(np.float32))
    normalized=np.round(np.clip(255.*gray/np.maximum(background,1.),0,255)).astype(np.uint8)
    votes=np.zeros((h,w),np.uint8)
    local=None
    for window in (31,75,151):
        foreground=normalized<threshold_sauvola(normalized,window_size=window,k=.2,r=128.)
        votes+=foreground
        if window==75:
            local=foreground
    return local.astype(np.uint8)*255,votes*85,normalized


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert cv2.imwrite(str(path),value),path


def main():
    selection=json.loads((OUT/'input_review/manifest.json').read_text())
    previous=json.loads((PRIOR/'manifest.json').read_text())
    previous_color=json.loads((COLOR/'manifest.json').read_text())
    assert sha(ROOT/'scripts/benchmark/compare_training_free_color_shape.py')==previous_color['source_sha256']
    pilot=OUT/'pilot';pilot.mkdir(exist_ok=True)
    manifest=dict(purpose='Fixed-parameter check on eight book IDs outside the four-page development set; no GT',
                  settings=previous['settings'],source_sha256=sha(Path(__file__)),
                  selection_manifest_sha256=sha(OUT/'input_review/manifest.json'),
                  development_foreground_manifest_sha256=sha(PRIOR/'manifest.json'),
                  frozen_color_manifest_sha256=sha(COLOR/'manifest.json'),
                  inputs=selection['inputs'],
                  versions=dict(python=sys.version,opencv=cv2.__version__,numpy=np.__version__,skimage=skimage.__version__))
    path=pilot/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('Existing inputs belong to a different run')
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(__file__,pilot/'source_snapshot.py')
    # Adapter parity is checked on one previously computed page.
    case='hnsd001_011'
    original=cv2.imread(str(PRIOR/case/'original.png'))
    local,votes,normalized=foreground_inputs(original)
    for value,filename in [(local,'sauvola_local/foreground.png'),(votes,'scale_votes.png'),(normalized,'normalized.png')]:
        expected=cv2.imread(str(PRIOR/case/filename),0)
        assert np.array_equal(value,expected),filename
    (pilot/'adapter_parity.json').write_text(json.dumps(dict(page=case,local_mask_equal=True,
        scale_votes_equal=True,normalized_equal=True),indent=2)+'\n')
    print('Adapter parity passed against frozen development outputs.',flush=True)
    for case in selection['cases']:
        source=ROOT/case['original']
        assert sha(source)==selection['inputs'][case['original']]
        original=cv2.imread(str(source));assert original is not None
        local,votes,normalized=foreground_inputs(original)
        target=pilot/case['id']
        write(target/'original.png',original)
        write(target/'sauvola_local/foreground.png',local)
        write(target/'scale_votes.png',votes)
        write(target/'normalized.png',normalized)
        print(f"{case['id']}: {original.shape[1]}x{original.shape[0]}, local foreground {int((local>0).sum())}",flush=True)
    outputs={str(p.relative_to(pilot)):sha(p) for p in sorted(pilot.rglob('*.png'))}
    (pilot/'complete.json').write_text(json.dumps(dict(status='complete',outputs=outputs),indent=2)+'\n')


if __name__=='__main__':
    main()
