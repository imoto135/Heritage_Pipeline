from pathlib import Path
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'tmp_work'
M = BASE / 'npj_manual_stain_oracle_20261004'
def a(p): return np.asarray(Image.open(p).convert('RGB'))
rows = []
for version in ['', '_v2', '_v3']:
    mask = np.asarray(Image.open(M / ('hnsd001_manual_mask' + version + '.png')).convert('L')) > 0
    out = a(M / ('hnsd001_harmonic_postprocess' + version + '.png'))
    baseline = a(M / 'hnsd001_output/hnsd001_011/result/hnsd001_011.png')
    change = np.any(out != baseline, axis=2)
    rows.append(dict(version=version or 'v1', mask_pixels=int(mask.sum()), changed_pixels=int(change.sum()), changed_outside=int((change & ~mask).sum()), shape=list(out.shape)))
print('RECOUNT',json.dumps(rows))
for folder in [
    'npj_dirt_clean_reference_20261002/native',
    'npj_literature_trial_review_20261002/native/07_red_seal_control',
    'npj_literature_trial_review_20261002/native/old_01_upper_faint_stroke',
    'npj_colour_independent_ridge_20261004/native/new_new_warm_lower_loss',
    'npj_ridge_saved_ranges_20261004/native/known_upper_stain',
]:
    print('FOLDER',folder)
    for p in sorted((BASE/folder).rglob('*.png')):
        print(str(p.relative_to(BASE)), Image.open(p).size)
(Path(__file__).parent/'recount_preliminary.json').write_text(json.dumps(rows,indent=2))
