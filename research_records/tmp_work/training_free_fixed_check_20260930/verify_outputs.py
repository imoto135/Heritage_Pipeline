"""Verify experiment artifacts without modifying computation outputs."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PILOT = OUT / 'pilot'
REVIEW = OUT / 'input_review'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, flag=cv2.IMREAD_COLOR):
    value = cv2.imread(str(path), flag)
    assert value is not None, path
    return value


manifest = json.loads((OUT/'manifest.json').read_text())
prior = json.loads((ROOT/'tmp_work/training_free_color_shape_20260930/manifest.json').read_text())
assert manifest['parameters']==prior['parameters']
selection = json.loads((REVIEW/'manifest.json').read_text())
assert len({c['book'] for c in selection['cases']})==8
assert not ({c['book'] for c in selection['cases']} & set(selection['excluded_development_books']))
complete = json.loads((OUT/'complete.json').read_text())
metrics = json.loads((OUT/'diagnostics.json').read_text())
cases = json.loads((REVIEW/'manifest.json').read_text())['cases']
assert sha(PILOT/'manifest.json') == manifest['parent_manifest_sha256']
assert sha(REVIEW/'rois.json') == manifest['rois_sha256']
for path, expected in manifest['input_hashes'].items():
    assert sha(ROOT/path) == expected, path
for path, expected in manifest['dependencies'].items():
    assert sha(PILOT/path) == expected, path
for name, expected in complete['outputs'].items():
    assert sha(OUT/name) == expected, name
    read(OUT/name)
for name, source, key in [
    ('source_snapshot.py', 'scripts/benchmark/compare_training_free_color_shape.py', 'source_sha256'),
    ('viewer_snapshot.html', 'scripts/benchmark/training_free_color_shape_viewer.html', 'viewer_template_sha256')]:
    assert sha(OUT/name) == sha(ROOT/source) == manifest[key]
ast.parse((OUT/'source_snapshot.py').read_text())

for case in cases:
    stem = case['id']
    original = read(OUT/stem/'original.png')
    assert np.array_equal(original, read(PILOT/stem/'original.png'))
    assert np.array_equal(original, read(ROOT/case['original']))
    h, w = original.shape[:2]
    foreground = read(PILOT/stem/'sauvola_local/foreground.png', 0)>127
    shape_masks = None
    for mode in manifest['modes']:
        masks = {}
        for key in ('black', 'red', 'stain', 'hold', 'ink'):
            value = read(OUT/stem/mode/(key+'.png'), 0)
            assert value.shape == (h, w)
            assert set(np.unique(value)).issubset({0, 255})
            masks[key] = value>127
        assert np.array_equal(masks['ink'], masks['black']|masks['red'])
        union = np.zeros((h, w), np.uint8)
        for key in ('black', 'red', 'stain', 'hold'):
            union += masks[key].astype(np.uint8)
        assert union.max()<=1
        if mode=='local_baseline':
            assert np.array_equal(masks['ink'], foreground)
        if mode=='color_shape':
            shape_masks=masks
        for selected in ('all', 'ink', 'black', 'red', 'stain', 'hold'):
            overlay = read(OUT/stem/mode/('overlay_'+selected+'.png'))
            selected_mask = union>0 if selected=='all' else masks[selected]
            assert np.array_equal(overlay[~selected_mask], original[~selected_mask])
        for metric in [m for m in metrics if m['page']==stem and m['mode']==mode]:
            slices = (slice(None), slice(None))
            if 'roi' in metric:
                roi = json.loads((REVIEW/'rois.json').read_text())[stem][metric['roi']-1]
                x,y,rw,rh = roi['xywh']
                assert x>=0 and y>=0 and x+rw<=w and y+rh<=h
                slices = (slice(y,y+rh), slice(x,x+rw))
            for key in ('black', 'red', 'stain', 'hold'):
                assert int(masks[key][slices].sum())==metric[key+'_pixels']
            ink, base = masks['ink'][slices], foreground[slices]
            assert int((ink&~base).sum())==metric['added_ink_pixels']
            assert int((base&~ink).sum())==metric['removed_ink_pixels']
    # All changes caused by shape gating must come from logged components.
    dense = read(OUT/stem/'dense_components.png',0)>127
    assert np.array_equal(foreground&~shape_masks['ink'],dense)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8),connectivity=8)
    ids = np.zeros(n, bool)
    for component in json.loads((OUT/stem/'components.json').read_text()):
        i = component['id']
        assert stats[i].tolist()==component['xywh']+[component['area']]
        ids[i]=True
    assert np.array_equal(dense,ids[labels]&~shape_masks['red'])

derivation = json.loads((OUT/'viewer_derivation.json').read_text())
assert sha(OUT/'index_frozen_template.html')==derivation['frozen_renderer_index_sha256']
assert sha(OUT/'index.html')==derivation['display_index_sha256']
assert sha(OUT/'build_audit.py')==derivation['postprocessor_sha256']
expected_html = (OUT/'index_frozen_template.html').read_text()
for before,after in derivation['substitutions'].items():
    expected_html=expected_html.replace(before,after)
assert expected_html==(OUT/'index.html').read_text()
prep_manifest=json.loads((PILOT/'manifest.json').read_text())
assert sha(ROOT/'scripts/benchmark/prepare_training_free_fixed_check.py')==sha(PILOT/'source_snapshot.py')==prep_manifest['source_sha256']
assert sha(REVIEW/'manifest.json')==prep_manifest['selection_manifest_sha256']
assert all(json.loads((PILOT/'adapter_parity.json').read_text())[k] for k in ['local_mask_equal','scale_votes_equal','normalized_equal'])
html = (OUT/'index.html').read_text()
script = re.search(r'<script>(.*?)</script>',html,re.S).group(1)
assert '__DATA__' not in script
data = json.loads(re.search(r'const DATA=(.*);\nconst \$=',script).group(1))
assert len(data['cases'])==8 and sum(len(c['rois']) for c in data['cases'])==24
for case in data['cases']:
    for roi in case['rois']+case['components']:
        assert (OUT/roi['image']).exists()
    for mode in data['modes']:
        for filename in ['classes.png']+['overlay_'+key+'.png' for key in ('all','ink','black','red','stain','hold')]:
            assert (OUT/case['id']/mode/filename).exists()
for href in re.findall(r'href="([^"]+)"',html):
    assert (OUT/href).resolve().exists(), href
js_path = OUT/'viewer_syntax_check.js'
js_path.write_text(script)
subprocess.run(['node','--check',str(js_path)],check=True)
auxiliary = {str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*.png')) if str(p.relative_to(OUT)) not in complete['outputs']}
result = dict(status='verified',pages=8,rois=24,modes=3,
              computation_pngs_checked=len(complete['outputs']),auxiliary_png_sha256=auxiliary,
              checks=['parameters equal frozen development version', 'eight book IDs outside development set', 'source/dependency/output hashes','original decoded pixels unchanged',
                      'native-size binary masks; disjoint candidate classes',
                      'ink equals black OR red; baseline equals previous local mask',
                      'overlays preserve all pixels outside selected masks',
                      'page and ROI diagnostic counts recomputed',
                      'shape exclusions equal logged component masks',
                      'viewer linked files and JavaScript syntax'],
              browser_ui_tested=False,manual_ground_truth_used=False)
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
