"""Render the main shortlist and verify saved PNGs and measurements."""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('deep_review',HERE/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
protocol=json.loads((HERE/'protocol.json').read_text())
rendering=json.loads((HERE/'rendering.json').read_text())
rendering=[r for r in rendering if not r['path'].endswith('_有力候補と基準.png')]
complete=json.loads((HERE/'complete.json').read_text())
assert run.sha(HERE/'run.py')==complete['source_script_sha256']
case_by_id={c['id']:c for c in protocol['cases']}
for c in protocol['cases']:
    masks={k:(run.read_mask(HERE/'masks'/c['id']/(k+'.png')) if (HERE/'masks'/c['id']/(k+'.png')).exists()
              else run.read_mask(run.OLD/'masks'/c['id']/(k+'.png'))) for k in ['wolf61','sauvola61','howe160']}
    raw=run.read(run.ROOT/c['original'])
    rendering.append(run.sheet(c,raw,masks,['original','sauvola61','wolf61','howe160'],'有力候補と基準'))
run.dump(HERE/'rendering.json',rendering)

panel_count=0
for record in rendering:
    path=HERE/record['path'];assert run.sha(path)==record['sha256']
    cid=path.name.split('_',1)[0];c=case_by_id[cid];x,y,w,h=c['roi_xywh']
    canvas=run.read(path)
    for panel in record['panels']:
        key=panel['method'];scale=panel['integer_scale']
        if key=='original':a=run.read(run.ROOT/c['original'])[y:y+h,x:x+w]
        else:
            if key in ['background','normalized']:
                p=HERE/'intermediates'/cid/(key+'.png')
            else:
                p=HERE/'masks'/cid/(key+'.png')
                if not p.exists():p=run.OLD/'masks'/cid/(key+'.png')
            a=run.read(p)[y:y+h,x:x+w]
        expected=np.repeat(np.repeat(a,scale,axis=0),scale,axis=1)
        px,py,pw,ph=panel['canvas_xywh']
        assert np.array_equal(canvas[py:py+ph,px:px+pw],expected),(path,key)
        panel_count+=1

old_protocol=json.loads((run.OLD/'protocol.json').read_text())
old_cases={c['id']:c for c in old_protocol['cases']}
annotations={a['id']:a for a in json.loads((run.OLD/'annotations_frozen.json').read_text())['cases']}
audit=json.loads((run.OLD/'annotation_audit.json').read_text())
rows=json.loads((HERE/'measurements.json').read_text())['records']
for row in rows:
    cid,key=row['id'],row['method']
    path=HERE/'masks'/cid/(key+'.png')
    if not path.exists():path=run.OLD/'masks'/cid/(key+'.png')
    b=run.read_mask(path);ox,oy,_,_=old_cases[cid]['roi_xywh']
    ann=annotations[cid]
    points=[ann['ink_points_xy'][i] for i in audit['valid_original_point_indices'][cid]]
    hits=[bool(b[oy+py,ox+px]) for px,py in points]
    assert hits==row['point_hits'] and sum(hits)==row['ink_center_hits']
    for patch in row['nonink_patches']:
        px,py,pw,ph=patch['old_roi_relative_xywh']
        assert int(b[oy+py:oy+py+ph,ox+px:ox+px+pw].sum())==patch['foreground_pixels']
for p,h in complete['source_sha256'].items():assert run.sha(run.ROOT/p)==h
sources={str(p.relative_to(HERE)):run.sha(p) for p in (HERE/'sources').rglob('*') if p.is_file()}
run.dump(HERE/'source_manifest.json',dict(urls={
    'Howe_code':'https://www.science.smith.edu/~nhowe/research/code/IJDAR-Binarization.zip',
    'Howe_paper':'https://www.science.smith.edu/~nhowe/research/pubs/divseg-ijdar.pdf',
    'maxflow':'https://pub.ist.ac.at/~vnk/software/maxflow-v3.01.src.tar.gz',
    'Bradley_Roth':'https://people.scs.carleton.ca/~roth/iit-publications-iti/docs/gerh-50002.pdf',
    'Phansalkar_ImageJ':'https://raw.githubusercontent.com/imagej/imagej-ops/master/src/main/java/net/imagej/ops/threshold/localPhansalkar/LocalPhansalkarThreshold.java'},sha256=sources))
run.dump(HERE/'final_verification.json',dict(figures=len(rendering),panels_checked=panel_count,
    saved_png_pixels_match_native_sources=True,measurements_recounted_from_saved_masks=len(rows),
    original_sources_unchanged=True,main_experiment_complete_sha256=run.sha(HERE/'complete.json'),
    rendering_sha256=run.sha(HERE/'rendering.json'),source_manifest_sha256=run.sha(HERE/'source_manifest.json'),
    finalizer_sha256=run.sha(Path(__file__))))
print('Verified',len(rendering),'PNGs,',panel_count,'panels,',len(rows),'method/case measurements.')
