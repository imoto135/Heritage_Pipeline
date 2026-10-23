"""Score added source-only text guards without altering the fixed26case metrics."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import evaluate_dark_nontext_teacher_control as ev
import evaluate_dark_nontext_source_context as ctx
import report_background_diversity_control as refs
from compare_v5_aligned_fade_control import rgb,mask,dump,sha
ROOT=ev.ROOT;OUT=ev.OUT

def freeze():
    ev.check();ctx.check();assert not (OUT/'text_guard_plan.json').exists();assert not list((OUT/'evaluation').glob('*/results.json'))
    ann=json.loads((OUT/'text_guard_annotations.json').read_text());files=[Path(__file__),ROOT/'scripts/benchmark/prepare_dark_nontext_text_guards.py',OUT/'text_guard_annotations.json']
    for r in ann['rows']:files += [ROOT/r['source_file'],ROOT/r['mask_file']]
    c=next(c for c in ev.old.cases() if c['id']=='D04');x,y,w,h=c['roi'];assert np.array_equal(rgb(OUT/'source_guard/D04_original.png'),c['page'][y:y+h,x:x+w])
    dump(OUT/'text_guard_plan.json',dict(status='Supplemental source-only visible ink retention guards, frozen before new experiment outputs. Not training or a replacement of26case partial labels. Prior development outputs seen.',annotations='text_guard_annotations.json',source_overlays_reviewed=True,core_counts={r['id']:r['ink_n'] for r in ann['rows']},score='Exact/radius2 covered source-supported traces; guard against erasing all text, not boundaryGT/F1/recognition.',D04='4conditions*3grids plusHowe, never new teaching',S03='New source fit native256 and full-page512/fixed256grids; same-source fit only, no generalization claim',frozen_files={str(f.relative_to(ROOT)):sha(f) for f in files}))
    print('Frozen D04 ink525 and S03 ink115 supplemental guards, not training',flush=True)
def check():
    ev.check();ctx.check();p=json.loads((OUT/'text_guard_plan.json').read_text())
    for n,h in p['frozen_files'].items():assert sha(ROOT/n)==h,n
    return p

def run():
    check();ann=json.loads((OUT/'text_guard_annotations.json').read_text());cores={r['id']:np.asarray(Image.open(ROOT/r['mask_file']))>0 for r in ann['rows']};rows=[];yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    def score(name,key,grid,path):
        pred=mask(path);core=cores[key];dist=ndi.distance_transform_edt(~pred) if pred.any() else np.full(pred.shape,np.inf);r2=int((core&(dist<=2)).sum());assert r2==int((core&ndi.binary_dilation(pred,structure=disk)).sum());rows.append(dict(condition=name,case=key,grid=grid,core_n=int(core.sum()),core_exact=int((core&pred).sum()),core_r2=r2,mask_file=str(path.relative_to(ROOT)),mask_sha256=sha(path)))
    for seed in [0,1]:
      for arm in ev.tr.ARMS:
        name=f'ragan_seed{seed}_{arm}'
        for grid in ev.GRIDS:score(name,'D04',grid,OUT/'evaluation'/name/'masks'/grid/'D04.png')
        score(name,'S03','native256_fit',OUT/'evaluation'/name/'stain_teacher_fit/S03.png')
        for grid in ev.GRIDS:score(name,'S03',grid,OUT/'source_context'/name/'masks'/grid/'S03.png')
    c=next(c for c in ev.old.cases() if c['id']=='D04');score('howe_fixed','D04','native_reference',refs.reference_path(c));assert len(rows)==29
    dump(OUT/'text_guard_results.json',dict(rows=rows,plan_sha256=sha(OUT/'text_guard_plan.json'),annotations_unchanged=True,no_training=True,formal_accuracy=False));print('Scored29 supplemental text-retention guards',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);v=a.parse_args();freeze() if v.action=='freeze' else run()
