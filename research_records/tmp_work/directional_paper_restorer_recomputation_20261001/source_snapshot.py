"""Check full restorer parity before a mask/colour factorial comparison."""
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import probe_observed_absolute_paper_patches as prior
from build_npj_v6_comparison import dump,read,sha
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/directional_paper_restorer_recomputation_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    manifest=dict(dependencies=deps,source_sha256=sha(Path(__file__)),settings=dict(pages=29,ROIs=90,mask='current effective applied only',reference='current frozen paper',solver='same16pair field_for including fineL seam',change='verification only'))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=json.loads((prior.OUT/'diagnostics.json').read_text());results=[];current=None
    for row in rows:
        key=row['page'];rid=row['id'];source=prior.OUT/'native'/rid
        if current!=key:
            original=read(ROOT/row['original_source']);full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB);current=key
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        app=read(source/'applied_context.png',True);known=read(source/'reference_context.png',True);base=read(source/'baseline.png');changed=0;fielderror=0.;applied_mismatch=0
        if app[s].any():
            result=prior.fill.field_for(full_lab[cs],known,app,16);field=result[4];accepted=result[1]
            applied_mismatch=int((accepted!=app).sum());value=prior.fill.frozen.compose(field,accepted,base,s)
            changed=int(np.any(value!=base,axis=2).sum());saved=np.load(source/'baseline_field_Lab.npy',allow_pickle=False)
            fielderror=float(np.abs(field[app]-saved[app]).max())
        results.append(dict(id=rid,applied_mismatch=applied_mismatch,changed_pixels=changed,maximum_applied_float_error=fielderror))
        print('Restorer parity:',len(results),'/90',rid,applied_mismatch,changed,fielderror,flush=True)
    dump(OUT/'verification.json',dict(status='completed',ROIs=90,pages=29,records=results,
        all_BRG_outputs_equal=all(r['changed_pixels']==0 for r in results),all_applied_equal=all(r['applied_mismatch']==0 for r in results),
        all_float_fields_equal=all(r['maximum_applied_float_error']==0 for r in results)))
    for p,h in deps.items():assert sha(ROOT/p)==h
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete: full current-restorer recomputation parity',flush=True)


if __name__=='__main__':main()
