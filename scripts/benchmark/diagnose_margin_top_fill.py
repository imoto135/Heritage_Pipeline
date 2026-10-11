"""Same top-completion masks with Telea; explicitly a non-DDRM diagnostic."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import check_margin_top_completion_ddrm as experiment
from build_npj_v6_comparison import dump, read, sha, write
from package_paper_blending_review import check_bundle

ROOT = experiment.ROOT
OUT = ROOT/'tmp_work/margin_top_completion_fill_diagnostic_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [experiment.OUT,experiment.mask_probe.OUT,experiment.prior_ddrm.OUT]:check_bundle(folder)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        **{str((f/n).relative_to(ROOT)):sha(f/n) for f in [experiment.OUT,experiment.mask_probe.OUT,experiment.prior_ddrm.OUT]
           for n in ['manifest.json','complete.json']}}
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=dependencies,regions=22,pages=14,
        settings=dict(method='OpenCV Telea',radius_px=3,context_px=64,mask_fixed=True,
        baseline='saved original-context Telea with old margin mask',DDRM=False,
        purpose='post-output diagnostic of dark generated strokes; no training or mask changes'))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=[]
    for source in experiment.mask_probe.old.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            original=read(ROOT/page['original']);H,W=original.shape[:2];key=page['page']
            masks={'old':read(experiment.mask_probe.old.OUT/'candidate_pages'/key/'dark_support/edit.png',True),
                **{m:read(experiment.mask_probe.OUT/'pages'/key/m/'edit.png',True) for m in experiment.MODES}}
            for roi in page['rois']:
                rid=roi['id'];x,y,w,h=roi['xywh'];x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(W,x+w+64),min(H,y+h+64)
                context=original[y0:y1,x0:x1];local=(slice(y-y0,y-y0+h),slice(x-x0,x-x0+w));crop=context[local]
                values={};reused={}
                for mode in ['old']+experiment.MODES:
                    missing=masks[mode][y0:y1,x0:x1]
                    if mode!='old' and np.array_equal(missing,masks['old'][y0:y1,x0:x1]):
                        value=values['old'].copy();reused[mode]=True
                    else:
                        value=cv2.inpaint(context,missing.astype(np.uint8)*255,3,cv2.INPAINT_TELEA) if missing.any() else context.copy()
                        value[~missing]=context[~missing];value=value[local];reused[mode]=False
                    assert not np.any(np.any(value!=crop,axis=2)&~missing[local])
                    values[mode]=value;write(OUT/'native'/rid/(mode+'_telea.png'),value)
                assert np.array_equal(values['old'],read(experiment.prior_ddrm.OUT/'native'/rid/'candidate_telea.png'))
                ddrm=read(experiment.OUT/'native'/rid/'completed.png')
                experiment.prior_ddrm.figure([crop,ddrm,values['completed']],
                    ['原画像','全成分マスク → DDRM','同じマスク → Telea'],
                    OUT/'comparisons'/(rid+'.png'),roi['title']+' · 同じマスクで復元器を分離',
                    'Teleaは局所補間でDDRMではない。同じ原画像・マスク・周囲64px。境界の紙色補正は加えない。',
                    '補助診断：DDRMで残る暗い模様を、同じ編集範囲の補間と比較。')
                rows.append(dict(id=rid,reused=reused,outside_edit_changed_pixels=0))
            del original,masks;gc.collect()
    dump(OUT/'diagnostics.json',rows)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',regions=22,pages=14,same_masks=True,DDRM=False,
        old_Telea_output_reproduced=True,outside_edit_changed_pixels=0))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json'] and p.name not in ['manifest.json','complete.json']}))
    print('Passed: same-mask Telea diagnostic, all 22 ROIs',flush=True)


if __name__=='__main__':main()
