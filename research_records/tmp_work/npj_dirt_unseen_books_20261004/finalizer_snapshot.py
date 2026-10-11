"""Audit fixed transfer comparisons, preserve counterexamples, freeze artifacts."""
import csv
import json
from pathlib import Path
import re
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, sha, dump


def finalize(name):
    out=ROOT/'tmp_work'/name
    assert not (out/'complete.json').exists(), 'Already frozen'
    selection=json.loads((out/'selection.json').read_text())
    for rel,digest in selection['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    for filename,folder in [('inference_complete.json',out),('complete.json',out/'classification'),('comparison_complete.json',out)]:
        bundle=json.loads((folder/filename).read_text())
        for rel,digest in bundle['outputs'].items():assert sha(folder/rel)==digest,rel
    records=[]
    for c in selection['cases']:
        x,y,w,h=c['xywh'];s=np.s_[y:y+h,x:x+w]
        n=out/'native'/c['id'];a={k:read(n/(k+'.png')) for k in ['original','baseline','v6_all_boundary','v6_clean_reference','ruled_clean_reference']}
        assert np.array_equal(a['original'],read(ROOT/c['original'])[s])
        for key,filename in [('npj_black','02_text_black.png'),('npj_red','03_text_red.png')]:
            expected=cv2.imread(str(ROOT/c[key]),0)[s]
            assert np.array_equal(expected,cv2.imread(str(out/'masks'/c['id']/filename),0))
        roles={k:read(out/'classification/v6_masks'/c['page']/(k+'.png'),True)[s]
               for k in ['edit','hold','protect','red','extra']}
        keep=roles['hold']|roles['protect']|roles['red']|roles['extra']|(cv2.imread(str(out/'masks'/c['id']/'03_text_red.png'),0)<127)
        applied=read(n/'v6_applied.png',True);update=read(n/'v6_update.png',True);known=read(n/'v6_reference.png',True);used=read(n/'v6_used_reference.png',True)
        assert np.array_equal(read(n/'keep.png',True),keep)
        assert not (applied&~update).any() and not (update&keep).any()
        assert not (used&~known).any() and not (known&update).any()
        assert np.array_equal(a['baseline'][~applied],a['v6_clean_reference'][~applied])
        assert np.array_equal(a['baseline'][keep],a['v6_clean_reference'][keep])
        ma=read(n/'ruled_applied.png',True);mu=read(n/'ruled_update.png',True)
        assert not (ma&~mu).any()
        assert np.array_equal(a['baseline'][~ma],a['ruled_clean_reference'][~ma])
        # Verify all overview panels independently from the renderer.
        canvas=read(out/(c['id']+'_comparison.png'))
        for i,key in enumerate(['original','baseline','v6_all_boundary','v6_clean_reference','ruled_clean_reference']):
            panel=cv2.resize(a[key],(540,540),interpolation=cv2.INTER_AREA)
            left=24+i*(540+24)
            assert np.array_equal(canvas[120:660,left:left+540],panel)
        changed=np.any(a['ruled_clean_reference']!=a['baseline'],axis=2)
        records.append(dict(id=c['id'],raw_crop_exact=True,stored_masks_exact=True,
                            full_page_v6_roles_exact=True,panel_pixels_exact=True,
                            v6_kept_pixels_exact=True,all_methods_outside_applied_exact=True,
                            ruled_changes_on_v6_keep=int((changed&keep).sum()),
                            v6_changed_pixels=int(np.any(a['v6_clean_reference']!=a['baseline'],axis=2).sum())))
    # Detect expected semantic failure, rather than treating a numerically valid solver as safe.
    if name=='npj_dirt_transfer_20261004':
        failing=next(r for r in records if r['id']=='seal_control')
        assert failing['ruled_changes_on_v6_keep']==11325
        assert (out/'failure_header_as_footer.png').exists()
    else:
        prior_books=set()
        for p in (ROOT/'tmp_work').glob('*/baseline_review.json'):
            if p.parent.name.startswith('npj_dirt_'):continue
            for c in json.loads(p.read_text()).get('cases',[]):
                if 'book' in c:prior_books.add(c['book'])
        for name0 in ['warm_ink_hold_v6_20260930','warm_ink_v6_other_books_fixed_20261001']:
            for c in json.loads((ROOT/'tmp_work'/name0/'manifest.json').read_text()).get('cases',[]):
                if 'book' in c:prior_books.add(c['book'])
        assert all(c['book'] not in prior_books for c in selection['cases'])
        assert all(r['v6_changed_pixels']==0 and r['ruled_changes_on_v6_keep']==0 for r in records)
    dump(out/'independent_verification.json',dict(cases=records,
         no_threshold_changes=True, no_text_generation=True,
         decision='Lower-rule-only mask rejected for general use; v6 preserves selected roles but misses observed defects',
         limitations='Role masks and dark-pixel counts are not semantic GT; visual counterexample is a required gate'))
    dump(out/'adoption_gate.json',dict(adopted=False, keep_saved_npj_and_frozen_v6=True,
         reason='Header/footer confusion deletes observed writing; residual wormhole/binding marks not selected',
         next='Use full-original page layout to distinguish frame sides; separately review damage rims currently classified as ink; no stroke completion'))
    for target in re.findall(r'\]\(([^)]+)\)',(out/'report.md').read_text()):
        if target=='comparison_pngs.zip':continue
        assert (out/target).exists(),target
    shutil.copyfile(__file__,out/'finalizer_snapshot.py')
    dump(out/'review_manifest.json',dict(finalizer_sha256=sha(Path(__file__)),
         selection_sha256=sha(out/'selection.json'),report_sha256=sha(out/'report.md'),
         comparison_count=len(list(out.glob('*.png'))),source_scope='Current frozen-code crop controls, not historical saved npj final pages'))
    pngs=sorted(out.glob('*.png'))
    with zipfile.ZipFile(out/'comparison_pngs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,p.name)
    with zipfile.ZipFile(out/'comparison_pngs.zip') as z:
        assert z.testzip() is None
        for p in pngs:assert z.read(p.name)==p.read_bytes()
    dump(out/'complete.json',dict(outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    for rel,digest in json.loads((out/'complete.json').read_text())['outputs'].items():assert sha(out/rel)==digest,rel
    print('Frozen and verified',out.name,len(pngs),'PNG',len(records),'frames',flush=True)


if __name__=='__main__':
    for name in ['npj_dirt_transfer_20261004','npj_dirt_unseen_books_20261004']:finalize(name)
