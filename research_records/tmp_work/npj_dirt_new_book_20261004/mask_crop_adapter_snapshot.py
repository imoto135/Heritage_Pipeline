"""Resume verified mask cropping without rerunning completed ACE/CSC.

Fresh generate_mask_csc writes raw original00; archived corpus00 is a processed
classification image. The original caller incorrectly required both to match
raw. Keep its recorded source and failed log, fix only the input audit adapter.
"""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT,read,write,sha,dump
from validate_npj_dirt_new_book import OUT


def main():
    assert not (OUT/'full_mask_complete.json').exists(),'Already prepared'
    selection=json.loads((OUT/'selection.json').read_text())
    for rel,digest in selection['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    audit=[];deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for c in selection['cases']:
        original=read(ROOT/c['original']);classification=read(ROOT/c['npj_input'])
        assert original.shape==classification.shape
        exact=np.array_equal(classification,original)
        if c['new_masks']:assert exact
        x,y,w,h=c['xywh']
        for key in ['npj_input','npj_black','npj_red','npj_degradation','npj_background']:
            deps[c[key]]=sha(ROOT/c[key])
        for key,filename in [('npj_black','02_text_black.png'),('npj_red','03_text_red.png')]:
            full=cv2.imread(str(ROOT/c[key]),0);assert full.shape==original.shape[:2]
            target=OUT/'masks'/c['id']/filename;crop=full[y:y+h,x:x+w]
            if target.exists():assert np.array_equal(cv2.imread(str(target),0),crop)
            else:write(target,crop)
        if not c.get('saved_full'):
            assert np.array_equal(read(OUT/'input'/(c['id']+'.png')),original[y:y+h,x:x+w])
        audit.append(dict(id=c['id'],classification00_matches_raw=bool(exact),
             classification00_mae_to_raw=float(np.abs(classification.astype(float)-original.astype(float)).mean()),
             masks_exact_from_recorded_full_source=True,
             raw_inference_crop_exact=not bool(c.get('saved_full')),
             note='Processed classification00 is not substituted for actual model input'))
    shutil.copyfile(__file__,OUT/'mask_crop_adapter_snapshot.py')
    dump(OUT/'mask_input_audit.json',dict(cases=audit,dependencies=deps,
         original_mask_command_not_rerun=True,
         source_expectation_correction='Fresh00 is raw; archived00 can be ACE/processed; masks copied unchanged, inference uses recorded raw original'))
    dump(OUT/'full_mask_complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p)
         for folder in [OUT/'full_npj_masks',OUT/'masks'] for p in sorted(folder.rglob('*.png'))},
         mask_generation_completed_before_input_audit_failure=True,mask_generation_not_repeated=True,
         adapter_source_sha256=sha(Path(__file__))))
    print('Full-page masks and six exact mask crops verified; no ACE/CSC rerun',flush=True)


if __name__=='__main__':main()
