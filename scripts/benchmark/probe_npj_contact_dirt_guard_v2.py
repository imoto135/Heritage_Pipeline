"""Reject abnormal chromatic boundary references; keep dirt/stroke rules fixed.

The first development run also exceeded the width of one seal-control input.
Fix that display frame before running; validate all frames before processing.
The nine valid old frames keep identical coordinates and candidate masks.
"""
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

import probe_npj_contact_dirt_guard as core
from build_npj_v6_comparison import ROOT,read,sha,dump

OUT=ROOT/'tmp_work/npj_contact_dirt_guard_v2_20261004'
INITIAL_REFERENCE=core.paper.paper_reference


def paper_reference(base,update,domain,mandatory):
    known,threshold=INITIAL_REFERENCE(base,update,domain,mandatory)
    if not known.any():return known,threshold
    ab=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,1:]
    median=np.median(ab[known],axis=0)
    distance=np.linalg.norm(ab-median,axis=2)
    return known&(distance<=4),threshold


if __name__=='__main__':
    digest=sha(Path(__file__))
    cases=core.source_cases()
    for c in cases:
        if c['id']=='known_200010454_v2':c['xywh']=[1164,1740,620,620]
        original=read(ROOT/c['original']);x,y,w,h=c['xywh']
        assert 0<=x and 0<=y and x+w<=original.shape[1] and y+h<=original.shape[0],c['id']
        base=read(ROOT/c['saved_npj'])
        if c.get('baseline_full_page'):assert original.shape==base.shape
        else:assert base.shape==(h,w,3)
    core.source_cases=lambda:cases
    core.OUT=OUT;core.SETTINGS=copy.deepcopy(core.SETTINGS)
    core.SETTINGS['reference_chroma']='Keep original brightness-qualified known pixels only if Lab ab distance to their median <=4; no correction to any known pixel'
    core.SETTINGS['wrapper_sha256_before_run']=digest
    core.SETTINGS['first_run_failure']='9 completed frames then seal-control xywh exceeds original width; first outputs preserved; corrected x=1164 for620px'
    core.paper.paper_reference=paper_reference
    core.main()
    assert sha(Path(__file__))==digest
    m=json.loads((OUT/'manifest.json').read_text())
    m['dependencies'][str(Path(__file__).relative_to(ROOT))]=digest
    m['entrypoint']=str(Path(__file__).relative_to(ROOT))
    dump(OUT/'manifest.json',m);shutil.copyfile(__file__,OUT/'entrypoint_snapshot.py')
