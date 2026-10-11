"""Preserve the complete observed span of the matched horizontal rule groups.

A fitted vertical side can be an internal table division. A writing envelope
must include the top/bottom rule observations beyond that side, not merely the
four fitted corner intersections. No damage threshold or renderer changes.
"""
import copy
import json
from pathlib import Path
import shutil

import numpy as np

import probe_npj_full_page_layout as core
import probe_npj_full_page_layout_v3 as fitted
from build_npj_v6_comparison import ROOT,sha,dump

OUT=ROOT/'tmp_work/npj_full_page_layout_v4_20261004'


def detect_frame(original):
    masks,diagnostic=fitted.detect_frame(original)
    chosen=diagnostic['chosen']
    if chosen is None:return masks,diagnostic
    old=chosen['box'].copy()
    horizontal=[]
    for line in chosen['raw_support_members']:
        x0,y0,x1,y1=line
        if abs(x1-x0)>0 and abs(y1-y0)<=.04*abs(x1-x0):horizontal.append(line)
    assert horizontal
    points=np.array(horizontal).reshape(-1,2)
    low=points.min(0);high=points.max(0)
    x0,y0=min(old[0],int(low[0])),min(old[1],int(low[1]))
    x1,y1=max(old[2],int(high[0])),max(old[3],int(high[1]))
    H,W=original.shape[:2]
    interior=np.zeros((H,W),bool)
    interior[max(0,y0-8):min(H,y1+9),max(0,x0-8):min(W,x1+9)]=True
    masks['interior']=interior
    masks['exterior']=~interior&masks['footprint']
    chosen['fitted_box_before_observed_span']=old
    chosen['box']=[x0,y0,x1,y1]
    chosen['envelope_includes_complete_horizontal_observations']=True
    return masks,diagnostic


if __name__=='__main__':
    digest=sha(Path(__file__))
    core.OUT=OUT;core.SETTINGS=copy.deepcopy(core.SETTINGS)
    core.SETTINGS['frame']['fragment_grouping']='Mid-page intercept complete-link span <= .012*orthogonal page size'
    core.SETTINGS['frame']['directional_filter']='31px horizontal/vertical morphology opening before Hough'
    core.SETTINGS['frame']['corners']='Intersections of median slope/intercept of four rule groups'
    core.SETTINGS['frame']['support_gate']='Each fitted rule >=45% directional ink occupancy within 3px'
    core.SETTINGS['frame']['selection']='Largest supported frame; writing envelope is union of fitted corners and complete observed top/bottom spans'
    core.SETTINGS['frame']['limitations']='Fitted vertical sides may be internal divisions; observations bound writing, not a proven physical frame GT'
    core.detect_frame=detect_frame
    core.main()
    assert sha(Path(__file__))==digest
    manifest=json.loads((OUT/'manifest.json').read_text())
    for p in [Path(__file__),Path(fitted.__file__),ROOT/'scripts/benchmark/probe_npj_full_page_layout_v2.py']:
        manifest['dependencies'][str(p.relative_to(ROOT))]=sha(p)
    manifest['entrypoint']=str(Path(__file__).relative_to(ROOT))
    manifest['entrypoint_sha256_recorded_before_run']=digest
    dump(OUT/'manifest.json',manifest);shutil.copyfile(__file__,OUT/'entrypoint_snapshot.py')
