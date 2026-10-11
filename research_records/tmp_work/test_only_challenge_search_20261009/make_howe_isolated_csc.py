"""Restrict the development proposal to strongly CSC-supported isolated marks."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import probe_howe_csc_fusion as probe

def main():
    initial=probe.OUT
    out=initial/'isolated_csc_v2';out.mkdir(parents=True,exist_ok=True)
    cases=json.loads((probe.BASE/'test_candidates.json').read_text())['cases']
    previous=json.loads((initial/'diagnostics.json').read_text())
    probe.OUT=out;rows=[]
    for case,prior in zip(cases,previous):
        assert case['id']==prior['id']
        page=case['name']
        h=probe.load(initial/'masks/howe'/page/'02_text_black.png',True)
        red=probe.load(initial/'masks/howe'/page/'03_text_red.png',True)
        _,labels,_,_=cv2.connectedComponentsWithStats(h.astype(np.uint8),8)
        selected=[]
        height,width=h.shape;pad=int(np.ceil(3*prior['stroke_width']))
        for c in prior['removed_components']:
            if c['reason']!='isolated_round_mark' or c['csc_fraction']<.5:continue
            x,y,w,hh=c['xywh']
            s=np.s_[max(0,y-pad):min(height,y+hh+pad),max(0,x-pad):min(width,x+w+pad)]
            # Protect detached tips of faint strokes using all Howe ink, including
            # components that have little or no CSC support.
            if np.any(h[s]&(labels[s]!=c['component'])):continue
            selected.append(c)
        removed=np.isin(labels,[c['component'] for c in selected])
        candidate=h&~removed
        assert not np.any(removed&red)
        for method,mask in [('howe',h),('isolated',candidate)]:
            probe.save(out/'masks'/method/page/'02_text_black.png',mask)
            probe.save(out/'masks'/method/page/'03_text_red.png',red)
        original=np.asarray(Image.open(probe.ROOT/case['source']).convert('RGB'))
        veto=probe.load(initial/'masks/pixel_veto'/page/'02_text_black.png',True)
        probe.render(case,original,h,veto,candidate,removed)
        probe.review_components(case,original,removed,selected)
        x,y,w,hh=probe.ROIS[case['id']];s=np.s_[y:y+hh,x:x+w]
        row=dict(id=case['id'],page=page,removed_components=selected,
                 removed_pixels_page=int(removed.sum()),removed_pixels_roi=int(removed[s].sum()),
                 mask_unchanged_page=bool(np.array_equal(h,candidate)),
                 stroke_width=prior['stroke_width'])
        rows.append(row);print(case['id'],len(selected),'components',row['removed_pixels_page'],'page pixels',row['removed_pixels_roi'],'ROI pixels',flush=True)
    protocol=json.loads((initial/'protocol.json').read_text())
    protocol['candidate_rule']={**probe.RULE,'degradation_components_removed':False,
                                'isolated_min_csc_fraction':.5,'other_howe_ink_exclusion_margin_strokes':3}
    protocol['source_sha256']=probe.sha(Path(__file__))
    protocol['scope']='development trial; only round isolated components with CSC black fraction >= 0.5; no pixel labels'
    probe.dump(out/'protocol.json',protocol);probe.dump(out/'diagnostics.json',rows)

if __name__=='__main__':main()
