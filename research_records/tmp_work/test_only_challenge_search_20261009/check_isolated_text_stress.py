"""Extra mask-only controls containing separated real text; frozen rule."""
from pathlib import Path
import importlib.util
import json
import cv2
import numpy as np
from PIL import Image
import validate_new_materials as trial
import run_ddrm_pilot as driver
import probe_howe_csc_fusion as probe


def main():
    out=trial.OUT/'isolated_text_stress';out.mkdir(exist_ok=True)
    cases=[dict(id='S01',page='200008003_00007_1',roi_xywh=[180,800,2380,850],
                protection_xywh=[280,850,2150,900],reason='離れた漢字の見出し'),
           dict(id='S02',page='200005798_00024_2',roi_xywh=[850,600,450,1300],
                protection_xywh=[870,660,430,1130],reason='単独の短い題字')]
    for c in cases:
        source=trial.ROOT/'data/test'/c['page'].split('_')[0]/(c['page']+'.jpg')
        c['source']=driver.relative(source);c['source_sha256']=driver.digest(source)
        c['input_size']=list(Image.open(source).size)
    path=out/'protocol.json'
    assert not path.exists()
    driver.save_json(path,dict(cases=cases,rule_frozen=True,rule_protocol=driver.relative(trial.OUT/'protocol.json'),
        rule_protocol_sha256=driver.digest(trial.OUT/'protocol.json'),
        selection_scope='Additional source-selected grayscale stress controls after initial color mask review; mask only, no DDRM',
        no_parameter_changes=True,script_sha256=driver.digest(Path(__file__))))
    spec=importlib.util.spec_from_file_location('classical_stress',trial.ROOT/'tmp_work/classical_deep_review_20261008/run.py')
    deep=importlib.util.module_from_spec(spec);spec.loader.exec_module(deep)
    cv2.setNumThreads(2);records=[]
    for c in cases:
        p=c['page'];raw=np.asarray(Image.open(trial.ROOT/c['source']).convert('RGB'))
        deep.SCRATCH=out/'scratch'/p;deep.SCRATCH.mkdir(parents=True,exist_ok=True)
        terms,term_meta=deep.howe_terms(deep.base.gray(raw));h,cut_meta=deep.howe(terms,160)
        d=trial.ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p
        red=probe.load(d/'03_text_red.png',True);h &= ~red
        csc=probe.load(d/'02_text_black.png',True)
        candidate,removed,meta=trial.frozen_candidate(h,csc,probe.load(d/'04_degradation.png',False),red)
        for method,m in [('howe',h),('isolated',candidate)]:probe.save(out/'masks'/method/p/'02_text_black.png',m)
        overlay=raw.copy();overlay[removed]=[255,40,40]
        def rgb(m):return np.repeat(np.where(m,0,255).astype(np.uint8)[:,:,None],3,axis=2)
        fig=trial.panel(out/'figures'/(c['id']+'_マスク比較.png'),[raw,rgb(csc),rgb(h),rgb(candidate),overlay],
                        ['原画像','CSC','Howe','孤立点条件','除去候補（赤）'],c['roi_xywh'],c['id'])
        fig.thumbnail((1800,1400),Image.Resampling.LANCZOS);fig.save(out/'figures'/(c['id']+'_preview.png'))
        for i,comp in enumerate(meta['removed_components']):
            x,y,w,hh=comp['xywh'];cx,cy=x+w//2,y+hh//2;x0,y0=max(0,cx-100),max(0,cy-100)
            roi=[x0,y0,min(200,raw.shape[1]-x0),min(200,raw.shape[0]-y0)]
            trial.panel(out/'component_figures'/f"{c['id']}_{i:03d}.png",[raw,overlay,rgb(h),rgb(candidate)],
                        ['原画像','除去（赤）','Howe','条件後'],roi,c['id'])
        x,y,w,hh=c['protection_xywh'];sl=np.s_[y:y+hh,x:x+w]
        row=dict(id=c['id'],page=p,removed_pixels_page=int(removed.sum()),
                 protection_xywh=c['protection_xywh'],removed_pixels_protection=int(removed[sl].sum()),
                 terms=term_meta,cut=cut_meta,**meta)
        records.append(row);print(c['id'],len(meta['removed_components']),int(removed.sum()),
                                  'protected text changes',int(removed[sl].sum()),flush=True)
        driver.save_json(out/'diagnostics.json',records)
    for c in cases:assert driver.digest(trial.ROOT/c['source'])==c['source_sha256']
    driver.save_json(out/'complete.json',dict(cases=len(cases),sources_unchanged=True,
                    diagnostics_sha256=driver.digest(out/'diagnostics.json')))


if __name__=='__main__':main()
