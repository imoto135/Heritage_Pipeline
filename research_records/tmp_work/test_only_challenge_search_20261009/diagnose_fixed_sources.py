"""Source-only neighbor review and separate saved-mask provenance checks.

No mask produced by this diagnostic is supplied to the frozen evaluation.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import validate_fixed_holdout as v


def sources():
    proto=json.loads((v.OUT/'protocol.json').read_text());records=[]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    for cid, neighbors in [('F02',['200021063_00006_2','200021063_00007_2']),
                           ('F04',['200021063_00002_1','200021063_00003_1'])]:
        case=next(c for c in proto['cases'] if c['id']==cid);src=v.ROOT/case['source']
        raw=cv2.imread(str(src))[:,:,::-1];h,w=raw.shape[:2];arrays=[raw];deps={v.driver.relative(src):v.driver.digest(src)}
        for page in neighbors:
            path=v.ROOT/'data/test'/page.split('_')[0]/(page+'.jpg')
            arrays.append(cv2.resize(cv2.imread(str(path))[:,:,::-1][:,::-1],(w,h)))
            deps[v.driver.relative(path)]=v.driver.digest(path)
        sheet=Image.new('RGB',(1800,1100),'#eee');draw=ImageDraw.Draw(sheet)
        for i,(arr,title) in enumerate(zip(arrays,[case['page']+' 原画像']+[n+' 左右反転' for n in neighbors])):
            im=Image.fromarray(arr);im.thumbnail((580,1030));sheet.paste(im,(i*600,60));draw.text((i*600+5,8),title,font=font,fill='black')
        sheet.save(v.OUT/(cid+'_隣接原画像の鏡像比較.png'))
        v.red_review.figure(v.OUT/(cid+'_隣接原画像_ROI比較.png'),arrays,
            ['原画像','直前ページ・反転','直後ページ・反転'],case['roi_xywh'],cid)
        records.append(dict(id=cid,neighbors=neighbors,dependencies=deps,
            transform='Horizontal mirror and resize only; no registration or GT',
            interpretation='Visual source context only; matching darkness is not ground-truth ink classification'))
    v.driver.save_json(v.OUT/'neighbor_source_review.json',records)
    records=[]
    for c in proto['cases']:
        page=c['page'];saved=v.ROOT/'data/split_dataset_csc_mask_v2/test'/page.split('_')[0]/page/'00_input.png'
        a=cv2.imread(str(v.ROOT/c['source']));b=cv2.imread(str(saved));assert a.shape==b.shape
        records.append(dict(id=c['id'],saved_ace_input=v.driver.relative(saved),saved_ace_sha256=v.driver.digest(saved),
            source_sha256=c['source_sha256'],shape_match=True,
            raw_vs_ace_changed_pixels=int(np.any(a!=b,axis=2).sum()),
            note='00_input.png is the saved ACE-corrected input; pixel inequality to raw JPEG is expected; exact ACE-generation settings not independently reproduced here'))
    v.driver.save_json(v.OUT/'saved_csc_source_check.json',records)


def red_recheck():
    cv2.setNumThreads(2)
    path=v.ROOT/'tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2/scripts/csc.py'
    spec=importlib.util.spec_from_file_location('frozen_csc_red_diagnostic',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    page='200021063_00002_2';folder=v.ROOT/'data/split_dataset_csc_mask_v2/test/200021063'/page
    ace=cv2.imread(str(folder/'00_input.png'))
    roi=[1650,2500,400,620];x,y,w,h=roi;sl=np.s_[y:y+h,x:x+w]
    trace={}
    def inspect_return(frame,event,arg):
        if event=='return' and frame.f_code.co_filename==str(path):
            local=frame.f_locals
            if frame.f_code.co_name=='color_space_clustering':
                trace['stage1_nonbackground_roi_pixels']=int(local['nonbg_mask_flat'].reshape(ace.shape[:2])[sl].sum())
                trace['stage2_mean_L']=list(map(float,local.get('cluster_mean_L_s2',[])))
                trace['stage2_mean_a']=list(map(float,local.get('cluster_mean_a_s2',[])))
                trace['red_cluster']=local.get('red_cluster')
                trace['black_cluster']=local.get('black_cluster')
                if 'candidate_2d' in local:
                    trace['stage2_red_candidate_roi_pixels']=int(local['candidate_2d'][sl].sum())
            elif frame.f_code.co_name=='_filter_red_candidate_by_component':
                trace['hsv_filter_candidate_roi_pixels']=int(local['candidate_mask'][sl].sum())
                trace['hsv_filter_kept_roi_pixels']=int(arg[sl].sum())
                trace['hsv_absolute_support_roi_pixels']=int(local['is_red_abs'][sl].sum())
        return inspect_return
    sys.settrace(inspect_return)
    try:
        result=module.color_space_clustering(ace)
    finally:
        sys.settrace(None)
    out=v.OUT/'red_recheck';out.mkdir(exist_ok=True)
    red=result.red_text_mask>0
    v.probe.save(out/'03_text_red.png',red)
    saved=v.load(folder/'03_text_red.png')
    raw=cv2.imread(str(v.ROOT/'data/test/200021063'/(page+'.jpg')))[:,:,::-1]
    rgb=lambda m:np.repeat(np.where(m,0,255).astype(np.uint8)[:,:,None],3,2)
    v.red_review.figure(out/'F04_保存赤と再計算赤.png',[raw,ace[:,:,::-1],rgb(saved),rgb(red)],
        ['原画像','保存ACE入力','保存CSC赤','凍結実装・既定値の赤'],roi,'F04')
    v.driver.save_json(out/'diagnostic.json',dict(page=page,roi_xywh=roi,
        csc_code=v.driver.relative(path),csc_sha256=v.driver.digest(path),
        ace_input_sha256=v.driver.digest(folder/'00_input.png'),saved_red_sha256=v.driver.digest(folder/'03_text_red.png'),
        parameters=dict(seed=42,max_samples=50000,border_margin_ratio=.04,red_a_min=133,
            red_hsv_support_min=.05,morph_close_k=3,morph_open_k=1,black_dilate_r=0),
        saved_red_pixels=int(saved.sum()),recomputed_red_pixels=int(red.sum()),
        recomputed_roi_red_pixels=int(red[sl].sum()),saved_recomputed_disagreement=int((saved!=red).sum()),
        recomputed_saved_black_disagreement=int(((result.black_text_mask>0)!=v.load(folder/'02_text_black.png')).sum()),
        intermediate_trace=trace,diagnostic_script_sha256=v.driver.digest(Path(__file__)),
        used_in_frozen_inference=False,
        limitation='Diagnostic of frozen implementation defaults on saved ACE. Saved-mask generation settings are not assumed identical.'))
    print('Red diagnostic',int(saved.sum()),int(red.sum()),'ROI',int(red[sl].sum()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['sources','red-recheck']);args=parser.parse_args()
    sources() if args.stage=='sources' else red_recheck()
