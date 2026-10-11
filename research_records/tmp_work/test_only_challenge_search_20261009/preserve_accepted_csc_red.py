"""Preserve existing CSC red inside accepted regions; add local red coverage."""
import argparse,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np
from PIL import Image
import validate_local_red as v

PARENT=v.OUT/'regional_refinement'
OUT=PARENT/'preserve_accepted_red'
METHOD='grow5_preserve'

def prepare():
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=True)
    proto=json.loads((PARENT/'protocol.json').read_text());params=proto['variants']['grow5_region']
    proto.update(created_utc=datetime.now(timezone.utc).isoformat(),variants={METHOD:dict(**params,preserve='Original CSC red intersect accepted strong-seed regions, union local red')},
        stage='Posthoc development: preserve accepted CSC red after observing faint-seal regression',
        parent_protocol_sha256=v.driver.digest(PARENT/'protocol.json'),script_sha256=v.driver.digest(Path(__file__)))
    assert not (OUT/'protocol.json').exists();v.driver.save_json(OUT/'protocol.json',proto)
    rows=[]
    for c in proto['cases']:
        p=c['page'];raw=np.asarray(Image.open(v.ROOT/c['source']).convert('RGB'));delta,chroma=v.color_fields(raw)
        ap=v.old.OUT/p/'red_without_global_gate.png'
        if not ap.exists():ap=v.OUT/'anchors'/p/'red_without_global_gate.png'
        anchor=v.review.load(ap);seed=anchor&chroma&(delta>=8)
        n,labels=cv2.connectedComponents(cv2.dilate(seed.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33))),connectivity=8)
        counts=np.bincount(labels[seed],minlength=n);keep=counts>=128;keep[0]=False;zone=keep[labels]
        previous=v.review.load(PARENT/'red_masks/grow5_region'/p/'03_text_red.png')
        savedpath=v.ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p/'03_text_red.png';saved=v.review.load(savedpath)
        red=previous|(saved&zone)
        assert not np.any((saved&zone)&~red)
        v.old.save(OUT/'red_masks'/METHOD/p/'03_text_red.png',red)
        arrays=[raw];titles=['原画像']
        for m,title in [(saved,'現行CSC赤（緑）'),(previous,'局所赤・既存保持なし（緑）'),(red,'局所赤＋既存CSC赤保持（緑）')]:
            a=raw.copy();a[m]=[20,235,20];arrays.append(a);titles.append(title)
        v.review.figure(OUT/'mask_figures'/(c['id']+'_既存赤保持_比較.png'),arrays,titles,c['roi_xywh'],c['id'])
        for k,r in c['controls_xywh'].items():v.review.figure(OUT/'mask_controls'/(c['id']+'_'+k+'_赤比較.png'),arrays,titles,r,c['id'])
        row=dict(id=c['id'],page=p,red_page_pixels=int(red.sum()),red_roi_pixels=v.roi_count(red,c['roi_xywh']),
            controls={k:v.roi_count(red,r) for k,r in c['controls_xywh'].items()},preserved_old_roi_pixels=v.roi_count(saved&red,c['roi_xywh']),
            excluded_old_roi_pixels=v.roi_count(saved&~red,c['roi_xywh']),restored_old_pixels=int((red&~previous).sum()),
            input_sha256=c['source_sha256'],anchor_sha256=v.driver.digest(ap),saved_red_sha256=v.driver.digest(savedpath),
            red_sha256=v.driver.digest(OUT/'red_masks'/METHOD/p/'03_text_red.png'))
        rows.append(row);print(c['id'],row['red_page_pixels'],'restored',row['restored_old_pixels'],'excluded old ROI',row['excluded_old_roi_pixels'],flush=True)
    v.driver.save_json(OUT/'mask_diagnostics.json',rows)

def infer():
    p='200021063_00002_2';method='howe_'+METHOD
    assert np.array_equal(v.review.load(OUT/'red_masks'/METHOD/p/'03_text_red.png'),v.review.load(PARENT/'masks/howe_grow5_region'/p/'03_text_red.png'))
    source=PARENT/'outputs/howe_grow5_region'/p/'result'/(p+'.png');dest=OUT/'outputs'/method/p/'result'/(p+'.png')
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    (OUT/'logs').mkdir(exist_ok=True);shutil.copyfile(PARENT/'logs/F04_howe_grow5_region.log',OUT/'logs'/('F04_'+method+'.log'))
    parentproto=json.loads((PARENT/'ddrm_protocol_grow5_region.json').read_text())
    for key,path in [('pipeline_sha256',v.driver.PIPELINE),('checkpoint_sha256',v.driver.CHECKPOINT),('background_sha256',v.driver.BACKGROUND)]:assert parentproto[key]==v.driver.digest(path)
    v.driver.save_json(OUT/'F04_reuse_provenance.json',dict(parent_source=v.driver.relative(source),source_sha256=v.driver.digest(source),reason='Same raw Howe and exactly same red mask, input and frozen inference'))
    v.OUT=OUT;v.infer(METHOD,['F04','R02','R01','R03'])
    for name in ['02_text_black.png','03_text_red.png']:
        assert v.driver.digest(OUT/'masks'/method/p/name)==v.driver.digest(PARENT/'masks/howe_grow5_region'/p/name)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else infer()
