"""Posthoc development refinement: require a region with substantial red seeds.

Parameters are shared across all ten observed pages; not an unbiased final test.
"""
import argparse,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np
from PIL import Image
import validate_local_red as v

PARENT=v.OUT
OUT=PARENT/'regional_refinement'
METHOD='grow5_region'

def prepare():
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=True)
    proto=json.loads((PARENT/'protocol.json').read_text())
    original=proto['variants']['grow5']
    proto.update(created_utc=datetime.now(timezone.utc).isoformat(),variants={METHOD:dict(**original,region_seed_radius=16,region_min_seed_pixels=128)},
        stage='Posthoc region support after observing F03 and R04 small false-positive fragments',
        parent_protocol_sha256=v.driver.digest(PARENT/'protocol.json'),script_sha256=v.driver.digest(Path(__file__)))
    assert not (OUT/'protocol.json').exists()
    v.driver.save_json(OUT/'protocol.json',proto)
    records=[]
    for c in proto['cases']:
        p=c['page'];raw=np.asarray(Image.open(v.ROOT/c['source']).convert('RGB'))
        delta,chroma=v.color_fields(raw)
        anchorpath=v.old.OUT/p/'red_without_global_gate.png'
        if not anchorpath.exists():anchorpath=PARENT/'anchors'/p/'red_without_global_gate.png'
        anchor=v.review.load(anchorpath)
        seed=anchor&chroma&(delta>=8)
        region=cv2.dilate(seed.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)))>0
        n,labels=cv2.connectedComponents(region.astype(np.uint8),connectivity=8)
        counts=np.bincount(labels[seed],minlength=n);keep=counts>=128;keep[0]=False
        before=v.review.load(PARENT/'red_masks/grow5'/p/'03_text_red.png');red=before&keep[labels]
        v.old.save(OUT/'red_masks'/METHOD/p/'03_text_red.png',red)
        saved=v.review.load(v.ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p/'03_text_red.png')
        arrays=[raw];titles=['原画像']
        for m,label in [(saved,'現行CSC赤（緑）'),(before,'局所赤・領域条件なし（緑）'),(red,'局所赤・領域条件あり（緑）')]:
            a=raw.copy();a[m]=[20,235,20];arrays.append(a);titles.append(label)
        v.review.figure(OUT/'mask_figures'/(c['id']+'_領域支持_赤比較.png'),arrays,titles,c['roi_xywh'],c['id'])
        for k,r in c['controls_xywh'].items():v.review.figure(OUT/'mask_controls'/(c['id']+'_'+k+'_赤比較.png'),arrays,titles,r,c['id'])
        row=dict(id=c['id'],page=p,red_page_pixels=int(red.sum()),red_roi_pixels=v.roi_count(red,c['roi_xywh']),
            removed_previous_pixels=int((before&~red).sum()),controls={k:v.roi_count(red,r) for k,r in c['controls_xywh'].items()},
            region_seed_counts=counts[keep].tolist(),input_sha256=c['source_sha256'],anchor_sha256=v.driver.digest(anchorpath),
            parent_red_sha256=v.driver.digest(PARENT/'red_masks/grow5'/p/'03_text_red.png'),red_sha256=v.driver.digest(OUT/'red_masks'/METHOD/p/'03_text_red.png'))
        records.append(row);print(c['id'],json.dumps(row,ensure_ascii=False),flush=True)
    v.driver.save_json(OUT/'mask_diagnostics.json',records)
    sheet=Image.new('RGB',(2000,2400),'#f4f2ec')
    for i,c in enumerate(proto['cases'][:5]):
        im=Image.open(OUT/'mask_figures'/(c['id']+'_領域支持_赤比較.png')).convert('RGB');im.thumbnail((1980,465));sheet.paste(im,(10,480*i+10))
    sheet.save(OUT/'5例_局所赤と領域支持_比較.png')

def infer():
    # F04 regional refinement is pixel-identical to parent grow5, so reuse only
    # after proving both inference masks, input, code and checkpoint agree.
    p='200021063_00002_2'
    a=v.review.load(OUT/'red_masks'/METHOD/p/'03_text_red.png')
    b=v.review.load(PARENT/'masks/howe_grow5'/p/'03_text_red.png')
    assert np.array_equal(a,b)
    result=PARENT/'outputs/howe_grow5'/p/'result'/(p+'.png')
    assert result.exists(), 'Run parent inference first'
    m='howe_'+METHOD
    dest=OUT/'outputs'/m/p/'result'/(p+'.png');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(result,dest)
    (OUT/'logs').mkdir(exist_ok=True)
    shutil.copyfile(PARENT/'logs/F04_howe_grow5.log',OUT/'logs'/('F04_'+m+'.log'))
    parentlog=json.loads((PARENT/'logs/F04_howe_grow5.json').read_text())
    expected=dict(pipeline_sha256=v.driver.digest(v.driver.PIPELINE),checkpoint_sha256=v.driver.digest(v.driver.CHECKPOINT),background_sha256=v.driver.digest(v.driver.BACKGROUND))
    parentproto=json.loads((PARENT/'ddrm_protocol_grow5.json').read_text())
    assert all(parentproto[k]==value for k,value in expected.items())
    v.driver.save_json(OUT/'F04_reuse_provenance.json',dict(parent_log=parentlog,parent_result_sha256=v.driver.digest(result),
        parent_red_sha256=v.driver.digest(PARENT/'masks/howe_grow5'/p/'03_text_red.png'),new_red_sha256=v.driver.digest(OUT/'red_masks'/METHOD/p/'03_text_red.png'),
        reason='Identical raw Howe, red pixels, input and frozen DDRM; no new F04 inference needed'))
    v.OUT=OUT
    v.infer(METHOD,['F04','R02','R01','R03'])
    assert v.driver.digest(OUT/'masks'/m/p/'02_text_black.png')==v.driver.digest(PARENT/'masks/howe_grow5'/p/'02_text_black.png')
    assert v.driver.digest(OUT/'masks'/m/p/'03_text_red.png')==v.driver.digest(PARENT/'masks/howe_grow5'/p/'03_text_red.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else infer()
