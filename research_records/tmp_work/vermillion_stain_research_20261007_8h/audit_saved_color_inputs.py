"""Distinguish saved classification input from newly frozen CPU ACE.

Same v2 algorithm name does not establish identical input or saved masks.
"""
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def main():
    protocol=json.loads((OUT/'cases_frozen.json').read_text());dest=OUT/'saved_color_input_audit';dest.mkdir(exist_ok=True)
    records=[];pages={r['page']:r for r in protocol['cases']}
    for page,r in pages.items():
        original=ROOT/r['original'];split=original.parts[-3]
        oldraw=ROOT/'data/split_dataset'/split/r['book']/(page+'.jpg')
        masks=ROOT/'data/split_dataset_csc_mask'/split/r['book']/page
        masks2=ROOT/'data/split_dataset_csc_mask_v2'/split/r['book']/page
        newdir=OUT/'fullpage_baselines'/page
        entry=dict(page=page,book=r['book'],original=r['original'],old_raw_exists=oldraw.exists(),
            saved_v1_exists=(masks/'00_input.png').exists(),saved_v2_exists=(masks2/'03_text_red.png').exists())
        if not oldraw.exists():records.append(entry);continue
        original_rgb=rgb(original);old_rgb=rgb(oldraw)
        entry['decoded_original_identical']=np.array_equal(original_rgb,old_rgb)
        entry['old_raw_sha256']=file_hash(oldraw)
        if not entry['decoded_original_identical']:
            entry['usable_saved_masks_for_current_image']=False;records.append(entry);continue
        entry['usable_saved_masks_for_current_image']=True
        if not (masks/'00_input.png').exists() or not (newdir/'ace.png').exists():records.append(entry);continue
        old=rgb(masks/'00_input.png');new=rgb(newdir/'ace.png');assert old.shape==new.shape==original_rgb.shape
        entry.update(saved_classifier_input=str((masks/'00_input.png').relative_to(ROOT)),
            saved_classifier_input_sha256=file_hash(masks/'00_input.png'),
            new_ACE_sha256=file_hash(newdir/'ace.png'),ACE_identical=np.array_equal(old,new),
            ACE_mean_abs_change_255=float(np.abs(old.astype(float)-new).mean()))
        hsvs=[cv2.cvtColor(a,cv2.COLOR_RGB2HSV) for a in [original_rgb,old,new]]
        entry['strict_red_pixels']={key:int((((hsv[:,:,0]<=12)|(hsv[:,:,0]>=168))&(hsv[:,:,1]>180)).sum())
            for key,hsv in zip(['original','saved_classifier_input','CPU_ACE_seed0_n200'],hsvs)}
        for kind,prefix in [('v1',masks),('v2',masks2)]:
            saved=prefix/'03_text_red.png';fresh=newdir/('csc_'+kind+'_red.png')
            if saved.exists() and fresh.exists():
                sm=np.asarray(Image.open(saved).convert('L'))<127;fm=np.asarray(Image.open(fresh).convert('L'))>127
                assert sm.shape==fm.shape
                entry[kind]=dict(saved_red_pixels=int(sm.sum()),new_red_pixels=int(fm.sum()),different_pixels=int((sm!=fm).sum()),
                    saved_mask_path=str(saved.relative_to(ROOT)),saved_mask_sha256=file_hash(saved))
                # Keep saved masks as separate observational baselines, white=positive.
                Image.fromarray(sm.astype(np.uint8)*255).save(newdir/('saved_csc_'+kind+'_red.png'))
        for roi in [c for c in protocol['cases'] if c['page']==page]:
            x,y,w,h=roi['ROI'];sl=np.s_[y:y+h,x:x+w]
            if roi['case_id'] not in ['dev_06','dev_08','dev_10']:continue
            inputs=[original_rgb[sl],old[sl],new[sl]]
            labels=['原画像','保存済み分類入力','CPU ACE・seed0・200参照']
            sheet=Image.new('RGB',(w*3,h+90),'#faf9f5');d=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,18)
            d.text((10,8),roi['name']+'／色補正入力の違い',font=font,fill='#20323b')
            for j,(im,label) in enumerate(zip(inputs,labels)):
                d.text((j*w+4,44),label,font=font,fill='#20323b');sheet.paste(Image.fromarray(im),(j*w,90))
            sheet.save(dest/(roi['case_id']+'_色補正入力.png'))
        records.append(entry)
    (dest/'audit.json').write_text(json.dumps(dict(records=records,
        warning='Generated CPU-ACE+CSC v2 and saved CSC v2 are distinct conditions; no assumption of published-mask reproduction',
        source_sha256=file_hash(__file__)),ensure_ascii=False,indent=2))
    for r in records:
        if 'ACE_mean_abs_change_255' in r:print(r['page'],'ACE MAE',round(r['ACE_mean_abs_change_255'],1),'strict red',r['strict_red_pixels'],flush=True)

if __name__=='__main__':main()
