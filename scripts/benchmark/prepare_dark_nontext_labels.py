"""Source-reviewed nontext stain and clear-paper boxes, identical size/shape."""
import json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from compare_v5_aligned_fade_control import rgb,sha,dump
from screen_dark_nontext_sources import ROOT,OUT
SELECT=[('S01',133,138),('S02',150,163),('S03',140,125),('S05',134,133),('S06',122,132),('S07',99,129),('S08',128,135),('S11',126,132)]
def main():
    assert not (OUT/'protocol.json').exists();candidates=json.loads((OUT/'candidate_sources.json').read_text())['cases'];rows=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    for j,(key,cx,cy) in enumerate(SELECT):
        c=next(c for c in candidates if c['id']==key);raw=rgb(ROOT/c['crop']);g=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY);masks={};stain_box=[cx-12,cy-12,24,24];clear_box=[32,32,24,24]
        for arm,box in [('clear',clear_box),('stain',stain_box)]:
            x,y,w,h=box;mask=np.zeros((256,256),np.uint8);mask[y:y+h,x:x+w]=255;dest=OUT/'paper_masks'/arm/f'{key}.png';dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(mask).save(dest);masks[arm]=mask>0
        assert not (masks['clear']&masks['stain']).any() and masks['clear'].sum()==masks['stain'].sum()==576
        over=raw.copy();over[masks['clear']]=np.rint(.3*raw[masks['clear']]+.7*np.array([0,190,100])).astype(np.uint8);over[masks['stain']]=np.rint(.3*raw[masks['stain']]+.7*np.array([255,0,180])).astype(np.uint8)
        canvas=Image.new('RGB',(1024,555),'white');d=ImageDraw.Draw(canvas);d.text((0,0),key+' 原画像 / 緑=明るい紙 紫=濃い非文字（各24×24画素、2倍最近傍）',font=font,fill='black');canvas.paste(Image.fromarray(raw).resize((512,512),Image.Resampling.NEAREST),(0,35));canvas.paste(Image.fromarray(over).resize((512,512),Image.Resampling.NEAREST),(512,35));canvas.save(OUT/f'{key}_原画像と部分教師.png')
        rows.append(dict(**c,stain_box_xywh=stain_box,clear_box_xywh=clear_box,n_each=576,stain_gray_median=float(np.median(g[masks['stain']])),clear_gray_median=float(np.median(g[masks['clear']])),mask_sha256={arm:sha(OUT/'paper_masks'/arm/f'{key}.png') for arm in masks}))
    dump(OUT/'selected_sources.json',dict(cases=rows,status='Only chosen clear and stain boxes labeled as paper. Outside unknown, visible ink in context remains unlabeled. Original source pixels never changed.'))
    dump(OUT/'label_preparation.json',dict(n_sources=8,n_books=len({r['book'] for r in rows}),teacher_pixels_each_arm=4608,mask_shape=[24,24],coefficient_each_arm=1,decision='Choose8 isolated nontext artifacts in source margins across6 training books; exclude ambiguous S12 ink marks, cover, broad low-contrast S09 and duplicate-page S04. Source-only review, before any new predictions.',status='Provisional source-reviewed nontext negative labels, not expertGT. No new positive ink labels or RGB repair. Adjacent clear paper may be included in stain box; supervised region not exact stain segmentation.'))
    print([(r['id'],r['clear_gray_median'],r['stain_gray_median']) for r in rows],flush=True)
if __name__=='__main__':main()
