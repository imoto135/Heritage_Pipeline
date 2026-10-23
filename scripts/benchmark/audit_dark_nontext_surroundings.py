"""Supplemental nontext surroundings, source-only, never extra training labels."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import evaluate_dark_nontext_teacher_control as ev
import evaluate_dark_nontext_source_context as ctx
from compare_v5_aligned_fade_control import rgb,mask,sha,dump
ROOT=ev.ROOT;OUT=ev.OUT

def freeze():
    ev.check();ctx.check();assert not (OUT/'surroundings_plan.json').exists();assert not list((OUT/'evaluation').glob('*/results.json'));cs=json.loads((OUT/'selected_sources.json').read_text())['cases'];rows=[];files=[Path(__file__)];canvas=Image.new('RGB',(4*268,2*290),'white');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
    for i,c in enumerate(cs):
        x,y,w,h=c['stain_box_xywh'];box=[x-20,y-20,64,64];assert box[0]>=0 and box[1]>=0 and box[0]+64<=256 and box[1]+64<=256
        m=np.zeros((256,256),np.uint8);m[box[1]:box[1]+64,box[0]:box[0]+64]=255;f=OUT/'source_guard/nontext_surroundings'/f"{c['id']}.png";f.parent.mkdir(exist_ok=True);Image.fromarray(m).save(f);files.append(f);rows.append(dict(id=c['id'],box_xywh=box,paper_n=4096,mask_file=str(f.relative_to(ROOT)),sha256=sha(f)))
        im=Image.fromarray(rgb(ROOT/c['crop']));dd=ImageDraw.Draw(im);dd.rectangle((box[0],box[1],box[0]+63,box[1]+63),outline=(0,200,70),width=1);dd.rectangle((x,y,x+w-1,y+h-1),outline=(255,0,180),width=1);xx=i%4*268;yy=i//4*290;d.text((xx,yy),c['id']+' 緑=補助64 紫=教師24',font=font,fill='black');canvas.paste(im,(xx,yy+28))
    canvas.save(OUT/'source_guard/非文字の周辺領域_原画像確認.png')
    dump(OUT/'surroundings_plan.json',dict(status='Source-only provisional nontext64x64 surroundings, fixed before model readouts. Additional source-fit diagnostic, never training. Original outside24 remains unknown in training; these are separate evaluation masks.',purpose='Detect stain-edge/halo remaining outside taught center. Not an exact full-stain segmentation, boundaries or held-out GT.',rows=rows,protocol_sha256=sha(OUT/'protocol.json'),source_context_plan_sha256=sha(OUT/'source_context_plan.json'),frozen_files={str(f.relative_to(ROOT)):sha(f) for f in files}))
    print('Prepared8 supplemental nontext surroundings,4096pixels each',flush=True)
def check():
    ev.check();ctx.check();p=json.loads((OUT/'surroundings_plan.json').read_text());assert p['protocol_sha256']==sha(OUT/'protocol.json')
    for n,h in p['frozen_files'].items():assert sha(ROOT/n)==h,n
    return p

def run():
    p=check();rows=[]
    for seed in [0,1]:
      for arm in ev.tr.ARMS:
        name=f'ragan_seed{seed}_{arm}'
        for c in p['rows']:
          m=np.asarray(Image.open(ROOT/c['mask_file']))>0
          for grid in ['native256_fit',*ev.GRIDS]:
            f=OUT/'evaluation'/name/'stain_teacher_fit'/f"{c['id']}.png" if grid=='native256_fit' else OUT/'source_context'/name/'masks'/grid/f"{c['id']}.png";a=mask(f);rows.append(dict(condition=name,case=c['id'],grid=grid,paper_n=4096,paper_ink=int(np.count_nonzero(a[m])),mask_sha256=sha(f)))
    assert len(rows)==128;dump(OUT/'surroundings_results.json',dict(rows=rows,plan_sha256=sha(OUT/'surroundings_plan.json'),same_training_sources=True,no_training_use=True,formal_accuracy=False));print('Scored128 source nontext-surroundings records',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);v=a.parse_args();freeze() if v.action=='freeze' else run()
