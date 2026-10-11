from pathlib import Path
import json,csv,hashlib
from collections import Counter
from PIL import Image,ImageDraw,ImageFont
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'tmp_work/npj_failure_collection_20261001'
rows={r['page']:r for r in csv.DictReader((OLD/'page_inventory.csv').open())}
scores=list(csv.DictReader((OLD/'screening_scores.csv').open()))
scores.sort(key=lambda s:-int(s['brown_blob_score']))
choices=[];counts=Counter()
for s in scores[:350]:
 r=rows[s['page']]
 if counts[r['book']]>=3:continue
 choices.append(s);counts[r['book']]+=1
 if len(choices)>=54:break
# Extend to more pages of the source collection of the initial contact-stain case.
for s in [s for s in scores if s['page'].startswith('hnsd')][:24]:
 if s['page'] not in {c['page'] for c in choices}:choices.append(s)
# Near-text brown spot noted in previous original/mask-only review.
for page in ['200021869_00014_2']:
 if page not in {c['page'] for c in choices}:choices.append(next(s for s in scores if s['page']==page))
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(FONT,24)
(HERE/'screening').mkdir(exist_ok=True)
prepared=[]
for idx,s in enumerate(choices):
 r=rows[s['page']];im=Image.open(ROOT/r['original']).convert('RGB');W,H=im.size
 x,y,w,h=json.loads(s['brown_blob_thumb_xywh']);tw,th=int(s['thumbnail_width']),int(s['thumbnail_height'])
 anchor=[round(x*W/tw),round(y*H/th),round(w*W/tw),round(h*H/th)]
 if r['page']=='200021869_00014_2':anchor=[1070,1700,190,200]
 cx,cy=anchor[0]+anchor[2]/2,anchor[1]+anchor[3]/2
 side=768;xx=min(max(0,round(cx-side/2)),W-side);yy=min(max(0,round(cy-side/2)),H-side)
 xywh=[xx,yy,side,side]
 prepared.append(dict(id=f'P{idx+1:02}',**r,score=int(s['brown_blob_score']),anchor_xywh=anchor,crop_xywh=xywh,
   original_sha256=hashlib.sha256((ROOT/r['original']).read_bytes()).hexdigest(),
   provenance='original + existing CSC v1 black mask only; not restored NPJ output'))
for start in range(0,len(prepared),3):
 canvas=Image.new('RGB',(1540,1850),'#faf9f5');d=ImageDraw.Draw(canvas)
 for j,r in enumerate(prepared[start:start+3]):
  yy=j*610;x,y,w,h=r['crop_xywh'];box=(x,y,x+w,y+h)
  im=Image.open(ROOT/r['original']).convert('RGB');m=Image.open(ROOT/r['npj_black']).convert('RGB')
  assert im.size==m.size
  d.text((20,yy+8),f'{r["id"]} {r["page"]} score={r["score"]} crop={r["crop_xywh"]}',font=font,fill='#202d38')
  d.text((20,yy+45),'原画像',font=font,fill='black');d.text((555,yy+45),'保存黒文字マスク（復元出力ではない）',font=font,fill='black')
  for img,px in [(im,20),(m,555)]:
   crop=img.crop(box);crop.thumbnail((512,512),Image.Resampling.LANCZOS if img is im else Image.Resampling.NEAREST)
   canvas.paste(crop,(px,yy+83))
  thumb=im.copy();thumb.thumbnail((350,460));td=ImageDraw.Draw(thumb);sx,sy=thumb.width/im.width,thumb.height/im.height
  td.rectangle((x*sx,y*sy,(x+w)*sx,(y+h)*sy),outline='red',width=3);canvas.paste(thumb,(1140,yy+83))
 canvas.save(HERE/'screening'/f'sheet_{start//3+1:02}.png')
(HERE/'screening.json').write_text(json.dumps(dict(original_score_corpus_pages=len(rows),screened_candidates=prepared,
 ranking_source=str((OLD/'screening_scores.csv').relative_to(ROOT)),selection='Top350 brown blobs with 3/book cap up to54, plus top24 hnsd pages and previous near-text spot',
 objective='Find brown stains crossing visible writing; illustrations, warm ink, binding shadows, isolated stains excluded by visual review'),ensure_ascii=False,indent=2))
print('original/mask candidate pages',len(prepared),'sheets',(len(prepared)+2)//3)
