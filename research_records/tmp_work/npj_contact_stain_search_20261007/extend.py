from pathlib import Path
import json,csv,hashlib
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'tmp_work/npj_failure_collection_20261001'
meta=json.loads((HERE/'screening.json').read_text()); oldids={r['page'] for r in meta['screened_candidates']}
rows={r['page']:r for r in csv.DictReader((OLD/'page_inventory.csv').open())}
scores=sorted([s for s in csv.DictReader((OLD/'screening_scores.csv').open()) if s['page'].startswith('hnsd007')],key=lambda s:-int(s['brown_blob_score']))
new=[]
for s in scores[:16]:
 if s['page'] in oldids:continue
 r=rows[s['page']];im=Image.open(ROOT/r['original']).convert('RGB');W,H=im.size
 x,y,w,h=json.loads(s['brown_blob_thumb_xywh']);tw,th=int(s['thumbnail_width']),int(s['thumbnail_height'])
 anchor=[round(x*W/tw),round(y*H/th),round(w*W/tw),round(h*H/th)]
 cx,cy=anchor[0]+anchor[2]/2,anchor[1]+anchor[3]/2
 xywh=[min(max(0,round(cx-384)),W-768),min(max(0,round(cy-384)),H-768),768,768]
 new.append(dict(id=f'P{76+len(new):02}',**r,score=int(s['brown_blob_score']),anchor_xywh=anchor,crop_xywh=xywh,
 original_sha256=hashlib.sha256((ROOT/r['original']).read_bytes()).hexdigest(),provenance='original and saved CSC v1 mask only'))
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',24)
for start in range(0,len(new),3):
 canvas=Image.new('RGB',(1540,1850),'#faf9f5');d=ImageDraw.Draw(canvas)
 for j,r in enumerate(new[start:start+3]):
  yy=j*610;x,y,w,h=r['crop_xywh'];box=(x,y,x+w,y+h);im=Image.open(ROOT/r['original']).convert('RGB');m=Image.open(ROOT/r['npj_black']).convert('RGB')
  d.text((20,yy+8),f'{r["id"]} {r["page"]} crop={r["crop_xywh"]}',font=font,fill='black')
  d.text((20,yy+45),'原画像',font=font,fill='black');d.text((555,yy+45),'保存黒文字マスク（復元出力ではない）',font=font,fill='black')
  for img,px in [(im,20),(m,555)]:
   t=img.crop(box);t.thumbnail((512,512));canvas.paste(t,(px,yy+83))
  t=im.copy();t.thumbnail((350,460));td=ImageDraw.Draw(t);sx,sy=t.width/im.width,t.height/im.height
  td.rectangle((x*sx,y*sy,(x+w)*sx,(y+h)*sy),outline='red',width=3);canvas.paste(t,(1140,yy+83))
 canvas.save(HERE/'screening'/f'extra_{start//3+1:02}.png')
meta['screened_candidates']+=new;meta['extension']='Top16 hnsd007-specific ranking; only unreviewed pages added'
(HERE/'screening.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print('added',len(new),'total',len(meta['screened_candidates']))
