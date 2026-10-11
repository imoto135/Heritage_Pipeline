from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
frames={f['id']:f for f in json.loads((HERE/'frames.json').read_text())}
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font=ImageFont.truetype(FONT,23)
rois=[
('S01','manual_control_hnsd007_042',[300,150,320,400]),
('S02','crop_hnsd',[0,270,540,330]),
('L01','known_200010454_v2',[1230,500,280,650]),
('L02','page_200019865',[1750,2490,280,550]),
('L03','saved_200018243_00002_2',[350,800,350,700]),
('L04','new_warm_faint_writing',[480,0,270,650]),
('L05a','new_warm_upper_loss',[390,290,360,450]),
('L05b','new_warm_lower_loss',[0,120,480,450]),
('N01','old_full_page',[760,1870,220,480]),
('R01','seal_control',[200,50,330,400]),
('N02','unseen_lower_binding_mark',[360,0,370,400])]
(HERE/'draft').mkdir(exist_ok=True)
for id,frame,roi in rois:
 f=frames[frame]; a=Image.open(ROOT/f['input']).convert('RGB'); b=Image.open(ROOT/f['output']).convert('RGB')
 if id=='S02': a=Image.open(ROOT/'tmp_work/npj_dirt_control_20261002/input/hnsd_raw_control.png').convert('RGB')
 x,y,w,h=roi; box=(x,y,x+w,y+h)
 scale=max(1,min(2,1000//h,850//w))
 a=a.crop(box).resize((w*scale,h*scale),Image.Resampling.NEAREST);b=b.crop(box).resize(a.size,Image.Resampling.NEAREST)
 canvas=Image.new('RGB',(w*scale*2+90,h*scale+110),'#ffffff');d=ImageDraw.Draw(canvas)
 d.text((20,10),f'{id} {f["page"]} {roi}',font=font,fill='black')
 d.text((20,48),'原画像（実入力と一致）',font=font,fill='black');d.text((w*scale+55,48),'NPJ 保存出力',font=font,fill='black')
 canvas.paste(a,(20,90));canvas.paste(b,(w*scale+55,90));canvas.save(HERE/'draft'/f'{id}.png')
(HERE/'draft_rois.json').write_text(json.dumps(rois,ensure_ascii=False,indent=2))
print('draft comparisons',len(rois))
