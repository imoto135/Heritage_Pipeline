from common import *
from render_full import records
import csv,time
ids=['T00','T01','T02','T03','T04','T06','N01','N02','F39','F48','F60','F62'];lookup={c['id']:c for c in records()}
canvas=Image.new('RGB',(1680,3030),'#faf9f5');draw=ImageDraw.Draw(canvas);f=ImageFont.truetype(FONT,22);sm=ImageFont.truetype(FONT,18)
draw.text((24,20),'本文接触の候補12ページ：原ページと比較位置',font=f,fill='#20313b');draw.text((24,58),'候補は成功数・シミの正解分類ではない。赤枠は表示位置、処理マスクではない。',font=sm,fill='#53636d');rows=[]
for i,id in enumerate(ids):
 c=lookup[id];path=ROOT/c['source']['original'];im=Image.open(path).convert('RGB');w,h=im.size;im.thumbnail((510,610),Image.Resampling.LANCZOS)
 x=(i%3)*560+24;y=(i//3)*715+105;draw.text((x,y),id+' '+c['page'],font=f,fill='black');px=x+(510-im.width)//2;py=y+42;canvas.paste(im,(px,py))
 xx,yy,ww,hh=c['detail_page_xywh'];roi=[xx,yy,ww,hh]
 if id=='N01':roi=[428,347,190,220]
 if id=='F62':roi=[xx+560,yy+300,180,270]
 xx,yy,ww,hh=roi;draw.rectangle((px+xx*im.width/w,py+yy*im.height/h,px+(xx+ww)*im.width/w,py+(yy+hh)*im.height/h),outline='#cf4c39',width=3)
 draw.text((x,y+665),'追加候補' if id.startswith(('N','F')) else '以前の候補',font=sm,fill='#53636d');rows.append(dict(id=id,page=c['page'],source_group=c['page'].split('_')[0],original=str(path.relative_to(ROOT)),display_page_xywh=roi,original_sha256=sha(path),material_GT=False,reading_GT=False))
canvas.save(HERE/'figures/07_contact_page_contexts.png');dump(HERE/'contact_candidates.json',dict(created=time.time(),candidates=rows,source_sha256=sha(__file__),groups=len({r['source_group'] for r in rows})))
with (HERE/'contact_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('contact page contexts',len(rows))
