"""Original-only coordinate sheets for audit. Never render predictions here."""
from pathlib import Path
import json
import hashlib
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
F = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def main():
    rows = json.loads((OUT/'relative_red_screen/queue.json').read_text())
    dest=OUT/'native_candidate_inspection'; dest.mkdir(exist_ok=True)
    records=[]
    for idx in [25,36,52,54,81,82,83]:
        row=rows[idx]; im=Image.open(ROOT/row['original']).convert('RGB')
        w,h=im.size
        # Overview coordinates show original pixel positions.
        preview=im.copy(); preview.thumbnail((1000,1250))
        s=preview.width/w
        canvas=Image.new('RGB',(preview.width,preview.height+80),'white')
        canvas.paste(preview,(0,80)); d=ImageDraw.Draw(canvas)
        d.text((10,10),f"{idx} {row['id']}  {w}x{h}",font=ImageFont.truetype(F,20),fill='black')
        d.text((10,40),'原画像のみ・500px間隔／予測を見ず候補を選ぶ',font=ImageFont.truetype(F,20),fill='black')
        for x in range(0,w,500):
            xx=int(x*s); d.line((xx,80,xx,80+preview.height),fill='#63b0c0',width=1)
            d.text((xx+3,84),str(x),font=ImageFont.truetype(F,16),fill='#184353')
        for y in range(0,h,500):
            yy=80+int(y*s); d.line((0,yy,preview.width,yy),fill='#63b0c0',width=1)
            d.text((3,yy+3),str(y),font=ImageFont.truetype(F,16),fill='#184353')
        canvas.save(dest/f"{idx:02d}_{row['id']}_overview.png")
        # Four broad quadrants at native pixels preserve faint markings.
        for j,(x,y) in enumerate([(0,0),(w//2,0),(0,h//2),(w//2,h//2)]):
            box=(x,y,min(w,x+w//2+80),min(h,y+h//2+80))
            im.crop(box).save(dest/f"{idx:02d}_{row['id']}_q{j}.png")
        records.append(dict(**row,width=w,height=h,source_sha256=hashlib.sha256((ROOT/row['original']).read_bytes()).hexdigest()))
    (dest/'sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print('Original coordinate inspection:',len(records),'pages')

if __name__=='__main__': main()
