from pathlib import Path
import csv,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/test';OUT=Path(__file__).resolve().parent
FONT=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
rows=[]
for book in sorted(BASE.iterdir()):
 if not book.is_dir():continue
 files=sorted(book.glob('*.jpg'))
 thumbs=[]
 for path in files:
  bgr=cv2.imread(str(path),cv2.IMREAD_REDUCED_COLOR_8)
  if bgr is None:continue
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  h,w=rgb.shape[:2];roi=rgb[int(h*.12):int(h*.88),int(w*.1):int(w*.9)].astype('float32')
  lab=cv2.cvtColor(roi.astype('uint8'),cv2.COLOR_RGB2LAB).astype('float32')
  # Source-only descriptors for shortlisting; none is an artifact label.
  chroma=float(np.sqrt((lab[:,:,1]-128)**2+(lab[:,:,2]-128)**2).mean())
  color_var=float(np.std(cv2.GaussianBlur(lab[:,:,2],(0,0),8)))
  dark=float((cv2.cvtColor(roi.astype('uint8'),cv2.COLOR_RGB2GRAY)<90).mean())
  blue=float(((roi[:,:,2]-roi[:,:,0]>-10)&(roi.mean(2)>85)&(roi.mean(2)<210)).mean())
  rows.append(dict(book=book.name,page=path.stem,original=str(path.relative_to(ROOT)),chroma=round(chroma,2),color_var=round(color_var,2),dark_fraction=round(dark,4),bluegray_fraction=round(blue,4)))
  im=Image.fromarray(rgb);im.thumbnail((210,280),Image.Resampling.LANCZOS)
  thumbs.append((path.stem,im.copy()))
 for part in range((len(thumbs)+79)//80):
  batch=thumbs[part*80:(part+1)*80];tw,th=230,320;sheet=Image.new('RGB',(tw*8,th*10),'#f5f4f0');d=ImageDraw.Draw(sheet)
  for i,(stem,im) in enumerate(batch):
   x=i%8*tw;y=i//8*th
   sheet.paste(im,(x+(tw-im.width)//2,y+23));d.text((x+5,y+3),stem[-7:],font=FONT,fill='#1d3040')
  sheet.save(OUT/f'{book.name}_part{part+1:02d}.jpg',quality=86)
 print(book.name,len(files),'pages',flush=True)
with (OUT/'original_features.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('total',len(rows),flush=True)
