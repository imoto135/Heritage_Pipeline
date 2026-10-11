from pathlib import Path
import json, hashlib
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'tmp_work/npj_contact_stain_search_20261007'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return np.asarray(Image.open(path).convert('RGB')).copy()
def write(path,a):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 Image.fromarray(np.clip(np.rint(a),0,255).astype('uint8')).save(path)
def cases():return json.loads((SOURCE/'benchmark_candidates.json').read_text())['cases']
def gray(a):return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype('float32')/255
def sauvola(a,window=61,k=.2):
 g=gray(a);m=cv2.boxFilter(g,-1,(window,window),borderType=cv2.BORDER_REFLECT_101)
 s=np.sqrt(np.maximum(cv2.boxFilter(g*g,-1,(window,window),borderType=cv2.BORDER_REFLECT_101)-m*m,0))
 t=m*(1+k*(s/.5-1));return g<t
def morph_contrast(a,radius=15):
 g=gray(a);kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
 b=cv2.morphologyEx(g,cv2.MORPH_CLOSE,kernel)
 return np.clip((b-g)/np.maximum(b,.1),0,1),b
def ink_render(alpha,paper=None):
 if paper is None:paper=np.array([235,224,200],dtype='float32')
 return paper*(1-alpha[...,None])
def card(path,arrays,labels,title,footer='',size=480):
 n=len(arrays);width=n*(size+18)+24;header=112;foot=106
 canvas=Image.new('RGB',(width,header+size+foot),'#faf9f5');d=ImageDraw.Draw(canvas)
 f=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,18)
 d.text((16,10),title,font=f,fill='#20313b')
 for i,(a,label) in enumerate(zip(arrays,labels)):
  im=Image.fromarray(np.clip(np.rint(a),0,255).astype('uint8'));im.thumbnail((size,size),Image.Resampling.NEAREST)
  x=16+i*(size+18);d.text((x,60),label,font=small,fill='black');canvas.paste(im,(x,header))
 for j,line in enumerate(footer.split('\n')):d.text((16,header+size+12+j*25),line,font=small,fill='#53636d')
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
