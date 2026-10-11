"""Known digital layers, deliberately overlapping color distributions.

Extracted source contrast is a procedural shape bank, NOT annotated real ink.
Labels refer only to added synthetic layers. No real restoration GT is claimed.
"""
from pathlib import Path
import json
import hashlib
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PRIOR=ROOT/'tmp_work/npj_contact_research_20261007/synthetic'
FONT='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
CHARSET='春秋山水月花風人心詩書天地文化物語遠近上下左右千百一二三四五六七八九十'

def file_hash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def local_paper(rgb):
    h,w=rgb.shape[:2]
    small=cv2.resize(rgb,(max(1,w//2),max(1,h//2)),interpolation=cv2.INTER_AREA)
    closed=cv2.morphologyEx(small,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)),borderType=cv2.BORDER_REFLECT)
    bg=cv2.resize(cv2.GaussianBlur(closed.astype(np.float32),(0,0),2),(w,h),interpolation=cv2.INTER_LINEAR)
    return np.maximum(bg,rgb.astype(np.float32))

def input_features(rgb,mode='relative'):
    raw=rgb.astype(np.float32)/255
    if mode=='rgb': return raw.transpose(2,0,1).copy()
    bg=local_paper(rgb)/255
    od=np.log((bg+1/255)/(raw+1/255))
    # No known synthesis background or labels enter the features.
    return np.concatenate([raw,od,bg],axis=2).transpose(2,0,1).copy()

def load_bank(split):
    rows=json.loads((PRIOR/'patches.json').read_text())
    protocol=json.loads((OUT/'cases_frozen.json').read_text())
    excluded=set(protocol['development_books']+protocol['validation_books']+['200010454'])
    good=[r for r in rows if r['split']==split and r['book'] not in excluded]
    data=np.load(PRIOR/'patches.npy',mmap_mode='r')
    return np.stack([data[r['index']] for r in good]),good

def shape_bank(patches):
    bank=[]
    for a in patches:
        gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32)
        bg=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
        contrast=np.clip((bg-gray-22)/40,0,1)
        # Exclude weak paper grain and isolated speckles from the shape bank.
        count,labels,stats,_=cv2.connectedComponentsWithStats((contrast>.25).astype(np.uint8),8)
        accepted=np.zeros(count,bool)
        if count>1:accepted[1:]=(stats[1:,cv2.CC_STAT_AREA]>=12)&(stats[1:,cv2.CC_STAT_AREA]<16000)
        contrast*=accepted[labels]
        contrast=cv2.GaussianBlur(contrast,(0,0),.45)
        bank.append(contrast)
    return bank

def warped(alpha,rng):
    n=alpha.shape[0]
    angle=float(rng.uniform(-25,25)); scale=float(rng.uniform(.6,1.35))
    mat=cv2.getRotationMatrix2D((n/2,n/2),angle,scale)
    mat[:,2]+=rng.uniform(-45,45,2)
    return cv2.warpAffine(alpha,mat,(n,n),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def glyphs(rng,n,kind):
    # Known generated structures, not readings of any historical writing.
    im=Image.new('L',(n*2,n*2),0); d=ImageDraw.Draw(im)
    if kind=='seal':
        x,y=rng.integers(12,70,2); ww=int(rng.integers(80,165)); hh=int(rng.integers(90,180))
        d.rectangle((int(x*2),int(y*2),int((x+ww)*2),int((y+hh)*2)),outline=255,width=int(rng.integers(2,7))*2)
        fs=int(min(ww/2,hh/3)*rng.uniform(.7,.95))
        font=ImageFont.truetype(FONT,fs*2)
        for ix in range(2):
            for iy in range(3):
                d.text((int((x+5+ix*ww/2)*2),int((y+iy*hh/3)*2)),str(rng.choice(list(CHARSET))),font=font,fill=255)
    elif kind=='line':
        xx=float(rng.uniform(25,n-25)); ys=np.linspace(rng.uniform(5,40),rng.uniform(n-50,n-5),10)
        pts=[(int((xx+8*math.sin(k/4))*2),int(y*2)) for k,y in enumerate(ys)]
        d.line(pts,fill=255,width=int(rng.integers(1,5))*2)
    elif kind=='punctuation':
        for _ in range(int(rng.integers(2,6))):
            x,y=rng.uniform(15,n-15,2); rad=float(rng.uniform(1.5,5))
            d.ellipse((int((x-rad)*2),int((y-rad)*2),int((x+rad)*2),int((y+rad)*2)),fill=255)
    else:
        font=ImageFont.truetype(FONT,int(rng.integers(28,65))*2)
        for _ in range(int(rng.integers(2,6))):
            x,y=rng.integers(-10,n-30,2)
            d.text((int(x*2),int(y*2)),str(rng.choice(list(CHARSET))),font=font,fill=255)
    return np.asarray(im.resize((n,n),Image.Resampling.LANCZOS),dtype=np.float32)/255

def stains(rng,n,stress=False):
    yy,xx=np.mgrid[:n,:n].astype(np.float32)
    v=np.zeros((n,n),np.float32)
    for _ in range(int(rng.integers(1,5))):
        x,y=rng.uniform(-20,n+20,2)
        major=rng.uniform(15,110); minor=rng.uniform(6,65)
        angle=rng.uniform(0,np.pi); u=(xx-x)*np.cos(angle)+(yy-y)*np.sin(angle); z=-(xx-x)*np.sin(angle)+(yy-y)*np.cos(angle)
        q=(u/major)**2+(z/minor)**2
        blob=np.exp(-q*2)
        if rng.random()<.4:
            # Sharp/ring boundaries are hard negatives for simple ridges.
            blob=np.clip((1-q)*4,0,1)
            if rng.random()<.5: blob*=1-.65*np.exp(-q*8)
        v=np.maximum(v,blob*float(rng.uniform(.35,1)))
    if rng.random()<.35 or stress:
        k=float(rng.uniform(0.8,4)); location=float(rng.uniform(10,n-10)); slope=float(rng.uniform(-.5,.5))
        fold=np.exp(-((xx-location-slope*yy)/k)**2/2)
        v=np.maximum(v,fold*float(rng.uniform(.4,1)))
    if stress:
        for _ in range(12):
            x,y=rng.uniform(0,n,2); rad=float(rng.uniform(1,4));v=np.maximum(v,np.exp(-((xx-x)**2+(yy-y)**2)/(2*rad*rad))*.6)
    noise=cv2.resize(rng.random((16,16),dtype=np.float32),(n,n),interpolation=cv2.INTER_CUBIC)
    return np.clip(v*(.65+.35*noise),0,1)

def color_direction(rng,red_biased=False):
    # RGB OD: red can be faint red-brown. Brown and red supports overlap.
    if red_biased:
        direction=np.array([rng.uniform(.02,.15),rng.uniform(.5,.8),rng.uniform(.65,1)],np.float32)
    else:
        direction=np.array([rng.uniform(.08,.6),rng.uniform(.45,.85),rng.uniform(.65,1)],np.float32)
    return direction/max(direction)

class LayerDataset(Dataset):
    def __init__(self,split='train',length=20000,mode='relative',training='overlap',seed=20261008,stress=False,paired=False):
        cv2.setNumThreads(1)
        self.patches,self.records=load_bank(split); self.shapes=shape_bank(self.patches)
        self.papers=np.stack([local_paper(a).clip(30,255).astype(np.uint8) for a in self.patches])
        self.length=length; self.mode=mode; self.training=training; self.seed=seed;self.stress=stress;self.paired=paired
    def __len__(self): return self.length
    def sample(self,index,variant=0):
        rng=np.random.default_rng(self.seed+index*104729)
        n=256; paper=self.papers[int(rng.integers(len(self.papers)))].astype(np.float32)/255
        # All source shapes are imposed digital geometry, irrespective of origin.
        black=warped(self.shapes[int(rng.integers(len(self.shapes)))],rng)
        if rng.random()<.55:
            mark=warped(self.shapes[int(rng.integers(len(self.shapes)))],rng);kind='source_contrast_shape'
        else:
            kind=str(rng.choice(['seal','text','line','punctuation']));mark=glyphs(rng,n,kind)
        if rng.random()<.12: mark*=0
        stain=stains(rng,n,self.stress)
        if rng.random()<.15:stain*=0
        # Same-color samples defeat a purely chromatic shortcut.
        mark_dir=color_direction(rng,self.training=='separated')
        stain_dir=color_direction(rng)
        if self.training=='separated':
            stain_dir=np.array([rng.uniform(.4,.7),rng.uniform(.6,.85),1],np.float32)
        elif rng.random()<.45: stain_dir=mark_dir.copy()
        opacity=float(np.exp(rng.uniform(np.log(.025),np.log(.8))))
        stain_opacity=float(np.exp(rng.uniform(np.log(.035),np.log(.7))))
        if variant:
            # Recolor without changing geometry or semantic labels.
            recolor=np.random.default_rng(self.seed+index*104729+911)
            mark_dir=color_direction(recolor);stain_dir=color_direction(recolor)
            if recolor.random()<.45:stain_dir=mark_dir.copy()
        mark_od=mark[:,:,None]*opacity*mark_dir
        stain_od=stain[:,:,None]*stain_opacity*stain_dir
        black_od=black[:,:,None]*float(rng.uniform(.3,1.8))*np.array([.93,1.,.99],np.float32)
        clean=paper*np.exp(-mark_od-black_od)
        rgb=np.clip(paper*np.exp(-mark_od-stain_od-black_od)*255+rng.normal(0,.45,(n,n,3)),0,255).astype(np.uint8)
        target=np.stack([mark,stain],axis=0).astype(np.float32)
        return rgb,target,(clean*255).clip(0,255).astype(np.uint8),dict(kind=kind,mark_opacity=opacity,stain_opacity=stain_opacity,
            same_color=bool(np.allclose(mark_dir,stain_dir)),seed=self.seed+index*104729)
    def __getitem__(self,index):
        rgb,target,_,_=self.sample(index)
        x=torch.from_numpy(input_features(rgb,self.mode));y=torch.from_numpy(target)
        if self.paired:
            other,yt,_,_=self.sample(index,variant=1)
            assert np.array_equal(target,yt)
            return x,torch.from_numpy(input_features(other,self.mode)),y
        return x,y

def audit():
    from collections import Counter
    dest=OUT/'synthetic_audit';dest.mkdir(exist_ok=True)
    p=json.loads((OUT/'cases_frozen.json').read_text())
    splits={}
    for split in ['train','dev','test']:
        _,rows=load_bank(split);splits[split]=sorted({r['book'] for r in rows})
        (dest/(split+'_sources.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    all_sets=[set(v) for v in splits.values()]
    assert all(not all_sets[i]&all_sets[j] for i in range(3) for j in range(i))
    assert not set.union(*all_sets)&set(p['validation_books'])
    dataset=LayerDataset('train',64)
    canvas=Image.new('RGB',(4*256,6*300),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT,15)
    infos=[]
    for i in range(12):
        rgb,y,clean,meta=dataset.sample(i);infos.append(meta)
        row=i//2; xx=(i%2)*512;yy=row*300
        canvas.paste(Image.fromarray(rgb),(xx,yy));canvas.paste(Image.fromarray(clean),(xx+256,yy))
        draw.text((xx,yy+260),f'{i} input / known added-stain removed',font=font,fill='black')
        draw.text((xx,yy+279),str(meta['kind']),font=font,fill='black')
        np.savez_compressed(dest/f'sample_{i:02d}.npz',rgb=rgb,target=y,clean=clean)
    canvas.save(dest/'合成入力と既知の除去対象.png')
    protocol=dict(seed=20261008,native_patch=256,splits=splits,samples_preview=infos,
        source_patches_sha256=file_hash(PRIOR/'patches.npy'),source_manifest_sha256=file_hash(PRIOR/'patches.json'),
        synthetic_module_sha256=file_hash(__file__),
        GT='Known digitally imposed mark and stain alpha layers only; shapes extracted from real contrast have no real ink GT',
        inputs='RGB or RGB+input-estimated OD+input-estimated paper; never known clean/GT paper',
        real_validation_books_excluded=p['validation_books'],
        limitation='Additive optical density is a controlled approximation, not verified physics of actual paper or red pigment',
        no_real_handwriting_GT=True)
    (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print(json.dumps(dict(books=splits,previews=12),ensure_ascii=False),flush=True)

if __name__=='__main__':audit()
