"""Classical black foreground comparison at native full-page resolution.

True means foreground. Gatos is the native M=1 version, Su is an explicitly
documented local reconstruction; neither is claimed to match author code.
"""
from pathlib import Path
import hashlib
import json
import time
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.special import expit
from skimage.feature import canny
from skimage.filters import threshold_otsu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
PRIMARY = ['sauvola61','wolf61','nick61','gatos_native','su2013_local']
LABELS = {'original':'原画像', 'sauvola61':'Sauvola 61 / k=.2',
 'wolf61':'Wolf–Jolion 61 / k=.5', 'nick61':'NICK 61 / k=-.1',
 'gatos_native':'Gatos 再実装 / 原寸 M=1', 'su2013_local':'Su 2013 再実装',
 'nick19':'NICK 19 / k=-.1', 'niblack61':'Niblack 61 / k=-.2',
 'otsu':'大津（全体閾値）', 'saved_v2':'保存CSC v2（ACE経路）',
 'sauvola31':'Sauvola 31 / k=.2', 'sauvola91':'Sauvola 91 / k=.2',
 'sauvola61_k01':'Sauvola 61 / k=.1', 'wolf61_k02':'Wolf 61 / k=.2',
 'su2013_initial':'Su 再実装 / 後処理前', 'gatos_initial':'Gatos 再実装 / 後処理前'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def read(p):return np.asarray(Image.open(p).convert('RGB')).copy()
def gray(a):return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype('float32')/255
def box(a,w):return cv2.boxFilter(a,-1,(w,w),borderType=cv2.BORDER_REFLECT_101)
def stats(g,w):
    m=box(g,w); v=np.maximum(box(g*g,w)-m*m,0)
    return m,np.sqrt(v),box(g*g,w)
def sauvola(g,w=61,k=.2):
    m,s,_=stats(g,w);return g<m*(1+k*(s/.5-1))
def wolf(g,w=61,k=.5):
    m,s,_=stats(g,w);R=float(s.max());M=float(g.min())
    t=m-k*(1-s/max(R,1e-8))*(m-M)
    return g<t,dict(window=w,k=k,global_min=M,global_max_std=R)
def nick(g,w=61,k=-.1):
    m,_,second=stats(g,w)
    t=m+k*np.sqrt(np.maximum(second-m*m/(w*w),0))
    return g<t

def remove_isolated(b):
    n=ndi.convolve(b.astype('uint8'),np.ones((3,3),dtype='uint8'),mode='constant')
    return b & (n>1)

def odd(v,minimum=3):
    w=max(minimum,int(round(v)));return w if w%2 else w+1

def gatos(g):
    # Eq3: classical local Wiener filter, floor negative variance/noise gains.
    m,s,_=stats(g,3);var=s*s;noise=float(var.mean())
    I=m+np.maximum(var-noise,0)/np.maximum(var,1e-10)*(g-m)
    S=sauvola(I,61,.2)
    nonink=(~S).astype('float32')
    den=box(nonink,181);num=box(I*nonink,181)
    # Background pixels remain observed; only foreground is interpolated (Eq4).
    B=np.where(S,num/np.maximum(den,1e-10),I)
    empty=S & (den<1e-8)
    # Undefined interpolation (no background in neighborhood) uses nearest known
    # background; this guard is recorded separately, not silently clipped to zero.
    if empty.any():
        inds=ndi.distance_transform_edt(S,return_distances=False,return_indices=True)
        B[empty]=I[tuple(inds[:,empty])]
    delta=float((B-I).sum(dtype='float64')/max(1,int(S.sum())))
    b=float(B[~S].mean()) if (~S).any() else float(I.mean())
    # Eq7 uses (1-S), despite inconsistent nearby prose in the paper.
    d=.6*delta*(.8+.2*expit(4*B/(max(b,1e-8)*.5)-2*1.5/.5))
    initial=(B-I)>d
    # Paper section3.6: estimate character height from component height mode.
    _,_,cs,_=cv2.connectedComponentsWithStats(initial.astype('uint8'),8)
    heights=cs[1:,cv2.CC_STAT_HEIGHT]
    # Exclude tiny punctuation/noise and whole page edges from height estimation.
    valid=heights[(heights>=10)&(heights<=min(g.shape)//4)]
    lh=int(np.bincount(valid).argmax()) if len(valid) else 20
    n=odd(.15*lh);N=n*n
    def count(a):return ndi.convolve(a.astype('float32'),np.ones((n,n),'float32'),mode='reflect')
    C=count(initial)
    shrink=initial & ((N-C)<=.9*N)
    C=count(shrink)
    # Symmetric displacement kernels provide centroid relative to center.
    rr=np.arange(-(n//2),n//2+1,dtype='float32')
    dx=ndi.correlate(shrink.astype('float32'),np.tile(rr,(n,1)),mode='reflect')/np.maximum(C,1)
    dy=ndi.correlate(shrink.astype('float32'),np.tile(rr[:,None],(1,n)),mode='reflect')/np.maximum(C,1)
    swell=shrink | ((C>.05*N)&(np.abs(dx)<.25*n)&(np.abs(dy)<.25*n))
    final=swell | (count(swell)>.35*N)
    meta=dict(noise_variance=noise,delta=delta,mean_background=b,background_window=181,
        undefined_background_pixels=int(empty.sum()),dominant_component_height=lh,post_window=n,
        native_upsampling_factor=1,initial_pixels=int(initial.sum()),final_pixels=int(final.sum()))
    return final,initial,meta

def su(g,override_window=None):
    mx=ndi.maximum_filter(g,size=3,mode='reflect')
    mn=ndi.minimum_filter(g,size=3,mode='reflect')
    gradient=mx-mn;contrast=gradient/np.maximum(mx+mn,1e-8)
    alpha=float(g.std()/ (128/255))  # gamma=1 (paper Eq4, 8bit Std/128)
    adaptive=alpha*contrast+(1-alpha)*gradient
    otsu=float(threshold_otsu(adaptive))
    # MATLAB-like automatic Canny threshold selection, explicitly not MATLAB parity.
    smooth=ndi.gaussian_filter(g,math.sqrt(2),mode='reflect')
    gx=ndi.sobel(smooth,axis=1,mode='reflect');gy=ndi.sobel(smooth,axis=0,mode='reflect')
    # skimage Canny uses the same unnormalized scipy Sobel magnitude.
    mag=np.hypot(gx,gy); high=float(np.quantile(mag,.7))
    can=canny(g,sigma=math.sqrt(2),low_threshold=.4*high,high_threshold=high,mode='reflect')
    edge=(adaptive>otsu)&can
    # Algorithm1: preceding background-to-edge transitions, brighter than next pixel.
    distances=[]
    for y in range(g.shape[0]):
        start=np.flatnonzero((~edge[y,:-1])&edge[y,1:]&(g[y,:-1]>=g[y,1:]))
        if len(start)>1:distances.extend(np.diff(start).tolist())
    widths=np.asarray(distances,dtype='int32')
    ew=int(np.bincount(widths).argmax()) if len(widths) else 3
    w=odd(2*ew) if override_window is None else int(override_window)
    ef=edge.astype('float32');den=box(ef,w)
    em=box(ef*g,w)/np.maximum(den,1e-10)
    es=np.sqrt(np.maximum(box(ef*g*g,w)/np.maximum(den,1e-10)-em*em,0))
    initial=(den>0)&(g<=em+es/2)
    # Algorithm2 raster-order relabeling of symmetric neighbor pairs.
    isolated_edge=remove_isolated(edge)
    final=initial.copy();h,ww=g.shape
    yy,xx=np.nonzero(isolated_edge)
    for y,x in zip(yy,xx):
        if 0<y<h-1 and 0<x<ww-1:
            for y1,x1,y2,x2 in [(y-1,x,y+1,x),(y,x-1,y,x+1)]:
                if final[y1,x1]==final[y2,x2] and g[y1,x1]!=g[y2,x2]:
                    lower=g[y1,x1]<g[y2,x2]
                    final[y1,x1]=lower;final[y2,x2]=not lower
    final=remove_isolated(final)
    meta=dict(alpha=alpha,gamma=1,contrast_otsu=otsu,canny_high=high,canny_low=.4*high,
        edge_pixels=int(edge.sum()),estimated_horizontal_edge_width=ew,threshold_window=w,
        edge_pair_samples=len(widths),window_override=override_window,
        initial_pixels=int(initial.sum()),final_pixels=int(final.sum()),
        implementation_limit='Canny uses skimage with automatic thresholds; final removal implements isolated pixels, not every boundary logical operator referenced in the paper. Author MATLAB p-code not executed.')
    return final,initial,meta

def binary(b):return np.repeat((~b)[...,None],3,axis=2).astype('uint8')*255
def save(p,a):
    p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(a.astype('uint8')).save(p)
def text(d,xy,t,size=20):d.text(xy,t,font=ImageFont.truetype(FONT,size),fill='#20313b')

def sheet(cid,raw,crops,keys,dest,title):
    scale=3 if cid=='N01' else 1 if cid=='T01' else 2
    h,w=raw.shape[:2];pw,ph=w*scale,h*scale
    columns=3;rows=math.ceil(len(keys)/columns);gap=22;top=105;header=45
    canvas=Image.new('RGB',(columns*(pw+gap)+gap,top+rows*(ph+header+gap)+80),'#faf9f5')
    d=ImageDraw.Draw(canvas);text(d,(18,14),title,26)
    text(d,(18,53),'同じ座標・整数倍表示。黒＝保存候補、白＝背景。朱とDDRMは含めない。',19)
    placements=[]
    for i,k in enumerate(keys):
        x=gap+(i%columns)*(pw+gap);y=top+(i//columns)*(ph+header+gap)
        text(d,(x,y),LABELS[k],21)
        a=raw if k=='original' else binary(crops[k])
        im=Image.fromarray(a).resize((pw,ph),Image.Resampling.NEAREST)
        canvas.paste(im,(x,y+header))
        assert np.array_equal(np.asarray(canvas)[y+header:y+header+ph,x:x+pw],np.repeat(np.repeat(a,scale,0),scale,1))
        placements.append(dict(method=k,xywh=[x,y+header,pw,ph],integer_scale=scale))
    text(d,(18,canvas.height-61),'ページ全体で計算してから切り出し。Gatos・Suは条件を記録した再実装。',19)
    text(d,(18,canvas.height-33),'開発用の選択範囲。精度の総合順位や原論文の完全再現を示す比較ではない。',18)
    dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest)
    return dict(path=str(dest.relative_to(HERE)),sha256=sha(dest),placements=placements)

def sanity():
    rng=np.random.default_rng(71);g=rng.random((13,15)).astype('float32');m,s,q=stats(g,3)
    for y,x in [(3,5),(8,8)]:
        v=g[y-1:y+2,x-1:x+2].astype('float64')
        assert abs(m[y,x]-v.mean())<1e-6
        assert abs(s[y,x]-v.std())<1e-6
        assert nick(g,3)[y,x]==(g[y,x]<v.mean()-.1*math.sqrt(float((v*v).mean())-v.mean()**2/9))
    # Uniform paper produces no foreground in three local methods.
    flat=np.full((50,60),.75,'float32')
    assert not sauvola(flat).any() and not nick(flat).any() and not wolf(flat)[0].any()
    return dict(local_moments_checked_against_direct_windows=True,uniform_paper_checks_passed=True)

def main():
    cv2.setNumThreads(2)
    protocol=json.loads((HERE/'protocol.json').read_text())
    annotation=json.loads((HERE/'annotations_frozen.json').read_text())
    ann={a['id']:a for a in annotation['cases']}
    checks=sanity(); records=[];panels=[];runtimes=[];parity=[]
    for c in protocol['cases']:
        cid=c['id'];p=ROOT/c['original'];assert sha(p)==c['source_sha256']
        raw=read(p);g=gray(raw);x,y,w,h=c['roi_xywh'];sl=np.s_[y:y+h,x:x+w]
        print('Start',cid,raw.shape,flush=True);masks={};details={}
        methods={
            'sauvola61':lambda:sauvola(g), 'wolf61':lambda:wolf(g),
            'nick61':lambda:nick(g), 'nick19':lambda:nick(g,19),
            'gatos_native':lambda:gatos(g), 'su2013_local':lambda:su(g),
            'sauvola31':lambda:sauvola(g,31), 'sauvola91':lambda:sauvola(g,91),
            'sauvola61_k01':lambda:sauvola(g,61,.1),'wolf61_k02':lambda:wolf(g,61,.2),
            'otsu':lambda:g<float(threshold_otsu(g)),
            'niblack61':lambda:g<(stats(g,61)[0]-.2*stats(g,61)[1])}
        for k,fn in methods.items():
            start=time.monotonic();res=fn()
            if k in ['gatos_native','su2013_local']:
                b,initial,meta=res
                ik='gatos_initial' if k=='gatos_native' else 'su2013_initial'
                masks[ik]=initial;details[k]=meta
            elif k.startswith('wolf'):
                b,meta=res;details[k]=meta
            else:b=res
            assert b.shape==g.shape and b.dtype==bool
            masks[k]=b
            dt=time.monotonic()-start;runtimes.append(dict(id=cid,method=k,seconds=dt))
            print(cid,k,round(dt,2),int(b[sl].sum()),flush=True)
        if c.get('saved_v2'):masks['saved_v2']=np.asarray(Image.open(ROOT/c['saved_v2']).convert('L'))<127
        if c.get('saved_sauvola'):
            prior=np.asarray(Image.open(ROOT/c['saved_sauvola']).convert('L'))<127
            diff=int((prior!=masks['sauvola61']).sum());assert diff==0
            parity.append(dict(id=cid,saved_sauvola_difference_pixels=diff))
        crops={k:b[sl] for k,b in masks.items()}
        for k,b in masks.items():
            save(HERE/'masks'/cid/(k+'.png'),binary(b))
            save(HERE/'crops'/cid/(k+'.png'),binary(b[sl]))
            a=ann[cid];hits=[bool(b[sl][yy,xx]) for xx,yy in a['ink_points_xy']]
            negative=np.zeros((h,w),bool)
            for xx,yy,ww,hh in a['nonink_regions_xywh']:negative[yy:yy+hh,xx:xx+ww]=True
            leak=int(b[sl][negative].sum());count=int(negative.sum())
            records.append(dict(id=cid,method=k,label=LABELS[k],ink_points_kept=sum(hits),ink_points_total=len(hits),
                ink_point_hits=hits,selected_nonink_pixels=leak,selected_nonink_total=count,
                selected_nonink_fraction=leak/count,roi_foreground_pixels=int(b[sl].sum()),
                crop_sha256=sha(HERE/'crops'/cid/(k+'.png'))))
        panels.append(sheet(cid,raw[sl],crops,['original']+PRIMARY,HERE/'figures'/(cid+'_古典5手法.png'),cid+' 黒文字：古典5手法'))
        keys=['original','sauvola61','sauvola61_k01','sauvola31','sauvola91','wolf61_k02',
              'nick19','niblack61','otsu']
        panels.append(sheet(cid,raw[sl],crops,keys,HERE/'figures'/(cid+'_設定感度と対照.png'),cid+' 設定感度と対照'))
        keys=['original','gatos_initial','gatos_native','su2013_initial','su2013_local']
        if 'saved_v2' in crops:keys.append('saved_v2')
        panels.append(sheet(cid,raw[sl],crops,keys,HERE/'figures'/(cid+'_後処理とCSC対照.png'),cid+' 後処理と既存CSCの対照'))
        dump(HERE/'method_details'/(cid+'.json'),details)
    dump(HERE/'measurements.json',dict(annotation_limits=annotation['caveat'],records=records))
    dump(HERE/'rendering.json',panels)
    dump(HERE/'complete.json',dict(cases=len(protocol['cases']),records=len(records),checks=checks,
        sauvola_parity=parity,runtimes=runtimes,
        input_hashes_unchanged=all(sha(ROOT/c['original'])==c['source_sha256'] for c in protocol['cases']),
        protocol_sha256=sha(HERE/'protocol.json'),annotation_sha256=sha(HERE/'annotations_frozen.json'),
        source_sha256=sha(__file__)))
    print('Complete',len(records),flush=True)

if __name__=='__main__':main()
