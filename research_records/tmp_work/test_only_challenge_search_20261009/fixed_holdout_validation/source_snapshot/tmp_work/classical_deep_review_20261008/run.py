"""Frozen classical-method probes on full native pages, black foreground only.

Howe's base energy is ported from the author's public MATLAB code. Canny is
skimage with normalized Sobel magnitude, not MATLAB canny_old; automatic
parameter selection is not run. Preserve that distinction in comparisons.
"""
from pathlib import Path
import hashlib
import importlib.util
import itertools
import json
import math
import struct
import subprocess
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.filters import threshold_otsu

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'tmp_work/classical_black_comparison_20261008'
SCRATCH=HERE/'scratch';SCRATCH.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('classical_previous',OLD/'compare.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
LABELS={
 'original':'原画像','wolf61':'Wolf / 窓61・k=.5','sauvola61':'Sauvola / 窓61・k=.2',
 'nick61':'NICK / 窓61・k=-.1','nick19':'NICK / 窓19・k=-.1',
 'gatos_native':'Gatos / 前回の原寸再実装','su2013_local':'Su / 前回の再実装',
 'bradley61':'Bradley / 窓61・t=15%', 'bradley_paper':'Bradley / 窓=幅の約1/8',
 'bradley61_t10':'Bradley / 窓61・t=10%', 'phansalkar61':'Phansalkar / 窓61',
 'phansalkar31':'Phansalkar / 窓31',
 'bg31_otsu':'背景補正31→大津','bg61_otsu':'背景補正61→大津','bg121_otsu':'背景補正121→大津',
 'bg61_wolf':'背景補正61→Wolf','bg61_sauvola':'背景補正61→Sauvola',
 'bg61_nick':'背景補正61→NICK',
 'howe80':'Howe基本エネルギー / c=80', 'howe160':'Howe基本エネルギー / c=160',
 'howe320':'Howe基本エネルギー / c=320',
 'background':'推定背景（灰色）','normalized':'背景で割った入力（灰色）',
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2))
def read(p):return np.asarray(Image.open(p).convert('RGB')).copy()
def save_mask(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.where(b,0,255).astype('uint8')).save(p)
def read_mask(p):return np.asarray(Image.open(p).convert('L'))==0

def cut(src,snk,down,right):
    h,w=src.shape
    assert snk.shape==(h,w) and down.shape==(h-1,w) and right.shape==(h,w-1)
    inp=SCRATCH/'graph_input.bin';out=SCRATCH/'graph_output.bin'
    with inp.open('wb') as f:
        f.write(struct.pack('<II',h,w))
        for a in (src,snk,down,right):np.asarray(a,dtype='<f8').tofile(f)
    result=subprocess.run([str(HERE/'graphcut_cli'),str(inp),str(out)],check=True,capture_output=True,text=True)
    b=np.fromfile(out,dtype='uint8').reshape(h,w).astype(bool)
    return b,float(result.stdout.strip())

def energy(b,src,snk,down,right):
    return float(np.where(b,src,snk).sum()+ (down*(b[:-1]!=b[1:])).sum()+(right*(b[:,:-1]!=b[:,1:])).sum())

def checks():
    rng=np.random.default_rng(124)
    tested=[]
    for shape in [(2,3),(3,3)]:
        src=rng.uniform(0,10,shape);snk=rng.uniform(0,10,shape)
        down=rng.uniform(0,5,(shape[0]-1,shape[1]));right=rng.uniform(0,5,(shape[0],shape[1]-1))
        b,flow=cut(src,snk,down,right)
        best=min(energy(np.array(bits,bool).reshape(shape),src,snk,down,right) for bits in itertools.product([False,True],repeat=math.prod(shape)))
        got=energy(b,src,snk,down,right)
        assert abs(got-best)<1e-8 and abs(got-flow)<1e-8
        tested.append(dict(shape=list(shape),enumerated_labelings=2**math.prod(shape),exact_energy=got))
    g=rng.uniform(.2,.9,(15,17)).astype('float32')
    m,s,_=base.stats(g,3)
    for yy,xx in [(4,4),(7,9)]:
        a=g[yy-1:yy+2,xx-1:xx+2].astype('float64')
        assert abs(float(m[yy,xx])-a.mean())<1e-6
        assert abs(float(s[yy,xx])-a.std())<1e-6
        direct=a.mean()*(1+2*np.exp(-10*a.mean())+.25*(a.std()/.5-1))
        assert abs(float(phansalkar_threshold(g,3)[yy,xx])-direct)<1e-6
    flat=np.full((100,110),.75,'float32')
    assert not (flat<base.box(flat,61)*.85).any()
    assert not (flat<phansalkar_threshold(flat,61)).any()
    B,n=background_normalize(flat,31)
    assert np.allclose(n,1) and np.allclose(B,flat)
    return dict(graphcut_exhaustive_energy_checks=tested,local_statistics_direct_check=True,
                phansalkar_direct_formula_check=True,uniform_paper_checks=True)

def phansalkar_threshold(g,w):
    m,s,_=base.stats(g,w)
    return m*(1+2*np.exp(-10*m)+.25*(s/.5-1))

def background_normalize(g,size):
    # Independent classical construction, not a claimed reproduction of Gatos
    # or the 2018 morphology/energy paper. Estimate stained paper using closing.
    B=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((size,size),'uint8'),borderType=cv2.BORDER_REFLECT_101)
    assert np.all(B>=g-1e-6)
    return B,np.clip(g/np.maximum(B,1/255),0,1).astype('float32')

def howe_terms(g):
    I=(g.astype('float64')*255)
    dx=I[:-1,1:]-I[:-1,:-1];dy=I[1:,:-1]-I[:-1,:-1]
    # MATLAB divergence uses centered differences, one-sided at the boundary.
    lap=np.gradient(dx,axis=1)+np.gradient(dy,axis=0)
    residual=I-ndi.gaussian_filter(I,20,mode='reflect',radius=60)
    rms=np.sqrt(ndi.gaussian_filter(residual**2,20,mode='constant',cval=0,radius=50))
    confident=residual/(rms+np.finfo(float).eps)>2
    lap[confident[:-1,:-1]]=-500
    sm=ndi.gaussian_filter(g,.6,mode='reflect',truncate=4)
    magnitude=np.hypot(ndi.sobel(sm,axis=0,mode='reflect'),ndi.sobel(sm,axis=1,mode='reflect'))
    maxmag=float(magnitude.max())
    edges=canny(g,sigma=.6,low_threshold=.1*maxmag,high_threshold=.4*maxmag,mode='reflect')
    down=~((edges[:-1,:-1]&(dy>0))|(edges[1:,:-1]&(dy<=0)))
    right=~((edges[:-1,:-1]&(dx>0))|(edges[:-1,1:]&(dx<=0)))
    meta=dict(canny_sigma=.6,canny_low_relative=.1,canny_high_relative=.4,
        canny_max_sobel_magnitude=maxmag,canny_edge_pixels=int(edges.sum()),
        background_bias_pixels=int(confident[:-1,:-1].sum()),laplacian_min=float(lap.min()),laplacian_max=float(lap.max()),
        implementation='Port of author IJDAR base energy, with skimage Canny normalized by maximum Sobel magnitude. MATLAB canny_old and automatic parameter tuning not reproduced.')
    return (1500-lap,1500+lap,down[:-1,:].astype('float64'),right[:,:-1].astype('float64')),meta

def howe(terms,c):
    src,snk,down,right=terms
    b,flow=cut(src,snk,c*down,c*right)
    actual=energy(b,src,snk,c*down,c*right)
    assert abs(actual-flow)<max(1e-5,abs(flow)*1e-9)
    expanded=np.pad(b,((0,1),(0,1)),mode='edge')
    above=np.concatenate([expanded[:1],expanded[:-1]],axis=0)
    left=np.concatenate([expanded[:,:1],expanded[:,:-1]],axis=1)
    final=expanded&~(expanded&~above&~left)
    return final,dict(c=c,mincut_energy=actual,solver_flow=flow,foreground_before_post=int(expanded.sum()),foreground_after_post=int(final.sum()))

def sheet(case,raw,masks,keys,name,continuous=None):
    x,y,w,h=case['roi_xywh'];sl=np.s_[y:y+h,x:x+w]
    scale=2 if case['id']=='N01' else 1
    pw,ph=w*scale,h*scale
    gap,top,heading=24,118,44
    canvas=Image.new('RGB',(2*pw+3*gap,top+2*(ph+heading+gap)+100),'#faf9f5')
    d=ImageDraw.Draw(canvas)
    def txt(pos,t,n):d.text(pos,t,font=ImageFont.truetype(FONT,n),fill='#20313b')
    txt((18,14),case['id']+' 黒文字：'+name,25)
    txt((18,53),'同じ範囲。原画像入力で全ページを処理。黒＝文字候補、白＝背景。',17)
    txt((18,79),'N01だけ整数2倍。T01・N05は原寸。表示の平滑化・輪郭強調なし。',17)
    placement=[]
    for i,k in enumerate(keys):
        px=gap+(i%2)*(pw+gap);py=top+(i//2)*(ph+heading+gap)
        txt((px,py),LABELS[k],20)
        if k=='original':a=raw[sl]
        elif continuous is not None and k in continuous:
            a=np.repeat(np.clip(continuous[k][sl]*255,0,255).astype('uint8')[:,:,None],3,axis=2)
        else:a=np.repeat(np.where(masks[k][sl],0,255).astype('uint8')[:,:,None],3,axis=2)
        tile=np.repeat(np.repeat(a,scale,axis=0),scale,axis=1)
        canvas.paste(Image.fromarray(tile),(px,py+heading))
        assert np.array_equal(np.asarray(canvas)[py+heading:py+heading+ph,px:px+pw],tile)
        placement.append(dict(method=k,canvas_xywh=[px,py+heading,pw,ph],integer_scale=scale))
    txt((18,canvas.height-81),'Howeは基本エネルギーのローカル移植。著者のCanny・自動調整との一致は未確認。',15)
    txt((18,canvas.height-55),'背景補正はグレーのclosingで推定した背景で割る独立した比較条件。',16)
    txt((18,canvas.height-29),'探索用の選択範囲。正解画素マスクに基づく精度や総合順位ではない。',16)
    dest=HERE/'figures'/(case['id']+'_'+name+'.png');dest.parent.mkdir(exist_ok=True)
    canvas.save(dest)
    return dict(path=str(dest.relative_to(HERE)),sha256=sha(dest),panels=placement)

def main():
    cv2.setNumThreads(2)
    prior=json.loads((OLD/'protocol.json').read_text())
    cases=[]
    for c in prior['cases']:
        c=dict(c)
        if c['id']=='T01':c['roi_xywh']=[760,360,600,600]
        if c['id']=='N05':c['roi_xywh']=[1019,1617,640,580]
        cases.append(c)
    protocol=dict(cases=cases,input='Full native raw page, OpenCV RGB2GRAY /255; crop only after processing; no ACE, red or DDRM.',
        predefined_conditions={
            'Bradley':dict(windows=[61,'nearest odd(width/8)'],relative_margin=[.15,.10],border='reflect101'),
            'Phansalkar':dict(windows=[61,31],k=.25,R=.5,p=2,q=10,variance='population'),
            'background_normalization':dict(closing_shape='square',sizes=[31,61,121],normalization='clip(I/max(B,1/255),0,1)',thresholds=['Otsu','Wolf61 k=.5','Sauvola61 k=.2','NICK61 k=-.1']),
            'Howe_base_local_port':dict(c=[80,160,320],sigmaE=.6,tlo=.1,thi=.4,implementation_limit='Skimage Canny with max-Sobel normalization, not MATLAB canny_old; no automatic tuning.'),
        },annotation_scope='Reuse 52 audited exploratory ink centers and 9 selected non-ink rectangles; no dense GT, no unbiased accuracy.',
        settings_frozen_before_new_outputs=True)
    dump(HERE/'protocol.json',protocol)
    verify=checks()
    ann=json.loads((OLD/'annotations_frozen.json').read_text())
    audit=json.loads((OLD/'annotation_audit.json').read_text())
    annotation={a['id']:a for a in ann['cases']}
    oldcases={c['id']:c for c in prior['cases']}
    records=[];runtimes=[];figures=[];details={};sources={}
    for c in cases:
        cid=c['id'];p=ROOT/c['original'];assert sha(p)==c['source_sha256']
        sources[str(p.relative_to(ROOT))]=sha(p)
        raw=read(p);g=base.gray(raw);masks={}
        def add(key,fn):
            started=time.monotonic();result=fn();meta={}
            if isinstance(result,tuple):b,meta=result
            else:b=result
            assert b.shape==g.shape and b.dtype==bool
            masks[key]=b
            dest=HERE/'masks'/cid/(key+'.png');save_mask(dest,b)
            assert np.array_equal(read_mask(dest),b)
            seconds=time.monotonic()-started
            runtimes.append(dict(id=cid,method=key,seconds=seconds))
            details[cid][key]=meta
            print(cid,key,round(seconds,2),flush=True)
        details[cid]={}
        for key in ['wolf61','sauvola61','nick61','nick19','gatos_native','su2013_local']:
            mp=OLD/'masks'/cid/(key+'.png');masks[key]=read_mask(mp);sources[str(mp.relative_to(ROOT))]=sha(mp)
        add('bradley61',lambda:g<base.box(g,61)*.85)
        bw=base.odd(g.shape[1]/8)
        add('bradley_paper',lambda:g<base.box(g,bw)*.85)
        details[cid]['bradley_paper']['window']=bw
        add('bradley61_t10',lambda:g<base.box(g,61)*.9)
        add('phansalkar61',lambda:g<phansalkar_threshold(g,61))
        add('phansalkar31',lambda:g<phansalkar_threshold(g,31))
        continuous={}
        for size in [31,61,121]:
            B,n=background_normalize(g,size)
            t=float(threshold_otsu(n))
            add('bg'+str(size)+'_otsu',lambda n=n,t=t:n<t)
            details[cid]['bg'+str(size)+'_otsu']['otsu_threshold']=t
            if size==61:
                continuous=dict(background=B,normalized=n)
                for key,a in continuous.items():
                    dest=HERE/'intermediates'/cid/(key+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray((a*255).astype('uint8')).save(dest)
                add('bg61_wolf',lambda n=n:base.wolf(n))
                add('bg61_sauvola',lambda n=n:base.sauvola(n))
                add('bg61_nick',lambda n=n:base.nick(n))
        started=time.monotonic();terms,tm=howe_terms(g)
        details[cid]['howe_terms']=tm
        details[cid]['howe_terms']['seconds']=time.monotonic()-started
        for weight in [80,160,320]:add('howe'+str(weight),lambda weight=weight:howe(terms,weight))
        a=annotation[cid];ox,oy,_,_=oldcases[cid]['roi_xywh']
        for key,b in masks.items():
            points=[a['ink_points_xy'][i] for i in audit['valid_original_point_indices'][cid]]
            hits=[bool(b[oy+py,ox+px]) for px,py in points]
            negative=[]
            for px,py,pw,ph in a['nonink_regions_xywh']:
                region=b[oy+py:oy+py+ph,ox+px:ox+px+pw]
                negative.append(dict(old_roi_relative_xywh=[px,py,pw,ph],foreground_pixels=int(region.sum()),pixels=region.size))
            records.append(dict(id=cid,method=key,ink_center_hits=sum(hits),ink_centers=len(hits),point_hits=hits,
                nonink_patches=negative,selected_nonink_foreground_pixels=sum(v['foreground_pixels'] for v in negative),
                selected_nonink_pixels=sum(v['pixels'] for v in negative),roi_foreground_pixels=int(b[c['roi_xywh'][1]:c['roi_xywh'][1]+c['roi_xywh'][3],c['roi_xywh'][0]:c['roi_xywh'][0]+c['roi_xywh'][2]].sum())))
        groups=[
            ('局所閾値の追加比較',['original','wolf61','bradley61','phansalkar61']),
            ('背景補正の比較',['original','wolf61','bg61_otsu','bg61_wolf']),
            ('筆画の連続性を狙う比較',['original','wolf61','nick61','howe160']),
            ('Bradleyの窓幅と閾値',['original','bradley61','bradley_paper','bradley61_t10']),
            ('背景推定サイズの感度',['original','bg31_otsu','bg61_otsu','bg121_otsu']),
            ('Howeの平滑化重みの感度',['original','howe80','howe160','howe320']),
            ('背景推定と補正入力',['original','background','normalized','bg61_otsu']),
            ('背景補正と筆画保持',['original','wolf61','bg61_sauvola','bg61_nick']),
            ('前回の再実装を広い範囲で確認',['original','wolf61','gatos_native','su2013_local']),
        ]
        for name,keys in groups:figures.append(sheet(c,raw,masks,keys,name,continuous))
        dump(HERE/'details.json',details)
        dump(HERE/'measurements.json',dict(scope=protocol['annotation_scope'],records=records))
    for p,h in sources.items():assert sha(ROOT/p)==h
    dump(HERE/'rendering.json',figures)
    dump(HERE/'complete.json',dict(cases=3,methods_per_case=len(masks),figures=len(figures),runtimes=runtimes,
        verification=verify,source_hashes_unchanged=True,source_sha256=sources,
        source_script_sha256=sha(__file__),graphcut_adapter_sha256=sha(HERE/'graphcut_cli.cpp'),
        graphcut_executable_sha256=sha(HERE/'graphcut_cli'),protocol_sha256=sha(HERE/'protocol.json'),
        annotation_sha256=sha(OLD/'annotations_frozen.json'),annotation_audit_sha256=sha(OLD/'annotation_audit.json')))
    print('complete',len(figures),flush=True)

if __name__=='__main__':main()
