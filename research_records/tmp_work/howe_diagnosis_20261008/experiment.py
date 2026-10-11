"""Howe mechanism diagnosis and Alg3 parameter-selection port.

The author selection logic is retained (33 c values, 3 high thresholds).
Canny remains the disclosed skimage approximation, NOT MATLAB canny_old.
Author copyright/license retained in Howe_LICENSE.txt.
"""
from pathlib import Path
import json, struct, subprocess, importlib.util, itertools, hashlib, time
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import canny

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PRIOR=ROOT/'tmp_work/classical_limits_by_category_20261008'
DEEP=ROOT/'tmp_work/classical_deep_review_20261008'
spec=importlib.util.spec_from_file_location('deep',DEEP/'run.py');deep=importlib.util.module_from_spec(spec);spec.loader.exec_module(deep)
deep.SCRATCH=HERE/'scratch';deep.SCRATCH.mkdir(exist_ok=True)
CS=np.exp(np.linspace(np.log(20),np.log(5120),33))
THS=[.25,.375,.5]
def dump(p,v):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return np.array(Image.open(p).convert('RGB'))
def mask(p):return np.array(Image.open(p).convert('L'))==0
def save(p,a):deep.save_mask(p,a)
def post(b):
    expanded=np.pad(b,((0,1),(0,1)),mode='edge')
    above=np.concatenate([expanded[:1],expanded[:-1]],0);left=np.concatenate([expanded[:,:1],expanded[:,:-1]],1)
    return expanded&~(expanded&~above&~left),expanded

def sweep(terms,cs,folder,tag):
    folder.mkdir(exist_ok=True,parents=True);src,snk,down,right=terms;h,w=src.shape
    inp=folder/(tag+'_input.bin');out=folder/(tag+'_output.bin')
    with inp.open('wb') as f:
        f.write(struct.pack('<III',h,w,len(cs)))
        for a in (cs,src,snk,down,right):np.asarray(a,dtype='<f8').tofile(f)
    with (folder/(tag+'_progress.log')).open('w') as log:
        subprocess.run([str(HERE/'sweep_cli'),str(inp),str(out)],check=True,stdout=log)
    n=h*w
    with out.open('rb') as f:
        for c in cs:
            flow=struct.unpack('<d',f.read(8))[0];b=np.frombuffer(f.read(n),dtype='uint8').reshape(h,w).astype(bool)
            e=deep.energy(b,src,snk,c*down,c*right)
            assert abs(e-flow)<max(1e-5,abs(flow)*1e-8),(c,e,flow)
            yield b,dict(c=float(c),flow=flow,actual_energy=e)
    inp.unlink();out.unlink()

def terms(g,thi=.4,tlo=.1,sigma=.6,bias=True,use_edges=True):
    I=g.astype('float64')*255
    dx=I[:-1,1:]-I[:-1,:-1];dy=I[1:,:-1]-I[:-1,:-1]
    lapraw=np.gradient(dx,axis=1)+np.gradient(dy,axis=0)
    residual=I-ndi.gaussian_filter(I,20,mode='reflect',radius=60)
    rms=np.sqrt(ndi.gaussian_filter(residual**2,20,mode='constant',cval=0,radius=50))
    confident=residual/(rms+np.finfo(float).eps)>2
    lap=lapraw.copy()
    if bias:lap[confident[:-1,:-1]]=-500
    sm=ndi.gaussian_filter(g,sigma,mode='reflect',truncate=4)
    maxmag=float(np.hypot(ndi.sobel(sm,axis=0,mode='reflect'),ndi.sobel(sm,axis=1,mode='reflect')).max())
    edges=canny(g,sigma=sigma,low_threshold=tlo*maxmag,high_threshold=thi*maxmag,mode='reflect') if use_edges else np.zeros_like(g,bool)
    down=~((edges[:-1,:-1]&(dy>0))|(edges[1:,:-1]&(dy<=0)))
    right=~((edges[:-1,:-1]&(dx>0))|(edges[:-1,1:]&(dx<=0)))
    return (1500-lap,1500+lap,down[:-1].astype('float64'),right[:,:-1].astype('float64')),dict(lapraw=lapraw,lap=lap,confident=confident,edges=edges,maxmag=maxmag)

def stability_select(bsd):
    kernel=np.exp(-.5*np.arange(-2.5,3,1)**2);kernel/=kernel.sum()
    d=np.convolve(bsd,kernel,mode='full')[2:-2]
    best=-np.inf;selected=None
    for i in range(len(d)-2):
        for j in range(i+2,len(d)):
            for k in range(i+1,j):
                score=float(d[i]+d[j]-2*d[k])
                if score>best:best=score;selected=(i,k,j)
    return selected[1],dict(bsd=[float(v) for v in bsd],smoothed=d.tolist(),qrs_zero_based=list(selected),score=best)

def checks():
    rng=np.random.default_rng(987);results=[]
    for shape in [(2,3),(3,3)]:
        for mode in ['dense','sparse','no_edges']:
            src=rng.uniform(-2,10,shape);snk=rng.uniform(-2,10,shape)
            down=rng.uniform(0,2,(shape[0]-1,shape[1]));right=rng.uniform(0,2,(shape[0],shape[1]-1))
            if mode=='sparse':down[down<1]=0;right[right<1]=0
            if mode=='no_edges':down[:]=0;right[:]=0
            t=(src,snk,down,right);cs=[0,.5,2,10]
            for (got,meta),c in zip(sweep(t,cs,HERE/'scratch','check'),cs):
                best=min(deep.energy(np.array(bits,bool).reshape(shape),src,snk,c*down,c*right) for bits in itertools.product([False,True],repeat=src.size))
                independent,flow=deep.cut(src,snk,c*down,c*right)
                assert abs(meta['actual_energy']-best)<1e-8
                assert abs(flow-best)<1e-8
                assert np.array_equal(got,independent)
                results.append(dict(shape=shape,mode=mode,c=c,enumerated_energy=best,labels_match_independent=True))
    # Independent scalar convolution and MATLAB-index-shaped brute-force rule.
    bsd=rng.uniform(0,.01,32);idx,m=stability_select(bsd)
    kernel=np.exp(-.5*np.arange(-2.5,3,1)**2);kernel/=kernel.sum()
    conv=np.array([sum(bsd[j]*kernel[i-j] for j in range(32) if 0<=i-j<6) for i in range(37)])[2:-2]
    assert np.allclose(conv,m['smoothed'],rtol=0,atol=1e-16) and len(conv)==33
    triples=[(float(conv[q-1]+conv[s-1]-2*conv[r-1]),q,r,s) for q in range(1,32) for s in range(q+2,34) for r in range(q+1,s)]
    chosen=max(triples,key=lambda v:v[0]);assert chosen[2]-1==idx
    uniform_idx,_=stability_select(np.zeros(32));assert uniform_idx==1
    g=rng.uniform(0,1,(50,60)).astype('float32');ours,aux=terms(g);old,_=deep.howe_terms(g)
    for a,b in zip(ours,old):assert np.array_equal(a,b)
    # Exact mathematical equivalence of separable Gaussian and explicit kernel.
    a=rng.uniform(0,255,(41,43));sigma=2;rad=5;k=np.exp(-np.arange(-rad,rad+1)**2/(2*sigma**2));k/=k.sum()
    ref=ndi.convolve(ndi.convolve(a,k[None,:],mode='constant'),k[:,None],mode='constant')
    assert np.allclose(ref,ndi.gaussian_filter(a,sigma,radius=rad,mode='constant'),atol=1e-12)
    result=dict(dynamic_graph_cut=results,author_stability_scalar_convolution_checked=True,author_qrs_one_based_indices_checked=True,tie_behavior_checked=True,base_terms_match_existing_port=True,separable_gaussian_checked=True,matlab_runtime_validation=False)
    dump(HERE/'implementation_checks.json',result);return result

def freeze():
    allcases=json.loads((PRIOR/'selection.json').read_text())['cases']
    cases=[c for c in allcases if c['id'][:2] in ['01','02','03','04']]
    protocol=dict(cases=cases,pages=sorted({c['page'] for c in cases}),input='Full native raw page; OpenCV RGB2GRAY /255. Crop after global processing.',
        auto=dict(author_algorithm=3,c_values=CS.tolist(),high_thresholds=THS,tlo=.1,sigma=.6,rule='Alg1 smoothed instability qrs criterion, Alg3 chooses low vs high output by lower mismatch to middle; strict < else high',reference='https://www.science.smith.edu/~nhowe/research/code/'),
        variants=['fixed c160 thi.4','pre_postprocessing','no_background_bias','no_edge_gating','data_only_c0','c40_thi.4','c80_thi.4','c320_thi.4','thi.25_c160','auto_Alg3_local_Canny'],
        interpretation='Mechanism ablations are diagnostics, not proposed methods. No pixel GT or test accuracy. Selected known development pages.',
        unresolved=['No MATLAB/Octave runtime; canny_old numerical parity not established','OpenCV uint8 gray vs MATLAB rgb2gray not cross-runtime verified','Dynamic solver tie choices checked on tiny random grids, not original MATLAB full pages'],
        frozen_before_new_outputs=True,spot_probe=dict(page='hnsd001_011',xywh=[460,590,90,60],role='Previously inspected isolated nontext spot; not dense GT'))
    dump(HERE/'protocol.json',protocol)
    sourcefiles=[DEEP/'run.py',DEEP/'graphcut_cli',DEEP/'sources/howe_ijdar/binarizeImage.m',DEEP/'sources/howe_ijdar/binarizeImageAlg1.m',DEEP/'sources/howe_ijdar/binarizeImageAlg3.m',DEEP/'sources/howe_ijdar/gsmooth.m',HERE/'sweep_cli.cpp',HERE/'sweep_cli',HERE/'experiment.py']
    dump(HERE/'sources.json',{str(p.relative_to(ROOT)):sha(p) for p in sourcefiles})

def worker(page):
    cv2.setNumThreads(1);start=time.monotonic()
    protocol=json.loads((HERE/'protocol.json').read_text());cases=[c for c in protocol['cases'] if c['page']==page];c=cases[0]
    assert sha(ROOT/c['original'])==c['source_sha256'];raw=read(ROOT/c['original']);g=deep.base.gray(raw)
    folder=HERE/'masks'/page;folder.mkdir(exist_ok=True,parents=True);scratch=HERE/'scratch'/page
    status={'page':page,'stage':'fixed','completed_sweeps':0};dump(HERE/'status'/f'{page}.json',status)
    t,aux=terms(g);oldt,_=deep.howe_terms(g)
    assert all(np.array_equal(a,b) for a,b in zip(t,oldt));del oldt
    fixed_results={};basepre=None
    for (pre,meta),weight in zip(sweep(t,[0,40,80,160,320],scratch,'fixed'),[0,40,80,160,320]):
        final,expanded=post(pre);name='fixed' if weight==160 else f'c{weight}'
        save(folder/(name+'.png'),final);fixed_results[name]=meta
        if weight==160:
            assert np.array_equal(final,mask(PRIOR/'masks'/page/'howe.png'))
            save(folder/'no_post.png',expanded);basepre=expanded
    save(folder/'sauvola.png',mask(PRIOR/'masks'/page/'sauvola.png'))
    for key,opts in [('no_bias',dict(bias=False)),('no_edges',dict(use_edges=False)),('thi025_c160',dict(thi=.25))]:
        tt,_=terms(g,**opts)
        pre,meta=next(sweep(tt,[160],scratch,key));final,expanded=post(pre);save(folder/(key+'.png'),final);fixed_results[key]=meta
    status['stage']='auto';dump(HERE/'status'/f'{page}.json',status)
    branches=[]
    for thi in THS:
        tt,_=terms(g,thi=thi);bs=[];bsd=[];energies=[]
        for idx,(pre,meta) in enumerate(sweep(tt,CS,scratch,f'auto_{thi}')):
            final,expanded=post(pre)
            if bs:bsd.append(float(np.mean(bs[-1]!=final)))
            bs.append(final);energies.append(meta)
        chosen,stability=stability_select(bsd);best=bs[chosen]
        key=f'auto_thi{thi}';save(folder/(key+'.png'),best)
        # Preserve all diagnostic ROI candidates, not hundreds of full-page files.
        for cc in cases:
            x,y,w,h=cc['roi_xywh']
            for idx,b in enumerate(bs):save(HERE/'sweep_crops'/cc['id']/f'thi{thi}_c{idx:02d}.png',b[y:y+h,x:x+w])
        branches.append(dict(thi=thi,c=float(CS[chosen]),index=chosen,key=key,stability=stability,energies=energies))
        status['completed_sweeps']+=1;status['last_thi']=thi;dump(HERE/'status'/f'{page}.json',status)
        print(page,'auto branch',thi,'c',float(CS[chosen]),flush=True);del bs
    blo,bmid,bhi=[mask(folder/(b['key']+'.png')) for b in branches]
    dlo=float(np.mean(blo!=bmid));dhi=float(np.mean(bhi!=bmid));chosen=0 if dlo<dhi else 2
    auto=branches[chosen];save(folder/'auto.png',blo if chosen==0 else bhi)
    # Save interpretable local arrays from full-page computation, not ROI reruns.
    for cc in cases:
        x,y,w,h=cc['roi_xywh'];sl=np.s_[y:y+h,x:x+w]
        dest=HERE/'intermediates'/cc['id'];dest.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(dest/'terms.npz',lapraw=aux['lapraw'][sl],lap=aux['lap'][sl],edges=aux['edges'][sl],confident=aux['confident'][sl],raw=raw[sl])
    audit=dict(page=page,baseline_exact=True,terms_exact=True,branches=branches,selected=auto['key'],c=auto['c'],thi=auto['thi'],dlo=dlo,dhi=dhi,fixed_energies=fixed_results,seconds=time.monotonic()-start,
       full_page_post_removed_pixels=int((basepre&~mask(folder/'fixed.png')).sum()),source_sha256=sha(ROOT/c['original']))
    dump(HERE/'results'/f'{page}.json',audit);status['stage']='complete';status['seconds']=audit['seconds'];dump(HERE/'status'/f'{page}.json',status)
    print(page,'DONE',auto['c'],auto['thi'],round(audit['seconds'],1),flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':checks();freeze();print('checks passed; protocol frozen')
    elif sys.argv[1]=='worker':worker(sys.argv[2])
