"""Saved v1/v2 audit and experimental removal of the global a gate.

Exploratory diagnostics only: no frozen masks or inference outputs are changed.
"""
from pathlib import Path
import importlib.util
import sys
import json
import hashlib
import cv2
import numpy as np
from PIL import Image
import validate_red_retention as review

ROOT=review.ROOT
OUT=Path(__file__).resolve().parent/'red_detection_followup'
V1=ROOT/'modules/ddrm_npj/scripts/csc.py'
V2=ROOT/'tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2/scripts/csc.py'
CASES=[('F04','200021063_00002_2',[1650,2500,400,620]),
       *[(c[0],c[1],c[2]) for c in review.SELECTION]]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def run_traced(mod,path,ace):
    trace={};candidate=np.zeros(ace.shape[:2],bool)
    def hook(frame,event,arg):
        if frame.f_code.co_filename!=str(path):return None
        if event=='return' and frame.f_code.co_filename==str(path) and frame.f_code.co_name=='color_space_clustering':
            d=frame.f_locals
            for k in ['cluster_mean_a','cluster_mean_a_s2','cluster_mean_L','cluster_mean_L_s2','red_cluster','black_cluster','black_text_cluster']:
                if k in d:trace[k]=list(map(float,d[k])) if isinstance(d[k],list) else int(d[k])
            idx=d.get('nonbg_idx',d.get('deg_idx'))
            if idx is not None and 'stage2_labels' in d:
                candidate.reshape(-1)[idx[d['stage2_labels']==d['red_cluster']]]=True
        return hook
    sys.settrace(hook)
    try:result=mod.color_space_clustering(ace)
    finally:sys.settrace(None)
    return result,trace,candidate

def component_filter_fast(image_bgr,candidate_mask,support_min):
    """Same component support rule, accumulated once instead of K full-page scans."""
    hsv=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
    seeds=((hsv[:,:,0]<=12)|(hsv[:,:,0]>=168))&(hsv[:,:,1]>180)
    n,labels=cv2.connectedComponents(candidate_mask.astype(np.uint8),connectivity=8)
    counts=np.bincount(labels.ravel(),minlength=n)
    support=np.bincount(labels[seeds],minlength=n)
    selected=(support/np.maximum(counts,1))>=support_min
    selected[0]=False
    return selected[labels]

def save(p,m):
    p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.where(m,0,255).astype(np.uint8)).save(p)

def main():
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=True)
    protocol=dict(cases=CASES,v1=str(V1.relative_to(ROOT)),v1_sha256=digest(V1),
        v2=str(V2.relative_to(ROOT)),v2_sha256=digest(V2),
        candidate='Remove only global red mean-a >=133 gate; retain stage2 red candidate, HSV component support >=.05 and morphology close3/open1',
        final_test=False,ddrm_changed=False,scope='Source/mask exploratory audit; ROI counts are not annotated ink recall')
    proto=OUT/'protocol.json'
    if proto.exists():assert json.loads(proto.read_text())==json.loads(json.dumps(protocol))
    else:proto.write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    v1=module(V1,'audit_csc_v1');v2=module(V2,'audit_csc_v2');records=[]
    # Preserve decisions exactly; verify against the original implementation on
    # deterministic samples and against complete saved red masks on every page.
    rng=np.random.default_rng(42)
    for _ in range(3):
        test_image=rng.integers(0,256,(40,50,3),dtype=np.uint8)
        test_candidate=rng.random((40,50))>.7
        assert np.array_equal(component_filter_fast(test_image,test_candidate,.05),
                              v2._filter_red_candidate_by_component(test_image,test_candidate,.05))
    v2._filter_red_candidate_by_component=component_filter_fast
    for cid,page,roi in CASES:
        dest=OUT/page;dest.mkdir(exist_ok=True)
        done=dest/'diagnostic.json'
        if done.exists():
            records.append(json.loads(done.read_text()));continue
        dirs=[ROOT/f'data/{name}/test'/page.split('_')[0]/page for name in ['split_dataset_csc_mask','split_dataset_csc_mask_v2']]
        ace=cv2.imread(str(dirs[1]/'00_input.png'));raw=cv2.imread(str(ROOT/'data/test'/page.split('_')[0]/(page+'.jpg')))[:,:,::-1]
        x,y,w,h=roi;sl=np.s_[y:y+h,x:x+w]
        rec=dict(id=cid,page=page,roi_xywh=roi,dependencies={},saved={},v1_recompute=None)
        colors=np.array([[245,242,226],[35,35,35],[240,30,40],[50,140,245]],np.uint8)
        arrays=[raw];titles=['原画像']
        for label,d in zip(['v1','v2'],dirs):
            masks=[cv2.imread(str(d/f),0) for f in ['05_background.png','02_text_black.png','03_text_red.png','04_degradation.png']]
            if any(m is None for m in masks):raise FileNotFoundError(d)
            masks=[m>127 if i in (0,3) else m<127 for i,m in enumerate(masks)]
            counts={k:int(m[sl].sum()) for k,m in zip(['background','black','red','degradation'],masks)}
            rec['saved'][label]=dict(roi_pixels=counts,red_page_pixels=int(masks[2].sum()),ace_matches_v2=bool(np.array_equal(cv2.imread(str(d/'00_input.png')),ace)))
            # Later classes take visual priority where morphology overlaps occur.
            classified=np.full(raw.shape,255,np.uint8)
            for c,m in zip(colors,masks):classified[m]=c
            arrays.append(classified);titles.append('保存CSC '+label+' 分類')
            for q in d.glob('*.png'):rec['dependencies'][str(q.relative_to(ROOT))]=digest(q)
        if cid=='F04':
            ace1=cv2.imread(str(dirs[0]/'00_input.png'))
            result,t,c=run_traced(v1,V1,ace1)
            rec['v1_recompute']=dict(trace=t,red_page_pixels=int((result.red_text_mask>0).sum()),
                red_saved_disagreement=int(((result.red_text_mask>0)!=(cv2.imread(str(dirs[0]/'03_text_red.png'),0)<127)).sum()),
                black_saved_disagreement=int(((result.black_text_mask>0)!=(cv2.imread(str(dirs[0]/'02_text_black.png'),0)<127)).sum()))
        result,t,candidate=run_traced(v2,V2,ace)
        kept=v2._filter_red_candidate_by_component(ace,candidate,.05)
        alternative=v2._apply_morphology(kept.astype(np.uint8)*255,3,1)>0
        saved_red=cv2.imread(str(dirs[1]/'03_text_red.png'),0)<127
        rec['v2_recompute']=dict(trace=t,saved_red_disagreement=int(((result.red_text_mask>0)!=saved_red).sum()))
        assert rec['v2_recompute']['saved_red_disagreement']==0
        rec['component_filter_implementation']='Vectorized identical HSV component support; deterministic sample equivalence and whole-page saved-red equivalence checked'
        rec['candidate']=dict(roi_pixels=int(candidate[sl].sum()),kept_roi_pixels=int(alternative[sl].sum()),
            kept_page_pixels=int(alternative.sum()),added_page_pixels=int((alternative&~saved_red).sum()),
            removed_page_pixels=int((saved_red&~alternative).sum()))
        if cid=='R02':
            rec['candidate']['brown_binding_pixels']=int(alternative[150:2950,70:270].sum())
        save(dest/'red_candidate_before_gate.png',candidate);save(dest/'red_without_global_gate.png',alternative)
        overlay=raw.copy();overlay[alternative]=[20,235,20]
        arrays.extend([np.repeat(np.where(saved_red,0,255).astype(np.uint8)[:,:,None],3,2),overlay])
        titles.extend(['現行赤マスク','平均aゲート除外候補（緑）'])
        review.figure(dest/(cid+'_CSC分類と赤候補.png'),arrays,titles,roi,cid)
        if cid=='F04':
            # Source-defined color proxy within the seal box; descriptive, not GT.
            r,g,b=raw.astype(np.int16).transpose(2,0,1)
            proxy=(r-g>15)&(r-b>15);box=np.zeros(raw.shape[:2],bool);box[2600:3000,1780:2020]=True;proxy &= box
            rec['source_red_proxy_definition']='Seal box [1780,2600,240,400], raw R-G>15 and R-B>15; not annotated GT'
            rec['source_red_proxy_pixels']=int(proxy.sum());rec['source_red_proxy_classes']={}
            for label,d in zip(['v1','v2'],dirs):
                rec['source_red_proxy_classes'][label]={k:int((proxy&(cv2.imread(str(d/f),0)>127 if k in ('background','degradation') else cv2.imread(str(d/f),0)<127)).sum()) for k,f in [('background','05_background.png'),('black','02_text_black.png'),('red','03_text_red.png'),('degradation','04_degradation.png')]}
        done.write_text(json.dumps(rec,ensure_ascii=False,indent=2));records.append(rec)
        print(cid,json.dumps(dict(saved=rec['saved'],v1=rec['v1_recompute'],v2=rec['v2_recompute'],candidate=rec['candidate']),ensure_ascii=False),flush=True)
    (OUT/'diagnostics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
