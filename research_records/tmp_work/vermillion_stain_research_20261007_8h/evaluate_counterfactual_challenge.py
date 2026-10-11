"""Native development perturbation diagnosis; no real clean/readability GT."""
import argparse,json,time,pickle
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
from prepare_fullpage_baselines import module,heuristic_masks
from fit_sparse_color_context import feature_maps
from evaluate_synthetic_selectors import load_selector
from predict_native_selectors import predict
from render_selector_diagnostics import mask_path

def diagnose(r,mask,original_mask):
    pts=[p for p in json.loads((OUT/'sparse_original_review/points_frozen.json').read_text())['points'] if p['case']==r['case']]
    groups={}
    for p in pts:
        x,y=p['xy_original'];g=groups.setdefault(p['label'],dict(points=0,before=0,after=0,newly_selected=0,newly_lost=0))
        b=bool(original_mask[y,x]);a=bool(mask[y,x]);g['points']+=1;g['before']+=b;g['after']+=a;g['newly_selected']+=a and not b;g['newly_lost']+=b and not a
    regions=json.loads((OUT/'original_paper_negative_regions/regions_frozen.json').read_text())['regions'];alpha=np.load(OUT/r['alpha']);xx,yy,ww,hh=r['ROI'];full=np.zeros(mask.shape,bool);full[yy:yy+hh,xx:xx+ww]=alpha>=.2
    neg=[]
    for idx,p in enumerate(regions):
        if p['case']!=r['case']:continue
        x,y,w,h=p['xywh_original'];eligible=full[y:y+h,x:x+w];n=int(eligible.sum());a=int((mask[y:y+h,x:x+w]&eligible).sum());b=int((original_mask[y:y+h,x:x+w]&eligible).sum())
        neg.append(dict(region=idx,added_support_pixels=n,before_selected=b,after_selected=a,
            interpretation='Known added field inside tentative no-visible-red rectangle, not authentic stain-pixel GT'))
    return dict(points=groups,selected_paper_region_added_support=neg)

def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['learned','forest','csc_v2','docres_chroma'],required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    cv2.setNumThreads(2);torch.set_num_threads(4);root=OUT/'counterfactual_stain_challenge';rows=json.loads((root/'inputs_frozen.json').read_text())['records']
    names=[];models={};resources={}
    if a.family=='learned':
        for variant in ['paper_texture','paper_texture_sparse']:
            name='texture_'+variant;path=OUT/'texture_transfer_trials'/f'{variant}_seed614/last.pth';model,cfg,ck=load_selector(path,a.device);assert ck['step']==10000;models[name]=model;resources[name]={str(path):file_hash(path)};names.append(name)
    elif a.family=='forest':
        info=json.loads((OUT/'region_supported_trials/models.json').read_text());r=next(r for r in info if r['name']=='region_spatial_forest');name=r['name'];models[name]=pickle.loads(Path(r['model_path']).read_bytes());resources[name]={r['model_path']:file_hash(r['model_path'])};names=[name]
    elif a.family=='csc_v2':
        cp=ROOT/'tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2/scripts/csc.py';ap=ROOT/'modules/ddrm_npj/scripts/ace.py';csc=module('counterfactual_frozen_csc',cp);ace=module('counterfactual_frozen_ace',ap);names=['csc_v2_red'];resources[names[0]]={str(f):file_hash(f) for f in [cp,ap]}
    else:
        import sys
        sys.path.insert(0,str(ROOT/'scripts/benchmark'))
        from run_npj_literature_models import load,prompts
        from run_direct_literature_baselines import tiles
        dm,res=load('docres',a.device);pn=prompts();names=['docres_foreground_chroma'];resources[names[0]]={str(f):file_hash(f) for f in res}
    original_cache={}
    for r in rows:
        assert file_hash(OUT/r['input'])==r['input_sha256'];rgb=None
        for name in names:
            dest=root/'predictions'/name/r['id'];dest.mkdir(parents=True,exist_ok=True);done=dest/'complete.json'
            if done.exists():
                old=json.loads(done.read_text());assert old['input_sha256']==r['input_sha256']
                assert file_hash(dest/'mark_mask.png')==old['mask_sha256'];continue
            start=time.monotonic()
            if rgb is None:rgb=np.asarray(Image.open(OUT/r['input']).convert('RGB'))
            if a.family=='learned':mask=predict(models[name],rgb,'relative',a.device)[0]>=.5
            elif a.family=='forest':
                maps,features=feature_maps(rgb,True);h,w=rgb.shape[:2];mask=np.zeros((h,w),bool)
                for y in range(0,h,96):
                    x=np.stack([m[y:y+96].reshape(-1) for m in maps],1);mask[y:y+96]=models[name].predict_proba(x)[:,1].reshape(min(96,h-y),w)>=.5
                del maps
            elif a.family=='csc_v2':
                bgr=rgb[:,:,::-1].copy();equalized=ace.automatic_color_equalization(bgr,alpha=5.,n_samples=200,seed=0)
                mask=csc.color_space_clustering(equalized,seed=42,max_samples=50000,border_margin_ratio=.04).red_text_mask>127
            else:
                bgr=rgb[:,:,::-1].copy();prompt=pn['binarization_promptv2'](bgr);prob=tiles(dm,np.concatenate([bgr,prompt],2),'docres_binarization',a.device)
                mask=(prob[:,:,0]<.5)&heuristic_masks(rgb)['relative_chroma']
            key=(name,r['page'])
            if key not in original_cache:
                if a.family=='docres_chroma':
                    original=np.asarray(Image.open(ROOT/r['original']).convert('RGB'));ink=np.asarray(Image.open(OUT/'direct_literature_baselines/docres_binarization'/r['page']/'ink_mask.png'))>127
                    original_cache[key]=ink&heuristic_masks(original)['relative_chroma']
                else:original_cache[key]=np.asarray(Image.open(mask_path(name,r['page'])).convert('L'))>127
            Image.fromarray(mask.astype(np.uint8)*255).save(dest/'mark_mask.png')
            rec=dict(id=r['id'],case=r['case'],name=name,family=a.family,input_sha256=r['input_sha256'],resources=resources[name],
                script_sha256=file_hash(__file__),mask_sha256=file_hash(dest/'mark_mask.png'),seconds=time.monotonic()-start,threshold=.5,
                diagnostic=diagnose(r,mask,original_cache[key]),no_real_pixel_or_reading_GT=True,
                original_training_points_may_have_been_seen=True,perturbation='Frozen development diagnostic, not final test')
            done.write_text(json.dumps(rec,ensure_ascii=False,indent=2));print('Counterfactual',name,r['id'],round(rec['seconds'],1),'sec',flush=True)
    summary=[json.loads(f.read_text()) for name in names for f in (root/'predictions'/name).glob('*/complete.json')]
    (root/(a.family+'_results.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
