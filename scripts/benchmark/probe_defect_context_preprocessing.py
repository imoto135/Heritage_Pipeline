"""Development prototype: paper-relative brown components and neutral ridges.

Infer on full pages, without ROI gating or restoration. Compare the same color
candidates with/without spatial support and neutral dark-stroke guards. Frozen
v6 remains read-only. This is an unvalidated heuristic, not a physical detector.
"""
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/defect_context_preprocessing_pilot_v1_20261001'
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import dump, sha, read, write, overlay, FONT_PATH
from collect_saved_npj_failures import saved_masks

cv2.setNumThreads(1)
SOURCES = [ROOT/'tmp_work/npj_failure_collection_20261001',
           ROOT/'tmp_work/npj_failure_followup_20261001']
SETTINGS = dict(paper=dict(scale=4,bright_window_small=33,bright_tolerance_L=8.,sigma_small=16.),
    color=dict(soft_delta_a=2.5,soft_L_drop=4.,a_min=2.,b_min=5.,b_over_a=.9,
               seed_delta_a=5.,seed_L_drop=8.,seed_b_min=12.),
    component=dict(connectivity=8,min_area=80,min_seed_pixels=12,min_seed_fraction=.03,
                   min_short_side=8,min_fill=.12,min_peak_radius=3.),
    ridge=dict(sigmas=[1.5,3.,6.],response_min=1.5,neutral_delta_a_max=4.,
               neutral_a_max=6.,local_L_drop_min=6.,guard_radius=2.),
    always_preserve='frozen v6 confirmed red and v6-specific extra warm hold',
    modes=dict(color_only='seeded paper-relative brown components, no shape/ridge gate',
               context_stroke='same color candidates + component support + neutral dark-ridge guard'),
    edit='frozen v6 edit OR additional candidate; clear protect/hold only in added edit',
    processing='full original pages, no ROI coordinates or labels used in inference',
    mask_dilation=False,training=False,pixel_GT=False,restoration=False,
    purpose='development diagnostic, not a frozen new method or independent validation')


def paper_estimate(lab):
    h,w=lab.shape[:2]
    small=cv2.resize(lab,((w+3)//4,(h+3)//4),interpolation=cv2.INTER_AREA)
    max_l=cv2.dilate(small[:,:,0],np.ones((33,33),np.uint8))
    bright=(small[:,:,0]>=max_l-8.).astype(np.float32)
    weight=cv2.GaussianBlur(bright,(0,0),16.,borderType=cv2.BORDER_REFLECT_101)
    weighted=cv2.GaussianBlur(small*bright[:,:,None],(0,0),16.,borderType=cv2.BORDER_REFLECT_101)
    valid=weight>1e-5
    fallback=cv2.GaussianBlur(small,(0,0),16.,borderType=cv2.BORDER_REFLECT_101)
    estimate=np.where(valid[:,:,None],weighted/np.maximum(weight[:,:,None],1e-5),fallback)
    return cv2.resize(estimate,(w,h),interpolation=cv2.INTER_LINEAR)


def ridge_response(light):
    best=np.zeros(light.shape,np.float32)
    for sigma in SETTINGS['ridge']['sigmas']:
        smoothed=cv2.GaussianBlur(light,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
        xx=cv2.Sobel(smoothed,cv2.CV_32F,2,0,ksize=3,scale=.25)*sigma**2
        yy=cv2.Sobel(smoothed,cv2.CV_32F,0,2,ksize=3,scale=.25)*sigma**2
        xy=cv2.Sobel(smoothed,cv2.CV_32F,1,1,ksize=3,scale=.25)*sigma**2
        root=np.sqrt((xx-yy)**2+4*xy**2)
        lo=(xx+yy-root)/2;hi=(xx+yy+root)/2
        response=np.maximum(hi,0)*(1-np.minimum(np.abs(lo)/np.maximum(np.abs(hi),1e-6),1))
        best=np.maximum(best,response)
    return best


def infer(original,v6):
    # No ROI or case labels are accepted by this function.
    median=cv2.medianBlur(original,3)
    lab=cv2.cvtColor(median.astype(np.float32)/255.,cv2.COLOR_BGR2LAB)
    paper=paper_estimate(lab)
    light,a,b=lab[:,:,0],lab[:,:,1],lab[:,:,2]
    da=a-paper[:,:,1];drop=paper[:,:,0]-light
    soft=(da>=2.5)&(drop>=4.)&(a>=2.)&(b>=5.)&(b>=.9*np.maximum(a,0))
    seed=soft&(da>=5.)&(drop>=8.)&(b>=12.)
    n,labels,stats,_=cv2.connectedComponentsWithStats(soft.astype(np.uint8),8)
    counts=np.bincount(labels.ravel(),weights=seed.ravel(),minlength=n)
    areas=stats[:,cv2.CC_STAT_AREA]
    valid=(areas>=80)&(counts>=12)&(counts/np.maximum(areas,1)>=.03);valid[0]=False
    widths=stats[:,cv2.CC_STAT_WIDTH];heights=stats[:,cv2.CC_STAT_HEIGHT]
    fill=areas/np.maximum(widths*heights,1)
    distance=cv2.distanceTransform(soft.astype(np.uint8),cv2.DIST_L2,5)
    peak=np.zeros(n,np.float32);np.maximum.at(peak,labels.ravel(),distance.ravel())
    spatial=valid&(np.minimum(widths,heights)>=8)&(fill>=.12)&(peak>=3.)
    response=ridge_response(light)
    neutral=(da<=4.)&(a<=6.)&(drop>=6.)
    ridge=(response>=1.5)&neutral
    guarded=cv2.dilate(ridge.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    mandatory=v6['red']|v6['extra']
    color_only=valid[labels]&~mandatory
    context=spatial[labels]&~mandatory&~guarded
    modes={}
    for name,candidate in [('color_only',color_only),('context_stroke',context)]:
        add=candidate&~v6['edit']
        edit=v6['edit']|add
        protect=v6['protect']&~add;hold=v6['hold']&~add
        assert not np.any(edit&(protect|hold|v6['red']|v6['extra']))
        assert not np.any(add&mandatory)
        modes[name]=dict(edit=edit,added=add,protect=protect,hold=hold,red=v6['red'])
    components=[]
    for i in np.flatnonzero(valid):
        x,y,w,h,area=map(int,stats[i]);s=(slice(y,y+h),slice(x,x+w));inside=labels[s]==i
        components.append(dict(id=int(i),xywh=[x,y,w,h],area=area,seed_pixels=int(counts[i]),
            fill=float(fill[i]),peak_radius=float(peak[i]),spatial_accept=bool(spatial[i]),
            median_delta_a=float(np.median(da[s][inside])),
            median_L_drop=float(np.median(drop[s][inside])),
            neutral_ridge_fraction=float(ridge[s][inside].mean()),
            v6_extra_pixels=int((v6['extra'][s]&inside).sum()),
            v6_red_pixels=int((v6['red'][s]&inside).sum())))
    features=dict(color_seed=seed,color_soft=soft,neutral_ridge=ridge,ridge_guard=guarded,
                  paper_bgr=cv2.cvtColor(paper,cv2.COLOR_LAB2BGR),
                  delta_a=da,light_drop=drop,ridge_response=response)
    return modes,features,components


def paint(original,masks):
    return overlay(original,[('edit',masks['edit']),('hold',masks['hold']),
                             ('blue',masks['protect']),('red',masks['red'])])


def figure(images,labels,path,title,note):
    h,w=images[0].shape[:2];scale=min(480/w,1000/h);dw,dh=round(w*scale),round(h*scale)
    width=len(images)*dw+(len(images)-1)*20+48
    canvas=Image.new('RGB',(width,dh+242),(250,248,243));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT_PATH,25);small=ImageFont.truetype(FONT_PATH,21)
    draw.text((24,14),title,font=font,fill='black')
    draw.text((24,51),'分類マスクの試作。復元なし・全ページから自動推定・開発例。',font=small,fill='black')
    for i,(im,label) in enumerate(zip(images,labels)):
        x=24+i*(dw+20);draw.text((x,91),label,font=font,fill='black')
        resized=cv2.resize(im,(dw,dh),interpolation=cv2.INTER_NEAREST)
        canvas.paste(Image.fromarray(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)),(x,128))
    draw.text((24,dh+148),'青：保護　赤：朱　黄：編集候補　紫：保留',font=small,fill='black')
    draw.text((24,dh+182),note,font=small,fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True);cases=[];dependencies={}
    for source in SOURCES:
        for name in ['baseline_review.json','comparison_manifest.json','comparison_complete.json']:
            dependencies[str((source/name).relative_to(ROOT))]=sha(source/name)
        previous=json.loads((source/'comparison_manifest.json').read_text())
        for rel,digest in previous['dependencies'].items():assert sha(ROOT/rel)==digest,rel
        outputs=json.loads((source/'comparison_complete.json').read_text())['outputs']
        for row in json.loads((source/'baseline_review.json').read_text())['cases']:
            assert not row['contains_illustration']
            for key in ['original','npj_black','npj_red','npj_degradation','npj_background']:
                path=ROOT/row[key];dependencies[str(path.relative_to(ROOT))]=sha(path)
            for key in ['edit','protect','hold','red','extra']:
                path=source/'v6_masks'/row['page']/(key+'.png')
                assert sha(path)==outputs[str(path.relative_to(source))]
                dependencies[str(path.relative_to(ROOT))]=sha(path)
            cases.append(dict(row=row,source=source))
    manifest=dict(settings=SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)),
        pages=len(cases),regions=sum(len(c['row']['rois']) for c in cases),
        scope='all previously compared pages are development; no independent test claim')
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Use a new version for changed settings/code'
    else:dump(path,manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');diagnostics=[];started=time.monotonic()
    for item in cases:
        row,source=item['row'],item['source'];page=row['page'];dest=OUT/'pages'/page
        original=read(ROOT/row['original']);npj=saved_masks(row)
        v6={k:read(source/'v6_masks'/page/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        modes,features,components=infer(original,v6)
        for method,masks in modes.items():
            for key,value in masks.items():write(dest/method/(key+'.png'),value)
        for key,value in features.items():
            if key=='paper_bgr':write(dest/'features'/(key+'.png'),np.rint(np.clip(value,0,1)*255).astype(np.uint8))
            elif value.dtype==bool:write(dest/'features'/(key+'.png'),value)
            else:
                (dest/'features').mkdir(parents=True,exist_ok=True)
                np.save(dest/'features'/(key+'.npy'),value)
        dump(dest/'components.json',components)
        regions=[dict(id=page+'_whole',title='ページ全体',xywh=[0,0,original.shape[1],original.shape[0]],baseline_failure=False)]+row['rois']
        for roi in regions:
            x,y,w,h=roi['xywh'];s=(slice(y,y+h),slice(x,x+w));id=roi['id'];crop=original[s]
            crops=[crop,overlay(crop,[('edit',npj['edit'][s]),('blue',npj['black'][s]),('red',npj['red'][s])]),
                   paint(crop,{k:v[s] for k,v in v6.items()}),
                   paint(crop,{k:v[s] for k,v in modes['context_stroke'].items()})]
            count={name:int(m['added'][s].sum()) for name,m in modes.items()}
            note=f'追加編集候補：色のみ {count["color_only"]:,} / 文脈＋筆画保護 {count["context_stroke"]:,}画素（精度ではない）'
            figure(crops,['原画像','保存済みnpj','固定v6','試作：文脈＋筆画保護'],OUT/'comparisons'/(id+'.png'),roi['title']+' · '+page,note)
            figure([crop,paint(crop,{k:v[s] for k,v in v6.items()}),
                    paint(crop,{k:v[s] for k,v in modes['color_only'].items()}),crops[-1]],
                    ['原画像','固定v6','試作：色差のみ','試作：文脈＋筆画保護'],
                    OUT/'ablation'/(id+'.png'),roi['title']+' · '+page,note)
            if not id.endswith('_whole'):
                write(OUT/'native'/id/'original.png',crop)
                for name,m in modes.items():
                    for key in ['edit','added','protect','hold']:
                        write(OUT/'native'/id/(name+'_'+key+'.png'),m[key][s])
                    write(OUT/'native'/id/(name+'_overlay.png'),paint(crop,{k:v[s] for k,v in m.items()}))
            diagnostics.append(dict(id=id,page=page,xywh=roi['xywh'],baseline_failure=roi['baseline_failure'],
                **{name+'_added_pixels':value for name,value in count.items()},
                v6_edit_pixels=int(v6['edit'][s].sum()),
                candidate_v6_protect_overlap=int((modes['context_stroke']['added'][s]&v6['protect'][s]).sum()),
                candidate_v6_hold_overlap=int((modes['context_stroke']['added'][s]&v6['hold'][s]).sum()),
                red_or_extra_added_pixels=int((modes['context_stroke']['added'][s]&(v6['red'][s]|v6['extra'][s])).sum())))
        print('Context prototype:',page,'components',len(components),flush=True)
        del original,v6,modes,features,npj;gc.collect()
    dump(OUT/'diagnostics.json',diagnostics)
    with (OUT/'diagnostics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',pages=11,regions=15,
        full_page_inference=True,ROI_gating=False,source_and_code_hashes=True,
        red_and_v6_extra_preserved=True,edit_protect_hold_disjoint=True,
        frozen_v6_unchanged=True,primary_artifact='PNG',pixel_GT=False,restoration=False,
        elapsed_seconds=time.monotonic()-started))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.suffix in ['.png','.npy']}))
    print('Prototype completed: 11 pages / 15 frozen ROIs; not independently validated',flush=True)


if __name__=='__main__':main()
