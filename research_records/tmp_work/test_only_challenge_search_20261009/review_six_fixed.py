"""Audit original six challenges with frozen red, text and DDRM conditions."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import argparse,json,shutil
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import validate_local_red as red_method
import run_ddrm_pilot as driver
import validate_red_retention as review

ROOT=driver.ROOT
BASE=Path(__file__).resolve().parent
OUT=BASE/'six_challenge_fixed_review'
PRIOR=BASE/'ddrm_pilot'
METHODS=['csc','sauvola','howe']
CONTEXT={
    '01':[500,500,900,1300], '02':[550,800,1050,1500],
    '03':[450,700,1300,1300], '04':[550,550,800,430],
    '06':[700,900,800,1000], '07':[350,600,1200,1800],
}

def prepare():
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    prior=json.loads((PRIOR/'protocol.json').read_text())
    selection=json.loads((BASE/'red_local_color_validation/selected_candidate.json').read_text())
    redproto=json.loads((ROOT/selection['protocol']).read_text())
    assert driver.digest(ROOT/selection['protocol'])==selection['protocol_sha256']
    assert driver.digest(driver.PIPELINE)==prior['ddrm']['code_sha256']
    assert driver.digest(driver.CHECKPOINT)==prior['ddrm']['checkpoint_sha256']
    assert driver.digest(driver.BACKGROUND)==prior['ddrm']['background_sha256']
    params=selection['parameters']
    for k,value in red_method.VARIANTS['grow5'].items():assert params[k]==value
    proto=dict(created_utc=datetime.now(timezone.utc).isoformat(),cases=prior['cases'],rois_xywh=prior['rois_xywh'],
        context_rois_xywh=CONTEXT,ddrm=prior['ddrm'],red_selection=selection,
        red_shared=redproto['shared'],text=dict(csc='Saved ACE+CSC v2',sauvola='window61 k=.20 R=.5',howe='c160 Canny .6/.1/.4; same local base-energy port'),
        inference_uses=['black','red'],csc_background_degradation_role='Diagnostic only; not read by frozen DDRM',
        optional_isolated_cleanup='Appendix only; unchanged previous isolated_csc_v2 rule',
        final_test=False,scope='Original six observed and tuned development cases from test; no dense ground truth',
        script_sha256=driver.digest(Path(__file__)))
    driver.save_json(OUT/'protocol.json',proto)
    red_method.OUT=OUT
    v2=red_method.old.module(red_method.old.V2,'six_review_csc');v2._filter_red_candidate_by_component=red_method.old.component_filter_fast
    rows=[]
    for c in proto['cases']:
        cid=c['id'];p=c['page'];raw=np.asarray(Image.open(ROOT/c['source']).convert('RGB'))
        assert driver.digest(ROOT/c['source'])==c['source_sha256']
        assert list(Image.open(ROOT/c['source']).size)==c['input_size']
        anchor,meta=red_method.anchor_for(c,v2);delta,chroma=red_method.color_fields(raw)
        local,_=red_method.detect(anchor,delta,chroma,params)
        seed=anchor&chroma&(delta>=params['strong_a'])
        radius=params['region_seed_radius'];region=cv2.dilate(seed.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
        n,labels=cv2.connectedComponents(region,connectivity=8);counts=np.bincount(labels[seed],minlength=n)
        keep=counts>=params['region_min_seed_pixels'];keep[0]=False;zone=keep[labels]
        oldred=review.load(ROOT/c['red']);assert driver.digest(ROOT/c['red'])==c['red_sha256']
        red=(local&zone)|(oldred&zone)
        red_method.old.save(OUT/'fixed_red'/p/'03_text_red.png',red)
        d=ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p
        classes={
            'black':review.load(d/'02_text_black.png'),'red':oldred,
            'degradation':np.asarray(Image.open(d/'04_degradation.png').convert('L'))>127,
            'background':np.asarray(Image.open(d/'05_background.png').convert('L'))>127,
        }
        assigned=np.stack(list(classes.values())).sum(axis=0)
        howe=review.load(ROOT/c['masks']['howe']['black'])&~red
        csc=classes['black']&~red
        x,y,w,h=proto['rois_xywh'][cid];sl=np.s_[y:y+h,x:x+w]
        extra=howe&~csc
        row=dict(id=cid,page=p,red_page_pixels=int(red.sum()),red_roi_pixels=int(red[sl].sum()),red_matches_old=bool(np.array_equal(red,oldred)),
            anchor=meta,methods={},csc_extra_association_roi={k:int((extra&m)[sl].sum()) for k,m in classes.items()},
            howe_extra_roi_pixels=int(extra[sl].sum()),howe_extra_unassigned_roi_pixels=int((extra&(assigned==0))[sl].sum()),
            csc_unassigned_page_pixels=int((assigned==0).sum()),csc_overlap_page_pixels=int((assigned>1).sum()),
            csc_masks_sha256={driver.relative(d/f):driver.digest(d/f) for f in ['02_text_black.png','03_text_red.png','04_degradation.png','05_background.png']})
        for method in METHODS:
            black=review.load(ROOT/c['masks'][method]['black']);assert driver.digest(ROOT/c['masks'][method]['black'])==c['masks'][method]['black_sha256']
            dest=OUT/'masks'/method/p
            red_method.old.save(dest/'02_text_black.png',black&~red);red_method.old.save(dest/'03_text_red.png',red)
            previous=PRIOR/'masks'/method/p
            identical=bool(np.array_equal(review.load(previous/'02_text_black.png'),black&~red) and np.array_equal(review.load(previous/'03_text_red.png'),red))
            row['methods'][method]=dict(black_pixels=int((black&~red).sum()),same_as_previous_input=identical,
                mask_sha256={driver.relative(q):driver.digest(q) for q in dest.glob('*.png')})
        stage=OUT/'inputs'/p;stage.mkdir(exist_ok=True,parents=True);link=stage/(p+'.jpg')
        if not link.exists():link.symlink_to(ROOT/c['source'])
        rows.append(row);driver.save_json(OUT/'mask_audit.json',rows)
        print(cid,'red',row['red_page_pixels'],'same inputs',[r['same_as_previous_input'] for r in row['methods'].values()],flush=True)

def infer():
    proto=json.loads((OUT/'protocol.json').read_text());audit=json.loads((OUT/'mask_audit.json').read_text())
    expected={(r['id'],r['method']):r for r in json.loads((PRIOR/'complete.json').read_text())['results']}
    driver.OUT=OUT
    def run_group(cases,gpu):
        results=[]
        for c in cases:
            row=next(r for r in audit if r['id']==c['id'])
            for method in METHODS:
                if row['methods'][method]['same_as_previous_input']:
                    previous=expected[c['id'],method];src=ROOT/previous['output'];assert driver.digest(src)==previous['sha256']
                    assert Image.open(src).size==tuple(c['input_size'])
                    log=PRIOR/'logs'/(c['id']+'_'+method+'.log');assert '[ERROR]' not in log.read_text(errors='replace')
                    dest=OUT/'outputs'/method/c['page']/'result'/(c['page']+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():dest.symlink_to(src)
                    results.append(dict(id=c['id'],method=method,reused=True,reason='Pixel-identical black/red and same source, model, code, background and sampling settings',
                        output=driver.relative(dest),sha256=driver.digest(dest),previous_output=previous['output'],previous_log_sha256=driver.digest(log)))
                else:
                    result=driver.run(c,method,gpu)
                    assert '[ERROR]' not in (OUT/'logs'/(c['id']+'_'+method+'.log')).read_text(errors='replace');results.append(result)
        return results
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_group,proto['cases'][i::2],i) for i in range(2)]
        results=[r for f in futures for r in f.result()]
    assert len(results)==18
    driver.save_json(OUT/'inference_audit.json',dict(results=results,new=sum(not r['reused'] for r in results),reused=sum(r['reused'] for r in results)))
    print('Inference audit: new',sum(not r['reused'] for r in results),'reused',sum(r['reused'] for r in results),flush=True)

def render():
    proto=json.loads((OUT/'protocol.json').read_text());results=json.loads((OUT/'inference_audit.json').read_text())['results']
    sheet=Image.new('RGB',(2200,6*650+65),'#f4f2ec');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(review.FONT,24)
    draw.text((12,10),'6課題：原画像／CSC文字／Sauvola文字／Howe文字 → 固定DDRM。赤判定は改善候補で固定。',font=font,fill='black')
    whole=Image.new('RGB',(2400,6*960),'#f4f2ec');wd=ImageDraw.Draw(whole)
    for i,c in enumerate(proto['cases']):
        cid=c['id'];p=c['page'];raw=np.asarray(Image.open(ROOT/c['source']).convert('RGB'));arrays=[raw]
        for method in METHODS:
            r=next(r for r in results if r['id']==cid and r['method']==method);assert driver.digest(ROOT/r['output'])==r['sha256']
            a=np.asarray(Image.open(ROOT/r['output']).convert('RGB'));assert a.shape==raw.shape
            red=review.load(OUT/'masks'/method/p/'03_text_red.png');assert not np.any(a[red]!=raw[red]);arrays.append(a)
        titles=['原画像','CSC文字 → DDRM','Sauvola文字 → DDRM','Howe文字 → DDRM']
        fig=review.figure(OUT/'figures'/(cid+'_6課題DDRM比較.png'),arrays,titles,proto['rois_xywh'][cid],cid)
        fig.thumbnail((2180,630));sheet.paste(fig,(10,65+650*i))
        review.figure(OUT/'text_context'/(cid+'_本文比較.png'),arrays,titles,CONTEXT[cid],cid)
        for j,(arr,title) in enumerate(zip(arrays,titles)):
            tile=Image.fromarray(arr);tile.thumbnail((580,880));whole.paste(tile,(j*600+10,i*960+65));wd.text((j*600+10,i*960+10),cid+' '+title,font=font,fill='black')
    sheet.save(OUT/'6課題_固定条件DDRM比較.png');whole.save(OUT/'6課題_全ページ比較.png')
    # Existing optional cleanup is a separate control, not the default method.
    iso=BASE/'howe_csc_fusion/isolated_csc_v2';p=next(c['page'] for c in proto['cases'] if c['id']=='04')
    assert np.array_equal(review.load(OUT/'masks/howe'/p/'03_text_red.png'),review.load(iso/'masks/isolated'/p/'03_text_red.png'))
    result=next(r for r in json.loads((iso/'ddrm_complete.json').read_text())['results'] if r['id']=='04')
    assert driver.digest(ROOT/result['output'])==result['sha256']
    c=next(c for c in proto['cases'] if c['id']=='04');raw=np.asarray(Image.open(ROOT/c['source']).convert('RGB'))
    basic=np.asarray(Image.open(OUT/'outputs/howe'/p/'result'/(p+'.png')).convert('RGB'));clean=np.asarray(Image.open(ROOT/result['output']).convert('RGB'))
    review.figure(OUT/'04_孤立点除去_追加条件比較.png',[raw,basic,clean],['原画像','Howe基本','Howe＋既存孤立点除去'],proto['rois_xywh']['04'],'04')
    spots=json.loads((iso/'04_spot_verification.json').read_text());counts=[]
    for s in spots['spots']:
        x,y,w,h=s['xywh'];sl=np.s_[y:y+h,x:x+w]
        counts.append(dict(xywh=s['xywh'],howe_dark_pixels=int((basic[sl].mean(axis=2)<=128).sum()),isolated_dark_pixels=int((clean[sl].mean(axis=2)<=128).sum())))
    x,y,w,h=CONTEXT['04'];sl=np.s_[y:y+h,x:x+w]
    driver.save_json(OUT/'04_optional_audit.json',dict(spots=counts,printing_changed_pixels=int(np.any(basic[sl]!=clean[sl],axis=2).sum()),
        previous_isolated_result=result,not_default=True,dark_counts_are_not_GT=True))
    print('Rendered six native ROI comparisons, six text contexts and optional isolated-point control',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer','render']);args=parser.parse_args()
    {'prepare':prepare,'infer':infer,'render':render}[args.stage]()
