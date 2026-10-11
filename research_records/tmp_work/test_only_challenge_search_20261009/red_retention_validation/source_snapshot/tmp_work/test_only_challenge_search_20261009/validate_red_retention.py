"""Audit saved CSC red with fixed CSC/Howe black masks and frozen DDRM.

Include a red-overlay-off control with exactly the same Howe black mask.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import argparse
import importlib.util
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import run_ddrm_pilot as driver
import probe_howe_csc_fusion as probe

ROOT=driver.ROOT
BASE=Path(__file__).resolve().parent
OUT=BASE/'red_retention_validation'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
SELECTION=[
    ('R01','200019865_00002_1',[1350,100,650,650],'濃淡のある二つの朱印',
     {'black_text':[950,1000,800,1000]}),
    ('R02','200017458_00003_2',[1790,80,320,620],'明瞭な朱印と茶色い綴じ側領域',
     {'brown_binding':[70,150,200,2800],'black_text':[650,900,850,900]}),
    ('R03','200010454_00002_2',[1220,1830,520,480],'墨と重なる朱印と別の小朱印',
     {'black_text':[650,800,700,800]}),
    ('R04','200003803_00003_1',[200,100,750,550],'既存06の淡い赤印状の像',
     {'black_text':[700,900,800,1000]}),
]


def load(path):
    a=np.asarray(Image.open(path).convert('L'))
    assert np.isin(a,[0,255]).all()
    return a<127


def figure(path,arrays,titles,roi,cid):
    x,y,w,h=roi
    canvas=Image.new('RGB',(len(arrays)*(w+12)+12,h+70),'#f4f2ec')
    draw=ImageDraw.Draw(canvas)
    for i,(arr,title) in enumerate(zip(arrays,titles)):
        text=cid+' '+title;size=24
        while size>10:
            font=ImageFont.truetype(FONT,size)
            if draw.textbbox((0,0),text,font=font)[2]<=w:break
            size-=1
        px=12+i*(w+12);draw.text((px,12),text,font=font,fill='black')
        tile=Image.fromarray(arr[y:y+h,x:x+w]).convert('RGB');canvas.paste(tile,(px,58))
        assert np.array_equal(np.asarray(canvas.crop((px,58,px+w,58+h))),np.asarray(tile))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    return canvas


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    old=json.loads((BASE/'ddrm_pilot/protocol.json').read_text())
    cases=[]
    for cid,p,roi,reason,controls in SELECTION:
        src=ROOT/'data/test'/p.split('_')[0]/(p+'.jpg')
        raw=np.asarray(Image.open(src).convert('RGB'));size=[raw.shape[1],raw.shape[0]]
        for x,y,w,h in [roi]+list(controls.values()):assert x>=0 and y>=0 and x+w<=size[0] and y+h<=size[1]
        cases.append(dict(id=cid,page=p,source=driver.relative(src),source_sha256=driver.digest(src),
                          input_size=size,roi_xywh=roi,controls_xywh=controls,reason=reason))
    driver.save_json(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases,ddrm=old['ddrm'],howe=dict(c=160,sigma=.6,low=.1,high=.4),
        black_red_priority='Remove all saved CSC-red pixels from both CSC and Howe black masks',
        saved_red_unchanged=True,selection_scope='Source-selected test examples, saved red inspected before new Howe/DDRM; not unbiased GT accuracy',
        red_off_control='R02: identical priority-resolved Howe black, red PNG all white; only final overlay differs',
        shared_stepwise_noise_guaranteed_between_black_methods=False,
        red_off_same_black_and_inputs=True,script_sha256=driver.digest(Path(__file__))))
    spec=importlib.util.spec_from_file_location('classical_red',ROOT/'tmp_work/classical_deep_review_20261008/run.py')
    deep=importlib.util.module_from_spec(spec);spec.loader.exec_module(deep)
    cv2.setNumThreads(2);rows=[]
    for case in cases:
        p=case['page'];src=ROOT/case['source'];raw=np.asarray(Image.open(src).convert('RGB'))
        d=ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p
        red=load(d/'03_text_red.png');csc=load(d/'02_text_black.png')
        if case['id']=='R04':
            howe=load(BASE/'three_method_comparison/masks'/p/'howe.png')
            howe_meta={'reused':'original six-case fixed Howe mask'}
        else:
            deep.SCRATCH=OUT/'scratch'/p;deep.SCRATCH.mkdir(parents=True,exist_ok=True)
            terms,term_meta=deep.howe_terms(deep.base.gray(raw));howe,cut_meta=deep.howe(terms,160)
            howe_meta=dict(terms=term_meta,cut=cut_meta)
        assert howe.shape==red.shape==csc.shape==raw.shape[:2]
        overlaps=dict(csc=int((csc&red).sum()),howe=int((howe&red).sum()))
        csc &= ~red;howe &= ~red
        methods={'csc':csc,'howe':howe}
        if case['id']=='R02':methods['howe_no_red']=howe
        hashes={}
        for method,black in methods.items():
            probe.save(OUT/'masks'/method/p/'02_text_black.png',black)
            probe.save(OUT/'masks'/method/p/'03_text_red.png',np.zeros_like(red) if method=='howe_no_red' else red)
            assert not (black&red).any()
            for q in (OUT/'masks'/method/p).glob('*.png'):hashes[driver.relative(q)]=driver.digest(q)
        inp=OUT/'inputs'/p;inp.mkdir(parents=True,exist_ok=True);dst=inp/src.name
        if not dst.exists():dst.symlink_to(src)
        overlay=raw.copy();overlay[red]=[20,235,20]
        figure(OUT/'mask_figures'/(case['id']+'_赤マスク比較.png'),
               [raw,np.repeat(np.where(red,0,255).astype(np.uint8)[:,:,None],3,2),overlay],
               ['原画像','保存CSC赤','赤候補（緑）'],case['roi_xywh'],case['id'])
        x,y,w,h=case['roi_xywh'];sl=np.s_[y:y+h,x:x+w]
        control_counts={}
        for key,r in case['controls_xywh'].items():
            xx,yy,ww,hh=r;control_counts[key]=int(red[yy:yy+hh,xx:xx+ww].sum())
            # Whole control rectangles remain the source-only audit scope.
        deps={driver.relative(q):driver.digest(q) for q in [src,d/'02_text_black.png',d/'03_text_red.png']}
        row=dict(id=case['id'],page=p,red_pixels_page=int(red.sum()),red_pixels_roi=int(red[sl].sum()),
                 red_pixels_outside_roi=int(red.sum()-red[sl].sum()),red_control_pixels=control_counts,
                 black_red_overlap_before_priority=overlaps,howe=howe_meta,dependencies=deps,masks_sha256=hashes)
        rows.append(row);driver.save_json(OUT/'mask_diagnostics.json',rows)
        print(case['id'],'red page/ROI',row['red_pixels_page'],row['red_pixels_roi'],
              'controls',control_counts,'overlap',overlaps,flush=True)
    driver.save_json(OUT/'masks_complete.json',dict(cases=len(cases)))


def infer():
    proto=json.loads((OUT/'protocol.json').read_text());diag=json.loads((OUT/'mask_diagnostics.json').read_text())
    assert (OUT/'masks_complete.json').exists()
    for attr,key in [('PIPELINE','code_sha256'),('CHECKPOINT','checkpoint_sha256'),('BACKGROUND','background_sha256')]:
        assert driver.digest(getattr(driver,attr))==proto['ddrm'][key]
    for row in diag:
        for rel,sha in {**row['dependencies'],**row['masks_sha256']}.items():assert driver.digest(ROOT/rel)==sha
    driver.OUT=OUT;cases=proto['cases']
    def group(case,methods,gpu):
        results=[]
        for method in methods:
            result=driver.run(case,method,gpu)
            assert '[ERROR]' not in (OUT/'logs'/f"{case['id']}_{method}.log").read_text(errors='replace')
            results.append(result)
        return results
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(group,cases[0],['csc','howe'],0),pool.submit(group,cases[1],['csc','howe'],1),
              pool.submit(group,cases[2],['csc','howe'],0),pool.submit(group,cases[1],['howe_no_red'],1)]
        results=[r for j in jobs for r in j.result()]
    # Reuse the original 06 outputs only when both masks, original, and settings match.
    oldproto=json.loads((BASE/'ddrm_pilot/protocol.json').read_text())
    oldcase=next(c for c in oldproto['cases'] if c['id']=='06')
    case=cases[3];p=case['page'];assert p==oldcase['page'] and case['source_sha256']==oldcase['source_sha256']
    assert proto['ddrm']==oldproto['ddrm']
    oldresults=json.loads((BASE/'ddrm_pilot/complete.json').read_text())['results']
    for method in ['csc','howe']:
        for name in ['02_text_black.png','03_text_red.png']:
            assert np.array_equal(np.asarray(Image.open(OUT/'masks'/method/p/name)),
                                  np.asarray(Image.open(BASE/'ddrm_pilot/masks'/method/p/name)))
        source=BASE/'ddrm_pilot/outputs'/method/p/'result'/(p+'.png')
        expected=next(r['sha256'] for r in oldresults if r['id']=='06' and r['method']==method)
        assert driver.digest(source)==expected
        dst=OUT/'outputs'/method/p/'result'/(p+'.png');dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():dst.symlink_to(source)
        assert driver.digest(dst)==expected
        results.append(dict(id=case['id'],method=method,reused=True,
                            reason='full black/red masks, source and DDRM settings identical to original 06',
                            output=driver.relative(dst),sha256=expected))
    boards=[];audits=[]
    for case,row in zip(cases,diag):
        p=case['page'];raw=cv2.imread(str(ROOT/case['source']))[:,:,::-1]
        red=load(OUT/'masks/howe'/p/'03_text_red.png');ov=raw.copy();ov[red]=[20,235,20]
        outputs=[np.asarray(Image.open(OUT/'outputs'/m/p/'result'/(p+'.png')).convert('RGB')) for m in ['csc','howe']]
        audit={}
        for method,array in zip(['csc','howe'],outputs):
            difference=int(np.any(array[red]!=raw[red],axis=1).sum())
            assert difference==0,'Red overlay did not copy the source pixels'
            audit[method]=dict(red_mask_source_rgb_changed_pixels=difference,
                output_sha256=driver.digest(OUT/'outputs'/method/p/'result'/(p+'.png')))
        board=figure(OUT/'ddrm_figures'/(case['id']+'_赤保持比較.png'),[raw,ov]+outputs,
                     ['原画像','CSC赤候補（緑）','CSC黒＋CSC赤','Howe黒＋CSC赤'],case['roi_xywh'],case['id'])
        boards.append(board);audits.append(dict(id=case['id'],red_pixels=int(red.sum()),methods=audit))
        for key,roi in case['controls_xywh'].items():
            figure(OUT/'control_figures'/(case['id']+'_'+key+'.png'),[raw,ov]+outputs,
                   ['原画像','CSC赤候補（緑）','CSC黒＋CSC赤','Howe黒＋CSC赤'],roi,case['id'])
    case=cases[1];p=case['page'];raw=cv2.imread(str(ROOT/case['source']))[:,:,::-1]
    red=load(OUT/'masks/howe'/p/'03_text_red.png')
    on=np.asarray(Image.open(OUT/'outputs/howe'/p/'result'/(p+'.png')).convert('RGB'))
    off=np.asarray(Image.open(OUT/'outputs/howe_no_red'/p/'result'/(p+'.png')).convert('RGB'))
    assert np.array_equal(load(OUT/'masks/howe'/p/'02_text_black.png'),load(OUT/'masks/howe_no_red'/p/'02_text_black.png'))
    recomposite=off.copy();recomposite[red]=raw[red]
    on_off=dict(id='R02',same_black_mask=True,
        outside_red_changed_pixels=int(np.any(on[~red]!=off[~red],axis=1).sum()),
        red_region_changed_pixels=int(np.any(on[red]!=off[red],axis=1).sum()),
        external_recomposite_vs_pipeline_on_changed_pixels=int(np.any(on!=recomposite,axis=2).sum()))
    assert on_off['outside_red_changed_pixels']==0 and on_off['external_recomposite_vs_pipeline_on_changed_pixels']==0
    for name,roi in [('朱印',case['roi_xywh']),('綴じ側',case['controls_xywh']['brown_binding'])]:
        x,y,w,h=roi;sl=np.s_[y:y+h,x:x+w]
        on_off[name]=dict(xywh=roi,red_mask_pixels=int(red[sl].sum()),
                         changed_output_pixels=int(np.any(on[sl]!=off[sl],axis=2).sum()))
        figure(OUT/'red_off_control'/('R02_'+name+'_赤合成ON_OFF.png'),[raw,off,on],
               ['原画像','赤合成OFF','赤合成ON'],roi,'R02')
    width=max(p.width for p in boards);height=sum(p.height for p in boards)+16*(len(boards)-1)
    sheet=Image.new('RGB',(width,height),'#ddd');y=0
    for board in boards:sheet.paste(board,(0,y));y+=board.height+16
    path=OUT/'4例_CSC赤とHowe併用_DDRM比較.png';sheet.save(path)
    sheet.resize((1800,round(height*1800/width)),Image.Resampling.LANCZOS).save(OUT/'4例_CSC赤とHowe併用_DDRM比較_preview.png')
    for row in diag:
        for rel,sha in {**row['dependencies'],**row['masks_sha256']}.items():assert driver.digest(ROOT/rel)==sha
    driver.save_json(OUT/'ddrm_complete.json',dict(results=results,red_source_copy_audit=audits,
        red_on_off_control=on_off,inputs_masks_unchanged=True,
        figure=driver.relative(path),figure_sha256=driver.digest(path)))
    print(json.dumps(on_off,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer']);a=p.parse_args()
    prepare() if a.stage=='prepare' else infer()
