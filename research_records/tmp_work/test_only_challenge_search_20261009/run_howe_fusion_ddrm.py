"""Run the fixed component-removal candidate with the previous DDRM settings."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image,ImageDraw,ImageFont

import run_ddrm_pilot as driver

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OUT=BASE/'howe_csc_fusion'
METHOD='component'
PRIOR=BASE/'ddrm_pilot'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def figure(case):
    page=case['page'];cid=case['id'];x,y,w,h=driver.ROIS[cid]
    original=np.asarray(Image.open(ROOT/case['source']).convert('RGB'))
    baseline=np.asarray(Image.open(PRIOR/'outputs/howe'/page/'result'/(page+'.png')).convert('RGB'))
    candidate=np.asarray(Image.open(OUT/'outputs'/METHOD/page/'result'/(page+'.png')).convert('RGB'))
    oldmask=np.asarray(Image.open(PRIOR/'masks/howe'/page/'02_text_black.png').convert('L'))<127
    newmask=np.asarray(Image.open(OUT/'masks'/METHOD/page/'02_text_black.png').convert('L'))<127
    assert not np.any(newmask&~oldmask)
    removed=oldmask&~newmask
    overlay=original.copy();overlay[removed]=[255,45,45]
    header,gap=68,16
    sheet=Image.new('RGB',(w*4+gap*5,h+header+gap),'#f4f2ec')
    draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,26)
    for i,(name,array) in enumerate(zip(['原画像','Howe → DDRM','試作 → DDRM','除去マスク（赤）'],[original,baseline,candidate,overlay])):
        px=gap+i*(w+gap)
        draw.text((px,10),cid+' '+name,font=font,fill='black')
        sheet.paste(Image.fromarray(array[y:y+h,x:x+w]),(px,header))
    path=OUT/'ddrm_figures'/f'{cid}_DDRM比較.png'
    path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path)
    s=np.s_[y:y+h,x:x+w]
    return dict(id=cid,roi_xywh=driver.ROIS[cid],removed_mask_pixels_roi=int(removed[s].sum()),
                changed_output_pixels_roi=int(np.any(baseline[s]!=candidate[s],axis=2).sum()),
                changed_output_pixels_page=int(np.any(baseline!=candidate,axis=2).sum()),
                baseline=driver.relative(PRIOR/'outputs/howe'/page/'result'/(page+'.png')),
                baseline_sha256=driver.digest(PRIOR/'outputs/howe'/page/'result'/(page+'.png')),
                figure=driver.relative(path))

def main():
    prior=json.loads((PRIOR/'protocol.json').read_text())
    baseline_expected={r['id']:r['sha256'] for r in json.loads((PRIOR/'complete.json').read_text())['results'] if r['method']=='howe'}
    cases=prior['cases']
    assert driver.digest(driver.PIPELINE)==prior['ddrm']['code_sha256']
    assert driver.digest(driver.CHECKPOINT)==prior['ddrm']['checkpoint_sha256']
    assert driver.digest(driver.BACKGROUND)==prior['ddrm']['background_sha256']
    dependencies=json.loads((OUT/'protocol.json').read_text())['dependencies']
    for rel,expected in dependencies.items():assert driver.digest(ROOT/rel)==expected,rel
    masks={}
    for case in cases:
        source=ROOT/case['source'];page=case['page']
        inp=OUT/'inputs'/page;inp.mkdir(parents=True,exist_ok=True)
        path=inp/source.name
        if not path.exists():path.symlink_to(source)
        oldred=PRIOR/'masks/howe'/page/'03_text_red.png'
        newred=OUT/'masks'/METHOD/page/'03_text_red.png'
        assert np.array_equal(np.asarray(Image.open(oldred)),np.asarray(Image.open(newred)))
        for filename in ['02_text_black.png','03_text_red.png']:
            p=OUT/'masks'/METHOD/page/filename
            masks[driver.relative(p)]=driver.digest(p)
        assert driver.digest(PRIOR/'outputs/howe'/page/'result'/(page+'.png'))==baseline_expected[case['id']]
    protocol=dict(cases=cases,ddrm=prior['ddrm'],
        candidate=('CSC-supported isolated round components only' if METHOD=='isolated' else 'CSC degradation evidence or isolated round-mark rule'),
        candidate_masks_sha256=masks,black_mask_only_changed=True,
        shared_stepwise_noise_guaranteed=False,development_examples=True,
        driver_sha256=driver.digest(Path(driver.__file__)),runner_sha256=driver.digest(Path(__file__)))
    driver.save_json(OUT/'ddrm_protocol.json',protocol)
    driver.OUT=OUT
    def group(items,gpu):
        results=[]
        for case in items:
            page=case['page'];oldmask=PRIOR/'masks/howe'/page/'02_text_black.png';newmask=OUT/'masks'/METHOD/page/'02_text_black.png'
            identical=np.array_equal(np.asarray(Image.open(oldmask)),np.asarray(Image.open(newmask)))
            if identical:
                src=PRIOR/'outputs/howe'/page/'result'/(page+'.png')
                dst=OUT/'outputs'/METHOD/page/'result'/(page+'.png')
                dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():dst.symlink_to(src)
                assert driver.digest(dst)==baseline_expected[case['id']]
                result=dict(id=case['id'],method=METHOD,reused=True,reason='full-page mask and red mask exactly identical',
                            output=driver.relative(dst),sha256=driver.digest(dst))
                print('REUSE EXACT',case['id'],flush=True)
            else:
                result=driver.run(case,METHOD,gpu)
                log=OUT/'logs'/f"{case['id']}_{METHOD}.log"
                assert '[ERROR]' not in log.read_text(errors='replace'),log
            results.append(result)
        return results
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(group,[cases[0],cases[1],cases[3]],0)
        b=pool.submit(group,[cases[2],cases[4],cases[5]],1)
        results=a.result()+b.result()
    statistics=[figure(case) for case in cases]
    panels=[Image.open(ROOT/s['figure']).convert('RGB') for s in statistics]
    width=max(p.width for p in panels);height=sum(p.height for p in panels)+20*(len(panels)-1)
    sheet=Image.new('RGB',(width,height),'#dddddd');y=0
    for p in panels:sheet.paste(p,(0,y));y+=p.height+20
    fig=OUT/'6課題_Howe_試作_DDRM比較.png';sheet.save(fig)
    for rel,expected in dependencies.items():assert driver.digest(ROOT/rel)==expected,rel
    for rel,expected in masks.items():assert driver.digest(ROOT/rel)==expected,rel
    driver.save_json(OUT/'ddrm_complete.json',dict(results=results,diagnostics=statistics,
        sources_unchanged=True,figure=driver.relative(fig),figure_sha256=driver.digest(fig)))
    print('FIGURE',fig,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--safe-isolated',action='store_true');args=parser.parse_args()
    if args.safe_isolated:OUT=OUT/'isolated_csc_v2';METHOD='isolated'
    main()
