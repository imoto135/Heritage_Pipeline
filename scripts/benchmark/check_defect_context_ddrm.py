"""Check the development candidate with the previously frozen DDRM operator.

Reuse saved npj/v6 outputs and couple candidate noise in exactly the same image
coordinates. No sampler, weights, coefficients or original images are changed.
"""
import csv
import json
from pathlib import Path
import shutil
import time

import compare_npj_v6_ddrm_edits as driver
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from build_npj_v6_comparison import sha,dump,read,write,FONT_PATH

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/defect_context_ddrm_pilot_v2_20261001'
CANDIDATE=ROOT/'tmp_work/defect_context_preprocessing_pilot_v2_20261001'
BASELINE=ROOT/'tmp_work/npj_v6_ddrm_edit_comparison_20261001'


def figure(images,path,title,note):
    h,w=images[0].shape[:2];scale=min(480/w,1000/h);dw,dh=round(w*scale),round(h*scale)
    canvas=Image.new('RGB',(4*dw+108,dh+243),(250,248,243));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT_PATH,25);small=ImageFont.truetype(FONT_PATH,21)
    draw.text((24,14),title,font=font,fill='black')
    draw.text((24,51),'同じDDRM・重み・20段・空間ノイズ。前処理マスクの違いを比較。',font=small,fill='black')
    for i,(image,label) in enumerate(zip(images,['原画像','npjマスク → DDRM','v6マスク → DDRM','試作マスク → DDRM'])):
        x=24+i*(dw+20);draw.text((x,91),label,font=font,fill='black')
        resized=cv2.resize(image,(dw,dh),interpolation=cv2.INTER_NEAREST)
        canvas.paste(Image.fromarray(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)),(x,128))
    draw.text((24,dh+148),'編集候補を欠損領域に接続。開発例での試作、元npj全工程の再現ではありません。',font=small,fill='black')
    draw.text((24,dh+182),note,font=small,fill='black');path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((BASELINE/'manifest.json').read_text())
    assert baseline['settings']==driver.SETTINGS
    for rel,digest in baseline['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    prior=json.loads((BASELINE/'complete.json').read_text())['outputs']
    candidate=json.loads((CANDIDATE/'manifest.json').read_text())
    for rel,digest in candidate['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    assert candidate['source_sha256']==sha(ROOT/'scripts/benchmark/probe_defect_context_preprocessing.py')
    coutputs=json.loads((CANDIDATE/'complete.json').read_text())['outputs']
    dependencies={str((BASELINE/'manifest.json').relative_to(ROOT)):sha(BASELINE/'manifest.json'),
                  str((CANDIDATE/'manifest.json').relative_to(ROOT)):sha(CANDIDATE/'manifest.json'),
                  str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                  str(driver.CKPT.relative_to(ROOT)):sha(driver.CKPT),
                  str(Path(driver.__file__).relative_to(ROOT)):sha(Path(driver.__file__))}
    specs=[]
    for source in driver.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            mask_path=CANDIDATE/'pages'/page['page']/'context_stroke/edit.png'
            assert sha(mask_path)==coutputs[str(mask_path.relative_to(CANDIDATE))]
            for path in [mask_path,ROOT/page['original'],source/'v6_masks'/page['page']/'edit.png']:
                dependencies[str(path.relative_to(ROOT))]=sha(path)
            specs.extend(dict(page=page,roi=roi,source=source) for roi in page['rois'])
    manifest=dict(settings=driver.SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)),
        candidate='development v2 paper-relative brown + neutral ridges + anchored v6 networks',
        pages=11,regions=15,selection='all 15 fixed ROIs; no outcome selection',
        sampler_or_weight_changes=False,original_npj_pipeline_reproduction=False,pixel_GT=False)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    (OUT/'snapshots').mkdir(exist_ok=True)
    for name in ['denoising.py','svd_replacement.py']:
        shutil.copyfile(BASELINE/'snapshots'/name,OUT/'snapshots'/name)
        assert sha(OUT/'snapshots'/name)==sha(driver.MODULE/'functions'/name)
    driver.OUT=OUT;runner=driver.DDRM('cuda:0');weight=driver.blend_weight();rows=[];start=time.monotonic()
    baseline_rows={r['id']:r for r in json.loads((BASELINE/'diagnostics.json').read_text())}
    repeated=False
    for spec in specs:
        page,roi,source=spec['page'],spec['roi'],spec['source'];id=roi['id'];dest=OUT/'native'/id
        original=read(ROOT/page['original']);newmask=read(CANDIDATE/'pages'/page['page']/'context_stroke/edit.png',True)
        oldmask=read(source/'v6_masks'/page['page']/'edit.png',True)
        H,W=original.shape[:2];x,y,w,h=roi['xywh'];x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(W,x+w+64),min(H,y+h+64)
        context=original[y0:y1,x0:x1];ch,cw=context.shape[:2];edit=newmask[y0:y1,x0:x1]
        s=(slice(y-y0,y-y0+h),slice(x-x0,x-x0+w));input_crop=context[s]
        assert np.all(oldmask[y0:y1,x0:x1]<=edit)
        base_native=BASELINE/'native'/id
        for name in ['original.png','npj.png','v6.png','v6_raw.png']:
            assert sha(base_native/name)==prior[str((base_native/name).relative_to(BASELINE))]
        assert np.array_equal(input_crop,read(base_native/'original.png'))
        reused=np.array_equal(edit,oldmask[y0:y1,x0:x1]);sampled=0
        if reused:
            final=read(base_native/'v6.png');raw=read(base_native/'v6_raw.png')
        else:
            padded=cv2.copyMakeBorder(context,0,128,0,128,cv2.BORDER_REFLECT_101)
            pmask=np.pad(edit,((0,128),(0,128)))
            accum=np.zeros((ch+128,cw+128,3),np.float32);wacc=np.zeros((ch+128,cw+128),np.float32)
            coords=[(yy,xx) for yy in range(0,ch,96) for xx in range(0,cw,96)]
            for index,(yy,xx) in enumerate(coords):
                patch=padded[yy:yy+128,xx:xx+128];mask=pmask[yy:yy+128,xx:xx+128]
                seed=driver.patch_seed(page['page'],x0+xx,y0+yy)
                result=runner.sample(patch,mask,seed);sampled+=int(mask.any())
                if mask.any() and not repeated:
                    assert np.array_equal(result,runner.sample(patch,mask,seed))
                    repeated=True
                accum[yy:yy+128,xx:xx+128]+=result.astype(np.float32)*weight[:,:,None]
                wacc[yy:yy+128,xx:xx+128]+=weight
            assert (wacc[:ch,:cw]>0).all()
            raw_full=np.clip(np.rint(accum[:ch,:cw]/wacc[:ch,:cw,None]),0,255).astype(np.uint8)
            final_full=raw_full.copy();final_full[~edit]=context[~edit]
            raw=raw_full[s];final=final_full[s]
        crop_edit=edit[s];added=crop_edit&~oldmask[y:y+h,x:x+w]
        changed=np.any(final!=input_crop,axis=2)
        assert not np.any(changed&~crop_edit)
        for name,value in [('original.png',input_crop),('candidate.png',final),('candidate_raw.png',raw),
                           ('candidate_edit.png',crop_edit),('added.png',added),('candidate_changed.png',changed)]:
            write(dest/name,value);assert np.array_equal(read(dest/name,value.dtype==bool),value)
        previous=baseline_rows[id]
        note=f'変更画素：v6 {previous["v6_changed_pixels"]:,} / 試作 {int(changed.sum()):,}　追加編集候補 {int(added.sum()):,}（精度ではない）'
        figure([input_crop,read(base_native/'npj.png'),read(base_native/'v6.png'),final],
               OUT/'comparisons'/(id+'.png'),roi['title']+' · '+page['page'],note)
        rows.append(dict(id=id,page=page['page'],xywh=roi['xywh'],baseline_failure=roi['baseline_failure'],
            candidate_changed_pixels=int(changed.sum()),candidate_edit_pixels=int(crop_edit.sum()),
            added_edit_pixels=int(added.sum()),v6_changed_pixels=previous['v6_changed_pixels'],
            candidate_vs_v6_changed_pixels=int(np.any(final!=read(base_native/'v6.png'),axis=2).sum()),
            sampled_patches=sampled,reused_exact_same_context_mask=reused,nonedit_changed_pixels=0))
        dump(OUT/'progress.json',dict(completed_rois=len(rows),total_rois=15,latest=id))
        print('Candidate DDRM:',len(rows),'/15',id,'sampled',sampled,flush=True)
        del original,newmask,oldmask
    dump(OUT/'diagnostics.json',rows)
    with (OUT/'diagnostics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',DDRM=True,pages=11,regions=15,
        sampler_and_weights_fixed=True,spatial_noise_coupled=True,native_pixels=True,
        same_mask_repeat_passed=repeated,outside_edit_changed_pixels=0,pixel_GT=False,
        source_hashes_checked=True,model_forward_calls=runner.model_calls,
        elapsed_seconds=time.monotonic()-start,independent_validation=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*.png')}))
    print('Candidate DDRM completed: 11 pages / 15 fixed ROIs',flush=True)


if __name__=='__main__':main()
