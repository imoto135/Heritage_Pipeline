"""Fixed-DDRM and separately labelled local-fill diagnostic for margin probe.

All 22 original-selected ROIs are included; the old 15 paired outputs are reused
after hashes are checked. There is no weight/sampler tuning or mask correction.
"""
import csv
import gc
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import compare_npj_v6_ddrm_edits as driver
import probe_paper_margin_defects as candidate
from collect_saved_npj_failures import saved_masks
from build_npj_v6_comparison import dump, sha, read, write, FONT_PATH

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/paper_margin_defect_ddrm_20261001'
MASKS = candidate.OUT
BASELINE = ROOT / 'tmp_work/npj_v6_ddrm_edit_comparison_20261001'


def figure(images, labels, path, title, note, kind):
    h,w=images[0].shape[:2];scale=min(480/w,900/h);dw,dh=round(w*scale),round(h*scale)
    canvas=Image.new('RGB',(len(images)*(dw+20)+28,dh+244),(250,248,243))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT_PATH,25);small=ImageFont.truetype(FONT_PATH,21)
    draw.text((24,14),title,font=font,fill='black');draw.text((24,51),kind,font=small,fill='black')
    for i,(im,label) in enumerate(zip(images,labels)):
        x=24+i*(dw+20);draw.text((x,91),label,font=font,fill='black')
        resized=cv2.resize(im,(dw,dh),interpolation=cv2.INTER_NEAREST)
        canvas.paste(Image.fromarray(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)),(x,128))
    draw.text((24,dh+148),'編集範囲外は原画素。開発診断で、元npj全工程の再現・失われた内容の真正性検証ではない。',font=small,fill='black')
    draw.text((24,dh+182),note,font=small,fill='black');path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def restore(runner, image, mask, page, gx, gy, repeat_state):
    h,w=image.shape[:2]
    padded=cv2.copyMakeBorder(image,0,128,0,128,cv2.BORDER_REFLECT_101)
    pmask=np.pad(mask,((0,128),(0,128)))
    accum=np.zeros((h+128,w+128,3),np.float32);weights=np.zeros((h+128,w+128),np.float32)
    weight=driver.blend_weight();sampled=0
    for y in range(0,h,96):
        for x in range(0,w,96):
            patch=padded[y:y+128,x:x+128];missing=pmask[y:y+128,x:x+128]
            seed=driver.patch_seed(page,gx+x,gy+y)
            result=runner.sample(patch,missing,seed);sampled+=int(missing.any())
            if missing.any() and not repeat_state['passed']:
                assert np.array_equal(result,runner.sample(patch,missing,seed))
                repeat_state['passed']=True
            accum[y:y+128,x:x+128]+=result.astype(np.float32)*weight[:,:,None]
            weights[y:y+128,x:x+128]+=weight
    assert (weights[:h,:w]>0).all()
    raw=np.clip(np.rint(accum[:h,:w]/weights[:h,:w,None]),0,255).astype(np.uint8)
    final=raw.copy();final[~mask]=image[~mask]
    return final,raw,sampled


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((BASELINE/'manifest.json').read_text())
    assert baseline['settings']==driver.SETTINGS
    for rel,digest in baseline['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    previous=json.loads((BASELINE/'complete.json').read_text())['outputs']
    manifest=json.loads((MASKS/'candidate_manifest.json').read_text())
    assert manifest['settings']==candidate.SETTINGS
    for rel,digest in manifest['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    outputs=json.loads((MASKS/'candidate_complete.json').read_text())['outputs']
    for rel,digest in outputs.items():assert sha(MASKS/rel)==digest,rel
    dependencies={str((MASKS/'candidate_manifest.json').relative_to(ROOT)):sha(MASKS/'candidate_manifest.json'),
        str((BASELINE/'manifest.json').relative_to(ROOT)):sha(BASELINE/'manifest.json'),
        str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        str(Path(driver.__file__).relative_to(ROOT)):sha(Path(driver.__file__)),
        str(driver.CKPT.relative_to(ROOT)):sha(driver.CKPT)}
    frozen=dict(settings=driver.SETTINGS,dependencies=dependencies,
        selection='all 22 fixed ROIs, 15 prior plus 7 new; no output-based omission',
        pages=14,regions=22,weight_sampler_changes=False,
        supplemental_fill='OpenCV Telea radius 3, same original context and candidate mask; explicitly not DDRM',
        original_npj_pipeline_reproduction=False,pixel_GT=False)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==frozen
    else:dump(OUT/'manifest.json',frozen)
    (OUT/'snapshots').mkdir(exist_ok=True)
    for name in ['denoising.py','svd_replacement.py']:
        shutil.copyfile(BASELINE/'snapshots'/name,OUT/'snapshots'/name)
        assert sha(OUT/'snapshots'/name)==sha(driver.MODULE/'functions'/name)
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    driver.OUT=OUT;runner=driver.DDRM('cuda:0');repeat_state=dict(passed=False)
    rows=[];started=time.monotonic()
    for source in candidate.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            idpage=page['page'];original=read(ROOT/page['original']);H,W=original.shape[:2]
            npj=saved_masks(page)['edit'];v6=read(source/'v6_masks'/idpage/'edit.png',True)
            proposed=read(MASKS/'candidate_pages'/idpage/'dark_support/edit.png',True)
            assert np.all(v6<=proposed)
            for roi in page['rois']:
                id=roi['id'];x,y,w,h=roi['xywh'];x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(W,x+w+64),min(H,y+h+64)
                context=original[y0:y1,x0:x1];s=(slice(y-y0,y-y0+h),slice(x-x0,x-x0+w));crop=context[s]
                masks=dict(npj=npj[y0:y1,x0:x1],v6=v6[y0:y1,x0:x1],candidate=proposed[y0:y1,x0:x1])
                finals={};raws={};samples={};reused={}
                old=BASELINE/'native'/id
                for name in ['npj','v6','candidate']:
                    if name in ['npj','v6'] and old.exists():
                        for file in ['original.png',name+'.png',name+'_raw.png']:
                            assert sha(old/file)==previous[str((old/file).relative_to(BASELINE))]
                        assert np.array_equal(crop,read(old/'original.png'))
                        finals[name]=read(old/(name+'.png'));raws[name]=read(old/(name+'_raw.png'))
                        samples[name]=0;reused[name]='frozen_previous_output'
                    elif name=='candidate' and np.array_equal(masks[name],masks['v6']):
                        finals[name]=finals['v6'].copy();raws[name]=raws['v6'].copy()
                        samples[name]=0;reused[name]='same_context_mask_as_v6'
                    else:
                        final,raw,sampled=restore(runner,context,masks[name],idpage,x0,y0,repeat_state)
                        finals[name]=final[s];raws[name]=raw[s];samples[name]=sampled;reused[name]=False
                    assert not np.any(np.any(finals[name]!=crop,axis=2)&~masks[name][s])
                if masks['candidate'].any():
                    local=cv2.inpaint(context,masks['candidate'].astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
                    local[~masks['candidate']]=context[~masks['candidate']]
                else:local=context.copy()
                local=local[s];assert not np.any(np.any(local!=crop,axis=2)&~masks['candidate'][s])
                dest=OUT/'native'/id
                write(dest/'original.png',crop);write(dest/'candidate_telea.png',local)
                for name in finals:
                    for file,value in [(name+'.png',finals[name]),(name+'_raw.png',raws[name]),(name+'_edit.png',masks[name][s])]:
                        write(dest/file,value);assert np.array_equal(read(dest/file,value.dtype==bool),value)
                added=masks['candidate'][s]&~masks['v6'][s];write(dest/'added.png',added)
                note=f'追加編集候補 {int(added.sum()):,}画素（精度ではない）。同じ原画像・重み・20段・空間ノイズ。'
                figure([crop,finals['npj'],finals['v6'],finals['candidate']],
                    ['原画像','npjマスク → DDRM','v6マスク → DDRM','余白試作 → DDRM'],
                    OUT/'comparisons'/(id+'.png'),roi['title']+' · '+idpage,note,
                    '同じDDRMで前処理マスクの差を比較。旧出力はハッシュ確認して再利用。')
                figure([crop,finals['candidate'],local],['原画像','余白試作 → DDRM','同じマスク → Telea'],
                    OUT/'fill_diagnostic'/(id+'.png'),roi['title']+' · 復元器の診断',
                    'Teleaは局所補間でDDRMではない。マスクを固定し、紙面生成時の差を見る。',
                    '同じ編集範囲・原画像・周辺64px。学習済みDDRMと局所補間を別々に表示。')
                rows.append(dict(id=id,page=idpage,xywh=roi['xywh'],baseline_failure=roi['baseline_failure'],
                    added_pixels=int(added.sum()),
                    **{name+'_changed_pixels':int(np.any(value!=crop,axis=2).sum()) for name,value in dict(**finals,telea=local).items()},
                    sampled_patches=samples,reused=reused,outside_edit_changed_pixels=0))
                dump(OUT/'progress.json',dict(completed_regions=len(rows),total_regions=22,latest=id))
                print('Margin DDRM:',len(rows),'/22',id,samples,flush=True)
            del original,npj,v6,proposed;gc.collect()
    dump(OUT/'diagnostics.json',rows)
    with (OUT/'diagnostics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',pages=14,regions=22,
        actual_DDRM=True,weight_sampler_fixed=True,spatial_noise_coupled=True,
        same_mask_repeat_passed=repeat_state['passed'],outside_edit_changed_pixels=0,
        source_output_hashes_checked=True,Telea_separately_labelled=True,
        elapsed_seconds=time.monotonic()-started,model_forward_calls=runner.model_calls,
        pixel_GT=False,independent_accuracy=False,adopted=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*.png')}))
    print('Fixed DDRM + separately labelled local-fill diagnostic complete: 22 ROIs',flush=True)


if __name__=='__main__':main()
