"""Materialize dense train/val I/O and score fixed classical validation controls."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from compare_v5_aligned_fade_control import dump,sha,save
from apply_ragan_real_regions import panel
from audit_ragan_reference_contracts import load_reference

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/ragan_reproduction_foundation_20261023'


def score(pred,gt):
    truth=gt==0
    tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum());tn=int((~pred&~truth).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,ink_f1=2*tp/max(1,2*tp+fp+fn),
        binary_psnr_db=None if fp+fn==0 else float(10*np.log10(gt.size/(fp+fn))))


def main():
    assert not (OUT/'patch_io.json').exists()
    manifest=json.loads((OUT/'dense_benchmark_manifest.json').read_text())
    assert not (OUT/'patch_io_plan.json').exists()
    dump(OUT/'patch_io_plan.json',dict(script_sha256=sha(__file__),manifest_sha256=sha(OUT/'dense_benchmark_manifest.json'),
        split='Use only train/val pages; reserved_test never materialized or scored.',
        auxiliary_scope='Compute Otsu/Sobel at full native page, then crop jointly. Otsu uses PIL L; Sobel uses cv2RGB2GRAY as author tools. This explicit recipe resolves ambiguous script ordering for this controlled benchmark, not an established exact paper recipe.',
        padding='256 nonoverlap; white255 right/bottom padding for every channel, as author crop utility; record valid_wh.',
        controls='Full-page Otsu and Sauvola61,k.2,R128 on same PIL-L grayscale; score only4validation pages. F1 and binaryPSNR, no pFM/DRD claim.',
        figures='Most-ink native grid patch per validation page chosen from GT for illustration only; all numeric scores use full page.',no_training=True))
    rows=[];controls=[];figures=[]
    for page in manifest['pages']:
        if page['split']=='reserved_test':continue
        assert sha(ROOT/page['image'])==page['image_sha256'] and sha(ROOT/page['gt'])==page['gt_sha256']
        raw=np.asarray(Image.open(ROOT/page['image']).convert('RGB'));gt=np.asarray(Image.open(ROOT/page['gt']).convert('L'))
        gray=np.asarray(Image.fromarray(raw).convert('L'))
        otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        cg=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY)
        edge=np.rint(np.clip(cv2.magnitude(cv2.Sobel(cg,cv2.CV_64F,1,0,ksize=3),cv2.Sobel(cg,cv2.CV_64F,0,1,ksize=3)),0,255)).astype(np.uint8)
        pp=[p for p in manifest['patches'] if p['page']==page['id']]
        for p in pp:
            x,y=p['xy'];w,h=p['valid_wh'];name=f"{page['id']}_{y//256}_{x//256}.png";files={}
            for kind,a in [('img',raw),('gt',gt),('ostu',otsu),('sobel',edge)]:
                canvas=np.full((256,256,3) if a.ndim==3 else (256,256),255,np.uint8);canvas[:h,:w]=a[y:y+h,x:x+w]
                f=OUT/'patch_io'/page['split']/kind/name;f.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(canvas).save(f)
                files[kind]=dict(path=str(f.relative_to(ROOT)),sha256=sha(f))
            rows.append(dict(page=page['id'],split=page['split'],xy=[x,y],valid_wh=[w,h],files=files))
        if page['split']=='val':
            f=gray.astype(np.float32);mean=cv2.boxFilter(f,-1,(61,61),borderType=cv2.BORDER_REFLECT_101)
            sd=np.sqrt(np.maximum(cv2.boxFilter(f*f,-1,(61,61),borderType=cv2.BORDER_REFLECT_101)-mean*mean,0))
            masks=dict(otsu=otsu<128,sauvola61=f<mean*(1+.2*(sd/128.-1)))
            crop=max(pp,key=lambda p:int((gt[p['xy'][1]:p['xy'][1]+256,p['xy'][0]:p['xy'][0]+256]==0).sum()))
            x,y=crop['xy'];w,h=crop['valid_wh'];sl=np.s_[y:y+h,x:x+w]
            def pad(a):
                b=np.full((256,256,3) if a.ndim==3 else (256,256),255,np.uint8);b[:h,:w]=a;return b
            items=[('原画像',pad(raw[sl])),('公式の正解',pad(gt[sl]))]
            errors=[]
            for method,pred in masks.items():
                target=OUT/'validation_controls'/page['id']/f'{method}.png';save(target,pred)
                controls.append(dict(page=page['id'],method=method,**score(pred,gt),mask_file=str(target.relative_to(ROOT)),mask_sha256=sha(target)))
                items.append((method,pad(np.where(pred[sl],0,255).astype(np.uint8))))
                overlay=raw.copy();overlay[pred&(gt!=0)]=[255,0,60];overlay[~pred&(gt==0)]=[0,120,255]
                errors.append((method+' 赤=余計な黒 青=文字欠落',pad(overlay[sl])))
            fig=OUT/'figures'/f"{page['id']}_原画像と公式正解.png";panel(fig,page['id']+' / 学習なし・検証ページのみ',items+errors);figures.append(str(fig.relative_to(OUT)))
        print('Prepared',page['id'],page['split'],len(pp),'patches',flush=True)
    dump(OUT/'patch_io.json',dict(plan_sha256=sha(OUT/'patch_io_plan.json'),rows=rows,patches=len(rows),
        train_patches=sum(p['split']=='train' for p in rows),val_patches=sum(p['split']=='val' for p in rows),reserved_test_patches=0,no_training=True))
    dump(OUT/'validation_controls.json',dict(rows=controls,figures={f:sha(OUT/f) for f in figures},
        macro_ink_f1={m:float(np.mean([r['ink_f1'] for r in controls if r['method']==m])) for m in ['otsu','sauvola61']},
        no_GAN_evaluation=True,reserved_test_scored=False))
    loader=load_reference();checks=[]
    for split in ['train','val']:
        ds=loader.DocData(str(OUT/'patch_io'/split/'img'),(256,256),training=False,only_one=True)
        expected=[r for r in rows if r['split']==split];assert len(ds)==len(expected)
        for index in [0,len(ds)-1]:
            item=ds[index];path=Path(ds.ImgFiles[index]);gt=np.asarray(Image.open(str(path).replace('/img/','/gt/')))
            assert np.array_equal(item[3][0].numpy(),gt.astype(np.float32)/255)
            assert all(tuple(t.shape)==s for t,s in zip(item[:4],[(3,256,256),(1,256,256),(1,256,256),(1,256,256)]))
            checks.append(dict(split=split,index=index,name=path.name,dense_gt_tensor_bitexact=True))
    dump(OUT/'loader_io_verification.json',dict(checks=checks,official_loader_accepts_materialized_patches=True,no_training=True))
    print('Ready:',len(rows),'dense train/val patches; test untouched; no training',flush=True)


if __name__=='__main__':main()
