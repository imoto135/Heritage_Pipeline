"""Report completed paired runs only; never train or evaluate reserved test."""
import argparse
import numpy as np
from PIL import Image
from compare_v5_aligned_fade_control import dump,sha,rgb,mask
from apply_ragan_real_regions import panel
import run_ragan_dense_reference_pair as run


def main(stage):
    run.check();out=run.OUT/stage
    completions={a:run.read(out/a/'completion.json') for a in ['no_gan','gan']}
    ca,cb=completions.values()
    assert ca['initial_G_digest']==cb['initial_G_digest'] and ca['input_hashes']==cb['input_hashes']
    scores={a:run.read(out/a/'validation.json') for a in completions}
    manifest=run.read(run.FOUNDATION/'dense_benchmark_manifest.json')
    pages={p['id']:p for p in manifest['pages'] if p['split']=='val'}
    assert len(pages)==4 and all(len(r['rows'])==4 for r in scores.values())
    controls=run.read(run.FOUNDATION/'validation_controls.json')['rows'];figures=[];table=[]
    for key,page in pages.items():
        original=rgb(run.ROOT/page['image']);gt=np.asarray(Image.open(run.ROOT/page['gt']).convert('L'))
        patches=[p for p in manifest['patches'] if p['page']==key]
        # Same frozen illustration rule as foundation; all metrics use full page.
        selected=max(patches,key=lambda p:int((gt[p['xy'][1]:p['xy'][1]+256,p['xy'][0]:p['xy'][0]+256]==0).sum()))
        x,y=selected['xy'];w,h=selected['valid_wh'];sl=np.s_[y:y+h,x:x+w]
        def pad(a):
            b=np.full((256,256,3) if a.ndim==3 else (256,256),255,np.uint8);b[:h,:w]=a;return b
        items=[('原画像',pad(original[sl])),('公式GT',pad(gt[sl]))]
        for method in ['otsu','sauvola61','no_gan','gan']:
            if method in scores:
                row=next(r for r in scores[method]['rows'] if r['page']==key);path=run.ROOT/row['mask']
            else:
                row=next(r for r in controls if r['page']==key and r['method']==method);path=run.ROOT/row['mask_file']
            assert sha(path)==row['mask_sha256'];pred=mask(path);truth=gt==0
            tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum());f1=2*tp/max(1,2*tp+fp+fn)
            assert abs(f1-row['ink_f1'])<1e-12
            table.append(dict(page=key,method=method,tp=tp,fp=fp,fn=fn,ink_f1=f1))
            items.append((f'{method} 全頁F1={f1:.4f}',pad(np.where(pred[sl],0,255).astype(np.uint8))))
        f=out/'figures'/f'{key}.png';panel(f,key+' / '+stage+'・同じ初期Gと入力',items);figures.append(str(f.relative_to(out)))
    lines=['# 密な正解を用いた公開コード由来GAN有無の比較','',
        ('64更新の動作確認。GANの有効性・収束・限界の結論には使わない。' if stage=='smoke' else '固定60周の小規模denseGT比較。原論文の全学習データ・報告値の再現ではない。'),'',
        '|ページ|手法|全文字F1|余計な黒FP|文字欠落FN|','|---|---|---:|---:|---:|']
    for r in table:lines.append(f"|{r['page']}|{r['method']}|{r['ink_f1']:.6f}|{r['fp']}|{r['fn']}|")
    lines += ['','数値は検証4ページ全体、PNGは事前に固定した最多墨パッチの原寸例示。評価保留集合は未使用。','']
    for f in figures:lines.append(f'[{f}]({f})')
    (out/'比較結果.md').write_text('\n'.join(lines)+'\n')
    dump(out/'paired_report.json',dict(stage=stage,rows=table,figures={f:sha(out/f) for f in figures},
        reporter_sha256=sha(__file__),matched_initial_G=True,matched_augmented_inputs=True,reserved_test_scored=False))
    print('Saved paired report and4original/GT/baseline/GAN comparison PNGs',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['smoke','full'],default='smoke');main(a.parse_args().stage)
