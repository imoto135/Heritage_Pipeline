"""Reuse the exact constituent outputs, change only blending support.

Margin64=(512-384)/2 is geometry-derived, never selected by image scores.
"""
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import decompose_binarization_tiles as tile
import isolate_binarization_causes as exp
from compare_v5_aligned_fade_control import dump,sha,mask,save,rgb
from apply_ragan_real_regions import panel
from report_background_diversity_control import table

ROOT=exp.ROOT;OUT=exp.OUT


def freeze():
    exp.check();assert not (OUT/'aggregation_plan.json').exists()
    dump(OUT/'aggregation_plan.json',dict(status='Adaptive diagnostic frozen before full tile decomposition readout',
        parent_protocol_sha256=sha(OUT/'protocol.json'),tile_plan_sha256=sha(OUT/'tile_plan.json'),script_sha256=sha(__file__),
        margin=64,reason='Half of512-384overlap; no margin search',
        fixed='Same model outputs, probability values, threshold.5 andHann; modify only support to exclude tile pixels within64 of edge',
        uncovered='Fall back to original weighted probability wherever restricted support denominator is0; record count',
        scope='All4models,bothsourcecases,bothauxvariants. Diagnostic only; no adoption from these2training sources.',no_training=True))
    print('Frozen geometry-derived64margin aggregation diagnostic',flush=True)


def run():
    parent,plan=tile.check();p=exp.read(OUT/'aggregation_plan.json')
    assert p['script_sha256']==sha(__file__) and p['tile_plan_sha256']==sha(OUT/'tile_plan.json')
    assert not (OUT/'aggregation.json').exists();d=exp.read(OUT/'tiles.json');assert d['tile_plan_sha256']==sha(OUT/'tile_plan.json')
    yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    weight=np.outer(np.maximum(np.hanning(512),.05),np.maximum(np.hanning(512),.05));support=np.zeros((512,512),bool);support[64:448,64:448]=True
    checked=[];rows=[];figures=[]
    def verify(pred,c,r):
        dist=ndi.distance_transform_edt(~pred) if pred.any() else np.full(pred.shape,np.inf)
        values=dict(core_n=int(c['core'].sum()),paper_n=int(c['paper'].sum()),core_exact=int((pred&c['core']).sum()),core_r2=int(((dist<=2)&c['core']).sum()),paper_ink=int((pred&c['paper']).sum()),total_ink=int(pred.sum()))
        assert all(r[k]==v for k,v in values.items())
    for n in exp.CONDITIONS:
        for c in parent['context']['cases']:
            k=c['id'];x0,y0,w,h=c['roi'];base=OUT/'inputs'/k
            core=np.asarray(Image.open(base/'core.png'))>0;paper=np.asarray(Image.open(base/'stain.png'))>0;labels=dict(core=core,paper=paper)
            annotation=rgb(base/'original.png').copy();annotation[core]=[0,100,255];annotation[paper]=[0,190,80]
            items=[('中央の原画像',rgb(base/'original.png'))]
            for variant in plan['variants']:
                subset=[r for r in d['rows'] if (r['condition'],r['case'],r['variant'])==(n,k,variant)]
                total=np.zeros((h,w),np.float64);den=total.copy();restricted=total.copy();restricted_den=total.copy()
                for r in subset:
                    a=np.load(ROOT/r['probability_file']);pred=mask(ROOT/r['mask_file']);assert np.array_equal(pred,a<.5)
                    assert sha(ROOT/r['probability_file'])==r['probability_sha256'] and sha(ROOT/r['mask_file'])==r['mask_sha256']
                    l,t,ww,hh=r['overlap_roi_xywh'];rs=np.s_[t:t+hh,l:l+ww];x,y=r['tile_xy'];ts=np.s_[y0+t-y:y0+t-y+hh,x0+l-x:x0+l-x+ww]
                    valid=np.zeros((h,w),bool);valid[rs]=True;verify(pred,dict(core=core&valid,paper=paper&valid),r)
                    checked.append(dict(path=r['mask_file'],sha256=r['mask_sha256']))
                    wt=weight[ts];total[rs]+=a[rs]*wt;den[rs]+=wt;rw=wt*support[ts];restricted[rs]+=a[rs]*rw;restricted_den[rs]+=rw
                assert (den>0).all();original=(total/den).astype(np.float32)
                before=next(r for r in d['blends'] if (r['condition'],r['case'],r['variant'])==(n,k,variant))
                assert np.array_equal(original,np.load(ROOT/before['probability_file']));verify(original<.5,labels,before)
                assert sha(ROOT/before['mask_file'])==before['mask_sha256'];checked.append(dict(path=before['mask_file'],sha256=before['mask_sha256']))
                out=original.copy();covered=restricted_den>0;out[covered]=(restricted[covered]/restricted_den[covered]).astype(np.float32)
                dest=OUT/'aggregation'/n/k;dest.mkdir(parents=True,exist_ok=True);np.save(dest/f'{variant}.npy',out);save(dest/f'{variant}.png',out<.5)
                metrics=exp.score(out,labels,.5);r=dict(condition=n,case=k,variant=variant,fallback_pixels=int((~covered).sum()),
                    fallback_core=int((~covered&core).sum()),fallback_paper=int((~covered&paper).sum()),original_core_r2=before['core_r2'],original_paper_ink=before['paper_ink'],
                    probability_file=str((dest/f'{variant}.npy').relative_to(ROOT)),probability_sha256=sha(dest/f'{variant}.npy'),
                    mask_file=str((dest/f'{variant}.png').relative_to(ROOT)),mask_sha256=sha(dest/f'{variant}.png'),**metrics)
                verify(out<.5,labels,r);rows.append(r);checked.append(dict(path=r['mask_file'],sha256=r['mask_sha256']))
                prefix='補助512' if variant=='native_aux' else '中央補助256'
                items.extend([(prefix+f" 元 墨{before['core_r2']} 黒{before['paper_ink']}",np.where(original<.5,0,255).astype(np.uint8)),
                              (prefix+f" 端64除外 墨{r['core_r2']} 黒{r['paper_ink']}",np.where(out<.5,0,255).astype(np.uint8))])
            items.append(('部分注釈 青=墨 緑=斑点',annotation));f=OUT/'figures'/f'{n}_{k}_aggregation.png';panel(f,k+' '+n+' / 同じタイル出力・合成だけ変更',items);figures.append(str(f.relative_to(OUT)))
    dump(OUT/'aggregation.json',dict(plan_sha256=sha(OUT/'aggregation_plan.json'),tile_results_sha256=sha(OUT/'tiles.json'),rows=rows,figures=figures,independent_masks=checked,original_blends_bit_exact=True,no_training=True))
    summary=[[r['condition'],r['case'],r['variant'],f"{r['original_core_r2']}→{r['core_r2']}/{r['core_n']}",f"{r['original_paper_ink']}→{r['paper_ink']}",r['fallback_pixels']] for r in rows]
    lines=['# 実ページのタイル分解と、合成だけを変える補助診断','',
        '中央寄りの512で消えたS07斑点が実ページで戻る理由を追う適応的な追加診断。先行の入力介入結果を見て計画し、タイル出力を読む前に計画を固定した。別冊への汎化評価ではない。','',
        '[タイル計画](tile_plan.json)・[合成計画](aggregation_plan.json)・[各タイルと元の合成](tiles.json)・[再採点と変更後](aggregation.json)。','',
        '元のページ512/stride384/末尾格子/Hann下限0.05をCPU上で再構成。各タイルの出力を保存し、同じRGB・同じタイル位置・同じ合成重みで、ROI内だけのOtsu/Sobelを単独256の値へ差し替える対照も実施した。タイル間では見える周囲・境界までの距離・内部位置が同時に異なるため、位置だけの因果とは呼ばない。','',
        '続いて保存した全タイル出力から、入力・重み・閾値を固定したまま合成の有効範囲だけを変更。512-384の重なりの半分に当たる64画素を各タイル端から除外した。64は画像で探索した値ではない。有効なタイルがなくなる画素は元の合成へ戻し、その数を記録した。これは有利なタイルを画像ごとに選ぶ操作ではない。','',
        table(['条件','原画像','補助入力','墨半径2','斑点黒/576','元合成へ戻した画素'],summary),'',
        'CPUで再構成した元の合成と、旧CUDA出力の判定差はtiles.jsonの各blendに記録。同一CPU出力から元の合成を再計算した値は全16件でビット一致した。タイル単位と合成後の全保存マスクを独立に再採点した。','',
        '## 原画像付きPNG','', '\n'.join(f'- [{f}]({f})' for f in d['figures']+figures),'',
        '[研究判断](研究判断.md)・[先行の閾値/中央RGB固定対照](比較と診断.md)。入力介入の改善を、新しい学習モデルの精度や実運用の採用結果として報告しない。','']
    (OUT/'タイル分解と合成の診断.md').write_text('\n'.join(lines))
    print('Verified actual tiles and16 original blends; created16 aggregation masks',len(checked),'mask records',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);args=a.parse_args();freeze() if args.action=='freeze' else run()
