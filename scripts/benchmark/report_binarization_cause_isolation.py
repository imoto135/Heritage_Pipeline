"""Post-readout reporting and independent artifact checks; no new selection."""
import json,csv
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import isolate_binarization_causes as exp
from compare_v5_aligned_fade_control import sha,dump,mask
from report_background_diversity_control import table

ROOT=exp.ROOT;OUT=exp.OUT


def main():
    plan=exp.check();cal=exp.read(OUT/'calibration.json');ctx=exp.read(OUT/'context.json')
    assert cal['protocol_sha256']==ctx['protocol_sha256']==sha(OUT/'protocol.json')
    cases={k:{n:np.asarray(Image.open(OUT/'labels'/k/f'{n}.png'))>0 for n in ['core','paper']} for k in exp.KEYS}
    yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4;verified=[]
    def verify(pred,c,r):
        dist=ndi.distance_transform_edt(~pred) if pred.any() else np.full(pred.shape,np.inf)
        independent=dict(core_n=int(c['core'].sum()),core_exact=int((pred&c['core']).sum()),core_r2=int(((dist<=2)&c['core']).sum()),
                         paper_n=int(c['paper'].sum()),paper_ink=int((pred&c['paper']).sum()),total_ink=int(pred.sum()))
        assert all(r[k]==v for k,v in independent.items()),r
    for r in cal['rows']:
        if r['operating']=='calibrated':
            f=ROOT/r['mask_file'];assert sha(f)==r['mask_sha256'];pred=mask(f);verified.append(dict(path=r['mask_file'],sha256=sha(f)))
        else:pred=np.load(exp.pfile(r['condition'],r['grid'],r['case']))<.5
        verify(pred,cases[r['case']],r)
    effects=[]
    for n in exp.CONDITIONS:
        for k in ['S03','S07']:
            inp=OUT/'inputs'/k;labels={'core':np.asarray(Image.open(inp/'core.png'))>0,'paper':np.asarray(Image.open(inp/'stain.png'))>0}
            by={r['variant']:r for r in ctx['rows'] if r['condition']==n and r['case']==k}
            for variant,r in by.items():
                f=ROOT/r['mask_file'];p=np.load(ROOT/r['probability_file']);assert sha(f)==r['mask_sha256'] and sha(ROOT/r['probability_file'])==r['probability_sha256']
                assert np.array_equal(mask(f),p<.5);verify(mask(f),labels,r);verified.append(dict(path=r['mask_file'],sha256=sha(f)))
            for factor,base,altered in [('outer','reflect_local','page_local'),('outer','reflect_page','page_page'),('central_aux','page_local','page_page'),('central_aux','reflect_local','reflect_page')]:
                a=np.load(ROOT/by[base]['probability_file']);b=np.load(ROOT/by[altered]['probability_file']);pa=a<.5;pb=b<.5
                effects.append(dict(condition=n,case=k,factor=factor,base=base,altered=altered,
                    core_r2_delta=by[altered]['core_r2']-by[base]['core_r2'],paper_black_delta=by[altered]['paper_ink']-by[base]['paper_ink'],
                    prediction_disagreement=int((pa!=pb).sum()),core_disagreement=int(((pa!=pb)&labels['core']).sum()),paper_disagreement=int(((pa!=pb)&labels['paper']).sum()),
                    mean_abs_probability_difference=float(np.abs(a-b).mean()),max_abs_probability_difference=float(np.abs(a-b).max())))
    # Selection objective must be reproduced entirely from calibration inputs.
    selections=[]
    for r in cal['thresholds']:
        keys=plan['calibration']['folds'][r['fold']]['calibration'];p=[np.load(exp.pfile(r['condition'],'full512',k)) for k in keys]
        again=exp.choose_threshold(p,[cases[k] for k in keys]);assert again['threshold']==r['threshold']
        selections.append(dict(condition=r['condition'],fold=r['fold'],D04_excluded=True,threshold=r['threshold']))
    # Numeric boundary regression: nextafter event must actually include ties.
    tiny=np.float32(1e-12);t=np.nextafter(float(tiny),np.inf)
    fixture=np.array([[tiny,.7]],np.float32);lab=dict(core=np.array([[True,False]]),paper=np.array([[False,True]]))
    assert exp.score(fixture,lab,t)['core_exact']==1 and exp.score(fixture,lab,float(tiny))['core_exact']==0
    assert exp.choose_threshold([fixture],[lab])['calibration_macro_balanced_error']==0
    figures=cal['figures']+ctx['figures'];assert len(figures)==24 and all((OUT/f).exists() for f in figures)
    assert len(verified)==136 and len(effects)==32
    dump(OUT/'factor_effects.json',dict(rows=effects,interpretation='Paired fixed512 input interventions, not identification of a layer or proof of dominant cause'))
    # Flat output avoids nested geometry/path fields and duplicate score tables.
    for name,rows in [('閾値の全条件.csv',cal['rows']),('文脈の全条件.csv',ctx['rows']),('文脈要因の差.csv',effects)]:
        fields=sorted({k for r in rows for k in r})
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    transfer=[]
    for chosen in cal['thresholds']:
        n,fold=chosen['condition'],chosen['fold'];faint=next(k for k in plan['calibration']['folds'][fold]['transfer'] if k!='D04')
        def row(k,o):return next(r for r in cal['rows'] if (r['condition'],r['fold'],r['grid'],r['case'],r['operating'])==(n,fold,'full512',k,o))
        a,b=row('D04','fixed'),row('D04','calibrated');c,d=row(faint,'fixed'),row(faint,'calibrated')
        transfer.append([n,fold,f"{chosen['threshold']:.10g}",f"{a['paper_ink']}→{b['paper_ink']}",f"{c['core_exact']}→{d['core_exact']}/{d['core_n']}",f"{c['core_r2']}→{d['core_r2']}/{d['core_n']}"])
    context_rows=[[r['condition'],r['case'],r['variant'],f"{r['core_r2']}/{r['core_n']}",r['paper_ink'],r.get('CPU_vs_saved_CUDA_disagreement_pixels','—')] for r in ctx['rows']]
    rankrows=[[r['condition'],r['grid'],r['mode'],f"{r['auc_paper_above_ink']:.5f}",r['foreground_zero'],r['background_zero']] for r in cal['rank_diagnostics']]
    lines=['# 原因切り分け：固定モデルでの閾値転用と入力介入','',
      '2026-10-23。追加学習・新注釈・GANなし。前実験の全4モデルを固定。前の開発結果を見た後に計画を固定した探索的診断であり、盲検評価・正式な精度比較ではない。','',
      '[計画と入力ハッシュ](protocol.json)・[研究判断](研究判断.md)。','',
      '## 1. D04を使わずに選んだ閾値は転用できるか','',
      'R03＋R05、R03＋R06の2通りで、各領域の墨/紙のクラス均衡誤り率を領域間で平均し、厳密画素一致で最小にする。半径2指標は閾値選択に使わない。同点は0.5に近いもの、その後小さいもの。候補は校正側のスコアのみ。512で選んだ一つの閾値を全3格子へ適用し、格子ごとには選び直さない。','',
      '**D04は選択に不使用。** R05_to_R06の評価薄字はR06、逆はR05。どれも既知の開発例であり、新しい独立検証集合ではない。この閾値を製品・採用重みに反映しない。','',
      table(['条件','校正→薄字確認','閾値','D04黒/5100','別領域墨・厳密','別領域墨・半径2'],transfer),'',
      'D04には文字の主注釈がない。D04黒化の減少だけを成功とせず、別の薄字と原寸PNGを併記する。D04自体の補助文字経路は前実験を参照し、本診断で新しいGTには変更していない。','',
      '## 2. 中央RGBを固定した外周×補助入力の2×2','',
      'S03（文字と斑点）とS07（黒い斑点）の学習原画像を使う。512入力で、外周5チャンネルを原本/中央256の反射に切り替える。中央RGBは全条件同じ、中央Otsu/Sobelは単独256での計算/原本512での計算を独立に切り替える。異なる外周でも中央5チャンネルが同一であることを検査した。中央補助を変える対では外周を含め他画素不変。','',
      '単独256は参考の基点で、512との比較では寸法・境界・計算文脈が同時に変わる。因果の主比較は512内の2×2。反射外周や補助チャンネルの貼り替えは人工的な介入で、推奨前処理ではない。S03はページ右端付近のため中央の左上が512内(139,128)、S07は(128,128)。原本512に外部のパディングは追加していない。','',
      'CPU同一バックエンド・float32で40条件と4反復を実行し、各モデルの同一512入力反復がビット一致することとモデル不変性を確認した。CUDA保存結果との比較は単独256基点での参考診断に限定し、512内の要因差はすべてCPU同士で取る。','',
      table(['条件','原画像','入力','墨半径2','斑点黒/576','単独256のCPU/CUDA判定差'],context_rows),'',
      '[要因ごとの予測変化と確率差](文脈要因の差.csv)。32対すべてを記録し、良い初期化だけを選ばない。S03の墨115点と斑点576画素、S07の斑点576画素は前実験の暫定部分注釈をそのまま使用。','',
      '## 3. 閾値を動かしても変わらないスコア順序','',
      'R05+R06の墨芯とD04の紙についてP(紙の紙確率 > 墨芯の紙確率)+0.5×同値確率を算出。順位AUCと独立な対数えを照合した。高いほどこの2群の順位分離が良いが、字形・線幅・認識率ではない。exactは墨芯そのもの、radius2_proxyは周囲半径2の最小紙確率であり、後者は墨芯そのものの分類ではない。','',
      table(['条件','格子','指標','順位AUC','墨スコア0','紙スコア0'],rankrows),'',
      '## 原画像付き比較PNG','', '\n'.join(f'- [{f}]({f})' for f in figures),'',
      '[閾値192件](閾値の全条件.csv)・[文脈40件](文脈の全条件.csv)・[検証記録](verification.json)。閾値192件には固定0.5の重複読出しが含まれる。保存した新しいマスクは校正96＋介入40＝136枚であり、独立評価画像数とは異なる。','']
    (OUT/'比較と診断.md').write_text('\n'.join(lines))
    dump(OUT/'verification.json',dict(protocol_sha256=sha(OUT/'protocol.json'),readout_files={f:sha(OUT/f) for f in ['calibration.json','context.json','factor_effects.json']},
        independent_masks=verified,calibration_rows=192,context_rows=40,factor_pairs=32,rank_rows=24,calibration_selections=selections,
        nextafter_boundary_fixture_passed=True,model_states_unchanged=ctx['unchanged_models'],repeated_forward=ctx['repeated_forward'],
        figures={f:sha(OUT/f) for f in figures},report_script_sha256=sha(__file__),formal_accuracy=False))
    print('Verified136 saved masks,8 calibration choices,32 factor pairs,24 PNGs',flush=True)


if __name__=='__main__':main()
