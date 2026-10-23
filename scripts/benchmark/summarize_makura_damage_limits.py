"""Summarize development stress tests without claiming benchmark accuracy."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ['MPLCONFIGDIR']='tmp_work/makura_damage_challenge_20261021/mpl_config'
import json
from pathlib import Path

import cv2
import numpy as np

from probe_makura_damage_limits import ROOT,OUT,FAMILIES,STRENGTHS,rgb,gray,dump,sha,materials,apply_damage

KEYS=['csc_v2','sauvola_w61_k0.20','howe_t0.400_c160.000','howe_t0.250_c640.000']
NAMES={'csc_v2':'CSC v2','sauvola_w61_k0.20':'Sauvola w61 k.20',
 'howe_t0.400_c160.000':'Howe c160 thi.4','howe_t0.250_c640.000':'Howe c640 thi.25'}


def aggregate(records,key):
    scores=[]
    for record in records:
        if key=='howe_reference_best':
            s=min([s for s in record['scores'] if s['method'].startswith('howe_')],key=lambda s:s['balanced_error'])
        else:s=next(s for s in record['scores'] if s['method']==key)
        scores.append(s)
    positive=sum(s['ink_n'] for s in scores);negative=sum(s['nonink_n'] for s in scores)
    return dict(cases=len(records),ink_retention=sum(s['ink_hit'] for s in scores)/positive,
        nonink_false_rate=sum(s['nonink_false'] for s in scores)/negative,
        ink_n=positive,nonink_n=negative,formal_accuracy=False)


def main():
    records=json.loads((OUT/'all_results.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    selections={s['patch_id']:s for s in protocol['selections']}
    summary=[];regional=[];visibility=[]
    for family in FAMILIES:
        for strength in STRENGTHS:
            subset=[r for r in records if r['family']==family and r['strength']==strength]
            for key in KEYS+['howe_reference_best']:
                summary.append(dict(family=family,strength=strength,method=key,**aggregate(subset,key)))
    for record in records:
        if record['family']=='identity':continue
        selection=selections[record['patch_id']];raw=rgb(ROOT/selection['original'])
        target=gray(ROOT/selection['target'])==0;valid=gray(ROOT/selection['valid'])>0
        valid[:64]=False;valid[-64:]=False;valid[:,:64]=False;valid[:,-64:]=False
        paper=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
        mat=materials(raw,target,valid,selection['split'],selection['patch_id'],record['family'])
        reconstructed=apply_damage(raw,paper,mat,record['family'],record['strength'])
        assert np.array_equal(reconstructed,rgb(ROOT/record['folder']/'input.png'))
        counter=apply_damage(paper,paper,mat,record['family'],record['strength'])
        signal=np.max(np.abs(reconstructed.astype(np.int16)-counter.astype(np.int16)),axis=2)
        core=(mat['field']>.8)&target&valid
        visibility.append(dict(patch_id=record['patch_id'],family=record['family'],strength=record['strength'],
            core_known_ink=int(core.sum()),core_median_contrast=float(np.median(signal[core])) if core.any() else None,
            core_fraction_below2=float((signal[core]<2).mean()) if core.any() else None,
            core_fraction_below8=float((signal[core]<8).mean()) if core.any() else None,
            estimated_paper_not_gt=True))
        support=mat['field']>.1
        for score in record['scores']:
            pred=gray(ROOT/record['folder']/(score['method']+'.png'))==0
            positive=target&valid&support;negative=~target&valid&support
            regional.append(dict(patch_id=record['patch_id'],split=record['split'],family=record['family'],strength=record['strength'],
                method=score['method'],affected_ink=int(positive.sum()),affected_ink_hit=int((positive&pred).sum()),
                affected_nonink=int(negative.sum()),affected_nonink_false=int((negative&pred).sum())))
    howe_keys=[s['method'] for s in records[0]['scores'] if s['method'].startswith('howe_')]
    training=[r for r in records if r['split']=='train' and r['family']!='identity']
    order=sorted(howe_keys,key=lambda key:sum(next(s['balanced_error'] for s in r['scores'] if s['method']==key) for r in training))
    chosen=order[0]
    # This is development tuning on provisional labels, not an independent test.
    common=dict(method=chosen,selected_on='48 damaged train calibration cases; mean masked balanced error',
        training=aggregate(training,chosen),validation=aggregate([r for r in records if r['split']=='val' and r['family']!='identity'],chosen),
        candidate_order=order,formal_accuracy=False)
    dump(OUT/'summary_metrics.json',dict(by_family_strength=summary,common_howe=common))
    dump(OUT/'damage_support_metrics.json',regional);dump(OUT/'core_signal_diagnostics.json',visibility)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,7),dpi=160)
    for col,family in enumerate(FAMILIES):
        for key in KEYS:
            points=[aggregate([r for r in records if r['family']=='identity'],key)]
            points += [next(s for s in summary if s['family']==family and s['strength']==strength and s['method']==key) for strength in STRENGTHS]
            xs=[0]+list(STRENGTHS)
            axes[0,col].plot(xs,[x['ink_retention'] for x in points],marker='o',label=NAMES[key])
            axes[1,col].plot(xs,[x['nonink_false_rate'] for x in points],marker='o',label=NAMES[key])
        axes[0,col].set_title(family);axes[0,col].set_ylim(.5,1.01)
        axes[1,col].set_ylim(0,.1);axes[1,col].set_xlabel('Severity (within family only)')
        for row in (0,1):axes[row,col].grid(alpha=.25)
    axes[0,0].set_ylabel('Retained provisional ink');axes[1,0].set_ylabel('False ink in provisional paper')
    axes[0,0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Development diagnostics on provisional labels: NOT formal accuracy',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'damage_tradeoffs.png');plt.close(fig)
    thin=json.loads((OUT/'source_thin_probe_diagnostics.json').read_text())
    total=sum(x['ink_center_pixels'] for x in thin['teacher_audit'])
    accepted=sum(x['old_teacher_valid_ink'] for x in thin['teacher_audit'])
    unknown=sum(x['old_teacher_unknown'] for x in thin['teacher_audit'])
    negative=sum(x['old_teacher_valid_paper'] for x in thin['teacher_audit'])
    lines=['# 損傷条件を広げた古典法の限界調査','',
        '2026-10-21。枕草子の固定6ページ・各1パッチに、4種類×3強度の損傷と正常入力を作成。78入力に対し、Howe8設定・Sauvola5設定・CSC v2の14条件、計1,092出力を比較した。',
        '', '**暫定教師による開発診断であり、専門家正解に対する正式な精度比較ではない。有限の設定探索から古典法全体の限界やGANの必要性を証明したとは扱わない。**','',
        '## 損傷と分割','',
        '原画像の選択は、前の試作で固定した学習側先頭4ページ、検証側先頭2ページの先頭512角パッチ。出力を見てページを選び直していない。最終評価資料は使っていない。同資料内の開発校正であり、独立した最終評価ではない。',
        '', '損傷は、不規則な染みの乗算透過、文字に接触する黒い斑点、濃い水染みの縁、局所退色と粒状ノイズ。形・位置・ノイズは強度間で固定し、教師も固定した。実損傷画像の汚れを素材として移植していない。輪郭や材質はまだ人工的であり、物理的な再現性や実損傷と同じ分布は未確認。DKDS公式合成の再現ではない。',
        '', '各手法は同じ損傷512角入力を処理し、中央384角の有効教師だけで文字保持と汚れ残存を別々に集計した。正常入力での出力も計算し、パッチ再処理による差と損傷効果を区別できるようにした。Howeは従来と同じ近似Cannyのローカル移植。',
        '', '## 確認できた違い','',
        '- 濃い不規則な染みでは、Sauvolaが文字を残す一方で紙を黒くする。Howeは汚れを抑える一方、濃い領域内の文字を失う。',
        '- 黒い接触斑点では、以前の実画像D04で除去できたHoweも残存する。濃度・形・接触状態が違えば、同じ「黒い斑点」でも挙動が変わる。',
        '- 水染みの縁ではHoweの設定によって残存が変わり、固定設定より調整済み設定が常に良いとは限らない。',
        '- 強い退色＋ノイズでは、Howeの探索内参考最良でも欠落が残る。観測可能な筆画と、ほぼ情報がなくなった筆画を分ける必要がある。',
        '', '以下は強度.90の6ページ合計における「既存の有効な文字候補を残した割合 / 紙候補を黒くした割合」。教師を作る際にHowe・Sauvolaの一致を利用しているため偏りがある。文字認識精度ではない。','',
        '| 損傷 | CSC v2 | Sauvola既定 | Howe固定 | Howe調整 | Howe探索内参考最良 |',
        '|---|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        values=[]
        for key in KEYS+['howe_reference_best']:
            s=next(s for s in summary if s['family']==family and s['strength']==.9 and s['method']==key)
            values.append(f"{s['ink_retention']:.1%} / {s['nonink_false_rate']:.1%}")
        lines.append('| '+family+' | '+' | '.join(values)+' |')
    lines += ['', '探索内参考最良は、入力ごとに暫定教師を見て8設定から選んだ後知恵の参照値。自動選択の性能ではない。図の開発難例も学習側の結果から選んだ探索例であり、検証側の固定例と区別した。',
        '', f"参考として、学習側の48損傷条件で共通Howe設定を選ぶと `{chosen}` となった。この設定も暫定教師での開発選択であり、実画像での最適設定や汎化性能は保証しない。",'',
        '## 教師の薄線を別に確認','',
        f'原画像を拡大し、3パッチの細い筆画を部分トレースした。高確信の中心線{total}画素のうち、前回教師で有効な文字だったのは{accepted}画素、不明として除外されていたのは{unknown}画素、紙ラベルだったのは{negative}画素。',
        '', '細線の大半が教師に入っていないことを確認した。紙ラベルとの衝突には座標配置や二値境界の誤差も含まれうるため、これをそのまま密な教師の誤り率とは呼ばない。半径0・1・2画素の被覆を別保存し、暫定注釈の位置誤差も確認する。',
        '', '最初の粗いトレースは原画像に対してずれていたため、予測を採点する前に原画像の行・列の暗さと拡大画像で位置を修正した。修正前の注釈と図はhistory_before_source_alignmentに保存。過去の開発出力は既知であり盲検注釈ではない。低確信の区間は主診断から外した。注釈外を紙とみなしていない。',
        '', '新しいトレースは確認用の部分教師で、既存の160パッチの教師を勝手に密な正解へ昇格していない。次の学習では、このような薄線を含む少量の精密な教師を先に整える必要がある。','',
        '## 観測できる情報の確認','',
        '損傷前の画像と、文字を形態学的処理で埋めた紙推定画像へ同じ損傷を適用し、残るRGB差を計算した。薄線や重なり領域で文字信号が失われる条件を識別するための診断で、人が読めることや学習で復元できることの証明ではない。紙推定も真の無文字原本ではない。強度を上げて古典法が失敗したことだけを学習法の採用根拠にしない。','',
        '## 次の実験','',
        '1. 細線・汚れを含む少量の原画像から確認した教師を作る。既存の一致領域のまま学習量だけを増やさない。',
        '2. 弱・中・強の損傷と正常入力を残し、文字保持と非文字残存を分けて評価する。比較しやすい難例だけに対象を限定しない。',
        '3. 論文式GA候補と公開版の差を明示し、同じ教師・入力・予算でU-Net、GAあり生成器、GANありを比較する。GA候補は未確認の出力接続仮定を含むため、論文完全再現とは呼ばない。','',
        '[論文と公開コードの照合](../ragan_paper_alignment_20261021/論文との照合.md) · [設定](protocol.json) · [検証](verification.json) · [全集計](summary_metrics.json) · [損傷内だけの指標](damage_support_metrics.json) · [文字信号診断](core_signal_diagnostics.json) · [部分注釈診断](source_thin_probe_diagnostics.json)','',
        '## 比較PNG','', '[損傷強度による変化](damage_tradeoffs.png)','']
    for f in json.loads((OUT/'figure_verification.json').read_text()):
        lines.append('- ['+f['id']+']('+str(Path(f['path']).relative_to(OUT.relative_to(ROOT)))+')')
    lines += ['', '## 教師候補v3への反映', '',
        'その後、確認した高確信の細線中心125画素を有効文字として確保した別版を作成した。線幅を決められない近傍は無効化し、旧ラベルと入力は保存したまま。新規に有効な正例となったのは104画素で、残る21画素は既存の有効文字。修正は3パッチの部分注釈に限られ、密な正解全体を完成させたものではない。', '',
        '[教師v3の変更と比較PNG](../makura_teacher_v3_20261021/教師v3の変更.md)']
    (OUT/'損傷条件と限界調査.md').write_text('\n'.join(lines)+'\n')
    print('Saved metrics, native comparison index, tradeoff plot, and Japanese report',flush=True)


if __name__=='__main__':main()
