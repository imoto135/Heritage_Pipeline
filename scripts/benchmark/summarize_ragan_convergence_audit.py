"""Compare fixed-duration endpoints, preserve image evidence, report implementation limits."""
from __future__ import annotations
import json
import re
import sys
import os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp_work/ragan_convergence_audit_20261022/matplotlib_config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_convergence_audit import OUT
from compare_v5_aligned_fade_control import sha,dump,rgb,mask
import apply_ragan_real_regions as real


def main():
    dest=OUT/'summary';dest.mkdir(exist_ok=True)
    sources=[OUT/'evaluation/results.json',OUT/'nogan_evaluation/results.json',OUT/'unet_evaluation/results.json']
    sets=[json.loads(p.read_text()) for p in sources]
    synthetic=sum([d['synthetic'] for d in sets],[])
    realrows=sum([d['real'] for d in sets],[])
    traced=sum([d['traced_fade'] for d in sets],[])
    names={'ragan_original':'RA公開版・旧汚れ','ragan_contact':'RA公開版・暗い汚れ',
        'ra_no_gan_contact':'同じ生成器・GANなし','unet_contact':'U-Net・暗い汚れ'}
    for condition in names:
        kind='unet' if condition=='unet_contact' else ('ra_no_gan' if condition=='ra_no_gan_contact' else 'ragan')
        completed=json.loads((OUT/condition/'models'/kind/'completion.json').read_text())
        assert completed['training_completed'] and completed['steps']==2160
    for condition in ('ra_no_gan_contact','unet_contact'):
        for name in ('train_manifest.jsonl','val_manifest.jsonl'):
            assert (OUT/condition/name).read_bytes()==(OUT/'ragan_contact'/name).read_bytes()
    def syn(condition,step,family='contact_stain'):
        rows=[r for r in synthetic if r['condition']==condition and r['step']==step and r['family']==family]
        return (sum(r['core_r2'] for r in rows),sum(r['core_n'] for r in rows),sum(r['paper_ink'] for r in rows),sum(r['paper_n'] for r in rows))
    def reg(condition,step,region):
        return next(r for r in realrows if r['condition']==condition and r['step']==step and r['region']==region)
    def spot(condition,step,case):
        return next(r for r in realrows if r['condition']==condition and r['step']==step and r['case']==case)['nonink_false']
    def tr(condition,step):return next(r for r in traced if r['condition']==condition and r['step']==step)
    synth_table=[];text_table=[];spot_table=[]
    for condition,label in names.items():
        for step in (432,2160):
            c,n,p,pn=syn(condition,step)
            synth_table.append(f'| {label} | {step//216} | {c}/{n} | {p}/{pn} |')
        text_table.append(f"| {label} | {reg(condition,432,'thin')['ink_r2']}→{reg(condition,2160,'thin')['ink_r2']} /225 | {reg(condition,432,'contact_upper')['ink_r2']}→{reg(condition,2160,'contact_upper')['ink_r2']} /124 | {tr(condition,432)['r2']}→{tr(condition,2160)['r2']} /49 |")
        spot_table.append(f'| {label} | {spot(condition,432,"D04")}→{spot(condition,2160,"D04")} /5,100 | {spot(condition,432,"T07")}→{spot(condition,2160,"T07")} /4,500 | {spot(condition,432,"C01")}→{spot(condition,2160,"C01")} /10,800 |')
    classic={c['id']:c for c in json.loads((ROOT/'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']}
    other={c['id']:c for c in json.loads((ROOT/'tmp_work/makura_contact_stain_train_control_20261022/other_book_transfer/protocol.json').read_text())['cases']}
    figures=[]
    for key in ('D01','D03','D04','T07','C01'):
        c=classic[key] if key in classic else other[key]
        source=rgb(ROOT/c['source']);x,y,w,h=c.get('roi',c.get('crop_xywh'));source=source[y:y+h,x:x+w]
        hp=(ROOT/'tmp_work/classical_limit_audit_20261021/crops'/key/'howe_fixed.png') if key in ('D01','D03','D04') else \
            ROOT/'tmp_work/makura_contact_stain_train_control_20261022/other_book_transfer/masks'/key/'howe_fixed.png'
        items=[('原本',source),('Howe固定',np.where(mask(hp),0,255).astype(np.uint8))]
        for folder,condition,step,label in [('evaluation','ragan_contact',432,'RA公開版・暗い汚れ 2周'),
                ('evaluation','ragan_contact',2160,'RA公開版・暗い汚れ 10周'),
                ('nogan_evaluation','ra_no_gan_contact',2160,'同じ生成器・GANなし 10周'),
                ('unet_evaluation','unet_contact',2160,'U-Net 10周')]:
            p=mask(OUT/folder/'real_masks'/condition/f'{step:05d}'/(key+'.png'))
            items.append((label,np.where(p,0,255).astype(np.uint8)))
        fig=dest/'figures'/(key+'_原本比較.png');real.panel(fig,key+' / 学習量とGANの対照',items)
        figures.append(dict(case=key,path=str(fig.relative_to(OUT)),sha256=sha(fig)))
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    fig.suptitle('Exploratory: no-GAN / U-Net transfer readouts at epochs 2 and 10 only',fontsize=11)
    english={'ragan_original':'RA public / old stains','ragan_contact':'RA public / dark stains',
        'ra_no_gan_contact':'Same generator / no GAN','unet_contact':'U-Net / dark stains'}
    for condition,label in english.items():
        kind='unet' if condition=='unet_contact' else ('ra_no_gan' if condition=='ra_no_gan_contact' else 'ragan')
        history=json.loads((OUT/condition/'models'/kind/'history.json').read_text())
        axes[0,0].plot([r['step']/216 for r in history],[r['validation']['ink_f1_to_provisional'] for r in history],marker='.',label=label)
        steps=sorted({r['step'] for r in synthetic if r['condition']==condition})
        axes[0,1].plot([s/216 for s in steps],[100*reg(condition,s,'thin')['ink_r2']/225 for s in steps],marker='.')
        axes[1,0].plot([s/216 for s in steps],[100*syn(condition,s)[2]/syn(condition,s)[3] for s in steps],marker='.')
        axes[1,1].plot([s/216 for s in steps],[100*spot(condition,s,'D04')/5100 for s in steps],marker='.')
    for ax,title,ylabel in zip(axes.flat,['Validation to provisional teacher','D01 visible ink-center coverage (r2)',
            'Synthetic contact stain: paper proxy ink','D04 selected spot: black area'],
            ['Ink F1 (provisional)','Coverage (%)','Ink / paper proxy (%)','Black / selected region (%)']):
        ax.set(title=title,xlabel='Epochs (216 updates each)',ylabel=ylabel);ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);axes[0,0].set_ylim(0,1.02)
    fig.savefig(dest/'学習量と誤黒化の推移.png',dpi=160);plt.close(fig)
    gradients=json.loads((OUT/'gradient_audit.json').read_text())['rows']
    lo=min(r['weighted_adversarial_to_supervised_norm'] for r in gradients)
    hi=max(r['weighted_adversarial_to_supervised_norm'] for r in gradients)
    assert all(len(r['attention_gradients'])==48 and all(a['supervised_max']==a['adversarial_max']==0 for a in r['attention_gradients']) for r in gradients)
    cap=json.loads((OUT/'fixed_batch_capacity_fresh/completion.json').read_text())
    last=cap['history'][-1]
    lines='\n'.join(synth_table);textlines='\n'.join(text_table);spotlines='\n'.join(spot_table)
    note=f'''# 学習量と公開RA-GAN実装の再監査

2026-10-22。**これまでの432更新・2周の試験だけでRA-GANの文字保持能力や手法の限界を判断したのは早かった。** 損傷を変えず10周まで学習すると細線が大きく改善した。一方、学習が進むほど斑点を黒く残す例もある。現在の問題は、学習量、公開注意機構、GAN寄与、合成と実損傷の差を切り分けることである。

## 固定した条件

旧茶色汚れと暗い乗算汚れの入力・暫定教師・切り出し規則をそのまま使い、各24ページ群・96パッチ・864レコード・batch4・lr3e-5・seed20261021で2,160更新（10周）まで新規学習した。2/4/6/8/10周の全チェックポイントを保存し、**同じ新規学習の432と2,160を比較**する。以前の432更新モデルとは検証頻度等が異なり、完全な数値再現を主張しない。重みは最終2,160を主に評価し、見た目に合わせた途中重み選択・閾値変更は行っていない。追加対照は同じ暗い汚れデータでのGANなし生成器とU-Net。いずれも10周・同じ共通教師あり損失である。単一seed、専門家未確認教師、既知の開発例であり、論文再現や正式順位ではない。

## 学習不足がどれだけ影響したか

値は2周→10周の半径2画素被覆。D01/D03は既存の原本のみの暫定部分注釈、退色49点も原本から追跡した中心点で、専門家GTではない。

| 学習条件 | D01細線 | D03上側接触 | 既知の合成退色α0.70 |
|---|---:|---:|---:|
{textlines}

Howe固定はD01 181/225、D03上側89/124、合成退色42/49。ただし被覆だけでは文字の正確さ・余計な線・線幅を表さない。[D01の原本比較](summary/figures/D01_原本比較.png)、[D03の原本比較](summary/figures/D03_原本比較.png)。D03の既存非文字注釈は染み中央全体を覆わず、誤黒化0という値でも原本比較では染み下端などに残りが見える。

## 汚れ抑制との関係

最初から固定していた同一本の2検証ページ群・各1パッチ。暗部骨格と遠方紙は原本由来の代理である。

| 学習条件 | 周数 | 汚れ下骨格 r2 | 汚れ内紙の黒判定 |
|---|---:|---:|---:|
{lines}

旧茶色汚れ条件では学習を延ばしても暗い合成汚れ全体を黒化する。旧条件にはその暗い損傷方式を学習させておらず、これは見た目の異なる損傷への転移試験であって、旧条件が学習不能という意味ではない。暗い汚れ条件は紙の黒化が少ないが、周数に対して単調に減らない。**暫定検証F値だけで停止時点を決めると、接触汚れ・孤立斑点の悪化を見逃す。**

| 学習条件 | D04斑点内の黒判定 | 別本T07染み内 | 別本C01斑点内 |
|---|---:|---:|---:|
{spotlines}

いずれも2周→10周。小領域を原本上で先に指定した暫定測定で、材質・隠れた墨が不明な領域もある。Howe固定はD04 0/5,100、T07 0/4,500、C01 7,035/10,800。[D04](summary/figures/D04_原本比較.png)、[T07](summary/figures/T07_原本比較.png)、[C01](summary/figures/C01_原本比較.png)。

## 現在の公開コード版で注意とGANは働いているか

公開生成器は4か所で`GA(y,y)`を呼び、最終式で注意の効果が相殺される。今回も2周・10周の計4重みで、注意枝48テンソルに対する教師あり・GAN双方の勾配が全て0だった。この条件を「領域適応注意の有効性」の実験とは呼べない。論文式候補にも出力への適用式などの仮定が残るため、推測による置換で論文再現を宣言しない。

同じ固定4学習画像で、重み0.1を含むGAN損失の生成器勾配ノルムは教師あり損失の **{lo:.2e}〜{hi:.2e}倍**。識別器の実/偽スコアはほぼ0.498で近い。これはこのバッチ・評価モードでの局所診断であり、GANが全学習で無効だという証明ではない。

同学習量のGANなしでもD01細線204/225、D03上側115/124と回復し、GANありの195/225、109/124を上回った。一方、合成退色はGANなし45/49、あり48/49で、別本T07/C01の汚れはGANありがやや少ない。単一seedで指標ごとの利点が分かれ、**現時点では文字形状学習の改善をGAN固有の効果と主張できない**。U-Netは同じ10周でも暗い合成汚れをほぼ全面黒化したため、生成器構造の違いは引き続き検討に値するが、GAは相殺されており、その差を領域適応注意へ帰属できない。[勾配監査](gradient_audit.json)、[GANなしの固定評価](nogan_evaluation/results.json)。

教師あり損失も著者版と完全には同じでなく、比較全モデルに共通のマスク付き墨Dice/Jaccard等を使っている。したがって今回の名称は「公開コード版に基づくパイロット」とする。[先行の論文との照合](../ragan_paper_alignment_20261021/論文との照合.md)。

## 固定4画像を学習できるか

クリーン2枚と暗い接触汚れ2枚の固定256角・同一教師を用い、生成器だけで当てはめた。400更新では暫定教師F値約0.75だったが、乱数初期化から連続して学習する容量試験では **{last['total_G_updates']}更新で{last['provisional_training_f1']:.6f}** に達した。データ対応を学習する能力自体はある。これは同じ4訓練画像への当てはめであり、未知損傷・別資料への形状学習を証明しない。[容量試験](fixed_batch_capacity_fresh/completion.json)。

別の補助試行では、400更新重みだけを読みAdamを初期化し直した継続が全面白へ崩壊した。optimizerを保持しない継続と連続学習を同一視しない。この補助試行を学習量の因果比較には使わない。[記録](fixed_batch_capacity_extension/completion.json)。

## 次に優先すること

1. 損傷の細かな調整を一旦止め、十分な学習量・保存optimizer・固定閾値・課題別の検証を標準手順にする。10周は学習量依存性の確認であり、論文60周の再現でも収束保証でもない。
2. 公開コードの注意相殺と論文との差を整理し、公開版・修正候補・GANなしを混ぜずに扱う。現行の結果からRA-GAN本来の限界を主張しない。
3. 実損傷の同一ROIで墨芯・汚れ内の非文字・不明を専門家確認し、評価用の別本を固定する。文字被覆の増加と染みの再導入を同時に数える。
4. その評価を固定してから、広い損傷分布で学習する必要性と、GAN・注意機構を加える利点を検討する。

[全チェックポイントの評価](evaluation/results.json)、[U-Net同学習量対照](unet_evaluation/results.json)、[学習曲線](summary/学習量と誤黒化の推移.png)。
'''
    (OUT/'学習量と実装の再監査.md').write_text(note)
    dump(dest/'verification.json',dict(results_sha256={str(p.relative_to(OUT)):sha(p) for p in sources},
        all_four_conditions_complete=True,threshold_fixed=.5,final_step=2160,figures=figures,
        capacity_criterion_reached=cap['criterion_reached'],formal_accuracy=False))
    for link in re.findall(r'\]\(([^)]+)\)',note):
        assert (OUT/link.split('#')[0]).exists(),link
    print('Saved',OUT/'学習量と実装の再監査.md',flush=True)


if __name__=='__main__':main()
