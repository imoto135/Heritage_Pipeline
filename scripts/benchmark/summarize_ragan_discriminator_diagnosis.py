"""Report paired critic rankings and feedback without claiming restoration gains."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp_work/ragan_discriminator_diagnosis_20261022/matplotlib_config'))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from diagnose_ragan_discriminator import OUT,CONDITIONS,VARIANTS
from compare_v5_aligned_fade_control import sha,dump


def main():
    dest=OUT/'summary';dest.mkdir(exist_ok=True)
    completed={c:json.loads((OUT/c/'completion.json').read_text()) for c in CONDITIONS}
    initial=json.loads((OUT/'reference_lr1e6/history.json').read_text())[0]
    rows=[('initial',initial)]+[(c,d['final']) for c,d in completed.items()]
    labels={'initial':'既存の学習済みD', 'reference_lr1e6':'元の損失・lr1e-6で追加800更新',
        'reference_lr1e4':'元の損失・lr1e-4', 'bce_lr1e4':'BCE・lr1e-4',
        'logit_hinge_lr1e4':'Sigmoid除去＋要素別hinge・lr1e-4',
        'bce_balanced_corruptions_lr1e4':'BCE＋4種類の負例'}
    keys=['generator_soft','added_stain','erased_overlap_ink','same_area_relocation']
    table=[];matrix=[]
    for key,d in rows:
        pairs={r['variant']:r for r in d['val']['pairs']}
        wins=[int(round(pairs[k]['paired_real_higher']*8)) for k in keys]
        table.append('| '+labels[key]+' | '+' | '.join(f'{x}/8' for x in wins)+' |')
        matrix.append(wins)
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    im=ax.imshow(np.asarray(matrix)/8,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
    ax.set_xticks(range(4),['Native G output','Add stain','Erase ink','Same-area relocation'])
    ax.set_yticks(range(len(rows)),['Original trained D','Reference +800 / lr1e-6','Reference +800 / lr1e-4',
        'BCE +800 / lr1e-4','Logit hinge +800 / lr1e-4','BCE + balanced negatives'])
    for y,line in enumerate(matrix):
        for x,n in enumerate(line):ax.text(x,y,f'{n}/8',ha='center',va='center',fontsize=12,color='black')
    ax.set_title('Paired ranking: provisional teacher preferred to altered output\n8 heldout page groups; discriminator-only diagnosis')
    fig.colorbar(im,ax=ax,label='Teacher preferred fraction');fig.savefig(dest/'識別器の改変別判定.png',dpi=160);plt.close(fig)
    index=[r for r in json.loads((OUT/'index.json').read_text())['rows'] if r['split']=='val']
    initial_scores=initial['val']['scores']
    balanced_scores=completed['bce_balanced_corruptions_lr1e4']['final']['val']['scores']
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
    titlefont=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
    variant_names={'real':'暫定教師','generator_soft':'実際の生成器出力（連続値）',
        'generator_binary':'生成器出力の二値版','added_stain':'教師に汚れを追加',
        'erased_overlap_ink':'教師の墨を消去','same_area_relocation':'黒画素数を保ち墨を移動',
        'softened_real':'教師をぼかす'}
    figures=[]
    for i,row in enumerate(index):
        a=np.load(OUT/row['cache']);valid=a['valid']
        tiles=[('原本',a['source'],None),('固定した損傷入力',a['image'],None)]
        for key in VARIANTS:
            tile=np.rint(np.where(valid,a['p_'+key]*255,170)).astype(np.uint8)
            note=None
            if key!='real':
                d0=initial_scores['real'][i]-initial_scores[key][i]
                d1=balanced_scores['real'][i]-balanced_scores[key][i]
                note=(f'既存D Δ={d0:+.2e}',f'負例追加D Δ={d1:+.2e}')
            tiles.append((variant_names[key],tile,note))
        gap=12;head=80;tilew=tileh=256;labelh=62
        canvas=Image.new('RGB',(3*(tilew+gap)+gap,head+3*(tileh+labelh+gap)+gap),'#faf9f5');draw=ImageDraw.Draw(canvas)
        draw.text((12,8),row['patch']+' / 原本と識別器の診断画像',font=titlefont,fill='#203442')
        draw.text((12,34),'Δ=D(教師)−D(改変)。正なら教師を高評価。灰色は教師未確定で評価対象外。',font=font,fill='#203442')
        draw.text((12,55),'生成器は固定。負例追加Dは診断専用で、二値化精度の改善は未検証。',font=font,fill='#203442')
        for j,(name,tile,note) in enumerate(tiles):
            x=gap+(j%3)*(tilew+gap);y=head+(j//3)*(tileh+labelh+gap)
            draw.text((x,y),name,font=font,fill='#203442')
            if note:
                draw.text((x,y+20),note[0],font=font,fill='#203442');draw.text((x,y+39),note[1],font=font,fill='#203442')
            if tile.ndim==2:tile=np.repeat(tile[:,:,None],3,2)
            canvas.paste(Image.fromarray(tile.astype(np.uint8)),(x,y+labelh))
        path=dest/'figures'/(row['patch']+'_原本と識別器判定.png');path.parent.mkdir(exist_ok=True)
        canvas.save(path);figures.append(dict(patch=row['patch'],path=str(path.relative_to(OUT)),sha256=sha(path)))
    native=json.loads((OUT/'native_generator_feedback.json').read_text())
    gradient_table=[]
    for condition in ['initial']+list(CONDITIONS):
        group=[r for r in native['rows'] if r['condition']==condition]
        vals=[r['weighted_GAN_to_supervised_parameter_grad_ratio'] for r in group]
        counts=[]
        for error in ('false_positive_ink','false_negative_ink'):
            q=[e for r in group for e in r['errors'] if e['type']==error]
            n=sum(e['n'] for e in q);hit=sum(e['desired_gradient_pixels'] for e in q)
            counts.append(f'{hit}/{n} ({hit/max(n,1):.1%})')
        gradient_table.append('| '+labels[condition]+f' | {min(vals):.2e}〜{max(vals):.2e} | '+' | '.join(counts)+' |')
    gap=next(r for r in initial['val']['pairs'] if r['variant']=='added_stain')['mean_real_minus_variant']
    rf=json.loads((OUT/'receptive_field_audit.json').read_text())
    note=f'''# 識別器は汚れと文字欠落を見抜き、修正できるか

2026-10-22。**現在の識別器が汚れを全く見抜けない、という説明は正確ではない。大きな黒い汚れは8/8例で教師より低く評価するが、文字を消した画像も高く評価することが多い。** 形状に関する望ましい評価と生成器への修正信号が揃っていない。

## 診断の条件

既存の学習24ページ群・検証8ページ群から1パッチずつ固定し、原本由来の暫定教師と強度0.65の暗い汚れ入力を用意した。生成器は10周モデルを固定し、更新していない。入力256角、生成器の出力は1画像ずつ同じ手順で保存した。教師未確定画素は実・偽とも0.5で同じように隠した。ただし、そのマスク境界自体が識別の手掛かりになる可能性は残る。

診断用の誤りは、(a)教師の有効紙領域に汚れを追加、(b)汚れと重なる有効墨を消す、(c)黒画素数を保って墨を別位置へ移す、である。これらは暫定教師に対する人工的な改変で、実際の隠れた文字の正解ではない。元の4条件では、これらをDの学習負例に入れず、生成器の連続値出力だけを負例に使った。

全条件は同じ学習済みD重みから開始し、Adamは新規作成、Dだけを800更新した。損失と学習率の対照であり、論文再現や生成器との同時学習の比較ではない。最後の「4種類の負例」は先の結果を見て固定した探索的な能力試験で、生成器出力・汚れ追加・墨消去・墨移動を1個ずつ各batchに混ぜた。BCE条件から変えたのは負例の構成だけである。

## どの誤りを低く評価するか

同じ入力に対し、D(教師)>D(改変)となった検証例数。8例は8ページ群。分類閾値0.5での正解数ではなく、**同一入力内の順位**である。全体AUC・点差・全途中結果もJSONへ残した。

| D条件 | 生成器の出力 | 汚れ追加 | 文字を消去 | 黒量を保つ移動 |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

既存Dの汚れ追加に対する平均点差は約{gap:.2e}。汚れを認識する順位はあっても点差が小さい。一方で、墨消去を低く評価できたのは2/8例だった。学習率を上げるだけでは「墨を残す方向」に偏り、汚れ追加を全例で教師より高く評価する条件になった。BCEへ変えた条件は学習画像の実/生成器出力をAUC=1.0まで分けたが、検証画像の黒量を保つ移動は1/8例しか低く評価しなかった。単なる学習速度の問題だけでは説明できない。

4種類の負例を教えると、汚れ・文字消去・黒量保存の移動は各8/8例で低く評価した。**面積だけでは区別できない改変も学べる余地はある。** ただし、人工負例の作り方や未確定領域の境界を識別している可能性があり、文字の意味・筆順・正しい字形を学んだ証拠ではない。実際の生成器出力を低く評価できたのは1/8例で、この負例設計をそのままGANへ採用できるとは言えない。

[判定の比較図](summary/識別器の改変別判定.png)、[全8例の原本比較の一覧](summary/verification.json)。

## 実際の生成器の誤りへ返す勾配

同じ8検証画像の固定生成器出力について、暫定有効紙を黒くした画素と有効墨を消した画素を別々に数えた。GAN損失の重みは0.1。各画素で勾配降下が正しい方向へ出力を動かす符号かを診断した。これは生成器を更新した結果ではなく、出力・パラメータでの局所微分である。

| D条件 | GAN/教師ありの生成器勾配ノルム比（2batch） | 誤黒化を白へ戻す符号 | 消した墨を戻す符号 |
|---|---:|---:|---:|
{chr(10).join(gradient_table)}

符号の割合だけでは勾配の強さ・画素間の結合・実際の更新後の変化を説明できない。高い割合や大きい勾配ノルムだけを採用基準にしない。識別器の分類順位が良くても、出力の現在位置で文字を戻す微分が得られるとは限らない。[全勾配診断](native_generator_feedback.json)。

## 局所形状の判定について分かった実装上の条件

公開Dは理論受容野{rf['theoretical_rf']}画素、学習入力256角では出力が1画像につき1個だった。512角なら25個になるが、現在の学習は256角である。したがって学習中は、小さな文字・汚れごとに別の判定を返す構造になっていない。これだけで局所欠陥を見抜けないとは言えないが、論文の局所パッチ判定という説明と実際の出力サイズを区別する。[受容野と実出力の監査](receptive_field_audit.json)。

## 現時点の説明と次の条件

今回の結果は、現在のDが「文字の形を守り、汚れだけを消す」という一貫した評価を学べていないことを示す。黒領域の量、連続値/二値値の違い、RGBと二値像の対応など、文字形状以外の手掛かりへ依存している可能性がある。ただし、どの手掛かりが主因かをこの試験だけで断定しない。

学習率やGAN係数を単に増やす修正は採用しない。候補Dには、(1)汚れ追加と墨消去を両方低く評価する、(2)黒量を保つ改変も識別する、(3)実際の生成器誤りへ適切な修正信号を返す、を課す。その後に生成器と同時学習し、GANなしより実染みの混入を減らしつつ墨を保持できるかを測る必要がある。**今回のD単独診断から二値化精度の改善を主張しない。**

現行教師は専門家確認済みGTではなく、評価画像も開発に使ってきた画像である。別資料の独立評価と、接触汚れ内の墨・非文字・判定不能の確認は引き続き必要。[元の学習条件監査](../ragan_convergence_audit_20261022/学習量と実装の再監査.md)。
'''
    (OUT/'識別器と文字形状の診断.md').write_text(note)
    dump(dest/'verification.json',dict(figures=figures,conditions=list(CONDITIONS),
        no_generator_updates=True,source_teacher_provisional=True,formal_accuracy=False))
    print('Saved critic diagnostic report and',len(figures),'annotated comparisons',flush=True)


if __name__=='__main__':main()
