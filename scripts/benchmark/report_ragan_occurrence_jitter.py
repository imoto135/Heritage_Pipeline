"""Report the preregistered per-draw crop control, including failed outcomes."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_occurrence_jitter_control import OUT,PARENT,CONDITIONS
from audit_ragan_inference_context import OUT as CONTEXT,HARD,cases
from evaluate_ragan_native_tiling import load_cases
from compare_v5_aligned_fade_control import dump,sha,rgb,mask
from apply_ragan_real_regions import panel
import apply_ragan_real_regions as real

LABELS={f'{kind}_order{o}':f'{name} 順序{o+1}' for o in (0,1) for kind,name in [('inkpaper','周内固定'),('jitter','提示毎変更')]}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def ink(path):return np.where(mask(path),0,255).astype(np.uint8)


def main():
    groups={};train={};origins={};tiling=[];tilereal=[]
    for step in (864,432):
        olddir=PARENT/('evaluation' if step==864 else 'evaluation_00432')
        old=json.loads((olddir/'results.json').read_text())
        oldtrain=json.loads((olddir/'training_reviewed_results.json').read_text())
        groups[step]={k:[r for r in old[k] if r['condition'].startswith('inkpaper')] for k in ['real','synthetic','traced_fade']}
        train[step]=[dict(r,view='full512') for r in oldtrain['rows'] if r['condition'].startswith('inkpaper')]
        for order in (0,1):
            dest=OUT/f'evaluation_{step:05d}_order{order}'
            new=json.loads((dest/'results.json').read_text())
            traces=json.loads((dest/'training_and_offsets.json').read_text())
            for k in groups[step]:groups[step][k]+=new[k]
            train[step]+=traces['rows']
            origins[(step,f'jitter_order{order}')]=traces['origin_rows']
            if step==864:
                a=json.loads((dest/'tiling_results.json').read_text());tiling+=a['rows'];tilereal+=a['real_regions']
                oldtiles=json.loads((CONTEXT/f'tiling_order{order}.json').read_text())
                tiling+=[r for r in oldtiles['rows'] if r['condition']==f'inkpaper_order{order}']
                tilereal+=[r for r in oldtiles['real_regions'] if r['condition']==f'inkpaper_order{order}']
        if step==864:
            for order in (0,1):
                a=json.loads((CONTEXT/f'audit_order{order}.json').read_text())
                c=f'inkpaper_order{order}'
                origins[(step,c)]=[dict(r,xy=[int(r['view'].split('x')[-1]),0]) for r in a['rows'] if r['condition']==c and r['patch']==HARD and r['view'].startswith('origin_x')]
    figures=[]
    for c in load_cases():
        key=c['id'];x,y,w,h=c['roi'];original=c['source'];damaged=c['page'][y:y+h,x:x+w]
        if c['group']=='real':
            hp=real.CLASSIC/'crops'/key/'howe_fixed.png'
            if not hp.exists():hp=ROOT/'tmp_work/makura_contact_stain_train_control_20261022/other_book_transfer/masks'/key/'howe_fixed.png'
            first,second=('原画像',damaged),('Howe固定',ink(hp))
        else:first,second=('損傷前原本',original),('同じ損傷入力',damaged)
        items=[]
        for order,reference in [(0,first),(1,second)]:
            items.append(reference)
            if c['group']=='train':
                old=PARENT/'evaluation/training_reviewed_masks'/f'inkpaper_order{order}'/(key+'_targeted_fade.png')
                new=OUT/f'evaluation_00864_order{order}/training_masks/targeted_fade'/(key+'_512.png')
            elif c['group']=='traced_fade':
                old=PARENT/'evaluation/traced_fade'/f'inkpaper_order{order}.png'
                new=OUT/f'evaluation_00864_order{order}/traced_fade'/f'jitter_order{order}.png'
            else:
                old=PARENT/'evaluation'/c['group']/f'inkpaper_order{order}'/(key+'.png')
                new=OUT/f'evaluation_00864_order{order}'/c['group']/f'jitter_order{order}'/(key+'.png')
            items.extend([(f'周内固定 順序{order+1}',ink(old)),(f'提示毎変更 順序{order+1}',ink(new))])
        path=OUT/'figures'/(key+'_512比較.png');panel(path,key+' / 切り出しだけの対照',items);figures.append(str(path.relative_to(OUT)))
    c=next(c for c in cases() if c['patch']==HARD);x,y=c['xy'];row=c['record']
    source=rgb(ROOT/row['clean'])[y:y+256,x:x+256];damaged=rgb(ROOT/row['input'])[y:y+256,x:x+256]
    items=[]
    for order,reference in [(0,('原本の固定256窓',source)),(1,('退色入力の固定256窓',damaged))]:
        items.extend([reference,(f'周内固定 順序{order+1}',ink(PARENT/'evaluation/training_native_masks'/f'inkpaper_order{order}'/(HARD+'_single_crop1.png'))),
            (f'提示毎変更 順序{order+1}',ink(OUT/f'evaluation_00864_order{order}/training_masks/targeted_fade'/(HARD+'_canonical256_local.png')))])
    path=OUT/'figures/残存44点_原位置の256窓比較.png';panel(path,'同じ原位置256窓 / 前回の保持も失うか',items);figures.append(str(path.relative_to(OUT)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained')
    for c,label in LABELS.items():
        xs=sorted(origins[(864,c)],key=lambda r:r['xy'][0]);ax.plot([r['xy'][0] for r in xs],[r['r2'] for r in xs],marker='o',label=label)
    ax.axhline(36,color='grey',linestyle='--',alpha=.7)
    ax.set(xticks=list(range(0,33,4)),yticks=[0,11,22,33,36,44],ylim=(-1,46))
    ax.set_title('提示ごとの位置変更：同じ44点・同じ補助画像',fontproperties=font)
    ax.set_xlabel('横の切り出し開始位置 [px]',fontproperties=font);ax.set_ylabel('中心被覆（半径2）/44',fontproperties=font)
    ax.legend(prop=font,ncol=2);ax.grid(alpha=.2)
    path=OUT/'figures/提示毎位置変更_位置別被覆.png';fig.savefig(path,dpi=180);plt.close(fig);figures.append(str(path.relative_to(OUT)))
    def fit(step,c,patch=None,view='full512',family='targeted_fade'):
        return sum(r['r2'] for r in train[step] if r['condition']==c and r['view']==view and r['family']==family and (patch is None or r['patch']==patch))
    def syn(step,c,f,key):return sum(r[key] for r in groups[step]['synthetic'] if r['condition']==c and r['family']==f)
    def thin(step,c):return next(r['r2'] for r in groups[step]['traced_fade'] if r['condition']==c)
    def rv(step,c,case,region,key):return next(r[key] for r in groups[step]['real'] if r['condition']==c and r['case']==case and r['region']==region)
    gates={}
    for c in CONDITIONS:
        gates[c]=dict(fixed512=fit(864,c,HARD),worst9_origins=min(r['r2'] for r in origins[(864,c)]),best9_origins=max(r['r2'] for r in origins[(864,c)]))
        gates[c]['passed']=gates[c]['fixed512']>=36 and gates[c]['worst9_origins']>=36
    gate=all(r['passed'] for r in gates.values())
    summaryrows=[];collateral=[]
    for c,label in LABELS.items():
        ors=origins[(864,c)]
        summaryrows.append([label,fit(864,c),fit(864,c,HARD),f"{min(r['r2'] for r in ors)}–{max(r['r2'] for r in ors)}",thin(864,c),syn(864,c,'contact_stain','core_r2'),syn(864,c,'contact_stain','paper_ink')])
        collateral.append([label,syn(864,c,'clean','core_r2'),syn(864,c,'uniform_fade_070','core_r2'),rv(864,c,'D01','thin','ink_r2'),rv(864,c,'D03','contact_upper','ink_r2'),
            *[rv(864,c,k,'selected_spot_or_stain','nonink_false') for k in ['D04','T07','C01']]])
    tilerows=[]
    for c,label in LABELS.items():
        for m in ['full512_cached','tiled0','tiled16']:
            rows=[r for r in tiling if r['condition']==c and r['method']==m]
            tilerows.append([label,m,sum(r['core_r2'] for r in rows if r['group']=='train'),next(r['core_r2'] for r in rows if r['case']==HARD),
                next(r['core_r2'] for r in rows if r['case']=='heldout49'),sum(r['paper_ink'] for r in rows if r['family']=='contact_stain'),next(r['paper_ink'] for r in rows if r['case']=='D04')])
    seen=[r for o in (0,1) for r in json.loads((OUT/f'jitter_seen_order{o}.json').read_text())['rows']]
    seenrows=[]
    for c in CONDITIONS:
        for view in ['actual_batch4','single_crop1']:
            rs=[r for r in seen if r['condition']==c and r['view']==view]
            seenrows.append([LABELS[c],view,sum(r['r2'] for r in rs),next(r['r2'] for r in rs if r['patch']==HARD)])
    lines=[
        '# 提示ごとの切り出し位置変更：対照学習の判断',
        f'事前の進行基準は **{"通過" if gate else "不通過"}**。位置変更だけでは、対象細線を保持しつつ位置依存を解消できなかった。今回の重みは採用せず、この生成器・損失構成に対する損傷や位置の追加調整を続ける段階は一旦区切る。',
        '## 固定した条件',
        '前回の墨芯＋近傍紙条件を基準に、同じ10周生成器・Adam状態から864更新（4周）を追加した。GANなし。入力RGBファイル・教師・損傷強度・提示配列・提示回数・補助損失・学習率・更新数・閾値0.5を固定し、既存7例の強退色.80の切り出し位置だけを提示ごとに変えた。周内固定の2順序は前回の凍結出力を再利用した。同じ初期Gへの2つの提示順であり、独立初期化反復ではない。',
        '全墨芯と同じ近傍紙を256窓に入れる合法範囲から座標を事前生成した。約100提示に対して各例97〜100の異なる位置となり、全692対象提示の墨芯・紙画素露出数は基準と完全一致した。それ以外の2764入力提示は元の座標のまま。窓内の他の文字・紙、再計算する補助画像、全体損失の対象画素は切り出しに伴って変わるため、縮小位相だけを変えた実験ではない。',
        '[事前プロトコル](protocol.json)、[切り出し多様性と露出監査](crop_exposure_audit.json)、[補助マスク一致](mask_identity_checks.json)、[評価計画](evaluation_plan.json)。',
        '## 主評価：最終864更新、全条件を報告',
        table(['条件','学習512/455','対象512/44','9開始位置の最小–最大/44','別ページ/49','合成汚れ墨/596','合成汚れ紙黒/18105'],summaryrows),
        '対象44点の512被覆は5/0→13/0点。前回の原位置256窓では44/44→12/0点に低下した。開始位置による変動幅は小さくなったが、特に順序2は全位置で墨を失っており、これを位置頑健性の改善とは扱わない。学習455点や別ページ49点の結果も順序ごとに増減し、一貫した保持改善はない。',
        '事前に、この試験で明確な回復と呼ぶ目安を「512の対象44点と、9開始位置の最悪値がともに36点以上、両順序で達成」とした。80%を超える中心被覆の運用上の目安であり、正式精度や字形の正しさを保証する基準ではない。この基準を結果を見て緩めない。',
        '[原本付き512比較](figures/'+HARD+'_512比較.png)、[原位置256窓比較](figures/残存44点_原位置の256窓比較.png)、[位置別の被覆](figures/提示毎位置変更_位置別被覆.png)。',
        '## 他の文字と実汚れ',
        table(['条件','無損傷墨/5500','退色墨/5500','D01/225','D03上側/124','D04黒/5100','T07黒/4500','C01黒/10800'],collateral),
        '合成汚れ内の紙は両順序0画素まで抑えたが、別ページ退色は49/47→43/42点に低下した。D04は2628/2250→2165/2152画素へ減るが、Howe固定0には届かない。D01は206/198→198/205、別本C01は8367/8222→7257/8466で順序による増減がある。紙を抑えることと薄線保持を両立できていない。黒画素数は汚れ下の不可視の墨まで判定した正解ではない。',
        '## 固定格子の推論',
        table(['条件','推論','学習/455','対象/44','別ページ/49','合成紙黒/18105','D04黒/5100'],tilerows),
        '分割は前回と同じ256・stride128・Hann、原点0と副格子16を全件測定した。文字の位置に合わせて有利な格子を選ばない。分割で良い数値だけを採用することもせず、512と併せて判断する。',
        '## 432更新時点も選別せず報告',
        table(['条件','学習512/455','対象512/44','別ページ/49','合成紙黒/18105'],[[label,fit(432,c),fit(432,c,HARD),thin(432,c),syn(432,c,'contact_stain','paper_ink')] for c,label in LABELS.items()]),
        '主評価は事前指定の864更新である。432更新の別ページが良くても、最終結果と入れ替えない。',
        '## 追加診断：今回実際に学習した最後の窓',
        '元の位置だけを試したために失敗した可能性を確認するため、主評価の不通過後に追加した診断である。各7例が最後に提示された正確な座標・4枚組を凍結計画から再構成し、同じ最終重みで再入力した。採用基準や主評価は変更しない。',
        table(['条件','入力','学習455点中','対象44点中'],seenrows),
        '対象44点は実際に学習した最後の4枚組でも14/0点で、同じ窓を単独入力しても14/0点だった。学習側の保持不足も残るため、原位置や512への切り替えだけの失敗ではない。[順序1](jitter_seen_order0.json)、[順序2](jitter_seen_order1.json)。',
        '## 研究上の判断',
        'この対照は「位置が主因なら、提示ごとに位置を変えることで保持が安定する」という説明を支持しなかった。元のモデルに位置感度があるという観測は残るが、それだけを直せば十分とは言えない。既存の固定4画像への適合試験では細線を表現できたため、生成器の表現不能やRA-GAN一般の限界が証明されたわけでもない。今回の短い追加学習だけで、あらゆるデータ拡張が無効とも結論しない。',
        '一方で、入力到達・教師・露出・BCE勾配・推論範囲・位置変更を順に確認しても、目的の薄線保持と実汚れ抑制が安定して両立していない。現行構成を損傷の微調整で救い続ける方針は見合わせる。次の比較は損傷データを凍結し、標準的な教師あり生成器と現在の公開由来生成器を、同じ入力・十分な学習量・同じ評価で比べる構成へ戻す。GAN固有の寄与は、教師ありで保持が成立した後に一因子で検証する。',
        '公開GAの注意相殺が残り、論文本来の領域適応注意を評価した結果ではない。単一本・暫定教師・既知開発例による診断であり、専門家GTの正式な精度比較でもない。今回をRA-GAN原論文の否定やHowe優位の一般証明には使わない。',
        '## 比較PNG',
        '\n'.join(f'- [{Path(f).name}]({f})' for f in figures),
    ]
    (OUT/'研究判断_提示毎位置変更の対照.md').write_text('\n\n'.join(lines)+'\n')
    dump(OUT/'combined_results.json',dict(primary=groups[864],secondary=groups[432],training_primary=train[864],training_secondary=train[432],
        origins_primary=[dict(r,condition=c) for (step,c),rs in origins.items() if step==864 for r in rs],tiling=tiling,tiling_real_regions=tilereal,seen_last_training_crops=seen,
        progression_gate=dict(conditions=gates,passed=gate),figures=figures,formal_accuracy=False))
    shutil.copy2(Path(__file__),OUT/'source_snapshot'/Path(__file__).name)
    print('Gate',gate,gates,'PNGs',len(figures),flush=True)


if __name__=='__main__':main()
