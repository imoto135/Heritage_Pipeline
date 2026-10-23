"""All seeds, checkpoints and original-image panels for the architecture control."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_matched_supervised_generators import OUT,CONDITIONS
from audit_ragan_inference_context import cases,HARD
from evaluate_ragan_native_tiling import load_cases,measure,CONTACT
from compare_v5_aligned_fade_control import dump,sha,rgb,mask
from apply_ragan_real_regions import panel
import apply_ragan_real_regions as real
import evaluate_classical_partial_labels as partial

LABELS={c:('U-Net' if conf['kind']=='unet' else '公開G')+f" 初期化{conf['seed_index']+1}" for c,conf in CONDITIONS.items()}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def ink(path):return np.where(mask(path),0,255).astype(np.uint8)


def howe_path(c):
    if c['group']=='real':
        p=real.CLASSIC/'crops'/c['id']/'howe_fixed.png'
        if not p.exists():p=CONTACT/'other_book_transfer/masks'/c['id']/'howe_fixed.png'
        return p
    if c['group']=='traced_fade':return ROOT/'tmp_work/targeted_thin_fade_20261021/cases/alpha_0.70/howe_fixed.png'
    if c['group']=='synthetic' and c['family'] in ['contact_stain','clean']:
        patch=c['id'].rsplit('_'+c['family'],1)[0]
        return CONTACT/'evaluation/masks'/patch/c['family']/'howe_fixed.png'
    return None


def main():
    results={step:{c:json.loads((OUT/f'evaluation_{step:05d}'/c/'results.json').read_text()) for c in CONDITIONS} for step in (3024,2160)}
    primary=results[3024];allcases=load_cases();case_by={c['id']:c for c in allcases};figures=[]
    classical=[];classical_real=[];classical_hashes={}
    for c in allcases:
        hp=howe_path(c)
        if hp is None:continue
        assert hp.exists();pred=mask(hp)
        classical_hashes[str(hp.relative_to(ROOT))]=sha(hp)
        classical.append(dict(case=c['id'],group=c['group'],family=c['family'],**measure(pred,c)))
        if c['group']=='real':
            if c['subregions']:
                for r in c['subregions']:
                    x,y,w,h=r['xywh'];classical_real.append(dict(case=c['id'],region=r['region'],**partial.measure(pred[y:y+h,x:x+w],r['labels'],r['paths'])))
            else:classical_real.append(dict(case=c['id'],region='selected_spot_or_stain',nonink_false=int((pred&c['paper']).sum())))
    dump(OUT/'classical_reference.json',dict(rows=classical,real_regions=classical_real,prediction_sha256=classical_hashes,methods='existing fixed Howe only; no rerun or tuning'))
    for c in allcases:
        key=c['id'];x,y,w,h=c['roi'];image=c['page'][y:y+h,x:x+w]
        if c['group']=='real':references=[('原画像',image),('Howe固定',ink(howe_path(c)))]
        else:references=[('損傷前原本',c['source']),('同じ損傷入力',image)]
        items=[]
        for seed in (0,1):
            items.append(references[seed])
            for kind in ['unet','ragan']:
                name=f'{kind}_seed{seed}';items.append((LABELS[name],ink(OUT/'evaluation_03024'/name/'masks'/(key+'.png'))))
        if c['group']=='traced_fade':items.append(('Howe固定',ink(howe_path(c))))
        path=OUT/'figures'/(key+'_生成器比較.png');panel(path,key+' / 同じ教師あり条件・14周',items);figures.append(str(path.relative_to(OUT)))
    hard=next(c for c in cases() if c['patch']==HARD);x,y=hard['xy'];r=hard['record']
    source=rgb(ROOT/r['clean'])[y:y+256,x:x+256];image=rgb(ROOT/r['input'])[y:y+256,x:x+256]
    items=[]
    for seed,ref in [(0,('原本の固定256窓',source)),(1,('退色入力の固定256窓',image))]:
        items.append(ref)
        for kind in ['unet','ragan']:
            name=f'{kind}_seed{seed}';items.append((LABELS[name],ink(OUT/'evaluation_03024'/name/'training'/HARD/'canonical_local.png')))
    path=OUT/'figures/対象44点_固定256窓比較.png';panel(path,'学習側の対象44点 / 原位置の256窓',items);figures.append(str(path.relative_to(OUT)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained')
    for c,label in LABELS.items():
        rs=sorted(primary[c]['origin_rows'],key=lambda r:r['xy'][0]);ax.plot([r['xy'][0] for r in rs],[r['r2'] for r in rs],marker='o',label=label)
    ax.set(xticks=list(range(0,33,4)),yticks=[0,11,22,33,44],ylim=(-1,46));ax.grid(alpha=.2);ax.legend(prop=font,ncol=2)
    ax.set_title('同じ補助画像・切り出し位置への感度：学習側44点',fontproperties=font)
    ax.set_xlabel('横の開始位置 [px]',fontproperties=font);ax.set_ylabel('中心被覆（半径2）/44',fontproperties=font)
    path=OUT/'figures/生成器と切り出し位置.png';fig.savefig(path,dpi=180);plt.close(fig);figures.append(str(path.relative_to(OUT)))
    def row(step,c,case):return next(r for r in results[step][c]['rows'] if r['case']==case)
    def total(step,c,f,key):return sum(r[key] for r in results[step][c]['rows'] if r['family']==f)
    def train_sum(step,c):return sum(r['core_r2'] for r in results[step][c]['rows'] if r['group']=='train')
    def realval(step,c,key,region,field):return next(r[field] for r in results[step][c]['real_regions'] if r['case']==key and r['region']==region)
    learning=[];synthetic=[];realrows=[];offsets=[];secondary=[];tiling=[]
    for c,label in LABELS.items():
        a=primary[c];seen=[r for r in a['seen_rows'] if r['view']=='actual_batch4'];canonical=[r for r in a['training'] if r['view']=='canonical_local']
        learning.append([label,train_sum(3024,c),row(3024,c,HARD)['core_r2'],sum(r['r2'] for r in canonical),sum(r['r2'] for r in seen),next(r['r2'] for r in seen if r['patch']==HARD),
            f"{min(r['r2'] for r in a['origin_rows'])}–{max(r['r2'] for r in a['origin_rows'])}"])
        synthetic.append([label,total(3024,c,'clean','core_r2'),total(3024,c,'uniform_fade_070','core_r2'),row(3024,c,'heldout49')['core_r2'],
            total(3024,c,'contact_stain','core_r2'),total(3024,c,'contact_stain','paper_ink')])
        realrows.append([label,realval(3024,c,'D01','thin','ink_r2'),realval(3024,c,'D03','contact_upper','ink_r2'),
            *[realval(3024,c,k,'selected_spot_or_stain','nonink_false') for k in ['D04','T07','C01']]])
        secondary.append([label,train_sum(2160,c),row(2160,c,HARD)['core_r2'],row(2160,c,'heldout49')['core_r2'],total(2160,c,'contact_stain','core_r2'),total(2160,c,'contact_stain','paper_ink'),realval(2160,c,'D04','selected_spot_or_stain','nonink_false')])
        for m in ['tiled0','tiled16']:
            rs=[r for r in a['tiling_rows'] if r['method']==m]
            tiling.append([label,m,sum(r['core_r2'] for r in rs if r['group']=='train'),next(r['core_r2'] for r in rs if r['case']==HARD),
                next(r['core_r2'] for r in rs if r['case']=='heldout49'),sum(r['paper_ink'] for r in rs if r['family']=='contact_stain'),next(r['paper_ink'] for r in rs if r['case']=='D04')])
    def cv(key,reg,field):return next(r[field] for r in classical_real if r['case']==key and r['region']==reg)
    realrows.append(['Howe固定',cv('D01','thin','ink_r2'),cv('D03','contact_upper','ink_r2'),*[cv(k,'selected_spot_or_stain','nonink_false') for k in ['D04','T07','C01']]])
    howe_transfer=['Howe固定',sum(r['core_r2'] for r in classical if r['family']=='clean'),'—',next(r['core_r2'] for r in classical if r['case']=='heldout49'),
        sum(r['core_r2'] for r in classical if r['family']=='contact_stain'),sum(r['paper_ink'] for r in classical if r['family']=='contact_stain')]
    synthetic.append(howe_transfer)
    lines=[
        '# 同一教師あり条件での生成器比較',
        'U-NetとRA-GAN公開由来生成器（以下「公開G」）を、いずれも未学習状態・新規Adamから14周学習した。全4条件は追加学習の重みを引き継いでいない。全期間GANなし。各2初期化の小規模な比較であり、正式精度や統計的優劣ではない。',
        '## 揃えた条件と揃えていない点',
        '元の864入力・96パッチ・教師・損傷・5入力チャンネル・提示順・切り出し座標・損失・学習率3e-5・3024更新・閾値.5を同じseedの2モデルで一致させた。全14周を通して7例の強退色を173/864回提示し、提示毎に合法位置を変えた。確認済み墨芯と近傍紙の補助BCEは従来と同じ係数で両モデルへ加える。これは標準U-Netのネットワークを共通の試験用損失で評価する対照であり、一般的なU-Net論文の学習条件の再現ではない。',
        'U-Netの単一出力logitと公開Gの積型出力の等価logitを使い、同じ確率なら同じ損失になること・従来の修正RA損失との値と勾配の一致・不明画素の除外・白に飽和した誤りへのBCE勾配を確認した。公開モデルのforwardは変更しない。',
        'U-Netは学習可能4,370,713パラメータ、公開Gは26,132,662。正規化・縮小拡大・出力層なども異なるため、同じ容量や単一部品の比較ではない。同じ学習率・更新数は、各モデルの最適条件や収束を保証しない。14周主評価・10周補助評価を事前固定し、良い重みだけを事後選択しない。',
        '過去の「10周重みから4周追加」とは、初期重み、BCE修正の適用時期、提示頻度の履歴、GAN学習歴が異なる。今回のモデル間比較は揃っているが、過去との差を一つの原因へ帰属しない。[事前プロトコル](protocol.json)、[損失の一致確認](loss_checks.json)、[固定評価計画](evaluation_plan.json)。',
        '## 学習側の保持：最終14周',
        table(['条件','512/455','対象512/44','原位置256/455','実学習4枚組/455','実学習対象/44','9位置の最小–最大/44'],learning),
        '455点と44点は学習用の確認済み中心であり、ここでの保持は汎化を意味しない。実学習4枚組は各checkpoint以前に最後に使われた正確な入力窓・同伴画像を再構成した。原位置は旧epoch13の固定窓で、今回最後に学習した窓とは区別する。',
        '[対象44点の原画像付き512比較](figures/'+HARD+'_生成器比較.png)、[固定256窓](figures/対象44点_固定256窓比較.png)、[位置別の被覆](figures/生成器と切り出し位置.png)。',
        '## 別ページ・合成損傷',
        table(['条件','無損傷墨/5500','退色墨/5500','既知別ページ/49','合成汚れ墨/596','合成汚れ紙黒/18105'],synthetic),
        '合成評価の墨芯・紙は原本由来の代理で、専門家GTではない。既知49点も以前から開発に見ている部分注釈で、盲検テストではない。中心から2画素以内の被覆は線幅・連結・字形の正しさを保証しない。',
        '## 実画像',
        table(['条件','D01/225','D03上側/124','D04黒/5100','T07黒/4500','C01黒/10800'],realrows),
        'Howeは既存の固定予測マスクを同じ評価領域で再集計した。実汚れ内の黒画素数は、不可視の墨の真値まで判定した精度ではない。既知開発例・暫定の部分注釈という制約を維持する。',
        '## 固定格子の256推論',
        table(['条件','格子','学習/455','対象/44','既知別ページ/49','合成紙黒/18105','D04黒/5100'],tiling),
        '512を主評価とし、同じ256・stride128・Hannの2格子も全条件で保存した。有利な位置・格子を注釈から選ばない。',
        '## 10周時点：補助評価',
        table(['条件','512学習/455','対象/44','既知別ページ/49','合成汚れ墨/596','合成紙黒/18105','D04黒/5100'],secondary),
        '## 解釈の範囲',
        '公開GAの注意相殺は維持され、今回も領域適応注意やGAN固有の効果を検証していない。単一本の暫定教師・共通の試験用損失による生成器対照である。両方が失敗する場合には共有する教師・損失・入力条件が残り、片方が良い場合にも、どの部品が原因かや最適性能の順位はこの比較だけでは決まらない。',
        '## 原画像付き比較PNG',
        '\n'.join(f'- [{Path(f).name}]({f})' for f in figures),
    ]
    (OUT/'比較結果_教師あり生成器.md').write_text('\n\n'.join(lines)+'\n')
    dump(OUT/'combined_results.json',dict(primary=primary,secondary=results[2160],figures=figures,learning_rows=learning,synthetic_rows=synthetic,real_rows=realrows,
        secondary_rows=secondary,tiling_summary=tiling,formal_accuracy=False))
    shutil.copy2(Path(__file__),OUT/'source_snapshot'/Path(__file__).name)
    print('Reported4 fresh models,2 checkpoints,',len(figures),'PNGs',flush=True)


if __name__=='__main__':main()
