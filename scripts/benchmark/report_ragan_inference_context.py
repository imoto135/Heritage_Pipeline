"""Original-size PNG panels and complete readout of frozen inference controls."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
from audit_ragan_inference_context import OUT, PREVIOUS, CONDITIONS, HARD, VIEWS, cases
from evaluate_ragan_native_tiling import load_cases, METHODS
from compare_v5_aligned_fade_control import dump, rgb, mask, sha
from apply_ragan_real_regions import panel
import apply_ragan_real_regions as real
import run_ragan_full_continuation as full

LABELS = {f'{kind}_order{o}': f'{name} 順序{o+1}' for o in (0, 1) for kind, name in [('baseline','補助項なし'), ('inkmean','墨芯のみ'), ('inkpaper','墨芯＋近傍紙')]}
SHORT = {'full512_cached':'512','tiled0':'256格子0','tiled16':'256格子16'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+'|'.join(['---']*len(headers))+'|'] + ['| '+' | '.join(map(str,row))+' |' for row in rows])


def monochrome(path):
    return np.where(mask(path), 0, 255).astype(np.uint8)


def main():
    audits = [json.loads((OUT/f'audit_order{o}.json').read_text()) for o in (0,1)]
    tiles = [json.loads((OUT/f'tiling_order{o}.json').read_text()) for o in (0,1)]
    ar = [r for a in audits for r in a['rows']]
    tr = [r for a in tiles for r in a['rows']]
    rr = [r for a in tiles for r in a['real_regions']]
    figures = []
    case = next(c for c in cases() if c['patch']==HARD)
    x,y = case['xy']
    row = case['record']
    source = rgb(ROOT/row['clean'])[y:y+256,x:x+256]
    damaged = rgb(ROOT/row['input'])[y:y+256,x:x+256]
    teacher = full.gray(ROOT/row['labels']['target'])[y:y+256,x:x+256].copy()
    teacher[full.gray(ROOT/row['labels']['valid'])[y:y+256,x:x+256]==0] = 128
    views = [('crop_local','256局所補助'), ('crop_full_aux','256広域補助'), ('full_global','512通常'),
             ('full_patch_local_aux','512窓内補助変更'), ('full_global_theta_crop','512ゲート変更'), ('phase_aligned','576格子合わせ')]
    for condition in CONDITIONS:
        if condition.startswith('baseline'):
            continue
        items = [('同じ原本の256窓',source),('同じ退色.80の256窓',damaged),('暫定教師・不明は灰',teacher)]
        for view,name in views:
            score = next(r['r2'] for r in ar if r['condition']==condition and r['patch']==HARD and r['view']==view)
            items.append((f'{name} {score}/44',monochrome(OUT/'masks'/condition/HARD/(view+'.png'))))
        path = OUT/'figures'/('残存44点_要因比較_'+condition+'.png')
        panel(path,LABELS[condition]+' / 補助・範囲・格子',items)
        figures.append(str(path.relative_to(OUT)))
    allcases = load_cases()
    for c in allcases:
        key = c['id']
        x0,y0,w,h = c['roi']
        inp = c['page'][y0:y0+h,x0:x0+w]
        overlay = c['source'].copy()
        overlay[c['paper']] = np.rint(overlay[c['paper']]*.4 + np.array([0,190,90])*.6).astype(np.uint8)
        overlay[c['core']] = [0,105,245]
        if c['group']=='real':
            hp = real.CLASSIC/'crops'/key/'howe_fixed.png'
            if not hp.exists():
                hp = ROOT/'tmp_work/makura_contact_stain_train_control_20261022/other_book_transfer/masks'/key/'howe_fixed.png'
            top = [('原画像',inp),('Howe固定',monochrome(hp)),('暫定評価領域：青墨・緑紙',overlay)]
        else:
            top = [('損傷前原本',c['source']),('同じ損傷入力',inp),('原本由来の墨・紙代理',overlay)]
        for order in (0,1):
            items = list(top)
            for kind,label in [('baseline','補助なし'),('inkpaper','墨芯＋紙')]:
                condition = f'{kind}_order{order}'
                for method in ['full512_cached',*METHODS]:
                    items.append((label+' '+SHORT[method],monochrome(OUT/'tiling/masks'/condition/method/(key+'.png'))))
            path = OUT/'figures'/f'{key}_分割推論比較_順序{order+1}.png'
            panel(path,key+f' / 固定格子の分割・順序{order+1}',items)
            figures.append(str(path.relative_to(OUT)))
    # A standalone research plot supplements the original-image comparison PNGs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig, ax = plt.subplots(figsize=(9,4.7),layout='constrained')
    for condition in CONDITIONS:
        if condition.startswith('baseline'):
            continue
        xs,ys = [],[]
        for origin in range(0,33,4):
            r = next(r for r in ar if r['condition']==condition and r['patch']==HARD and r['view']==f'origin_x{origin:02d}')
            xs.append(origin);ys.append(r['r2'])
        ax.plot(xs,ys,marker='o',label=LABELS[condition])
    ax.axvline(12,color='grey',linestyle='--',alpha=.6)
    ax.set(xticks=list(range(0,33,4)),yticks=[0,11,22,33,44],ylim=(-1,46))
    ax.set_title('同じ広域補助画像から切り出す位置を変更：学習側44点',fontproperties=font)
    ax.set_xlabel('横方向の開始位置 [px]（最終学習周は12）',fontproperties=font)
    ax.set_ylabel('2px以内に墨がある中心数 / 44',fontproperties=font)
    ax.legend(prop=font,ncol=2);ax.grid(alpha=.2)
    path = OUT/'figures/切り出し位置と44点被覆.png'
    fig.savefig(path,dpi=180);plt.close(fig)
    figures.append(str(path.relative_to(OUT)))

    def hard(c,view):
        return next(r['r2'] for r in ar if r['condition']==c and r['patch']==HARD and r['view']==view)
    def group(c,m,f,key):
        return sum(r[key] for r in tr if r['condition']==c and r['method']==m and r['family']==f)
    def one(c,m,case,key):
        return next(r[key] for r in tr if r['condition']==c and r['method']==m and r['case']==case)
    def realval(c,m,case,region,key):
        return next(r[key] for r in rr if r['condition']==c and r['method']==m and r['case']==case and r['region']==region)
    tileconditions = json.loads((OUT/'tiling_protocol.json').read_text())['conditions']
    mainrows,collateral = [],[]
    for c in tileconditions:
        for m in ['full512_cached',*METHODS]:
            label = LABELS[c]+' '+SHORT[m]
            mainrows.append([label,group(c,m,'training_targeted_fade','core_r2'),one(c,m,HARD,'core_r2'),one(c,m,'heldout49','core_r2'),
                group(c,m,'contact_stain','core_r2'),group(c,m,'contact_stain','paper_ink')])
            collateral.append([label,group(c,m,'clean','core_r2'),group(c,m,'uniform_fade_070','core_r2'),
                realval(c,m,'D01','thin','ink_r2'),realval(c,m,'D03','contact_upper','ink_r2'),
                one(c,m,'D04','paper_ink'),one(c,m,'T07','paper_ink'),one(c,m,'C01','paper_ink')])
    seamrows = []
    for c in tileconditions:
        if c.startswith('baseline'):
            continue
        for m in METHODS:
            for key in [HARD,'heldout49',*[a['id'] for a in allcases if a['family']=='contact_stain']]:
                r = next(r for r in tr if r['condition']==c and r['method']==m and r['case']==key)
                seamrows.append([LABELS[c]+' '+SHORT[m],key,r['core_disagree'],r['core_n'],f"{r['disagree_n']}/{r['overlap_n']}",
                    f"{r['seam_paper_ink']}/{r['seam_paper_n']}",f"{r['other_paper_ink']}/{r['other_paper_n']}"])
    hardrows = [[view,*[hard(c,view) for c in CONDITIONS]] for view in [*VIEWS,'full_global_theta_crop','crop_full_aux_theta_full','phase_original','phase_aligned']]
    positions = json.loads((OUT/'crop_position_audit.json').read_text())
    replay = [r for a in audits for r in a['cached_replay']]
    maxflip = max(r['changed_pixels'] for r in replay)
    r2diff = sum(r['new_trace']['r2'] != r['cached_trace']['r2'] for r in replay)
    assert r2diff == 0
    report = [
        '# 推論範囲・補助チャンネル・切り出し位置の診断',
        '今回の固定重みでは、残存細線44点の欠落は補助画像の作成範囲やGGCの平均係数だけでは説明できず、切り出し位置・縮小格子・文脈の組合せに強い感度がある。単純な256画素分割では回復せず、採用しない。次の学習対照では、同じ周で同じ切り出しを繰り返す構成を見直すのが優先となる。',
        '新規学習なし。前回の補助項なし・墨芯のみ・墨芯＋近傍紙を各2順序、計6重みで診断した。入力RGB・教師・重み・閾値0.5は固定。分割推論は事前指定した補助項なしと墨芯＋紙の4重みで行い、7学習画像・既知49点・6合成例・5実画像、計19画像を全て報告する。2順序は共通初期Gへの感度確認で、独立初期化ではない。公開GAの注意相殺が残る試験生成器であり、RA-GAN論文の再現評価ではない。',
        '## 44点の要因対照',
        table(['介入',*[LABELS[c] for c in CONDITIONS]],hardrows),
        '`crop_local`は256窓から補助を再計算、`crop_full_otsu/sobel/aux`は同じ256 RGBに512由来の該当チャンネルを渡す。`full_patch_local_*`は512入力のうち同じ256窓内の補助だけを差し替え、窓外は固定する。`theta`対照は4つのGGC平均係数を反対側の入力で観測した値に固定し、5チャンネル自体は変えない。いずれも原因診断専用で、採用する推論法ではない。',
        '墨芯＋紙では、256の補助チャンネルを512由来に替えても44/44点を保つ。逆に512窓内を局所補助へ替えるだけでは回復しない。GGC平均の相互置換でも256は44/44、512は5/0点のまま。したがって、この44点については補助のスケール差や平均係数を単独の主因とはできない。他の6学習パッチについても全8条件を保存しており、他の画像で補助の影響がないと一般化しない。',
        '## 切り出し位置・格子への感度',
        '[原本を並べた要因比較・墨芯＋紙順序1](figures/残存44点_要因比較_inkpaper_order0.png)、[順序2](figures/残存44点_要因比較_inkpaper_order1.png)、[開始位置ごとの被覆図](figures/切り出し位置と44点被覆.png)。',
        '512で作った補助画像を固定して256窓の横開始位置だけを0,4,...32へ動かすと、墨芯＋紙の順序1はx=0で4点、x=12で44点、x=16で38点、x=32で0点。順序2はそれぞれ0、44、31、0点だった。RGB・補助の共通部分は同じだが、入力範囲・畳み込みの境界・縮小格子上の位置が変わる。この比較だけを厳密な平行移動同変性の検証とはしない。',
        'さらに、元の512画像を同じ576角キャンバスへ反射paddingし、4つのGGC係数を元の512値へ固定した。通常配置と、対象窓の開始位置を32の倍数へ合わせる配置を比べると、墨芯＋紙は0→37点、0→28点に増えた。ゲート平均以外の配置依存が大きいことを支持する。ただし遠方paddingの配置も変わり、文脈・境界効果と縮小位相の寄与を完全には分離していない。',
        '学習側の切り出しを再集計すると、対象44点の強退色画像はepoch10/11/12/13でx=1/20/16/12、y=0だった。同じepoch内の約25提示は全て同じ座標である。周ごとのjitterは既にあるため「jitterがない」は誤りであり、問題候補は提示ごとの位置多様性と位置間での保持である。[提示位置の監査](crop_position_audit.json)。',
        '## 固定格子の256分割推論',
        '主条件は256画素、stride128、Hann重みの下限.05、補助画像は各タイルで再計算、batch1。ページ原点0を起点とし、端の最終タイルも固定する。副条件は内部格子を16画素ずらす。格子は文字・注釈の座標を一切使わず、両格子を全件報告して良い方だけを選ばない。512欄は前回保存済みの同じ重みの出力である。',
        table(['条件・推論','学習/455','残存/44','別ページ/49','合成汚れ墨/596','合成汚れ紙黒/18105'],mainrows),
        '墨芯＋紙の主格子では残存44点は両順序0点であり、512の5/0点も改善しない。副格子でも7/0点に留まる。単独の有利な256窓で44点を拾うことと、任意位置でページ全体を安定して読むことは異なる。確率を平均すると他タイルでの白判定も加わるため、1つの窓の回復がそのまま残る保証はない。',
        '## 清潔な文字と実汚れへの影響',
        table(['条件・推論','無損傷墨/5500','退色墨/5500','D01/225','D03上側/124','D04黒/5100','T07黒/4500','C01黒/10800'],collateral),
        '分割で合成汚れの紙や実斑点の黒画素が減る場合もあるが、文字も失う。D04は墨芯＋紙の主格子でも2613/2236画素が黒く、Howe固定0に届かない。D01の被覆も206→201、198→194へ下がった。実画像の紙領域は暫定の原本確認範囲で、不可視の墨の真値を保証しない。合成の原本由来proxyも専門家GTではなく、正式な精度順位ではない。',
        '## 重なりと境界の診断',
        '重なったタイルで墨/紙の判定が割れる画素と、確率の重み付き標準偏差を保存した。境界帯は各内部タイル端から4画素以内で、Hannで混ぜるため実際の硬い貼り合わせ線を指すものではない。境界帯とそれ以外の既知紙の黒画素を分け、境界帯の分母も併記する。',
        table(['条件','画像','判定が割れる墨芯','墨芯分母','全重なりの判定割れ','境界帯紙黒/紙','他の紙黒/紙'],seamrows),
        '## 再計算と検証',
        f'6重みのstate_dictは診断前後、4重みは分割推論前後でも完全一致した。入力・教師・注釈のハッシュを照合。旧キャッシュとの再計算は512が一致し、256マスクは最大{maxflip}画素の差があったが、全ての追跡被覆r2は一致した。入力メモリ配置だけを替えてもこの小差は解消せず、原因は未分離である。新しい要因対照は同一読み出し処理内で測定した。初回の根拠のない「最大2画素」照合ガードが停止した記録はattempt1へ残し、差を隠さず報告する方式へ修正した。',
        '[事前プロトコル](protocol.json)、[固定格子・入力マニフェスト](tiling_protocol.json)、[混合計算の確認](tiling_blend_checks.json)、[順序1の要因全数値](audit_order0.json)、[順序2](audit_order1.json)、[順序1の分割全数値](tiling_order0.json)、[順序2](tiling_order1.json)。',
        '今回の結果から損傷合成を追加する必要はまだ判断しない。次はRGBファイル・教師・モデル・提示回数・損失を揃え、確認済み対象を窓内に保ったまま提示ごとに位置を変える学習対照を用意する。学習窓への当てはまりだけでなく、固定512推論・両固定格子・別ページ・実汚れを同時に判定する。局所墨芯の満点を、線幅や字形の正しさ・GAN固有の学習効果とは扱わない。',
        '## 原本付きPNG',
        '\n'.join(f'- [{Path(f).name}]({f})' for f in figures),
    ]
    (OUT/'研究判断_推論範囲と切り出し位置.md').write_text('\n\n'.join(report)+'\n')
    dump(OUT/'combined_results.json',dict(audit_rows=ar,tiling_rows=tr,real_regions=rr,figures=figures,
        source_json_sha256={name:sha(OUT/name) for name in ['audit_order0.json','audit_order1.json','tiling_order0.json','tiling_order1.json']},
        no_weight_updates=True,formal_accuracy=False))
    shutil.copy2(Path(__file__),OUT/'source_snapshot'/Path(__file__).name)
    print('Reported:',len(ar),'audit rows,',len(tr),'tiling rows,',len(figures),'PNG comparisons',flush=True)


if __name__=='__main__':
    main()
