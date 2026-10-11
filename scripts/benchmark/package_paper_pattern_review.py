"""Check observed-paper donor provenance and package the full fixed-ROI review."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import dump, read, sha, FONT_PATH
from package_defect_context_review import stack
from package_paper_blending_review import boundary_metric, check_bundle
import probe_paper_pattern_continuity as probe

ROOT = probe.ROOT
OUT = ROOT/'tmp_work/paper_pattern_review_20261001'
PILOT = probe.OUT
PRIOR = probe.prior.OUT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in [PILOT, PRIOR, probe.prior.DDRM]:
        check_bundle(folder)
    rows = json.loads((PILOT/'diagnostics.json').read_text())
    assert len(rows) == len({r['id'] for r in rows}) == 22
    donor_stats = []; thumbnails = []
    pages = {}
    for source in probe.prior.previous.margin_probe.SOURCES:
        for p in json.loads((source/'baseline_review.json').read_text())['cases']:
            pages[p['page']] = p
    assert len(pages) == 14
    independent = []
    for row in rows:
        key = row['id']; native = PILOT/'native'/key
        x, y, w, h = row['xywh']; original = read(native/'original.png')
        base = read(native/'base.png'); edit = read(native/'edit.png', True)
        adjusted = read(native/'adjusted.png', True)
        assert np.array_equal(base, read(PRIOR/'native'/key/'boundary.png'))
        assert np.array_equal(edit, read(probe.prior.DDRM/'native'/key/'candidate_edit.png', True))
        assert np.array_equal(original, read(probe.prior.DDRM/'native'/key/'original.png'))
        assert not np.any(adjusted & ~read(PRIOR/'native'/key/'adjusted.png', True))
        assert not np.any(adjusted & ~edit)
        reference = read(native/'reference_context.png', True)
        used = read(native/'donor_context.png', True)
        donors = json.loads((native/'donors.json').read_text())
        cx, cy, cw, ch = donors['context_xywh']; oy, ox = y-cy, x-cx
        local = (slice(oy, oy+h), slice(ox, ox+w))
        assert reference.shape == (ch, cw)
        assert np.array_equal(reference, read(PRIOR/'native'/key/'reference_context.png', True))
        assert not np.any(reference[local] & edit)
        union = np.zeros_like(reference)
        for dx, dy in donors['patch_xy']:
            patch = reference[dy:dy+24, dx:dx+24]
            assert patch.shape == (24, 24) and patch.all()
            union[dy:dy+24, dx:dx+24] = True
        assert np.array_equal(union, used)
        assert len(donors['patch_xy']) == row['donor_patches']
        evaluation = read(probe.prior.previous.OUT/'native'/key/'adjusted.png', True)
        for mode in ['base']+probe.MODES:
            value = read(native/(mode+'.png'))
            assert not np.any(np.any(value!=base, axis=2) & ~adjusted)
            assert not np.any(np.any(value!=original, axis=2) & ~edit)
            count, median = boundary_metric(value, evaluation, reference[local])
            recorded = row['metrics'][mode]
            assert count == recorded['boundary_pairs']
            if median is None: assert recorded['median_boundary_deltaLab'] is None
            else: assert abs(median-recorded['median_boundary_deltaLab']) < 1e-6
        independent.append(key)
        if donors['patch_xy']:
            full = read(ROOT/pages[row['page']]['original'])
            context = full[cy:cy+ch, cx:cx+cw]
            light = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)[:,:,0]
            stds = []
            for dx, dy in donors['patch_xy']:
                patch = light[dy:dy+24, dx:dx+24]
                stds.append(float(np.std(patch-cv2.GaussianBlur(patch, (0,0), 1.2))))
            donor_stats.append(dict(id=key, patches=len(stds), median_donor_L_highpass_std=float(np.median(stds)),
                                    q10=float(np.quantile(stds, .1)), q90=float(np.quantile(stds, .9))))
            thumbnails.append((key, [context[dy:dy+24, dx:dx+24].copy()
                for dx, dy in [donors['patch_xy'][i] for i in np.linspace(0, len(stds)-1, 8, dtype=int)]]))
    overview = []
    for filename, group, keys, title, notes in [
        ('01_paper_pattern.png', 'comparisons', ['hole_edge_new','new_margin_hole'], '紙色と紙模様の連続性の比較',
         ['マスクとDDRMを固定。紙色のつながり／紙模様／両方を分けた試作。',
          '違和感は十分に減らない。今回の方式を前回の境界補正に置き換えない。']),
        ('02_remaining_mask.png', 'mask_diagnostic', ['hole_edge_new','new_margin_hole'], '残った損傷は編集範囲の内か外か',
         ['赤線は現在の編集範囲の輪郭。新しい穴の右上の濃い部分はマスク外。',
          'マスク外は今回の合成条件では変わらない。模様の改善とは別の課題。']),
        ('05_controls.png', 'comparisons', ['new_frame_text','new_header_ink','new_mono_text'], '枠・見出し・本文の対照',
         ['補正対象0の対照。全18箇所で前回の保存出力と完全一致。']),
        ('06_paper_and_stamp.png', 'comparisons', ['brown_false_red_new','new_brown_stamp'], '茶色い紙面と印影の対照',
         ['補正画素は前回と同じ。印影の小さな点の消失は以前のマスクに由来する。'])]:
        stack([PILOT/group/(key+'.png') for key in keys], OUT/filename, title, notes)
        overview.append(filename)
    crops = {}
    for filename, key in [('03_old_hole_closeup.png','hole_edge_new'),('04_new_hole_closeup.png','new_margin_hole')]:
        native = PILOT/'native'/key; target = read(native/'adjusted.png', True)
        yy, xx = np.nonzero(target); h, w = target.shape
        x0, y0 = max(0, int(xx.min())-12), max(0, int(yy.min())-12)
        x1, y1 = min(w, int(xx.max())+13), min(h, int(yy.max())+13)
        crops[key] = [x0,y0,x1-x0,y1-y0]
        images = [read(native/(name+'.png'))[y0:y1,x0:x1] for name in ['original','base']+probe.MODES]
        probe.prior.previous.figure(images, ['原画像','前回の境界補正','紙色のつながり','紙模様のみ','紙色＋紙模様'],
            OUT/filename, key+' · 補正範囲の拡大', '同じ切り出しを最近傍で拡大。境界と塗り跡は残る。')
        overview.append(filename)
    filename = '07_known_paper_donors.png'
    canvas = Image.new('RGB', (1200, 640), (250,248,243)); draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 22); small = ImageFont.truetype(FONT_PATH, 18)
    draw.text((20,12), '模様の参照候補：未編集紙面の24px角パッチ', fill='black', font=font)
    draw.text((20,46), '各パッチ全画素が参照条件を満たす。候補の並びから等間隔に8個を表示。', fill='black', font=small)
    for i, (key, patches) in enumerate(thumbnails):
        y = 90+i*130; draw.text((20,y), key, fill='black', font=small)
        for j, patch in enumerate(patches):
            im = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)).resize((96,96), Image.Resampling.NEAREST)
            canvas.paste(im, (370+j*100,y))
    canvas.save(OUT/filename); overview.append(filename)
    dump(OUT/'diagnostics.json', dict(donors=donor_stats, closeup_xywh=crops, regions=rows))
    items = [(OUT/name, 'overview/'+name) for name in overview]
    for group in ['comparisons','mask_diagnostic']:
        items.extend((PILOT/group/(r['id']+'.png'), group+'/'+r['id']+'.png') for r in rows)
    zip_path = OUT/'paper_pattern_comparison_images.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(zip_path) as archive:
        assert len(archive.namelist()) == len(items) == 51 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    fmt = lambda val: '—' if val is None else f'{val:.3f}'
    boundaries = '\n'.join('| `'+r['id']+'` | '+' | '.join(fmt(r['metrics'][m]['median_boundary_deltaLab'])
        for m in ['base']+probe.MODES)+' |' for r in rows if r['adjusted_pixels'])
    texture = '\n'.join('| `'+r['id']+'` | '+' | '.join(fmt(r['metrics'][m]['core_highpass_std'])
        for m in ['base']+probe.MODES)+' |' for r in rows if r['adjusted_pixels'])
    donor_table = '\n'.join(f'| `{r["id"]}` | {r["patches"]} | {r["median_donor_L_highpass_std"]:.3f} | {r["q10"]:.3f}〜{r["q90"]:.3f} |' for r in donor_stats)
    all_rows = '\n'.join(f'| [{r["title"]}](../{PILOT.name}/comparisons/{r["id"]}.png) · `{r["id"]}` | {r["adjusted_pixels"]:,} | {r["donor_patches"]} |' for r in rows)
    report = f'''# 紙模様の連続性とマスク外の残存損傷

2026-10-01。**未編集紙面の模様を移す試作では、残る違和感を十分に減らせなかった。新しい穴では補修部分がさらに平滑になる傾向があり、今回の方式へ置き換えず、前回の隣接境界補正を基準として維持する。** 濃く残る損傷がマスク外であることも比較PNGで分離した。

今回も14ページ・11冊・22箇所の開発例を全て扱い、同じ編集マスクと保存済みDDRMを使用した。新しいDDRM推論・学習・マスクの拡張はない。比較の基準は[前回](../paper_blending_review_20261001/report.md)の**境界のみ**の補正である。

## 比較PNG

- [虫食い2例の要素比較](01_paper_pattern.png)：原画像、前回の境界補正、紙色のつながり、紙模様のみ、両方。
- [編集境界と残存損傷](02_remaining_mask.png)：赤線は現在の編集境界。マスク外の損傷は変わらない。
- [既存の穴の拡大](03_old_hole_closeup.png)、[新しい穴の拡大](04_new_hole_closeup.png)。切り出しは補正範囲の外接矩形＋12pxで全条件共通。
- [本文・見出し・枠の対照](05_controls.png)、[紙面上端と印影](06_paper_and_stamp.png)、[参照紙面のパッチ](07_known_paper_donors.png)。
- [全51枚のPNG ZIP](paper_pattern_comparison_images.zip)：概要7、全ROIの要素比較22、編集境界の診断22。

## 分けた条件

補正対象は前回の対象マスクと完全に同じ4箇所で、残る18箇所は変更しない。元のマスク外にある画素も変更しない。紙色のつながりは、未編集の参照紙面からσ6pxの正規化低周波Labを求め、境界条件として編集内部へ補間した。4近傍の離散ラプラシアンと弱い既存生成色への拘束（1/48²）を用い、既存生成画素のσ3pxからの細かい差分は残す。

紙模様のみは、保存済みの紙面参照マスク内に全画素が入る24×24pxパッチを8px間隔で集める。候補は最大512、4個未満なら補正しない。元のインク保護・編集対象の画素を候補に含めない。原画像の参照紙面だけで正規化したσ3px低周波との差をパッチの模様として用い、8px重複で敷き詰める。重複部分のLab差と再利用回数で候補を選び、重複部分を加重平均する。生成された低周波の紙色は維持する。両方は低周波補間と、この観測紙面の模様を合わせる。

全条件とも編集内部4pxで前回出力へ弱めて接続する。元の損傷画素を混ぜ戻さず、新しい乱数の粒状模様も生成しない。ただしパッチの再配置なので、歴史資料の失われた紙模様を真に復元したという意味ではない。印影の消失した点も今回回復していない。

## 診断値と目視

境界値は前回と同じ境界対でのLab距離中央値。ΔE2000・GT精度・目視の品質点ではない。

| ROI | 前回の境界補正 | 紙色のつながり | 紙模様のみ | 両方 |
| --- | ---: | ---: | ---: | ---: |
{boundaries}

既存の穴では新しい3条件の境界値はいずれも前回より増えた。新しい穴は両方で2.039→1.961と小さくなるが、平坦な領域と薄い輪郭が目視で残る。値の改善だけで自然さの向上と判断しない。印影と上端紙面でも両方が常に良いわけではない。

編集内部6px以上のL高周波（σ1.2px）標準偏差も比較した。これは模様の量の診断で、値の大小自体を成功の判定に使わない。

| ROI | 前回の境界補正 | 紙色のつながり | 紙模様のみ | 両方 |
| --- | ---: | ---: | ---: | ---: |
{texture}

新しい穴では模様を入れ替えた側の高周波量が大きく下がった。参照が明るく平滑な紙面に偏ること、パッチの重複平均と8bit量子化が模様を弱めることが考えられるが、各要因の寄与は分離していない。模様が少ない／多いだけでは自然さを説明できず、今回の移植を改善と確定しない。

候補パッチの原画像L高周波標準偏差（σ1.2px、パッチごとの中央値と10〜90%範囲）は次の通り。こちらは原パッチごとに平滑化した診断で、編集内部と領域サイズ・境界処理が違い、直接の品質比較ではない。

| ROI | 候補パッチ数 | 中央値 | 10〜90% |
| --- | ---: | ---: | ---: |
{donor_table}

## 全22箇所

| ROI | 補正対象画素 | 参照候補パッチ数 |
| --- | ---: | ---: |
{all_rows}

マスク外に残る濃い右上部分は、今回の補正の効き不足とは区別する。既存の穴の薄い輪郭には、マスク外に残る部分と編集内部の塗り跡の両方がある。本文・枠・見出しなど18箇所は補正対象0で前回と完全一致した。条件による対象外であり、未知資料の文字保存精度の実証ではない。利用者が許容した以前の印影の小さな点の変化は、新しい阻害要因として再扱いしない。

## 確認と次の段階

全参照候補の24×24画素が未編集の紙面参照に入ること、保存入力・マスク・DDRM・ソース・PNGのハッシュ、22箇所の補正範囲外／編集範囲外の不変、境界値の別実装での再計算、ZIP内51PNGのハッシュを確認した。参照領域を固定して原画像の編集対象を任意の色に置き換えても、穴2例と印影の全条件の出力は一致した。[確認JSON](verification.json)。参照ゲート自体の正しさを画素GTで評価した確認ではない。

次の優先事項は、模様の条件を追加で調整することより、濃い残存損傷をマスク外に置いている理由を検証すること。特に新しい穴の上余白帯の境界に接する成分の一括除外を、本文からの距離・印影保護と両立して見直す。マスクの変更を先にPNGで確認し、その候補のみを固定DDRM・前回の境界補正へ接続する。今回の模様試作は履歴に保存し、その実験には混ぜない。本文と重なるシミ・欠損文字の復元は未解決のまま保持する。

[試作の固定条件](../{PILOT.name}/manifest.json)、[全診断値](../{PILOT.name}/diagnostics.json)、[原寸出力](../{PILOT.name}/native)、[実装](../../scripts/benchmark/probe_paper_pattern_continuity.py)。
'''
    (OUT/'report.md').write_text(report)
    dump(OUT/'verification.json', dict(status='passed', pages=14, regions=independent,
        donor_entire_24px_patch_within_known_reference=True, donor_map_equals_patch_union=True,
        outside_adjusted_changed_pixels=0, outside_edit_changed_pixels=0,
        unchanged_regions=[r['id'] for r in rows if not r['adjusted_pixels']],
        independent_boundary_metric=True, source_and_input_hashes=True, PNG_zip_count=51, zip_hashes=True))
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)),
        dependencies={str((f/n).relative_to(ROOT)):sha(f/n) for f in [PILOT,PRIOR,probe.prior.DDRM]
                      for n in ['manifest.json','complete.json']},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print('Passed: all 22 ROIs; donor provenance; 51 PNGs packaged', flush=True)


if __name__ == '__main__':
    main()
