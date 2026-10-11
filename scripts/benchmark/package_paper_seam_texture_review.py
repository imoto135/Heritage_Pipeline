"""Verify donor provenance and matched overlap averages; deliver comparison PNGs."""
import gc
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_paper_seam_texture as probe
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_paper_blending_review import boundary_metric, check_bundle

ROOT = probe.ROOT
PILOT = probe.OUT
BASE = probe.prior.OUT
OUT = ROOT / 'tmp_work/paper_seam_texture_review_20261001'
MODES = probe.MODES


def verify_texture(lab, known, applied, native):
    """Check each saved seam pixel against a recorded donor tile, without quilting."""
    metadata = json.loads((native / 'donors.json').read_text())
    seam = np.load(native / 'texture_seam.npy', allow_pickle=False)
    average = np.load(native / 'texture_average.npy', allow_pickle=False)
    sx = np.load(native / 'source_x.npy', allow_pickle=False)
    sy = np.load(native / 'source_y.npy', allow_pickle=False)
    if not metadata['patch_xy']:
        assert not applied.any()
        assert not seam.any() and not average.any()
        assert (sx == -1).all() and (sy == -1).all()
        return dict(pixels=0, average_error=0., distinct_donors=0)
    assert np.array_equal(sx >= 0, applied)
    assert np.array_equal(sy >= 0, applied)
    assert known[sy[applied], sx[applied]].all()
    assert not seam[~applied].any() and not average[~applied].any()
    light = lab[:, :, 0]
    weights = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)
    low = cv2.GaussianBlur(np.where(known, light, 0), (0, 0), 6) / np.maximum(weights, 1e-6)
    signal = light - low
    patches = []
    for px, py in metadata['patch_xy']:
        assert known[py:py+24, px:px+24].shape == (24, 24)
        assert known[py:py+24, px:px+24].all()
        patch = signal[py:py+24, px:px+24].copy()
        patches.append(np.clip(patch - float(patch.mean()), -3, 3))
    assert float(np.abs(seam).max()) <= 3
    n, labels, stats, _ = cv2.connectedComponentsWithStats(applied.astype(np.uint8), connectivity=4)
    assert len(metadata['components']) == n - 1
    axis = np.array([min(1., (i+1)/8., (24-i)/8.) for i in range(24)], np.float32)
    taper = axis[:, None] * axis[None, :]
    coverage = np.zeros_like(applied)
    reconstructed = np.zeros_like(average)
    donors_used = set()
    for component in metadata['components']:
        index = component['id']
        x, y, w, h, area = map(int, stats[index])
        assert component['xywh'] == [x, y, w, h] and component['area'] == area
        canvas_shape = (max(h, 24)+24, max(w, 24)+24)
        total = np.zeros(canvas_shape, np.float32)
        mass = np.zeros(canvas_shape, np.float32)
        for tile in component['tiles']:
            tx, ty, index = tile['x'], tile['y'], tile['donor']
            px, py = metadata['patch_xy'][index]
            assert tile['source_xy'] == [px, py]
            donors_used.add(index)
            patch = patches[index]
            total[ty:ty+24, tx:tx+24] += patch * taper
            mass[ty:ty+24, tx:tx+24] += taper
            tw, th = min(24, w-tx), min(24, h-ty)
            if tw <= 0 or th <= 0:
                continue
            ys, xs = slice(y+ty, y+ty+th), slice(x+tx, x+tx+tw)
            yy, xx = np.indices((th, tw))
            # A seam pixel must equal a placed, unaveraged observed donor pixel.
            matches = ((sx[ys, xs] == px+xx) & (sy[ys, xs] == py+yy)
                       & np.isclose(seam[ys, xs], patch[:th, :tw], rtol=0, atol=1e-6))
            coverage[ys, xs] |= matches & (labels[ys, xs] == component['id'])
        local = labels[y:y+h, x:x+w] == component['id']
        reconstructed[y:y+h, x:x+w][local] = (total[:h, :w] / np.maximum(mass[:h, :w], 1e-6))[local]
    assert coverage[applied].all()
    error = float(np.abs(reconstructed - average).max())
    assert error < 1e-6, error
    return dict(pixels=int(applied.sum()), average_error=error, distinct_donors=len(donors_used))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    masks = probe.prior.previous.mask_probe.OUT
    ddrm = probe.prior.previous.OUT
    for folder in [PILOT, BASE, masks, ddrm]:
        check_bundle(folder)
    # Exhaustive small path costs check the seam optimizer independently.
    for cost in [np.arange(12).reshape(3, 4), np.array([[8, 1, 9, 9], [0, 9, 1, 9], [9, 0, 9, 1]])]:
        candidates = [p for p in itertools.product(range(4), repeat=3)
                      if all(abs(p[i]-p[i-1]) <= 1 for i in range(1, 3))]
        optimum = min(sum(cost[y, p[y]] for y in range(3)) for p in candidates)
        result = probe.minimum_path(cost)
        assert sum(cost[y, result[y]] for y in range(3)) == optimum
    rows = json.loads((PILOT / 'diagnostics.json').read_text())
    lookup = {r['id']: r for r in rows}
    assert len(rows) == len(lookup) == 22
    provenance = []
    pages, books = set(), set()
    for source in probe.prior.previous.mask_probe.old.SOURCES:
        for page in json.loads((source / 'baseline_review.json').read_text())['cases']:
            pages.add(page['page']); books.add(page['book'])
            original = read(ROOT / page['original'])
            full_edit = read(masks / 'pages' / page['page'] / 'completed/edit.png', True)
            for roi in page['rois']:
                rid = roi['id']; row = lookup[rid]
                native, previous = PILOT / 'native' / rid, BASE / 'native' / rid
                cx, cy, cw, ch = row['context_xywh']; x, y, w, h = row['xywh']
                context = original[cy:cy+ch, cx:cx+cw]
                s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
                base = read(native / 'base.png')
                assert np.array_equal(base, read(previous / 'boundary_smooth.png'))
                assert np.array_equal(read(native / 'original.png'), context[s])
                for name in ['edit', 'applied', 'reference_context', 'applied_context']:
                    assert sha(native / (name+'.png')) == sha(previous / (name+'.png'))
                edit = read(native / 'edit.png', True)
                applied = read(native / 'applied_context.png', True)
                known = read(native / 'reference_context.png', True)
                assert np.array_equal(edit, full_edit[y:y+h, x:x+w])
                assert not np.any(known & full_edit[cy:cy+ch, cx:cx+cw])
                lab = cv2.cvtColor(context.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
                proof = verify_texture(lab, known, applied, native)
                provenance.append(dict(id=rid, **proof))
                raw = np.load(previous / 'boundary_raw_field_Lab.npy', allow_pickle=False)
                smooth = np.load(previous / 'boundary_smooth_field_Lab.npy', allow_pickle=False)
                dist = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
                bw = np.exp(-(np.maximum(dist-1, 0)/8.)**2) * applied
                tw = np.clip((dist-1)/4., 0, 1) * applied
                boundary = smooth + (raw-smooth) * bw[:, :, None]
                core = applied[s] & (dist[s] >= 6)
                for mode in ['base'] + MODES:
                    value = read(native / (mode+'.png'))
                    assert not np.any(np.any(value != base, axis=2) & ~applied[s])
                    assert not np.any(np.any(value != context[s], axis=2) & ~edit)
                    if mode != 'base':
                        field = (boundary if mode in ['boundary', 'combined'] else smooth).copy()
                        if mode in ['texture_seam', 'combined', 'texture_average']:
                            name = 'texture_average' if mode == 'texture_average' else 'texture_seam'
                            field[:, :, 0] += np.load(native / (name+'.npy'), allow_pickle=False) * tw
                        expected = cv2.cvtColor(field[s], cv2.COLOR_LAB2BGR)
                        expected = np.rint(np.clip(expected, 0, 1)*255).astype(np.uint8)
                        assert np.array_equal(value[applied[s]], expected[applied[s]])
                    count, median = boundary_metric(value, applied[s], known[s])
                    stats = row['diagnostics'][mode]
                    assert count == stats['boundary_pairs']
                    recorded_median = stats['median_boundary_deltaLab']
                    assert (median is None and recorded_median is None) or np.isclose(median, recorded_median, rtol=0, atol=1e-6)
                    light = cv2.cvtColor(value.astype(np.float32)/255, cv2.COLOR_BGR2LAB)[:, :, 0]
                    hp = light - cv2.GaussianBlur(light, (0, 0), 1.2)
                    assert stats['core_L_highpass_std'] == (float(hp[core].std()) if core.any() else None)
                    assert stats['L_le_80_applied_pixels'] == int(((light <= 80) & applied[s]).sum())
                    assert stats['changed_vs_base'] == int(np.any(value != base, axis=2).sum())
                assert row['applied_pixels'] == int(applied[s].sum())
                print('Verified:', rid, flush=True)
            del original, full_edit
            gc.collect()
    assert len(pages) == 14 and len(books) == 11
    unchanged = [r['id'] for r in rows if not r['applied_pixels']]
    assert len(unchanged) == 18
    overview = []
    for filename, keys, group, title, notes in [
        ('01_boundary_and_texture.png', ['new_margin_hole', 'hole_edge_new'], 'comparisons', '境界の紙色と紙模様を分けて比較',
         ['境界紙色は馴染みが前進。紙模様は既存の穴で見えるが、新しい穴では変化が小さい。',
          '薄い縁と塗り跡は残る。原画像・前回・境界のみ・模様のみ・両方の順。']),
        ('02_overlap_ablation.png', ['new_margin_hole', 'hole_edge_new'], 'overlap_ablation', '同じ紙面パッチ：重複平均と継ぎ目',
         ['パッチ選択・配置・明度成分・強さは同一。重複部分のつなぎ方だけを変える。',
          '継ぎ目は平均による減衰を一部避けるが、平均との差は小さい。']),
        ('05_controls.png', ['new_frame_text', 'new_header_ink', 'new_mono_text'], 'comparisons', '枠・見出し・本文の対照',
         ['適用0。今回の全条件は前回の保存出力と完全一致。']),
        ('06_stamp_and_paper.png', ['new_brown_stamp', 'brown_false_red_new'], 'comparisons', '印影と広い紙色ムラの対照',
         ['印影の以前の点の消失は解決していない。今回のマスク追加は0。',
          '広い茶色い紙面は224画素だけ適用。全体の色ムラを解決したと数えない。'])]:
        stack([PILOT / group / (key+'.png') for key in keys], OUT / filename, title, notes)
        overview.append(filename)
    crops = json.loads((ROOT / 'tmp_work/known_paper_harmonic_fill_review_20261001/diagnostics.json').read_text())['closeup_xywh']
    figure = probe.prior.previous.prior_ddrm.figure
    for filename, rid in [('03_new_hole_closeup.png', 'new_margin_hole'), ('04_old_hole_closeup.png', 'hole_edge_new')]:
        x, y, w, h = crops[rid]; native = PILOT / 'native' / rid
        modes = ['original', 'base', 'boundary', 'texture_seam', 'combined']
        figure([read(native / (m+'.png'))[y:y+h, x:x+w] for m in modes],
               ['原画像', '前回の紙色補間', '境界紙色のみ', '紙模様のみ・継ぎ目', '境界＋紙模様'],
               OUT / filename, rid+' · 前回と同じ範囲を拡大',
               '固定マスク。紙色場の境界補正と、未編集紙面の明度模様を比較。',
               '全条件で同じ切り出しと最近傍拡大。薄い縁・塗り跡は残る。')
        overview.append(filename)
    rid = 'new_margin_hole'; native = PILOT / 'native' / rid
    old = probe.prior.previous.prior_ddrm.OUT / 'native' / rid
    figure([read(p) for p in [native/'original.png', old/'npj.png', old/'v6.png',
                             BASE/'native'/rid/'ddrm.png', native/'base.png', native/'combined.png']],
           ['原画像', 'npj → DDRM', 'v6 → DDRM', '回収マスク → DDRM', '前回の紙色補間', '境界＋紙模様'],
           OUT/'07_npj_v6_context.png', '保存済みnpj・固定v6との位置づけ',
           '同じ原画像。保存済みDDRMを再利用し、紙色補間の適用部分だけを変更。',
           'マスクと復元方式の両方が異なる。最新列はDDRMだけの結果ではない。')
    overview.append('07_npj_v6_context.png')
    items = [(OUT/name, 'overview/'+name) for name in overview]
    for group in ['comparisons', 'overlap_ablation']:
        items.extend((PILOT/group/(r['id']+'.png'), group+'/'+r['id']+'.png') for r in rows)
    archive_path = OUT / 'paper_seam_texture_comparison_images.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items:
            archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(items) == 51 and archive.testzip() is None
        for path, name in items:
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    boundary_table = '\n'.join(f'| `{r["id"]}` | {r["applied_pixels"]:,} | {r["diagnostics"]["base"]["median_boundary_deltaLab"]:.3f} | {r["diagnostics"]["boundary"]["median_boundary_deltaLab"]:.3f} |' for r in rows if r['applied_pixels'])
    texture_table = '\n'.join('| `'+rid+'` | '+' | '.join(f'{lookup[rid]["diagnostics"][m]["core_L_highpass_std"]:.4f}' for m in ['base', 'texture_average', 'texture_seam', 'combined'])+' |' for rid in ['hole_edge_new', 'new_margin_hole'])
    all_table = '\n'.join(f'| [{r["title"]}](../{PILOT.name}/comparisons/{r["id"]}.png) · `{r["id"]}` | {r["applied_pixels"]:,} | {r["donor_patches"]} |' for r in rows)
    report = f'''# 固定した紙色補間の境界と紙模様を改善する

2026-10-01。**虫食い2例で境界の紙色差が小さくなった。未編集紙面の模様を加えると既存の穴の平坦さは和らぐが、新しい穴では変化が小さい。薄い縁・塗り跡は残り、自然な修復の完成には至っていない。**

同じ14ページ・11冊・22箇所の開発例で、前回の全成分マスク、紙色補間の適用範囲、紙面参照、保存された直接／平滑のLab補間場を固定した。前回の平滑境界画像を比較基準にする。今回もマスク追加・学習・新しいDDRM推論はない。

## 比較PNG

- [境界と紙模様の分離比較](01_boundary_and_texture.png)。原画像、前回、境界のみ、紙模様のみ、両方。
- [同じパッチの重複平均と継ぎ目](02_overlap_ablation.png)。
- [新しい穴の上方の拡大](03_new_hole_closeup.png)、[既存の穴の拡大](04_old_hole_closeup.png)。前回と同じ切り出しを最近傍で拡大。
- [枠・見出し・本文の対照](05_controls.png)、[印影と広い紙色ムラ](06_stamp_and_paper.png)。
- [保存済みnpj・固定v6・今回の位置づけ](07_npj_v6_context.png)。npj/v6列は保存したDDRM比較、最新列は紙色補間とその後処理であり、マスクと復元方式の両方が異なる。元npj全工程の再現ではない。
- [全51PNG ZIP](paper_seam_texture_comparison_images.zip)。概要7枚、全22箇所の分離比較、全22箇所の重複方式比較。

## 境界の紙色

保存された平滑境界場を、補間内部の最初の8px程度で直接境界場に近づける。最初の内側画素では直接境界を使い、内部ほど平滑境界に戻す。未編集の損傷画素やDDRMの模様を混ぜる操作ではない。参照が欠ける外周へ新たな紙色条件を作ることもしていない。

| ROI | 固定適用画素 | 前回の境界Lab距離中央値 | 境界補正後 |
| --- | ---: | ---: | ---: |
{boundary_table}

同じ参照境界対を使った診断値で、境界補正が目的とする量そのもの。独立した自然さの点数・損傷除去精度ではない。紙模様は最初の内側画素では重み0なので、模様のみの境界値は前回と同じ、両方の境界値は境界のみと同じになる。

PNGでは境界の急な紙色差が和らぐ一方、既存の穴の輪郭や新しい穴の薄い茶色い残りは見える。保存マスク外の画素は原画像のままなので、そこに残る縁を今回の後処理で消したとは扱わない。補間内部と周囲の大きな色ムラも一致していない。

## 紙模様と重複平均の比較

保存された紙面参照に24×24pxの全画素が入るパッチだけを候補にする。8px間隔で候補を取り、最大512個に制限した。参照だけを使った正規化Gaussian（σ6px）との差から明度Lの模様を取り出し、各パッチの平均を引き、±3Lに制限する。色成分a/bの模様、乱数模様、DDRM由来の模様は使わない。

24pxパッチを16px間隔で配置し、8pxの重複部分は明度差の小さい継ぎ目でつなぐ。同じ選択パッチ・同じ配置を重複平均でも合成して比較する。模様の強さと境界から5pxまでのフェザーも同じ。**今の平均条件と継ぎ目条件だけが、重複処理の比較として対応している。** 過去の模様試作とは基準画像・マスク・参照・低周波条件が異なるので、継ぎ目だけによる改善とは比較しない。

| ROI | 前回の内部高周波量 | 同じパッチ・重複平均 | 同じパッチ・継ぎ目 | 境界＋継ぎ目 |
| --- | ---: | ---: | ---: | ---: |
{texture_table}

高周波量は元解像度の出力Lからσ1.2pxのGaussianを引いた標準偏差。固定適用範囲の境界から6px以上離れた画素だけを数える。模様の振幅を観測する量であり、大きいほど自然・正確とは限らない。

既存の穴では粒のある紙面に近づく変化を確認できた。平均処理でも模様は戻り、継ぎ目との差は小さい。新しい穴は参照にできる紙面自体が滑らかで、模様を加えても見た目の変化は小さく、平坦な補修跡が残る。したがって「前回の失敗は重複平均だけが原因」「継ぎ目だけで解決」とは結論しない。

穴2例の固定適用範囲ではL≤80の画素数は全条件で0を維持した。灰色の生成模様を再導入した状態ではないが、これも暗さの診断であり、真正な紙模様を復元した証明ではない。印影の紙面は元から灰色なので、555画素中L≤80の数を穴と同じ品質尺度にはしない。印影の以前からの点の消失は今回も残る。利用者が許容した変化を新しいマスク誤編集として数えない。

広い上端の茶色い紙面は今回も224画素だけ適用。大部分に保存DDRMが残り、全体のシミを解決していない。ほか18箇所は全条件で前回画像と完全一致した。この対象外の対照を、未知資料の文字保存精度の証明にしない。

## 全22箇所の適用範囲

| ROI | 固定適用画素 | 参照パッチ候補 |
| --- | ---: | ---: |
{all_table}

## 確認と次の段階

保存入力・マスク・紙色場・ソース・出力のハッシュを確認した。全参照パッチが編集範囲外の固定紙面参照に入り、各継ぎ目画素が記録された配置パッチの観測明度模様と一致することを、継ぎ目処理を再実行せず座標と画素値から確認した。同じ配置記録から重複平均を別に再構成し、保存場との誤差は1e-6未満だった。小さな継ぎ目の最小経路は全列挙の最小コストとも一致した。

4適用例で編集対象の原画素を任意の色に変えても出力場が完全一致した。全22例の元解像度で、適用範囲外は前回画像、編集範囲外は原画素と一致した。境界対を別方向の列挙で再計算し、明度・高周波・変更画素数とZIP51PNGのハッシュも確認した。[確認JSON](verification.json)。

**境界補正と観測紙模様を組み合わせた条件を、次の開発比較の候補として保存する。** 本番採用や新手法の確立とはしない。繰り返し調整した22箇所であり、画素正解と未使用冊の独立評価はない。参照条件は文字がないことの保証でも、失われた歴史的な紙模様の真正な復元でもない。

次は塗り跡の残る新しい穴で、固定マスク外の薄い縁と、内部・周囲の低周波紙色ムラを分けて測る。今回の候補を保存したまま、輪郭の回収を変える比較と、紙色だけを変える比較を分離する。全資料への展開より先に、未使用冊の文字のない損傷例と本文対照で同じ固定条件を確かめる。イラストのページは引き続き対象外。

[固定条件](../{PILOT.name}/manifest.json)、[全診断](../{PILOT.name}/diagnostics.json)、[原寸出力・参照元座標](../{PILOT.name}/native)、[実装](../../scripts/benchmark/probe_paper_seam_texture.py)、[確認・梱包](../../scripts/benchmark/package_paper_seam_texture_review.py)、[前回の紙色補間](../known_paper_harmonic_fill_review_20261001/report.md)。
'''
    (OUT / 'report.md').write_text(report)
    dump(OUT / 'diagnostics.json', dict(donor_provenance=provenance, closeup_xywh=crops))
    dump(OUT / 'verification.json', dict(status='passed', pages=14, books=11, regions=22,
        masks_fields_applied_references_fixed=True, matched_donor_provenance=True,
        independent_overlap_reconstruction=True, maximum_average_error=max(p['average_error'] for p in provenance),
        exhaustive_small_seam_paths=True, outside_applied_changed_pixels=0, outside_edit_changed_pixels=0,
        original_damage_poison_cases=json.loads((PILOT/'verification.json').read_text())['original_damage_poison_invariance'],
        unchanged_regions=unchanged, independent_boundary_metrics=True, zip_PNG_count=51, zip_hashes=True))
    dependencies = {str((folder/name).relative_to(ROOT)): sha(folder/name)
                    for folder in [PILOT, BASE, masks, ddrm] for name in ['manifest.json', 'complete.json']}
    dump(OUT / 'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: fixed scope, observed donor pixels, matched averages, 51 PNGs', flush=True)


if __name__ == '__main__':
    main()
