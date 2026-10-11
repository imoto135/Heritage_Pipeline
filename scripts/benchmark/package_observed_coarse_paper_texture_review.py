"""Independent donor-coordinate band and mean-preserving correction verification."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_observed_coarse_paper_texture as probe
from build_npj_v6_comparison import dump, read, sha
from check_paper_margin_defect_ddrm import figure
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT = probe.ROOT
OUT = ROOT/'tmp_work/observed_coarse_paper_texture_review_20261001'


def lowpass(light, known, sigma):
    mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
    return cv2.GaussianBlur(np.where(known, light, 0), (0, 0), sigma)/np.maximum(mass, 1e-6)


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(probe.OUT)
    rows = json.loads((probe.OUT/'diagnostics.json').read_text()); assert len(rows) == 42
    textures = json.loads((probe.prior.OUT/'texture_sources.json').read_text()); pages = {}
    for source in probe.prior.references.prior.prior.mask.prior.MARGIN.SOURCES+[probe.prior.references.prior.prior.mask.new.INPUT]:
        for p in json.loads((source/'baseline_review.json').read_text())['cases']: pages[p['page']] = p
    current = None; errors = []; measured = []
    for r in rows:
        rid = r['id']; key = r['page']; native = probe.OUT/'native'/rid; previous = probe.prior.OUT/'native'/rid
        if current != key: original = read(ROOT/pages[key]['original']); current = key
        x, y, w, h = r['xywh']; cx, cy, cw, ch = r['context_xywh']
        context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w)); crop = context[s]
        known = read(native/'reference_context.png', True); applied = read(native/'applied_context.png', True)
        base = read(native/'base.png'); assert np.array_equal(base, read(previous/'base.png'))
        assert np.array_equal(crop, read(native/'original.png'))
        distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
        if applied[s].any():
            lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            source = ROOT/textures[rid]; sx = np.load(source/'source_x.npy', allow_pickle=False); sy = np.load(source/'source_y.npy', allow_pickle=False)
            selected = (sx >= 0)&(sy >= 0); assert not np.any(selected&~applied)
            assert known[sy[selected], sx[selected]].all()
            base_field = np.load(previous/'base_field_Lab.npy', allow_pickle=False)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(applied.astype(np.uint8), connectivity=4)
        for sigma in probe.SIGMAS:
            mode = 'coarse'+str(sigma); md = native/mode; value = read(md/'output.png')
            assert not np.any(np.any(value != base, axis=2)&~applied[s])
            if applied[s].any():
                band = lowpass(lab[:, :, 0], known, 6)-lowpass(lab[:, :, 0], known, sigma)
                borrowed = np.zeros(applied.shape, np.float32)
                borrowed[selected] = np.clip(band[sy[selected], sx[selected]], -2, 2)
                assert np.array_equal(borrowed, np.load(md/'borrowed_L.npy', allow_pickle=False))
                mass = cv2.GaussianBlur(selected.astype(np.float32), (0, 0), 6)
                smooth = cv2.GaussianBlur(borrowed, (0, 0), 6)/np.maximum(mass, 1e-6)
                assert np.array_equal(smooth, np.load(md/'smoothed_L.npy', allow_pickle=False))
                weight = np.clip((distance-1)/4., 0, 1)*selected
                delta = np.zeros(applied.shape, np.float32)
                for i in range(1, n):
                    target = labels == i; weights = weight[target].astype(np.float64); values = smooth[target].astype(np.float64)
                    if weights.sum() > 0:
                        center = float(np.sum(values*weights)/weights.sum()); correction = (values-center)*weights
                        scale = min(1., 2./max(float(np.abs(correction).max()), 1e-6))
                        delta[target] = (correction*scale).astype(np.float32)
                saved = np.load(md/'coarse_delta_L.npy', allow_pickle=False)
                assert np.array_equal(delta, saved) and not delta[distance <= 1].any()
                assert float(np.abs(delta).max()) <= 2+1e-6
                field = np.load(md/'result_field_Lab.npy', allow_pickle=False)
                expected = base_field.copy(); expected[:, :, 0] += delta
                assert np.array_equal(field, expected) and np.array_equal(field[:, :, 1:], base_field[:, :, 1:])
                for i in range(1, n):
                    target = labels == i
                    correction_mean = abs(float(delta[target].mean(dtype=np.float64)))
                    field_mean = abs(float((field[target, 0]-base_field[target, 0]).mean(dtype=np.float64)))
                    assert correction_mean < 1e-5 and field_mean < 1e-5
                    errors.append(dict(id=rid, sigma=sigma, component=i, correction_mean_error=correction_mean, field_mean_error=field_mean))
                converted_float = cv2.cvtColor(field[s], cv2.COLOR_LAB2BGR)
                converted = np.rint(np.clip(converted_float, 0, 1)*255).astype(np.uint8)
                assert np.array_equal(value[applied[s]], converted[applied[s]])
                measured.append(dict(id=rid, sigma=sigma, correction_std=float(delta[applied].std()),
                    converted_out_of_gamut_pixels=int((np.any((converted_float < 0)|(converted_float > 1), axis=2)&applied[s]).sum())))
            else: assert np.array_equal(value, base)
        print('Verified observed coarse paper:', rid, flush=True)
    items = []
    for name, ids, title, notes in [
        ('01_holes_coarse_texture.png', ['hole_edge_new', 'new_margin_hole'], '平均の紙色を変えずに観測の粗い明度模様を移す',
         ['明度ムラは少し増えるが、視覚効果は小さく、残存輪郭と塗り跡は残る。']),
        ('02_crack_and_fold.png', ['unseen_thin_crack', 'unseen_fold'], '裂けと折れにも同じ条件',
         ['成分の平均Lとa/bの場を保つ。模様だけで補修の形を隠せるとは判断しない。']),
        ('03_controls.png', ['unseen_seal', 'unseen_dense_text', 'new_uncertain_trace'], '本文・印影・不確かな跡',
         ['この対照は前回と完全一致。全42ROIを保存。'])]:
        stack([probe.OUT/'comparisons'/(rid+'.png') for rid in ids], OUT/name, title, notes); items.append((OUT/name, 'overview/'+name))
    # Context gallery makes the npj/v6 comparison and restoration differences explicit.
    for rid in ['hole_edge_new', 'new_margin_hole', 'unseen_thin_crack']:
        native = probe.OUT/'native'/rid
        if rid == 'unseen_thin_crack':
            original_review = probe.prior.references.prior.prior.fill.new.OUT/'native'/rid
            images = [read(native/'original.png'), read(original_review/'npj/output.png'), read(original_review/'v6/output.png'), read(native/'base.png'), read(native/'coarse48/output.png')]
            labels = ['原画像', 'npj→同じ紙面補間', 'v6→同じ紙面補間', '薄いコア＋枝端→補間', '平均色固定・粗い模様']
            kind = '同じ基本復元器へ違うマスク。最後の列だけ観測粗模様を追加。DDRMではない。'
        else:
            previous = ROOT/'tmp_work/paper_margin_defect_ddrm_20261001/native'/rid
            images = [read(native/'original.png'), read(previous/'npj.png'), read(previous/'v6.png'), read(native/'base.png'), read(native/'coarse48/output.png')]
            labels = ['原画像', 'npj→DDRM', 'v6→DDRM', '輪郭回収→紙面補間', '平均色固定・粗い模様']
            kind = 'マスクと復元方式の両方が異なる開発比較。前処理だけのDDRM精度比較ではない。'
        assert all(image.shape == images[0].shape for image in images)
        figure(images, labels, OUT/'baseline_context'/(rid+'.png'), rid+' · npjとv6からの位置づけ',
            '元npj全工程の再現ではない。保存された出力を原画素位置のまま再利用。', kind)
        items.append((OUT/'baseline_context'/(rid+'.png'), 'baseline_context/'+rid+'.png'))
    stack([OUT/'baseline_context'/(rid+'.png') for rid in ['hole_edge_new', 'new_margin_hole', 'unseen_thin_crack']],
        OUT/'04_npj_v6_context.png', '元のnpj・固定v6と最新の開発比較',
        ['旧穴はnpj/v6の保存DDRMと紙面補間を比較。新しい裂けは同じ紙面補間のマスク差。', '粗い模様の効果は小さい。前処理と復元器の変更を区別して読む。'])
    items.append((OUT/'04_npj_v6_context.png', 'overview/04_npj_v6_context.png'))
    items += [(probe.OUT/'comparisons'/(r['id']+'.png'), 'all_ROIs/'+r['id']+'.png') for r in rows]
    archive_path = OUT/'observed_coarse_paper_comparisons.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == 49 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    (OUT/'report.md').write_text('''# 平均紙色を保って観測した粗い明度模様を追加

2026-10-01。**観測された低周波の明度ムラを移すと補間部分のムラは少し増えるが、視覚効果は小さい。塗り跡・残存輪郭は解決しないため粗い模様の追加採用は保留し、次は条件を固定して別の冊へ検証を広げる。**

[虫食いの比較PNG](01_holes_coarse_texture.png)、[裂けと折れ](02_crack_and_fold.png)、[対照](03_controls.png)、[npj・v6との位置づけ](04_npj_v6_context.png)、[全49PNG ZIP](observed_coarse_paper_comparisons.zip)。全42ROIに原画像を並べた。

## 観測模様だけの変更

薄いコア＋近接8pxの元の紙面補間を基準にし、マスク・適用範囲・参照資格・細かい明度模様の元画素座標を固定した。既知紙面だけのGaussian L σ6とσ24／48の差を、保存された同じ元画素から借りる。±2Lに限定し、移植した値だけをσ6で正規化平滑化。成分ごとの羽根加重平均を引き、最大振幅が2Lを超える場合だけ線形縮小した。距離1pxの最初の画素は0、5px以上で全強度。a/bの場と成分ごとの平均Lは維持。編集内の原損傷・DDRM・乱数から模様を作らない。

既存の穴の大きな成分では追加L標準偏差が約.160／.186、新しい穴で.070／.109、裂けで.161／.159と小さい。これらは移した観測値の振幅で、独立品質・除去精度ではない。平均色を守る小さな変化では形状の塗り跡を十分に減らせない。大きく増幅して自然さと呼ぶ根拠もない。RGB変換後には丸め・色域制限があるため、画像から計算した平均Lが厳密に同じとは言わず、合成前のLab場で保存を確認した。[全確認値](diagnostics.json)。

## npj・v6比較の読み方

旧穴2例の文脈PNGは保存されたnpj／v6マスクのDDRMと、輪郭回収マスク＋紙面補間、粗い模様を比較する。マスクと復元器の両方が異なり、マスク精度だけによるDDRMの差ではない。新しい裂けはnpj／v6マスクを同じ紙面補間に渡した保存診断と、薄いコア＋枝端補完の紙面補間を比べる。こちらはDDRMではない。最後の粗模様の列だけ、平均色を保った追加の模様因子がある。元npj全工程の再現・真正な紙形状の復元・画素正解率は未実証。

## 検証と次の段階

同じ参照元座標が全て既知紙面にあること、既知画素だけの低周波差分・保存移植値・正規化平滑・成分平均引き・最大振幅・羽根0を別に再計算。a/b完全一致、各成分の平均L差、Lab合成とPNG、全84条件の適用範囲外一致、非参照異色置換不変、保存入力・ソース・出力とZIP49PNGのハッシュを確認。[検証JSON](verification.json)。

条件調整済み18ページの中だけで細かな色・模様変更を繰り返すより、薄いコア＋近接8pxを固定した開発候補として別の冊の原画像から例を選び、回収と筆画への影響を検証する。イラストは除外し、原因不明の曲線も正解として数えない。今の粗い模様・狭い紙色資格・保存境界色の補正をまとめて採用しない。利用者の停止まで改善を継続する。
''')
    dump(OUT/'diagnostics.json', dict(mean_L_errors=errors, observed_band_and_gamut=measured))
    dump(OUT/'verification.json', dict(status='passed', ROIs=42, variants=84,
        max_mean_L_field_error=max(r['field_mean_error'] for r in errors), chroma_field_identical=True,
        direct_observed_source_reconstruction=True, mask_applied_fine_texture_fixed=True,
        outside_applied_changes=0, correction_limit2=True, first_pixel_zero=True,
        ZIP_PNG_count=49, zip_hashes=True, pixel_GT=False))
    dependencies = {str((probe.OUT/name).relative_to(ROOT)): sha(probe.OUT/name) for name in ['manifest.json', 'complete.json']}
    for path, _ in items: dependencies[str(path.relative_to(ROOT))] = sha(path)
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: observed coarse band, 84 variants, 49 PNG', flush=True)


if __name__ == '__main__': main()
