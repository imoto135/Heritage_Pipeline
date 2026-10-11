"""Independent source-map chroma reconstruction and comparison packaging."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_observed_paper_chroma as probe
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT = probe.ROOT
OUT = ROOT/'tmp_work/observed_paper_chroma_review_20261001'


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(probe.OUT)
    rows = json.loads((probe.OUT/'diagnostics.json').read_text()); assert len(rows) == 34
    originals = {}
    for source in probe.prior.prior.frozen.MARGIN.SOURCES + [probe.prior.prior.new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']: originals[page['page']] = page['original']
    current = None; measured = []
    for r in rows:
        rid = r['id']; native = probe.OUT/'native'/rid
        src = probe.prior.prior.OUT/'native'/rid; baseline = probe.prior.OUT/'native'/rid
        if current != r['page']: original = read(ROOT/originals[r['page']]); current = r['page']
        x, y, w, h = r['xywh']; cx, cy, cw, ch = r['context_xywh']
        context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
        crop = context[s]; base = read(native/'base.png'); value = read(native/'chroma.png')
        assert np.array_equal(read(native/'original.png'), crop)
        assert np.array_equal(base, read(baseline/'guide32.png'))
        extra = read(src/'newly_applied_context.png', True); applied = read(src/'applied8_context.png', True)
        known = read(src/'reference_context.png', True)
        assert not np.any(np.any(value != base, axis=2)&~extra[s])
        if extra.any():
            lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            sx = np.load(src/'source_x.npy', allow_pickle=False); sy = np.load(src/'source_y.npy', allow_pickle=False)
            selected = (sx >= 0)&(sy >= 0)
            assert known[sy[selected], sx[selected]].all()
            mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)
            low = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), 6)/np.maximum(mass[:, :, None], 1e-6)
            residual = lab[:, :, 1:]-low[:, :, 1:]
            center = residual[known].mean(axis=0, dtype=np.float64).astype(np.float32)
            delta = np.zeros_like(lab)
            delta[selected, 1:] = np.clip(residual[sy[selected], sx[selected]]-center,
                -np.array([1.5, 2.0], np.float32), np.array([1.5, 2.0], np.float32))
            distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
            delta *= (np.clip((distance-1)/4., 0, 1)*applied)[:, :, None]
            assert np.array_equal(delta, np.load(native/'chroma_delta_Lab.npy', allow_pickle=False))
            original_field = np.load(baseline/'guide32/result_field_Lab.npy', allow_pickle=False)
            field = np.load(native/'result_field_Lab.npy', allow_pickle=False)
            assert np.array_equal(field, original_field+delta) and np.array_equal(field[:, :, 0], original_field[:, :, 0])
            converted = np.rint(np.clip(cv2.cvtColor(field[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
            assert np.array_equal(value[extra[s]], converted[extra[s]])
        else: assert np.array_equal(value, base)
        measured.append(dict(id=rid, changed=int(np.any(value != base, axis=2).sum()),
            maximum_ab=r['maximum_delta_ab']))
        print('Verified observed chroma:', rid, flush=True)
    items = []
    for name, ids, title, notes in [
        ('01_spot_color_texture.png', ['unseen_brown_spot'], '観測された紙模様の明度と色を比較',
         ['同じ明度の場に小さな色模様を追加した。見た目の差は小さく、色塊・角張った縁は残る。']),
        ('02_other_components.png', ['unseen_fold', 'brown_false_red_new'], '他の新規受理成分への影響',
         ['色だけを加え、以前の補間画素は保持。折れと広い茶色い紙面の問題を解決したとは扱わない。']),
        ('03_controls.png', ['unseen_dense_text', 'unseen_seal', 'new_uncertain_trace'], '本文・印影・不確かな跡',
         ['この3ROIは前回と完全一致。全34ROIを保存。'])]:
        stack([probe.OUT/'comparisons'/(rid+'.png') for rid in ids], OUT/name, title, notes)
        items.append((OUT/name, 'overview/'+name))
    items += [(probe.OUT/'comparisons'/(r['id']+'.png'), 'all_ROIs/'+r['id']+'.png') for r in rows]
    archive_path = OUT/'observed_chroma_comparisons.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == 37 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    (OUT/'report.md').write_text('''# 固定した明度模様へ観測紙面の色模様を追加

2026-10-01。**明度だけの模様に観測紙面の色の揺らぎを加えたが、見た目の差は小さい。斑点の平坦な色塊と角張った縁は解決しないため、追加採用は保留する。**

[斑点の比較PNG](01_spot_color_texture.png)、[他の成分](02_other_components.png)、[対照](03_controls.png)、[全37PNG ZIP](observed_chroma_comparisons.zip)。原画像を並べ、全34ROIを保存した。DDRM再推論・学習はなし。

## 観測値を分けた条件

前回の[分布紙色σ32](../distributed_paper_guidance_review_20261001/report.md)を比較の土台にし、マスク、参照資格、受理成分、明度場、明度模様の元画素座標を固定。既知紙面だけをσ6で平滑化したa/b差分を、保存された明度模様と同じ元画素から移した。参照全体の差分平均を差し引き、aを±1.5、bを±2に限定して既存の距離羽根で加えた。原損傷・DDRMの色や乱数は使わない。境界8組で新規受理した成分だけへ合成し、以前の補間画素は完全保持した。薄いコアの回収とは組み合わせていない。

斑点の既知参照を9px角で侵食した105,659画素のL高周波標準偏差は約0.663で、前回の補間成分内部は約0.569だった。参照パッチには既に明度模様があり、前回の512候補の残差標準偏差中央値は約1.196。**現状を単に「紙模様を全く移せていない」とは診断しない。** a/b高周波は約0.126／0.179と小さく、色模様追加の視覚効果も小さい。この原画像の測定は損傷GTでも品質指標でもない。[参照の測定JSON](../distributed_paper_texture_diagnostic_20261001/diagnostics.json)、[変更数・振幅](diagnostics.json)。

紙色と明度・色模様を分離した結果、模様だけを追加して固定マスクの不自然な形を隠す方針には進めない。次は、回収した薄いコアを同じ復元器に渡して、未補間の理由・残存輪郭・原因不明の曲線への影響を直接確認する。全ページの追加5成分も比較対象に含め、ROI内の成功だけで一律採用しない。

## 検証と適用範囲

参照元の全画素、正規化平滑差分と平均・クリップ・距離合成を別に再計算し、保存Lab場と画像を確認した。L場の完全一致、範囲外・以前の補間画素の一致、全非参照画素の異色置換に対する不変、入力・凍結ソース・保存出力とZIP37PNGのハッシュを確認。[検証JSON](verification.json)。RGB変換後の丸めで画像をLabに戻したL値は微差があり得るが、合成前のL場は厳密に不変。

旧14ページの土台には保存DDRM、新4ページの未補間箇所には原画像が残る。全18ページは調整に使った開発例で、画素GTと学習データとの独立性は不明。利用者の停止指示まで改善を継続する。
''')
    dump(OUT/'diagnostics.json', measured)
    dump(OUT/'verification.json', dict(status='passed', ROIs=34, L_field_identical=True,
        observed_source_known_only=True, direct_chroma_reconstruction=True, outside_extra_changes=0,
        ZIP_PNG_count=37, zip_hashes=True, pixel_GT=False))
    dependencies = {str((probe.OUT/name).relative_to(ROOT)): sha(probe.OUT/name) for name in ['manifest.json', 'complete.json']}
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: observed chroma, 34 ROIs, 37 PNG', flush=True)


if __name__ == '__main__': main()
