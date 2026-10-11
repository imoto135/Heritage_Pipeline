"""Independent chroma-reference gate, fixed texture and fallback checks."""
import gc
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_paper_color_reference_gate_v2 as probe
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_blending_review import check_bundle

ROOT = probe.ROOT
OUT = ROOT/'tmp_work/paper_color_reference_gate_review_20261001'


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(probe.OUT)
    dependencies = json.loads((probe.OUT/'final_dependencies.json').read_text())
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    rows = json.loads((probe.OUT/'diagnostics.json').read_text()); assert len(rows) == 42
    pages = []
    for source in probe.prior.prior.mask.prior.MARGIN.SOURCES+[probe.prior.prior.mask.new.INPUT]:
        pages += json.loads((source/'baseline_review.json').read_text())['cases']
    residuals = []; boundary_gaps = []; checked = []
    for pg in pages:
        key = pg['page']; original = read(ROOT/pg['original'])
        median_lab = cv2.cvtColor(cv2.medianBlur(original, 3).astype(np.float32)/255, cv2.COLOR_BGR2LAB)
        paper = probe.prior.prior.mask.prior.MARGIN.first.paper_estimate(median_lab)
        differences = np.abs(median_lab[:, :, 1:]-paper[:, :, 1:])
        full_edit = read(probe.prior.OUT/'pages'/key/'near8/edit.png', True)
        for r in [r for r in rows if r['page'] == key]:
            rid = r['id']; native = probe.OUT/'native'/rid; baseline = probe.prior.OUT/'native'/rid/'near8'
            x, y, w, h = r['xywh']; cx, cy, cw, ch = r['context_xywh']
            cs = (slice(cy, cy+ch), slice(cx, cx+cw)); s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
            context = original[cs]; crop = context[s]
            known = read(native/'baseline_reference_context.png', True); old_applied = read(native/'baseline_applied_context.png', True)
            base = read(native/'base.png'); edit = read(native/'edit.png', True)
            assert np.array_equal(crop, read(native/'original.png'))
            assert np.array_equal(base, read(baseline/'output.png'))
            assert np.array_equal(edit, full_edit[cs][s])
            assert np.array_equal(differences[cs], np.load(native/'reference_chroma_difference.npy', allow_pickle=False))
            source_info = json.loads((native/'texture_source.json').read_text()); texture_source = ROOT/source_info['path'] if source_info['path'] else None
            if texture_source:
                seam = np.load(texture_source/'texture_seam.npy', allow_pickle=False)
                sx = np.load(texture_source/'source_x.npy', allow_pickle=False); sy = np.load(texture_source/'source_y.npy', allow_pickle=False)
                selected = sx >= 0; assert known[sy[selected], sx[selected]].all()
                assert not seam[~old_applied].any()
            else: seam = np.zeros(old_applied.shape, np.float32); assert not old_applied.any()
            dist = cv2.distanceTransform(old_applied.astype(np.uint8), cv2.DIST_L2, 5)
            bw = np.exp(-(np.maximum(dist-1, 0)/8.)**2)*old_applied
            lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            for mode, thresholds in probe.LEVELS.items():
                md = native/mode; strict = read(md/'reference_context.png', True); applied = read(md/'applied_context.png', True)
                value = read(md/'output.png')
                assert np.array_equal(strict, known&np.all(differences[cs] <= np.array(thresholds, np.float32), axis=2))
                assert not np.any(applied&~old_applied)
                assert not np.any(np.any(value != base, axis=2)&~applied[s])
                assert not np.any(np.any(value != crop, axis=2)&~edit)
                stats = r['color_reference_diagnostics'][mode]
                assert stats['tone_updated_pixels'] == int(applied[s].sum())
                assert stats['baseline_retained_due_insufficient_reference'] == int((old_applied[s]&~applied[s]).sum())
                if (md/'harmonic_fields_Lab.npz').exists():
                    fields = np.load(md/'harmonic_fields_Lab.npz', allow_pickle=False)
                    mass = cv2.GaussianBlur(strict.astype(np.float32), (0, 0), 6)
                    smooth = cv2.GaussianBlur(np.where(strict[:, :, None], lab, 0), (0, 0), 6)/np.maximum(mass[:, :, None], 1e-6)
                    for name, target in [('boundary_raw', lab), ('boundary_smooth', smooth)]:
                        error, used = verify_field(fields[name], applied, strict, target)
                        assert np.array_equal(used, read(md/'used_context.png', True))
                        residuals.append(dict(id=rid, mode=mode, field=name, residual=error))
                    field = fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*bw[:, :, None]
                    field[:, :, 0] += seam*np.clip((dist-1)/4., 0, 1)*old_applied
                    assert np.array_equal(field, np.load(md/'result_field_Lab.npy', allow_pickle=False))
                    converted = np.rint(np.clip(cv2.cvtColor(field[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
                    assert np.array_equal(value[applied[s]], converted[applied[s]])
                else: assert np.array_equal(value, base)
            yy, xx = np.nonzero(old_applied); H, W = old_applied.shape
            counts = dict(known_paper_pairs=0, preserved_outside_edit_unqualified_pairs=0, unapplied_edit_pairs=0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = yy+dy, xx+dx; valid = (ny >= 0)&(ny < H)&(nx >= 0)&(nx < W)
                ny, nx = ny[valid], nx[valid]; outside = ~old_applied[ny, nx]
                ny, nx = ny[outside], nx[outside]
                counts['known_paper_pairs'] += int(known[ny, nx].sum())
                counts['preserved_outside_edit_unqualified_pairs'] += int((~known[ny, nx]&~full_edit[cs][ny, nx]).sum())
                counts['unapplied_edit_pairs'] += int((~known[ny, nx]&full_edit[cs][ny, nx]).sum())
            boundary_gaps.append(dict(id=rid, **counts)); checked.append(rid)
            print('Verified fixed-texture color gate:', rid, flush=True)
        del original, median_lab, paper, differences, full_edit; gc.collect()
    # The first attempt only wrote 8 unchanged ROIs before source resolution failed.
    first = ROOT/'tmp_work/paper_color_reference_gate_pilot_20261001'; identical = 0
    for path in (first/'native').rglob('*'):
        if path.is_file():
            assert sha(path) == sha(probe.OUT/path.relative_to(first)); identical += 1
    items = []
    for name, ids, title, notes in [
        ('01_holes_and_crack.png', ['unseen_thin_crack', 'new_margin_hole', 'hole_edge_new'], '紙色参照の資格を絞った比較',
         ['全体の改善は小さく、平坦な塗り跡は残る。a1/b2では裂けの参照が足りず前回を保持。']),
        ('02_unknown_curves.png', ['200021712_00082_1_component_28', '200021712_00082_1_component_36'], '参照不足も残す',
         ['厳しい色差で一部の成分を再補間できない。画像不変を改善成功と数えない。']),
        ('03_stamp_and_fold.png', ['new_brown_stamp', 'unseen_fold'], '印影と折れの対照',
         ['既存の編集内だけで紙色が変わる。印影の旧消失を今回の成功の根拠にしない。']),
        ('04_controls.png', ['unseen_seal', 'unseen_dense_text', 'new_uncertain_trace'], '本文・印影・不確かな跡',
         ['この3ROIは前回と完全一致。全42ROIの結果を省略せず保存。'])]:
        stack([probe.OUT/'comparisons'/(rid+'.png') for rid in ids], OUT/name, title, notes)
        items.append((OUT/name, 'overview/'+name))
    items += [(probe.OUT/'comparisons'/(r['id']+'.png'), 'all_ROIs/'+r['id']+'.png') for r in rows]
    archive_path = OUT/'paper_color_reference_comparisons.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == 46 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    table = '\n'.join(f'| {r["id"]} | '+ ' | '.join(str(r['color_reference_diagnostics'][m][k]) for m in probe.LEVELS for k in ['tone_updated_pixels', 'baseline_retained_due_insufficient_reference'])+' |' for r in rows)
    (OUT/'report.md').write_text(f'''# 固定マスク・固定明度模様で紙色参照を絞る

2026-10-01。**紙色参照のa/b色差を絞っても塗り跡の改善は小さく、厳しい条件では裂けなどの参照不足が増える。参照を絞る条件へ置き換えず、前回を比較基準に維持する。**

[虫食いと裂けの比較PNG](01_holes_and_crack.png)、[参照不足の成分](02_unknown_curves.png)、[印影・折れ](03_stamp_and_fold.png)、[対照](04_controls.png)、[全46PNG ZIP](paper_color_reference_comparisons.zip)。全42ROI・84条件に原画像を並べた。

## 紙色と模様の参照を分ける

薄いコアと近接8px補完のマスク、以前の適用成分、保存された観測明度模様と元画素座標を固定。紙色補間に使う参照だけ、元の局所紙色とのa/b色差2/4から1.5/3と1/2へ絞った。他の参照条件は変更しない。元と同じ全ページ中央値3pxのLab・紙色推定で資格を計算し、4近傍の直接／σ6平滑境界から補間した。明度模様は元の資格を満たす保存紙面からのままで、新しい狭い紙色資格とは区別する。参照不足になった成分は前回出力を保持し、消えた損傷を原画像へ戻したようには表示しない。

穴2例や裂けの補修色は少し変わるが、平坦な塗り跡・薄い残存輪郭は解決しない。a1/b2では裂け1,522画素への再補間ができず全て前回を保持。原因不明の追加成分でも320／1,011／563画素などが参照不足になる。全体に紙面参照があっても直接境界に足りるとは限らない。原画像との差・明るさ・再補間面積を正解や品質指標にはしない。

| ROI | a1.5/b3再補間 | 同参照不足保持 | a1/b2再補間 | 同参照不足保持 |
| --- | ---: | ---: | ---: | ---: |
{table}

## 確認と次の課題

全入力と保存場・凍結ソースのハッシュ、資格差分の全ページ再計算、参照の包含、固定明度模様の元画素と合成、直接4近傍残差、全84条件の再補間外・編集外一致、非参照異色置換、ZIP46PNGを確認。[検証JSON](verification.json)、[境界参照不足の内訳](diagnostics.json)。初回は旧模様の参照先解決で8ROIの後に停止し、従前の模様フォルダを参照先に加えたv2で完了。初回のネイティブ{identical}ファイルはv2とハッシュ一致、数値式は変更していない。

次は、元の参照資格を満たさず境界条件に使っていない、編集範囲外の保存画素と補修色の間の段差を診断する。マスクを広げず、疑わしい画素を紙面の正解とはせず、限られた境界帯の色連続性だけを別因子として試す。暗い損傷を内部へ引き継ぐ反例も保持する。

旧14ページの固定ROIの範囲外には保存DDRM、新4ページと追加成分の範囲外には原画像が残る。DDRM再推論・学習はない。全例は開発例、物理原因・画素GT・学習データ独立性は未確定。利用者の停止まで改善を継続する。
''')
    dump(OUT/'diagnostics.json', dict(residuals=residuals, boundary_gaps=boundary_gaps, initial_native_hashes_identical=identical))
    dump(OUT/'verification.json', dict(status='passed', pages=18, ROIs=42, variants=84,
        fixed_observed_texture_verified=True, color_reference_subset=True, fallback_verified=True,
        direct_residual_max=max(r['residual'] for r in residuals), outside_reapplied_changes=0,
        outside_edit_changes=0, ZIP_PNG_count=46, zip_hashes=True, pixel_GT=False))
    deps = {str((probe.OUT/name).relative_to(ROOT)): sha(probe.OUT/name) for name in ['manifest.json', 'complete.json', 'final_dependencies.json']}
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=deps,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: color gate only, 84 variants, 46 PNG', flush=True)


if __name__ == '__main__': main()
