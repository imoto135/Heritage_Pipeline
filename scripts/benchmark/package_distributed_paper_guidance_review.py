"""Check screened fields directly and diagnose texture loss on fixed masks."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_distributed_paper_guidance as probe
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT = probe.ROOT
OUT = ROOT/'tmp_work/distributed_paper_guidance_review_20261001'


def direct_residual(field, target, applied, known, strength, guide):
    yy, xx = np.nonzero(applied); H, W = applied.shape
    residual = strength[yy, xx, None].astype(np.float64)*(field[yy, xx]-guide[yy, xx])
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = yy+dy, xx+dx; valid = (ny >= 0)&(ny < H)&(nx >= 0)&(nx < W)
        ids = np.flatnonzero(valid); ny, nx = ny[valid], nx[valid]
        inside = applied[ny, nx]; ii = ids[inside]
        residual[ii] += field[yy[ii], xx[ii]]-field[ny[inside], nx[inside]]
        boundary = known[ny, nx]; ii = ids[boundary]
        residual[ii] += field[yy[ii], xx[ii]]-target[ny[boundary], nx[boundary]]
    error = float(np.abs(residual).max()) if len(yy) else 0
    assert error < 1e-3, error
    return error


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(probe.OUT); check_bundle(probe.prior.OUT)
    rows = json.loads((probe.OUT/'diagnostics.json').read_text()); assert len(rows) == 34
    originals = {}
    for source in probe.prior.frozen.MARGIN.SOURCES + [probe.prior.new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']: originals[page['page']] = page['original']
    results = []; residuals = []; current = None
    for row in rows:
        rid = row['id']; key = row['page']; dest = probe.OUT/'native'/rid
        native = probe.prior.OUT/'native'/rid
        if current != key: original = read(ROOT/originals[key]); current = key
        x, y, w, h = row['xywh']; cx, cy, cw, ch = row['context_xywh']
        context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
        lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
        crop = context[s]; extra = read(native/'newly_applied_context.png', True)
        applied = read(native/'applied8_context.png', True); known = read(native/'reference_context.png', True)
        assert np.array_equal(read(dest/'original.png'), crop)
        base = read(dest/'base.png'); assert np.array_equal(base, read(native/'pairs8.png'))
        measurements = {}
        distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
        core = extra[s]&(distance[s] >= 6)
        for sigma in probe.SIGMAS:
            mode = 'guide'+str(sigma); md = dest/mode; value = read(dest/(mode+'.png'))
            assert not np.any(np.any(value != base, axis=2)&~extra[s])
            if extra.any():
                fields = np.load(md/'screened_fields_Lab.npz', allow_pickle=False)
                mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
                guide = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), sigma)/np.maximum(mass[:, :, None], 1e-6)
                strength = np.where(mass >= .05, mass/256., 0).astype(np.float32)*applied
                assert np.array_equal(guide, np.load(md/'guide_Lab.npy', allow_pickle=False))
                assert np.array_equal(mass, np.load(md/'reference_mass.npy', allow_pickle=False))
                assert np.array_equal(strength, np.load(md/'screening_strength.npy', allow_pickle=False))
                weight = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)
                smooth = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), 6)/np.maximum(weight[:, :, None], 1e-6)
                for name, target in [('boundary_raw', lab), ('boundary_smooth', smooth)]:
                    residuals.append(dict(id=rid, sigma=sigma, field=name,
                        residual=direct_residual(fields[name], target, applied, known, strength, guide)))
                bw = np.exp(-(np.maximum(distance-1, 0)/8.)**2)*applied
                seam = np.load(native/'texture_seam.npy', allow_pickle=False)
                combined = fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*bw[:, :, None]
                combined[:, :, 0] += seam*np.clip((distance-1)/4., 0, 1)*applied
                assert np.array_equal(combined, np.load(md/'result_field_Lab.npy', allow_pickle=False))
                converted = np.rint(np.clip(cv2.cvtColor(combined[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
                assert np.array_equal(value[extra[s]], converted[extra[s]])
            else: assert np.array_equal(value, base)
        for mode in ['original', 'base', 'guide16', 'guide32']:
            value = read(dest/(mode+'.png')); ll = cv2.cvtColor(value.astype(np.float32)/255, cv2.COLOR_BGR2LAB)[:, :, 0]
            hp = ll-cv2.GaussianBlur(ll, (0, 0), 1.2)
            measurements[mode] = dict(core_pixels=int(core.sum()),
                core_L_HP_std=float(hp[core].std()) if core.any() else None,
                median_core_L=float(np.median(ll[core])) if core.any() else None)
        texture = None
        if extra.any():
            donors = json.loads((native/'donors.json').read_text())
            pp, _ = probe.prior.frozen.prior.donors(lab, known)
            seam = np.load(native/'texture_seam.npy', allow_pickle=False)
            texture = dict(candidate_patches=len(donors['patch_xy']),
                candidate_std_quantiles=np.quantile(pp.std(axis=(1, 2)), [0, .25, .5, .75, 1]).tolist() if pp is not None else [],
                saved_seam_std_newly_applied=float(seam[extra].std()))
        results.append(dict(id=rid, newly_applied=row['newly_applied_pixels'], measurements=measurements, texture=texture))
        print('Verified screened paper:', rid, flush=True)
    specifications = [
        ('01_brown_spot.png', ['unseen_brown_spot'], '斑点の疎な境界参照を補う',
         ['周囲の観測紙色を加えると暗い塊は少し薄くなる。平坦さと角張った縁は残る。', 'マスク・参照資格・明度模様は固定。σ32を完成形として採用しない。']),
        ('02_other_changes.png', ['unseen_fold', 'brown_false_red_new'], '新規受理した他の成分',
         ['旧補間画素は完全保持。変化は小さく、折れの塗り跡は解決していない。']),
        ('03_controls.png', ['unseen_dense_text', 'unseen_seal', 'new_uncertain_trace'], '本文・印影・不確かな跡の対照',
         ['両σ条件ともこの対照を変更しない。新規受理した成分だけの比較。'])]
    items = []
    for name, ids, title, notes in specifications:
        stack([probe.OUT/'comparisons'/(rid+'.png') for rid in ids], OUT/name, title, notes)
        items.append((OUT/name, 'overview/'+name))
    items += [(probe.OUT/'comparisons'/(r['id']+'.png'), 'all_ROIs/'+r['id']+'.png') for r in rows]
    archive_path = OUT/'distributed_paper_comparisons.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == 37 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    spot = next(r for r in results if r['id'] == 'unseen_brown_spot')
    table = '\n'.join(f'| {mode} | {v["median_core_L"]:.3f} | {v["core_L_HP_std"]:.4f} |' for mode, v in spot['measurements'].items())
    report = f'''# 固定マスクで周囲の紙色を分布参照

2026-10-01。**隣接参照が偏った斑点に周囲の観測紙色を加えると、暗い平坦な塊は少し薄くなった。ただし角張った縁と紙模様不足が残り、自然な復元には届かない。**

[斑点の比較PNG](01_brown_spot.png)、[折れと茶色い紙面](02_other_changes.png)、[対照](03_controls.png)、[全37PNG ZIP](distributed_paper_comparisons.zip)。すべて原画像を並べた。DDRMの追加推論・学習はない。

## 条件と範囲

[境界16→8組の試作](../thin_core_reference_gate_review_20261001/report.md)で新規受理した成分だけを変更した。マスク・参照画素資格・受理範囲・保存された観測明度模様を固定し、以前から補間した画素は完全保持する。参照紙面だけの正規化Gaussian σ16／32を内部の弱い紙色項とし、4近傍の調和補間に加えた。強さは参照密度／256、参照密度.05未満は0。原損傷・非参照画素・DDRMの色は使わない。隣接の直接／σ6平滑境界と境界8pxの合成は従来どおり。全34ROIを確認した。薄いコアの変更とは組み合わせていない。

## 斑点と残る問題

新規5,081画素の斑点は、以前12組・異なる9画素だけの直接境界に依存していた。周囲の既知紙面を内部へ反映することで色が明るくなるが、復元の形は同じ固定マスクのまま。周囲との色ムラと段差、平坦な模様を同時には解決しない。下表の明度・高周波標準偏差は診断で、損傷除去精度や自然さの正解得点ではない。原画像の値には損傷も含む。

| 条件 | 成分内部L中央値 | 内部L高周波標準偏差 |
| --- | ---: | ---: |
{table}

明度模様候補と保存された模様の振幅も[全診断](diagnostics.json)へ保存した。紙面参照の資格が細かな暗い模様を除き、全面既知24px角パッチという条件も候補を滑らかに偏らせる可能性がある。次は紙色項とは分離して、紙面の模様を参照する資格・範囲と実際の振幅を検証する。模様を損傷から直接引き継ぐ方法にはしない。σ16／32の一律採用は保留する。

折れの新規225画素と広い茶色い紙面124画素への効果は小さい。本文・印影・原因不明の跡など31ROIは追加補間0で前回と完全一致。対照が対象外だったことを自動的なインク識別の成功とは数えない。旧14ページの範囲外は保存DDRM、新4ページの範囲外は原画像。新4冊も条件調整に使った開発例で、画素GTと学習データからの独立性は不明。

## 検証

入力・保存出力・凍結ソースのハッシュ、原画像一致、変更範囲外と以前の補間画素の完全一致を確認。既知参照だけからの分布参照と強さを再計算し、行列を組み立てずに4近傍差と内部項から残差を直接確認した。保存のLab合成式・観測明度模様の一致、定数紙面、全非参照画素を異色にした場の不変、ZIP37PNGの内容ハッシュも確認。[検証JSON](verification.json)。利用者の停止指示まで改善を継続する。
'''
    (OUT/'report.md').write_text(report)
    dump(OUT/'diagnostics.json', dict(regions=results, residuals=residuals))
    dump(OUT/'verification.json', dict(status='passed', ROIs=34, variants=68,
        directly_checked_residual_max=max(r['residual'] for r in residuals), old_applied_preserved=True,
        outside_newly_applied_changes=0, guides_from_known_paper_only=True, texture_fixed=True,
        ZIP_PNG_count=37, zip_hashes=True, pixel_GT=False))
    dependencies = {str((folder/name).relative_to(ROOT)): sha(folder/name)
        for folder in [probe.OUT, probe.prior.OUT] for name in ['manifest.json', 'complete.json']}
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: distributed tone, 68 variants, 37 PNG', flush=True)


if __name__ == '__main__': main()
