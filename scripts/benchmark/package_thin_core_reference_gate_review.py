"""Independently check two separate ablations and show unclassified additions."""
import gc
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_thin_margin_components as thin
import probe_paper_boundary_pair_gate_v2 as pairs
from build_npj_v6_comparison import dump, read, sha
from check_paper_margin_defect_ddrm import figure
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_blending_review import check_bundle
from package_paper_seam_texture_review import verify_texture

ROOT = thin.ROOT
OUT = ROOT/'tmp_work/thin_core_reference_gate_review_20261001'


def show_mask(original, before, added, path, title):
    old = original.copy(); old[before] = (0, 200, 255)
    new = old.copy(); new[added] = (255, 120, 0)
    figure([original, old, new], ['原画像', '黄＝従来の編集候補', '青＝今回の追加候補'], path,
           title, f'追加 {int(added.sum()):,}画素。正解数ではない。復元なし。',
           '太さ条件だけを変更。青は追加編集候補。原因不明の筆画状の跡も含む。')


def boundary_positions(component, known):
    """List actual adjacent reference pairs, retaining duplicates for pair count."""
    yy, xx = np.nonzero(component); H, W = known.shape; points = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = yy+dy, xx+dx
        valid = (ny >= 0)&(ny < H)&(nx >= 0)&(nx < W)
        ny, nx = ny[valid], nx[valid]; good = known[ny, nx]
        points.extend(zip(nx[good].tolist(), ny[good].tolist()))
    return np.array(points, dtype=np.int32).reshape(-1, 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_bundle(thin.OUT); check_bundle(pairs.OUT)
    tr = json.loads((thin.OUT/'diagnostics.json').read_text())
    pr = json.loads((pairs.OUT/'diagnostics.json').read_text())
    rows = {r['id']: r for r in pr}; assert len(rows) == len(pr) == 34
    mask_rows = {r['id']: r for r in tr['regions']}
    residuals = []; proofs = []; components = []; component_images = []; boundary_stats = []
    checked = []; page_checks = []; items = []
    for source in thin.prior.MARGIN.SOURCES + [thin.new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key = page['page']; original = read(ROOT/page['original'])
            folder = thin.OUT/'pages'/key
            before = read(folder/'baseline_edit.png', True); after = read(folder/'edit.png', True)
            added = read(folder/'added.png', True); domain = read(folder/'domain.png', True)
            guard = read(folder/'guard.png', True); core = read(folder/'new_core.png', True)
            strict = read(folder/'strict_core.png', True); relaxed = read(folder/'relaxed_core.png', True)
            assert np.array_equal(after, before|added) and not np.any(before&added)
            assert np.array_equal(core, relaxed&~strict) and not np.any(strict&~relaxed)
            assert not np.any(added & (~domain|guard))
            halo = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
            assert not np.any(added&~halo)
            notes = json.loads((folder/'components.json').read_text())
            new_notes = [n for n in notes if n['newly_accepted']]
            for n in new_notes:
                assert 3.5 <= n['peak'] < 6 and n['area'] >= 128 and n['seeds'] >= 32
                x, y, w, h = n['xywh']; H, W = before.shape
                x0, y0, x1, y1 = max(0, x-45), max(0, y-45), min(W, x+w+45), min(H, y+h+45)
                s = (slice(y0, y1), slice(x0, x1)); name = key+'_'+str(n['id'])+'.png'
                show_mask(original[s], before[s], added[s], OUT/'components'/name,
                          key+' · 追加成分 '+str(n['id']))
                components.append(dict(page=key, component=n, review='unclassified appearance; no damage/ink GT'))
                component_images.append(OUT/'components'/name)
            show_mask(original, before, added, OUT/'whole_pages'/(key+'.png'), key+' · ページ全体')
            items.append((OUT/'whole_pages'/(key+'.png'), 'whole_pages/'+key+'.png'))
            assert mask_rows[key+'_whole']['added_pixels'] == int(added.sum())
            page_checks.append(dict(page=key, additional_pixels=int(added.sum()), new_components=len(new_notes)))
            for roi in page['rois']:
                rid = roi['id']; r = rows[rid]; x, y, w, h = r['xywh']
                rs = (slice(y, y+h), slice(x, x+w)); crop = original[rs]
                assert np.array_equal(crop, read(thin.OUT/'native'/rid/'original.png'))
                assert mask_rows[rid]['added_pixels'] == int(added[rs].sum())
                show_mask(crop, before[rs], added[rs], OUT/'thin_masks'/(rid+'.png'), r['title'])
                items.append((OUT/'thin_masks'/(rid+'.png'), 'thin_masks/'+rid+'.png'))
                native = pairs.OUT/'native'/rid
                cx, cy, cw, ch = r['context_xywh']; cs = (slice(cy, cy+ch), slice(cx, cx+cw))
                s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w)); context = original[cs]
                known = read(native/'reference_context.png', True); active = read(native/'eligible_context.png', True)
                old = read(native/'old_applied_context.png', True); applied = read(native/'applied8_context.png', True)
                extra = read(native/'newly_applied_context.png', True)
                base = read(native/'base.png'); value = read(native/'pairs8.png'); edit = read(native/'edit.png', True)
                assert np.array_equal(read(native/'original.png'), crop)
                assert not np.any(applied&~active) and not np.any(known&active)
                assert np.array_equal(extra[s], applied[s]&~old[s]) and not np.any(old[s]&~applied[s])
                assert not np.any(np.any(value != base, axis=2)&~extra[s])
                assert np.array_equal(value[old[s]], base[old[s]])
                assert not np.any(np.any(value != crop, axis=2)&~edit)
                assert r['newly_applied_pixels'] == int(extra[s].sum())
                if extra.any():
                    lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                    fields = np.load(native/'harmonic_fields_Lab.npz', allow_pickle=False)
                    mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)
                    smooth = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), 6)/np.maximum(mass[:, :, None], 1e-6)
                    for name, boundary in [('boundary_raw', lab), ('boundary_smooth', smooth)]:
                        error, used = verify_field(fields[name], applied, known, boundary)
                        assert np.array_equal(used, read(native/'used_context.png', True))
                        residuals.append(dict(id=rid, field=name, residual=error))
                    donors = json.loads((native/'donors.json').read_text())
                    if donors['components']: proofs.append(dict(id=rid, **verify_texture(lab, known, applied, native)))
                    else:
                        assert len(donors['patch_xy']) < 4
                        assert not np.load(native/'texture_seam.npy', allow_pickle=False).any()
                    distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
                    weight = np.exp(-(np.maximum(distance-1, 0)/8.)**2)*applied
                    combined = fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:, :, None]
                    combined[:, :, 0] += np.load(native/'texture_seam.npy', allow_pickle=False)*np.clip((distance-1)/4., 0, 1)*applied
                    assert np.array_equal(combined, np.load(native/'result_field_Lab.npy', allow_pickle=False))
                    converted = np.rint(np.clip(cv2.cvtColor(combined[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
                    assert np.array_equal(value[extra[s]], converted[extra[s]])
                    count, labels, stats, _ = cv2.connectedComponentsWithStats(extra.astype(np.uint8), connectivity=4)
                    for i in range(1, count):
                        target = labels == i; points = boundary_positions(target, known)
                        if not len(points): continue
                        unique = np.unique(points, axis=0); ux, uy = unique[:, 0], unique[:, 1]
                        x0, y0, ww, hh, area = map(int, stats[i]); center = (x0+(ww-1)/2, y0+(hh-1)/2)
                        quadrants = (ux >= center[0]).astype(int)+2*(uy >= center[1]).astype(int)
                        boundary_stats.append(dict(id=rid, component=i, area=area, boundary_pairs=len(points),
                            distinct_reference_pixels=len(unique), reference_bbox=[int(ux.min()), int(uy.min()), int(ux.max()-ux.min()+1), int(uy.max()-uy.min()+1)],
                            component_xywh=[x0, y0, ww, hh], quadrants=np.bincount(quadrants, minlength=4).tolist(),
                            boundary_L_min=float(lab[uy, ux, 0].min()), boundary_L_max=float(lab[uy, ux, 0].max()),
                            known_reference_pixels=int(known.sum()), donor_patch_count=len(donors['patch_xy'])))
                items.append((pairs.OUT/'comparisons'/(rid+'.png'), 'pair_gate/'+rid+'.png'))
                checked.append(rid); print('Verified separate factors:', rid, flush=True)
            del original, before, after, added, domain, guard, core, strict, relaxed
            gc.collect()
    assert len(page_checks) == 18 and len(checked) == 34 and len(components) == 8
    for path in component_images: items.append((path, 'new_components/'+path.name))
    overview = []
    specifications = [
        ('01_thin_crack_mask.png', [OUT/'thin_masks'/('unseen_thin_crack.png'), OUT/'thin_masks'/('unseen_brown_spot.png')],
         '細いコアの回収：分類だけを変更', ['薄い裂け状のROIに1,207画素追加。復元はこの因子でまだ比較していない。']),
        ('02_whole_page_unknown_traces.png', component_images[:5], '固定ROIの外で増えた原因不明の曲線',
         ['200021712のページ全体に3,331画素追加。筆画状の跡を損傷の正解として数えない。', '34ROIの対照が不変でも、全ページに安全な条件とは言えない。']),
        ('03_reference_pair_fill.png', [pairs.OUT/'comparisons'/(k+'.png') for k in ['unseen_brown_spot', 'unseen_fold', 'brown_false_red_new']],
         '参照境界16→8組：マスクを固定', ['斑点5,081画素などを追加補間。斑点の平坦な色塊・薄い輪郭は残る。', '太さ条件の因子とは組み合わせていない。旧14は範囲外に保存DDRM、新4は原画像。']),
        ('04_controls.png', [OUT/'thin_masks'/(k+'.png') for k in ['unseen_dense_text', 'unseen_seal', 'new_uncertain_trace']],
         '本文・印影・既存の不確かな跡', ['この3ROIの細いコア追加0。ページ全体の新しい曲線も併せて判断する。'])]
    for name, paths, title, notes in specifications:
        stack(paths, OUT/name, title, notes); overview.append(name); items.append((OUT/name, 'overview/'+name))
    archive_path = OUT/'separate_factor_comparisons.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(items) and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    table = '\n'.join(f'| {r["id"]} | {mask_rows[r["id"]]["added_pixels"]:,} | {r["newly_applied_pixels"]:,} |' for r in pr)
    report = f'''# 細いコアと紙面参照の受理条件を別々に検証

2026-10-01。**細い裂けの検出は増えたが、原因不明の筆画状の曲線もページ全体で増えた。境界対数を緩めると斑点を補間できるが、平坦な色塊が残る。両条件とも一律採用せず、改善を継続する。**

## 比較PNG

- [細い裂けの分類](01_thin_crack_mask.png)。原画像・固定の輪郭マスク・細いコア追加。黄は従来の編集候補、青は今回の追加。
- [固定ROIの外に増えた曲線](02_whole_page_unknown_traces.png)。原画像を全成分に並べた。損傷かインクかは確定していない。
- [固定マスクの紙面補間](03_reference_pair_fill.png)。原画像・境界16組・境界8組。同じ補間と観測明度模様。
- [本文・印影・不確かな跡の対照](04_controls.png)、[全{len(items)}PNG ZIP](separate_factor_comparisons.zip)。全18ページの分類、全34ROIの分類、全34ROIの補間、全8追加成分と概要4枚。

## 独立した二つの変更

太さ因子はコアのピーク半径だけを6→3.5pxに変更した。面積128、茶色種32、サイズ、上部余白境界、安全領域・保護、6px近傍と明度差2の成長条件は従来どおり。全ページで処理し、ROI名で対象を絞らない。18ページのうち2ページに追加：200017458は1,335画素、200021712は3,331画素。他16ページは追加0。薄い裂けROIは1,207画素増え、隣の斑点は128画素増えた。旧22ROIは追加0だが、旧ページのROI外に5個の筆画状成分が増えた。**対照ROIの不変を全ページの安全性の根拠にはしない。** この段階は分類だけで、新しい薄いコアの復元はまだない。

参照因子はマスク・参照画素と数値補間を固定し、隣接参照境界16→8組だけを変えた。既に補間した画素は完全保持。追加は斑点5,081画素、折れ225画素、茶色い紙面124画素。ほか31ROIは追加0。斑点付近の黒い点は保持されるが、暗い暖色の平坦な塊と残存縁が見える。数学的に埋められることと自然に馴染むことは分けて扱う。斑点の境界位置・明度を[参照診断](diagnostics.json)に保存した。

| ROI | 太さ因子の追加編集 | 参照因子の追加補間 |
| --- | ---: | ---: |
{table}

二つは別実験であり合成していない。旧14ページの参照因子は保存された輪郭回収＋DDRM出力を土台に、新4ページは紙面補間＋未補間の原画素を土台にした。DDRMの再推論・学習はない。新4冊を条件調整に使ったので開発側へ移した。[条件固定の初回結果](../unseen_margin_validation_20261001/review/report.md)は維持。画素GTも学習データからの独立性も不明。

## 確認と次の改善

保存ソース・入力・出力のハッシュ、全18ページの編集包含・保護・成長範囲、全34ROIの追加数を確認した。補間の新規範囲外と旧補間画素の完全一致、編集外の原画素一致、直接4近傍残差、観測明度模様の参照元と保存Lab合成式、ZIPの内容ハッシュを別に確認。[確認JSON](verification.json)。細いコア試作の中間図にある定型の青＝保護の凡例はこの実験の表示に合わないため、ここでは保存マスクから全図を再描画し、青＝追加編集に統一した。計算マスクは変更していない。

次は、疎な参照境界の偏りと局所紙色・模様を測定し、斑点の平坦な塊を減らす。薄いコアは全ページの不確かな曲線を保持して検証を続け、一律には採用しない。利用者の停止まで継続する。
'''
    (OUT/'report.md').write_text(report)
    dump(OUT/'diagnostics.json', dict(pages=page_checks, new_components=components, boundary_coverage=boundary_stats,
         harmonic_residuals=residuals, observed_texture=proofs))
    dump(OUT/'verification.json', dict(status='passed', pages=18, ROIs=34, independently_composed_fields=True,
         discrete_residual_max=max(r['residual'] for r in residuals), old_applied_unchanged=True,
         outside_extra_changes=0, outside_edit_changes=0, zip_PNG_count=len(items), zip_hashes=True,
         factors_combined=False, pixel_GT=False, global_thin_gate_adopted=False, DDRM_recomputed=False))
    dependencies = {str((folder/name).relative_to(ROOT)): sha(folder/name)
                    for folder in [thin.OUT, pairs.OUT] for name in ['manifest.json', 'complete.json']}
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
         outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: separate factors,', len(items), 'PNG', flush=True)


if __name__ == '__main__': main()
