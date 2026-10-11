"""Check uncertain preserved-color sources and bounded boundary-only changes."""
import gc
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_preserved_paper_boundary as probe
from build_npj_v6_comparison import dump, read, sha
from check_paper_margin_defect_ddrm import figure
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_blending_review import boundary_metric, check_bundle

ROOT = probe.ROOT
OUT = ROOT/'tmp_work/preserved_paper_boundary_review_20261001'


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(probe.OUT)
    rows = json.loads((probe.OUT/'diagnostics.json').read_text()); assert len(rows) == 42
    textures = json.loads((probe.OUT/'texture_sources.json').read_text()); pages = []
    for source in probe.references.prior.prior.mask.prior.MARGIN.SOURCES+[probe.references.prior.prior.mask.new.INPUT]:
        pages += json.loads((source/'baseline_review.json').read_text())['cases']
    residuals = []; metrics = []; targets = ['hole_edge_new', 'new_margin_hole', 'unseen_thin_crack']; items = []
    for pg in pages:
        key = pg['page']; original = read(ROOT/pg['original'])
        lab_m = cv2.cvtColor(cv2.medianBlur(original, 3).astype(np.float32)/255, cv2.COLOR_BGR2LAB)
        paper = probe.references.prior.prior.mask.prior.MARGIN.first.paper_estimate(lab_m)
        drop = paper[:, :, 0]-lab_m[:, :, 0]; diff = np.abs(lab_m[:, :, 1:]-paper[:, :, 1:])
        full_edit = read(probe.BASE/'pages'/key/'near8/edit.png', True)
        gd = probe.references.prior.prior.mask.OUT/'pages'/key
        domain = read(gd/'domain.png', True); guard = read(gd/'guard.png', True)
        available = ~full_edit&domain&~guard&(lab_m[:, :, 0] >= 50)&(drop >= -6)&(drop <= 8)&np.all(diff <= [3., 6.], axis=2)
        for r in [r for r in rows if r['page'] == key]:
            rid = r['id']; native = probe.OUT/'native'/rid; baseline = probe.BASE/'native'/rid/'near8'
            x, y, w, h = r['xywh']; cx, cy, cw, ch = r['context_xywh']
            cs = (slice(cy, cy+ch), slice(cx, cx+cw)); s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w)); context = original[cs]; crop = context[s]
            base = read(native/'base.png'); applied = read(native/'applied_context.png', True)
            known = read(native/'baseline_reference_context.png', True); broad = read(native/'broad_color_context.png', True)
            assert np.array_equal(crop, read(native/'original.png')) and np.array_equal(base, read(baseline/'output.png'))
            assert np.array_equal(broad, known|available[cs]) and not np.any(broad&full_edit[cs])
            assert np.array_equal(applied, read(baseline/'applied_context.png', True))
            distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
            if applied[s].any():
                lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                old_fields = np.load(native/'baseline_fields_Lab.npz', allow_pickle=False)
                new_fields = np.load(native/'broad_fields_Lab.npz', allow_pickle=False)
                for prefix, fields, ref in [('baseline', old_fields, known), ('broad', new_fields, broad)]:
                    mass = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), 6)
                    smooth = cv2.GaussianBlur(np.where(ref[:, :, None], lab, 0), (0, 0), 6)/np.maximum(mass[:, :, None], 1e-6)
                    for name, target in [('boundary_raw', lab), ('boundary_smooth', smooth)]:
                        error, used = verify_field(fields[name], applied, ref, target)
                        assert np.array_equal(used, read(native/('used_'+prefix+'.png'), True))
                        residuals.append(dict(id=rid, source=prefix, field=name, residual=error))
                bw = np.exp(-(np.maximum(distance-1, 0)/8.)**2)*applied
                old = old_fields['boundary_smooth']+(old_fields['boundary_raw']-old_fields['boundary_smooth'])*bw[:, :, None]
                new = new_fields['boundary_smooth']+(new_fields['boundary_raw']-new_fields['boundary_smooth'])*bw[:, :, None]
                source = ROOT/textures[rid]; seam = np.load(source/'texture_seam.npy', allow_pickle=False)
                sx = np.load(source/'source_x.npy', allow_pickle=False); sy = np.load(source/'source_y.npy', allow_pickle=False)
                selected = sx >= 0; assert known[sy[selected], sx[selected]].all()
                base_field = old.copy(); base_field[:, :, 0] += seam*np.clip((distance-1)/4., 0, 1)*applied
                assert np.array_equal(base_field, np.load(native/'base_field_Lab.npy', allow_pickle=False))
                converted = np.rint(np.clip(cv2.cvtColor(base_field[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
                assert np.array_equal(base[applied[s]], converted[applied[s]])
                delta = np.clip(new-old, -probe.LIMITS, probe.LIMITS)*applied[:, :, None]
                assert np.array_equal(delta, np.load(native/'bounded_delta_Lab.npy', allow_pickle=False))
            measurement = dict(id=rid, conditions={})
            for mode in ['base']+['band'+str(radius) for radius in probe.RADII]:
                value = base if mode == 'base' else read(native/mode/'output.png')
                if mode != 'base':
                    radius = int(mode[4:]); support = applied&(distance <= 1+3*radius)
                    assert not np.any(np.any(value != base, axis=2)&~support[s])
                    if applied[s].any():
                        weight = np.exp(-(np.maximum(distance-1, 0)/radius)**2)*support
                        correction = delta*weight[:, :, None]
                        assert np.all(np.abs(correction) <= probe.LIMITS+1e-6)
                        assert np.array_equal(correction, np.load(native/mode/'correction_Lab.npy', allow_pickle=False))
                        field = base_field+correction
                        assert np.array_equal(field, np.load(native/mode/'result_field_Lab.npy', allow_pickle=False))
                        converted = np.rint(np.clip(cv2.cvtColor(field[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
                        assert np.array_equal(value[applied[s]], converted[applied[s]])
                    else: assert np.array_equal(value, base)
                assert not np.any(np.any(value != crop, axis=2)&~full_edit[cs][s])
                pairs, median = boundary_metric(value, applied[s], broad[s])
                measurement['conditions'][mode] = dict(boundary_pairs=pairs, median_deltaLab=median)
            metrics.append(measurement)
            if rid in targets:
                overlay = crop.copy(); overlay[known[s]] = (80, 210, 80); overlay[(broad&~known)[s]] = (255, 120, 0); overlay[applied[s]] = (0, 200, 255)
                figure([crop, overlay], ['原画像', '緑＝元参照・青＝追加色・黄＝補間'], OUT/'references'/(rid+'.png'),
                    r['title']+' · 色の参照元を区別', '青は原画素を残す明るい余白。薄い汚れを含む可能性がありGTではない。',
                    '青は追加編集ではなく境界補正の色参照。マスクと模様は固定。')
                items.append((OUT/'references'/(rid+'.png'), 'references/'+rid+'.png'))
                yy, xx = np.nonzero(applied[s]); x0, y0 = max(0, int(xx.min())-12), max(0, int(yy.min())-12)
                x1, y1 = min(w, int(xx.max())+13), min(h, int(yy.max())+13)
                images = [read(native/name)[y0:y1, x0:x1] for name in ['original.png', 'base.png', 'band2/output.png', 'band4/output.png']]
                figure(images, ['原画像', '前回', '境界2px帯', '境界4px帯'], OUT/'closeups'/(rid+'.png'),
                    r['title']+' · 同じ画素の拡大', '最近傍で同じ範囲を拡大。補修色差と残存縁を見る。',
                    '追加色源を清浄紙面と確定せず、有限帯の小さな色補正だけを比較。')
                items.append((OUT/'closeups'/(rid+'.png'), 'closeups/'+rid+'.png'))
            print('Verified preserved light boundary:', rid, flush=True)
        del original, lab_m, paper, drop, diff, full_edit, domain, guard, available; gc.collect()
    for name, paths, title, notes in [
        ('01_holes.png', [probe.OUT/'comparisons'/(k+'.png') for k in ['hole_edge_new', 'new_margin_hole']], '穴の境界帯だけを補正',
         ['境界色はわずかに変わるが、平坦な塗り跡・残存輪郭の改善は十分でない。']),
        ('02_crack_and_fold.png', [probe.OUT/'comparisons'/(k+'.png') for k in ['unseen_thin_crack', 'unseen_fold']], '裂け・折れの局所色',
         ['マスク・適用範囲・模様は固定。暗い薄い境界色を内部へ引き継ぐ可能性も残る。']),
        ('03_reference_sources.png', [OUT/'references'/(k+'.png') for k in targets], '追加編集と色参照を区別',
         ['緑＝元の参照、青＝追加した保存境界色、黄＝既存補間。青は編集候補ではない。']),
        ('04_controls.png', [probe.OUT/'comparisons'/(k+'.png') for k in ['unseen_seal', 'unseen_dense_text', 'new_uncertain_trace']], '本文・印影・不確かな跡',
         ['この3ROIは不変。参照色の借用がインク識別を証明するものではない。']),
        ('05_stamp_and_unknown.png', [probe.OUT/'comparisons'/(k+'.png') for k in ['new_brown_stamp', '200021712_00082_1_component_21']], '印影・原因不明の曲線を省略しない',
         ['以前の編集消失を今回の成功と扱わない。旧マスクの曖昧な曲線は残る課題。'])]:
        stack(paths, OUT/name, title, notes); items.append((OUT/name, 'overview/'+name))
    items += [(probe.OUT/'comparisons'/(r['id']+'.png'), 'all_ROIs/'+r['id']+'.png') for r in rows]
    archive_path = OUT/'preserved_boundary_comparisons.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == 53 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    table = '\n'.join('| '+r['id']+' | '+' | '.join('—' if v['median_deltaLab'] is None else f'{v["median_deltaLab"]:.4f}' for v in r['conditions'].values())+' |' for r in metrics if r['id'] in targets)
    (OUT/'report.md').write_text(f'''# 保存する境界色との連続性だけを有限帯で補正

2026-10-01。**編集外に残す明るい境界色を有限帯に反映すると色はわずかに変わるが、平坦な塗り跡・残存輪郭を十分には解決しない。追加参照には薄い汚れも含み得るため、現状の補正を一律採用しない。**

[穴の比較PNG](01_holes.png)、[裂けと折れ](02_crack_and_fold.png)、[色参照の内訳](03_reference_sources.png)、[対照](04_controls.png)、[印影と不確かな曲線](05_stamp_and_unknown.png)、[全53PNG ZIP](preserved_boundary_comparisons.zip)。全42ROIの原画像比較と対象3ROIの拡大・色参照図。

## 境界だけを調べた条件

前回の薄いコア＋近接8px補完のマスク・適用範囲・観測明度模様を固定。既知紙面としては選ばなかった編集外の画素を、安全領域内・保護外・L≥50・紙色差L[-6,8]・a/b差3/6以内に限って追加色源にした。v6の保護・保留や薄い汚れを含む可能性がある。**清浄な紙面の正解ではなく、その画素を原画素のまま残すことと補修色との連続性を調べる参照。** 編集内の原損傷とDDRMの色は使わない。

元と追加参照の4近傍調和紙色場の差をLab±3/1/1.5に限定し、距離2／4pxの指数羽根で加えた。距離1+3倍の半径より奥は補正0。マスク・受理範囲・模様は変えず、適用内の境界色だけを比較。元の保存場を全42ROIで完全再現してから試作を凍結した。

元の不参照境界は裂け300組、新しい穴1,319組などで、多数を境界条件として使っていなかった。追加参照でその一部に連続性を与えるが、色源の汚れや損傷も反映し得る。下表は追加色源との4近傍Lab色差中央値で最適化対象に近い診断値であり、独立品質・除去精度ではない。画像では塗り跡・紙模様差・薄い残存縁が残る。

| ROI | 前回 | 2px帯 | 4px帯 |
| --- | ---: | ---: | ---: |
{table}

次は色補正の増幅を続けず、紙面の低周波の明度模様を観測紙面から移す因子を、マスク・平均紙色・細かい模様を固定して比較する。元の細かい模様が移っていることは確認済みで、単に模様を強めればよいとは判断しない。旧マスクの原因不明の曲線も反例として維持する。

## 検証と適用範囲

追加参照の位置・色条件・全編集外を元画像から再計算。両調和場の直接4近傍残差、保存境界場と元模様の合成、補正量・有限帯外0・Lab合成、全84条件の適用外・編集外原画素一致を確認した。全非参照画素の異色置換不変、全保存依存とZIP53PNGの内容ハッシュも確認。[検証JSON](verification.json)、[全色差と残差](diagnostics.json)。

旧14ページの固定ROIの範囲外は保存DDRM、新4ページと追加成分の範囲外は原画像。DDRM再推論・学習はなし。全例は開発例で、原因・画素GT・学習データ独立性は未確定。利用者の停止まで改善を継続する。
''')
    dump(OUT/'diagnostics.json', dict(residuals=residuals, boundary_color_metrics=metrics))
    dump(OUT/'verification.json', dict(status='passed', ROIs=42, variants=84,
        direct_residual_max=max(r['residual'] for r in residuals), masks_applied_texture_fixed=True,
        correction_limits_verified=True, finite_band_support=True, outside_applied_changes=0,
        outside_edit_changes=0, ZIP_PNG_count=53, zip_hashes=True, pixel_GT=False))
    dependencies = {str((probe.OUT/name).relative_to(ROOT)): sha(probe.OUT/name) for name in ['manifest.json', 'complete.json']}
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: bounded preserved boundary, 84 variants, 53 PNG', flush=True)


if __name__ == '__main__': main()
