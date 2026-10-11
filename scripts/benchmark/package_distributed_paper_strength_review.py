"""Directly verify strength-only screened harmonic ablations."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_distributed_paper_strength as probe
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_distributed_paper_guidance_review import direct_residual
from package_paper_blending_review import check_bundle

ROOT = probe.ROOT
OUT = ROOT/'tmp_work/distributed_paper_strength_review_20261001'


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(probe.OUT)
    rows = json.loads((probe.OUT/'diagnostics.json').read_text()); assert len(rows) == 34
    previous = probe.prior; pairs = previous.prior; originals = {}
    for source in pairs.frozen.MARGIN.SOURCES+[pairs.new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']: originals[page['page']] = page['original']
    current = None; residuals = []; checked = []
    for r in rows:
        rid = r['id']; native = probe.OUT/'native'/rid; src = pairs.OUT/'native'/rid
        if current != r['page']: original = read(ROOT/originals[r['page']]); current = r['page']
        x, y, w, h = r['xywh']; cx, cy, cw, ch = r['context_xywh']
        context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w)); crop = context[s]
        known = read(src/'reference_context.png', True); applied = read(src/'applied8_context.png', True)
        extra = read(src/'newly_applied_context.png', True); base = read(native/'base.png')
        assert np.array_equal(base, read(previous.OUT/'native'/rid/'guide32.png'))
        assert np.array_equal(crop, read(native/'original.png'))
        for denominator in probe.DENOMINATORS:
            mode = 'strength'+str(denominator); md = native/mode; value = read(md/'output.png')
            assert not np.any(np.any(value != base, axis=2)&~extra[s])
            if extra.any():
                lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 32)
                guide = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), 32)/np.maximum(mass[:, :, None], 1e-6)
                strength = np.where(mass >= .05, mass*(1/denominator), 0).astype(np.float32)*applied
                assert np.array_equal(mass, np.load(md/'reference_mass.npy', allow_pickle=False))
                assert np.array_equal(guide, np.load(md/'guide_Lab.npy', allow_pickle=False))
                assert np.array_equal(strength, np.load(md/'screening_strength.npy', allow_pickle=False))
                fields = np.load(md/'screened_fields_Lab.npz', allow_pickle=False)
                mm = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)
                smooth = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), 6)/np.maximum(mm[:, :, None], 1e-6)
                for name, target in [('boundary_raw', lab), ('boundary_smooth', smooth)]:
                    residuals.append(dict(id=rid, denominator=denominator, field=name,
                        residual=direct_residual(fields[name], target, applied, known, strength, guide)))
                distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
                bw = np.exp(-(np.maximum(distance-1, 0)/8.)**2)*applied
                field = fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*bw[:, :, None]
                field[:, :, 0] += np.load(src/'texture_seam.npy', allow_pickle=False)*np.clip((distance-1)/4., 0, 1)*applied
                assert np.array_equal(field, np.load(md/'result_field_Lab.npy', allow_pickle=False))
                converted = np.rint(np.clip(cv2.cvtColor(field[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
                assert np.array_equal(value[extra[s]], converted[extra[s]])
            else: assert np.array_equal(value, base)
        checked.append(rid); print('Verified screening strength:', rid, flush=True)
    items = []
    for name, ids, title, notes in [
        ('01_spot_strength.png', ['unseen_brown_spot'], '固定マスク・σ32で紙色項の強さだけを比較',
         ['内部は少し明るくなるが、弱→中→強の差は小さく、平坦な色塊と角張った縁は残る。']),
        ('02_other_components.png', ['unseen_fold', 'brown_false_red_new'], '他の新規受理成分への影響',
         ['局所色が明るくなるが、折れの塗り跡や広い色ムラの解消ではない。']),
        ('03_controls.png', ['unseen_dense_text', 'unseen_seal', 'new_uncertain_trace'], '本文・印影・不確かな跡の対照',
         ['既存の補間画素とこの対照は保持。全34ROIを保存した。'])]:
        stack([probe.OUT/'comparisons'/(rid+'.png') for rid in ids], OUT/name, title, notes)
        items.append((OUT/name, 'overview/'+name))
    items += [(probe.OUT/'comparisons'/(r['id']+'.png'), 'all_ROIs/'+r['id']+'.png') for r in rows]
    archive_path = OUT/'paper_strength_comparisons.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == 37 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    (OUT/'report.md').write_text('''# 分布する観測紙色への強さだけを比較

2026-10-01。**周囲の観測紙色に近づける強さを増すと内部が少し明るくなるが、効果は小さく、斑点の平坦な色塊と角張った縁が残る。強さの増加だけを追加採用しない。**

[斑点の比較PNG](01_spot_strength.png)、[他の成分](02_other_components.png)、[対照](03_controls.png)、[全37PNG ZIP](paper_strength_comparisons.zip)。すべて原画像を並べた。DDRM再推論・学習はなし。

σ32、マスク、参照資格、受理範囲、明度模様、直接／平滑境界と距離合成は固定。内部項の強さだけを参照密度／256→／64→／16に変え、境界8組で新規受理した成分へ合成した。以前の補間画素は保持。薄いコア・枝端補完・色模様の変更とは組み合わせていない。

斑点の新規成分L中央値は／256で69.153、／64で69.800、／16で69.891。これは内部項の目的に沿う明度診断で、自然さ・除去精度ではない。折れの新規225画素と茶色い紙面124画素にも変化するが、範囲は狭く、問題全体の解決とは言えない。他31ROIは前回と完全一致。[全診断](../distributed_paper_strength_pilot_20261001/diagnostics.json)。

元の弱い／256の保存場を画素単位で完全再現してから比較した。定数紙面、全非参照画素の異色置換に対する不変を実行。保存入力・ソース・出力のハッシュ、全68条件の範囲外・以前の補間画素一致、既知紙面だけからの分布参照・密度・強さの再計算、行列を再構築しない直接4近傍差＋内部項残差、固定明度模様とのLab合成、ZIP37PNGの内容ハッシュを確認。[検証JSON](verification.json)。

旧14ページの土台には保存DDRM、新4ページの未補間部は原画像。全例は調整した開発例で、画素GTと学習データ独立性は不明。次は増幅を止め、補間の紙色参照に残る薄い暖色の偏りを診断する。固定マスク・固定明度模様で紙色参照の資格だけを比較し、参照不足になった成分も省略せず記録する。利用者の停止指示まで改善を継続する。
''')
    dump(OUT/'verification.json', dict(status='passed', ROIs=34, variants=68,
        direct_residual_max=max(r['residual'] for r in residuals), reference_texture_mask_fixed=True,
        outside_newly_applied_changes=0, ZIP_PNG_count=37, zip_hashes=True, pixel_GT=False))
    dump(OUT/'diagnostics.json', dict(residuals=residuals))
    dependencies = {str((probe.OUT/name).relative_to(ROOT)): sha(probe.OUT/name) for name in ['manifest.json', 'complete.json']}
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: strength alone, 68 variants, 37 PNG', flush=True)


if __name__ == '__main__': main()
