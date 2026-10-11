"""Verify the thin-mask restorer and all additional whole-page components."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import restore_thin_margin_core as probe
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_blending_review import check_bundle
from package_paper_seam_texture_review import verify_texture

ROOT = probe.ROOT
OUT = ROOT/'tmp_work/thin_margin_restoration_review_20261001'


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(probe.OUT)
    rows = json.loads((probe.OUT/'diagnostics.json').read_text()); assert len(rows) == 42
    originals = {}
    for source in probe.mask.prior.MARGIN.SOURCES+[probe.mask.new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']: originals[page['page']] = page['original']
    current = None; residuals = []; provenance = []; checked = []
    for r in rows:
        rid = r['id']; native = probe.OUT/'native'/rid
        if current != r['page']: original = read(ROOT/originals[r['page']]); current = r['page']
        x, y, w, h = r['xywh']; cx, cy, cw, ch = r['context_xywh']
        context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w)); crop = context[s]
        base = read(native/'base.png'); value = read(native/'thin16.png')
        assert np.array_equal(read(native/'original.png'), crop)
        full = probe.mask.OUT/'pages'/r['page']
        edit = read(native/'edit.png', True); added = read(native/'added.png', True)
        assert np.array_equal(edit, read(full/'edit.png', True)[y:y+h, x:x+w])
        assert np.array_equal(added, read(full/'added.png', True)[y:y+h, x:x+w])
        applied = read(native/'applied_context.png', True); known = read(native/'reference_context.png', True)
        active = read(native/'eligible_context.png', True); old_applied = read(native/'old_applied_context.png', True)
        assert not np.any(applied&~active) and not np.any(known&active)
        assert not np.any(np.any(value != base, axis=2)&~applied[s])
        assert not np.any(np.any(value != crop, axis=2)&~edit)
        assert r['total_applied'] == int(applied[s].sum())
        assert r['filled_additional_edit'] == int((applied[s]&added).sum())
        assert r['newly_filled'] == int((applied[s]&~old_applied[s]).sum())
        if (native/'harmonic_fields_Lab.npz').exists():
            lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            fields = np.load(native/'harmonic_fields_Lab.npz', allow_pickle=False)
            mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)
            smooth = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), 6)/np.maximum(mass[:, :, None], 1e-6)
            for name, boundary in [('boundary_raw', lab), ('boundary_smooth', smooth)]:
                error, used = verify_field(fields[name], applied, known, boundary)
                assert np.array_equal(used, read(native/'used_context.png', True))
                residuals.append(dict(id=rid, field=name, residual=error))
            donors = json.loads((native/'donors.json').read_text())
            if donors['components']: provenance.append(dict(id=rid, **verify_texture(lab, known, applied, native)))
            else:
                assert not np.load(native/'texture_seam.npy', allow_pickle=False).any()
                assert len(donors['patch_xy']) < 4 or not applied.any()
            distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
            bw = np.exp(-(np.maximum(distance-1, 0)/8.)**2)*applied
            field = fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*bw[:, :, None]
            field[:, :, 0] += np.load(native/'texture_seam.npy', allow_pickle=False)*np.clip((distance-1)/4., 0, 1)*applied
            assert np.array_equal(field, np.load(native/'result_field_Lab.npy', allow_pickle=False))
            converted = np.rint(np.clip(cv2.cvtColor(field[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
            assert np.array_equal(value[applied[s]], converted[applied[s]])
        else: assert np.array_equal(value, base)
        checked.append(rid); print('Verified thin mask restoration:', rid, flush=True)
    items = []
    old_components = [r['id'] for r in rows if not r['fixed_ROI'] and r['page'] == '200021712_00082_1']
    for name, ids, title, notes in [
        ('01_thin_crack.png', ['unseen_thin_crack'], '太さ条件だけで細い裂けを復元へ渡す',
         ['追加候補1,207画素のうち1,206画素を補間。既存候補も接続して合計1,424画素を新規補間。',
          '中央と右枝は薄くなるが、左の小さな枝端と補修色の違いは残る。']),
        ('02_unknown_curves.png', old_components, '原因不明の曲線も消える：全ページの追加を保持',
         ['全5成分を表示。損傷か筆記か確定していないため、除去の成功数にはしない。',
          'これら8追加成分の診断はDDRMなし。補間外は原画像。近接ROIの数は重複する。']),
        ('03_spot_and_controls.png', ['unseen_brown_spot', 'unseen_seal', 'new_uncertain_trace'], '斑点の未解決と対照',
         ['斑点に128画素追加しても境界16組では補間0。太さ変更と参照8組は組み合わせない。',
          '全34固定ROI＋全8追加成分を省略せず保存。'])]:
        stack([probe.OUT/'comparisons'/(rid+'.png') for rid in ids], OUT/name, title, notes)
        items.append((OUT/name, 'overview/'+name))
    items += [(probe.OUT/'comparisons'/(r['id']+'.png'), ('fixed_ROIs/' if r['fixed_ROI'] else 'new_components/')+r['id']+'.png') for r in rows]
    archive_path = OUT/'thin_mask_restoration_comparisons.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == 45 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    table = '\n'.join(f'| {r["id"]} | {r["additional_edit"]:,} | {r["filled_additional_edit"]:,} | {r["newly_filled"]:,} |' for r in rows)
    (OUT/'report.md').write_text(f'''# 薄いコアを同じ紙面復元へ渡した結果

2026-10-01。**太さ条件だけの変更で細い裂け状の跡を1,424画素補間でき、中央と右枝が薄くなった。ただし左の小さな枝端と補修色の違いが残り、原因不明の曲線も消す。薄いコアの一律採用は保留して改善を続ける。**

[細い裂けの比較PNG](01_thin_crack.png)、[全ページで増えた原因不明の曲線5成分](02_unknown_curves.png)、[斑点と対照](03_spot_and_controls.png)、[全45PNG ZIP](thin_mask_restoration_comparisons.zip)。原画像を並べ、全34固定ROIと全8追加成分を含めた。

## 一因子の接続確認

[分類のピーク半径6→3.5px](../thin_core_reference_gate_review_20261001/report.md)だけを変えた保存マスクに、元の境界16組以上・面積64以上の調和紙色補間と観測明度模様を適用した。境界8組の変更、分布紙色、色模様の変更は加えていない。元の紙面参照から追加編集画素を除き、新しいマスクに従って補間場と模様候補を計算し直す。全18ページのマスクを使い、ROI名で候補を制限しない。

裂けROIの追加編集1,207画素中1,206画素を補間。元から編集候補だった隣接217画素も同じ成分に接続し、計1,424画素を新規補間した。新成分の参照境界は98組で、従来の16組条件を満たす。左上の小さな枝端は追加コアの外に残った。追加編集と新規補間は定義が異なり、画素の正解数ではない。

旧ページの5個の原因不明の筆画状成分にも補間が入った。外観だけでは損傷か筆記か確定できないので、この消失を除去成功と扱わない。追加成分の矩形・文脈が重なり、表の新規補間画素を単純合計して全ページの除去量にはしない。

斑点は追加128画素を得ても紙面参照16組の条件では未補間。本文・印影などほか32固定ROIはこの変更で元出力と完全一致。対象外の対照の不変を、全ページのインク識別精度にはしない。

| ROI | 追加編集候補 | うち補間した画素 | 新規補間全体 |
| --- | ---: | ---: | ---: |
{table}

## 復元比較と検証の意味

旧14ページの34固定ROI中の旧22ROIは保存された輪郭回収の紙面補間と範囲外のDDRMが土台。新4ページと今回追加した8成分の診断は紙面補間のみで、補間外は原画像。新しいDDRM推論・学習はない。紙面補間が内容を真正に復元したとも、原因を自動識別できたとも言わない。すべて調整に使った開発例で、画素GTと学習データ独立性は不明。

全ページの基準参照・適用候補と保存文脈の一致、入力・保存場・凍結ソースのハッシュ、全42ROIの範囲外と編集外の不変を確認。保存された場の4近傍残差を直接計算し、観測模様の元画素と重複平均を再構成、Lab合成と保存PNGを確認した。全非参照画素の異色置換に対する不変とZIP45PNGの内容ハッシュも確認。[検証JSON](verification.json)、[数値と参照元](diagnostics.json)。

次は、残った細い枝端の拒否理由と新規成分の形を測り、受理した損傷からの連続性を使えるか検証する。筆画状の反例を維持し、単なるサイズ・太さ緩和を一律には採用しない。利用者の停止指示まで継続する。
''')
    dump(OUT/'diagnostics.json', dict(harmonic_residuals=residuals, observed_texture_provenance=provenance))
    dump(OUT/'verification.json', dict(status='passed', ROIs=42, fixed_ROIs=34, new_components=8,
        directly_checked_residual_max=max(r['residual'] for r in residuals), outside_applied_changes=0,
        outside_edit_changes=0, observed_donors_verified=True, ZIP_PNG_count=45, zip_hashes=True,
        pixel_GT=False, one_mask_factor=True))
    dependencies = {str((probe.OUT/name).relative_to(ROOT)): sha(probe.OUT/name) for name in ['manifest.json', 'complete.json']}
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: unchanged restorer on thin mask, 42 ROIs, 45 PNG', flush=True)


if __name__ == '__main__': main()
