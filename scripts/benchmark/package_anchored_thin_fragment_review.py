"""Check all anchored-fragment mask levels and unchanged paper equations."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_anchored_thin_fragments as probe
from build_npj_v6_comparison import dump, read, sha
from check_paper_margin_defect_ddrm import figure
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_blending_review import check_bundle
from package_paper_seam_texture_review import verify_texture

ROOT = probe.ROOT
OUT = ROOT/'tmp_work/anchored_thin_fragment_review_20261001'


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(probe.OUT)
    data = json.loads((probe.OUT/'diagnostics.json').read_text()); rows = data['regions']; pages = data['pages']
    assert len(rows) == 42 and len(pages) == 18
    originals = {}
    for source in probe.prior.mask.prior.MARGIN.SOURCES+[probe.prior.mask.new.INPUT]:
        for p in json.loads((source/'baseline_review.json').read_text())['cases']: originals[p['page']] = p['original']
    for pg in pages:
        key = pg['page']; folder = probe.OUT/'pages'/key
        base = read(probe.prior.mask.OUT/'pages'/key/'edit.png', True)
        domain = read(probe.prior.mask.OUT/'pages'/key/'domain.png', True)
        guard = read(probe.prior.mask.OUT/'pages'/key/'guard.png', True)
        a8 = read(folder/'near8/added.png', True); a16 = read(folder/'near16/added.png', True)
        assert not np.any(a8&~a16)
        for radius in probe.RADII:
            mode = 'near'+str(radius); added = read(folder/mode/'added.png', True)
            edit = read(folder/mode/'edit.png', True); core = read(folder/mode/'fragment_core.png', True)
            assert np.array_equal(edit, base|added) and not np.any(base&added)
            assert not np.any(added&(~domain|guard))
            halo = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
            assert not np.any(added&~halo)
            assert pg['diagnostics'][mode]['added'] == int(added.sum())
            for component in json.loads((folder/mode/'fragments.json').read_text()):
                assert 16 <= component['area'] < 128 and component['brown_seeds'] >= 8 and component['peak'] >= 2
                assert 0 < component['distance'] <= radius
    current = None; residuals = []; provenance = []; checked = []
    for r in rows:
        rid = r['id']; key = r['page']; native = probe.OUT/'native'/rid
        if current != key: original = read(ROOT/originals[key]); current = key
        x, y, w, h = r['xywh']; cx, cy, cw, ch = r['context_xywh']
        context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w)); crop = context[s]
        assert np.array_equal(read(native/'original.png'), crop)
        base = read(native/'base.png'); assert np.array_equal(base, read(probe.prior.OUT/'native'/rid/'thin16.png'))
        original_known = read(probe.prior.OUT/'native'/rid/'reference_context.png', True)
        original_active = read(probe.prior.OUT/'native'/rid/'eligible_context.png', True)
        old_applied = read(native/'old_applied_context.png', True)
        for radius in probe.RADII:
            mode = 'near'+str(radius); md = native/mode
            known = read(md/'reference_context.png', True); active = read(md/'eligible_context.png', True)
            applied = read(md/'applied_context.png', True); added = read(md/'added.png', True)
            value = read(md/'output.png'); edit = read(md/'edit.png', True)
            full = probe.OUT/'pages'/key/mode
            context_added = read(full/'added.png', True)[cy:cy+ch, cx:cx+cw]
            assert np.array_equal(added, context_added[s])
            assert np.array_equal(known, original_known&~context_added)
            assert np.array_equal(active, original_active|context_added)
            assert not np.any(applied&~active) and not np.any(known&active)
            assert not np.any(np.any(value != base, axis=2)&~applied[s])
            assert not np.any(np.any(value != crop, axis=2)&~edit)
            stats = r['fragment_diagnostics'][mode]
            assert stats['filled_added'] == int((applied[s]&added).sum())
            assert stats['newly_filled'] == int((applied[s]&~old_applied[s]).sum())
            assert stats['changed_vs_base'] == int(np.any(value != base, axis=2).sum())
            if (md/'harmonic_fields_Lab.npz').exists():
                lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                fields = np.load(md/'harmonic_fields_Lab.npz', allow_pickle=False)
                mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)
                smooth = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), 6)/np.maximum(mass[:, :, None], 1e-6)
                for name, boundary in [('boundary_raw', lab), ('boundary_smooth', smooth)]:
                    error, used = verify_field(fields[name], applied, known, boundary)
                    assert np.array_equal(used, read(md/'used_context.png', True))
                    residuals.append(dict(id=rid, mode=mode, field=name, residual=error))
                donors = json.loads((md/'donors.json').read_text())
                if donors['components']: provenance.append(dict(id=rid, mode=mode, **verify_texture(lab, known, applied, md)))
                else: assert not np.load(md/'texture_seam.npy', allow_pickle=False).any()
                distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
                bw = np.exp(-(np.maximum(distance-1, 0)/8.)**2)*applied
                field = fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*bw[:, :, None]
                field[:, :, 0] += np.load(md/'texture_seam.npy', allow_pickle=False)*np.clip((distance-1)/4., 0, 1)*applied
                assert np.array_equal(field, np.load(md/'result_field_Lab.npy', allow_pickle=False))
                converted = np.rint(np.clip(cv2.cvtColor(field[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
                assert np.array_equal(value[applied[s]], converted[applied[s]])
            else: assert np.array_equal(value, base)
        checked.append(rid); print('Verified anchored fragments:', rid, flush=True)
    specifications = [
        ('01_thin_crack.png', ['unseen_thin_crack'], '小さな枝端を近接条件で補完',
         ['8px条件は104画素追加し97画素を補間。左の濃い枝端も薄くなるが、補修色・模様差は残る。']),
        ('02_unknown_trace.png', ['200021712_00082_1_component_16', '200021712_00082_1_component_21'], '16pxでは原因不明の曲線の周囲も増える',
         ['8pxではこの旧ページへの追加0。16pxは189画素追加・182画素補間。正解とは数えない。']),
        ('03_whole_page_masks.png', [], '2ページのマスクを全体で確認', []),
        ('04_controls.png', ['unseen_seal', 'unseen_dense_text', 'new_uncertain_trace'], '本文・印影・既存の不確かな跡',
         ['両距離でこの3ROIの追加・出力変更0。既存の薄いコアの曖昧さは継続。'])]
    items = []
    for name, ids, title, notes in specifications:
        paths = [probe.OUT/'comparisons'/(rid+'.png') for rid in ids] if ids else [probe.OUT/'whole_page_masks'/(key+'.png') for key in ['200021712_00082_1', '200017458_00004_2']]
        stack(paths, OUT/name, title, notes); items.append((OUT/name, 'overview/'+name))
    rid = 'unseen_thin_crack'; row = next(r for r in rows if r['id'] == rid); native = probe.OUT/'native'/rid
    x, y, w, h = row['xywh']; key = row['page']
    core = read(probe.prior.mask.OUT/'pages'/key/'new_core.png', True)[y:y+h, x:x+w]
    fragment = read(probe.OUT/'pages'/key/'near8/fragment_core.png', True)[y:y+h, x:x+w]
    yy, xx = np.nonzero(core|fragment); x0, y0 = max(0, int(xx.min())-12), max(0, int(yy.min())-12)
    x1, y1 = min(w, int(xx.max())+13), min(h, int(yy.max())+13)
    images = [read(native/name)[y0:y1, x0:x1] for name in ['original.png', 'base.png', 'near8/output.png', 'near16/output.png']]
    figure(images, ['原画像', '前回', '近接8px', '近接16px'], OUT/'05_branch_closeup.png', '細い裂け · 同じ画素範囲の拡大',
        '最近傍で同じ範囲を拡大。枝端の残存と塗り跡を別に確認。', '位置・復元器は固定。物理原因と復元の真正性のGTはない。')
    items.append((OUT/'05_branch_closeup.png', 'overview/05_branch_closeup.png'))
    items += [(probe.OUT/'comparisons'/(r['id']+'.png'), 'all_ROIs/'+r['id']+'.png') for r in rows]
    items += [(probe.OUT/'whole_page_masks'/(p['page']+'.png'), 'whole_page_masks/'+p['page']+'.png') for p in pages]
    archive_path = OUT/'anchored_fragment_comparisons.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == 65 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    table = '\n'.join(f'| {r["id"]} | '+ ' | '.join(str(r['fragment_diagnostics'][m][k]) for m in ['near8', 'near16'] for k in ['added_edit', 'filled_added', 'changed_vs_base'])+' |' for r in rows)
    (OUT/'report.md').write_text(f'''# 受理済み薄いコアに近い小成分を補完

2026-10-01。**近接8pxの小成分補完で、残っていた左の濃い枝端が薄くなった。16pxでは原因不明の曲線の周りも増え、枝端の効果は8pxと同じだった。8pxを狭い開発候補として残すが、元の薄いコアの曖昧な曲線と補修色差は未解決。**

[裂けの比較PNG](01_thin_crack.png)、[枝端の拡大](05_branch_closeup.png)、[16pxで増えた原因不明の跡](02_unknown_trace.png)、[全ページのマスク](03_whole_page_masks.png)、[対照](04_controls.png)、[全65PNG ZIP](anchored_fragment_comparisons.zip)。全18ページ・42ROIを保持。

## 小成分を単独で拾わない条件

[前回の薄いコア＋境界16組](../thin_margin_restoration_review_20261001/report.md)を土台にし、受理済みの新しい薄いコアから8／16px以内の茶色い小成分だけを一段補完した。面積16..127、茶色種8以上、ピーク半径2px以上、余白境界に非接触、従来のサイズ制限。補完からさらに次の成分を連鎖的に拾わない。従来の6px近傍・明度差2・安全領域・保護条件を維持し、全ページで同じ条件を使った。ROI名は受理条件に使わない。

枝端は面積36・茶色種23・ピーク2px・コアから6.083pxで、以前の単独面積128条件を満たさず残っていた。8pxで編集候補を104画素追加し、その97画素を補間した。既存候補も接続し98画素の新規補間。16pxでも枝端は同じで追加の改善はない。旧ページは8pxで追加0、16pxで面積71・茶色種22・距離11.180pxの不確かな小成分を拾い、189画素追加・182画素補間した。ほか16ページのマスクは両距離とも不変。

マスクに合わせて参照画素・場・明度模様の候補を再計算するので、変化は新しい104画素だけに限定されない。裂けの固定ROIでは出力1,172画素が前回と異なる。これは元から補間した箇所の紙色・模様の再計算も含み、除去量ではない。本文・印影などほか33固定ROIは前回と完全一致。追加8成分の矩形は互いに重なり、画素数を合計して全ページの量にはしない。

| ROI | 8px追加編集 | 8pxうち補間 | 8px出力変更 | 16px追加編集 | 16pxうち補間 | 16px出力変更 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

## 残る問題と比較範囲

枝端は薄くなるが、原画素の微小な残存と紙色・模様の補修跡が残る。以前から薄いコア条件で受理した旧ページの原因不明5曲線はこの8px補完でも消えており、8pxだけを見て安全なインク識別とは判断しない。8pxをさらに検証する候補とし、16pxへ広げる根拠はない。画素GTも物理原因の確定もなく、一律の本番採用はしない。

境界16組・調和紙色・観測明度模様は前回どおり。参照8組、分布紙色、色模様は混ぜていない。旧14ページの固定ROIの範囲外は保存DDRM、新4ページと追加成分の診断の範囲外は原画像。新しいDDRM推論・学習はない。全例は調整した開発例、学習データとの独立性は不明。

## 確認と継続

全18ページのマスク包含・8px⊆16px・保護・成長範囲と受理成分の条件を確認。全42ROI・84条件で、基準参照からの追加編集除外、適用範囲・出力変更数、範囲外と編集外一致を確認した。保存場に4近傍差を足して残差を直接確認し、観測模様の元画素座標・重複平均・Lab合成を再構成。全非参照画素の異色置換不変、入力・ソース・出力とZIP65PNGのハッシュも確認。[確認JSON](verification.json)、[場と参照元の診断](diagnostics.json)。

次は、紙色の平坦な塊に対する分布参照の強さを、同じマスク・参照・模様で比較する。今回の8px補完と混ぜず、分類と復元の因子を維持する。利用者の停止まで継続する。
''')
    dump(OUT/'diagnostics.json', dict(residuals=residuals, observed_texture_provenance=provenance,
        closeup_xywh=[x0, y0, x1-x0, y1-y0]))
    dump(OUT/'verification.json', dict(status='passed', pages=18, ROIs=42, variants=84,
        residual_max=max(r['residual'] for r in residuals), no_recursive_chaining=True,
        outside_applied_changes=0, outside_edit_changes=0, ZIP_PNG_count=65, zip_hashes=True, pixel_GT=False))
    dependencies = {str((probe.OUT/name).relative_to(ROOT)): sha(probe.OUT/name) for name in ['manifest.json', 'complete.json']}
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: anchored fragments, 84 variants, 65 PNG', flush=True)


if __name__ == '__main__': main()
