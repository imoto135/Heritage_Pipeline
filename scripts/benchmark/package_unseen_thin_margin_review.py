"""Audit original-first selection, full-page masks and equal-restorer PNGs."""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import validate_unseen_thin_margin_candidate_v2 as masks
import restore_unseen_thin_margin_paper as fill
from build_npj_v6_comparison import dump, read, sha
from check_paper_margin_defect_ddrm import figure
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_seam_texture_review import verify_texture
from package_paper_blending_review import check_bundle

ROOT = masks.ROOT
INPUT = masks.INPUT
OUT = INPUT/'review'


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(masks.OUT); check_bundle(fill.OUT)
    selected = json.loads((INPUT/'baseline_review.json').read_text())
    for relative, digest in selected['inputs'].items(): assert sha(ROOT/relative) == digest
    assert selected['selection_before_new_v6_and_candidate'] and selected['original_only']
    assert not set(p['book'] for p in selected['cases'])&set(selected['excluded_books'])
    assert len(set(p['book'] for p in selected['cases'])) == 3
    md = json.loads((masks.OUT/'diagnostics.json').read_text())
    fd = json.loads((fill.OUT/'diagnostics.json').read_text()); by_id = {r['id']:r for r in fd}
    region_masks = {r['id']:r for r in md['regions']}; records = []; residuals = []; texture = []
    for page in selected['cases']:
        key = page['page']; original = read(ROOT/page['original']); directory = masks.OUT/'pages'/key
        v6 = read(directory/'v6/edit.png', True); domain = read(directory/'features/safe_domain.png', True)
        guard = read(directory/'features/rim_guard.png', True); covered = np.zeros(v6.shape, bool)
        edits = {m:read(directory/m/'edit.png', True) for m in masks.MODES}
        for m in ['rim', 'thin', 'near8']:
            added = edits[m]&~v6
            assert np.array_equal(added, read(directory/m/'added.png', True))
            assert not np.any(v6&~edits[m]) and not np.any(added&(guard|~domain))
            for name in ['red', 'extra']: assert np.array_equal(read(directory/m/(name+'.png'), True), read(directory/'v6'/(name+'.png'), True))
            for name in ['protect', 'hold']: assert np.array_equal(read(directory/m/(name+'.png'), True), read(directory/'v6'/(name+'.png'), True)&~added)
        for roi in page['rois']:
            rid = roi['id']; row = by_id[rid]; x,y,w,h = roi['xywh']; rs = (slice(y,y+h),slice(x,x+w)); covered[rs] = True
            crop = original[rs]; native = fill.OUT/'native'/rid
            assert np.array_equal(read(native/'original.png'), crop)
            cx,cy,cw,ch = row['context_xywh']; cs=(slice(cy,cy+ch),slice(cx,cx+cw)); s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            lab=cv2.cvtColor(original[cs].astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            for mode in fill.MODES:
                local=native/mode; value=read(local/'output.png'); edit=read(local/'edit.png',True)
                active=read(local/'eligible_context.png',True); applied=read(local/'applied_context.png',True)
                known=read(local/'reference_context.png',True); used=read(local/'used_context.png',True)
                assert not np.any(known&active) and not np.any(applied&~active)
                assert not np.any(np.any(value!=crop,axis=2)&~applied[s])
                assert not np.any(np.any(value!=crop,axis=2)&~edit)
                expected=read(fill.OUT/'reference_pages'/key/(mode+'_paper.png'),True)[cs]
                assert np.array_equal(known,expected)
                if mode!='npj':
                    assert np.array_equal(edit,edits[mode][rs])
                    assert np.array_equal(active,edits[mode][cs]&domain[cs]&~guard[cs])
                stat=row['diagnostics'][mode]
                assert stat['applied_pixels']==int(applied[s].sum()) and stat['changed_pixels']==int(np.any(value!=crop,axis=2).sum())
                if applied.any():
                    fields=np.load(local/'harmonic_fields_Lab.npz',allow_pickle=False)
                    mass=cv2.GaussianBlur(known.astype(np.float32),(0,0),6)
                    smooth=cv2.GaussianBlur(np.where(known[:,:,None],lab,0),(0,0),6)/np.maximum(mass[:,:,None],1e-6)
                    for name,target in [('boundary_raw',lab),('boundary_smooth',smooth)]:
                        error, actual_used=verify_field(fields[name],applied,known,target)
                        assert np.array_equal(actual_used,used)
                        residuals.append(dict(id=rid,mode=mode,field=name,residual_max=error))
                    donor=json.loads((local/'donors.json').read_text())
                    if donor['components']: texture.append(dict(id=rid,mode=mode,**verify_texture(lab,known,applied,local)))
                    else:
                        assert len(donor['patch_xy'])<4
                        assert not np.load(local/'texture_seam.npy',allow_pickle=False).any()
                        assert np.all(np.load(local/'source_x.npy',allow_pickle=False)==-1)
                    distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5)
                    weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*applied
                    result=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
                    result[:,:,0]+=np.load(local/'texture_seam.npy',allow_pickle=False)*np.clip((distance-1)/4.,0,1)*applied
                    assert np.array_equal(result,np.load(local/'result_field_Lab.npy',allow_pickle=False))
                    assert np.array_equal(value,fill.frozen.compose(result,applied,crop,s))
                else: assert np.array_equal(value,crop)
            # Additional candidate stages are no-ops on this selected set, including outside ROIs.
            assert all(np.array_equal(read(native/'v6/output.png'),read(native/m/'output.png')) for m in ['rim','thin','near8'])
            records.append(dict(id=rid,page=key,additional_vs_v6=region_masks[rid]['diagnostics']['near8']['added_vs_v6'],
                                npj_applied=row['diagnostics']['npj']['applied_pixels'], v6_applied=row['diagnostics']['v6']['applied_pixels']))
        assert all(np.array_equal(v6,edits[m]) for m in ['rim','thin','near8'])
        record=next(r for r in md['pages'] if r['page']==key)
        dump(OUT/'page_checks'/(key+'.json'),dict(**record,additional_outside_review_ROIs=int((edits['near8']&~v6&~covered).sum())))
    for folder,name in [(masks.OUT/'comparisons','masks'),(masks.OUT/'additions','additions'),(fill.OUT/'comparisons','paper')]:
        (OUT/name).mkdir(exist_ok=True)
        for p in folder.glob('*.png'): shutil.copyfile(p,OUT/name/p.name)
    shutil.copyfile(INPUT/'selected_originals.png',OUT/'00_original_selection.png')
    focus=['fresh_page_edge','fresh_margin_flecks','fresh_header_ink','fresh_red_seal','fresh_small_top_trace']
    for i,rid in enumerate(focus,1):
        stack([OUT/'masks'/(rid+'.png'),OUT/'paper'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',
              rid+' · マスクと同一条件の紙面復元',
              ['上：分類、下：同じ紙面補間。原画像を必ず併記。','v6以後の追加0。DDRMではない。npj全工程の品質比較ではない。'])
    report='''# 固定した薄い跡回収条件：別3冊の検証

原画像だけで選んだ3冊・3ページ・12 ROIに、輪郭回収、薄いコア半径3.5、一段だけの近接8pxを変更せず適用した。以前調整に使った30冊を除外し、イラスト3ページと強い装飾背景1ページも主評価から外した。画素の正解ラベルとモデル学習との独立性は不明。

**今回、v6からの追加編集は全3ページで0画素。v6以後の4条件の復元も全12 ROIで原画像と同じ。** 別冊で薄い損傷を回収できる証拠は得られなかった。元の小さな茶色い跡・紙端・折れは残る。追加がないため、保護の一般的な正確さも証明していない。

|ページ|原画像で選んだ対象|横罫線y|v6以後の追加|
|---|---|---:|---:|
|200003076_00054_1|右紙端・折れ、小さな茶色い跡、余白と本文|480|0|
|200021853_00014_2|実際の見出し文字・枠、余白と本文|検出なし|0|
|200021644_00003_2|上部の薄い跡、朱印、余白と本文|396|0|

見出しのページは横罫線未検出で安全余白領域が空になり、追加処理が停止した。よって「文字があっても回収処理が正しく保護した」とは扱えない。朱印のページではv6からの追加も紙面復元も0で、印影は原画像のまま。

保存済みnpjマスクにも**同じ16境界組の紙色補間と観測明度模様**を適用した診断を並べる。朱印ROIでは4,726画素が適用対象となり、薄い印影が一部弱くなる。これは保存済みマスクと現在の紙面補間を組み合わせた診断であり、元npjのDDRM工程の再現・評価ではない。全モードとも未適用画素は原画像を保持。

- [原画像だけの選択](00_original_selection.png)
- [紙端：分類と復元](01_fresh_page_edge.png)
- [小さな茶色い跡](02_fresh_margin_flecks.png)
- [見出し文字・枠](03_fresh_header_ink.png)
- [朱印](04_fresh_red_seal.png)
- [上部の薄い跡](05_fresh_small_top_trace.png)
- [全比較PNGのZIP](comparison_pngs.zip)
- [独立した数値検証](verification.json)

処理は全ページのマスクで行い、ROI名や対照ラベルを判定に使用していない。近接関数のコピーでは入出力先だけを変更し、以前の18ページ・8/16px両方について全マスクと成分情報の一致を確認した。参照元と出力のハッシュ、編集範囲外の不変、補間の4近傍残差、模様の原画像上の参照位置を別に検証した。

初回の保存用ラッパーは重複した辞書キーで途中停止した。`fixed_masks/`には途中結果のみを残し、辞書の保存処理だけ修正した`fixed_masks_v2/`を採用した。条件や計算関数の変更はない。

次は原画像の候補探索を見直し、余白処理が実際に動く別冊の細い損傷候補を追加する。この3冊は検証済みの開発集合へ移し、以後の独立した冊の候補選択から除外する。紙端・薄い跡の見逃しと罫線未検出の制約を残し、色・模様の小変更はまだ併用しない。
'''
    (OUT/'report.md').write_text(report)
    dump(OUT/'verification.json',dict(status='passed',selected_before_outputs=True,excluded_books=30,pages=3,ROIs=12,
        mask_addition_pixels=0,all_full_page_candidate_masks_equal_v6=True,all_v6_candidate_ROI_outputs_original=True,
        no_rule_page='200021853_00014_2',routing_parity_pages=18,outside_applied_changes=0,
        residuals=residuals,texture_provenance=texture,records=records,DDRM=False,pixel_GT=False))
    pngs=sorted(OUT.rglob('*.png'))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==len(pngs) and all(n.endswith('.png') for n in z.namelist())
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=len(pngs),
        dependencies={str((folder/name).relative_to(ROOT)):sha(folder/name) for folder in [masks.OUT,fill.OUT] for name in ['manifest.json','complete.json']},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered fixed thin margin validation:',len(pngs),'PNGs',flush=True)


if __name__=='__main__':main()
