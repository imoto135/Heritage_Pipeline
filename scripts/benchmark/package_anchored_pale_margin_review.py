"""Audit two fixed mask extensions, field provenance, and all67 comparisons."""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import restore_anchored_pale_margin as prior
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_seam_texture_review import verify_texture
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/anchored_pale_margin_review_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.mask.OUT);check_bundle(prior.OUT)
    masks=json.loads((prior.mask.OUT/'diagnostics.json').read_text())
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());assert len(rows)==67
    originals={r['page']['page']:r['page']['original'] for r in prior.mask.prior.descriptors()}
    mask_checks=[];field_checks=[];texture=[];checked=[];unchanged={6:[],4:[]}
    for page in masks['pages']:
        key=page['page'];folder=prior.mask.OUT/'pages'/key
        base=read(folder/'base_edit.png',True);domain=read(folder/'domain.png',True);guard=read(folder/'guard.png',True)
        for level in prior.LEVELS:
            added=read(folder/f'drop{level}/added.png',True);edit=read(folder/f'drop{level}/edit.png',True)
            assert np.array_equal(edit,base|added) and not np.any(base&added) and not np.any(added&(guard|~domain))
            assert int(added.sum())==page['additions'][str(level)]
            for component in page['components'][str(level)]:
                if component['accepted']:
                    assert component['core_overlap']>=32 and component['other_contacts']==0
                    assert 128<=component['area']<=.01*base.size
                    assert max(component['xywh'][2:])<=.2*min(base.shape)
            mask_checks.append(dict(page=key,level=level,additional_pixels=int(added.sum())))
    for key,path in originals.items():
        original=read(ROOT/path);page_folder=prior.mask.OUT/'pages'/key
        for row in [r for r in rows if r['page']==key]:
            rid=row['id'];native=prior.OUT/'native'/rid;old=ROOT/row['baseline_local'];x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh']
            cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            crop=original[y:y+h,x:x+w];base=read(native/'baseline.png');known=read(native/'baseline_reference_context.png',True)
            active=read(native/'baseline_eligible_context.png',True);old_applied=read(native/'baseline_applied_context.png',True)
            assert np.array_equal(read(native/'original.png'),crop) and np.array_equal(base,read(old/'output.png'))
            for label,value in [('reference_context',known),('eligible_context',active),('applied_context',old_applied)]:
                assert np.array_equal(value,read(old/(label+'.png'),True))
            lab=cv2.cvtColor(original[cs].astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            for level in prior.LEVELS:
                mode=f'drop{level}';md=native/mode;stat=row['pale_diagnostics'][mode]
                new_known=read(md/'reference_context.png',True);new_active=read(md/'eligible_context.png',True)
                applied=read(md/'applied_context.png',True);retained=read(md/'retained_applied_context.png',True)
                used=read(md/'used_context.png',True);value=read(md/'output.png');edit=read(md/'edit.png',True);added=read(md/'added.png',True)
                whole_added=read(page_folder/mode/'added.png',True)[cs]
                assert np.array_equal(added,whole_added[s])
                assert np.array_equal(new_known,known&~whole_added) and np.array_equal(new_active,active|whole_added)
                assert np.array_equal(retained,old_applied|applied)
                assert not np.any(applied&~new_active) and not np.any(new_known&new_active)
                assert not np.any(np.any(value!=base,axis=2)&~applied[s])
                assert not np.any(np.any(value!=crop,axis=2)&~edit)
                assert stat['added_edit']==int(added.sum()) and stat['filled_added']==int((applied[s]&added).sum())
                assert stat['paper_applied']==int(retained[s].sum()) and stat['newly_filled']==int((applied[s]&~old_applied[s]).sum())
                assert stat['changed_vs_base']==int(np.any(value!=base,axis=2).sum())
                if not stat['changed_vs_base']:unchanged[level].append(rid)
                if stat['recomputed']:
                    fields=np.load(md/'harmonic_fields_Lab.npz',allow_pickle=False)
                    if applied.any():
                        mass=cv2.GaussianBlur(new_known.astype(np.float32),(0,0),6)
                        smooth=cv2.GaussianBlur(np.where(new_known[:,:,None],lab,0),(0,0),6)/np.maximum(mass[:,:,None],1e-6)
                        for name,boundary in [('boundary_raw',lab),('boundary_smooth',smooth)]:
                            error,references=verify_field(fields[name],applied,new_known,boundary)
                            assert np.array_equal(references,used)
                            field_checks.append(dict(id=rid,mode=mode,field=name,residual_max=error))
                        notes=json.loads((md/'donors.json').read_text())
                        if notes['components']:texture.append(dict(id=rid,mode=mode,**verify_texture(lab,new_known,applied,md)))
                        else:
                            assert len(notes['patch_xy'])<4 and not np.load(md/'texture_seam.npy',allow_pickle=False).any()
                            assert np.all(np.load(md/'source_x.npy',allow_pickle=False)==-1)
                    distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5)
                    weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*applied
                    combined=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
                    combined[:,:,0]+=np.load(md/'texture_seam.npy',allow_pickle=False)*np.clip((distance-1)/4.,0,1)*applied
                    assert np.array_equal(combined,np.load(md/'result_field_Lab.npy',allow_pickle=False))
                    assert np.array_equal(value,prior.fill.frozen.compose(combined,applied,base,s))
                else:
                    assert not whole_added.any() or not new_active[s].any()
                    assert np.array_equal(value,base) and np.array_equal(applied,old_applied) and not used.any()
                checked.append(dict(id=rid,mode=mode,**stat))
    for folder,name in [(prior.mask.OUT/'comparisons','masks'),(prior.OUT/'comparisons','paper')]:
        (OUT/name).mkdir(exist_ok=True)
        for p in folder.glob('*.png'):shutil.copyfile(p,OUT/name/p.name)
    focus=['extra_top_branch','extra_second_branch','unseen_thin_crack','unseen_brown_spot','extra_brown_stain','extra_isolated_marks','new_brown_stamp']
    for i,rid in enumerate(focus,1):
        stack([OUT/'masks'/(rid+'.png'),OUT/'paper'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',
              rid+' · 淡い支持への接続：マスクと同じ復元器',
              ['上：黄＝前回、青＝追加。下：原画像・前回・差6・差4。',
               '差4は斑点・印影の追加と塗り跡も記録。一律採用はしない。'])
    # The additional old components have no fixed mask ROI PNG; paper comparisons expose all of them.
    ambiguous=[r for r in rows if r['page']=='200021712_00082_1' and not r.get('fixed_ROI',False)]
    stack([OUT/'paper'/(r['id']+'.png') for r in ambiguous],OUT/'08_unknown_old_components.png',
          '原因不明の筆画状成分も全5つ保持して比較',
          ['同じ追加4画素を共有する切り出しもある。数を重ねて合計しない。',
           '前回から削る5曲線自体の正誤は未確定。損傷の正解には数えない。'])
    proof=dict(status='passed',pages=25,ROIs=67,variants=134,mask_checks=mask_checks,
               direct_field_residuals=field_checks,texture_provenance=texture,regions=checked,
               unchanged_ROIs={str(k):v for k,v in unchanged.items()},outside_edit_changes=0,
               all_prior_filled_pixels_retained_on_solver_failure=True,pixel_GT=False,new_DDRM=False)
    dump(OUT/'verification.json',proof)
    report='''# 受理済みの暗いコアに連なる淡い部分の回収

新しい冊の上端損傷候補で、暗さ差10の支持が上の枝を分断していた。固定した受理コアと同じ領域・保護を維持し、暗さ差6／4の連結支持だけを加えた。全25ページ・固定59ROIと旧追加8成分の計67ROIを開発評価した。全ページ判定を先に行いROI名・対照ラベルを使っていない。画素GTはなく独立精度の評価ではない。

L上限80、3px閉処理、固定コアとの重なり32画素、面積128〜全体1%、長辺は短辺20%以内。安全領域上端16pxだけの接触は許可し、左右端・本文側の接触を拒否。新しい膨張・再帰的なコア更新・印影保護解除はない。診断段階のL上限85は採用せず、元の80を維持した。

**差6で新しい枝状候補の上部も大きく薄くなった。ただし薄い縁・平坦な補修跡と、右上の横枝や別ページの左枝は残る。** 差4はさらに面積が増えるが、古い斑点に大きく平坦な色塊が生じる。差6を次の狭い開発候補に残し、差4は追加採用しない。差6も全資料への採用ではない。

|対象|差6：追加編集／そのうち補間|差4：追加編集／そのうち補間|所見|
|---|---:|---:|---|
|新しい枝・00004_2|3,249／3,249|5,443／5,443|上の両枝が薄くなるが紙色の補修跡が残る|
|同じ冊の枝・00002_2|1,351／1,351|1,665／1,665|右上の枝が薄くなる。濃い左上の枝は残る|
|前の細い裂けROI|0／0|133／133|差4の広がりでも見た目への効果は小さい|
|前の茶色い斑点ROI|692／0|2,179／2,169|差4で旧候補もつながり計7,808画素を新規補間。塗り跡が悪化|
|広いシミROI|0／0|5／5|広いシミは解決しない|
|孤立茶色跡ROI|0／0|20／20|見た目の差は小さい|
|茶色い印影対照|0／0|1／1|差4は場・模様の再計算で出力104画素も変わる|

旧原因不明の筆画状5成分があるページにも4画素を追加し、その周辺の2切り出しの出力319／307画素が変わる。切り出しは重なるので合計しない。もともと回収した5曲線の正誤は未確定で、この問題を解決したとは扱わない。本文・上余白の大きな実筆画の対照は追加0で前回出力と一致。

同じ16境界組の紙色補間と既知紙面の細かい明度模様を使い、追加領域を参照から外して場・模様を再計算した。色模様・粗い模様・分布紙色・境界8組の試作は併用しない。旧14ページの固定ROIは紙面補間以外に保存済みDDRMを残すが、その部分は今回不変。新しい冊と旧追加成分は未補間部分が原画像。新しいDDRM推論・学習は行っていない。

- [枝の上部の回収](01_extra_top_branch.png)
- [別ページの枝と未回収の左枝](02_extra_second_branch.png)
- [以前の細い裂け](03_unseen_thin_crack.png)
- [差4で塗り跡が増える斑点](04_unseen_brown_spot.png)
- [広いシミ](05_extra_brown_stain.png)
- [孤立した跡](06_extra_isolated_marks.png)
- [印影対照](07_new_brown_stamp.png)
- [原因不明の全5成分](08_unknown_old_components.png)
- [全比較PNGのZIP](comparison_pngs.zip)
- [直接残差・観測模様・不変の検証](verification.json)

マスクの領域・保護・元の編集、全67ROIの原寸入力・保存基準・追加参照除外・4近傍残差・明度模様の元画素・合成・範囲外不変を別に確認した。出力画素の変化は新しい編集数だけに限らず、既存の補間領域の色・模様の再計算にも及ぶ。

次は残った濃い左枝が、既存の保護で止まるのか、固定コアとの接続がないためかを測定する。差6の場・模様を固定し、残った縁の未編集と紙面の色差を分ける。未使用の冊群は残り8で白黒資料と図の比率が高く、今回の調整25ページは開発集合として保持する。
'''
    (OUT/'report.md').write_text(report)
    pngs=sorted(OUT.rglob('*.png'))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==len(pngs) and all(n.endswith('.png') for n in z.namelist())
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=len(pngs),
        dependencies={str((folder/name).relative_to(ROOT)):sha(folder/name) for folder in [prior.mask.OUT,prior.OUT] for name in ['manifest.json','complete.json']},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered anchored pale margin:',len(pngs),'PNGs',flush=True)


if __name__=='__main__':main()
