"""Verify frozen paper fields with a separate discrete residual calculation."""
import gc
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_known_paper_harmonic_fill_v2 as probe
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT=probe.ROOT
PILOT=probe.OUT
OUT=ROOT/'tmp_work/known_paper_harmonic_fill_review_20261001'
FIRST=ROOT/'tmp_work/known_paper_harmonic_fill_pilot_20261001'


def verify_field(field,applied,reference,target):
    """Evaluate neighbor differences directly, without rebuilding solver matrix."""
    H,W=applied.shape;yy,xx=np.nonzero(applied)
    residual=np.zeros((len(yy),3),np.float64);used=np.zeros_like(reference)
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny,nx=yy+dy,xx+dx;valid=(ny>=0)&(ny<H)&(nx>=0)&(nx<W)
        i=np.flatnonzero(valid);oy,ox=ny[valid],nx[valid]
        unknown=applied[oy,ox];paper=reference[oy,ox]
        j=i[unknown];residual[j]+=field[yy[j],xx[j]]-field[oy[unknown],ox[unknown]]
        j=i[paper];residual[j]+=field[yy[j],xx[j]]-target[oy[paper],ox[paper]]
        used[oy[paper],ox[paper]]=True
    error=float(np.abs(residual).max()) if len(yy) else 0.
    assert error<1e-3,error
    return error,used


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [FIRST,PILOT,probe.previous.OUT,probe.previous.mask_probe.OUT,probe.telea.OUT]:check_bundle(folder)
    # A known constant paper must remain constant, even with black missing pixels.
    synthetic=np.full((32,32,3),(90,3,12),np.float32)
    active=np.zeros((32,32),bool);active[10:22,10:22]=True
    synthetic[active]=0
    fields,applied,_,_=probe.solve_fields(synthetic,~active,active)
    assert np.array_equal(applied,active)
    for field in fields.values():assert np.allclose(field[active],(90,3,12),atol=1e-4)
    _,empty,_,_=probe.solve_fields(synthetic,np.zeros_like(active),active)
    assert not empty.any()
    rows=json.loads((PILOT/'diagnostics.json').read_text());lookup={r['id']:r for r in rows}
    assert len(rows)==len(lookup)==22
    checked=[];residuals=[];books=set();page_count=0;summaries=[]
    for source in probe.previous.mask_probe.old.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key=page['page'];page_count+=1;books.add(page['book'])
            original=read(ROOT/page['original']);H,W=original.shape[:2]
            directory=probe.previous.mask_probe.OUT/'pages'/key
            edit=read(directory/'completed/edit.png',True)
            domain=read(directory/'features/safe_domain.png',True)
            guard=read(directory/'features/long_line_guard.png',True)
            anchor=read(directory/'features/warm_anchor_guard.png',True)
            eligible=read(PILOT/'reference_pages'/key/'eligible.png',True)
            assert np.array_equal(eligible,edit&domain&~guard&~anchor)
            reference_full=read(PILOT/'reference_pages'/key/'paper.png',True)
            assert not np.any(reference_full&edit)
            protected=np.zeros((H,W),bool)
            for name in ['protect','hold','red','extra']:protected|=read(source/'v6_masks'/key/(name+'.png'),True)
            assert not np.any(reference_full&((protected|anchor)&~edit))
            for roi in page['rois']:
                rid=roi['id'];row=lookup[rid];native=PILOT/'native'/rid
                for path in native.glob('*.png'):
                    assert sha(path)==sha(FIRST/'native'/rid/path.name),path
                x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh']
                context=original[cy:cy+ch,cx:cx+cw];s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
                crop=context[s];local_edit=read(native/'edit.png',True)
                assert np.array_equal(local_edit,edit[y:y+h,x:x+w])
                assert np.array_equal(local_edit,read(probe.previous.OUT/'native'/rid/'completed_edit.png',True))
                assert np.array_equal(read(native/'original.png'),crop)
                baseline=read(native/'ddrm.png')
                assert np.array_equal(baseline,read(probe.previous.OUT/'native'/rid/'completed.png'))
                assert np.array_equal(read(native/'telea.png'),read(probe.telea.OUT/'native'/rid/'completed_telea.png'))
                known=read(native/'reference_context.png',True)
                applied=read(native/'applied_context.png',True)
                active=read(native/'eligible_context.png',True)
                assert np.array_equal(active,eligible[cy:cy+ch,cx:cx+cw])
                assert np.array_equal(known,reference_full[cy:cy+ch,cx:cx+cw])
                assert not np.any(known&edit[cy:cy+ch,cx:cx+cw])
                assert not np.any(applied&~active)
                assert np.array_equal(applied[s],read(native/'applied.png',True))
                lab=cv2.cvtColor(context.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
                weight=cv2.GaussianBlur(known.astype(np.float32),(0,0),6)
                smooth=cv2.GaussianBlur(np.where(known[:,:,None],lab,0),(0,0),6)/np.maximum(weight[:,:,None],1e-6)
                for mode,target in [('boundary_raw',lab),('boundary_smooth',smooth)]:
                    field=np.load(native/(mode+'_field_Lab.npy'),allow_pickle=False)
                    error,used=verify_field(field,applied,known,target)
                    assert np.array_equal(used,read(native/'used_boundary_context.png',True))
                    assert not np.any(used&~known)
                    output=read(native/(mode+'.png'))
                    assert not np.any(np.any(output!=baseline,axis=2)&~applied[s])
                    assert not np.any(np.any(output!=crop,axis=2)&~local_edit)
                    converted=probe.previous.blending.previous.from_lab(field[s])
                    assert np.array_equal(output[applied[s]],converted[applied[s]])
                    residuals.append(dict(id=rid,mode=mode,max_discrete_residual=error))
                # Recompute diagnostic brightness on exactly the same applied set.
                for mode in ['original','ddrm','telea']+probe.MODES:
                    value=read(native/(mode+'.png'));light=cv2.cvtColor(value.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
                    stats=row['diagnostics'][mode]
                    assert stats['pixels']==int(applied[s].sum())
                    assert stats['L_le_80_pixels']==int(((light<=80)&applied[s]).sum())
                    if applied[s].any():assert stats['median_L']==float(np.median(light[applied[s]]))
                overlay=crop.copy();ref_roi=known[s];target_roi=applied[s]
                overlay[ref_roi]=np.rint(.65*overlay[ref_roi]+.35*np.array([150,210,80])).astype(np.uint8)
                overlay[target_roi]=np.rint(.65*overlay[target_roi]+.35*np.array([20,200,255])).astype(np.uint8)
                unavailable=active[s]&~target_roi
                overlay[unavailable]=np.rint(.65*overlay[unavailable]+.35*np.array([190,50,190])).astype(np.uint8)
                contours,_=cv2.findContours(local_edit.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                boundary=crop.copy();cv2.drawContours(boundary,contours,-1,(0,0,255),1)
                probe.previous.prior_ddrm.figure([crop,boundary,overlay],['原画像','赤線＝固定の編集境界','緑＝参照・黄＝補間・紫＝未適用'],
                    OUT/'reference_diagnostic'/(rid+'.png'),row['title']+' · 補間の範囲と参照',
                    f'補間 {row["applied_pixels"]:,} / 適用候補 {row["eligible_pixels"]:,}画素。候補外・参照不足は保存DDRMを維持。',
                    '表示用の色。参照条件を満たす紙面であり、画素GTではない。')
                summaries.append(dict(id=rid,edit_pixels=int(local_edit.sum()),eligible=row['eligible_pixels'],applied=row['applied_pixels']))
                checked.append(rid)
            del original,edit,eligible,reference_full,protected,domain,guard,anchor;gc.collect()
    assert page_count==14 and len(books)==11 and len(checked)==22
    overview=[]
    for filename,keys,group,folder,title,notes in [
        ('01_hole_fill.png',['new_margin_hole','hole_edge_new'],'comparisons',PILOT,'固定マスクで紙色から埋める',
         ['DDRMの暗い模様を引き継がない。直接境界と平滑境界を分けて比較。',
          '暗い生成模様は抑えられるが、平坦な塗り跡・紙色差・マスク外の薄い縁は残る。']),
        ('04_controls.png',['new_frame_text','new_header_ink','new_mono_text'],'comparisons',PILOT,'枠・見出し・本文の対照',
         ['紙色補間の適用0。今回の2方式とも保存済みDDRM出力と完全一致。']),
        ('05_stamp_and_paper.png',['new_brown_stamp','brown_false_red_new'],'comparisons',PILOT,'印影と広い紙色ムラの対照',
         ['印影の小さな点の消失は以前のマスクに由来する。今回もマスクを拡張しない。',
          '広い紙色ムラは参照不足でほぼ未適用。補間で解決したと数えない。']),
        ('06_reference_scope.png',['new_margin_hole','hole_edge_new'],'reference_diagnostic',OUT,'紙面の参照と補間範囲',
         ['緑は参照候補、黄は補間、紫は適用候補だが参照不足・小成分などで未適用。'])]:
        stack([folder/group/(key+'.png') for key in keys],OUT/filename,title,notes);overview.append(filename)
    crops={}
    for filename,rid in [('02_new_hole_closeup.png','new_margin_hole'),('03_old_hole_closeup.png','hole_edge_new')]:
        native=PILOT/'native'/rid
        selected=read(probe.previous.mask_probe.OUT/'native'/rid/'completed_added.png',True) if rid=='new_margin_hole' else read(native/'applied.png',True)
        yy,xx=np.nonzero(selected);H,W=selected.shape
        x0,y0=max(0,int(xx.min())-12),max(0,int(yy.min())-12);x1,y1=min(W,int(xx.max())+13),min(H,int(yy.max())+13)
        crops[rid]=[x0,y0,x1-x0,y1-y0]
        images=[read(native/(m+'.png'))[y0:y1,x0:x1] for m in ['original','ddrm','telea']+probe.MODES]
        probe.previous.prior_ddrm.figure(images,['原画像','DDRM','Telea','紙色補間・直接境界','紙色補間・平滑境界'],
            OUT/filename,rid+' · 原寸画素の拡大','同じ切り出しを最近傍で拡大。紙模様を新しく追加していない。',
            '暗い線状模様と、補修の平坦さ・残存輪郭を区別して見る。');overview.append(filename)
    rid='new_margin_hole';native=PILOT/'native'/rid;old=probe.previous.prior_ddrm.OUT/'native'/rid
    images=[read(p) for p in [native/'original.png',old/'npj.png',old/'v6.png',native/'ddrm.png',native/'boundary_raw.png',native/'boundary_smooth.png']]
    probe.previous.prior_ddrm.figure(images,['原画像','npj → DDRM','v6 → DDRM','今回マスク → DDRM','紙色補間・直接境界','紙色補間・平滑境界'],
        OUT/'07_npj_v6_context.png','保存済みnpj・固定v6との位置づけ',
        '保存出力を再利用。紙色補間はDDRMではなく、マスクの紙面対象だけに適用する。',
        'npjは編集候補をDDRMへ接続した比較で、元npj全工程の再現ではない。');overview.append('07_npj_v6_context.png')
    items=[(OUT/name,'overview/'+name) for name in overview]
    items.extend((PILOT/'comparisons'/(r['id']+'.png'),'comparisons/'+r['id']+'.png') for r in rows)
    items.extend((OUT/'reference_diagnostic'/(r['id']+'.png'),'reference_scope/'+r['id']+'.png') for r in rows)
    archive_path=OUT/'known_paper_fill_comparison_images.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path,name in items:archive.write(path,name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist())==len(items)==51 and archive.testzip() is None
        for path,name in items:assert hashlib.sha256(archive.read(name)).hexdigest()==sha(path)
    unchanged=[r['id'] for r in rows if not r['applied_pixels']];assert len(unchanged)==18
    brightness='\n'.join('| `'+r['id']+'` | '+str(r['applied_pixels'])+' | '+' | '.join(str(r['diagnostics'][m]['L_le_80_pixels']) for m in ['ddrm','telea']+probe.MODES)+' |' for r in rows if r['applied_pixels'])
    all_table='\n'.join(f'| [{lookup[r["id"]]["title"]}](../{PILOT.name}/comparisons/{r["id"]}.png) · `{r["id"]}` | {r["edit_pixels"]:,} | {r["eligible"]:,} | {r["applied"]:,} |' for r in summaries)
    report=f'''# 固定マスクの紙面を、未編集紙色だけから補間する

2026-10-01。**問題の虫食い領域では、DDRMの灰色の線状模様を引き継がずに埋められた。一方、平坦な塗り跡・紙色差・薄い輪郭が残り、自然な紙面としての完成はまだ未達。** 前回の全成分回収マスクを固定し、復元方法だけを比較した。

全14ページ・11冊・22箇所を同じ開発例として保持した。比較するのは保存済みDDRM、同じマスクのTelea、紙色補間の直接境界と平滑境界。紙色補間・TeleaはDDRMではない。マスクの再計算・拡張、モデルの再学習、新しいDDRM推論はしていない。

## 比較PNG

- [虫食い2例の復元比較](01_hole_fill.png)。
- [新しい穴の上方の拡大](02_new_hole_closeup.png)、[既存の穴の拡大](03_old_hole_closeup.png)。全条件で同じ切り出し・最近傍拡大を使う。
- [枠・見出し・本文](04_controls.png)、[印影と広い紙色ムラ](05_stamp_and_paper.png)。
- [参照紙面と補間範囲](06_reference_scope.png)：緑＝参照、黄＝補間、紫＝適用候補だが未適用。赤線＝固定の編集境界。
- [保存済みnpj・固定v6からの比較](07_npj_v6_context.png)、[全51PNG ZIP](known_paper_fill_comparison_images.zip)。ZIPは概要7、全ROI復元比較22、全ROIの範囲診断22。

## 変更した復元条件

前回の[上端接触の回収](../margin_top_completion_review_20261001/report.md)で保存した全成分マスクをそのまま使う。紙色補間の候補は、その編集領域と固定の上余白安全領域の共通部分で、長い横線の保護幅と暖色接続保護を除いた領域。元の編集マスクは全14ページで固定し、どこを紙色補間に渡すかだけをこの条件で判定する。ROIの意味ラベルや「穴」の座標で適用領域を作っていない。

この位置・保護条件は文字がないことの証明ではない。今回も印影の以前からの誤編集箇所が候補に入る。したがって、未知資料の余白へ自動的に紙色補間を適用する仕組みが確立したとは扱わない。

参照は元の編集領域**全体の外**から、以前と同じ局所Lab紙色条件で絞り、残存インク保護と暖色保護を除く。全ページで参照条件を計算し、出力の補間はROIの周囲64pxの文脈で行う。4近傍でつながる64画素以上の成分を対象にし、16組以上の隣接紙面の境界対がある場合だけ補間する。参照不足や小さな成分は変更せず、保存DDRM出力を維持する。

編集内部のLabを4近傍の調和補間で求める。直接境界は未編集紙面のLabをそのまま境界値にする。平滑境界は参照に指定した紙面だけの正規化Gaussian（σ6px）で境界値を作る。参照条件に入らない隣接画素と文脈端からは値を取り込まない。DDRMによる初期値・拘束・低周波・高周波模様は補間に使わず、生成模様へのフェザーも行わない。

これは保存DDRMを基準にした**一部領域の置き換え**である。紙色補間の適用部分はDDRMから独立するが、未適用の編集領域には保存DDRMが残る。TeleaとDDRMは元の編集範囲全体を扱うため、範囲の差も次の表と診断PNGで示す。編集範囲外はどの条件でも原画素と一致する。紙の粒や模様は今回追加していない。

## 暗い模様と見た目

新しい穴では補間対象14,038画素に含まれるL≤80の画素が、保存DDRMの1,092から両補間条件で0になった。灰色の線状模様も拡大PNGで目立たなくなる。これは固定した適用領域での暗さの観測で、除去精度・損傷の正解率・目視の品質点ではない。

| ROI | 共通の適用画素 | DDRMのL≤80 | TeleaのL≤80 | 直接境界のL≤80 | 平滑境界のL≤80 |
| --- | ---: | ---: | ---: | ---: | ---: |
{brightness}

印影の紙面は元々灰色で、周囲の紙色にもL≤80が多い。この555画素を、暗い生成模様が残った穴と同じ失敗として数えない。原画像のインクの点は以前のマスクで既に失われており、今回の補間はその点の復元でも保護の解決でもない。利用者が許容した変化を今回の追加誤編集と扱わない。

既存の穴では直接境界が周辺の黄みを強く引き継ぎ、平滑境界はより明るい塗り跡になる。新しい穴も直接境界と平滑境界で紙色が違い、両方に平坦さと輪郭が残る。σ6pxの参照平滑化による紙色のずれは残る比較事項であり、平滑境界を自動的に最良条件としない。マスク外に残った薄い傷や、保護された小領域は補間だけでは消せない。

広い上端の茶色い紙面では、適用候補15,577画素のうち224画素だけが境界参照の条件を満たした。大部分は保存DDRMのままで、紙色ムラ全体を解決した結果ではない。ほか18箇所も紙色補間を適用せず、2方式とも保存DDRMと完全一致した。条件による対象外であり、未知資料の文字保存精度の証明ではない。

## 全22箇所の範囲

| ROI | 固定編集マスク | 紙色補間の適用候補 | 実際に補間 |
| --- | ---: | ---: | ---: |
{all_table}

## 確認と次の段階

保存入力・マスク・DDRM・Telea・ソースのハッシュと原寸画素を確認した。全参照が編集範囲外であること、固定の適用候補との一致、補間範囲外は保存DDRM、編集範囲外は原画素との一致を確認した。原画像の編集対象を任意の色に置き換えても、また補間部分のDDRMを黒で置き換えても、4適用例の出力は完全一致した。これは参照領域を固定した補間段階の確認であり、参照ゲート自体の正しさの証明ではない。

定数の紙面を欠損の内外で同じ値に復元する確認と、参照がない場合に補間しない確認を実施した。初回の定数テストでは、float32の境界値から微小許容誤差を引く検証式で許容幅が丸められ、正しい解を拒否することが分かった。初回結果とソースを保存し、境界値の比較だけをfloat64へ修正したv2で再確認した。全22箇所の全ネイティブPNGは初回とハッシュが同一で、補間画像の算出は変えていない。

保存された場に対して、行列を組む実装を使わず4近傍の差を足し合わせ、調和方程式の残差を再計算した。解が参照の各Lab成分の最小・最大を超えないことも実行時に確認した。この性質は正しい参照が得られた場合の色の制約であり、紙模様や資料の内容を真正に復元する保証ではない。[確認JSON](verification.json)。ZIP51PNGの内容ハッシュも一致した。

紙色のみの補間を、暗い生成模様を引き継がない**開発用の復元基準**として保存する。マスクをさらに広げず、次は固定した補間結果について、塗り跡の原因である境界の紙色差と内部の紙模様不足を分けて比較する。前回の模様移植は参照の平滑さ・重複平均でさらに平坦化したため、その条件を自動的に足さない。紙面の参照が不足する例と、印影・本文との適用区別も引き続き必要。新手法の確立・全資料への採用はまだ判断しない。

[固定条件](../{PILOT.name}/manifest.json)、[全診断](../{PILOT.name}/diagnostics.json)、[原寸出力](../{PILOT.name}/native)、[補間実装v2](../../scripts/benchmark/probe_known_paper_harmonic_fill_v2.py)、[初回ソース](../../scripts/benchmark/probe_known_paper_harmonic_fill.py)、[別実装による確認・梱包](../../scripts/benchmark/package_known_paper_fill_review.py)。
'''
    (OUT/'report.md').write_text(report)
    dump(OUT/'diagnostics.json',dict(scope=summaries,independent_residuals=residuals,closeup_xywh=crops))
    dump(OUT/'verification.json',dict(status='passed',pages=14,books=11,regions=checked,
        mask_fixed=True,reference_excludes_entire_edit=True,outside_applied_changed_pixels=0,
        outside_edit_changed_pixels=0,unchanged_regions=unchanged,
        independent_discrete_residual=True,residual_max=max(r['max_discrete_residual'] for r in residuals),
        synthetic_constant_paper=True,no_reference_no_fill=True,zip_PNG_count=51,zip_hashes=True))
    dump(OUT/'manifest.json',dict(source_sha256=sha(Path(__file__)),
        dependencies={str((f/n).relative_to(ROOT)):sha(f/n) for f in [FIRST,PILOT,probe.previous.OUT,probe.previous.mask_probe.OUT,probe.telea.OUT]
                      for n in ['manifest.json','complete.json']},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    print('Passed: fixed masks, known-paper references, harmonic residuals; 51 PNGs',flush=True)


if __name__=='__main__':main()
