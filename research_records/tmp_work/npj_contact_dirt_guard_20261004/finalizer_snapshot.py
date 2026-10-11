"""Audit guard/color trials, preserve failure, and compose verified local edits."""
import json
from pathlib import Path
import re
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT,read,write,sha,dump
from package_saved_npj_role_separated_review import figure
from finalize_npj_full_page_layout import panel_audit
import probe_npj_dirt_clean_reference as renderer

OLD=ROOT/'tmp_work/npj_contact_dirt_guard_20261004'
OUT=ROOT/'tmp_work/npj_contact_dirt_guard_v2_20261004'
LAYOUT=ROOT/'tmp_work/npj_full_page_layout_v4_20261004'
MODES=['no_guard','ridge_guard','ridge_and_colour']


def lab(a):return cv2.cvtColor(a.astype(np.float32)/255,cv2.COLOR_BGR2LAB)


def compare(path,arrays,labels,title,notes,overview=False):
    figure(arrays,labels,path,title,notes,overview=overview)
    panel_audit(path,arrays,overview)


def audit(folder,chroma):
    manifest=json.loads((folder/'manifest.json').read_text())
    for rel,digest in manifest['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(folder/'source_snapshot.py')==manifest['source_sha256']
    if chroma:
        assert sha(folder/'entrypoint_snapshot.py')==manifest['settings']['wrapper_sha256_before_run']
    records=[]
    for c in manifest['cases']:
        n=folder/'native'/c['id']
        if not (n/'ridge_and_colour/repaired.png').exists():continue
        base=read(n/'baseline.png');original=read(ROOT/c['original'])
        x,y,w,h=c['xywh'];sl=np.s_[y:y+h,x:x+w]
        assert np.array_equal(original[sl],read(n/'original.png'))
        expected=read(ROOT/c['saved_npj'])
        if c.get('baseline_full_page'):expected=expected[sl]
        assert np.array_equal(base,expected)
        mandatory=read(n/'mandatory.png',True);guard=read(n/'guard.png',True)
        broad=read(n/'broad.png',True);candidate=read(n/'candidate.png',True)
        domain=read(n/'domain.png',True);colour=read(n/'colour.png',True)
        assert not (candidate&(~broad|mandatory)).any()
        row=dict(id=c['id'],original_and_baseline_exact=True,methods={})
        for mode in MODES:
            m={k:read(n/mode/(k+'.png'),True) for k in ['update','reference','applied','used_reference']}
            expected=candidate.copy()
            if mode!='no_guard':expected&=~guard
            if mode=='ridge_and_colour':expected&=colour
            assert np.array_equal(expected,m['update'])
            assert not (m['applied']&~m['update']).any()
            assert not (m['used_reference']&~m['reference']).any()
            known,_=renderer.paper_reference(base,m['update'],domain,mandatory)
            median=np.median(lab(base)[:,:,1:][known],axis=0) if known.any() else None
            if chroma and known.any():known &= np.linalg.norm(lab(base)[:,:,1:]-median,axis=2)<=4
            assert np.array_equal(known,m['reference'])
            repaired=read(n/mode/'repaired.png')
            assert np.array_equal(repaired[~m['applied']|mandatory],base[~m['applied']|mandatory])
            if mode!='no_guard':assert np.array_equal(repaired[guard],base[guard])
            after=lab(repaired);before=lab(base)
            green=m['applied']&(after[:,:,0]>60)&(after[:,:,1]<-8)
            row['methods'][mode]=dict(applied_pixels=int(m['applied'].sum()),
                 changed_pixels=int(np.any(repaired!=base,2).sum()),
                 dark_to_bright_pixels=int((m['applied']&(before[:,:,0]<30)&(after[:,:,0]>60)).sum()),
                 bright_green_in_applied=int(green.sum()),mandatory_exact=True,outside_applied_exact=True,
                 reference_selection_independently_recomputed=True)
        arrays=[read(n/'original.png'),base,read(n/'previous.png'),
                read(n/'no_guard/repaired.png'),read(n/'ridge_guard/repaired.png')]
        panel_audit(folder/(c['id']+'_comparison.png'),arrays,True)
        records.append(row)
    return manifest,records


def freeze(folder,report,records,partial=False):
    assert not (folder/'complete.json').exists(),'Already frozen'
    (folder/'report.md').write_text(report)
    dump(folder/'independent_verification.json',dict(cases=records,
         complete_planned_run=not partial,partial_run_due_to_out_of_bounds_control=partial,
         source_dependencies_unchanged=True,native_masks_and_reference_audited=True,
         pixel_GT=False,adopted=False))
    dump(folder/'adoption_gate.json',dict(adopted=False,
         reason='Residual defect ridges retained; faint-writing controls change; no material GT',
         no_stroke_generation=True,baseline_npj_v6_preserved=True))
    shutil.copyfile(__file__,folder/'finalizer_snapshot.py')
    pngs=sorted(folder.glob('*.png'))
    with zipfile.ZipFile(folder/'comparison_pngs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,p.name)
    with zipfile.ZipFile(folder/'comparison_pngs.zip') as z:
        assert z.testzip() is None
        for p in pngs:assert z.read(p.name)==p.read_bytes()
    for rel in re.findall(r'\]\(([^)]+)\)',report):assert (folder/rel).exists(),rel
    dump(folder/'complete.json',dict(complete_planned_run=not partial,
         outputs={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
    for rel,digest in json.loads((folder/'complete.json').read_text())['outputs'].items():assert sha(folder/rel)==digest,rel
    print('Frozen',folder.name,len(pngs),'PNG',len(records),'frames','partial' if partial else 'complete',flush=True)


def main():
    assert not (OUT/'complete.json').exists() and not (OLD/'complete.json').exists()
    old_manifest,old_records=audit(OLD,False)
    manifest,records=audit(OUT,True)
    assert len(old_records)==9 and len(records)==11
    # Candidate comparisons are exact for the nine identically framed inputs.
    for r in old_records:
        for mode in MODES:
            for key in ['update']:
                assert np.array_equal(read(OLD/'native'/r['id']/mode/(key+'.png'),True),
                                      read(OUT/'native'/r['id']/mode/(key+'.png'),True))
    old_green=next(r for r in old_records if r['id']=='saved_hnsd')['methods']['ridge_guard']['bright_green_in_applied']
    new_green=next(r for r in records if r['id']=='saved_hnsd')['methods']['ridge_guard']['bright_green_in_applied']
    assert old_green==3429 and new_green==0
    combinations=[]
    for c in manifest['cases']:
        if not c.get('previous_output'):continue
        n=OUT/'native'/c['id'];p=LAYOUT/'native'/c['id']
        base=read(n/'baseline.png');previous=read(n/'previous.png')
        prior_applied=read(p/'applied.png',True)
        applied=read(n/'ridge_guard/applied.png',True)
        repaired=read(n/'ridge_guard/repaired.png')
        additional=applied&~prior_applied
        combined=previous.copy();combined[additional]=repaired[additional]
        assert np.array_equal(combined[prior_applied],previous[prior_applied])
        assert np.array_equal(combined[~(prior_applied|additional)],base[~(prior_applied|additional)])
        for k,a in dict(combined=combined,additional_applied=additional).items():write(n/(k+'.png'),a)
        before,after=lab(previous),lab(combined)
        interior=read(p/'interior.png',True)
        combinations.append(dict(id=c['id'],additional_applied=int(additional.sum()),
             additional_inside_frame=int((additional&interior).sum()),
             additional_dark_to_bright=int((additional&(before[:,:,0]<30)&(after[:,:,0]>60)).sum()),
             additional_dark_to_bright_inside=int((additional&interior&(before[:,:,0]<30)&(after[:,:,0]>60)).sum()),
             prior_applied_preserved_exact=True,not_adopted=True))
    dump(OUT/'compositions.json',dict(cases=combinations,
         recipe='Keep previous frozen layout edit exactly; only apply guarded chroma-qualified fill outside its applied mask',
         no_new_selection_thresholds=True,no_solver_rerun=True,automatic_adoption=False))
    n=OUT/'native/saved_hnsd';arrays={k:read(n/(k+'.png')) for k in ['original','baseline','previous','combined']}
    current=read(n/'ridge_guard/repaired.png');old=read(OLD/'native/saved_hnsd/ridge_guard/repaired.png')
    s=np.s_[240:385,250:600]
    compare(OUT/'focus_inside_frame_dirt.png',[arrays[k][s] for k in ['original','baseline','previous','combined']],
            ['原画像','保存npjと一致対照','前回：枠外だけ','枠外を維持＋線保持で追加'],
            '本文側の黒い塊：観測された線を保持しながら局所除去',
            ['追加は原画像の線保持外。文字生成なし。塊はなお残り、材質のGTもない。',
             'この拡大に対応する候補選別は全原ページの特徴から行う。'])
    s=np.s_[380:550,350:560]
    compare(OUT/'focus_reference_colour_failure.png',[arrays['original'][s],arrays['baseline'][s],old[s],current[s]],
            ['原画像','同じnpj基準','初版：異色参照が入る','参照の色条件を追加'],
            '緑の点の原因：保存npjの異色画素を紙面参照に混ぜる',
            ['同じ汚れ候補・保持・補間。参照の色条件だけ変更。',
             '強い緑の補修画素はhnsd guarded全範囲3,429→0。黒い残存は解決していない。'])
    s=np.s_[50:310,30:190]
    compare(OUT/'focus_hnsd_text_control.png',[arrays['original'][s],arrays['baseline'][s],arrays['combined'][s]],
            ['原画像','保存npj基準','追加汚れ補修'],
            'hnsdの既知本文対照：この範囲の追加変更0',
            ['この対照での保持確認。全本文の保存や文字の正解率ではない。'])
    assert np.array_equal(arrays['combined'][s],arrays['baseline'][s])
    # Expose every changed component in the thin-writing control, not a selected success.
    n=OUT/'native/old_full_page'
    ori=read(n/'original.png');base=read(n/'baseline.png');after=read(n/'ridge_guard/repaired.png')
    changes=np.any(after!=base,2)
    _,labels,stats,_=cv2.connectedComponentsWithStats(changes.astype(np.uint8),connectivity=8)
    assert len(stats)-1==4 and int(changes.sum())==104
    witness=[]
    for i in range(1,len(stats)):
        x,y,w,h,count=map(int,stats[i]);x0=max(0,x-24);y0=max(0,y-24)
        x1=min(base.shape[1],x+w+24);y1=min(base.shape[0],y+h+24);s=np.s_[y0:y1,x0:x1]
        filename=f'thin_control_change_{i:02d}.png'
        compare(OUT/filename,[ori[s],base[s],after[s]],['原画像','保存npj','線保持で補修'],
                f'薄い文字の対照：変更成分{i}、{count}画素',
                ['変更は汚れ・筆画端・裏写りのいずれか未確定。成功と数えない。'])
        witness.append(dict(filename=filename,crop_xywh=[x0,y0,x1-x0,y1-y0],changed_pixels=count,
                            observed_original_class='unconfirmed; includes writing surroundings'))
    dump(OUT/'semantic_witnesses.json',dict(thin_control_all_changed_components=witness,
         hnsd_text_control_change_pixels=0,hnsd_guarded_bright_green_before=old_green,
         hnsd_guarded_bright_green_after=new_green,pixel_GT=False))
    old_report='''# 枠内の汚れ選別・初版の失敗記録

2026-10-04。11範囲を計画し、9範囲の比較まで生成した後、朱印対照の切り出しが原画像幅1784を96px越える設定を検出して実行が終了した。全11例の完了・検証ではない。保存された9範囲を独立に監査し、失敗版として凍結する。

同じ広い暗部とnpj黒残留を使い、原画像の中性線状応答と暖色接続領域を保持する／しない／茶色色差条件を加える比較。線保持なしは本文を大きく削除する。線保持側も虫食いの縁を線として残し、汚れの全面除去にはならない。

hnsdの保持付き補修で、明るさ条件を通過した強い緑色のnpj境界画素を参照して緑の点が発生した。補修範囲のLab L>60かつa<-8は3,429画素。保存npjの外側の画素を補間に用いる際の異色参照であり、RGB/BGRの取り違えや新しい拡散モデルの出力ではない。入力色の全体補正は行わない。

[hnsdの比較](saved_hnsd_comparison.png)・[本文消失の対照](fresh_lower_paper_comparison.png)・[27比較PNG ZIP](comparison_pngs.zip)。[修正版](../npj_contact_dirt_guard_v2_20261004/report.md)では朱印対照の幅を修正し、汚れ候補を固定したまま異色参照を除いた。既存出力は変更していない。
'''
    rows=['| 範囲 | 線保持なし：暗→明 | 線保持：適用 | 線保持：暗→明 | 線保持＋色条件：適用 |','| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        m=r['methods'];col=next(d for d in json.loads((OUT/'diagnostics.json').read_text())['cases'] if d['id']==r['id'])
        rows.append(f"| [{r['id']}]({r['id']}_comparison.png) | {m['no_guard']['dark_to_bright_pixels']:,} | {m['ridge_guard']['applied_pixels']:,} | {m['ridge_guard']['dark_to_bright_pixels']:,} | {m['ridge_and_colour']['applied_pixels']:,} |")
    body=next(c for c in combinations if c['id']=='saved_hnsd')
    report=f'''# 枠内の残留汚れ：原画像の線保持と紙面参照の色条件を比較

2026-10-04。**hnsdの本文側に残る黒い塊を一部減らせたが、線保持は汚れの縁も保持する。薄い文字対照に小変更も残り、一般採用はしない。** 新しいDDRM・筆画補完・学習は行っていない。

まず見る画像：

- [本文側の汚れ：原画像／npj／前回／追加補修](focus_inside_frame_dirt.png)
- [hnsdの本文保持対照・追加変更0](focus_hnsd_text_control.png)
- [緑の点が発生した失敗と参照修正](focus_reference_colour_failure.png)
- [薄い文字の追加変更1](thin_control_change_01.png)・[2](thin_control_change_02.png)・[3](thin_control_change_03.png)・[4](thin_control_change_04.png)
- [実保存npjに残る朱印の保持対照](known_200010454_v2_comparison.png)

## 汚れと文字保持を分けた結果

前回は本文外包を全部保持したため、枠内の塊に手を付けなかった。今回の比較は、全原ページから推定した広い暗部があり、対応する保存npjに黒い残留がある成分を選び、原画像に観測される中性の線・暖色ネットワークを保持する。文字を生成し戻す処理はない。

線保持なしでは実際の本文や罫線を大きく消すため不採用。線保持はこの失敗を抑えるが、欠損の縁や汚れの細部も線に見えて残る。汚れの黒い構造が全て消えたわけではない。薄い文字のold_full_page対照では104画素が4成分に分かれて変更されており、全成分を拡大保存した。筆画端・裏写り・汚れの区別が未確定なので、無誤編集を主張しない。

茶色色差を追加するとhnsdの保持付き適用は8,191→0になる。原画像の濃いシミが紙色と近いため、茶色色差だけで扱えない。これは同じ候補へ色条件を追加したアブレーションで、以前の全成分分類器の再現ではない。

{chr(10).join(rows)}

暗→明は適用画素のLab L<30→L>60。汚れGT精度や本文GT精度ではない。編集0も解決ではない。全11範囲は既知の開発ページであり、独立した未使用冊の評価ではない。イラストは対象外。

## 補間色の失敗を分離

初版のhnsdでは明るさ条件を通る異色参照があり、線保持付き適用範囲に明るい緑（Lab L>60,a<-8）を3,429画素作った。元の参照831画素のうち264画素は、明るさで選んだ紙面候補のab中央値から距離4を越えていた。保存npjには局所的な異色画素があり、文字用復元の出力すべてを紙面参照にしてよいとは限らない。

修正版は明るさで認めた紙面候補のab中央値との距離4以内の参照だけを残す。9共通範囲の汚れ候補マスクは全画素不変。同じ補間・線保持条件で、hnsd保持付き補修の緑画素は3,429→0。色出力の後補正ではなく、補間へ入れる参照の限定である。参照が不足した成分は黒い基準を維持するため、適用量は11,230→8,191へ減る。黒が残ることを成功として隠さない。

## 前回の効果を保持した組合せ

前回の枠外補修の適用範囲を全画素保持し、その外だけへ今回の線保持付き出力を追加した。候補閾値・補間器の追加調整はしていない。hnsdの追加適用は{body['additional_applied']:,}画素、本文外包内は{body['additional_inside_frame']:,}画素、そのうち暗→明は{body['additional_dark_to_bright_inside']:,}画素。既知の左本文対照は追加変更0。ただし塊・周囲の小片が残り、他の本文を全て保持できた保証はない。

虫食いページは今回の線保持だけだと縁を残し、前回より悪い。組合せは前回の適用範囲を維持するので、その局所的な除去を戻さない。前回の枠推定も一般化未確定なため、組合せを完成した手法として採用しない。

## 入力と固定条件

最初の7範囲は既に保存した固定ローカルnpjコードの新規crop対照。hnsdは歴史的保存とその対照が画素一致。追加3範囲は原画像と実推論入力が全画素一致する実保存npjの切り出し。追加推論なし。hnsdで元入力の濃淡が異なる問題は、原画素対照と保存出力の一致を確認済みであり、濃淡差だけを黒い塊の原因としない。

広い暗部は前回同様sigma8輝度差5／8、元画像b≥5、成分面積128・seed32・半径4、npj残差32画素かつ3%。そこからnpj L<30と2px周囲だけへ編集を限定。線保持は以前のsigma1.5／3／6・応答1.5・中性色条件と4px周囲、暖色接続保持は5px closingと2px周囲。朱・extra・CSC朱は必須保持。白黒2例は元のb条件のため未解決。

初版は朱印対照の幅を越える切り出しで9例後に終了した。[初版の記録](../npj_contact_dirt_guard_20261004/report.md)。修正版は範囲を変更前に確認して11例を完了。共通9例で候補マスク一致を確認した。新しい参照条件はこの既知失敗を見て設計したので独立した色評価ではない。

[全40比較PNG ZIP](comparison_pngs.zip)・[固定条件と入力](manifest.json)・[独立画素監査](independent_verification.json)・[組合せ範囲](compositions.json)・[未確定な変更の全拡大](semantic_witnesses.json)。

次は、原画像で紙面と確かめられる領域と、実際の筆画がある領域の小さな固定注釈を用い、本文側の追加除去と誤編集を別に測る。形だけで線を全部保持する方法では欠損縁が残り、形を無視すると本文が消える。この両方を判定できる条件が必要。文字の生成・補完を目的にはしない。
'''
    assert len(list(OUT.glob('*.png')))==40
    # Freeze corrected report before old one: their cross-links must both exist.
    (OLD/'report.md').write_text(old_report);(OUT/'report.md').write_text(report)
    freeze(OLD,old_report,old_records,partial=True)
    freeze(OUT,report,records)


if __name__=='__main__':main()
