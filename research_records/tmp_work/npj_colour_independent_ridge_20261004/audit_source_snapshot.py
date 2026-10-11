"""Saved-pixel and semantic-control audit of the colour-independent guard."""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT,read,write,sha,dump,overlay
from finalize_npj_full_page_layout import render,panel_audit
import probe_npj_colour_independent_ridge_guard as experiment

OUT=experiment.OUT


def main():
    assert not (OUT/'complete.json').exists(), 'Already frozen'
    manifest=json.loads((OUT/'manifest.json').read_text())
    diagnostics={c['id']:c for c in json.loads((OUT/'diagnostics.json').read_text())['cases']}
    checked_full=set();records=[]
    for c in manifest['cases']:
        n=OUT/'native'/c['id'];d=diagnostics[c['id']];full=OUT/d['full_geometry']
        if full not in checked_full:
            original=read(ROOT/c['original']);protect=read(ROOT/c['roles']/'protect.png',True)
            maps=experiment.full_geometry(original,protect)
            for key,a in maps.items():assert np.array_equal(a,read(full/(key+'.png'),True))
            checked_full.add(full)
        x,y,w,h=c['xywh'];sl=np.s_[y:y+h,x:x+w]
        original=read(n/'original.png');base=read(n/'baseline.png');prior=read(n/'prior.png')
        before=read(n/'before.png');veto=read(n/'veto.png')
        assert np.array_equal(original,read(ROOT/c['original'])[sl])
        assert np.array_equal(base,read(ROOT/c['native']/'baseline.png'))
        old_update=read(n/'old_update.png',True);domain=read(n/'old_domain.png',True)
        mandatory=read(n/'mandatory.png',True);old_guard=read(n/'old_guard.png',True)
        prior_applied=read(n/'prior_applied.png',True)
        before_added_changed=np.any(before!=prior,2)
        results=[];mchecks={}
        for mode in experiment.MODES:
            f=n/mode;guard=read(f/'guard.png',True)
            m={k:read(f/(k+'.png'),True) for k in ['update','domain','reference','applied','used_reference','additional']}
            repaired=read(f/'repaired.png');combined=read(f/'combined.png')
            assert np.array_equal(guard,read(full/(mode+'.png'),True)[sl])
            assert np.array_equal(m['update'],old_update&~guard)
            assert np.array_equal(m['domain'],domain&~guard)
            assert not (m['applied']&~m['update']).any()
            assert not (m['reference']&(m['update']|mandatory|~m['domain'])).any()
            assert not (m['used_reference']&~m['reference']).any()
            known,t=experiment.reference.paper_reference(base,m['update'],m['domain'],mandatory)
            assert np.array_equal(known,m['reference']) and t==d['methods'][mode]['reference_L_threshold']
            assert np.array_equal(repaired[~m['applied']|guard|old_guard|mandatory],base[~m['applied']|guard|old_guard|mandatory])
            assert np.array_equal(m['additional'],m['applied']&~prior_applied)
            expected=prior.copy();expected[m['additional']]=repaired[m['additional']]
            assert np.array_equal(expected,combined)
            assert np.array_equal(combined[prior_applied],prior[prior_applied])
            # Distinguish genuine guard coverage from solver abstention.
            mchecks[mode]=dict(previous_changed=int(before_added_changed.sum()),
                previous_changes_covered_by_new_guard=int((before_added_changed&guard).sum()),
                changed_from_prior=int(np.any(combined!=prior,2).sum()),
                combined_exact_to_previous=np.array_equal(combined,before),
                original_guard_and_mandatory_exact=True,prior_applied_exact=True,
                reference_recomputed=True)
            results.append(combined)
        panel_audit(OUT/(c['id']+'_comparison.png'),[original,base,before,veto,*results],True)
        records.append(dict(id=c['id'],methods=mchecks))
    byid={r['id']:r for r in records}
    for mode in experiment.MODES:
        assert byid['old_saved_hnsd']['methods'][mode]['combined_exact_to_previous']
        assert byid['old_wormhole']['methods'][mode]['combined_exact_to_previous']
        for cid,expected in [('new_new_warm_lower_loss',37),('new_new_warm_faint_writing',74)]:
            d=byid[cid]['methods'][mode]
            assert d['previous_changes_covered_by_new_guard']==expected
            assert d['changed_from_prior']==0
    # Two full masks explain the causal change without claiming material ground truth.
    for cid,box,name in [
        ('new_new_warm_lower_loss',[615,411,80,79],'guard_explains_stroke_preservation'),
        ('old_saved_hnsd',[250,240,350,150],'guard_keeps_dirt_candidate')]:
        n=OUT/'native'/cid;x,y,w,h=box;sl=np.s_[y:y+h,x:x+w]
        original=read(n/'original.png');old_guard=read(n/'old_guard.png',True)
        new_guard=read(n/'all_colour_ridge/guard.png',True)|old_guard
        added=read(n/'all_colour_ridge/additional.png',True)
        render(OUT/(name+'.png'),[original[sl],overlay(original[sl],[('blue',old_guard[sl])]),
            overlay(original[sl],[('blue',new_guard[sl])]),overlay(original[sl],[('edit',added[sl])]),
            read(n/'all_colour_ridge/combined.png')[sl]],
            ['原画像','青＝従来の線保持','青＝色条件を外した線保持','黄＝今回の追加補修','同じ補修結果'],
            cid+'：保持と補修の範囲',
            ['青は筆画GT、黄は汚れGTではない。原画像に観測される形状の固定条件。',
             '本文の再生成なし。前回枠外補修は、この追加保持の適用対象外。'])
    dump(OUT/'independent_verification.json',dict(cases=records,
        full_geometry_recomputed=len(checked_full),original_crops_exact=True,
        references_recomputed=True,mask_roles_verified=True,panels_verified=True,
        hnsd_and_wormhole_combined_exact_to_previous=True,
        all_111_previous_brown_writing_changes_covered_by_guard=True,
        no_pixel_GT=True,independent_test=False))
    shutil.copyfile(__file__,OUT/'audit_source_snapshot.py')
    report(records)
    print('Verified 17 cases;111 previous stroke-area changes covered;hnsd and wormhole exact',flush=True)


def report(records):
    diagnostics=json.loads((OUT/'diagnostics.json').read_text())['cases']
    lines=['# 茶色い筆画の誤編集を防ぎ、既知の黒残留除去を維持', '',
        '2026-10-04。原画像の線状応答から中性色の制限を外すと、前回の茶色い本文の変更3成分・111画素を全て保持できた。'
        '同時にhnsdの本文側の追加補修3,781画素（暗→明2,109画素）と虫食いの前回結果を全画素維持できた。'
        '汚れ除去と観測された筆画の保持を両立する手掛かりだが、既知の開発対照であり一般採用はまだしない。', '',
        '最初に見る比較：', '',
        '- [hnsd：成分ごとの見送りで戻った黒い残留を、今回の線保持では維持](old_saved_hnsd_inside_frame_residual.png)',
        '- [37画素の筆画削除を防ぐ](new_new_warm_lower_loss_previous_change_01.png)',
        '- [52画素の筆画端削除を防ぐ](new_new_warm_faint_writing_previous_change_02.png)',
        '- [前回残った22画素の筆画縁変更も0](new_new_warm_faint_writing_previous_change_01.png)',
        '- [線保持マスクが筆画を覆う理由](guard_explains_stroke_preservation.png)',
        '- [汚れ候補と保持の範囲](guard_keeps_dirt_candidate.png)',
        '- [全比較PNG](comparison_png.zip)', '',
        '## 変更した条件', '',
        '従来は、線の形があっても原画像のLab a>6、または紙面からのa差>4だと中性線の保持対象から外れた。'
        '今回は既存の線状応答（sigma1.5/3/6、応答1.5以上、局所L低下6以上、4画素の保持幅）をそのまま使い、'
        'この中性色条件を外した。全ページ原画像で処理し、比較用の座標やケース名を判定へ入れない。', '',
        '比較する2条件は、色を問わず線を保持する条件と、その線の中心の8連結成分が固定v6のprotectに接続する場合だけ保持する条件。'
        '従来の保持も残し、前回の追加候補から新しい保持を除く。紙面参照も新しい保持から外し、同じ明るさ・局所残差・ab距離4の条件と補間を使う。'
        '朱・追加暖色保留・適用外はnpj基準と一致する。前回の全ページ枠外補修は全画素維持するため、この追加保持は枠外補修の正しさを新たに証明しない。', '',
        '17範囲すべてを同じ条件で処理した。新しいDDRM推論・学習・文字生成・筆画補完・色の全体補正はない。'
        '比較のnpj基準は前回の実保存出力または凍結コードの新規crop対照を再利用し、その由来を変更していない。', '',
        '## 汚れ除去と保持の結果', '',
        '| 範囲 | 色を問わず線保持：追加適用 | protect接続線のみ：追加適用 | 色を問わず線保持：暗→明 |',
        '| --- | ---: | ---: | ---: |']
    for c in diagnostics:
        a,b=[c['methods'][m] for m in experiment.MODES]
        lines.append('| [%s](%s_comparison.png) | %d | %d | %d |'%
                     (c['id'],c['id'],a['additional'],b['additional'],a['additional_dark_to_bright']))
    lines.extend(['', '茶色い本文の37＋22＋52画素は全て新しい保持マスクが直接覆っている。'
        '単に参照不足で補間が止まったことを筆画認識の改善と数えた結果ではない。両条件で追加変更0。'
        'hnsdは追加適用3,781・変更3,777・暗→明2,109画素で、前回の出力と全画素一致する。'
        '前回の成分見送りでは追加適用474・暗→明136まで減ったため、今回の形状保持ではその局所除去を失わずに済んだ。'
        '左本文の固定対照は追加変更0。暗→明はLab L<30→L>60であり、汚れの正解数ではない。', '',
        '虫食いの組合せ結果も前回と全画素一致。これは前回の枠外補修を維持した結果で、今回新しく虫食いを解いたという意味ではない。'
        '白黒の4範囲は元の暖色条件で候補0のまま、汚れ除去の成功に数えない。朱印の既知対照も追加変更0。', '',
        '薄い文字の実保存対照old_full_pageでは、4成分・104変更画素が両条件で残る。'
        '20／16／21／47画素を全て拡大して確認した。灰色の別の像・紙面の点・筆画端との区別は未確定で、無誤編集とは呼ばない。'
        '主要な黒い筆画の大きな欠落はこの4拡大では見えないが、それだけで変更を汚れ除去と認定しない。', '',
        '- [残る変更1](old_old_full_page_thin_control_01.png)',
        '- [残る変更2](old_old_full_page_thin_control_02.png)',
        '- [残る変更3](old_old_full_page_thin_control_03.png)',
        '- [残る変更4](old_old_full_page_thin_control_04.png)', '',
        '紙模様・綴じ側の跡にも2条件の差がある。色を問わない保持は追加適用155／0、protect接続線だけの保持は587／1,656画素。'
        '材質の正解がないため、除去量が大きい方を採用する判断はしない。線の形に見える汚れを保持してしまう限界も残る。', '',
        '## 検証と次段階', '',
        '全ページの形状マスクを再計算し、原画像crop・保存基準・候補の包含・必須保持・適用外・参照条件・前回枠外結果・表示パネルを照合した。'
        '補間時は非参照画素をマゼンタへ置換しても適用範囲の結果と参照範囲が変わらないことを検証した。'
        '全111画素の新保持との重なり、およびhnsdと虫食いの前回出力との一致を独立に検査した。', '',
        '今回の17範囲は既に確認済みの開発例で、未使用冊での独立精度や一般化の根拠ではない。'
        '次段階はこの条件を固定し、実保存npjの別ページ／別範囲の汚れと茶色い本文で検証する。'
        '薄い像の4変更も残したまま評価し、本文保持だけでなく追加の汚れ除去があることを確認する。', '',
        '[固定条件・依存関係](manifest.json)／[実行診断](diagnostics.json)／'
        '[事前に固定した表示範囲](witnesses.json)／[範囲別の変化](witness_metrics.json)／'
        '[独立検証](independent_verification.json)'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
