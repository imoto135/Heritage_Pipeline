"""Audit fixed-policy transfer; render every added changed component.

Zero edits and dark-pixel reductions are not restoration accuracy. Originals
and witnesses were selected before policy output, independently of thresholds.
"""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, write, sha, dump, overlay
from finalize_npj_full_page_layout import render, panel_audit
import validate_npj_dirt_new_book as run
import probe_npj_dirt_clean_reference as paper
import probe_npj_contact_dirt_guard_v2 as chroma
import probe_npj_contact_dirt_guard as contact
import probe_npj_full_page_layout as broad
from probe_saved_npj_role_separated_postprocess import frame_guard

OUT = run.OUT
METHODS = ['v6', 'full_page_layout', 'contact_guard']
LABELS = ['原画像', '同じnpj基準', '固定v6紙面補修', '固定全ページ枠外', '枠外維持＋線保持補修']


def verify():
    assert not (OUT/'complete.json').exists(), 'Already frozen'
    selection = json.loads((OUT/'selection.json').read_text())
    for rel, digest in selection['dependencies'].items():
        assert sha(ROOT/rel) == digest, rel
    assert sha(OUT/'source_snapshot.py') == sha(Path(run.__file__))
    for folder, name in [(OUT, 'full_mask_complete.json'), (OUT, 'inference_complete.json'),
                         (OUT/'classification', 'complete.json'), (OUT, 'comparison_complete.json')]:
        for rel, digest in json.loads((folder/name).read_text())['outputs'].items():
            assert sha(folder/rel) == digest, rel
    diagnostics = {c['id']:c for c in json.loads((OUT/'diagnostics.json').read_text())['cases']}
    witnesses = json.loads((OUT/'witnesses_before_comparison.json').read_text())
    audits, regions, components = [], [], []
    for c in selection['cases']:
        n = OUT/'native'/c['id']; x,y,w,h = c['xywh']; sl = np.s_[y:y+h,x:x+w]
        original = read(ROOT/c['original'])[sl]
        base = read(OUT/'npj_output'/c['id']/'result'/(c['id']+'.png'))
        assert np.array_equal(original, read(n/'original.png'))
        assert np.array_equal(base, read(n/'baseline.png'))
        if c.get('saved_full'):
            assert np.array_equal(base, read(ROOT/c['saved_full'])[sl])
        else:
            assert np.array_equal(original, read(OUT/'input'/(c['id']+'.png')))
        roles = {k:read(OUT/'classification/v6_masks'/c['page']/(k+'.png'),True)[sl]
                 for k in ['protect','hold','red','extra']}
        for key, filename in [('npj_black','02_text_black.png'),('npj_red','03_text_red.png')]:
            assert np.array_equal(cv2.imread(str(ROOT/c[key]),0)[sl],
                                  cv2.imread(str(OUT/'masks'/c['id']/filename),0))
        mandatory = roles['red']|roles['extra']|(cv2.imread(str(ROOT/c['npj_red']),0)[sl]<127)
        assert np.array_equal(mandatory, read(n/'mandatory.png',True))
        guard = read(n/'guard.png',True); interior = read(n/'frame_interior.png',True)
        full_roles={k:read(OUT/'classification/v6_masks'/c['page']/(k+'.png'),True)
                    for k in ['edit','protect','hold','red','extra']}
        features,_=contact.features(read(ROOT/c['original']),full_roles)
        f={k:v[sl] for k,v in features.items()}; del features,full_roles
        assert np.array_equal(f['guard'],guard)
        expected_broad,_=broad.select_update(base,f['drop'],f['b'],f['footprint'],mandatory)
        assert np.array_equal(expected_broad,read(n/'broad.png',True))
        chosen = diagnostics[c['id']]['layout']['chosen']
        expected = np.ones((h,w),bool)
        if chosen:
            x0,y0,x1,y1 = chosen['box']; yy,xx = np.ogrid[y:y+h,x:x+w]
            expected = (xx>=max(0,x0-8))&(xx<=x1+8)&(yy>=max(0,y0-8))&(yy<=y1+8)
        assert np.array_equal(interior, expected)
        results, masks, method_checks = {}, {}, {}
        L = cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        for method in METHODS:
            folder = n/method
            m = {k:read(folder/(k+'.png'),True) for k in
                 ['update','domain','reference','applied','used_reference']}
            result = read(folder/'repaired.png'); changed = np.any(result!=base,2)
            assert not (m['applied']&~m['update']).any()
            assert not (m['update']&mandatory).any()
            assert not (m['reference']&(m['update']|mandatory|~m['domain'])).any()
            assert not (m['used_reference']&~m['reference']).any()
            assert np.array_equal(result[~m['applied']|mandatory],base[~m['applied']|mandatory])
            fn = chroma.paper_reference if method=='contact_guard' else paper.paper_reference
            known, threshold = fn(base,m['update'],m['domain'],mandatory)
            assert np.array_equal(known,m['reference'])
            assert threshold == diagnostics[c['id']]['methods'][method]['reference_L_threshold']
            regenerated, applied, used, solves = paper.fill(base,m['update'],known)
            assert np.array_equal(result,regenerated)
            assert np.array_equal(applied,m['applied']) and np.array_equal(used,m['used_reference'])
            poison = base.copy(); poison[~known]=[255,0,255]
            poisoned, poisoned_applied, _, _ = paper.fill(poison,m['update'],known)
            assert np.array_equal(poisoned[applied],result[applied])
            assert np.array_equal(poisoned_applied,applied)
            if method=='full_page_layout':
                assert not (m['update']&interior).any()
                assert np.array_equal(result[interior],base[interior])
            if method=='contact_guard':
                residual=cv2.dilate((L<30).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
                expected_update=expected_broad&residual&~guard&~mandatory&frame_guard((h,w),2)
                assert np.array_equal(expected_update,m['update'])
                assert not (m['update']&guard).any()
                assert np.array_equal(result[guard],base[guard])
            if method=='v6':
                keep=roles['protect']|roles['hold']|mandatory
                assert np.array_equal(result[keep],base[keep])
            after=cv2.cvtColor(result.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            d=diagnostics[c['id']]['methods'][method]
            assert int(changed.sum())==d['changed'] and int(applied.sum())==d['applied']
            assert int((applied&(L<30)&(after[:,:,0]>60)).sum())==d['dark_to_bright']
            assert int((applied&(after[:,:,0]>60)&(after[:,:,1]<-8)).sum())==d['bright_green_in_applied']
            results[method]=result; masks[method]=m
            method_checks[method]=dict(changed=int(changed.sum()),applied=int(applied.sum()),
                reference_and_solver_recomputed=True,poison_invariant=True,
                mandatory_and_outside_applied_exact=True)
        prior=masks['full_page_layout']['applied']
        additional=masks['contact_guard']['applied']&~prior
        combined=results['full_page_layout'].copy();combined[additional]=results['contact_guard'][additional]
        assert np.array_equal(combined,read(n/'combined.png'))
        assert np.array_equal(additional,read(n/'additional_applied.png',True))
        assert np.array_equal(combined[prior],results['full_page_layout'][prior])
        panels=[original,base,results['v6'],results['full_page_layout'],combined]
        panel_audit(OUT/(c['id']+'_comparison.png'),panels,True)
        panel_audit(OUT/(c['id']+'_guard_only.png'),[original,base,results['contact_guard']],True)
        for role, box in witnesses['regions'][c['id']].items():
            wx,wy,ww,wh=box; assert wx>=0 and wy>=0 and wx+ww<=w and wy+wh<=h
            crop=np.s_[wy:wy+wh,wx:wx+ww]
            filename=c['id']+'_'+role+'.png'
            render(OUT/filename,[a[crop] for a in panels],LABELS,c['id']+'：'+role,
                ['原画像だけで事前に選んだ同じ範囲。色補正なし・元画素の整数反復。',
                 '欠損範囲にも文字・罫線が含まれる。画素単位の汚れ正解ではない。'])
            counts={k:int(np.any(v[crop]!=base[crop],2).sum()) for k,v in results.items()}
            counts['combined']=int(np.any(combined[crop]!=base[crop],2).sum())
            regions.append(dict(id=c['id'],role=role,xywh=box,filename=filename,changed=counts))
        # Every changed component of the proposed combination, without cherry picking.
        change=np.any(combined!=base,2)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(change.astype(np.uint8),8)
        for i in range(1,count):
            cx,cy,cw,ch,area=map(int,stats[i]); bx=max(0,cx-35);by=max(0,cy-35)
            ex=min(w,cx+cw+35);ey=min(h,cy+ch+35);cut=np.s_[by:ey,bx:ex]
            filename=c['id']+'_all_change_%02d.png'%i
            render(OUT/filename,[original[cut],base[cut],
                overlay(base[cut],[('blue',change[cut])]),combined[cut]],
                ['原画像','同じnpj基準','青＝全変更','線保持補修後'],
                c['id']+'：変更成分%d／%d'%(i,count-1),
                ['今回の組合せで変わった全成分を掲載。位置は出力確認後の監査。',
                 '青は汚れの正解ではない。筆画端・別の像・汚れの材質は未確定。'])
            components.append(dict(id=c['id'],component=i,bbox=[cx,cy,cw,ch],
                changed_pixels=area,display_xywh=[bx,by,ex-bx,ey-by],filename=filename))
            own=labels==i
            lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            components[-1].update(original_Lab_median=np.median(lab[own],axis=0).tolist(),
                original_ridge_guard_overlap=int((own&f['ridge_guard']).sum()),
                warm_anchored_guard_overlap=int((own&f['warm_guard']).sum()),
                fixed_v6_protect_or_hold_overlap=int((own&(roles['protect']|roles['hold'])).sum()),
                full_csc_black_overlap=int((own&(cv2.imread(str(ROOT/c['npj_black']),0)[sl]<127)).sum()))
            render(OUT/(c['id']+'_guard_diagnostic_%02d.png'%i),
                [original[cut],overlay(original[cut],[('blue',guard[cut])]),
                 overlay(original[cut],[('blue',change[cut])]),combined[cut]],
                ['原画像','青＝固定線保持','青＝実変更','同じ補修結果'],
                c['id']+'：原画像の線保持と誤編集候補',
                ['保持対象の画素は保存されるが、観測される全筆画を保持できる保証はない。',
                 '中性色・線状応答と印影接続の固定条件。閾値調整・追加推論なし。'])
        audits.append(dict(id=c['id'],original_and_baseline_exact=True,
            archived_or_new_mask_crop_exact=True,mandatory_recomputed=True,
            frame_interior_reconstructed=True,methods=method_checks,
            combination_exact=True,panels_verified=True))
    dump(OUT/'independent_verification.json',dict(cases=audits,regions=regions,
        all_combination_changed_components=components,adopted=False,pixel_GT=False,
        independent_model_test=False,no_threshold_changes=True,no_stroke_generation=True))
    shutil.copyfile(__file__,OUT/'finalizer_snapshot.py')
    print('Verified 6 cases, %d preregistered regions, %d total changed components'%
          (len(regions),len(components)),flush=True)


def package():
    selection=json.loads((OUT/'selection.json').read_text())
    verification=json.loads((OUT/'independent_verification.json').read_text())
    diagnostics=json.loads((OUT/'diagnostics.json').read_text())
    judgments={
        'new_warm_upper_loss':'欠損縁はnpj対照で既に目立たなくなる。追加除去なし。罫線は一部残る',
        'new_warm_lower_loss':'事前の欠損範囲は追加変更0。別位置の観測筆画を37画素削る',
        'new_warm_faint_writing':'追加74画素。筆画縁22画素と筆画端52画素を変更。汚れ改善としない',
        'new_gray_binding_mark':'元の跡の材質未確定。候補0のまま未解決',
        'new_gray_writing':'候補0の対照。汚れ除去成功としない',
        'known_saved_red_seal':'既知の実保存朱印。組合せ変更0。新規朱印精度・朱印復元ではない'}
    lines=['# 固定条件を別冊へ適用：追加除去の利点を確認できず、筆画誤編集が見つかった', '',
        '2026-10-04。汚れ除去を主目的に、これまでの実験登録にない暖色・白黒の2冊を原画像から選んだ。'
        '今回の線保持付き補修は一般採用しない。新しい暖色冊ではnpj対照が欠損の縁を既にかなり除去しており、'
        '追加補修は別位置の実際に観測される筆画を削った。文字の生成や欠けた筆画の補完は実施していない。', '',
        'まず見る比較：', '',
        '- [欠損の縁：npj対照で既に除去](new_warm_upper_loss_defect_and_rule.png)',
        '- [筆画の誤編集37画素：原画像と全変更](new_warm_lower_loss_all_change_01.png)',
        '- [筆画端52画素の誤編集](new_warm_faint_writing_all_change_02.png)',
        '- [残る筆画縁の変更22画素](new_warm_faint_writing_all_change_01.png)',
        '- [既知の実保存朱印対照](known_saved_red_seal_actual_seal_and_text_control.png)',
        '- [全比較PNG一括](comparison_png.zip)', '',
        '各PNGは原画像と同一のnpj基準を並べている。主比較の右端は前回の全ページ枠外補修を維持し、'
        'その外に線保持補修を追加した組合せ。今回は全4原ページで支持された枠が見つからず、枠外補修は全て0。'
        'したがって右端と線保持補修単独は全画素同じ。0変更を認識精度や汚れ除去の成功として数えない。', '',
        '| 範囲 | v6適用 | 枠外適用 | 線保持適用 | 線保持の暗→明 | 目視判定 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for d in diagnostics['cases']:
        m=d['methods']; cid=d['id']
        lines.append('| [%s](%s_comparison.png) | %d | %d | %d | %d | %s |'%
            (cid,cid,m['v6']['applied'],m['full_page_layout']['applied'],
             m['contact_guard']['applied'],m['contact_guard']['dark_to_bright'],judgments[cid]))
    lines.extend(['', '暗→明はLab L<30からL>60への変化であり、汚れの正解画素数ではない。'
        '事前に指定した2つの欠損範囲と3つの本文範囲は、今回の組合せで全て変更0。'
        '範囲外の全変更3成分・計111画素を省略せず拡大したため、事前の本文範囲が無変更でも誤編集を見つけられた。'
        '特に37画素と52画素は、原画像に筆画が見える場所でnpj筆画の形が削られている。'
        '22画素も筆画縁に位置し、汚れ除去の肯定例にはできない。画素単位の材質GTはない。', '',
        'なぜ線保持があっても削るのか：保持マスクは中性の線状応答と朱／暖色保留に接続する領域に限る。'
        'マスクに入った画素の保存は検証できるが、茶色い本文の全てがこの条件に入るわけではない。'
        '広い暗部候補が本文まで含み、保持条件から外れた部分を紙色で埋めていた。'
        'これは参照色の緑化やRGB/BGR交換とは別の、汚れと本文の選別の問題。'
        '今回の変更範囲に明るい緑（L>60,a<-8）は0だが、色が正しくても筆画誤編集は解決しない。', '',
        '次は補修量を増やす条件調整を先に行わず、実保存npjに残っている汚れの例へ戻る。'
        '原画像で見える欠損の縁・シミと実際の筆画を固定した小範囲で別々に確認し、'
        '今回の誤編集3成分を回帰対照へ入れる。新しいページでnpjが既に十分除去している例は追加手法の肯定例にしない。'
        '枠の支持不足を安易に緩和して本文を枠外へ出す変更もしない。', '',
        '## 条件と評価の範囲', '',
        '新規2冊は200021925（暖色、2原ページ・3crop）と200004148（白黒、1原ページ・2crop）。'
        'Top-levelのmanifest／selection／baseline_reviewの登録を調べて選定した。'
        '一般の画像一覧・原画像スクリーニングには既出の場合があり、全履歴で未見／モデル学習から独立とは主張しない。'
        '暖色冊1冊だけのため冊間成功率を出さない。白黒の0編集も肯定例に加えない。', '',
        '200021925は挿絵を含む冊だが、本文のみの00007_1と00004_1を原画像で確認して使った。'
        '表紙・挿絵・装飾された扉は除外。新しい冊で朱印が見つからなかったので、'
        '200010454の既知の実保存npjを別の保持対照にした。独立した新規冊の朱印評価ではない。', '',
        '新規5cropは原画像の画素をそのまま入力し、凍結Oct2コード・同じモデル・背景・seed999・20stepsを用いた新規対照。'
        '過去の全文書推論結果や論文手法の厳密な再現と同一視しない。'
        '暖色冊は保存CSCがなかったため、凍結したローカルCSCで2全ページのマスクを新規生成した。'
        '白黒冊と既知朱印は保存corpus-v1マスクを変更せず使った。マスクの由来は混在している。'
        '既知朱印は実保存の全ページ出力を切り出し、当時の実入力と原画像の画素一致も確認した。', '',
        '固定v6は全原ページで算出。候補・線保持・参照色距離4・補間条件は前回から変更していない。'
        'この比較は保存／新規npjの後に局所的な紙面補修を行う試作であり、DDRMの新設計を適用した比較ではない。', '',
        '## 検証と実行記録', '',
        '新規推論入力・実保存出力・原画像crop・マスクcrop・必須保持・適用外を画素で照合した。'
        '全原ページ由来の線保持と暗部候補、参照条件、補間結果、非参照画素をマゼンタにする反証試験、'
        '組合せ、PNGパネル、ZIP内の全画素とハッシュを検証。内部の画素一致は汚れの認識精度を保証しない。', '',
        '最初のマスク監査は保存分類00_inputが原画像と同じと仮定して停止した。'
        '保存分類用画像にはACE等の処理があり、この仮定は誤りだった。新規モデル入力は原画像のままで、'
        '既知朱印の当時の実入力も原画像と一致する。推論後の監査も、既知朱印には新規inputがないことを見落として停止した。'
        'いずれも実生成／推論は終了済みで、監査用アダプタで由来を分けて検証し直した。'
        'マスク生成やモデル推論を再実行して良い結果を選んでいない。元の停止ログも保存する。', '',
        '[選定と依存関係](selection.json)／[実行診断](diagnostics.json)／'
        '[独立再計算と全変更記録](independent_verification.json)／'
        '[分類画像と実入力の監査](mask_input_audit.json)／[推論の監査](inference_complete.json)', '',
        '## 事前指定範囲の比較', ''])
    for r in verification['regions']:
        lines.append('- [%s：%s](%s)（組合せ変更%d画素）'%
                     (r['id'],r['role'],r['filename'],r['changed']['combined']))
    lines.extend(['', '## 全変更成分の監査', ''])
    for c in verification['all_combination_changed_components']:
        diag=c['id']+'_guard_diagnostic_%02d.png'%c['component']
        lines.append('- [%s成分%d：%d画素](%s)／[保持マスクとの関係](%s)'%
                     (c['id'],c['component'],c['changed_pixels'],c['filename'],diag))
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    figures=sorted(OUT.glob('*.png'))
    with zipfile.ZipFile(OUT/'comparison_png.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in figures:z.write(p,p.name)
    with zipfile.ZipFile(OUT/'comparison_png.zip') as z:
        assert sorted(z.namelist())==sorted(p.name for p in figures)
        for p in figures:
            decoded=cv2.imdecode(np.frombuffer(z.read(p.name),np.uint8),cv2.IMREAD_COLOR)
            assert np.array_equal(decoded,read(p))
    shutil.copyfile(__file__,OUT/'finalizer_snapshot.py')
    outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
             if p.is_file() and not p.is_symlink() and p.name!='complete.json'}
    dump(OUT/'complete.json',dict(outputs=outputs,comparison_png_count=len(figures),
        adopted=False,method_thresholds_unchanged=True,new_books=2,new_full_pages=3,
        fresh_crop_inferences=5,known_actual_saved_crop=1,all_combined_changes_inspected=True,
        added_dirt_benefit_confirmed=False,observed_stroke_overediting=True,
        source_dependencies_unchanged=True,archive_pixels_verified=True))
    print('Frozen %d comparison PNG; fixed transfer rejected for observed stroke edits'%len(figures),flush=True)


if __name__=='__main__':
    verify()
    package()
