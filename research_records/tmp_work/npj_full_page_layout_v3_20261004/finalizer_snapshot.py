"""Freeze development layout branches with provenance and semantic controls.

This audits saved pixels independently of the processing entrypoints. It does
not equate internal mask consistency with recognition accuracy or adopt a model.
"""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, dump, overlay, read, sha
from package_saved_npj_role_separated_review import figure

NAMES = ['npj_full_page_layout_20261004'] + [
    f'npj_full_page_layout_v{i}_20261004' for i in range(2, 5)]
LABELS = ['原画像', '固定npj対照・新規crop', '前回の切出し内判定', '全原ページの枠文脈']
NOTES = [
    '追加DDRM・文字生成なし。表示の範囲外は評価していない。',
    '本文外包・朱／追加暖色保留は維持。従来の黒保護／holdは汚れ候補と重なり得る。',
    '既知8ページの開発比較。未編集は汚れ除去成功と数えない。']


def panel_audit(path, arrays, overview):
    h, w = arrays[0].shape[:2]
    if overview:
        ratio = min(540/w, 900/h, 1)
        dw, dh = round(w*ratio), round(h*ratio)
        panels = [cv2.resize(a, (dw, dh), interpolation=cv2.INTER_AREA) for a in arrays]
    else:
        scale = min(12, max(1, min(680//w, 1000//h)))
        dw, dh = w*scale, h*scale
        panels = [np.repeat(np.repeat(a, scale, 0), scale, 1) for a in arrays]
    canvas = read(path)
    for i, a in enumerate(panels):
        x = 24+i*(dw+24)
        assert np.array_equal(canvas[120:120+dh, x:x+dw], a), path


def render(path, arrays, labels, title, notes, overview=False):
    figure(arrays, labels, path, title, notes, overview=overview)
    panel_audit(path, arrays, overview)


def writing_witness(out, manifests, diagnostics):
    """Layout-only control in observed left text; no full-page output exists."""
    case = next(c for c in manifests['frames'] if c['id'] == 'wormhole')
    original = read(ROOT/case['original'])
    x, y, w, h = 130, 1250, 240, 1100
    cut = np.s_[y:y+h, x:x+w]
    panels = [original[cut]]
    counts = []
    for name in [NAMES[0], NAMES[-1]]:
        diag = json.loads((ROOT/'tmp_work'/name/'diagnostics.json').read_text())
        c = next(c for c in diag['cases'] if c['id'] == 'wormhole')
        x0, y0, x1, y1 = c['layout']['chosen']['box']
        yy, xx = np.ogrid[y:y+h, x:x+w]
        outside = ~((xx >= x0-8) & (xx <= x1+8) & (yy >= y0-8) & (yy <= y1+8))
        panels.append(overlay(original[cut], [('blue', outside)]))
        counts.append(int(outside.sum()))
    assert counts == [w*h, 0], counts
    render(out/'failure_inner_division_layout_only.png', panels,
           ['原ページ・実際の左列', '初版：青＝外包から外れる', '観測幅版：この列を保持'],
           '本文を含む左列：初版の内側罫線誤認を確認',
           ['配置判定だけの比較。ここに対応する全ページ復元は実行していない。',
            '初版はこの列全体を外と判定。観測幅版は保持するが全本文のGTではない。'], overview=True)
    return dict(page=case['page'], original_xywh=[x,y,w,h],
                old_exterior_pixels=counts[0], current_exterior_pixels=counts[1],
                scope='layout-only witness; not restoration or full-page safety proof')


def finalize(name):
    out = ROOT/'tmp_work'/name
    assert not (out/'complete.json').exists(), 'Already frozen'
    manifest = json.loads((out/'manifest.json').read_text())
    for rel, digest in manifest['dependencies'].items():
        assert sha(ROOT/rel) == digest, rel
    assert sha(out/'source_snapshot.py') == manifest['source_sha256']
    if 'entrypoint' in manifest:
        assert sha(out/'entrypoint_snapshot.py') == manifest['entrypoint_sha256_recorded_before_run']
    diag = json.loads((out/'diagnostics.json').read_text())
    diagnostics = {c['id']:c for c in diag['cases']}
    records = []
    for c in manifest['frames']:
        native = out/'native'/c['id']
        a = {k:read(native/(k+'.png')) for k in ['original','baseline','legacy','repaired']}
        m = {k:read(native/(k+'.png'),True) for k in
             ['interior','exterior','mandatory','update','applied','reference','used_reference']}
        x,y,w,h = c['xywh']; sl = np.s_[y:y+h, x:x+w]
        assert np.array_equal(a['original'], read(ROOT/c['original'])[sl])
        assert np.array_equal(a['baseline'], read(ROOT/c['saved_npj']))
        assert np.array_equal(a['legacy'], read(ROOT/c['legacy_output']))
        roles = {k:read(ROOT/c['v6_roles']/(k+'.png'),True)[sl]
                 for k in ['red','extra','protect','hold']}
        red = cv2.imread(str(ROOT/c['npj_red']),0)[sl] < 127
        assert np.array_equal(m['mandatory'], roles['red']|roles['extra']|red)
        chosen = diagnostics[c['id']]['layout']['chosen']
        expected = np.ones((h,w),bool)
        if chosen:
            x0,y0,x1,y1 = chosen['box']
            yy,xx = np.ogrid[y:y+h,x:x+w]
            expected = (xx>=max(0,x0-8)) & (xx<=x1+8) & (yy>=max(0,y0-8)) & (yy<=y1+8)
        assert np.array_equal(expected,m['interior'])
        assert not (m['exterior'] & m['interior']).any()
        assert not (m['update'] & (~m['exterior'] | m['mandatory'])).any()
        assert not (m['applied'] & ~m['update']).any()
        assert not (m['reference'] & (m['update'] | m['mandatory'] | ~m['exterior'])).any()
        assert not (m['used_reference'] & ~m['reference']).any()
        assert np.array_equal(a['repaired'][~m['applied']],a['baseline'][~m['applied']])
        assert np.array_equal(a['repaired'][m['interior']|m['mandatory']],
                              a['baseline'][m['interior']|m['mandatory']])
        changed = np.any(a['repaired']!=a['baseline'],2)
        L = cv2.cvtColor(a['baseline'].astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        afterL = cv2.cvtColor(a['repaired'].astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        d = diagnostics[c['id']]
        assert int(m['applied'].sum())==d['applied_pixels']
        assert int(changed.sum())==d['changed_pixels']
        assert int((m['applied']&(L<30)&(afterL>60)).sum())==d['removed_dark_pixels']
        labels = LABELS.copy()
        if c['id']=='saved_hnsd':
            labels[1]='歴史的保存npj・一致対照';labels[2]='前回の紙面参照補修'
        render(out/(c['id']+'_comparison.png'),[a[k] for k in a],labels,
               c['page']+'：残留汚れと本文保持の開発比較',NOTES,overview=True)
        records.append(dict(id=c['id'],source_crop_and_baseline_exact=True,
             envelope_independently_reconstructed=True,mandatory_exact=True,
             reference_roles_disjoint=True,outside_applied_and_kept_exact=True,
             panels_independently_verified=True,
             changes_on_black_protect_or_hold=int((changed&(roles['protect']|roles['hold'])).sum()),
             whole_frame_changes=int(changed.sum()),applied_pixels=int(m['applied'].sum())))
    labels = ['原画像','固定npj対照・新規crop','固定v6の紙面補修','旧：切出し内判定','今回：全ページ文脈']
    focus = {}
    if name==NAMES[-1]:
        for fid,box,filename in [
            ('wormhole',[230,150,400,420],'focus_wormhole_rim.png'),
            ('seal_control',[20,450,550,280],'focus_body_preservation.png')]:
            n=out/'native'/fid;x,y,w,h=box;sl=np.s_[y:y+h,x:x+w]
            arrays=[read(n/(k+'.png'))[sl] for k in ['original','baseline']]
            arrays.append(read(ROOT/'tmp_work/npj_dirt_transfer_20261004/native'/fid/'v6_clean_reference.png')[sl])
            arrays.extend(read(n/(k+'.png'))[sl] for k in ['legacy','repaired'])
            render(out/filename,arrays,labels,fid+'：同じ範囲・色補正なしの比較',
                   ['今回の原ページ判定で、虫食いの黒い縁を選ぶ／本文を消す旧版を見送る。',
                    '印影は基準で既に薄く／欠落している。今回の保持を朱印復元と呼ばない。'])
            focus[fid]=dict(crop_xywh=box,filename=filename,
                previous_changes=int(np.any(arrays[3]!=arrays[1],2).sum()),
                current_changes=int(np.any(arrays[4]!=arrays[1],2).sum()))
        n=out/'native/saved_hnsd';sl=np.s_[370:590,250:600]
        arrays=[read(n/(k+'.png'))[sl] for k in ['original','baseline','legacy','repaired']]
        render(out/'focus_saved_hnsd_residual.png',arrays,
               ['原画像','歴史的保存npj','前回の紙面参照補修','今回：全ページ文脈'],
               'hnsd：枠下の黒い塊、残存点、紙面の平坦さ',
               ['本文側の塊は未解決。黒画素減少は汚れのGT精度ではない。'])
        for fid in ['wormhole','saved_hnsd']:
            n=out/'native'/fid
            arrays=[read(n/(k+'.png')) for k in ['original','baseline']]
            arrays.append(overlay(arrays[1],[('blue',read(n/'update.png',True))]))
            arrays.append(read(n/'repaired.png'))
            render(out/(fid+'_edit_scope.png'),arrays,
                   ['原画像','同一npj基準','青＝更新候補','今回の補修結果'],
                   fid+'：編集範囲を表示',
                   ['朱／追加暖色保留・本文外包・適用外は基準と全画素一致。',
                    '更新候補は黒保護／holdと重なる。文字か汚れかは別途目視が必要。'],overview=True)
        witness=writing_witness(out,manifest,diagnostics)
        dump(out/'semantic_witnesses.json',dict(focus=focus,left_text_column=witness,
             full_page_material_GT=False,seal_preserved_by_abstention=True,
             binding_shadow_or_bleedthrough_on_right_unconfirmed=True))
        assert focus['seal_control']['previous_changes']>0 and focus['seal_control']['current_changes']==0
    dump(out/'independent_verification.json',dict(cases=records,
         focus=focus,no_new_diffusion_inference=True,no_stroke_generation=True,
         previous_processing_checks=json.loads((out/'verification.json').read_text()),
         source_dependencies_unchanged=True,independent_test=False,pixel_GT=False))
    conclusions = {
        NAMES[0]:'初版は虫食いを局所的に除去するが、表の内側罫線を左外枠と誤認し実際の本文列を外へ出す。一般採用しない。',
        NAMES[1]:'線分群の全端点を外包する版は遠い文字列・線分まで混ぜ、虫食いとhnsdの条件が成立しなくなる。編集0は解決ではなく退行。採用しない。',
        NAMES[2]:'方向別の実線支持と交点で局所除去が戻るが、紙模様対照48,018画素まで変更した。観測幅を含まないため一般採用しない。',
        NAMES[3]:'観測した上下横線の全幅を本文保持外包へ加える。虫食いの黒い縁が局所的に除去され、既知の本文消失を見送る。ただし配置判定・汚れの材質・一般化は未確定なので自動採用しない。'}
    table=['| 範囲 | 適用画素 | 変更画素 | 暗→明画素 | 判定 |','| --- | ---: | ---: | ---: | --- |']
    judgments={
        'wormhole':'見える欠損縁の局所例。全ページ修復ではない',
        'saved_hnsd':'枠下のみ。本文側の塊・点・平坦面が残る',
        'fresh_lower_paper':'紙模様／本文対照。変更増を改善と呼ばない',
        'fresh_binding_mark':'枠が成立せず、綴じ側の跡は未解決',
        'seal_control':'保持の対照。今回v4は判定不能で見送り',
        'fresh_printed_text':'本文対照。除去成功には数えない',
        'unseen_lower_frame_mark':'白黒断片は未解決',
        'unseen_lower_binding_mark':'白黒断片は未解決'}
    for c in manifest['frames']:
        d=diagnostics[c['id']]
        table.append(f"| [{c['id']}]({c['id']}_comparison.png) | {d['applied_pixels']:,} | {d['changed_pixels']:,} | {d['removed_dark_pixels']:,} | {judgments[c['id']]} |")
    current = name==NAMES[-1]
    extra=''
    if current:
        extra='''
最初に見る画像：

- [虫食いの縁：原画像／npj／v6／旧試作／今回](focus_wormhole_rim.png)
- [本文消失の反例と今回の保持](focus_body_preservation.png)
- [枠下の黒い塊と残存点](focus_saved_hnsd_residual.png)
- [虫食いの編集範囲](wormhole_edit_scope.png)
- [内側罫線を外枠にする反例・配置だけの比較](failure_inner_division_layout_only.png)

虫食いの対応範囲では15,825画素を適用し、15,085画素が変化した。Lab L<30→L>60は4,212画素。固定v6が保持する黒い縁も候補へ含められた。ただしこれは1既知ページの局所結果で、精度や成功率ではない。補修面の平坦さが残る。初版／v3／v4のこの局所出力は全画素一致するため3成功例には数えない。

印影ページは枠判定が成立せず更新0になり、本文を消す旧試作を見送れた。正しく外枠を検出できた証明ではない。薄い朱印はnpj対照で既に失われており、今回の後処理では取り戻していない。

全ページの左本文列は初版で外包外へ出ていたが、横線観測幅を含めた版では保持される。右側の密集した像は綴じ・裏写り等との区別が未確定で、先に「右端にも誤認」とした解釈は確定していない。枠外に本物の文字がないことを全面的に検証したわけではない。
'''
    report=f'''# 原ページの枠文脈で残留汚れを選ぶ開発比較

2026-10-04。{conclusions[name]}
{extra}
## 同一条件の8ページ

{chr(10).join(table)}

暗→明は適用域でLab L<30がL>60へ変わった画素数。汚れのGT・本文のGT精度ではない。既知の開発8ページで、真正の未使用2冊も前回に観察済みなので独立テストではない。イラストは対象外。編集0は保持または見送りで、汚れの除去成功ではない。

## npjからの変更

保存したnpj出力を土台に、原ページ全体から本文外包と紙面内の外側を求め、観測された広い暗部かつnpjにも残る成分だけを紙面参照補修する。原画像の全体へ合成し戻さない。追加DDRM・再学習・筆画補完はなし。

7範囲の基準は10月2日の固定ローカルコード・既存CSC v1・同モデル／背景で新規推論したcrop対照で、歴史的保存最終画像ではない。hnsdのみ歴史的保存npjと新規対照が全画素一致したもの。全ページ復元や論文再現と呼ばない。

全文脈版では本文外包、朱候補、v6 extra、CSC朱を必須保持とする。v6の黒protect／holdは一律保持せず、枠外の汚れ候補と重なり得る。暖色holdすべてを保持するという初期PNGの説明を訂正した。編集対象を増やすほど文字損失リスクがあるため、原画像・範囲・出力を同時に確認する。

汚れ条件はsigma8輝度差soft≥5／seed≥8、元画像b≥5、面積128・seed32・半径4以上、npj残差32画素かつ3%以上。紙面参照条件・sigma2・四近傍補間は10月2日版から不変。数値条件は各版を実行する前に記録したが、配置設計は同じ8例を見て変更した開発探索である。

## 採否と次の評価

一般採用は保留。固定npj／v6を継続基準とする。枠内の接触汚れ、枠不成立の綴じ側、白黒の欠損断片、紙面補修の平坦さが残る。本文や朱印の既存欠損を直すことを研究の前提にしない。

次はこの版を凍結し、別の損傷ページと、枠外に実際の見出し・注記がある対照を設定変更なしで評価する。先に原ページで配置と材質の候補を確認し、実保存npjに残るかを確認してから局所効果を評価する。失敗を見て新しい設定に変えた場合は次版の開発例として分ける。全文書の失敗率や新手法成立は未確定。

[全比較PNG ZIP](comparison_pngs.zip)・[独立画素監査](independent_verification.json)・[条件と入力](manifest.json)・[数値記録](diagnostics.json)。8組の原ページ配置画像は入力側の診断であり、全ページ復元結果ではない。
'''
    (out/'report.md').write_text(report)
    dump(out/'adoption_gate.json',dict(adopted=False,decision=conclusions[name],
         no_stroke_generation=True,independent_test=False,retain_npj_v6_baselines=True))
    shutil.copyfile(__file__,out/'finalizer_snapshot.py')
    pngs=sorted(p for p in out.glob('*.png') if not p.name.endswith('_original.png'))
    with zipfile.ZipFile(out/'comparison_pngs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,p.name)
    with zipfile.ZipFile(out/'comparison_pngs.zip') as z:
        assert z.testzip() is None
        for p in pngs:assert z.read(p.name)==p.read_bytes()
    dump(out/'review_manifest.json',dict(finalizer_sha256=sha(Path(__file__)),
         comparison_count=len(pngs),report_sha256=sha(out/'report.md'),
         supplementary_original_detail_excluded_from_comparison_zip=True))
    dump(out/'complete.json',dict(outputs={str(p.relative_to(out)):sha(p)
         for p in sorted(out.rglob('*')) if p.is_file()}))
    for rel,digest in json.loads((out/'complete.json').read_text())['outputs'].items():
        assert sha(out/rel)==digest,rel
    print('Frozen and verified',name,len(pngs),'comparison PNG',flush=True)


if __name__=='__main__':
    for name in NAMES:finalize(name)
