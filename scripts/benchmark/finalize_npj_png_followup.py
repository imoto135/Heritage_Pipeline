"""Record provisional visual findings and package PNG-only follow-up reviews."""
import json
from pathlib import Path
import zipfile

from build_npj_v6_comparison import dump, sha
from package_npj_comparison_images import board

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / 'tmp_work/npj_failure_followup_20261001'
PROBE = ROOT / 'tmp_work/npj_v6_edit_effect_probe_20261001'

MASK_FINDINGS = {
    'brown_near_text': ('unresolved', '未解決',
        '本文に接する茶色い部分の中心は保護・保留側に残る。汚れと損傷のどちらかは画像だけでは確定できない。'),
    'dark_text_control': ('control_preserved', '対照：主要筆画を維持',
        '主要な黒筆画を維持。紙面の影・ムラも保留されるため、編集面積の減少を除去品質の改善とは数えない。'),
    'faint_left_text': ('partial', '部分改善',
        '細い端部・払いの編集候補が減る。極薄い線の分類には欠落が残り、全筆画の正解判定とは言えない。'),
    'damage_control': ('unresolved', '未解決',
        '穴・裂けの濃い縁が保護・保留側に残る。このIDは対照ではなく失敗例として登録済み。'),
}

EDIT_FINDINGS = {
    'dense_stain': ('fill_artifact', '濃い汚れは編集されるが、平坦な茶色の塗り跡と灰色の円状部分が生じる。マスク改善を復元成功と数えない。'),
    'thin_black_water': ('stroke_preserved_stain_retained', '旧npjマスクの補間では薄い筆画の一部と紙面の質感が変わる。v6はほぼ原画像を維持するが、水染み・ムラも残る。'),
    'brown_as_red': ('fill_artifact', '旧npjで朱候補だった茶色い中心をv6で編集できるが、平坦な塗り跡が残る。'),
    'thin_black_loop': ('stroke_preservation', '旧npjマスクの補間で細い輪状筆画などが途切れる。v6はこの矩形の編集画素が0で、原画像と完全一致して筆画を維持する。'),
    'thin_black_text': ('control_preserved', '主要本文は両方で維持。旧npjマスクの補間は紙面を平滑化し、一部に平坦な点を作る。v6は影・ムラも含めほぼ原画像を維持。'),
    'warm_stroke': ('stroke_preservation', 'v6では赤褐色の長い筆画の端部・周縁が残る。確定朱への分類成功ではなく、保留による編集回避。'),
    'control_black': ('control_preserved', '主要な黒筆画は両方で維持。v6は紙面の灰色成分もほぼ維持する。'),
    'dense_stain_new': ('stain_retained', '本文・枠とつながる濃い茶色の塊は両方で残る。v6の変更は39画素で、中心の除去には至らない。'),
    'brown_false_red_new': ('fill_artifact_and_stain_retained', 'v6は茶色い領域の一部を編集するが、塗り跡が出て広い薄茶色の染みも残る。'),
    'hole_edge_new': ('damage_retained', '穴の濃い縁は両方で残る。v6で周縁を編集しても損傷の修復にはならない。'),
    'control_text_new': ('control_preserved', 'v6はこの矩形を変更せず本文を維持する。同時に紙面の影も維持する。'),
    'brown_near_text': ('stain_or_damage_retained', '本文に接する茶色い中心は両方で残る。v6は主に周縁を変更する。原因が汚れか損傷かは未確定。'),
    'dark_text_control': ('control_preserved', '主要な濃い本文は両方で維持。v6は43画素だけ変更し、紙面のムラも維持する。'),
    'faint_left_text': ('stroke_preservation', '旧npjマスクの補間では細い輪・払いの一部が消える。v6は薄い筆画をほぼ原状で残す。'),
    'damage_control': ('damage_retained', '別冊の穴・裂けの縁は両方で残る。v6は112画素だけ変更し、濃い縁の修復には至らない。'),
}


def main():
    baseline = json.loads((FOLLOWUP / 'baseline_review.json').read_text())
    diagnostics = {r['id']: r for r in json.loads((FOLLOWUP / 'comparison_diagnostics.json').read_text())}
    findings = []
    for page in baseline['cases']:
        for roi in page['rois']:
            verdict, label, observation = MASK_FINDINGS[roi['id']]
            findings.append(dict(id=roi['id'], page=page['page'], title=roi['title'],
                baseline_failure=roi['baseline_failure'], verdict=verdict,
                verdict_label=label, observation=observation,
                v6_extra_pixels=diagnostics[roi['id']]['v6_extra_pixels']))
    assert len(findings) == 4 and all(r['v6_extra_pixels'] == 0 for r in findings)
    review = dict(pages=3, books=3, failure_rois=3, control_rois=1,
        failure_verdicts=dict(partial=1, unresolved=2), findings=findings,
        reviewer='assistant_visual_review', pixel_GT=False,
        human_confirmation_pending=True, representative_accuracy=False,
        v6_specific_increment_in_selected_rois=False)
    dump(FOLLOWUP / 'v6_visual_review.json', review)
    board([FOLLOWUP / 'comparisons' / (r['id'] + '.png') for r in findings],
          [r['verdict_label'] + '：' + r['title'] for r in findings],
          FOLLOWUP / 'comparison_summary.png', '追加3ページ：固定v6と保存済みnpjの比較')
    archive = FOLLOWUP / 'npj_v6_followup_comparison_images.zip'
    pngs = sorted((FOLLOWUP / 'comparisons').glob('*.png'))
    assert len(pngs) == 7
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        z.write(FOLLOWUP / 'comparison_summary.png', 'overview/comparison_summary.png')
        for p in pngs:
            z.write(p, 'individual/' + p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == 8
    (FOLLOWUP / 'report.md').write_text('''# 追加3ページの固定比較

[比較PNG](comparison_summary.png) / [全8枚のZIP](npj_v6_followup_comparison_images.zip)

原画像と保存済みnpjだけで追加候補13ページを目視し、イラスト3ページを除外。v6を見る前に3ページ・3冊の問題3箇所と本文対照1箇所を固定した。開発冊以外は200014740の1ページ、残り2ページは既知冊の新しいページ。代表抽出ではない。

薄い筆画は部分改善1、本文に接する茶色い部分と別冊の穴・裂けは未解決2。対照の主要筆画は維持された。すべて補助者の暫定目視で、画素GT・全資料の精度ではない。茶色い部分の原因が汚れか損傷かは未確定。

4箇所ともv6固有の追加暖色保留は0画素。今回の差をv6固有の改良効果とは数えず、固定v6一式と旧npjの比較として扱う。暖色保留・基礎処理の閾値は変更していない。

マスクの色だけでは除去品質を判断できないため、前回と合わせた11ページ・15箇所に同一の局所補間を適用した。[筆画の実際の変化](../npj_v6_edit_effect_probe_20261001/02_stroke_edit_effect.png)、[残る汚れ・損傷](../npj_v6_edit_effect_probe_20261001/03_remaining_edit_effect.png)、[補間の所見と次の検証](../npj_v6_edit_effect_probe_20261001/report.md)。

再実行：`python -B scripts/benchmark/run_npj_failure_followup.py`。画像のまとめと所見：`python -B scripts/benchmark/finalize_npj_png_followup.py`。入力・固定コードのハッシュ、分類の排他性、原寸マスクと比較画像の一致を確認済み。原画像・保存済みnpjを変更していない。
''')
    rows = json.loads((PROBE / 'diagnostics.json').read_text())
    assert {r['id'] for r in rows} == set(EDIT_FINDINGS) and len(rows) == 15
    edit_findings = []
    for r in rows:
        verdict, observation = EDIT_FINDINGS[r['id']]
        edit_findings.append(dict(id=r['id'], page=r['page'], verdict=verdict,
            observation=observation, baseline_failure=r['baseline_failure'],
            npj_changed_pixels=r['npj_changed_pixels'], v6_changed_pixels=r['v6_changed_pixels']))
    dump(PROBE / 'visual_findings.json', dict(findings=edit_findings,
        reviewer='assistant_visual_review', human_confirmation_pending=True,
        pixel_GT=False, restoration_accuracy=False, full_npj_or_DDRM_reproduction=False,
        interpretation='Same local operator isolates edit-mask consequences. It does not establish full restoration quality.'))
    protocol = dict(status='planned_after_completed_fixed_probe',
        completed=dict(additional_pages=3, paired_probe_pages=11, paired_probe_rois=15,
                       primary_artifact='PNG'),
        frozen_cases=[r['id'] for r in rows],
        next_questions=[
            '本文に接する汚れ・穴の濃い縁を編集対象に戻せるか。周囲の実筆画を対照にする。',
            '編集対象に戻した汚れに、平坦な塗り跡・灰色点・筆画の消失を作らない復元ができるか。'],
        restoration_integration=[
            'DDRMの既知・欠損／本文マスクとv6のprotect・hold・editの意味を明記した接続仕様を作る。v6 editを本文マスクとして直接渡さない。',
            '旧npjの全工程再現と、同じ復元器でのマスク差比較を別の実験として記録する。',
            '復元器・重み・乱数種・解像度・前後処理を対で固定し、最終画像の編集範囲外は原画素を維持する。'],
        acceptance=[
            'thin_black_loop・faint_left_text・warm_strokeと本文対照で実筆画を消さない。',
            'dense_stain_new・brown_near_text・hole_edge_new・damage_controlの誤保護が残るかを確認する。',
            'dense_stain・brown_as_red・brown_false_red_newの塗り跡と紙面の質感を確認する。',
            '固定15箇所を省略せず比較PNGで提示する。編集画素数を精度・除去率と呼ばない。',
            'これらを調整に使ったら開発例と明記し、別冊の未使用ページで確認する。イラストページは主評価から除外する。'],
        method_thresholds_changed=False, new_method_implemented=False)
    dump(PROBE / 'next_step_protocol.json', protocol)
    (PROBE / 'report.md').write_text('''# 編集マスクが見た目に与える影響

[筆画の比較PNG](02_stroke_edit_effect.png) / [汚れを編集した比較PNG](01_stain_edit_effect.png) / [未解決の比較PNG](03_remaining_edit_effect.png) / [本文対照PNG](04_control_edit_effect.png) / [全19枚のZIP](npj_v6_edit_effect_images.zip)

前回8ページと追加3ページの固定矩形15箇所すべてで、同じ原画像に同じ局所補間を適用した。原寸、周辺64px、OpenCV Telea半径3px、マスク膨張なし。旧npjの保存済み04_degradationと固定v6のeditだけを切り替え、編集範囲外は原画素を維持した。分類マスク比較から一段進め、実際に筆画や紙面がどう変化するかを確認した。DDRMや旧npj全工程の再現ではない。

## 確認できたこと

細い輪状筆画と別ページの薄い払いは、旧npjマスクの補間で消える部分があり、v6では残る。輪状筆画の矩形ではv6の編集画素が0で、結果は原画像と完全一致する。赤褐色の長い筆画も保留で端部を維持する。一方、水染み・影・紙面のムラも維持されるため、変更が少ないだけで全面的な品質改善とは言えない。

孤立した濃い汚れや誤って朱候補に入った茶色い領域は、v6で編集できても平坦な塗り跡が出る。孤立汚れには灰色の円状部分も生じた。マスクの誤判定を直すことと、自然な紙面を復元することは別々に検証する必要がある。

本文・枠につながる汚れ、本文に接する茶色い部分、穴・裂けの濃い縁は両方で残る。追加ページの茶色い部分が汚れか損傷かは画像だけでは未確定。本文対照の主要筆画は維持されたが、灰色成分が裏写りか影かの正解ラベルはない。

所見は補助者の暫定目視。画素GT・正解復元画像はなく、変更画素数は診断値であり精度・除去率ではない。固定v6一式の差には土台の処理の変更も含まれ、すべてをv6固有の暖色保留の寄与とは数えない。

## 次の検証条件

次は固定例を保ったまま、(1)汚れ・損傷と実筆画の分離、(2)編集した場所に塗り跡を作らない復元、の二つを確認する。薄い輪・払い・赤褐色筆画と本文対照を、汚れ除去に伴う筆画消失の対照として使う。現段階で閾値変更・新手法作成・DDRMへの接続はしていない。

DDRM接続にはマスクの意味の整理が必要。v6のprotect・hold・editと、既存復元器の既知・欠損／本文マスクの対応を決めてから実装する。旧npj全工程の再現と、同一復元器でマスク差だけを見る実験は区別して記録し、対ごとに重み・乱数種・解像度・前後処理を固定する。受け入れ条件と15固定例は[next_step_protocol.json](next_step_protocol.json)に保存した。

## 保存と確認

各矩形の原寸画像・出力・編集マスク・変更マスクをnativeに保存。入力とマスクのハッシュ、出力の原寸一致、編集範囲外の不変、ZIP整合性は11ページ・15箇所すべてで確認済み。[個別所見](visual_findings.json)、[診断値](diagnostics.csv)、[検証記録](verification.json)。

再実行：`python -B scripts/benchmark/probe_npj_v6_edit_effect.py`。実行環境：`/home/imoto/miniconda3/envs/diffusion/bin/python`。
''')
    dump(FOLLOWUP / 'png_review_complete.json', dict(source_sha256=sha(Path(__file__)),
        outputs={p.name: sha(p) for p in [FOLLOWUP / 'comparison_summary.png',
                 FOLLOWUP / 'v6_visual_review.json', FOLLOWUP / 'report.md', archive]},
        probe_review_outputs={p.name: sha(p) for p in [PROBE / 'visual_findings.json',
                 PROBE / 'next_step_protocol.json', PROBE / 'report.md']},
        archive_files=8, pixel_GT=False))
    print('Recorded provisional findings; follow-up ZIP verified: 8 PNGs.')


if __name__ == '__main__':
    main()
