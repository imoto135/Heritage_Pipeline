"""Record visual findings after fixed saved-output transfer; no policy tuning."""
import json
from pathlib import Path
import shutil
import numpy as np
from build_npj_v6_comparison import ROOT,dump,read
import validate_npj_ridge_saved_ranges as run

OUT=run.OUT


def main():
    assert not (OUT/'complete.json').exists()
    selection=json.loads((OUT/'selection.json').read_text())
    diagnostics=json.loads((OUT/'diagnostics.json').read_text())['cases']
    audit=json.loads((OUT/'verification.json').read_text())
    previous=json.loads((run.ridge.OUT/'manifest.json').read_text())['cases']
    overlaps=[]
    for c in selection['cases']:
        n=OUT/'native'/c['id']
        assert np.array_equal(read(n/'all_colour_ridge/combined.png'),read(n/'protect_anchored_ridge/combined.png'))
        x,y,w,h=c['xywh']
        for old in previous:
            if c['page']!=old['page']:continue
            a,b,u,v=old['xywh']
            area=max(0,min(x+w,a+u)-max(x,a))*max(0,min(y+h,b+v)-max(y,b))
            assert area==0,(c['id'],old['id'],area)
            overlaps.append(dict(current=c['id'],previous=old['id'],overlap_pixels=area))
    assert all(c['combined_equals_old'] for c in audit['cases'])
    assert sum(c['candidate_changed'] for c in audit['cases'])==107
    notes=[dict(id='known_right_writing',component=1,pixels=13,
                observation='Isolated small dot beside an observed dark stroke; main stroke unchanged in the displayed component'),
           dict(id='known_right_writing',component=2,pixels=17,
                observation='Isolated small dot near a stroke end; material unconfirmed'),
           dict(id='known_lower_writing',component=1,pixels=55,
                observation='Dark fragments adjoining an observed stroke and gray secondary image; whether all fragments are unwanted is unresolved'),
           dict(id='known_lower_writing',component=2,pixels=22,
                observation='Small dot close to a vertical stroke and gray secondary image; material unconfirmed')]
    dump(OUT/'visual_review.json',dict(components=notes,
        all_candidate_changes_reviewed=True,no_new_large_stroke_loss_observed_in_these_four_components=True,
        all_stroke_preservation_proven=False,material_pixel_GT=False,
        previous_policy_ranges_disjoint=overlaps,
        important_dirt_improvement_confirmed=False,adopted=False))
    lines=['# 実保存npjの別範囲：目立つ汚れの追加改善は確認できなかった', '',
        '2026-10-04。形状による線保持を固定し、実保存npjの4ページ・6範囲へ適用した。'
        '茶色いシミや灰色の別の像は保存npjで既にかなり除去されていた。新候補の変更は2範囲・4成分・107画素だけで、'
        '従来の線保持と全6範囲で全画素一致した。今回、新しい大きな汚れ除去の肯定例は得られていない。', '',
        '前回の「茶色い筆画周辺111画素の誤編集を防ぎ、hnsdの局所除去を維持」という開発結果は残るが、'
        'そこからnpj全般の改善や別資料への一般化はまだ主張できない。今回条件の一般採用は保留する。', '',
        'まず見る比較：', '',
        '- [茶色いシミ：実保存npjで既に除去](known_upper_stain_brown_spot.png)',
        '- [別ページの下部：原画像／npj／固定条件](new_page_lower_comparison.png)',
        '- [残る灰色の点：固定条件では変更なし](new_page_lower_gray_residual.png)',
        '- [筆画近くの濃い破片：材質の判断が残る](known_lower_writing_all_change_01.png)',
        '- [全23比較PNG](comparison_png.zip)', '',
        '## 結果', '',
        '| 範囲 | 固定v6紙面補修の変更 | 従来線保持の変更 | 色を問わない線保持の変更 | protect接続線保持の変更 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for c in diagnostics:
        m=c['methods']
        lines.append('| [%s](%s_comparison.png) | %d | %d | %d | %d |'%
            (c['id'],c['id'],m['v6']['changed'],m['old_contact']['additional_changed'],
             m['all_colour_ridge']['additional_changed'],m['protect_anchored_ridge']['additional_changed']))
    lines.extend(['', '今回の全ページ枠外補修は全6範囲で適用0。線保持3条件の組合せ結果は全6範囲で同じ。'
        '固定v6紙面補修の2,224／224変更画素は、既に保存npjで除去されたシミ周辺などの紙色の変更も含む。'
        'これを追加の汚れ除去の証拠にしない。', '',
        '新候補の全変更は、右本文対照の13＋17画素と下部本文対照の55＋22画素。'
        '4成分を全て原画像と照合し、今回の拡大範囲には主要な濃い筆画が大きく消える様子は見えなかった。'
        'ただし55画素は筆画の近くの濃い破片で、別の灰色像との重なりがある。'
        '小さな点も汚れ・裏写り・筆画の一部の材質GTがないため、107画素を除去成功画素数として数えない。', '',
        '- [13画素の小点](known_right_writing_all_change_01.png)',
        '- [17画素の小点](known_right_writing_all_change_02.png)',
        '- [55画素の筆画近くの破片](known_lower_writing_all_change_01.png)',
        '- [22画素の細線近くの小点](known_lower_writing_all_change_02.png)', '',
        '原画像に見える茶色い斑点を選んでも、npj出力に大きな汚れが残るとは限らない。'
        '今回の結果から、追加処理の必要性は保存npj側の失敗を見て判断する必要がある。'
        '現方式は広い暗部とnpjの黒残留を対象にするため、残った灰色の点や細い余白の跡には作用しない例もある。'
        '変更0を解決として扱わず、今回の範囲に合わせた閾値調整は行っていない。', '',
        '## 入力と範囲', '',
        '200018243_00002_2は今回の形状条件で初めて試す1ページ。過去の別実験では既出で、未使用冊ではない。'
        '保存された外部紙面参照のローカルnpj v2実行ログと出力を確認した。'
        '当時のソース全体のスナップショットはないため、論文の厳密な再現とは呼ばない。'
        '他の200010454_00002_2、200019865_00003_1、200018243_00003_2は既知ページの別範囲。'
        '全6範囲は前回の17範囲と画素領域が重ならないことを確認したが、同じページ・冊なので独立評価ではない。', '',
        '全4ページで実推論入力と原画像が全画素一致し、保存された全ページ出力から同じ位置を切り出した。'
        '新規DDRM・npj用CSCマスク再生成・文字生成・筆画補完はない。本文ページのみでイラストは除外。'
        '新ページの固定v6のみ全原画像から計算し、既知ページの固定v6は保存分を利用した。'
        '新しい形状条件・旧線保持・枠検出・補間条件は全範囲で共通のまま。', '',
        '入力が原画像と異なるtrial_run、異なる処方／人工汚れを含むfidelity実験は、この実保存比較へ混ぜなかった。'
        '1,127件のOtsu結果を含む保存画像一覧全体を、通常npjの推論済み文書数とは扱わない。', '',
        '## 検証と次の作業', '',
        '原画像・実入力・保存出力の対応、全ページ由来の分類、マスク包含、参照条件の再計算、'
        '必須保持・適用外の一致、参照外画素を異色にする補間の反証試験、組合せ、全パネル、ZIPとハッシュを確認。'
        '事前指定7範囲に加え、出力後の全変更4成分を拡大して見落としを避けた。', '',
        '次は、実保存npjに大きな残留汚れがある例を優先して増やす。'
        '保存済みの対応例が不足する場合は、原画像で損傷を選んでから凍結コードの対照推論を作り、'
        '歴史的保存出力とは区別する。汚れを残していないページへ処理量を増やす方向には進めない。', '',
        '[選定・入力・由来](selection.json)／[固定範囲](witnesses_before_run.json)／'
        '[処理診断](diagnostics.json)／[画素と参照の検証](verification.json)／'
        '[目視記録と前回範囲との重複検査](visual_review.json)'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    shutil.copyfile(__file__,OUT/'package_source_snapshot.py')
    assert len(list(OUT.glob('*.png')))==23
    print('Recorded 6 fixed ranges, 4 tiny changed components, no new large-dirt benefit',flush=True)


if __name__=='__main__':main()
