# 上余白の損傷の検出：濃い縁の回収と印影の誤編集

2026-10-01。**前処理を変えるだけで穴の濃い縁をより広く編集対象に戻し、同じDDRMで薄くできた。ただし上余白の印影の点まで消す失敗があり、今回の試作も採用しない。** 前回の11ページ・15箇所に、原画像から先に選んだ新しい3ページ・3冊・7箇所を加え、計14ページ・11冊・22箇所で確認した。閾値を出力結果で再調整していない。画素GT、全資料の改善率、新手法としての確立を示した実験ではない。

## 比較PNG

- [穴の分類比較](01_hole_masks.png)：原画像、保存済みnpj、固定v6、今回の試作。
- [同じDDRM後の穴](02_hole_ddrm.png)：既存の穴の濃い部分は大きく薄くなる。新しい穴は部分的。
- [印影の点が消える反例](03_stamp_regression.png)：追加編集555画素。v6では残る図柄内の点が試作では消える。
- [同じマスクでDDRMと局所補間を比較](04_fill_diagnostic.png)：TeleaはDDRMではない。印影の点は両方で消える。
- [未解決例](05_remaining.png)、[新しい冊の枠・見出し・本文対照](06_ink_controls.png)。
- [全比較108枚ZIP](paper_margin_defect_comparison_images.zip)：概要6、全ページ分類14、ROI分類22、マスクのアブレーション22、同じDDRM22、復元器診断22。

分類の青＝保護、赤＝朱候補として保護、黄＝編集候補、紫＝保留。候補数は対象面積であり、Accuracy・Recall・改善率ではない。

## 入力の選定

過去の固定v6・別冊比較・npj失敗比較で扱った冊を除外し、既存の原画像走査スコアから18ページを目視した。茶色色差のランキングにはイラストや表紙が多く、濃淡差のランキングは白黒の文字・枠・余白でも高得点となる。これらの走査スコアを損傷の正解にしない。

イラストのページ自体は対象外にしたうえで、同じ冊の次の6ページを原画像だけで追加確認した。上余白に穴・裂けがあり本文と離れている最初の本文ページ`100249416_00024_1`を固定。`brsk003_003`には原因未確定の細い曲線、本文外の茶色い印影、囲み付き見出しを選び、`200008003_00004_2`は白黒の余白・本文の対照にした。計24ページの原画像レビューでイラスト6・表紙1を除外。採用3ページは、v6や今回の候補の出力を見る前に[選定JSON](../paper_margin_defect_pilot_20261001/baseline_review.json)へ保存した。

[原画像走査と除外の記録](../paper_defect_screening_20261001/visual_selection.json)、[最初の原画像シート](../paper_defect_screening_20261001/original_sheet_01.png)、[隣接ページの原画像シート](../paper_defect_screening_20261001/original_followup_book.png)。新しい冊は比較履歴から除外した冊であり、DDRMの学習資料からの独立は確認していない。正式な独立精度評価でも代表抽出でもない。今回出力を確認した3ページは、以後の調整に使うなら開発側に置く。

## 試作の判断規則

固定v6を土台に、原画像の長い中性の横線から本文の上枠を探し、その上の帯を余白候補とした。横線が見つからないページは追加しない。帯の境界やページ端には余裕を残す。前回のページ全体に対する茶色の追加編集を引き継がず、今回の追加はこの上余白候補に限定する。

近傍の明るい紙からの色差で茶色の種を作り、同じ帯にある暗い支持領域に接続して、色が中性に近い濃い縁にも候補を広げた。面積・太さ・大きさの条件と、帯の境界に触れる成分の除外を入れた。固定v6の朱候補・追加暖色保留および前回の暖色接続保護は維持する。マスクは全ページから推定し、ROI座標や問題ラベルで候補を作っていない。

茶色のみと暗い支持領域への接続の比較は、同じ帯・種・成分選択・周囲3px・暖色保護を使い、最終候補の色条件だけを変えた。既存の穴では追加7,879→8,987画素で、中性寄りの縁を1,108画素追加。新しい穴では5,270→7,215画素で1,945画素追加。一方、印影にも247→555画素の追加となる。[マスクのアブレーション](../paper_margin_defect_pilot_20261001/candidate_ablation/hole_edge_new.png)、[印影のアブレーション](../paper_margin_defect_pilot_20261001/candidate_ablation/new_brown_stamp.png)。各アブレーションのDDRMを実行した比較ではない。

上余白という位置は「文字がない」という証明ではない。実際に印影の小さな点を孤立汚れとして拾った。余白内の小さな成分を色・太さだけで損傷と判定する規則は不十分だった。保護を外した点がDDRMでもTeleaでも消えるため、この例は復元器の問題に帰属させない。

## 全22箇所の所見

| 箇所・分類PNG | 茶色のみ追加 | 暗い縁への接続後 | 分類の所見 | 同じDDRM後の所見・PNG |
| --- | ---: | ---: | --- | --- |
| [濃い茶色の斑点を墨と判定](../paper_margin_defect_pilot_20261001/candidate_comparisons/dense_stain.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/dense_stain.png) |
| [水染みの中の薄い本文の欠損](../paper_margin_defect_pilot_20261001/candidate_comparisons/thin_black_water.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/thin_black_water.png) |
| [茶色い斑点を朱インクと判定](../paper_margin_defect_pilot_20261001/candidate_comparisons/brown_as_red.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/brown_as_red.png) |
| [薄い黒筆画のループが編集候補に入る](../paper_margin_defect_pilot_20261001/candidate_comparisons/thin_black_loop.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/thin_black_loop.png) |
| [対照：本文と薄い青灰色の線](../paper_margin_defect_pilot_20261001/candidate_comparisons/thin_black_text.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/thin_black_text.png) |
| [赤褐色の長い筆画が分断される](../paper_margin_defect_pilot_20261001/candidate_comparisons/warm_stroke.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/warm_stroke.png) |
| [対照：比較的濃い黒本文](../paper_margin_defect_pilot_20261001/candidate_comparisons/control_black.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/control_black.png) |
| [枠線・本文に接する濃い茶色の汚れ](../paper_margin_defect_pilot_20261001/candidate_comparisons/dense_stain_new.png) | 0 | 0 | 追加0。本文に重なるオリーブ色の汚れは今回の対象外。 | [v6と同一出力。未解決。](../paper_margin_defect_ddrm_20261001/comparisons/dense_stain_new.png) |
| [上端の茶色い紙面を朱インクと判定](../paper_margin_defect_pilot_20261001/candidate_comparisons/brown_false_red_new.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/brown_false_red_new.png) |
| [虫食い状の穴・裂けの縁を墨と判定](../paper_margin_defect_pilot_20261001/candidate_comparisons/hole_edge_new.png) | 7,879 | 8,987 | 茶色から暗い縁への接続で追加8,987画素。前回より濃い縁を広く編集候補化。 | [DDRMで濃い部分が大きく薄くなる。薄い縁・平坦な塗り跡は残る。局所補間にも色差・ぼけ。](../paper_margin_defect_ddrm_20261001/comparisons/hole_edge_new.png) |
| [対照：穴のあるページの通常本文](../paper_margin_defect_pilot_20261001/candidate_comparisons/control_text_new.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/control_text_new.png) |
| [文字のすぐ上の濃い茶色い斑点](../paper_margin_defect_pilot_20261001/candidate_comparisons/brown_near_text.png) | 0 | 0 | 追加0。本文内は今回の追加編集対象外。 | [v6と同一出力。茶色い部分は残る。前回試作の文字状生成を解決したとは数えない。](../paper_margin_defect_ddrm_20261001/comparisons/brown_near_text.png) |
| [対照：同じページの濃い本文](../paper_margin_defect_pilot_20261001/candidate_comparisons/dark_text_control.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/dark_text_control.png) |
| [別ページの細い筆画の分断](../paper_margin_defect_pilot_20261001/candidate_comparisons/faint_left_text.png) | 0 | 0 | 追加0。今回の範囲を限定した結果で、一般的な筆画判別の改善ではない。 | [v6と同一出力。旧結果を維持し、新しい修復の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/faint_left_text.png) |
| [別冊の穴・裂けの濃い縁](../paper_margin_defect_pilot_20261001/candidate_comparisons/damage_control.png) | 0 | 0 | 追加0。細い損傷は色の種・太さ等の条件を満たさず、濃い縁は残る。 | [v6と同一出力。未解決。IDにcontrolとあるが問題例。](../paper_margin_defect_ddrm_20261001/comparisons/damage_control.png) |
| [新しい冊の上余白の穴・裂け](../paper_margin_defect_pilot_20261001/candidate_comparisons/new_margin_hole.png) | 5,270 | 7,215 | 新しい冊で追加7,215画素。上端に触れる成分は除外され、欠損の右上は保護が残る。 | [左側・下側は薄くなるが右上が残る。両復元器とも部分的な処理で、自然な全体修復は未達。](../paper_margin_defect_ddrm_20261001/comparisons/new_margin_hole.png) |
| [対照：穴の下の枠と本文](../paper_margin_defect_pilot_20261001/candidate_comparisons/new_frame_text.png) | 0 | 0 | 追加0。穴の下の枠・本文を新たに編集しない。 | [概観で枠と本文を維持。周辺マスクが違うためDDRMは再推論。](../paper_margin_defect_ddrm_20261001/comparisons/new_frame_text.png) |
| [上余白の細い曲線：原因未確定](../paper_margin_defect_pilot_20261001/candidate_comparisons/new_uncertain_trace.png) | 0 | 0 | 追加0。曲線の原因は未確定で、損傷の正解例ではない。 | [v6と同じ原画素。保持できたことを損傷検出の成功として数えない。](../paper_margin_defect_ddrm_20261001/comparisons/new_uncertain_trace.png) |
| [対照：本文外の茶色い印影](../paper_margin_defect_pilot_20261001/candidate_comparisons/new_brown_stamp.png) | 247 | 555 | 対照の印影に555画素追加。印影内の小さな点を汚れ候補化する明確な失敗。 | [DDRM・局所補間の両方で印影の点が消える。前処理由来の誤編集、採用不可。](../paper_margin_defect_ddrm_20261001/comparisons/new_brown_stamp.png) |
| [対照：余白内の見出しと囲み](../paper_margin_defect_pilot_20261001/candidate_comparisons/new_header_ink.png) | 0 | 0 | 追加0。上余白内の見出しと囲みを新たに編集しない。 | [v6と同じ原画素。見出し・囲みを維持。](../paper_margin_defect_ddrm_20261001/comparisons/new_header_ink.png) |
| [対照：白黒資料の上余白](../paper_margin_defect_pilot_20261001/candidate_comparisons/new_blank_paper.png) | 0 | 0 | 追加0。余白ROIは枠と本文端も含む。走査の濃淡スコアを損傷の正解にしない。 | [v6と同じ原画素。細かい濃淡は残る。](../paper_margin_defect_ddrm_20261001/comparisons/new_blank_paper.png) |
| [対照：白黒資料の本文](../paper_margin_defect_pilot_20261001/candidate_comparisons/new_mono_text.png) | 0 | 0 | 追加0。白黒資料の本文を新たに編集しない。 | [v6と同じ原画素。npj編集候補接続で発生する周縁の変形を避ける。](../paper_margin_defect_ddrm_20261001/comparisons/new_mono_text.png) |

既存の穴は中心・縁を広く処理できたが、薄い縁と平坦な塗り跡が残る。新しい穴の右上は帯の境界に触れる成分として除外され、保護された部分が残る。境界ガードを外して良いかは、紙の端や筆画の誤編集と合わせて検証する必要がある。別冊の細い裂けは太さ・茶色の種等の条件を満たさず追加0。本文に接する茶色い部分や重なるオリーブ色の汚れも今回の対象外で、解決したと数えない。

元の本文対照や暖色筆画の追加編集は0だが、多くは対象帯の外か横線が見つからないページだった。これを一般的な筆画判別性能の改善と数えない。全ページの追加は、既存の穴のページ17,319、新しい穴のページ7,215、印影のページ555画素、他11ページは0。切り出し以外の追加画素の正誤を全て判定したものではなく、全ページDDRMは実行していない。

## 復元条件と検証

旧npjの`04_degradation`、固定v6の`edit`、今回の候補`edit`を同じDDRM inpaintingの欠損領域として接続。旧npjの黒文字再生成・朱合成を含む全工程の再現ではない。原画像、文字モデル`ckpt_epoch0350.pth`、RGB [-1,1]、20段、sigma=0.05、etaB=1、etaA/etaC=0.5、128pxパッチ・重なり32px・周辺64px・画像座標にそろえたGaussianノイズ・合成規則を固定した。samplerや学習済み重みは変更していない。[以前のDDRM対応条件](../npj_v6_ddrm_edit_comparison_20261001/report.md)。

旧15箇所のnpj/v6出力はハッシュ確認した保存結果を再利用。新しい7箇所のnpj/v6は同じ条件で今回実行。周辺も含めて候補マスクがv6と一致した18箇所はv6出力を再利用した。ROI内追加0でも周辺の候補が違う場合は再推論し、編集領域外は原画素へ戻した。

補助診断は同じ候補マスク・原画像・周辺64pxを使ったOpenCV Telea半径3。DDRMでないことを各PNGに明記した。両方に塗り跡は残り、今回は局所補間が自然な修復を達成したという結果でもない。今回の穴2箇所で前回のような目立つ文字状の生成は見られないが、他の編集領域での生成問題を解決したとは言わない。

入力・ソース・重み・保存結果のSHA-256、14ページのマスク演算、朱／暖色保留の維持、保護と編集の非重複、全22箇所の原寸入力・編集マスク・出力・編集範囲外の不変、同じマスクとノイズによる出力一致、ZIPのCRCと全画像ハッシュを検証した。[独立した整合性検証](verification.json)、[DDRM実行時の検証](../paper_margin_defect_ddrm_20261001/verification.json)、[暫定目視所見](visual_review.json)。整合性が通ったことは精度や真正性の保証ではない。

## 次の課題

前処理に取り組む意味は確認できたが、今回は採用せず固定v6を比較基準として残す。次は印影の反例を必須対照にして、**離れた小さな点を含む筆記・印影の構造を保護する判断**を、損傷の縁の回収と同時に評価する。閾値をこの印影の点だけに合わせて変える前に、印影内の点・細い筆画・孤立汚れ・穴の縁の少数評価領域を整理し、追加の未使用冊で条件を固定した確認を行う。

本文内・筆画と重なる損傷への一般化は今回達成していない。復元はマスクを固定した比較を継続し、前処理の印影消失と紙の塗り跡を別に記録する。今回の結果だけを理由に拡散モデル本体の学習変更を開始しない。

実装：[全ページ推定](../../scripts/benchmark/probe_paper_margin_defects.py)、[固定DDRMと補間診断](../../scripts/benchmark/check_paper_margin_defect_ddrm.py)、[検証・画像整理](../../scripts/benchmark/package_paper_margin_defect_review.py)。保存済みnpj・固定v6・以前の試作・利用者の既存変更を維持した。
