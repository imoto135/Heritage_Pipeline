# 上余白帯の上端で除外された損傷の回収

2026-10-01。**右上に残っていた損傷を編集対象へ回収できた。前処理は前進したが、同じDDRMでは暗い線状の模様と塗り跡が発生し、自然な修復はまだ未達。** 同じマスクのTeleaでは暗い模様が少なくなる一方、ぼけ・色ムラが残る。今回のマスク候補を開発比較用に保存し、復元品質を改善した完成版とは扱わない。

前回[紙模様の試作](../paper_pattern_review_20261001/report.md)で、濃い残存損傷はマスク外にあると分離した。今回は前処理だけを変更し、全14ページ・11冊・22箇所を保持して比較した。調整済みの開発例であり、独立評価・画素GT・全資料の改善率はない。イラストページは従来どおり対象外。

## 比較PNG

- [マスクのアブレーション](01_mask_ablation.png)：原画像、前回、上端解除の帯内だけ、上方まで全成分を回収。黄＝編集、青＝保護、赤＝朱保護、紫＝保留。
- [同じDDRM後](02_same_ddrm.png)：重み・20段・空間ノイズを固定し、マスクだけ変更した主要比較。
- [前回と同じ境界補正](03_fixed_boundary.png)：補助比較。補正対象の旧余白帯も固定したため、新しく上方へ追加した領域には境界補正を加えない。
- [同じマスクのDDRMとTelea](04_same_mask_fill.png)、[上方の拡大](06_top_closeup.png)。Teleaは局所補間でDDRMではない。
- [本文・枠・印影・見出しの対照](05_controls.png)、[保存済みnpj・固定v6から今回まで](07_npj_v6_context.png)。
- [全比較109PNG ZIP](margin_top_completion_comparison_images.zip)：概要7、全ページのマスク14、ROIマスク22、同じDDRM22、固定境界補正22、復元器診断22。

## 変更した判定

従来は上余白帯のどの端に触れた成分も一括除外していた。問題のページの旧帯はy=127〜307、本文の上枠はy=340。右上の損傷2成分は帯の上端に接し、それ以外の境界には接していなかった。

今回も色の種32画素以上、面積128〜ページ1%、最大半径6px以上、最大辺が短辺20%以下という以前の形状条件を使う。**上端だけ**に触れる成分に限定して除外を解除し、側端・本文側の端に触れた成分は解除しない。帯内のみの条件は元の帯の中で3pxの周囲を回収する。

全成分の条件では同じ成分を、ページ上端16px・左右6%・本文枠の上32pxを除く領域で上方へ追う。暗い支持領域の閾値と3pxの周囲の条件も固定した。横に長い暗い線（ページ幅55%以上）は8pxの保護幅を付けて接続の障壁にする。元の朱・追加暖色保留と暖色接続保護を解除しない。既存の編集領域は全て維持し、その外だけを追加する。

ROIの座標や「穴」のラベルでマスクを作っていない。全ページに同じ規則を適用した。新しい帯を一律に汚れと認定せず、旧帯内で条件を満たした上端接触成分だけを追う。余白の位置を文字なしの保証にする方法ではない。

全14ページのうち変更は`100249416_00024_1`だけ。帯内解除は前回から3,585画素、全成分回収は6,730画素を追加した。全成分のうち3,144画素は旧帯の上方、3,586画素は帯内。全成分化による周囲の形状差で帯内解除から帯内にも1画素の差がある。面積は回収候補の大きさで、正解・検出率ではない。

## 同じDDRMと復元器の切り分け

旧のnpj・v6・余白DDRMはハッシュ確認した保存出力を使用した。今回の2マスクは同じモデル・原画像・128pxパッチ・重複32px・周囲64px・20段・画像空間ガウスノイズで比較した。欠損に原画像を混ぜ戻す処理はなく、編集範囲外は原画素へ厳密に戻す。変更したROIだけ各13パッチを再推論し、ほか21箇所は周辺マスクまで同一なので保存DDRMを再利用した。同一マスク・ノイズの再実行の一致も確認した。

帯内だけの解除では、帯より上にある暗い部分が原画素として残る。全成分回収はそこも大きく薄くする一方、上方に灰色の線・塊と平坦な塗り跡を生む。前回の境界補正を同じ旧帯に適用しても、上方の新しい編集部分には適用されず、暗い生成模様は解決しない。境界補正の範囲や紙模様の方式を同時に変えて結果を混ぜなかった。

DDRMの出力を見てから、同じ原画像・マスク・64pxの文脈をTelea半径3で補間する診断を追加した。全22箇所を扱い、旧マスクのTeleaは以前の保存出力と完全一致した。全成分回収のTeleaではDDRMの上方の灰色の模様は目立たないが、暖色のにじみや平坦な色の塊が残る。従って「編集対象へ戻せない」問題は低減したが、「自然な紙面へ復元する」問題は続いている。

全成分回収で追加した**同じ6,730画素**に対し、L≤80の画素数とLの中央値を測った。暗い部分の残り方の観測値で、除去精度・損傷の正解率・真正な復元の指標ではない。原画像と前回DDRMが一致するのは、この画素集合が以前は編集範囲外だったため。

| 条件 | L≤80の画素数 | L中央値 |
| --- | ---: | ---: |
| 原画像 | 5,101 | 69.855 |
| 前回のマスク→DDRM | 5,101 | 69.855 |
| 帯内解除→DDRM | 2,511 | 88.177 |
| 全成分回収→DDRM | 1,092 | 94.376 |
| 全成分回収→Telea | 116 | 95.148 |

Telea側の暗い画素が少なくても、明るく平滑にするだけで良い品質と数えない。既存の128pxパッチ用DDRMモデルを使い、紙面専用には再学習していない。紙面だけの編集にも同じ接続を用いているが、暗い模様の原因を学習データだけと断定はしない。周囲の暗い枠、欠損形状、モデル・サンプラーの影響を今回分離していない。

## 全22箇所

| ROI | 帯内解除の追加 | 全成分回収の追加 | 実際のDDRM |
| --- | ---: | ---: | --- |
| [濃い茶色の斑点を墨と判定](../margin_top_completion_pilot_20261001/comparisons/dense_stain.png) · `dense_stain` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [水染みの中の薄い本文の欠損](../margin_top_completion_pilot_20261001/comparisons/thin_black_water.png) · `thin_black_water` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [茶色い斑点を朱インクと判定](../margin_top_completion_pilot_20261001/comparisons/brown_as_red.png) · `brown_as_red` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [薄い黒筆画のループが編集候補に入る](../margin_top_completion_pilot_20261001/comparisons/thin_black_loop.png) · `thin_black_loop` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [対照：本文と薄い青灰色の線](../margin_top_completion_pilot_20261001/comparisons/thin_black_text.png) · `thin_black_text` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [赤褐色の長い筆画が分断される](../margin_top_completion_pilot_20261001/comparisons/warm_stroke.png) · `warm_stroke` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [対照：比較的濃い黒本文](../margin_top_completion_pilot_20261001/comparisons/control_black.png) · `control_black` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [枠線・本文に接する濃い茶色の汚れ](../margin_top_completion_pilot_20261001/comparisons/dense_stain_new.png) · `dense_stain_new` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [上端の茶色い紙面を朱インクと判定](../margin_top_completion_pilot_20261001/comparisons/brown_false_red_new.png) · `brown_false_red_new` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [虫食い状の穴・裂けの縁を墨と判定](../margin_top_completion_pilot_20261001/comparisons/hole_edge_new.png) · `hole_edge_new` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [対照：穴のあるページの通常本文](../margin_top_completion_pilot_20261001/comparisons/control_text_new.png) · `control_text_new` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [文字のすぐ上の濃い茶色い斑点](../margin_top_completion_pilot_20261001/comparisons/brown_near_text.png) · `brown_near_text` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [対照：同じページの濃い本文](../margin_top_completion_pilot_20261001/comparisons/dark_text_control.png) · `dark_text_control` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [別ページの細い筆画の分断](../margin_top_completion_pilot_20261001/comparisons/faint_left_text.png) · `faint_left_text` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [別冊の穴・裂けの濃い縁](../margin_top_completion_pilot_20261001/comparisons/damage_control.png) · `damage_control` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [新しい冊の上余白の穴・裂け](../margin_top_completion_pilot_20261001/comparisons/new_margin_hole.png) · `new_margin_hole` | 3,585 | 6,730 | 再推論、復元品質は未完成 |
| [対照：穴の下の枠と本文](../margin_top_completion_pilot_20261001/comparisons/new_frame_text.png) · `new_frame_text` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [上余白の細い曲線：原因未確定](../margin_top_completion_pilot_20261001/comparisons/new_uncertain_trace.png) · `new_uncertain_trace` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [対照：本文外の茶色い印影](../margin_top_completion_pilot_20261001/comparisons/new_brown_stamp.png) · `new_brown_stamp` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [対照：余白内の見出しと囲み](../margin_top_completion_pilot_20261001/comparisons/new_header_ink.png) · `new_header_ink` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [対照：白黒資料の上余白](../margin_top_completion_pilot_20261001/comparisons/new_blank_paper.png) · `new_blank_paper` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |
| [対照：白黒資料の本文](../margin_top_completion_pilot_20261001/comparisons/new_mono_text.png) · `new_mono_text` | 0 | 0 | 周辺マスク同一・保存出力と完全一致 |

13ページの全ページマスクと、ほか21ROIの周辺マスク・DDRM出力・固定境界補正出力は以前と完全一致した。印影の以前からの小さな点の変化は利用者レビューで許容されたものとして保持し、今回の追加変化と扱わない。本文と重なるシミ、細い裂け、欠損文字は未解決のまま。

## 確認と次の段階

入力・既存マスク・ソース・モデル重み・DDRM核・保存出力のハッシュ、全ページのマスクの演算と保護、原寸のPNG、編集範囲外の不変、周辺マスクが同一の場合の出力再利用、前回境界補正と旧Teleaの再現、ZIP内109PNGのハッシュを確認した。[確認JSON](verification.json)。

上端接触を回収する候補マスクを次の**開発用の検出基準**として保存する。今回のDDRM経路が自然な紙面を作るとは確認できなかったため、実運用のv6や全資料へ置き換えない。次はこの固定マスクのまま、紙面と確認した領域について、周囲の紙色を使う復元とDDRMの接続を比較する。紙色の低周波と暗い生成模様を分け、損傷を除去しても新たな線を生まない条件を調べる。本文・印影との区別を別に検証し、余白全部を紙面補間へ流さない。紙模様の前回試作をこの比較へ自動的に足さない。

[マスクの固定条件](../margin_top_completion_pilot_20261001/manifest.json)・[成分の判定](../margin_top_completion_pilot_20261001/pages/100249416_00024_1/components.json)、[DDRM条件](../margin_top_completion_ddrm_20261001/manifest.json)、[復元器診断の条件](../margin_top_completion_fill_diagnostic_20261001/manifest.json)。実装：[前処理](../../scripts/benchmark/probe_margin_top_completion.py)、[固定DDRM](../../scripts/benchmark/check_margin_top_completion_ddrm.py)、[Telea診断](../../scripts/benchmark/diagnose_margin_top_fill.py)、[検証・梱包](../../scripts/benchmark/package_margin_top_completion_review.py)。
