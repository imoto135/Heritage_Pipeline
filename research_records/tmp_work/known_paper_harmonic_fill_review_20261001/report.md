# 固定マスクの紙面を、未編集紙色だけから補間する

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
| `brown_false_red_new` | 224 | 0 | 0 | 0 | 0 |
| `hole_edge_new` | 9394 | 12 | 0 | 0 | 0 |
| `new_margin_hole` | 14038 | 1092 | 117 | 0 | 0 |
| `new_brown_stamp` | 555 | 551 | 555 | 555 | 555 |

印影の紙面は元々灰色で、周囲の紙色にもL≤80が多い。この555画素を、暗い生成模様が残った穴と同じ失敗として数えない。原画像のインクの点は以前のマスクで既に失われており、今回の補間はその点の復元でも保護の解決でもない。利用者が許容した変化を今回の追加誤編集と扱わない。

既存の穴では直接境界が周辺の黄みを強く引き継ぎ、平滑境界はより明るい塗り跡になる。新しい穴も直接境界と平滑境界で紙色が違い、両方に平坦さと輪郭が残る。σ6pxの参照平滑化による紙色のずれは残る比較事項であり、平滑境界を自動的に最良条件としない。マスク外に残った薄い傷や、保護された小領域は補間だけでは消せない。

広い上端の茶色い紙面では、適用候補15,577画素のうち224画素だけが境界参照の条件を満たした。大部分は保存DDRMのままで、紙色ムラ全体を解決した結果ではない。ほか18箇所も紙色補間を適用せず、2方式とも保存DDRMと完全一致した。条件による対象外であり、未知資料の文字保存精度の証明ではない。

## 全22箇所の範囲

| ROI | 固定編集マスク | 紙色補間の適用候補 | 実際に補間 |
| --- | ---: | ---: | ---: |
| [濃い茶色の斑点を墨と判定](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/dense_stain.png) · `dense_stain` | 3,356 | 0 | 0 |
| [水染みの中の薄い本文の欠損](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/thin_black_water.png) · `thin_black_water` | 10 | 0 | 0 |
| [茶色い斑点を朱インクと判定](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/brown_as_red.png) · `brown_as_red` | 3,661 | 0 | 0 |
| [薄い黒筆画のループが編集候補に入る](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/thin_black_loop.png) · `thin_black_loop` | 0 | 0 | 0 |
| [対照：本文と薄い青灰色の線](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/thin_black_text.png) · `thin_black_text` | 2 | 0 | 0 |
| [赤褐色の長い筆画が分断される](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/warm_stroke.png) · `warm_stroke` | 386 | 0 | 0 |
| [対照：比較的濃い黒本文](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/control_black.png) · `control_black` | 4 | 0 | 0 |
| [枠線・本文に接する濃い茶色の汚れ](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/dense_stain_new.png) · `dense_stain_new` | 39 | 0 | 0 |
| [上端の茶色い紙面を朱インクと判定](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/brown_false_red_new.png) · `brown_false_red_new` | 15,577 | 15,577 | 224 |
| [虫食い状の穴・裂けの縁を墨と判定](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/hole_edge_new.png) · `hole_edge_new` | 9,632 | 9,632 | 9,394 |
| [対照：穴のあるページの通常本文](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/control_text_new.png) · `control_text_new` | 0 | 0 | 0 |
| [文字のすぐ上の濃い茶色い斑点](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/brown_near_text.png) · `brown_near_text` | 1,357 | 0 | 0 |
| [対照：同じページの濃い本文](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/dark_text_control.png) · `dark_text_control` | 43 | 0 | 0 |
| [別ページの細い筆画の分断](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/faint_left_text.png) · `faint_left_text` | 66 | 0 | 0 |
| [別冊の穴・裂けの濃い縁](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/damage_control.png) · `damage_control` | 116 | 114 | 0 |
| [新しい冊の上余白の穴・裂け](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/new_margin_hole.png) · `new_margin_hole` | 14,061 | 14,051 | 14,038 |
| [対照：穴の下の枠と本文](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/new_frame_text.png) · `new_frame_text` | 0 | 0 | 0 |
| [上余白の細い曲線：原因未確定](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/new_uncertain_trace.png) · `new_uncertain_trace` | 0 | 0 | 0 |
| [対照：本文外の茶色い印影](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/new_brown_stamp.png) · `new_brown_stamp` | 562 | 562 | 555 |
| [対照：余白内の見出しと囲み](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/new_header_ink.png) · `new_header_ink` | 0 | 0 | 0 |
| [対照：白黒資料の上余白](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/new_blank_paper.png) · `new_blank_paper` | 0 | 0 | 0 |
| [対照：白黒資料の本文](../known_paper_harmonic_fill_pilot_v2_20261001/comparisons/new_mono_text.png) · `new_mono_text` | 0 | 0 | 0 |

## 確認と次の段階

保存入力・マスク・DDRM・Telea・ソースのハッシュと原寸画素を確認した。全参照が編集範囲外であること、固定の適用候補との一致、補間範囲外は保存DDRM、編集範囲外は原画素との一致を確認した。原画像の編集対象を任意の色に置き換えても、また補間部分のDDRMを黒で置き換えても、4適用例の出力は完全一致した。これは参照領域を固定した補間段階の確認であり、参照ゲート自体の正しさの証明ではない。

定数の紙面を欠損の内外で同じ値に復元する確認と、参照がない場合に補間しない確認を実施した。初回の定数テストでは、float32の境界値から微小許容誤差を引く検証式で許容幅が丸められ、正しい解を拒否することが分かった。初回結果とソースを保存し、境界値の比較だけをfloat64へ修正したv2で再確認した。全22箇所の全ネイティブPNGは初回とハッシュが同一で、補間画像の算出は変えていない。

保存された場に対して、行列を組む実装を使わず4近傍の差を足し合わせ、調和方程式の残差を再計算した。解が参照の各Lab成分の最小・最大を超えないことも実行時に確認した。この性質は正しい参照が得られた場合の色の制約であり、紙模様や資料の内容を真正に復元する保証ではない。[確認JSON](verification.json)。ZIP51PNGの内容ハッシュも一致した。

紙色のみの補間を、暗い生成模様を引き継がない**開発用の復元基準**として保存する。マスクをさらに広げず、次は固定した補間結果について、塗り跡の原因である境界の紙色差と内部の紙模様不足を分けて比較する。前回の模様移植は参照の平滑さ・重複平均でさらに平坦化したため、その条件を自動的に足さない。紙面の参照が不足する例と、印影・本文との適用区別も引き続き必要。新手法の確立・全資料への採用はまだ判断しない。

[固定条件](../known_paper_harmonic_fill_pilot_v2_20261001/manifest.json)、[全診断](../known_paper_harmonic_fill_pilot_v2_20261001/diagnostics.json)、[原寸出力](../known_paper_harmonic_fill_pilot_v2_20261001/native)、[補間実装v2](../../scripts/benchmark/probe_known_paper_harmonic_fill_v2.py)、[初回ソース](../../scripts/benchmark/probe_known_paper_harmonic_fill.py)、[別実装による確認・梱包](../../scripts/benchmark/package_known_paper_fill_review.py)。
