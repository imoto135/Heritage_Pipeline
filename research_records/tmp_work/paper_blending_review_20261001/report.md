# 保存済みDDRMの紙色・境界補正

2026-10-01。**隣接する未編集紙面を参照すると、虫食い周辺の境界の色差を小さくできた。目視の改善は控えめで、平坦な塗り跡・薄い輪郭はまだ残る。** 利用者の「失敗例は特に問題なさそう」「まだ違和感が残るが前進している」というレビューを受け、現在の余白試作を基準に、背景への馴染みを改善した。

今回の比較は**DDRM使用後の合成後処理**である。前処理のマスク、DDRMの重み・保存済み出力を固定し、追加推論も再学習もしていない。前回と同じ14ページ・11冊・22箇所を全て扱った。npjやv6のマスク精度を新しく比較した実験ではない。前回のnpj→DDRM比較は[前回報告](../paper_margin_defect_review_20261001/report.md)に保存している。

## 比較PNG

- [虫食い2例の補正前後](01_hole_blending.png)：原画像、固定v6→DDRM、現在の余白試作、隣接紙面による紙色＋境界補正。
- [紙色のみ・境界のみ・両方](02_tone_boundary_ablation.png)：両方を加えることが常に最良ではない。
- [初回の紙面平面と隣接紙面の比較](03_reference_ablation.png)：改善しなかった初回条件も残した。
- [枠・見出し・本文の対照](04_controls.png)、[茶色い紙面と印影](05_paper_and_stamp.png)、[この2例の要素比較](06_boundary_only.png)。
- [全比較94枚ZIP](paper_blending_comparison_images.zip)：概要6、隣接紙面比較22、隣接補正の要素比較22、参照方法の比較22、初回の要素比較22。

## 補正の条件

対象は保存済み編集マスクと、前回の上余白候補の共通部分。面積64画素以上の連結成分だけを扱う。本文やマスク外へ補正範囲を広げない。原画像から参照するのは、編集マスク・残存インク保護を除き、局所紙色とのLab差で絞った未編集紙面。参照領域の判定は全ページから作り、補正は保存済みROIと周囲64pxを使用する。全ページの新しい復元画像を生成した実験ではない。

初回は、成分から3〜48px離れた明るい紙面を、暗部の4pxガードと勾配条件でさらに絞り、ロバストなLab平面を当てた。この参照では4箇所全てで境界の色差が増えた。明るく平滑な紙の参照が直近の黄み・陰影からずれることが一因と考えられるが、この説明は参照条件と出力からの推測である。

次に、同じLabの紙面条件を使いつつ、編集領域に直接隣接する未編集画素を参照した。1成分あたり16組以上の4近傍の境界対が得られる場合に補正する。紙色のみは境界対のLab差の中央値を成分全体へ加える。境界のみは、生成画素だけから得た低周波成分（σ12px）と隣接紙面との差を境界上で取り、σ8pxで平滑化して内側へ6pxの尺度で弱める。両方は紙色補正後に境界の残差を加える。補正量はL±6、a/b±4に制限し、新しい粒状模様は加えない。

原画像の編集対象画素を出力に混ぜず、既存の生成画素に色の差分を加える。参照領域を固定して原画像の編集対象を任意の色に置き換えても、穴2例と印影1例の全3方式の出力が完全一致した。これは補正量を計算する段階での不変確認であり、元の参照領域の判定まで損傷画素と独立だと証明するものではない。

## 境界の診断値

編集側の境界画素と、紙面条件を満たす隣接未編集画素とのLab距離の中央値。値が小さいほど、その境界での色の段差が小さい。ΔE2000・画素正解との精度・目視品質の評点ではない。全条件を**初回の対象マスクから作った同じ境界対**で測った。隣接方式の対象が少ない上端紙面も、評価対を選び直していない。

| ROI | 境界対 | 補正前 | 初回平面・両方 | 隣接紙色のみ | 隣接境界のみ | 隣接・両方 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `brown_false_red_new` | 54 | 2.187 | 2.481 | 1.262 | 1.326 | 1.326 |
| `hole_edge_new` | 325 | 3.550 | 3.674 | 2.403 | 1.870 | 1.870 |
| `new_margin_hole` | 205 | 3.526 | 4.425 | 2.467 | 2.039 | 2.086 |
| `new_brown_stamp` | 110 | 2.551 | 5.078 | 1.629 | 1.818 | 1.818 |

穴2例は境界のみで3.550→1.870、3.526→2.039。紙色＋境界は後者が2.086で、境界のみよりわずかに大きい。紙面上端と印影は紙色のみの値が最も小さい。4例から全資料の改善率を推定せず、両方を自動的に最良条件として採用しない。この診断値は補正が参照する境界と共通なので、独立した品質評価でもない。

## 全22箇所と残る問題

| ROI | 初回補正の対象画素 | 隣接方式の対象画素 | 今回の状態 |
| --- | ---: | ---: | --- |
| [濃い茶色の斑点を墨と判定](../adjacent_paper_boundary_pilot_20261001/comparisons/dense_stain.png) · `dense_stain` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [水染みの中の薄い本文の欠損](../adjacent_paper_boundary_pilot_20261001/comparisons/thin_black_water.png) · `thin_black_water` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [茶色い斑点を朱インクと判定](../adjacent_paper_boundary_pilot_20261001/comparisons/brown_as_red.png) · `brown_as_red` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [薄い黒筆画のループが編集候補に入る](../adjacent_paper_boundary_pilot_20261001/comparisons/thin_black_loop.png) · `thin_black_loop` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [対照：本文と薄い青灰色の線](../adjacent_paper_boundary_pilot_20261001/comparisons/thin_black_text.png) · `thin_black_text` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [赤褐色の長い筆画が分断される](../adjacent_paper_boundary_pilot_20261001/comparisons/warm_stroke.png) · `warm_stroke` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [対照：比較的濃い黒本文](../adjacent_paper_boundary_pilot_20261001/comparisons/control_black.png) · `control_black` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [枠線・本文に接する濃い茶色の汚れ](../adjacent_paper_boundary_pilot_20261001/comparisons/dense_stain_new.png) · `dense_stain_new` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [上端の茶色い紙面を朱インクと判定](../adjacent_paper_boundary_pilot_20261001/comparisons/brown_false_red_new.png) · `brown_false_red_new` | 15,079 | 364 | 紙色・境界の診断対象 |
| [虫食い状の穴・裂けの縁を墨と判定](../adjacent_paper_boundary_pilot_20261001/comparisons/hole_edge_new.png) · `hole_edge_new` | 9,460 | 9,460 | 紙色・境界の診断対象 |
| [対照：穴のあるページの通常本文](../adjacent_paper_boundary_pilot_20261001/comparisons/control_text_new.png) · `control_text_new` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [文字のすぐ上の濃い茶色い斑点](../adjacent_paper_boundary_pilot_20261001/comparisons/brown_near_text.png) · `brown_near_text` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [対照：同じページの濃い本文](../adjacent_paper_boundary_pilot_20261001/comparisons/dark_text_control.png) · `dark_text_control` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [別ページの細い筆画の分断](../adjacent_paper_boundary_pilot_20261001/comparisons/faint_left_text.png) · `faint_left_text` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [別冊の穴・裂けの濃い縁](../adjacent_paper_boundary_pilot_20261001/comparisons/damage_control.png) · `damage_control` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [新しい冊の上余白の穴・裂け](../adjacent_paper_boundary_pilot_20261001/comparisons/new_margin_hole.png) · `new_margin_hole` | 7,312 | 7,312 | 紙色・境界の診断対象 |
| [対照：穴の下の枠と本文](../adjacent_paper_boundary_pilot_20261001/comparisons/new_frame_text.png) · `new_frame_text` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [上余白の細い曲線：原因未確定](../adjacent_paper_boundary_pilot_20261001/comparisons/new_uncertain_trace.png) · `new_uncertain_trace` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [対照：本文外の茶色い印影](../adjacent_paper_boundary_pilot_20261001/comparisons/new_brown_stamp.png) · `new_brown_stamp` | 555 | 555 | 紙色・境界の診断対象 |
| [対照：余白内の見出しと囲み](../adjacent_paper_boundary_pilot_20261001/comparisons/new_header_ink.png) · `new_header_ink` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [対照：白黒資料の上余白](../adjacent_paper_boundary_pilot_20261001/comparisons/new_blank_paper.png) · `new_blank_paper` | 0 | 0 | 補正を加えず保存出力と完全一致 |
| [対照：白黒資料の本文](../adjacent_paper_boundary_pilot_20261001/comparisons/new_mono_text.png) · `new_mono_text` | 0 | 0 | 補正を加えず保存出力と完全一致 |

穴の既存例は色の境目が少し弱くなるが、薄い縁と平坦な領域が残る。新しい冊の穴は左・下側の色が隣接紙面へ近づく一方、右上の濃い損傷は前回からマスク外にあり、そのまま残る。ここは色の補正だけでは消せない。茶色い紙面上端は大半の成分に十分な境界対がなく、初回15,079画素に対して今回は364画素だけを補正した。広い紙色ムラを解決した結果ではない。

印影の小さな点は前回マスクで既に消えている。この変化は利用者レビューで現在の研究継続の阻害要因とは扱わない判断になった。今回の補正でその点を回復したとも、印影保護を解決したとも数えない。ほか18箇所は補正対象0で、現在の試作出力と完全一致した。条件で対象外だった結果であり、未知資料の文字保存精度を示す数字ではない。

紙の質感については、Lの高周波成分（σ1.2px）の標準偏差も保存した。穴の既存例では初回の参照0.491に対して生成側0.628、新しい穴では参照0.205に対して生成側0.401だった。参照側は平滑な紙面に偏る条件なので、この差を真の紙の分散差と見なさない。少なくとも「平坦に見えるので一律にノイズを加える」という根拠は得られていない。粒の量だけでなく模様の空間的なつながりが残る診断課題である。

## 確認と次の段階

入力・マスク・重み・保存DDRM・ソース・PNG/診断値のハッシュ、22箇所の原寸画素、補正対象外の保存DDRMとの一致、編集対象外の原画素との一致、参照と編集マスクの非重複を確認した。境界対を別実装で列挙して全条件の診断値を再計算し、ZIPの94PNGも内容ハッシュで確認した。[確認JSON](verification.json)。

次の開発比較では、隣接紙面による境界補正を基準として残す。紙色のみ・境界のみ・両方の画像も保持する。残る違和感は、マスク外に残った細い損傷と、編集内部の紙模様の不連続を分けて検証する。前者のマスク範囲を変える比較と、後者の合成を変える比較は別条件にする。今回の22箇所は開発例であり、正式な精度評価・新手法の確立・全資料への適用判断はまだ行わない。

初回[条件と全診断](../paper_tone_boundary_pilot_20261001/diagnostics.json)、隣接方式[条件](../adjacent_paper_boundary_pilot_20261001/manifest.json)・[全診断](../adjacent_paper_boundary_pilot_20261001/diagnostics.json)。実装：[初回紙面平面](../../scripts/benchmark/probe_paper_tone_boundary.py)、[隣接境界](../../scripts/benchmark/probe_adjacent_paper_boundary.py)、[検証・PNG梱包](../../scripts/benchmark/package_paper_blending_review.py)。[利用者レビューによる方針更新](../paper_margin_defect_user_review_20261001.md)。
