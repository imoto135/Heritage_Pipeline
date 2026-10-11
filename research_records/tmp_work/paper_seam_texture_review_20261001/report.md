# 固定した紙色補間の境界と紙模様を改善する

2026-10-01。**虫食い2例で境界の紙色差が小さくなった。未編集紙面の模様を加えると既存の穴の平坦さは和らぐが、新しい穴では変化が小さい。薄い縁・塗り跡は残り、自然な修復の完成には至っていない。**

同じ14ページ・11冊・22箇所の開発例で、前回の全成分マスク、紙色補間の適用範囲、紙面参照、保存された直接／平滑のLab補間場を固定した。前回の平滑境界画像を比較基準にする。今回もマスク追加・学習・新しいDDRM推論はない。

## 比較PNG

- [境界と紙模様の分離比較](01_boundary_and_texture.png)。原画像、前回、境界のみ、紙模様のみ、両方。
- [同じパッチの重複平均と継ぎ目](02_overlap_ablation.png)。
- [新しい穴の上方の拡大](03_new_hole_closeup.png)、[既存の穴の拡大](04_old_hole_closeup.png)。前回と同じ切り出しを最近傍で拡大。
- [枠・見出し・本文の対照](05_controls.png)、[印影と広い紙色ムラ](06_stamp_and_paper.png)。
- [保存済みnpj・固定v6・今回の位置づけ](07_npj_v6_context.png)。npj/v6列は保存したDDRM比較、最新列は紙色補間とその後処理であり、マスクと復元方式の両方が異なる。元npj全工程の再現ではない。
- [全51PNG ZIP](paper_seam_texture_comparison_images.zip)。概要7枚、全22箇所の分離比較、全22箇所の重複方式比較。

## 境界の紙色

保存された平滑境界場を、補間内部の最初の8px程度で直接境界場に近づける。最初の内側画素では直接境界を使い、内部ほど平滑境界に戻す。未編集の損傷画素やDDRMの模様を混ぜる操作ではない。参照が欠ける外周へ新たな紙色条件を作ることもしていない。

| ROI | 固定適用画素 | 前回の境界Lab距離中央値 | 境界補正後 |
| --- | ---: | ---: | ---: |
| `brown_false_red_new` | 224 | 1.399 | 0.547 |
| `hole_edge_new` | 9,394 | 3.832 | 0.983 |
| `new_margin_hole` | 14,038 | 3.172 | 1.043 |
| `new_brown_stamp` | 555 | 2.916 | 0.760 |

同じ参照境界対を使った診断値で、境界補正が目的とする量そのもの。独立した自然さの点数・損傷除去精度ではない。紙模様は最初の内側画素では重み0なので、模様のみの境界値は前回と同じ、両方の境界値は境界のみと同じになる。

PNGでは境界の急な紙色差が和らぐ一方、既存の穴の輪郭や新しい穴の薄い茶色い残りは見える。保存マスク外の画素は原画像のままなので、そこに残る縁を今回の後処理で消したとは扱わない。補間内部と周囲の大きな色ムラも一致していない。

## 紙模様と重複平均の比較

保存された紙面参照に24×24pxの全画素が入るパッチだけを候補にする。8px間隔で候補を取り、最大512個に制限した。参照だけを使った正規化Gaussian（σ6px）との差から明度Lの模様を取り出し、各パッチの平均を引き、±3Lに制限する。色成分a/bの模様、乱数模様、DDRM由来の模様は使わない。

24pxパッチを16px間隔で配置し、8pxの重複部分は明度差の小さい継ぎ目でつなぐ。同じ選択パッチ・同じ配置を重複平均でも合成して比較する。模様の強さと境界から5pxまでのフェザーも同じ。**今の平均条件と継ぎ目条件だけが、重複処理の比較として対応している。** 過去の模様試作とは基準画像・マスク・参照・低周波条件が異なるので、継ぎ目だけによる改善とは比較しない。

| ROI | 前回の内部高周波量 | 同じパッチ・重複平均 | 同じパッチ・継ぎ目 | 境界＋継ぎ目 |
| --- | ---: | ---: | ---: | ---: |
| `hole_edge_new` | 0.0382 | 0.4219 | 0.4557 | 0.4572 |
| `new_margin_hole` | 0.0290 | 0.0674 | 0.0712 | 0.0957 |

高周波量は元解像度の出力Lからσ1.2pxのGaussianを引いた標準偏差。固定適用範囲の境界から6px以上離れた画素だけを数える。模様の振幅を観測する量であり、大きいほど自然・正確とは限らない。

既存の穴では粒のある紙面に近づく変化を確認できた。平均処理でも模様は戻り、継ぎ目との差は小さい。新しい穴は参照にできる紙面自体が滑らかで、模様を加えても見た目の変化は小さく、平坦な補修跡が残る。したがって「前回の失敗は重複平均だけが原因」「継ぎ目だけで解決」とは結論しない。

穴2例の固定適用範囲ではL≤80の画素数は全条件で0を維持した。灰色の生成模様を再導入した状態ではないが、これも暗さの診断であり、真正な紙模様を復元した証明ではない。印影の紙面は元から灰色なので、555画素中L≤80の数を穴と同じ品質尺度にはしない。印影の以前からの点の消失は今回も残る。利用者が許容した変化を新しいマスク誤編集として数えない。

広い上端の茶色い紙面は今回も224画素だけ適用。大部分に保存DDRMが残り、全体のシミを解決していない。ほか18箇所は全条件で前回画像と完全一致した。この対象外の対照を、未知資料の文字保存精度の証明にしない。

## 全22箇所の適用範囲

| ROI | 固定適用画素 | 参照パッチ候補 |
| --- | ---: | ---: |
| [濃い茶色の斑点を墨と判定](../paper_seam_texture_pilot_20261001/comparisons/dense_stain.png) · `dense_stain` | 0 | 0 |
| [水染みの中の薄い本文の欠損](../paper_seam_texture_pilot_20261001/comparisons/thin_black_water.png) · `thin_black_water` | 0 | 0 |
| [茶色い斑点を朱インクと判定](../paper_seam_texture_pilot_20261001/comparisons/brown_as_red.png) · `brown_as_red` | 0 | 0 |
| [薄い黒筆画のループが編集候補に入る](../paper_seam_texture_pilot_20261001/comparisons/thin_black_loop.png) · `thin_black_loop` | 0 | 0 |
| [対照：本文と薄い青灰色の線](../paper_seam_texture_pilot_20261001/comparisons/thin_black_text.png) · `thin_black_text` | 0 | 0 |
| [赤褐色の長い筆画が分断される](../paper_seam_texture_pilot_20261001/comparisons/warm_stroke.png) · `warm_stroke` | 0 | 0 |
| [対照：比較的濃い黒本文](../paper_seam_texture_pilot_20261001/comparisons/control_black.png) · `control_black` | 0 | 0 |
| [枠線・本文に接する濃い茶色の汚れ](../paper_seam_texture_pilot_20261001/comparisons/dense_stain_new.png) · `dense_stain_new` | 0 | 0 |
| [上端の茶色い紙面を朱インクと判定](../paper_seam_texture_pilot_20261001/comparisons/brown_false_red_new.png) · `brown_false_red_new` | 224 | 100 |
| [虫食い状の穴・裂けの縁を墨と判定](../paper_seam_texture_pilot_20261001/comparisons/hole_edge_new.png) · `hole_edge_new` | 9,394 | 389 |
| [対照：穴のあるページの通常本文](../paper_seam_texture_pilot_20261001/comparisons/control_text_new.png) · `control_text_new` | 0 | 0 |
| [文字のすぐ上の濃い茶色い斑点](../paper_seam_texture_pilot_20261001/comparisons/brown_near_text.png) · `brown_near_text` | 0 | 0 |
| [対照：同じページの濃い本文](../paper_seam_texture_pilot_20261001/comparisons/dark_text_control.png) · `dark_text_control` | 0 | 0 |
| [別ページの細い筆画の分断](../paper_seam_texture_pilot_20261001/comparisons/faint_left_text.png) · `faint_left_text` | 0 | 0 |
| [別冊の穴・裂けの濃い縁](../paper_seam_texture_pilot_20261001/comparisons/damage_control.png) · `damage_control` | 0 | 0 |
| [新しい冊の上余白の穴・裂け](../paper_seam_texture_pilot_20261001/comparisons/new_margin_hole.png) · `new_margin_hole` | 14,038 | 341 |
| [対照：穴の下の枠と本文](../paper_seam_texture_pilot_20261001/comparisons/new_frame_text.png) · `new_frame_text` | 0 | 0 |
| [上余白の細い曲線：原因未確定](../paper_seam_texture_pilot_20261001/comparisons/new_uncertain_trace.png) · `new_uncertain_trace` | 0 | 0 |
| [対照：本文外の茶色い印影](../paper_seam_texture_pilot_20261001/comparisons/new_brown_stamp.png) · `new_brown_stamp` | 555 | 512 |
| [対照：余白内の見出しと囲み](../paper_seam_texture_pilot_20261001/comparisons/new_header_ink.png) · `new_header_ink` | 0 | 0 |
| [対照：白黒資料の上余白](../paper_seam_texture_pilot_20261001/comparisons/new_blank_paper.png) · `new_blank_paper` | 0 | 0 |
| [対照：白黒資料の本文](../paper_seam_texture_pilot_20261001/comparisons/new_mono_text.png) · `new_mono_text` | 0 | 0 |

## 確認と次の段階

保存入力・マスク・紙色場・ソース・出力のハッシュを確認した。全参照パッチが編集範囲外の固定紙面参照に入り、各継ぎ目画素が記録された配置パッチの観測明度模様と一致することを、継ぎ目処理を再実行せず座標と画素値から確認した。同じ配置記録から重複平均を別に再構成し、保存場との誤差は1e-6未満だった。小さな継ぎ目の最小経路は全列挙の最小コストとも一致した。

4適用例で編集対象の原画素を任意の色に変えても出力場が完全一致した。全22例の元解像度で、適用範囲外は前回画像、編集範囲外は原画素と一致した。境界対を別方向の列挙で再計算し、明度・高周波・変更画素数とZIP51PNGのハッシュも確認した。[確認JSON](verification.json)。

**境界補正と観測紙模様を組み合わせた条件を、次の開発比較の候補として保存する。** 本番採用や新手法の確立とはしない。繰り返し調整した22箇所であり、画素正解と未使用冊の独立評価はない。参照条件は文字がないことの保証でも、失われた歴史的な紙模様の真正な復元でもない。

次は塗り跡の残る新しい穴で、固定マスク外の薄い縁と、内部・周囲の低周波紙色ムラを分けて測る。今回の候補を保存したまま、輪郭の回収を変える比較と、紙色だけを変える比較を分離する。全資料への展開より先に、未使用冊の文字のない損傷例と本文対照で同じ固定条件を確かめる。イラストのページは引き続き対象外。

[固定条件](../paper_seam_texture_pilot_20261001/manifest.json)、[全診断](../paper_seam_texture_pilot_20261001/diagnostics.json)、[原寸出力・参照元座標](../paper_seam_texture_pilot_20261001/native)、[実装](../../scripts/benchmark/probe_paper_seam_texture.py)、[確認・梱包](../../scripts/benchmark/package_paper_seam_texture_review.py)、[前回の紙色補間](../known_paper_harmonic_fill_review_20261001/report.md)。
