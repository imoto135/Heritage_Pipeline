# 残る輪郭と緩やかな紙色ムラを分けて比較する

2026-10-01。**輪郭回収の条件では、虫食い2例の薄い縁が減った。固定マスクで紙色だけを変える条件の効果は小さく、輪郭や塗り跡は残る。輪郭回収は印影対照にも110画素の候補を追加したため、全資料へ採用せず開発候補として保存する。**

前回の「境界＋観測紙模様」を基準に、同じ14ページ・11冊・22箇所で比較した。輪郭マスクだけを変える条件と、固定マスクの紙色だけを変える2条件（σ24／σ48）を分離し、両方を混ぜた条件は作らなかった。DDRMの追加推論・再学習・乱数模様はない。

## 比較PNG

- [輪郭と紙色の分離比較](01_rim_and_paper_color.png)。原画像、前回、輪郭のみ、紙色σ24、紙色σ48。
- [回収・保護・色差不足のマスク診断](02_rim_masks.png)。黄＝追加、紫＝保護、灰＝色差不足、青線＝前回の編集。
- [新しい穴の拡大](03_new_hole_closeup.png)、[既存の穴の拡大](04_old_hole_closeup.png)。前回と同じ切り出し・最近傍拡大。
- [枠・見出し・本文](05_controls.png)、[印影と広い紙色ムラ](06_stamp_and_paper.png)、[印影の追加箇所の拡大](08_stamp_closeup.png)。
- [保存済みnpj・固定v6との位置づけ](07_npj_v6_context.png)。保存済みDDRM列と最新の紙色補間列ではマスクと復元方式の両方が違う。元npj全工程の再現ではない。
- [全66PNG ZIP](paper_rim_lowfrequency_comparison_images.zip)。概要8枚、全22箇所の復元比較、全22箇所のマスク診断、全14ページのマスク比較。全ページ画像はマスクの診断で、復元出力は固定ROIだけ。

## 輪郭を回収する条件

保存された上余白の受理済み暗色成分と上方の全成分回収コアを種にする。各コアの6px近傍だけで、元の紙色推定との差Lが2以上の画素を候補に追加する。以前はコアの3px近傍・色差4以上だった。本文の枠から離した既存の安全領域、長い横線の保護、朱・暖色接続保護を維持し、元の編集マスクはすべて残す。ROI名や「穴」の意味ラベルで種を作っていない。

これは距離と色差を同時に緩めた**輪郭回収の1条件**であり、どちらの変更だけが効いたかの比較ではない。6px近傍と色差2未満の部分も診断PNGへ残した。

| 全ページの変更 | 追加候補 | 近傍内で保護 | 近傍内で色差2未満 |
| --- | ---: | ---: | ---: |
| [200021712_00082_1](whole_page_masks/200021712_00082_1.png) | 4,266 | 0 | 5,340 |
| [100249416_00024_1](whole_page_masks/100249416_00024_1.png) | 2,158 | 192 | 3,375 |
| [brsk003_003](whole_page_masks/brsk003_003.png) | 110 | 0 | 386 |

ほか11ページの全編集マスクは前回と一致した。既存の穴のページでは全ページ4,266画素の候補に対し固定ROIでは1,387画素。ROI外の候補も全ページPNGに示すが、そこでの復元品質は今回評価していない。

追加した候補を紙色補間へ渡し、以前と同じ4近傍の調和補間、直接／平滑境界の混合、観測明度パッチの継ぎ目合成を適用する。参照は以前の紙面参照から新しい編集画素を除くので、補間場と模様候補も変わる。**輪郭列はマスク変更に伴う復元全体の比較で、既存内部の紙色・パッチ配置まで固定した比較ではない。** 新しいマスクでDDRMを推論した結果でもない。参照不足の小成分は前回出力を保持する。

| ROI | 追加候補 | 追加して補間 | 追加候補だが未補間 | 旧編集内で新たに補間 |
| --- | ---: | ---: | ---: | ---: |
| `hole_edge_new` | 1,387 | 1,335 | 52 | 171 |
| `new_margin_hole` | 2,158 | 2,150 | 8 | 13 |
| `new_brown_stamp` | 110 | 107 | 3 | 0 |

追加候補数を検出精度や損傷の正解数と数えない。既存の穴は1,335画素、新しい穴は2,150画素を追加補間し、薄い茶色い輪郭が減った。塗り跡は残り、新しい穴には保護された192画素と、近傍内でも色差が小さい3,375画素がある。これらがすべて損傷とは確定していないので、残存損傷の正解数とも数えない。

印影対照では以前からの点の消失に加えて110画素が候補となり、107画素を追加補間した。点の周辺への新しい変更であり、以前の変化についての利用者の許容を、今回の追加の正解判定として使わない。大きな印影の形は概観で維持されるが、位置・色・太さだけで文字なしを認定できる状態ではない。この例を比較から除外しない。

## マスクを固定して紙色だけを変える条件

前回の編集マスク、補間範囲、紙面参照、紙模様の場と配置をそのまま使う。参照紙面だけを正規化Gaussianでσ24px／48pxに平滑化し、緩やかな紙色場を作る。その境界値の調和補間を引いた差を、前回の補間場へ足す。これで境界に整合する範囲で紙色の曲がりを試す。

補正は最初の内側画素では0、距離9px以上で全強度とする。参照重み0.05未満では適用せず、Lは±3、a/bは±1.5以内に制限する。元の損傷画素・DDRM模様を補正に取り込まず、紙模様を選び直すこともない。

| ROI | σ24の補正最大絶対値 L/a/b | σ48の補正最大絶対値 L/a/b |
| --- | --- | --- |
| `hole_edge_new` | 0.756/0.311/0.425 | 0.396/0.189/0.079 |
| `new_margin_hole` | 0.573/0.360/1.072 | 0.466/0.195/1.091 |
| `new_brown_stamp` | 0.007/0.010/0.014 | 0.011/0.007/0.007 |

今回の紙色補正は小さく、PNGでの改善も控えめだった。輪郭自体は変わらず、σ24／48のどちらでも平坦な塗り跡と薄い縁が残る。紙色補正を追加採用せず、前回の条件を比較基準に残す。参照から推定した紙色場と近づくことを、失われた紙色の真正な復元や自然さの独立評価と扱わない。

穴2例の旧補間範囲では、全条件でL≤80の画素は0を維持した。固定の追加候補範囲では、前回のL≤80は既存の穴3画素、新しい穴38画素、輪郭条件では両方0。これも明度の観測であり、薄い縁の除去精度ではない。印影の紙面は元から灰色なので同じしきい値を品質評価に使わない。

全18箇所は今回の3条件とも前回と完全一致した。広い上端の茶色い紙面は今回も224画素だけ紙色補間する状態で、大部分のシミは未解決。本文に重なる汚れ、細い未検出の裂け、イラストの問題も今回の成功に含めない。イラストは対象外。

## 全22箇所

| ROI | 輪郭追加候補 | 追加補間 | 固定の旧補間画素 |
| --- | ---: | ---: | ---: |
| [濃い茶色の斑点を墨と判定](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/dense_stain.png) · `dense_stain` | 0 | 0 | 0 |
| [水染みの中の薄い本文の欠損](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/thin_black_water.png) · `thin_black_water` | 0 | 0 | 0 |
| [茶色い斑点を朱インクと判定](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/brown_as_red.png) · `brown_as_red` | 0 | 0 | 0 |
| [薄い黒筆画のループが編集候補に入る](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/thin_black_loop.png) · `thin_black_loop` | 0 | 0 | 0 |
| [対照：本文と薄い青灰色の線](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/thin_black_text.png) · `thin_black_text` | 0 | 0 | 0 |
| [赤褐色の長い筆画が分断される](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/warm_stroke.png) · `warm_stroke` | 0 | 0 | 0 |
| [対照：比較的濃い黒本文](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/control_black.png) · `control_black` | 0 | 0 | 0 |
| [枠線・本文に接する濃い茶色の汚れ](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/dense_stain_new.png) · `dense_stain_new` | 0 | 0 | 0 |
| [上端の茶色い紙面を朱インクと判定](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/brown_false_red_new.png) · `brown_false_red_new` | 0 | 0 | 224 |
| [虫食い状の穴・裂けの縁を墨と判定](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/hole_edge_new.png) · `hole_edge_new` | 1,387 | 1,335 | 9,394 |
| [対照：穴のあるページの通常本文](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/control_text_new.png) · `control_text_new` | 0 | 0 | 0 |
| [文字のすぐ上の濃い茶色い斑点](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/brown_near_text.png) · `brown_near_text` | 0 | 0 | 0 |
| [対照：同じページの濃い本文](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/dark_text_control.png) · `dark_text_control` | 0 | 0 | 0 |
| [別ページの細い筆画の分断](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/faint_left_text.png) · `faint_left_text` | 0 | 0 | 0 |
| [別冊の穴・裂けの濃い縁](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/damage_control.png) · `damage_control` | 0 | 0 | 0 |
| [新しい冊の上余白の穴・裂け](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/new_margin_hole.png) · `new_margin_hole` | 2,158 | 2,150 | 14,038 |
| [対照：穴の下の枠と本文](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/new_frame_text.png) · `new_frame_text` | 0 | 0 | 0 |
| [上余白の細い曲線：原因未確定](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/new_uncertain_trace.png) · `new_uncertain_trace` | 0 | 0 | 0 |
| [対照：本文外の茶色い印影](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/new_brown_stamp.png) · `new_brown_stamp` | 110 | 107 | 555 |
| [対照：余白内の見出しと囲み](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/new_header_ink.png) · `new_header_ink` | 0 | 0 | 0 |
| [対照：白黒資料の上余白](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/new_blank_paper.png) · `new_blank_paper` | 0 | 0 | 0 |
| [対照：白黒資料の本文](../paper_rim_lowfrequency_pilot_v2_20261001/comparisons/new_mono_text.png) · `new_mono_text` | 0 | 0 | 0 |

## 確認と次の段階

全ページの入力・ソース・保存マスク・前回出力のハッシュと、元マスクの包含、保護・領域条件、追加と除外の内訳を確認した。紙色条件では固定マスクと参照が前回と同一で、固定模様を使った合成式と元解像度の出力が一致した。参照だけのGaussian場を別に再計算し、その境界の調和場に4近傍の差を直接足して方程式残差を確認した。定数紙面に紙色補正が生じないことも確認した。

輪郭条件では新しい編集画素が参照パッチに入らないこと、元画素座標と記録された配置パッチの模様値の一致、同じ配置からの重複平均の再構成を確認した。補間場を再計算したうえで残差を別の近傍列挙で確認し、記録済み模様を足した場が保存場と完全一致した。補間不足の画素は消したと数えず、上表へ記録した。[確認JSON](verification.json)。

初回は、紙色のみの条件にも輪郭用の拡張マスクで損傷画素の置換テストをかけており、紙色条件では正当な参照画素まで変えたため検証で停止した。初回のソース・途中出力を残し、各条件のマスクで検証するv2へ修正した。復元式・パラメータは変更せず、初回に保存された全ネイティブファイル251件はv2とハッシュ同一だった。4適用例で、それぞれの編集画素を任意色に変えても補正・復元場は完全一致した。

全22例の編集範囲外は原画素、各条件の適用範囲外は前回画像と一致した。境界対と明度を別に計算し、ZIP66PNGの内容ハッシュも確認した。今回の全例は調整を重ねた開発例で、画素GT・未使用冊の精度はない。

**次の優先は、新しい紙色補正を重ねることより、輪郭回収の条件を未使用冊で確かめること。** 輪郭回収の候補と前回基準を保存し、原画像から文字のない損傷と本文・印影の対照を先に選ぶ。固定した6px・色差2の条件をそのまま適用し、本文や印影の追加編集も同じPNGで確認する。紙色σ24／48は大きな改善を示していないため追加採用しない。位置・参照条件の文字なし判定や、新手法の確立はまだ実証していない。

[固定条件](../paper_rim_lowfrequency_pilot_v2_20261001/manifest.json)、[全診断](../paper_rim_lowfrequency_pilot_v2_20261001/diagnostics.json)、[原寸出力](../paper_rim_lowfrequency_pilot_v2_20261001/native)、[実装v2](../../scripts/benchmark/probe_paper_rim_lowfrequency_v2.py)、[初回ソース](../../scripts/benchmark/probe_paper_rim_lowfrequency.py)、[確認・梱包](../../scripts/benchmark/package_paper_rim_lowfrequency_review.py)、[前回の境界・紙模様](../paper_seam_texture_review_20261001/report.md)。
