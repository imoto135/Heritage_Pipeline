# 紙模様の連続性とマスク外の残存損傷

2026-10-01。**未編集紙面の模様を移す試作では、残る違和感を十分に減らせなかった。新しい穴では補修部分がさらに平滑になる傾向があり、今回の方式へ置き換えず、前回の隣接境界補正を基準として維持する。** 濃く残る損傷がマスク外であることも比較PNGで分離した。

今回も14ページ・11冊・22箇所の開発例を全て扱い、同じ編集マスクと保存済みDDRMを使用した。新しいDDRM推論・学習・マスクの拡張はない。比較の基準は[前回](../paper_blending_review_20261001/report.md)の**境界のみ**の補正である。

## 比較PNG

- [虫食い2例の要素比較](01_paper_pattern.png)：原画像、前回の境界補正、紙色のつながり、紙模様のみ、両方。
- [編集境界と残存損傷](02_remaining_mask.png)：赤線は現在の編集境界。マスク外の損傷は変わらない。
- [既存の穴の拡大](03_old_hole_closeup.png)、[新しい穴の拡大](04_new_hole_closeup.png)。切り出しは補正範囲の外接矩形＋12pxで全条件共通。
- [本文・見出し・枠の対照](05_controls.png)、[紙面上端と印影](06_paper_and_stamp.png)、[参照紙面のパッチ](07_known_paper_donors.png)。
- [全51枚のPNG ZIP](paper_pattern_comparison_images.zip)：概要7、全ROIの要素比較22、編集境界の診断22。

## 分けた条件

補正対象は前回の対象マスクと完全に同じ4箇所で、残る18箇所は変更しない。元のマスク外にある画素も変更しない。紙色のつながりは、未編集の参照紙面からσ6pxの正規化低周波Labを求め、境界条件として編集内部へ補間した。4近傍の離散ラプラシアンと弱い既存生成色への拘束（1/48²）を用い、既存生成画素のσ3pxからの細かい差分は残す。

紙模様のみは、保存済みの紙面参照マスク内に全画素が入る24×24pxパッチを8px間隔で集める。候補は最大512、4個未満なら補正しない。元のインク保護・編集対象の画素を候補に含めない。原画像の参照紙面だけで正規化したσ3px低周波との差をパッチの模様として用い、8px重複で敷き詰める。重複部分のLab差と再利用回数で候補を選び、重複部分を加重平均する。生成された低周波の紙色は維持する。両方は低周波補間と、この観測紙面の模様を合わせる。

全条件とも編集内部4pxで前回出力へ弱めて接続する。元の損傷画素を混ぜ戻さず、新しい乱数の粒状模様も生成しない。ただしパッチの再配置なので、歴史資料の失われた紙模様を真に復元したという意味ではない。印影の消失した点も今回回復していない。

## 診断値と目視

境界値は前回と同じ境界対でのLab距離中央値。ΔE2000・GT精度・目視の品質点ではない。

| ROI | 前回の境界補正 | 紙色のつながり | 紙模様のみ | 両方 |
| --- | ---: | ---: | ---: | ---: |
| `brown_false_red_new` | 1.326 | 1.276 | 1.261 | 1.375 |
| `hole_edge_new` | 1.870 | 2.101 | 1.920 | 1.994 |
| `new_margin_hole` | 2.039 | 2.028 | 2.009 | 1.961 |
| `new_brown_stamp` | 1.818 | 2.062 | 1.793 | 2.026 |

既存の穴では新しい3条件の境界値はいずれも前回より増えた。新しい穴は両方で2.039→1.961と小さくなるが、平坦な領域と薄い輪郭が目視で残る。値の改善だけで自然さの向上と判断しない。印影と上端紙面でも両方が常に良いわけではない。

編集内部6px以上のL高周波（σ1.2px）標準偏差も比較した。これは模様の量の診断で、値の大小自体を成功の判定に使わない。

| ROI | 前回の境界補正 | 紙色のつながり | 紙模様のみ | 両方 |
| --- | ---: | ---: | ---: | ---: |
| `brown_false_red_new` | — | — | — | — |
| `hole_edge_new` | 0.616 | 0.607 | 0.379 | 0.378 |
| `new_margin_hole` | 0.404 | 0.407 | 0.082 | 0.054 |
| `new_brown_stamp` | 2.140 | 2.120 | 0.553 | 0.541 |

新しい穴では模様を入れ替えた側の高周波量が大きく下がった。参照が明るく平滑な紙面に偏ること、パッチの重複平均と8bit量子化が模様を弱めることが考えられるが、各要因の寄与は分離していない。模様が少ない／多いだけでは自然さを説明できず、今回の移植を改善と確定しない。

候補パッチの原画像L高周波標準偏差（σ1.2px、パッチごとの中央値と10〜90%範囲）は次の通り。こちらは原パッチごとに平滑化した診断で、編集内部と領域サイズ・境界処理が違い、直接の品質比較ではない。

| ROI | 候補パッチ数 | 中央値 | 10〜90% |
| --- | ---: | ---: | ---: |
| `brown_false_red_new` | 98 | 0.418 | 0.298〜0.591 |
| `hole_edge_new` | 389 | 0.520 | 0.404〜0.638 |
| `new_margin_hole` | 341 | 0.085 | 0.019〜0.149 |
| `new_brown_stamp` | 512 | 0.792 | 0.653〜1.006 |

## 全22箇所

| ROI | 補正対象画素 | 参照候補パッチ数 |
| --- | ---: | ---: |
| [濃い茶色の斑点を墨と判定](../paper_pattern_continuity_pilot_20261001/comparisons/dense_stain.png) · `dense_stain` | 0 | 0 |
| [水染みの中の薄い本文の欠損](../paper_pattern_continuity_pilot_20261001/comparisons/thin_black_water.png) · `thin_black_water` | 0 | 0 |
| [茶色い斑点を朱インクと判定](../paper_pattern_continuity_pilot_20261001/comparisons/brown_as_red.png) · `brown_as_red` | 0 | 0 |
| [薄い黒筆画のループが編集候補に入る](../paper_pattern_continuity_pilot_20261001/comparisons/thin_black_loop.png) · `thin_black_loop` | 0 | 0 |
| [対照：本文と薄い青灰色の線](../paper_pattern_continuity_pilot_20261001/comparisons/thin_black_text.png) · `thin_black_text` | 0 | 0 |
| [赤褐色の長い筆画が分断される](../paper_pattern_continuity_pilot_20261001/comparisons/warm_stroke.png) · `warm_stroke` | 0 | 0 |
| [対照：比較的濃い黒本文](../paper_pattern_continuity_pilot_20261001/comparisons/control_black.png) · `control_black` | 0 | 0 |
| [枠線・本文に接する濃い茶色の汚れ](../paper_pattern_continuity_pilot_20261001/comparisons/dense_stain_new.png) · `dense_stain_new` | 0 | 0 |
| [上端の茶色い紙面を朱インクと判定](../paper_pattern_continuity_pilot_20261001/comparisons/brown_false_red_new.png) · `brown_false_red_new` | 364 | 98 |
| [虫食い状の穴・裂けの縁を墨と判定](../paper_pattern_continuity_pilot_20261001/comparisons/hole_edge_new.png) · `hole_edge_new` | 9,460 | 389 |
| [対照：穴のあるページの通常本文](../paper_pattern_continuity_pilot_20261001/comparisons/control_text_new.png) · `control_text_new` | 0 | 0 |
| [文字のすぐ上の濃い茶色い斑点](../paper_pattern_continuity_pilot_20261001/comparisons/brown_near_text.png) · `brown_near_text` | 0 | 0 |
| [対照：同じページの濃い本文](../paper_pattern_continuity_pilot_20261001/comparisons/dark_text_control.png) · `dark_text_control` | 0 | 0 |
| [別ページの細い筆画の分断](../paper_pattern_continuity_pilot_20261001/comparisons/faint_left_text.png) · `faint_left_text` | 0 | 0 |
| [別冊の穴・裂けの濃い縁](../paper_pattern_continuity_pilot_20261001/comparisons/damage_control.png) · `damage_control` | 0 | 0 |
| [新しい冊の上余白の穴・裂け](../paper_pattern_continuity_pilot_20261001/comparisons/new_margin_hole.png) · `new_margin_hole` | 7,312 | 341 |
| [対照：穴の下の枠と本文](../paper_pattern_continuity_pilot_20261001/comparisons/new_frame_text.png) · `new_frame_text` | 0 | 0 |
| [上余白の細い曲線：原因未確定](../paper_pattern_continuity_pilot_20261001/comparisons/new_uncertain_trace.png) · `new_uncertain_trace` | 0 | 0 |
| [対照：本文外の茶色い印影](../paper_pattern_continuity_pilot_20261001/comparisons/new_brown_stamp.png) · `new_brown_stamp` | 555 | 512 |
| [対照：余白内の見出しと囲み](../paper_pattern_continuity_pilot_20261001/comparisons/new_header_ink.png) · `new_header_ink` | 0 | 0 |
| [対照：白黒資料の上余白](../paper_pattern_continuity_pilot_20261001/comparisons/new_blank_paper.png) · `new_blank_paper` | 0 | 0 |
| [対照：白黒資料の本文](../paper_pattern_continuity_pilot_20261001/comparisons/new_mono_text.png) · `new_mono_text` | 0 | 0 |

マスク外に残る濃い右上部分は、今回の補正の効き不足とは区別する。既存の穴の薄い輪郭には、マスク外に残る部分と編集内部の塗り跡の両方がある。本文・枠・見出しなど18箇所は補正対象0で前回と完全一致した。条件による対象外であり、未知資料の文字保存精度の実証ではない。利用者が許容した以前の印影の小さな点の変化は、新しい阻害要因として再扱いしない。

## 確認と次の段階

全参照候補の24×24画素が未編集の紙面参照に入ること、保存入力・マスク・DDRM・ソース・PNGのハッシュ、22箇所の補正範囲外／編集範囲外の不変、境界値の別実装での再計算、ZIP内51PNGのハッシュを確認した。参照領域を固定して原画像の編集対象を任意の色に置き換えても、穴2例と印影の全条件の出力は一致した。[確認JSON](verification.json)。参照ゲート自体の正しさを画素GTで評価した確認ではない。

次の優先事項は、模様の条件を追加で調整することより、濃い残存損傷をマスク外に置いている理由を検証すること。特に新しい穴の上余白帯の境界に接する成分の一括除外を、本文からの距離・印影保護と両立して見直す。マスクの変更を先にPNGで確認し、その候補のみを固定DDRM・前回の境界補正へ接続する。今回の模様試作は履歴に保存し、その実験には混ぜない。本文と重なるシミ・欠損文字の復元は未解決のまま保持する。

[試作の固定条件](../paper_pattern_continuity_pilot_20261001/manifest.json)、[全診断値](../paper_pattern_continuity_pilot_20261001/diagnostics.json)、[原寸出力](../paper_pattern_continuity_pilot_20261001/native)、[実装](../../scripts/benchmark/probe_paper_pattern_continuity.py)。
