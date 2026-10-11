# 同じDDRMにnpjとv6の編集マスクを渡した比較

[汚れの比較PNG](01_stain_ddrm.png) / [筆画の比較PNG](02_stroke_ddrm.png) / [残る問題の比較PNG](03_remaining_ddrm.png) / [本文対照PNG](04_controls_ddrm.png) / [全19枚ZIP](npj_v6_ddrm_comparison_images.zip)

今回の画像は実際にDDRMで推論した結果。前回のTeleaによる局所補間とは別の実験。前回から固定している11ページ・15箇所をすべて使い、モデル・入力・解像度・設定・合成を固定し、npjとv6の編集候補マスクだけを変えた。イラストページは評価に含めていない。

## マスクをどう接続したか

保存済みnpj v1の04_degradation（白=編集候補）と固定v6のeditを、DDRM inpaintingの欠損領域として使用した。欠損領域以外は同じ原画像を観測値とし、生成後に元の画素へ戻す。v6のprotect・hold・redは欠損領域に含めない。赤・保留の別途の色調変更は行っていない。両方ともTeleaの事前穴埋め・背景参照は使っていない。

既存npjパイプラインの黒文字マスクは文字の再生成を指示するため、v6 editをそこへ直接渡していない。この実験は**編集候補を欠損に対応させたDDRM比較で、元のnpj全工程や黒文字の再生成経路の再現ではない**。モデルへ渡す値も学習時に合わせてRGB [-1,1]としている。旧コードの文字用xt抑制は両方とも無効。したがって、この結果から元のnpjパイプライン全体よりv6が優れるとは結論しない。

## 固定条件

- 重み：model_charのckpt_epoch0350.pth。同じUNet（基底幅64、残差ブロック2、attention 16/8）をstrict load。
- DDRM：リポジトリのefficient_generalized_steps。1000段線形beta=0.0001〜0.02から20段（0,50,…,950）、sigma0=0.05、etaB=1、etaA=etaC=0.5。
- 原寸、矩形周辺64px、128pxパッチ、重なり32px、共通の窓重みで合成。端は共通の反射パディング、パディング領域は既知。マスク膨張・解像度変更・新しい閾値調整なし。
- float32、同じGPU、seed=999からページ・座標ごとのseedを固定。観測ノイズ・初期ノイズ・各段のノイズは同じ画像座標のGaussian場から生成し、各マスクのSVD座標へ対応させた。欠損数が異なるため、単なる乱数seedのリセットだけに依存していない。
- 編集候補が空のパッチは原画像を維持。DDRMの生出力と、編集範囲だけを合成した最終出力を両方保存した。

## 今回の所見

薄い輪状筆画では、npj側で筆画の消失・変形が起こり、v6側では編集を回避して原形を保つ。赤褐色の長い筆画にも同じ傾向がある。ただし、別ページの細い本文では差が小さい例もあり、一律の改善率にはしない。

本文・枠につながる茶色い汚れ、本文に接する茶色い部分、穴・裂けの濃い縁は両方で残る。孤立した汚れはv6で編集できても明るい塗り跡・境界が残る。本文対照もnpj側では筆画周縁に生成による変形が見える一方、v6側はほぼ原画像を維持する。この保守性は紙面の影・ムラを残すことにもつながる。

すべて補助者の暫定目視で、画素GT・正解復元画像による精度ではない。v6一式と旧npjの違いには土台の分類処理の変更も含まれる。前回の分類比較でv6固有の追加暖色保留があったのは15箇所中warm_strokeだけであり、他の結果をv6固有の追加保留の成果とは数えない。

## 全15箇所の比較

変更画素数は診断値。精度・汚れ除去率ではない。

| 個別比較PNG | npj変更画素 | v6変更画素 | 暫定目視 |
| --- | ---: | ---: | --- |
| [dense_stain](comparisons/dense_stain.png) | 14,005 | 3,356 | v6は中心を編集できるが、明るい塗り跡と境界が残る。 |
| [thin_black_water](comparisons/thin_black_water.png) | 114,126 | 10 | v6は薄い本文を維持する一方、水染みも残る。 |
| [brown_as_red](comparisons/brown_as_red.png) | 5,246 | 3,661 | v6は茶色い中心を編集するが、明るい塗り跡が残る。 |
| [thin_black_loop](comparisons/thin_black_loop.png) | 6,154 | 0 | npj側で筆画が消失・変形。v6側は原画像と完全一致。 |
| [thin_black_text](comparisons/thin_black_text.png) | 135,624 | 2 | 本文対照：v6は原形をほぼ維持。npj側に生成による変形。 |
| [warm_stroke](comparisons/warm_stroke.png) | 62,850 | 386 | v6は赤褐色の長い筆画を維持。npj側に欠落・灰色の生成。 |
| [control_black](comparisons/control_black.png) | 45,678 | 4 | 本文対照：v6はほぼ維持。npj側は周縁・内側を変形。 |
| [dense_stain_new](comparisons/dense_stain_new.png) | 27,101 | 39 | 本文・枠とつながる汚れは両方で残る。 |
| [brown_false_red_new](comparisons/brown_false_red_new.png) | 4,266 | 15,576 | v6は一部を編集するが、広い薄茶色と塗り跡が残る。 |
| [hole_edge_new](comparisons/hole_edge_new.png) | 3,681 | 644 | 穴の濃い縁は両方で残る。npj側には細い生成跡。 |
| [control_text_new](comparisons/control_text_new.png) | 37,303 | 0 | 本文対照：v6は原画像と完全一致。npj側は文字周囲を変形。 |
| [brown_near_text](comparisons/brown_near_text.png) | 25,613 | 1,357 | 本文に接する茶色い中心は両方で残る。 |
| [dark_text_control](comparisons/dark_text_control.png) | 27,131 | 43 | 本文対照：v6はほぼ維持。npj側には文字内側の生成跡。 |
| [faint_left_text](comparisons/faint_left_text.png) | 20,149 | 66 | この別ページの差は小さい。v6は薄い筆画をほぼ維持。 |
| [damage_control](comparisons/damage_control.png) | 13,525 | 116 | 別冊の穴・裂けの濃い縁は両方で残る。 |

## 判断と次の検証

この条件では、編集候補に実筆画を入れるとDDRMでも筆画の形が変わり、v6で編集を避ける意義がある。一方、保護を広げたままでは汚れ・穴も残る。編集候補に戻せた汚れにも境界や塗り跡が生じるため、分類と復元の両方に課題がある。v6の全面置換や新手法の有効性はまだ示していない。

次は本文・汚れ・損傷が接する固定例の分離を優先し、薄い輪状筆画・払い・赤褐色筆画・通常本文を対照として残す。復元側の調整を試す場合はこの20段・同じマスクの結果を基準として保存し、マスクを同時に変更しない。これらの例で調整したら開発例と明記し、別冊の未使用ページで確認する。

## 保存と検証

native/<id>に原画像、DDRM生出力、最終出力、編集マスク、変更マスクを保存。入力・重み・実装のSHA256、マスクの原寸一致、編集範囲外の不変、同じマスク・同じ乱数での完全再現、SVDの画素対応、PNGとZIPを確認した。[実行条件](manifest.json)、[実行検証](verification.json)、[成果物検証](final_verification.json)、[個別所見](visual_review.json)。

実行：`/home/imoto/miniconda3/envs/diffusion/bin/python -B scripts/benchmark/compare_npj_v6_ddrm_edits.py`

比較画像のまとめ：`/home/imoto/miniconda3/envs/diffusion/bin/python -B scripts/benchmark/package_npj_v6_ddrm_comparison.py`
