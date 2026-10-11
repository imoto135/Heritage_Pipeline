# 虫食い状の損傷・シミの前処理試作と同条件DDRM確認

2026-10-01。**前処理で誤保護を減らす手掛かりは得たが、虫食い・シミの解決やv6を超える新手法の確立には至っていない。** 既存の固定11ページ・8冊・15箇所（問題11、本文対照4）を使い、全ページから推定する試作を2段階で実行した。修正版の全15箇所を、前回と同じDDRMで確認した。画像を見て調整した開発例であり、独立した別冊評価でも画素GTによる精度でもない。イラストとして除外済みのページは追加していない。

## 比較画像

- [穴・茶色い部分の分類PNG](01_mask_targets.png)：原画像、保存済みnpj、固定v6、修正版。
- [同じDDRM後のPNG](02_ddrm_targets.png)：穴の縁の残存と新しい文字状の生成。
- [別冊の未解決分類PNG](03_remaining_masks.png)、[筆画のDDRM確認PNG](04_stroke_controls.png)。
- [初回試作の筆画侵食と修正版PNG](05_initial_regression.png)、[文字のない紙面での生成失敗PNG](06_paper_artifact.png)。
- [全比較画像ZIP](defect_context_comparison_images.zip)。全ページ分類11枚、ROI分類15枚、同じDDRM15枚、色のみとの比較15枚、要素を切り分けた3枚、初回失敗1枚、概要6枚。合計66枚。

分類PNGの青＝保護、赤＝朱として保護、黄＝編集候補、紫＝保留。黄の増加は処理対象面積であり、精度や改善率ではない。DDRM PNGには分類色を重ねていない。

## 試作の内容

試作v1では、近傍の明るい紙から推定した紙色との差、茶色い領域の連結・太さ、黒い線状応答を使った。原画像のページ全体と固定v6のマスクだけを推定入力とし、ROI座標・問題ラベルで処理を限定していない。固定v6の編集候補に新しい候補を加え、その部分だけ保護・保留から外す。v6の朱候補と追加暖色保留は保持する。

ただし、赤褐色の実筆画にも黄が入り、`warm_stroke`で5,195画素を追加した。これを失敗例として保存した。色と局所形状だけで汚れと暖色筆画を分離できたとは言えない。

修正版v2では、v6の朱候補・追加暖色保留につながる保護／保留領域全体と2pxの周囲を守り、黒い線状応答の周囲の保護を2pxから4pxへ広げた。アンカーに接続する汚れも残しうる保守的な仮説であり、領域が実インクだという証明ではない。ここでの「文脈」は局所紙色・連結・線状応答を指し、文書の意味や損傷の物理原因を認識したものではない。

[要素別CSV](factor_ablation.csv)ではv1から「接続保護のみ」「墨ガード拡張のみ」「両方」を同じ全ページマスクから作り、効果を分けた。暖色筆画の追加候補はv1 5,195、接続保護のみ 95、墨ガード拡張のみ 5,086、両方 81画素。[暖色の要素別PNG](factor_ablation/warm_stroke.png)、[穴](factor_ablation/hole_edge_new.png)、[文字に接する茶色](factor_ablation/brown_near_text.png)。これらは分類比較で、各アブレーションのDDRMは実行していない。

## 全15箇所の所見

| 箇所・分類PNG | v1追加画素 | v2追加画素 | 分類の所見 | 同条件DDRM後の所見・PNG |
| --- | ---: | ---: | --- | --- |
| [dense_stain](../defect_context_preprocessing_pilot_v2_20261001/comparisons/dense_stain.png) | 103 | 35 | 追加は周縁35画素。中心はv6で既に編集対象。 | [明るい平坦な塗り跡・境界が残る。解決と数えない。](../defect_context_ddrm_pilot_v2_20261001/comparisons/dense_stain.png) |
| [thin_black_water](../defect_context_preprocessing_pilot_v2_20261001/comparisons/thin_black_water.png) | 0 | 0 | ROI内追加0。 | [v6と同一出力。薄い本文を維持するが影・水染みは残る。](../defect_context_ddrm_pilot_v2_20261001/comparisons/thin_black_water.png) |
| [brown_as_red](../defect_context_preprocessing_pilot_v2_20261001/comparisons/brown_as_red.png) | 0 | 0 | ROI内追加0。 | [v6と同一出力。既存の編集境界・色差が残る。](../defect_context_ddrm_pilot_v2_20261001/comparisons/brown_as_red.png) |
| [thin_black_loop](../defect_context_preprocessing_pilot_v2_20261001/comparisons/thin_black_loop.png) | 0 | 0 | ROI内追加0。 | [v6と同じ原画素。npj側の消失・変形を避ける状態を維持。](../defect_context_ddrm_pilot_v2_20261001/comparisons/thin_black_loop.png) |
| [thin_black_text](../defect_context_preprocessing_pilot_v2_20261001/comparisons/thin_black_text.png) | 172 | 63 | 本文対照に63画素追加。完全な無変更ではない。 | [主要筆画は概観で維持。小さな変更の正誤は未判定。](../defect_context_ddrm_pilot_v2_20261001/comparisons/thin_black_text.png) |
| [warm_stroke](../defect_context_preprocessing_pilot_v2_20261001/comparisons/warm_stroke.png) | 5,195 | 81 | v1では実筆画も編集候補化。v2の追加は5,195→81画素。 | [主要な暖色筆画を概観で維持。残る81画素の正誤は未判定。](../defect_context_ddrm_pilot_v2_20261001/comparisons/warm_stroke.png) |
| [control_black](../defect_context_preprocessing_pilot_v2_20261001/comparisons/control_black.png) | 279 | 35 | 本文対照に35画素追加。 | [主要筆画は概観で維持。小さな変更の正誤は未判定。](../defect_context_ddrm_pilot_v2_20261001/comparisons/control_black.png) |
| [dense_stain_new](../defect_context_preprocessing_pilot_v2_20261001/comparisons/dense_stain_new.png) | 0 | 0 | 濃い黄褐色・オリーブ色部分に追加0。色差条件の対象外。 | [v6と同一出力。本文・枠に重なる汚れは残り、未解決。](../defect_context_ddrm_pilot_v2_20261001/comparisons/dense_stain_new.png) |
| [brown_false_red_new](../defect_context_preprocessing_pilot_v2_20261001/comparisons/brown_false_red_new.png) | 13,497 | 12,411 | 紙面上端の茶色い部分に12,411画素追加。 | [新しい暗い線・文字状の断片が発生。広い色ムラも残り、復元成功と数えない。](../defect_context_ddrm_pilot_v2_20261001/comparisons/brown_false_red_new.png) |
| [hole_edge_new](../defect_context_preprocessing_pilot_v2_20261001/comparisons/hole_edge_new.png) | 6,407 | 4,960 | 文字のない穴周辺の濃い部分を4,960画素、編集側に戻す。 | [濃い中心は薄くなるが茶色い縁・不均一な塗り跡が残る。自然な修復は未達。](../defect_context_ddrm_pilot_v2_20261001/comparisons/hole_edge_new.png) |
| [control_text_new](../defect_context_preprocessing_pilot_v2_20261001/comparisons/control_text_new.png) | 287 | 24 | 本文対照に24画素追加。 | [主要筆画は概観で維持。小さな変更の正誤は未判定。](../defect_context_ddrm_pilot_v2_20261001/comparisons/control_text_new.png) |
| [brown_near_text](../defect_context_preprocessing_pilot_v2_20261001/comparisons/brown_near_text.png) | 16,102 | 15,426 | 茶色い部分に15,426画素追加。直下の黒い横画は保護。 | [茶色い中心に原画像で確認できない文字状の線を生成。復元成功と数えない。](../defect_context_ddrm_pilot_v2_20261001/comparisons/brown_near_text.png) |
| [dark_text_control](../defect_context_preprocessing_pilot_v2_20261001/comparisons/dark_text_control.png) | 113 | 35 | 本文対照に35画素追加。 | [主要筆画は概観で維持。小さな変更の正誤は未判定。](../defect_context_ddrm_pilot_v2_20261001/comparisons/dark_text_control.png) |
| [faint_left_text](../defect_context_preprocessing_pilot_v2_20261001/comparisons/faint_left_text.png) | 0 | 0 | ROI内追加0。 | [v6と同一出力。元の薄い本文を維持。](../defect_context_ddrm_pilot_v2_20261001/comparisons/faint_left_text.png) |
| [damage_control](../defect_context_preprocessing_pilot_v2_20261001/comparisons/damage_control.png) | 601 | 310 | 損傷の周縁310画素を追加。濃い中性の縁は保護が残る。 | [大きな損傷の線はほぼ残る。未解決。このIDは対照でなく問題例。](../defect_context_ddrm_pilot_v2_20261001/comparisons/damage_control.png) |

穴の中心、文字の直上の茶色い部分の誤保護は減る。しかし、穴の濃い縁を全部検出できていない。黒い損傷の縁は本物の黒筆画と同様に線状応答が強く、保護されて残る。本文に重なるオリーブ色のシミは今回の茶色色差条件を満たさず、追加0で未解決。物理的な虫食い・シミ・焼損の原因は画像から確定していない。

復元では、茶色い部分を編集対象に戻すと、新しい文字状の線が生成される例がある。元の文字を正しく取り戻したという根拠はなく、成功として数えない。文字用事前分布と紙面修復という接続の適合性、残存する境界・編集範囲の不確かさは別々に検証すべきで、今回だけで生成原因を特定してはいない。

## ページ全体での編集範囲

切り出し以外も自動推定している。特に穴のあるページでは追加が336,882画素、紙面上端が茶色いページでは125,344画素となり、側端の変色等にも広がる。全ページの縮小比較は確認したが、全追加画素の正誤を判定したものではない。本文対照4箇所にも24〜63画素の追加があり、無誤編集とは言えない。DDRMは15箇所の周辺64pxを含む局所実験であり、全ページ復元を完了したわけではない。

| ページ・全体分類PNG | 固定v6編集画素 | v2追加画素 | ページ面積に対する追加割合 |
| --- | ---: | ---: | ---: |
| [hnsd001_011](../defect_context_preprocessing_pilot_v2_20261001/comparisons/hnsd001_011_whole.png) | 16,189 | 36 | 0.00% |
| [200010454_00002_2](../defect_context_preprocessing_pilot_v2_20261001/comparisons/200010454_00002_2_whole.png) | 12,553 | 644 | 0.02% |
| [200003803_00009_2](../defect_context_preprocessing_pilot_v2_20261001/comparisons/200003803_00009_2_whole.png) | 33,360 | 11,088 | 0.16% |
| [200021763_00002_2](../defect_context_preprocessing_pilot_v2_20261001/comparisons/200021763_00002_2_whole.png) | 6,674 | 8,532 | 0.07% |
| [200021869_00010_2](../defect_context_preprocessing_pilot_v2_20261001/comparisons/200021869_00010_2_whole.png) | 11,945 | 6,076 | 0.09% |
| [hnsd007_042](../defect_context_preprocessing_pilot_v2_20261001/comparisons/hnsd007_042_whole.png) | 6,736 | 292 | 0.01% |
| [100249476_00003_1](../defect_context_preprocessing_pilot_v2_20261001/comparisons/100249476_00003_1_whole.png) | 60,484 | 125,344 | 2.33% |
| [200021712_00082_1](../defect_context_preprocessing_pilot_v2_20261001/comparisons/200021712_00082_1_whole.png) | 47,777 | 336,882 | 3.68% |
| [200021869_00014_2](../defect_context_preprocessing_pilot_v2_20261001/comparisons/200021869_00014_2_whole.png) | 14,300 | 21,264 | 0.32% |
| [200010454_00026_1](../defect_context_preprocessing_pilot_v2_20261001/comparisons/200010454_00026_1_whole.png) | 21,248 | 166 | 0.00% |
| [200014740_00065_2](../defect_context_preprocessing_pilot_v2_20261001/comparisons/200014740_00065_2_whole.png) | 30,950 | 4,726 | 0.08% |

## DDRM条件と検証

前回の[条件と結果](../npj_v6_ddrm_edit_comparison_20261001/report.md)を固定。保存済みnpjの`04_degradation`、固定v6の`edit`、修正版の`edit`を同じinpaintingの欠損領域に接続した。**旧npjの黒文字再生成と朱の合成を含む全工程の再現ではない。** npj/v6出力はハッシュ確認した前回の保存結果を再利用した。

同じ原画像、`model_char/ckpt_epoch0350.pth`、RGB [-1,1]、20段、sigma=0.05、etaB=1、etaA/etaC=0.5。128pxパッチ、重なり32px、周辺64px、同じ画像座標のノイズ場、同じ重み付き合成。文字用xt抑制は全方式で無効。編集範囲外は原画素へ戻す。sampler・学習済み重みは変更していない。ROIの周辺も含めてマスクが完全一致する4箇所は、v6保存出力を完全一致で再利用した。ROI内追加0でも周辺でマスクが変わる場合は再推論した。

入力・保存済み比較・ソース・重み・全出力のSHA-256、全11ページのマスク演算、朱／暖色保留の維持、編集と保護・保留の非重複、全15箇所の原寸出力・編集範囲外の不変を検証。同じマスクとノイズでDDRM出力が画素単位で一致することも実行時に確認した。追加候補画素数と出力差は一致するとは限らず、マスク変更によって以前から編集対象だった画素の生成も変わる。

[独立した整合性検証](verification.json)、[DDRM実行時の検証](../defect_context_ddrm_pilot_v2_20261001/verification.json)、[全例の原寸データ](../defect_context_ddrm_pilot_v2_20261001/native)、[暫定目視所見](visual_review.json)。いずれも精度・真正性の検証結果ではない。

## 次に進める課題

固定v6を比較基準として残し、この試作は採用しない。次は前処理の課題を、**文字のない紙面の損傷の検出**と**文字に接する汚れの安全な分離**に分ける。前者では黒い損傷の縁・紙色に近いシミ、後者では薄い黒／赤褐色筆画の侵食を同時に評価する。今回の15箇所は開発側に置き、少数の手動正解領域と未使用の別冊を確保してから改善率を測る。

紙面修復で文字状の線を生成しない条件も、マスクの正しさと分けて検証する。現段階で拡散モデル本体の学習変更を始める根拠にはせず、まず編集領域が紙面か文字を含むかの接続仕様と評価を整理する。重なりが不明な箇所は保留する。前処理を続ける研究上の理由は残るが、前処理だけで全損傷が自然に復元できるとは結論しない。

実装：[初回試作](../../scripts/benchmark/probe_defect_context_preprocessing.py)、[接続保護を加えた修正版](../../scripts/benchmark/probe_defect_context_anchors.py)、[同条件DDRM確認](../../scripts/benchmark/check_defect_context_ddrm.py)、[検証・画像整理](../../scripts/benchmark/package_defect_context_review.py)。旧npj、固定v6、過去の比較結果と利用者の既存変更は更新していない。
