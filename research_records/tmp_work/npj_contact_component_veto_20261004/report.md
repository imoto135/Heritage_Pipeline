# 保護に触れる成分を見送る：筆画削除と汚れ残留を別々に確認

2026-10-04。前回の線保持が茶色い本文を取りこぼしたため、追加補修の8連結成分が固定v6のprotect、またはprotect|holdに1画素でも触れたら、成分を丸ごと見送る2条件を全17範囲へ同じように適用した。これは結果を見た後の開発比較であり、新しい独立評価ではない。前回の枠外補修は全画素維持する。前回の組合せがない実保存3対照はnpj基準に線保持補修を加えたものを起点にする。追加DDRM・学習・補間・文字生成・色補正・閾値の調整はない。

同じ基準で出力の一部を見送るため、見送り箇所はnpj基準／前回枠外補修へ戻る。筆画を生成して直す処理ではない。既存v6のprotectも汚れを含み得るので、見送り量を改善量と数えない。

| 範囲 | 前回追加 | protect接触見送り後 | protect／hold接触見送り後 |
| --- | ---: | ---: | ---: |
| [old_fresh_lower_paper](old_fresh_lower_paper_comparison.png) | 806 | 628 | 598 |
| [old_fresh_binding_mark](old_fresh_binding_mark_comparison.png) | 2956 | 1474 | 686 |
| [old_wormhole](old_wormhole_comparison.png) | 0 | 0 | 0 |
| [old_seal_control](old_seal_control_comparison.png) | 0 | 0 | 0 |
| [old_fresh_printed_text](old_fresh_printed_text_comparison.png) | 0 | 0 | 0 |
| [old_unseen_lower_frame_mark](old_unseen_lower_frame_mark_comparison.png) | 0 | 0 | 0 |
| [old_unseen_lower_binding_mark](old_unseen_lower_binding_mark_comparison.png) | 0 | 0 | 0 |
| [old_saved_hnsd](old_saved_hnsd_comparison.png) | 3781 | 474 | 474 |
| [old_page_200019865](old_page_200019865_comparison.png) | 99 | 0 | 0 |
| [old_known_200010454_v2](old_known_200010454_v2_comparison.png) | 0 | 0 | 0 |
| [old_old_full_page](old_old_full_page_comparison.png) | 106 | 106 | 37 |
| [new_new_warm_upper_loss](new_new_warm_upper_loss_comparison.png) | 0 | 0 | 0 |
| [new_new_warm_lower_loss](new_new_warm_lower_loss_comparison.png) | 37 | 0 | 0 |
| [new_new_warm_faint_writing](new_new_warm_faint_writing_comparison.png) | 74 | 22 | 22 |
| [new_new_gray_binding_mark](new_new_gray_binding_mark_comparison.png) | 0 | 0 | 0 |
| [new_new_gray_writing](new_new_gray_writing_comparison.png) | 0 | 0 | 0 |
| [new_known_saved_red_seal](new_known_saved_red_seal_comparison.png) | 0 | 0 | 0 |

新しい冊で見つかった全変更3成分：

- [new_new_warm_lower_loss・成分1](new_new_warm_lower_loss_component_01.png)：変更画素 {'protect_component_veto': 0, 'protect_hold_component_veto': 0}
- [new_new_warm_faint_writing・成分1](new_new_warm_faint_writing_component_01.png)：変更画素 {'protect_component_veto': 22, 'protect_hold_component_veto': 22}
- [new_new_warm_faint_writing・成分2](new_new_warm_faint_writing_component_02.png)：変更画素 {'protect_component_veto': 0, 'protect_hold_component_veto': 0}

原画像の筆画形と汚れの材質の判定は目視が必要。黒画素減少やマスク保存の検証だけで汚れ精度・筆画の完全保持は決められない。小さな縁の変更が残る条件は未解決の回帰例として保持する。

[全比較PNG](comparison_png.zip)／[固定条件](manifest.json)／[画素数](diagnostics.json)／[全変更成分の比較記録](regression_controls.json)

## 前回報告の数え方の訂正

[別冊転用の凍結報告](../npj_dirt_new_book_20261004/report.md)の本文に「2つの欠損範囲と3つの本文範囲」とあるが、正しくは本文範囲4つ（暖色上端・暖色下端・薄い本文・白黒本文）。保存した8範囲のJSONと比較PNGは4本文範囲を含み、いずれも組合せ変更0という結果は変わらない。凍結成果物を変更せずここに訂正を記録する。

元の26PNGの比較と今回の20PNGは役割が異なる。前者は固定条件の2新規冊への転用、後者は転用後に考えた見送り条件を既知の11範囲と転用6範囲へ適用した開発比較。冊数・独立評価数を足し合わせない。
