# 茶色い筆画の誤編集を防ぎ、既知の黒残留除去を維持

2026-10-04。原画像の線状応答から中性色の制限を外すと、前回の茶色い本文の変更3成分・111画素を全て保持できた。同時にhnsdの本文側の追加補修3,781画素（暗→明2,109画素）と虫食いの前回結果を全画素維持できた。汚れ除去と観測された筆画の保持を両立する手掛かりだが、既知の開発対照であり一般採用はまだしない。

最初に見る比較：

- [hnsd：成分ごとの見送りで戻った黒い残留を、今回の線保持では維持](old_saved_hnsd_inside_frame_residual.png)
- [37画素の筆画削除を防ぐ](new_new_warm_lower_loss_previous_change_01.png)
- [52画素の筆画端削除を防ぐ](new_new_warm_faint_writing_previous_change_02.png)
- [前回残った22画素の筆画縁変更も0](new_new_warm_faint_writing_previous_change_01.png)
- [線保持マスクが筆画を覆う理由](guard_explains_stroke_preservation.png)
- [汚れ候補と保持の範囲](guard_keeps_dirt_candidate.png)
- [全比較PNG](comparison_png.zip)

## 変更した条件

従来は、線の形があっても原画像のLab a>6、または紙面からのa差>4だと中性線の保持対象から外れた。今回は既存の線状応答（sigma1.5/3/6、応答1.5以上、局所L低下6以上、4画素の保持幅）をそのまま使い、この中性色条件を外した。全ページ原画像で処理し、比較用の座標やケース名を判定へ入れない。

比較する2条件は、色を問わず線を保持する条件と、その線の中心の8連結成分が固定v6のprotectに接続する場合だけ保持する条件。従来の保持も残し、前回の追加候補から新しい保持を除く。紙面参照も新しい保持から外し、同じ明るさ・局所残差・ab距離4の条件と補間を使う。朱・追加暖色保留・適用外はnpj基準と一致する。前回の全ページ枠外補修は全画素維持するため、この追加保持は枠外補修の正しさを新たに証明しない。

17範囲すべてを同じ条件で処理した。新しいDDRM推論・学習・文字生成・筆画補完・色の全体補正はない。比較のnpj基準は前回の実保存出力または凍結コードの新規crop対照を再利用し、その由来を変更していない。

## 汚れ除去と保持の結果

| 範囲 | 色を問わず線保持：追加適用 | protect接続線のみ：追加適用 | 色を問わず線保持：暗→明 |
| --- | ---: | ---: | ---: |
| [old_fresh_lower_paper](old_fresh_lower_paper_comparison.png) | 155 | 587 | 25 |
| [old_fresh_binding_mark](old_fresh_binding_mark_comparison.png) | 0 | 1656 | 0 |
| [old_wormhole](old_wormhole_comparison.png) | 0 | 0 | 0 |
| [old_seal_control](old_seal_control_comparison.png) | 0 | 0 | 0 |
| [old_fresh_printed_text](old_fresh_printed_text_comparison.png) | 0 | 0 | 0 |
| [old_unseen_lower_frame_mark](old_unseen_lower_frame_mark_comparison.png) | 0 | 0 | 0 |
| [old_unseen_lower_binding_mark](old_unseen_lower_binding_mark_comparison.png) | 0 | 0 | 0 |
| [old_saved_hnsd](old_saved_hnsd_comparison.png) | 3781 | 3781 | 2109 |
| [old_page_200019865](old_page_200019865_comparison.png) | 0 | 0 | 0 |
| [old_known_200010454_v2](old_known_200010454_v2_comparison.png) | 0 | 0 | 0 |
| [old_old_full_page](old_old_full_page_comparison.png) | 106 | 106 | 11 |
| [new_new_warm_upper_loss](new_new_warm_upper_loss_comparison.png) | 0 | 0 | 0 |
| [new_new_warm_lower_loss](new_new_warm_lower_loss_comparison.png) | 0 | 0 | 0 |
| [new_new_warm_faint_writing](new_new_warm_faint_writing_comparison.png) | 0 | 0 | 0 |
| [new_new_gray_binding_mark](new_new_gray_binding_mark_comparison.png) | 0 | 0 | 0 |
| [new_new_gray_writing](new_new_gray_writing_comparison.png) | 0 | 0 | 0 |
| [new_known_saved_red_seal](new_known_saved_red_seal_comparison.png) | 0 | 0 | 0 |

茶色い本文の37＋22＋52画素は全て新しい保持マスクが直接覆っている。単に参照不足で補間が止まったことを筆画認識の改善と数えた結果ではない。両条件で追加変更0。hnsdは追加適用3,781・変更3,777・暗→明2,109画素で、前回の出力と全画素一致する。前回の成分見送りでは追加適用474・暗→明136まで減ったため、今回の形状保持ではその局所除去を失わずに済んだ。左本文の固定対照は追加変更0。暗→明はLab L<30→L>60であり、汚れの正解数ではない。

虫食いの組合せ結果も前回と全画素一致。これは前回の枠外補修を維持した結果で、今回新しく虫食いを解いたという意味ではない。白黒の4範囲は元の暖色条件で候補0のまま、汚れ除去の成功に数えない。朱印の既知対照も追加変更0。

薄い文字の実保存対照old_full_pageでは、4成分・104変更画素が両条件で残る。20／16／21／47画素を全て拡大して確認した。灰色の別の像・紙面の点・筆画端との区別は未確定で、無誤編集とは呼ばない。主要な黒い筆画の大きな欠落はこの4拡大では見えないが、それだけで変更を汚れ除去と認定しない。

- [残る変更1](old_old_full_page_thin_control_01.png)
- [残る変更2](old_old_full_page_thin_control_02.png)
- [残る変更3](old_old_full_page_thin_control_03.png)
- [残る変更4](old_old_full_page_thin_control_04.png)

紙模様・綴じ側の跡にも2条件の差がある。色を問わない保持は追加適用155／0、protect接続線だけの保持は587／1,656画素。材質の正解がないため、除去量が大きい方を採用する判断はしない。線の形に見える汚れを保持してしまう限界も残る。

## 検証と次段階

全ページの形状マスクを再計算し、原画像crop・保存基準・候補の包含・必須保持・適用外・参照条件・前回枠外結果・表示パネルを照合した。補間時は非参照画素をマゼンタへ置換しても適用範囲の結果と参照範囲が変わらないことを検証した。全111画素の新保持との重なり、およびhnsdと虫食いの前回出力との一致を独立に検査した。

今回の17範囲は既に確認済みの開発例で、未使用冊での独立精度や一般化の根拠ではない。次段階はこの条件を固定し、実保存npjの別ページ／別範囲の汚れと茶色い本文で検証する。薄い像の4変更も残したまま評価し、本文保持だけでなく追加の汚れ除去があることを確認する。

[固定条件・依存関係](manifest.json)／[実行診断](diagnostics.json)／[事前に固定した表示範囲](witnesses.json)／[範囲別の変化](witness_metrics.json)／[独立検証](independent_verification.json)
