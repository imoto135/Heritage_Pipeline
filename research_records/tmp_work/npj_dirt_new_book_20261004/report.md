# 固定条件を別冊へ適用：追加除去の利点を確認できず、筆画誤編集が見つかった

2026-10-04。汚れ除去を主目的に、これまでの実験登録にない暖色・白黒の2冊を原画像から選んだ。今回の線保持付き補修は一般採用しない。新しい暖色冊ではnpj対照が欠損の縁を既にかなり除去しており、追加補修は別位置の実際に観測される筆画を削った。文字の生成や欠けた筆画の補完は実施していない。

まず見る比較：

- [欠損の縁：npj対照で既に除去](new_warm_upper_loss_defect_and_rule.png)
- [筆画の誤編集37画素：原画像と全変更](new_warm_lower_loss_all_change_01.png)
- [筆画端52画素の誤編集](new_warm_faint_writing_all_change_02.png)
- [残る筆画縁の変更22画素](new_warm_faint_writing_all_change_01.png)
- [既知の実保存朱印対照](known_saved_red_seal_actual_seal_and_text_control.png)
- [全比較PNG一括](comparison_png.zip)

各PNGは原画像と同一のnpj基準を並べている。主比較の右端は前回の全ページ枠外補修を維持し、その外に線保持補修を追加した組合せ。今回は全4原ページで支持された枠が見つからず、枠外補修は全て0。したがって右端と線保持補修単独は全画素同じ。0変更を認識精度や汚れ除去の成功として数えない。

| 範囲 | v6適用 | 枠外適用 | 線保持適用 | 線保持の暗→明 | 目視判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| [new_warm_upper_loss](new_warm_upper_loss_comparison.png) | 0 | 0 | 0 | 0 | 欠損縁はnpj対照で既に目立たなくなる。追加除去なし。罫線は一部残る |
| [new_warm_lower_loss](new_warm_lower_loss_comparison.png) | 0 | 0 | 37 | 15 | 事前の欠損範囲は追加変更0。別位置の観測筆画を37画素削る |
| [new_warm_faint_writing](new_warm_faint_writing_comparison.png) | 83 | 0 | 74 | 29 | 追加74画素。筆画縁22画素と筆画端52画素を変更。汚れ改善としない |
| [new_gray_binding_mark](new_gray_binding_mark_comparison.png) | 0 | 0 | 0 | 0 | 元の跡の材質未確定。候補0のまま未解決 |
| [new_gray_writing](new_gray_writing_comparison.png) | 0 | 0 | 0 | 0 | 候補0の対照。汚れ除去成功としない |
| [known_saved_red_seal](known_saved_red_seal_comparison.png) | 861 | 0 | 0 | 0 | 既知の実保存朱印。組合せ変更0。新規朱印精度・朱印復元ではない |

暗→明はLab L<30からL>60への変化であり、汚れの正解画素数ではない。事前に指定した2つの欠損範囲と3つの本文範囲は、今回の組合せで全て変更0。範囲外の全変更3成分・計111画素を省略せず拡大したため、事前の本文範囲が無変更でも誤編集を見つけられた。特に37画素と52画素は、原画像に筆画が見える場所でnpj筆画の形が削られている。22画素も筆画縁に位置し、汚れ除去の肯定例にはできない。画素単位の材質GTはない。

なぜ線保持があっても削るのか：保持マスクは中性の線状応答と朱／暖色保留に接続する領域に限る。マスクに入った画素の保存は検証できるが、茶色い本文の全てがこの条件に入るわけではない。広い暗部候補が本文まで含み、保持条件から外れた部分を紙色で埋めていた。これは参照色の緑化やRGB/BGR交換とは別の、汚れと本文の選別の問題。今回の変更範囲に明るい緑（L>60,a<-8）は0だが、色が正しくても筆画誤編集は解決しない。

次は補修量を増やす条件調整を先に行わず、実保存npjに残っている汚れの例へ戻る。原画像で見える欠損の縁・シミと実際の筆画を固定した小範囲で別々に確認し、今回の誤編集3成分を回帰対照へ入れる。新しいページでnpjが既に十分除去している例は追加手法の肯定例にしない。枠の支持不足を安易に緩和して本文を枠外へ出す変更もしない。

## 条件と評価の範囲

新規2冊は200021925（暖色、2原ページ・3crop）と200004148（白黒、1原ページ・2crop）。Top-levelのmanifest／selection／baseline_reviewの登録を調べて選定した。一般の画像一覧・原画像スクリーニングには既出の場合があり、全履歴で未見／モデル学習から独立とは主張しない。暖色冊1冊だけのため冊間成功率を出さない。白黒の0編集も肯定例に加えない。

200021925は挿絵を含む冊だが、本文のみの00007_1と00004_1を原画像で確認して使った。表紙・挿絵・装飾された扉は除外。新しい冊で朱印が見つからなかったので、200010454の既知の実保存npjを別の保持対照にした。独立した新規冊の朱印評価ではない。

新規5cropは原画像の画素をそのまま入力し、凍結Oct2コード・同じモデル・背景・seed999・20stepsを用いた新規対照。過去の全文書推論結果や論文手法の厳密な再現と同一視しない。暖色冊は保存CSCがなかったため、凍結したローカルCSCで2全ページのマスクを新規生成した。白黒冊と既知朱印は保存corpus-v1マスクを変更せず使った。マスクの由来は混在している。既知朱印は実保存の全ページ出力を切り出し、当時の実入力と原画像の画素一致も確認した。

固定v6は全原ページで算出。候補・線保持・参照色距離4・補間条件は前回から変更していない。この比較は保存／新規npjの後に局所的な紙面補修を行う試作であり、DDRMの新設計を適用した比較ではない。

## 検証と実行記録

新規推論入力・実保存出力・原画像crop・マスクcrop・必須保持・適用外を画素で照合した。全原ページ由来の線保持と暗部候補、参照条件、補間結果、非参照画素をマゼンタにする反証試験、組合せ、PNGパネル、ZIP内の全画素とハッシュを検証。内部の画素一致は汚れの認識精度を保証しない。

最初のマスク監査は保存分類00_inputが原画像と同じと仮定して停止した。保存分類用画像にはACE等の処理があり、この仮定は誤りだった。新規モデル入力は原画像のままで、既知朱印の当時の実入力も原画像と一致する。推論後の監査も、既知朱印には新規inputがないことを見落として停止した。いずれも実生成／推論は終了済みで、監査用アダプタで由来を分けて検証し直した。マスク生成やモデル推論を再実行して良い結果を選んでいない。元の停止ログも保存する。

[選定と依存関係](selection.json)／[実行診断](diagnostics.json)／[独立再計算と全変更記録](independent_verification.json)／[分類画像と実入力の監査](mask_input_audit.json)／[推論の監査](inference_complete.json)

## 事前指定範囲の比較

- [new_warm_upper_loss：defect_and_rule](new_warm_upper_loss_defect_and_rule.png)（組合せ変更0画素）
- [new_warm_upper_loss：text_control](new_warm_upper_loss_text_control.png)（組合せ変更0画素）
- [new_warm_lower_loss：defect_and_rule](new_warm_lower_loss_defect_and_rule.png)（組合せ変更0画素）
- [new_warm_lower_loss：text_control](new_warm_lower_loss_text_control.png)（組合せ変更0画素）
- [new_warm_faint_writing：text_control](new_warm_faint_writing_text_control.png)（組合せ変更0画素）
- [new_gray_binding_mark：unconfirmed_binding_mark](new_gray_binding_mark_unconfirmed_binding_mark.png)（組合せ変更0画素）
- [new_gray_writing：text_control](new_gray_writing_text_control.png)（組合せ変更0画素）
- [known_saved_red_seal：actual_seal_and_text_control](known_saved_red_seal_actual_seal_and_text_control.png)（組合せ変更0画素）

## 全変更成分の監査

- [new_warm_lower_loss成分1：37画素](new_warm_lower_loss_all_change_01.png)／[保持マスクとの関係](new_warm_lower_loss_guard_diagnostic_01.png)
- [new_warm_faint_writing成分1：22画素](new_warm_faint_writing_all_change_01.png)／[保持マスクとの関係](new_warm_faint_writing_guard_diagnostic_01.png)
- [new_warm_faint_writing成分2：52画素](new_warm_faint_writing_all_change_02.png)／[保持マスクとの関係](new_warm_faint_writing_guard_diagnostic_02.png)
