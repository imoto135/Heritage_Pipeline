# 文献の手掛かりを保存npjの実例で検証

2026-10-02。「その手順で進めてみよう」を受け、保存済みnpjを固定したまま、背景補償＋対向エッジの構造試作、StainRestorer、DocResを実画像へ適用した。6保存フレーム・5ページ・20固定ROI。すべて既知の開発例で、独立した新規資料の精度評価ではない。イラストページは対象外。ROIは外部モデル出力を得る前に固定し、結果に応じた選別はしない。

## 今回の結論と目視所見

今回いちばん次へつながるのは **DocResを文字の補助候補として使う方向**。保存npj全体を置き換える根拠は得られていない。全20固定ROIを目視し、筆画の連続性、濃い紙面模様の除外、朱印、小さい筆点を分けて判断した。以下は開発例の目視所見で、人による正解注釈や精度確定ではない。

- **薄い筆画：** [上部の薄い曲線](old_01_upper_faint_stroke.png)は原画像と保存npj入力が同一。保存npjで細い部分に切れ目が見えるのに対し、DocResの二値候補は曲線を連続して拾う。ここは次に調べる価値がある。一方、右下の2つの筆点はDocResで小さくなる／薄い方がほぼ落ちるため、候補をそのまま保護の全範囲とすると危険。[中央列](old_06_page_faint_middle.png)・[長い線](old_08_page_long_stroke.png)では主要筆画を拾いながら周囲の淡い像や粒を減らす。淡い像の材質・裏写りかどうかは未確定。
- **濃い汚れと枠線：** hnsdの主比較は入力の濃淡が異なるので、[過去npjと同じ保存入力の対照](matched_10_rule_crossing_dark_mass.png)を追加した。同じ入力でもDocResは枠線下の濃い塊の大部分をインク候補から外し、[本文対照](matched_11_crop_text_control.png)の主要文字を残す。ただし枠線の一部は切れる。これは二値化での分離の手掛かりで、色画像から汚れを補修できた結果ではない。hnsdの参照原画像のタイル／全体推論では大局的な除外傾向が共通だが、細かな線・点が変わる。
- **朱印：** [朱印対照](07_red_seal_control.png)では保存npjは印影の赤を残す。通常StainDoc重みでもStainRestorerは朱を大きく薄め、背景を緑がかった色へ変える。DocResは大きい印を黒候補として部分的に拾うが、小さい印影を大幅に落とす。DocResにも朱保護を任せない。朱領域と周辺を別の保護役割にする方針は引き続き必要。
- **暖色斑点・余白：** [上余白](01_upper_slender_marks.png)・[左余白](02_left_margin_marks.png)では、npjが取り除く暖色の細線をDocResがインクとして拾う例がある。DocRes候補があるという理由だけで文字とは断定できない。200017197の実保存入力には参照原画像にないピンク色の斑点がある。固定ROI名の「紙模様」は確認した材質名ではない。同じ入力の[本文上](matched_05_pattern_text_control.png)・[下部](matched_06_lower_thin_text.png)ではDocResに斑点の縁が黒い輪郭として残り、npjのほうがそれらを整理している。これもDocRes単独採用の反例。斑点の付与目的や材質は未確定。
- **茶色のシミ：** [旧切り出し](old_03_upper_brown_stain.png)には保存npjで茶色の残りがあるが、同じページの[CSC v2保存版](08_known_brown_stain.png)は既にかなり整理されている。DocResがシミを白へ落とすことはインク候補からの除外で、色付き紙面の自然な再構成を証明しない。保存版を混ぜて「npjはこのシミを除去できない」と結論しない。
- **StainRestorer：** 200010454や200017197で背景の格子状の色むらと色変化が目立つ。hnsdでは汚れの塊の改善が小さい。今回の原寸512pxタイルという条件では採用しない。著者設定は256×256へ縮小する評価で、本試行はその再現ではない。したがってモデル自体の有効性を一般に否定せず、学習資料・入力スケール・適用方法の差が残ると扱う。
- **構造ペア試作：** [薄い曲線の分類比較](mask_old_01_upper_faint_stroke.png)・[枠線下の分類比較](mask_10_rule_crossing_dark_mass.png)で紙の繊維、シミ内部の細模様、淡い像まで多数拾う。単純な対向エッジ＋局所背景補償だけでは、紙と文字を十分に分離できない。これは独自試作の限界であり、元の論文の完全実装の評価ではない。

次の検証は、DocResの**候補／確率を保護と保留の補助情報**として扱い、小さな確認済み筆画・筆点・朱印・汚れ・正常紙面を別々に評価する。まずこの固定20ROIで文字を落とす箇所と汚れを拾う箇所に注釈を付け、閾値は開発例で固定し、その後に未使用の別冊で確認する。既存protectを一括してDocResへ置き換えない。DocResで背景とされた全画素を編集せず、汚れ編集候補・朱保護・不確実領域・正常紙面参照を明確に分ける。現行npjの良好な筆画と紙面を保つ前提で、新しい役割分けによる追加効果を同じ復元器で検証する。

今回の到達点は「学習済みの二値化に、細い曲線と濃い紙面模様を分離する手掛かりがある」。筆画保持率、虫食い除去率、シミ除去率、npjへの優位、新手法としての成立は未確定。


## 比較の意味と実装条件

保存npjは実際の既存推論出力そのもの。npj再推論、DDRMの再実行、文字の描き足し、出力の選択合成はしていない。新候補は参照原画像だけから推定。npjの版は混在しており、同じ版の統一ベンチマークではない。各全体PNGの見出しに保存版を示す。200010454の旧切り出しとCSC v2全体は同じページの別条件で、独立2資料として数えない。

StainRestorer列は通常のStainDoc重みによる色出力。DocRes列はbinarizationの文字候補で、白黒の結果を色復元の成功・失敗と混同しない。マスク列は推定の比較で、マスク精度によるDDRM後の差ではない。従来protect／holdは文字だけの正解マスクではなく、朱・未確定の材質も含む。

全画像を原寸で推定。外部モデルは512pxタイル、64px周辺余裕、384pxコアを一度ずつ採用し、端はreflect101。入力縮小なし、float32、TF32無効、全重みstrict一致。StainRestorerはRGBの0〜1入力、返却リスト先頭を採用。DocResは著者コードのBGR＋原画像全体で求めたSauvola/threshold/gradient prompt、最初の2チャネルをsoftmaxして背景クラス1を白へ復号した。著者のモデル構造・重みは改変していないが、原寸タイル推論は本プロジェクトの適用方法であり、論文のベンチマーク条件の忠実再現ではない。hnsdでは全体推論の対照も提示する。

構造試作はSobel対向勾配・3〜32pxの暗い内部・幅推定・形態学的背景補償・再判定を使用。Xiong/Jiaの着想を試した独自診断で、論文方式の完全再現ではない。ページ／ROI固有の座標や正解ラベルを推定へ入れていない。

## 重み・ソースの出自

- [StainRestorer論文](https://arxiv.org/abs/2410.22922)／[著者コード](https://github.com/CXH-Research/StainRestorer)：コミット f25239eb0ea817ce04bdfc2a3209157aa2e7998a、著者公開releaseの[通常StainDoc.pth](https://github.com/CXH-Research/StainRestorer/releases/download/Weights/StainDoc.pth)。mark／seal用重みは使わない。現代の染み付き紙の学習で、古文書での筆画・印章保全を保証しない。
- [DocRes論文](https://arxiv.org/abs/2405.04408)／[著者コード](https://github.com/ZZZHANG-jx/DocRes)：コミット d3e3a18c7c7ad10e4615ca2eb6a10f83128d8560。著者OneDriveからの取得は認証で通らず、[公開再配布](https://huggingface.co/DaVinciCode/doctra-docres-main/blob/main/docres.pkl)を利用。SHA256 `1d6a89d754fe1e58ffd1865eab0ef3f03344798d39197b2d9a77ce4fbc8c02fd` が別公開mirrorとも一致し、著者モデルへ全パラメータ一致。ただし著者配布原本との同一性を直接確認できていない。
- [背景補償の論文](https://link.springer.com/article/10.1186/s13640-021-00556-4)／[筆画の構造対称性の論文](https://doi.org/10.1016/j.patcog.2017.09.032)：着想として参照。[前回の文献調査](../document_restoration_literature_20261002/report.md)に適用範囲と次候補を記録。

## 入力の違い

4フレームは参照原画像と実保存npj入力が全画素一致。200017197は実保存入力に暖色の斑点があり、参照原画像には同じ斑点が見られない。圧縮差だけとは扱わず、付与目的・材質の正解は未確定とする。hnsdは濃淡が大きく異なる。主比較では新モデルは参照原画像、保存npjは過去の保存入力に由来する。特にhnsdでの汚れ残留差を主比較だけでモデルの効果と解釈しない。保存入力自体も並べた[200017197出自比較](input_provenance_page_200017197.png)、[hnsd出自比較](input_provenance_crop_hnsd.png)を参照。

この入力差を切り分けるため、両新モデルにも過去の実保存入力をそのまま渡す追加対照を実行した。設定・重みは主比較と同じ。新たな独立2資料ではない。[200017197の同一入力対照](matched_overview_page_200017197.png)、[hnsdの同一入力対照](matched_overview_crop_hnsd.png)。対応する6固定ROIも全件保存し、特に[濃い塊の同一入力対照](matched_10_rule_crossing_dark_mass.png)で確認する。

## 全固定ROIの比較PNG

- [上余白の細い暖色跡](01_upper_slender_marks.png)／[対応する分類候補](mask_01_upper_slender_marks.png)
- [左余白の線状跡](02_left_margin_marks.png)／[対応する分類候補](mask_02_left_margin_marks.png)
- [薄い文字の対照](03_faint_text_control.png)／[対応する分類候補](mask_03_faint_text_control.png)
- [暖色の紙模様・汚れと断定しない](04_warm_paper_pattern.png)／[対応する分類候補](mask_04_warm_paper_pattern.png)
  [過去のnpjと同一入力の追加対照](matched_04_warm_paper_pattern.png)
- [紙模様上の文字対照](05_pattern_text_control.png)／[対応する分類候補](mask_05_pattern_text_control.png)
  [過去のnpjと同一入力の追加対照](matched_05_pattern_text_control.png)
- [下部の細い筆画と紙模様](06_lower_thin_text.png)／[対応する分類候補](mask_06_lower_thin_text.png)
  [過去のnpjと同一入力の追加対照](matched_06_lower_thin_text.png)
- [朱印の対照・既知ページの別保存版](07_red_seal_control.png)／[対応する分類候補](mask_07_red_seal_control.png)
- [既知の茶色い斑点・別保存版](08_known_brown_stain.png)／[対応する分類候補](mask_08_known_brown_stain.png)
- [既知の濃い筆画・別保存版](09_known_dense_text.png)／[対応する分類候補](mask_09_known_dense_text.png)
- [枠線に接する濃い塊](10_rule_crossing_dark_mass.png)／[対応する分類候補](mask_10_rule_crossing_dark_mass.png)
  [過去のnpjと同一入力の追加対照](matched_10_rule_crossing_dark_mass.png)
- [濃い文字と枠線の対照](11_crop_text_control.png)／[対応する分類候補](mask_11_crop_text_control.png)
  [過去のnpjと同一入力の追加対照](matched_11_crop_text_control.png)
- [下端の紙面対照](12_crop_lower_paper.png)／[対応する分類候補](mask_12_crop_lower_paper.png)
  [過去のnpjと同一入力の追加対照](matched_12_crop_lower_paper.png)
- [上部の薄い筆画](old_01_upper_faint_stroke.png)／[対応する分類候補](mask_old_01_upper_faint_stroke.png)
- [濃い筆画の対照](old_02_upper_dense_control.png)／[対応する分類候補](mask_old_02_upper_dense_control.png)
- [筆画の上方の茶色い斑点](old_03_upper_brown_stain.png)／[対応する分類候補](mask_old_03_upper_brown_stain.png)
- [淡い茶色の広がり](old_04_upper_pale_stain.png)／[対応する分類候補](mask_old_04_upper_pale_stain.png)
- [左列の薄い筆画](old_05_page_faint_left.png)／[対応する分類候補](mask_old_05_page_faint_left.png)
- [中央列の細い筆画](old_06_page_faint_middle.png)／[対応する分類候補](mask_old_06_page_faint_middle.png)
- [右列の濃い筆画の対照](old_07_page_dense_control.png)／[対応する分類候補](mask_old_07_page_dense_control.png)
- [左寄りの長い細線](old_08_page_long_stroke.png)／[対応する分類候補](mask_old_08_page_long_stroke.png)

## 全体・タイル感度・検証

- [200019865_00003_1・保存npj：ddrm_visual_check既存版](overview_page_200019865.png)
- [200017197_00008_1・保存npj：trial_run既存版](overview_page_200017197.png)
- [200010454_00002_2・保存npj：CSC v2・同ページ追加対照](overview_known_200010454_v2.png)
- [hnsd001_011_crop・保存npj：旧v1・濃淡の異なる保存入力](overview_crop_hnsd.png)
- [200010454_00002_2・保存npj：旧v1マスク・selfref](overview_old_upper_crop.png)
- [200018243_00003_2・保存npj：v1true・ページ出力](overview_old_full_page.png)

[Stainタイル感度](tile_sensitivity_stain.png)、[DocResタイル感度](tile_sensitivity_docres.png)、[全58比較PNG ZIP](comparison_pngs.zip)。拡大パネルは同じ座標の整数最近傍表示で色補正なし。全体像のみ閲覧用に縮小。

[数値診断](diagnostics.json)の候補画素数・信頼度・出力差は精度、筆画保持率、汚れ除去率ではない。手描きGT、材質確定、未知の別冊評価は未実施。画像を見るための診断値に留める。

入力・保存npj・全依存ソース・モデル出力ハッシュ、全パネル画素一致、原寸・有限値・DocRes背景確率からの独立復号、全ZIP項目CRCとバイト一致を確認。[検証](verification.json)に保存。今回モデルを実行しただけで、現行処理への採用・NPJへの上書きはしていない。
