# 現状の課題を打破する文献候補と検証順

調査日：2026-10-02。論文本文・出版社・著者公式リポジトリを確認した。推薦の根拠と、このプロジェクトへの適用案を分けて記載する。ここに挙げた外部モデルは、まだ手元の古文書では推論していない。

**優先するのは、背景補正後の筆画抽出と、汚れ除去を直接学習したモデルとの比較。保存済みnpjを基準として維持し、まず前処理・分類を改善する方針が現実的だと考える。** 拡散モデルの再設計は後順位にする。これは手元の失敗結果と文献からの判断であり、解決を実証した結論ではない。

## 手元の失敗から、何を探したか

1. v6でも濃い汚れがprotectへ入り、実際の保存npjに黒い塊が残る。
2. 原画像と歴史的保存入力の濃淡差が大きいと、同じ固定分類の候補が変わる。原画像へ戻すだけでも塊全体は解決しない。
3. [枠外補修試作の訂正版](../saved_npj_ruled_margin_pilot_v2_20261002/report.md)では、枠内を保存npjのまま維持し、余白の更新候補は作れた。しかし直近の黒い外周を参照する調和補間は灰色のぼかしになり、細線保持は汚れ内の構造まで保持した。両候補とも未採用。

したがって、暗さや細線だけによる保護を増やすより、「背景の補正」「筆画を識別する構造」「汚れと文字の文脈」「インクと紙面参照の役割分離」を探す必要がある。比較は[実際の保存npj](../saved_npj_paper_only_followup_review_v2_20261002/report.md)を使う。過去の劣化マスク穴埋め診断を通常npjの結果として扱わない。

## 優先文献

| 優先する用途 | 文献 | 手掛かり | このプロジェクトでの位置付け |
| --- | --- | --- | --- |
| 汚れ除去の直接的な比較 | StainRestorer、WACV 2025 | 実際に付着させた汚れと清浄文書の対、汚れ表現の学習 | まず失敗例で追加効果と筆画消失を確認 |
| 前処理の実装候補 | Xiong et al.、EURASIP JIVP 2021 | 背景補正→筆幅推定→エネルギー最小化→接続保持 | 文字候補の作り方を現v6と比較 |
| 筆画保護の設計 | Jia et al.、Pattern Recognition 2018 | 推定筆幅内の対向するエッジから筆画を抽出 | 細線なら何でも保護する条件を見直す |
| 学習済み文字抽出の対照 | DocRes、CVPR 2024 | 二値化・勾配・閾値の事前情報を使う統合復元 | 二値化結果を文字候補として検証 |
| 将来の分類器設計 | DR-Mamba、2026プレプリント | 細部経路と背景経路、入力依存の背景抑制 | 学習を導入する場合の構造参考 |
| 筆画が本当に劣化した場合 | DocDiff、ACM MM 2023 | 低周波の粗復元と高周波残差の拡散モデル | 汚れ分類が改善してから検討 |

### 1. High-Fidelity Document Stain Removal via A Large-Scale Real-World Dataset and A Memory-Augmented Transformer

Mingxian Li et al.、WACV 2025。モデル名StainRestorer。

**論文の内容：** 汚れ除去を直接扱い、約300文書から茶・赤青インク等を実際に付着させて撮影した5,060対のStainDocを構築。DocMemoryとTransformerで汚れ表現を学習する。標準A4紙を使い、低コントラストや大きな細部喪失等の撮影品質が悪い対は除外する。歴史的和紙やくずし字での有効性をこの論文から断定できない。[論文本文](https://arxiv.org/html/2410.22922v1)、[WACV正式版PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Li_High-Fidelity_Document_Stain_Removal_via_A_Large-Scale_Real-World_Dataset_and_WACV_2025_paper.pdf)。

**適用案：** 原画像を入力する独立した比較対象として試す。保存npjを置き換える前に、濃い塊の除去、薄い筆画・朱印・紙模様の変化を確認する。出力差だけを「汚れマスク」として採用しない。汚れ表現の学習が手作り条件より役立つかを確かめる。

**再現性：** 著者の[公式コード](https://github.com/CXH-Research/StainRestorer)と[学習済み重みのリリース](https://github.com/CXH-Research/StainRestorer/releases/tag/Weights)を確認。各重みの用途、入力正規化、タイル境界は実行前に確認する。StainDoc_Sealの印章除去は、本プロジェクトの朱印保持とは目的が異なる。[公式説明](https://github.com/CXH-Research/StainRestorer)。

### 2. An enhanced binarization framework for degraded historical document images

Wei Xiong, Lei Zhou, Ling Yue, Lirong Li, Song Wang、EURASIP Journal on Image and Video Processing、2021。

**論文の内容：** 歴史的手書き文書向け。最小エントロピーによるStroke Width Transformで背景補正の構造要素の大きさを決め、Laplacianとエッジに基づくエネルギー最小化で文字を分離する。接続保持と孤立ノイズ除去を後処理する。手設計特徴には強い劣化で限界があり、グラフカットのメモリも課題と明記される。[論文全文](https://link.springer.com/article/10.1186/s13640-021-00556-4)。

**適用案：** 紙面の色を変えるための補正画像と、文字判定用の補正画像を分ける。補正後の文字マスクだけを新候補に使い、実保存npjの筆画は比較基準として残す。現在の固定31px等より筆幅に合わせた背景尺度が有効か調べる。後処理の接続補完は、裏写りをつなぐ可能性があるため独立に評価する。

**再現性：** 著者の[H-DIBCO-2018配布](https://github.com/beargolden/H-DIBCO-2018)を確認。公開ルートにはEXEがあり、Linux向けソースは確認できていない。そのまま即時に実行できる候補とは扱わず、論文の背景補正・筆幅推定を小さく試す場合も再実装との区別が必要。「歴史文書に強い」という一般論だけで、今回の濃い接触汚れが解決するとは扱わない。

### 3. Degraded document image binarization using structural symmetry of strokes

Fuxi Jia, Cunzhao Shi, Kun He, Chunheng Wang, Baihua Xiao、Pattern Recognition 74、225–240、2018（オンライン・著者READMEの年表記は2017）。

**論文の内容：** Structural Symmetric Pixels（SSP）は、筆画の両側にある大きな勾配と反対向きのエッジを手掛かりにする。背景補正、筆幅推定、複数局所閾値の投票を組み合わせる。色や明るさの閾値だけに依存しない筆画抽出の参考になる。[出版社の論文説明・抄録](https://www.sciencedirect.com/science/article/pii/S0031320317303849)。この論文は公開抄録・説明を確認し、出版社の全文は未取得。

**適用案：** 「細い暗部」から一歩進め、推定筆幅と対向エッジが整合する部分を筆画保持候補にする。汚れや虫食いの輪郭にも対向エッジがあり得るため、これだけで確定しない。枠線・筆画・損傷を別に表示する診断として試す。

**再現性：** [著者公式配布](https://github.com/FuxiJia/DocumentBinarizationSSP)にはEXEのZIP/RARがある。Pythonソースがそのまま使えるとは確認できず、手元で使う場合は実装手段を確認する。これは新手法の完成形というより、保護マスクの構造条件の根拠として優先する。

### 4. DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks

Jiaxin Zhang et al.、CVPR 2024。

**論文の内容：** 5種類の文書復元を単一モデルで扱い、Dynamic Task-Specific Promptでタスクに応じた事前情報を与える。二値化ではSauvola出力・閾値マップ・勾配を使い、歴史文書を含む二値化データで学習する。全タスクで専用モデルを上回るという主張ではない。[論文本文](https://arxiv.org/html/2405.04408v1)。

**適用案：** `binarization`の出力を、現在の文字保護範囲と比較する。`appearance`による背景整理も別列で確認できるが、明瞭な出力を正解文字と自動認定しない。復元画像・文字候補・汚れ候補の評価を分ける。

**再現性：** [公式コード・重み案内](https://github.com/ZZZHANG-jx/DocRes)とタスク別推論の説明を確認。まず学習済みモデルの対照として利用する余地がある。現在のnpjモデルを再設計する必要はない。

### 5. DR-Mamba: Automatic Inference-Time Domain Adaptation for Document Image Binarization via Sample-Conditioned Detail-Background Suppression

Sheng-Wei Chan, Jen-Shiun Chiang、2026-06-21公開のarXivプレプリント。

**論文の内容：** 細部経路と背景経路を分け、入力依存のゲートで背景を抑制し、筆画の接続を意識した損失で学習する。DIBCO年度を一つずつ除外する評価を報告。推論時の対象ドメインラベルや勾配更新は不要だが、モデル本体は教師あり学習されている。「学習不要」ではない。他手法との数値は異なる訓練分割を含む参考比較と明記。[論文全文](https://arxiv.org/html/2606.22625v1)。

**適用案：** 異なる冊・紙色・濃淡に対して、文字と背景を別経路で扱う分類器の設計参考。薄い筆画保持と背景抑制を別の評価項目にする。今回確認した範囲では著者の公開コード・重みを特定できておらず、即時の再現候補より将来の設計候補とする。査読済み論文として扱わない。

### 6. DocDiff: Document Enhancement via Residual Diffusion Models

Zongyuan Yang et al.、ACM Multimedia 2023。

**論文の内容：** Coarse Predictorで低周波を復元し、高周波残差を拡散モデルで補う。HRRによる他手法の文字エッジ改善も示す。文書復元と残差拡散の構造的参考になる。[論文](https://arxiv.org/abs/2305.03892)。[公式実装](https://github.com/Royalvice/DocDiff)には推論例と学習済みモデル案内があり、READMEで確認できる重みは文書のぼけ除去用。汚れ除去の万能重みと考えない。

**適用案：** 現在は汚れが文字として保護される問題が先にあるため後順位。筆画保持の有効性を保存npjに対して実証した後、細部の補修が必要なら比較する。npjのDDRMへHRRを追加するだけで、そのまま適切な結果になるとは確認されていない。全体を鮮鋭化すれば汚れも強くなる可能性がある。

## 新しい統合復元研究も確認したが、今回の優先度は下げる

- **Uni-DocDiff（2025、arXiv版参照）：** 局所高周波と広域低周波のPrior Poolをタスクに応じて融合する。文字細部と背景情報を別に扱う設計参考になるが、6タスクには手書き除去も含まれるため、手書きを保持する今回の目的とタスクを混同しない。今回、即時利用できる著者コード・重みは確認できていない。[本文](https://arxiv.org/html/2508.04055v1)。
- **MMDIR（CVPR 2026）：** ぼけ・影・文字透かし・印章の混合劣化を扱い、MixedDocを提供する。ただし、歴史文書の虫食い・シミと朱印保持を直接実証したものではない。確認した[著者README](https://github.com/xiaomore/MMDIR)はデータと予測結果の案内が中心で、モデル再現の手順・重みは未確認。[正式版論文](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_MMDIR_Multimodal_Instruction-Driven_Framework_for_Mixed-Degradation_Document_Image_Restoration_CVPR_2026_paper.pdf)。

## 次の検証に落とすなら

ここからは当プロジェクトへの提案であり、上記論文の既存結果ではない。

1. **保存npjを固定し、原画像から背景補正後の文字候補を得る。** 原入力の濃淡差を明示し、Xiong系の筆幅に合わせた背景補正とSSPの対向エッジが、濃い塊を過剰に保護しなくなるか確認する。背景補正画像をそのまま最終復元へ貼らない。
2. **StainRestorerとDocResを同じ失敗例・対照へ適用する。** 本当に汚れが減るか、薄い主線が消えないか、朱印が保持されるかを見る。モデルを入れること自体を改善としない。出力差を汚れGTや保護解除の根拠にしない。
3. **文字保持・汚れ編集・不確実領域・紙面参照を分ける。** 保持したインクを紙面色の参照には使わない。どの候補を処理しても、保持帯の外周が黒い場合に、単純な調和補間が黒さを伝播させる問題を別途評価する。文字マスクの補集合をすべて汚れにしない。
4. **結果が良ければ小さい人手注釈で確かめ、条件を固定して別冊へ進む。** 汚れ候補と主要筆画・薄い筆画の小領域を注釈すれば、除去の精度と筆画保持を別に測れる。現在の開発例を独立した未使用冊の評価として数えない。OCRだけではくずし字・朱印の保持を十分に判断できないので比較PNGを主にする。イラストページは除外を継続する。

現在の選択なら、**前処理の構造的な見直し＋公開汚れ除去モデルとの比較**を優先し、DocDiff／Uni-DocDiff等の拡散モデル再設計はその結果を見て判断する。

## 新手法としての位置付け

汚れ除去そのものはStainRestorer等で既に研究されている。したがって、「シミを消せた」だけで初めての手法とは主張できない。今回目指せる貢献は、たとえば**くずし字・朱印・薄い筆画を保持し、既存npjを利用しながら接触汚れを選択的に除去する役割分離と、その有効性の検証**になる。ただし、それも先行研究との比較と別冊評価を経てから判断する。

## 保存した資料

[papers](papers/)に取得できた正式版／arXiv PDFと検索用テキスト、[取得記録](source_downloads.json)にURL・ハッシュ・取得状態を保存。[公式資料](official_resources/)には著者README・一部設定のみ保存し、外部コードを実行したりモデル重みを導入したりしていない。GitHub APIはレート制限で取得できなかったため、公式raw READMEを取得した。PDF入手と、モデルの再現実行の成功は区別する。
