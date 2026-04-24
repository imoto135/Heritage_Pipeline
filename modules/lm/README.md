# Language Model Module

このディレクトリはOCR後段の言語モデル評価・再ランキングを扱います。

論文方針として、最良ルートは KenLM -> BERT パイプラインです。
そのため、通常利用は次の入口を推奨します。

- `run_best_pipeline.py`（互換ラッパー）
- 実体: `pipelines/run_best_pipeline.py` -> `pipelines/eval_kenlm_then_bert.py`

## Structure

- `pipelines/`: 評価パイプライン本体（KenLM/BERT/比較系）
- `training/`: 学習・学習前処理
- `utils/`: 読順・bboxなどの補助処理
- ルート直下の同名スクリプト: 後方互換ラッパー

## Recommended Flow

1. OCR top-k 候補を入力
2. KenLM で系列スコアリング（beam search）
3. BERT で文脈再ランキング

## Main Scripts

- `pipelines/run_best_pipeline.py`: 推奨実行入口（主要パラメータ既定値を設定）
- `pipelines/eval_kenlm_then_bert.py`: KenLM -> BERT の本体評価スクリプト
- `pipelines/eval_kenlm.py`: KenLMのみ評価
- `pipelines/eval_bert.py`: BERTのみ評価

## Secondary / Utility Scripts

- `pipelines/eval_kenlm_q.py`, `pipelines/eval_bert_q.py`: 可視化・統計の派生版
- `pipelines/eval_nanogpt.py`: nanoGPT 比較実験
- `training/insertspace.py`: KenLM学習用テキスト整形
- `training/bert.py`: BERT学習
- `utils/checkbbox.py`, `utils/xycut.py`: 前処理・読順処理ユーティリティ

## Data / Model Assets

- `*.arpa`: KenLMモデル
- `handtrain_*.txt`, `train_char.txt`: 学習用テキスト
- `gt_pages/`: 評価用GT

## Quick Start

例:

```bash
python run_best_pipeline.py \
	--coord-csv /path/to/<book>/<book>_coordinate.csv \
	--images-dir /path/to/<book>/images \
	--gt-dir /path/to/gt_pages/<book> \
	--ckpt /path/to/ocr/best.pth \
	--classes /path/to/ocr/classes.txt \
	--bert-model /path/to/bert/final \
	--stats --annotate-all --freeze-if-not-in-topk
```

設定テンプレートは `configs/best_pipeline.env.example` を参照してください。

補足: 旧コマンド（例: `python eval_kenlm_then_bert.py`）もラッパー経由でそのまま実行可能です。
