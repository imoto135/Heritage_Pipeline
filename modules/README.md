# Modules Structure

このディレクトリはタスク単位で整理しています。

- `detection/`: 文字検出（YOLOX関連）
- `lm/`: 言語モデル学習・評価（KenLM/BERT/nanoGPT）
- `restoration/`: 修復モデル（NAFNet/Unet++）
- `stain_removal/`: 染み除去パイプライン

## Current Script Policy

- `stain_removal/scripts/` を実装の正本にする
- `stain_removal/main1.py` と `stain_removal/main_high_inference.py` は後方互換ラッパー
- `stain_removal/pipeline.py` は `scripts/pipeline.py` へのエントリポイント

## Path Handling Policy

- スクリプト内のモデル・設定ファイル参照は `__file__` 基準の相対解決を優先
- 実行カレントディレクトリ依存を避ける

## Recommended Entrypoints

- `python modules/stain_removal/scripts/main1.py`
- `python modules/stain_removal/scripts/main_high_inference.py`
- `python modules/stain_removal/scripts/pipeline.py`
