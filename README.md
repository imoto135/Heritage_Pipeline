# Heritage_Pipeline

古文書OCR・修復パイプラインをまとめたルートリポジトリです。

## 構成

- `modules/detection`: 文字検出
- `modules/lm`: 言語モデル後処理（推奨: KenLM -> BERT）
- `modules/stain_removal`: 染み除去・修復
- `models/`: 学習済み重みなど（`.gitignore` で除外）

## 運用方針

- Git管理はルート（このリポジトリ）で一元化
- 以前のネストリポジトリの `.git` は `.git_backups/` に退避済み
- 大きい成果物（重み、出力、wandbログ等）は `.gitignore` で除外

## メモ

詳細な実行方法は各モジュールの README を参照してください。
