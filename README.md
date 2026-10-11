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

## バックアップ前の整理（2026-10-11）

研究再開用の元画像・使用中モデル・外部資料・注釈をローカルに残し、生成物と不要な途中チェックポイントを整理する。
研究記録と比較画像は [research_records](research_records/README.md)、再開とマスク再生成の手順は [maintenance/RESTART.md](maintenance/RESTART.md) を参照。
Gitには大型の元画像とモデルを含めないため、別媒体へバックアップするときは整理後のプロジェクト全体をコピーする。
