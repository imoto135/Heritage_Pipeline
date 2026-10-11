# 2026-10-11 整理後の研究再開

元画像、外部の原資料、注釈、学習済みモデル、実行コード、設定、研究記録と報告書から参照する比較画像を残す。大量のACE/CSC出力、推論の中間配列、計算入力、生ログ、参照されていない途中チェックポイントは削除対象とする。

原画像と学習済みモデルはこのサーバー上に残す。Gitへpushする対象には含めていないため、別媒体へのバックアップではプロジェクトフォルダ全体をコピーする。

## 保存場所

- 元画像：`data/train`、`data/val`、`data/test`、`data/split_dataset_jogai`、`data/makura_kiridashi`。
- splitの入口：`data/split_dataset/{train,val,test}` の既存リンクを維持する。
- 使用中・再開用の重み：`modules/stain_removal/model`、`modules/stain_removal/experiments`。各系列の最終epoch500、現行コードのepoch350/400/450、学習再開例で参照するepoch100を系列別に残す。
- 小型モデル・外部資料・注釈・入力資料：`tmp_work` の元のパスに残す。
- 文書・コードの履歴・CSV・JSONの実験条件・参考PDF・報告書でリンクされる比較画像：`research_records`。元の文書・スクリプトも元のパスに残す。
- 保存ファイルのハッシュ：`research_records/preservation_manifest.json`。
- 削除対象と保護対象のファイル一覧：`maintenance/cleanup_plan_20261011.json`。
- 実行結果：整理後に `maintenance/cleanup_result_20261011.json` へ記録する。

## マスクを再生成する

プロジェクト直下から、既存の `diffusion` 環境で必要なsplitだけ生成する。以下の処理は画像を書き込み、実行時間と保存容量を要する。

```bash
conda run -n diffusion python scripts/generate_csc_masks.py --split test
conda run -n diffusion python scripts/generate_csc_masks_v2.py --split test --workers 4
conda run -n diffusion python scripts/generate_makura_csc_masks.py --workers 4
```

train/valが必要なら `--split train` / `--split val` で実行する。生成スクリプトはACE処理画像も原画像から作成する。

CSCコードには未コミットだった修正も今回の保存コミットに含める。旧実験の条件・ソーススナップショットも記録に残すが、現在のコードから過去の出力画像が完全一致で再生成されるとは限らない。過去の全量出力を再利用するスクリプトは、対象を再生成してから実行する。

## Gitから戻す場合

Gitには大型の元画像とモデルを含めていない。別途保存した画像・モデルを上記のパスへ戻し、環境を用意する。ローカルに残したモデルの一覧は削除計画の `protected` に記録している。
