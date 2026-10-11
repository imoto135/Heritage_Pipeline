# 画像データの場所

| ディレクトリ | 内容 |
| --- | --- |
| `split_dataset/<split>/<book>/<page>.jpg` | 原画像 |
| `split_dataset_csc/<split>/<book>/<page>.jpg` | ACEで色・コントラストを処理した分類用画像 |
| `split_dataset_csc_mask/<split>/<book>/<page>/` | **全資料に適用した旧npj（v1）の分類出力**。6,072ページ・42冊 |
| `split_dataset_csc_mask_v2/` | npj v2の出力。旧npjとの比較基準に混ぜない |
| `makura_csc_mask/` | 枕草子の別処理（v3）。旧npjの全量出力とは別 |

`split` は train / val / test。既存スクリプトから参照されるため、ディレクトリ名は変更していない。

旧npjの失敗収集と固定v6の比較は `../tmp_work/npj_failure_collection_20261001/` に保存する。`page_inventory.csv` が全量出力の索引。イラストを含むページは主評価から除外する。
