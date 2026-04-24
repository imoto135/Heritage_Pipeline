# Diffusion-based Restoration for Historical Documents

古代文献（徒然草・枕草子・源氏物語など）の画像から汚れ・にじみをDiffusionモデルで除去するシステム。

**推論専用**のリポジトリです。学習済みモデルファイル（`.pth`）を別途用意する必要があります。データセットは今後追加予定です。

---

## 必要なもの

- Python 3.8+
- PyTorch（CUDA推奨）
- 学習済みモデルファイル（別途入手）
  - `model/model_page.pth` — ページ全体修復用（3ch UNet）
  - `model/model_char.pth` — パッチ・文字修復用（3ch UNet）

### 依存ライブラリのインストール

```bash
pip install torch torchvision opencv-python pillow scikit-image tqdm matplotlib
```

---

## ディレクトリ構成

```
diffusion/
├── src/                        # コアモジュール
│   ├── models/
│   │   ├── unet/               # UNetアーキテクチャ（3ch）
│   │   └── unet_1ch/           # UNetアーキテクチャ（1ch）
│   ├── inpainting.py           # Diffusionクラス（推論メイン）
│   ├── diffusion.py            # Diffusionモデル定義
│   ├── denoising.py            # DDIM逆拡散ステップ
│   ├── random_crop.py          # ページ分割・リサイズ処理
│   ├── orig_dataset.py         # データセットクラス
│   ├── ckpt_util.py            # チェックポイント読み込み
│   └── svd_replacement.py      # SVDベースの修復補助
├── scripts/                    # 実行スクリプト
│   ├── main1.py                # 標準ページ修復
│   ├── main_high_inference.py  # パッチ分割高精度推論
│   └── postprocess_redtext.py  # 赤文字後処理（旧バージョン）
├── datasets/                   # データセット定義（追加予定）
├── model/                      # 学習済みモデル置き場（要別途配置）
│   ├── model_page.pth          # ← ここに配置
│   └── model_char.pth          # ← ここに配置
├── assets/                     # 背景画像・テスト用素材
└── results/                    # 評価結果
```

---

## 入力データの構成

各文献ページは以下のサブフォルダ構成が必要です。

```
<入力ルート>/
└── <ページID>/               # 例: 200009896_00005_2
    ├── 00_input.png          # 元の文献画像
    ├── 02_text_black.png     # 黒文字マスク（GMMなどで事前生成）
    └── 03_text_red.png       # 赤文字マスク（省略可）
```

> マスク画像は別途GMMや二値化処理で生成してください。  
> `03_text_red.png` がある場合は、赤文字領域を元画像から復元して最終出力に合成します。

---

## 使い方

### データ分割

`data/full_dataset` を train / val / test に分けるには、次のスクリプトを使います。

```bash
python scripts/split_full_dataset.py
```

既定では `data/split_dataset/` に以下の構造を作ります。

```
data/split_dataset/
├── train/
│   └── <文書ID>/
├── val/
│   └── <文書ID>/
└── test/
  └── <文書ID>/
```

既定では元データを変更せず、実体コピーで分割ビューを作ります。symlink が必要なら `--mode symlink` を指定してください。

スクリプトはプロジェクトルートから実行してください。

```bash
cd /path/to/diffusion
```

### 1. 標準ページ修復 — `scripts/main1.py`

ページ全体を `image_size`（デフォルト: 1024px）にリサイズしてDiffusionを適用します。

**事前設定（スクリプト内 `Args` クラスを編集）:**

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `image_name` | 入力フォルダのパス | `./徒然草/turezuregusa_ace_ver2` |
| `image_folder` | 出力先フォルダのパス | `./heritage_result/page_base/...` |
| `back_ground` | 背景画像パス（背景色推定用） | `./assets/image.png` |
| `timesteps` | 逆拡散ステップ数 | `20` |
| `mask_timesteps` | マスク適用開始タイムステップ | `50` |
| `eta` | DDIMノイズパラメータ（0=決定論的） | `0.5` |
| `device` | 使用GPU | `cuda:1` |

```bash
python scripts/main1.py
```

**出力:**
```
<image_folder>/
└── <ページID>/
    ├── used_mask_raw.png          # 使用したマスク
    └── result/
        ├── <ページID>.png         # 修復結果（元解像度）
        └── <ページID>_fin.png     # 赤文字復元済みの最終結果
```

---

### 2. パッチ分割高精度推論 — `scripts/main_high_inference.py`

画像を128pxのパッチに分割して推論し、フェザーブレンドで合成します。高解像度画像でも元サイズのまま処理でき、**処理済みページは自動スキップ**します。

**事前設定（スクリプト内 `Args` クラスを編集）:**

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `image_name` | 入力フォルダのパス | `./徒然草/turezuregusa_ace_ver2` |
| `image_folder` | 出力先フォルダのパス | `./heritage_result/patch_base/...` |
| `device` | 使用GPU | `cuda:0` |

```bash
python scripts/main_high_inference.py
```

---

## モデルとパラメータの目安

`src/inpainting.py` の `sample()` 内でモデルファイルを指定しています。

| モデルファイル | 用途 |
|---|---|
| `model/model_page.pth` | ページ全体をまとめて修復（`main1.py`推奨） |
| `model/model_char.pth` | パッチ分割推論（`main_high_inference.py`推奨） |

**パラメータ調整の目安:**

| パラメータ | 小さい値 | 大きい値 |
|---|---|---|
| `timesteps` | 高速・粗い（10〜20） | 低速・高品質（50〜） |
| `mask_timesteps` | マスク部分のみ修復 | 周辺領域も強めに修復（50が標準） |
| `eta` | 再現性が高い（0=決定論的） | ランダム性が増す（1） |

---

## 学習

### 3. モデルの学習 — `scripts/train_patch.py` / `scripts/train_page.py`

**汚れのないクリーン画像**でDDPMを学習します。用途に応じて以下を使い分けます。

- `scripts/train_patch.py`: 128pxパッチ推論向け（`model_char.pth`）
- `scripts/train_page.py`: ページ全体推論向け（`model_page.pth`）

#### データセット準備

```
data/train/
└── *.png  （汚れのないクリーンな文献画像）
```

#### 事前設定（スクリプト内 `TrainConfig` クラスを編集）

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `image_dir` | クリーン画像フォルダ | `./data/train` |
| `image_size` | 学習解像度（高品質: 256 or 512） | `256` |
| `batch_size` | バッチサイズ（VRAM 24GB で 256px: 4〜8） | `4` |
| `n_epochs` | 学習エポック数 | `500` |
| `save_freq` | チェックポイント保存間隔（エポック数） | `50` |
| `save_name` | 保存するモデルファイル名 | `model_page.pth` |
| `resume` | 途中再開用チェックポイントパス | `None` |
| `device` | 使用GPU | `cuda:0` |

```bash
python scripts/train_patch.py
python scripts/train_page.py
```

#### 学習と解像度の目安

| `image_size` | 品質 | VRAM目安 | 学習時間目安 |
|---|---|---|---|
| 128px | 低 | 8GB | 速い |
| 256px | 中〜高 | 16GB | 標準 |
| 512px | 高 | 24GB以上 | 遅い |

> 高品質推論（`main_high_inference.py`）は128pxパッチで推論するため、`image_size=128` で学習したモデルと相性が良いです。  
> ページ全体修復（`main1.py`、`image_size=1024`）には `image_size=256` 以上で学習したモデルを推奨します。

#### 途中から再開する場合

`TrainConfig` の `resume` にチェックポイントパスを指定してください。

```python
resume = './model/ckpt_epoch0200.pth'
```

---

## 今後の予定

- [ ] 学習用データセットの追加
- [ ] モデルの公開

---

## 参考

このシステムのDiffusionモデル部分はDDIM（Denoising Diffusion Implicit Models）をベースにしています。
