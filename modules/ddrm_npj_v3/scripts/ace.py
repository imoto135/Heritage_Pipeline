"""
Automatic Color Equalization (Rizzi et al., 2003) の実装。

照明ムラ・黄ばみによる色の偏りを補正し、文字/背景のクラスタリング分離性を上げる
前処理として使う（npj論文 Methods: "Automatic Color Equalization (ACE) is applied
to the input image to normalize the color distribution"）。

アルゴリズム（Rizzi 2003 / IPOL 2012 Getreuer）:
    R_c(x) = Σ_{y≠x} [ r_alpha(I_c(x) - I_c(y)) / dist(x,y) ]
              / Σ_{y≠x} [ 1 / dist(x,y) ]
    L_c(x) = stretch(R_c(x))   ← チャンネル独立で [min,max] → [0,1]

    r_alpha(t) = clip(alpha * t, -1, 1)   ← slope関数
    dist(x,y)  = Euclidean distance (ピクセル座標間)

高速化: 全画素対ではなく N_samples 個のランダムサンプル画素を参照画素として使う
（IPOL 2012 の近似手法）。元解像度のまま計算するのでアップサンプルによるブラーがない。
ACE出力をそのまま最終画像として返す（ゲインマップ経由にしない）。
"""

import numpy as np
import cv2


def automatic_color_equalization(
    image: np.ndarray,
    alpha: float = 5.0,
    n_samples: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """
    ACEによる色補正を行い、補正後のBGR uint8画像を返す。

    全画素対全画素の O(N²) 計算を N_samples 個のランダム参照画素で近似する。
    元解像度のまま計算するのでダウンサンプルによるブラーが生じない。

    Args:
        image: BGR uint8画像 (H, W, 3)
        alpha: slope関数の傾き（論文推奨 3〜8、デフォルト 5.0）
        n_samples: 参照画素のランダムサンプル数（多いほど精度↑・速度↓）
        seed: 乱数シード
    Returns:
        補正後のBGR uint8画像 (H, W, 3)
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    img_f = image.astype(np.float32) / 255.0  # [0,1], (H,W,3)

    # 全画素の座標
    all_y = np.arange(h).repeat(w)          # (N,)
    all_x = np.tile(np.arange(w), h)        # (N,)
    N = h * w

    # ランダムサンプル画素のインデックス
    sample_idx = rng.integers(0, N, size=n_samples)
    sy = all_y[sample_idx].astype(np.float32)  # (S,)
    sx = all_x[sample_idx].astype(np.float32)  # (S,)

    # 全画素とサンプル画素間のユークリッド距離 (N, S)
    dy = all_y[:, None].astype(np.float32) - sy[None, :]  # (N,S)
    dx = all_x[:, None].astype(np.float32) - sx[None, :]  # (N,S)
    dist = np.sqrt(dy**2 + dx**2)
    dist = np.maximum(dist, 1.0)  # 自分自身 (dist=0) を 1 に置き換え
    inv_dist = 1.0 / dist  # (N,S)

    R = np.zeros((N, 3), dtype=np.float32)
    denom = inv_dist.sum(axis=1)  # (N,)  ← サンプル画素への 1/dist 和

    for c in range(3):
        flat = img_f[:, :, c].reshape(-1)          # (N,)
        s_flat = flat[sample_idx]                  # (S,)  サンプル画素の値
        diff = flat[:, None] - s_flat[None, :]     # (N,S): I(x) - I(y)
        clipped = np.clip(alpha * diff, -1.0, 1.0) # slope r_alpha
        R[:, c] = (clipped * inv_dist).sum(axis=1) / denom

    R = R.reshape(h, w, 3)

    # 3チャンネル共通レンジで [min, max] → [0, 1] に線形伸張。
    # チャンネル独立にすると過剰なコントラスト強調が起きて汚れ・裏写りが強調される。
    lo, hi = R.min(), R.max()
    if hi - lo > 1e-6:
        L = (R - lo) / (hi - lo)
    else:
        L = img_f.copy()

    return (np.clip(L, 0.0, 1.0) * 255).astype(np.uint8)


def automatic_color_equalization_torch(
    image: np.ndarray,
    alpha: float = 5.0,
    n_samples: int = 200,
    seed: int = 0,
    device: str = 'cuda',
    chunk_size: int = 500_000,
) -> np.ndarray:
    """
    automatic_color_equalization のGPU(torch)版。アルゴリズムはCPU版と同一だが、
    (N,S)の距離・差分行列を行方向にチャンク分割してGPU上で計算する。CPU版は
    1ページ(3216x2396)あたりピークRSS約46GB・64秒かかるが、GPU版はVRAM数GB・
    数秒で完了する(RTX A6000で実測)。

    サンプル画素の選び方(np.random.default_rng(seed))はCPU版と共通だが、
    浮動小数点演算順序がGPU側で異なるためビット単位の再現性はない。
    train/val/testの既存確定データセットの再生成には使わず、CPU版
    (automatic_color_equalization)を使い続けること。新規データ(枕草子など)の
    処理にのみ使う。
    """
    import torch

    h, w = image.shape[:2]
    N = h * w
    dev = torch.device(device)

    img_f = image.astype(np.float32) / 255.0
    flat = torch.from_numpy(img_f.reshape(-1, 3)).to(dev)  # (N,3)

    all_y = np.arange(h, dtype=np.float32).repeat(w)
    all_x = np.tile(np.arange(w, dtype=np.float32), h)

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, N, size=n_samples)
    sample_idx_t = torch.from_numpy(sample_idx).to(dev)

    sy = torch.from_numpy(all_y[sample_idx]).to(dev)  # (S,)
    sx = torch.from_numpy(all_x[sample_idx]).to(dev)  # (S,)
    s_flat = flat[sample_idx_t]                        # (S,3)

    all_y_t = torch.from_numpy(all_y).to(dev)
    all_x_t = torch.from_numpy(all_x).to(dev)

    R = torch.empty((N, 3), dtype=torch.float32, device=dev)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dy = all_y_t[start:end, None] - sy[None, :]
        dx = all_x_t[start:end, None] - sx[None, :]
        dist = torch.sqrt(dy * dy + dx * dx).clamp_(min=1.0)
        inv_dist = 1.0 / dist          # (chunk,S)
        denom = inv_dist.sum(dim=1)    # (chunk,)

        chunk_flat = flat[start:end]                        # (chunk,3)
        diff = chunk_flat[:, None, :] - s_flat[None, :, :]   # (chunk,S,3)
        clipped = torch.clamp(alpha * diff, -1.0, 1.0)
        weighted = clipped * inv_dist[:, :, None]
        R[start:end] = weighted.sum(dim=1) / denom[:, None]

    # 3チャンネル共通レンジで [min, max] → [0, 1] に線形伸張(CPU版と同じ)
    lo = R.min()
    hi = R.max()
    if (hi - lo).item() > 1e-6:
        L = (R - lo) / (hi - lo)
    else:
        L = flat

    out = (L.clamp(0.0, 1.0) * 255.0).to(torch.uint8).reshape(h, w, 3)
    return out.cpu().numpy()
