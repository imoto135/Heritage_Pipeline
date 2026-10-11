"""
Color Space Clustering (CSC)。

npj論文 Methods の記述に基づく2段階クラスタリング：
  Stage 1: 輝度ベース特徴量で PCA -> GMM(3クラス) => 背景 / 黒文字 / 劣化
  Stage 2: 劣化クラスのみ対象に、彩度重視特徴量で PCA -> GMM(2クラス)
           => 赤文字 / 実劣化（平均a値が高い方を赤文字と判定）

特徴ベクトルは RGB + CIELab + CIELuv を連結する（論文 Results 記載通り）。

クラスタ→意味の対応づけ：
  Stage1: 各クラスタの平均輝度(CIELab の L) が最も高い -> 背景
                                        最も低い -> 黒文字
                                        残り       -> 劣化
  Stage2: 各クラスタの平均 a値(CIELab の赤み成分) が高い方 -> 赤文字
                                              低い方 -> 実劣化
          ただし赤文字クラスタの平均a値が閾値未満なら赤文字なしと判定し全て劣化扱い
"""

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


@dataclass
class CSCResult:
    background_mask: np.ndarray   # uint8 0/255, 255=背景
    black_text_mask: np.ndarray   # uint8 0/255, 255=黒文字
    red_text_mask: np.ndarray     # uint8 0/255, 255=赤文字
    degradation_mask: np.ndarray  # uint8 0/255, 255=実劣化


def _to_feature_image(image_bgr: np.ndarray) -> np.ndarray:
    """RGB + CIELab + CIELuv を連結した9チャンネル特徴量画像を作る。"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    luv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LUV).astype(np.float32)

    lab_norm = lab / np.array([100.0, 127.0, 127.0], dtype=np.float32)
    luv_norm = luv / np.array([100.0, 127.0, 127.0], dtype=np.float32)

    return np.concatenate([rgb, lab_norm, luv_norm], axis=-1)  # (H, W, 9)


def _fit_gmm_clusters(features: np.ndarray, n_clusters: int, n_components_pca: int,
                      seed: int = 42, max_samples: int = 50000) -> np.ndarray:
    """
    features: (N, D) の特徴量配列
    Returns: (N,) のクラスタラベル
    """
    n_components_pca = min(n_components_pca, features.shape[1], features.shape[0])

    n = features.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        fit_features = features[idx]
    else:
        fit_features = features

    pca = PCA(n_components=n_components_pca, random_state=seed)
    pca.fit(fit_features)
    fit_reduced = pca.transform(fit_features)

    gmm = GaussianMixture(n_components=n_clusters, random_state=seed, max_iter=200)
    gmm.fit(fit_reduced)

    reduced_all = pca.transform(features)
    labels = gmm.predict(reduced_all)
    return labels


def color_space_clustering(
    image_bgr: np.ndarray,
    seed: int = 42,
    max_samples: int = 50000,
    border_margin_ratio: float = 0.04,
    red_a_min: float = 133.0,  # Stage2で赤文字クラスタの平均a値がこれ未満なら赤文字なしと判定
) -> CSCResult:
    """
    ACE補正済み画像に対してCSCを行い、4種類のマスクを返す。

    Args:
        image_bgr: ACE補正済みのBGR uint8画像
        seed: 再現性のための乱数シード
        max_samples: GMM学習時のサブサンプル数（速度対策）
        border_margin_ratio: 画像外周のこの比率分を強制的に背景扱いにする
        red_a_min: Stage2赤文字クラスタの平均a値下限（これ未満なら赤文字なし判定）
    """
    h, w = image_bgr.shape[:2]
    features = _to_feature_image(image_bgr)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1]

    # 外周ボーダーマスク（スキャン時の黒帯・製本の影対策）
    margin_h = int(h * border_margin_ratio)
    margin_w = int(w * border_margin_ratio)
    border_mask = np.zeros((h, w), dtype=bool)
    if margin_h > 0:
        border_mask[:margin_h, :] = True
        border_mask[-margin_h:, :] = True
    if margin_w > 0:
        border_mask[:, :margin_w] = True
        border_mask[:, -margin_w:] = True
    border_mask_flat = border_mask.reshape(-1)

    # ---- Stage 1: 全画素を輝度ベース特徴量でGMM(3クラス) ----
    # 輝度(L)のみの特徴量で 背景/黒文字/劣化 を分離する
    luminance_features = np.stack([
        lab[:, :, 0] / 100.0,
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LUV).astype(np.float32)[:, :, 0] / 100.0,
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
    ], axis=-1).reshape(-1, 3)

    stage1_labels = _fit_gmm_clusters(
        luminance_features, n_clusters=3, n_components_pca=3,
        seed=seed, max_samples=max_samples,
    )

    # クラスタ平均L値でラベルを意味に対応（ボーダー除外して計算）
    L_flat = L.reshape(-1)
    non_border = ~border_mask_flat
    cluster_mean_L = [
        L_flat[(stage1_labels == c) & non_border].mean()
        if np.any((stage1_labels == c) & non_border) else -1.0
        for c in range(3)
    ]
    background_cluster  = int(np.argmax(cluster_mean_L))
    black_text_cluster  = int(np.argmin(cluster_mean_L))
    degradation_cluster = [c for c in range(3)
                           if c != background_cluster and c != black_text_cluster][0]

    background_mask_flat  = (stage1_labels == background_cluster)  & ~border_mask_flat
    black_text_mask_flat  = (stage1_labels == black_text_cluster)   & ~border_mask_flat
    degradation_mask_flat = (stage1_labels == degradation_cluster)  & ~border_mask_flat
    # ボーダーは強制的に背景扱い
    background_mask_flat  = background_mask_flat | border_mask_flat

    # ---- Stage 2: 劣化クラスのみを彩度特徴でGMM(2クラス) → 赤文字/劣化 ----
    red_text_mask_flat       = np.zeros(h * w, dtype=bool)
    real_degradation_mask_flat = degradation_mask_flat.copy()

    deg_idx = np.where(degradation_mask_flat)[0]

    if deg_idx.size >= 2:
        feat_flat = features.reshape(-1, features.shape[-1])
        # 彩度重視: a,b(Lab) + u,v(Luv) の4次元
        chroma_features = feat_flat[deg_idx][:, [4, 5, 7, 8]]

        stage2_labels = _fit_gmm_clusters(
            chroma_features, n_clusters=2,
            n_components_pca=min(4, chroma_features.shape[1]),
            seed=seed, max_samples=max_samples,
        )

        A_flat = A.reshape(-1)
        A_deg = A_flat[deg_idx]
        cluster_mean_a = [
            A_deg[stage2_labels == c].mean() if np.any(stage2_labels == c) else -1e9
            for c in range(2)
        ]
        red_cluster  = int(np.argmax(cluster_mean_a))
        none_cluster = int(np.argmin(cluster_mean_a))

        # 赤文字クラスタの平均a値が閾値未満 → 朱文字なし文献として全て劣化扱い
        if cluster_mean_a[red_cluster] >= red_a_min:
            is_red = stage2_labels == red_cluster
            red_text_mask_flat[deg_idx[is_red]]   = True
            real_degradation_mask_flat[deg_idx[is_red]] = False  # 劣化から除外
        # 閾値未満の場合は real_degradation_mask_flat をそのまま維持（全て劣化）

    def _to_mask(flat: np.ndarray) -> np.ndarray:
        return (flat.reshape(h, w).astype(np.uint8)) * 255

    return CSCResult(
        background_mask=_to_mask(background_mask_flat),
        black_text_mask=_to_mask(black_text_mask_flat),
        red_text_mask=_to_mask(red_text_mask_flat),
        degradation_mask=_to_mask(real_degradation_mask_flat),
    )
