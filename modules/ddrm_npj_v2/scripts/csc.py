"""
Color Space Clustering (CSC)。

改良版2段階クラスタリング：
  Stage 1: 輝度ベース特徴量で GMM(2クラス) => 背景 / 非背景
           （輝度のみで背景と非背景を分離。文字周辺ピクセルの劣化誤分類を防ぐ）
  Stage 2: 非背景クラスのみを対象に、RGB+Lab+Luv全特徴量で GMM(3クラス)
           => 黒文字 / 赤文字 / 実劣化
           クラスタ判定: 平均L最低 -> 黒文字、平均a値最高 -> 赤文字、残り -> 劣化
           赤文字クラスタの平均a値が閾値未満なら赤文字なし文献として全て劣化 or 黒文字扱い

特徴ベクトルは RGB + CIELab + CIELuv を連結する（論文 Results 記載通り）。
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


def _filter_red_candidate_by_component(image_bgr: np.ndarray, candidate_mask: np.ndarray,
                                       support_min: float) -> np.ndarray:
    """
    赤文字候補マスク（2D bool, H×W）を連結成分ごとに絶対色基準（ddnm_npj.detect_red_seal
    と同じ実測済みルール: hue<=12|hue>=168 かつ sat>180）で判定し、支持率が閾値未満の
    成分を除外する。

    ページ全体を1つの塊として平均を取ると、本物の朱印1つが強い信号を持つだけで、
    シミ周辺などに散らばる無関係な小さい誤検出片（数百〜数千個）が一括で通過して
    しまう（本物の朱印がページ平均を押し上げるため）。連結成分単位で個別に判定する
    ことで、朱印だけを残しシミ由来のノイズ片を確実に除外する。

    Returns: 2D bool、支持率が閾値以上の成分のみTrue
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int32)
    sat = hsv[:, :, 1].astype(np.int32)
    is_red_abs = ((hue <= 12) | (hue >= 168)) & (sat > 180)

    n_labels, labels = cv2.connectedComponents(candidate_mask.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(candidate_mask, dtype=bool)
    for i in range(1, n_labels):
        comp = labels == i
        if is_red_abs[comp].mean() >= support_min:
            kept |= comp
    return kept


def _apply_morphology(mask: np.ndarray, close_k: int, open_k: int) -> np.ndarray:
    """
    クロージング（断片化ストロークを繋ぐ）→ オープニング（孤立ノイズ除去）を適用する。
    mask: uint8 0/255
    close_k: クロージングのカーネルサイズ（奇数）
    open_k:  オープニングのカーネルサイズ（奇数）
    """
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def color_space_clustering(
    image_bgr: np.ndarray,
    seed: int = 42,
    max_samples: int = 50000,
    border_margin_ratio: float = 0.04,
    red_a_min: float = 133.0,
    red_hsv_support_min: float = 0.05,  # 候補クラスタ中、絶対HSV基準で赤と判定される画素の最低割合
    morph_close_k: int = 3,       # クロージングカーネルサイズ（0で無効）
    morph_open_k: int = 1,        # オープニングカーネルサイズ（0で無効）
    black_dilate_r: int = 7,      # 黒文字マスク膨張半径（文字近傍の劣化誤検出除去、0で無効）
) -> CSCResult:
    """
    ACE補正済み画像に対してCSCを行い、4種類のマスクを返す。

    Args:
        image_bgr: ACE補正済みのBGR uint8画像
        seed: 再現性のための乱数シード
        max_samples: GMM学習時のサブサンプル数（速度対策）
        border_margin_ratio: 画像外周のこの比率分を強制的に背景扱いにする
        red_a_min: Stage2赤文字クラスタの平均a値下限（これ未満なら赤文字なし判定）
        red_hsv_support_min: 赤文字クラスタ候補のうち、絶対HSV基準（hue<=12|hue>=168 & sat>180）
                        で本物の赤と判定される画素の最低割合。red_a_minは満たすが
                        この割合に届かない場合は茶色いシミの誤検出とみなし劣化のまま残す
        morph_close_k: クロージングカーネルサイズ（断片化ストロークを繋ぐ、0で無効）
        morph_open_k: オープニングカーネルサイズ（孤立ノイズ除去、0で無効）
        black_dilate_r: 黒文字マスクを膨張させて近傍の劣化誤検出を除外する半径（0で無効）
                        red_text_maskがある領域は除外対象外として保護される
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

    # ---- Stage 1: 全画素を輝度ベース特徴量でGMM(2クラス) → 背景/非背景 ----
    luv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LUV).astype(np.float32)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    luminance_features = np.stack([
        L / 100.0,
        luv[:, :, 0] / 100.0,
        gray / 255.0,
    ], axis=-1).reshape(-1, 3)

    stage1_labels = _fit_gmm_clusters(
        luminance_features, n_clusters=2, n_components_pca=3,
        seed=seed, max_samples=max_samples,
    )

    # 平均L値が高い方が背景
    L_flat = L.reshape(-1)
    non_border = ~border_mask_flat
    cluster_mean_L = [
        L_flat[(stage1_labels == c) & non_border].mean()
        if np.any((stage1_labels == c) & non_border) else -1.0
        for c in range(2)
    ]
    background_cluster  = int(np.argmax(cluster_mean_L))
    nonbg_cluster       = int(np.argmin(cluster_mean_L))

    background_mask_flat = (stage1_labels == background_cluster) & ~border_mask_flat
    nonbg_mask_flat      = (stage1_labels == nonbg_cluster)      & ~border_mask_flat
    # ボーダーは強制的に背景扱い
    background_mask_flat = background_mask_flat | border_mask_flat

    # ---- Stage 2: 非背景クラスを全特徴量でGMM(3クラス) → 黒文字/赤文字/劣化 ----
    black_text_mask_flat     = np.zeros(h * w, dtype=bool)
    red_text_mask_flat       = np.zeros(h * w, dtype=bool)
    real_degradation_mask_flat = nonbg_mask_flat.copy()

    nonbg_idx = np.where(nonbg_mask_flat)[0]

    if nonbg_idx.size >= 3:
        feat_flat = features.reshape(-1, features.shape[-1])
        nonbg_features = feat_flat[nonbg_idx]  # 全9次元特徴量

        stage2_labels = _fit_gmm_clusters(
            nonbg_features, n_clusters=3,
            n_components_pca=min(9, nonbg_features.shape[1]),
            seed=seed, max_samples=max_samples,
        )

        A_flat = A.reshape(-1)
        A_nonbg = A_flat[nonbg_idx]
        L_nonbg = L_flat[nonbg_idx]

        cluster_mean_L_s2 = [
            L_nonbg[stage2_labels == c].mean() if np.any(stage2_labels == c) else 1e9
            for c in range(3)
        ]
        cluster_mean_a_s2 = [
            A_nonbg[stage2_labels == c].mean() if np.any(stage2_labels == c) else -1e9
            for c in range(3)
        ]

        # 平均L最低 → 黒文字
        black_cluster = int(np.argmin(cluster_mean_L_s2))
        # 残り2クラスのうち平均a値が高い方 → 赤文字候補
        remaining = [c for c in range(3) if c != black_cluster]
        red_cluster  = remaining[int(np.argmax([cluster_mean_a_s2[c] for c in remaining]))]
        deg_cluster  = remaining[int(np.argmin([cluster_mean_a_s2[c] for c in remaining]))]

        black_text_mask_flat[nonbg_idx[stage2_labels == black_cluster]] = True
        real_degradation_mask_flat[nonbg_idx[stage2_labels == black_cluster]] = False

        # 赤文字クラスタの平均a値が閾値未満 → 朱文字なし文献として劣化扱い
        if cluster_mean_a_s2[red_cluster] >= red_a_min:
            is_red = stage2_labels == red_cluster
            candidate_flat = np.zeros(h * w, dtype=bool)
            candidate_flat[nonbg_idx[is_red]] = True
            candidate_2d = candidate_flat.reshape(h, w)
            # 相対a値だけでなく絶対HSV基準でも赤の裏付けがあるかを連結成分単位で確認する。
            # 茶色いシミは劣化クラスタ内で相対的にa値が高くなりがちだが、
            # 実際の彩度は低いため、この二次ゲートで弾ける。
            kept_2d = _filter_red_candidate_by_component(image_bgr, candidate_2d, red_hsv_support_min)
            kept_flat = kept_2d.reshape(-1)
            red_text_mask_flat[kept_flat] = True
            real_degradation_mask_flat[kept_flat] = False
            # 除外された成分は相対a値は高いが実際の朱色画素がほぼ無い＝誤検出と判断し、
            # 劣化のまま残る（real_degradation_mask_flatは変えない）
        else:
            # 赤なし → 赤候補クラスも劣化のまま（real_degradation_mask_flatは変えない）
            pass

    def _to_mask(flat: np.ndarray) -> np.ndarray:
        return (flat.reshape(h, w).astype(np.uint8)) * 255

    black_text_mask = _to_mask(black_text_mask_flat)
    red_text_mask   = _to_mask(red_text_mask_flat)

    # モルフォロジー処理: 断片化ストロークを繋ぎ、孤立ノイズを除去
    black_text_mask = _apply_morphology(black_text_mask, morph_close_k, morph_open_k)
    if red_text_mask.any():
        red_text_mask = _apply_morphology(red_text_mask, morph_close_k, morph_open_k)

    degradation_mask = _to_mask(real_degradation_mask_flat)

    # モルフォロジーで拡張されたred/blackマスクとdegradationの重複を除去
    degradation_mask = cv2.bitwise_and(degradation_mask, cv2.bitwise_not(red_text_mask))
    degradation_mask = cv2.bitwise_and(degradation_mask, cv2.bitwise_not(black_text_mask))

    # 黒文字マスク近傍の劣化誤検出を除外（文字ストローク周辺の誤分類対策）
    # red_text_maskがある領域は保護して除外しない
    if black_dilate_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (black_dilate_r*2+1, black_dilate_r*2+1))
        black_dilated = cv2.dilate(black_text_mask, k)
        # 赤文字領域を保護ゾーンとして膨張させる
        if red_text_mask.any():
            red_dilated = cv2.dilate(red_text_mask, k)
            black_dilated = cv2.bitwise_and(black_dilated, cv2.bitwise_not(red_dilated))
        degradation_mask = cv2.bitwise_and(degradation_mask, cv2.bitwise_not(black_dilated))

    return CSCResult(
        background_mask=_to_mask(background_mask_flat),
        black_text_mask=black_text_mask,
        red_text_mask=red_text_mask,
        degradation_mask=degradation_mask,
    )
