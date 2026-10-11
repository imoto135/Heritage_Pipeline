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


def _rescue_ink_like_degradation(image_bgr: np.ndarray, degradation_mask: np.ndarray,
                                 gray_max: float, fill_ratio_max: float) -> np.ndarray:
    """
    劣化クラスタの連結成分のうち、(a)暗い(gray_max未満)かつ(b)バウンディングボックス
    充填率が低い(fill_ratio_max未満、細長い線状=文字ストロークらしい形状)ものを、
    Stage2 GMMが濃淡の連続体を誤って割ったことによる黒文字の見逃しとみなし、
    黒文字側に救済する。

    Stage2は「最も暗いクラスタ→黒文字」で判定するため、同一ページ内で筆致の
    濃淡に幅がある（かすれた/薄い筆致がある）と、薄めの本物のインクまで
    "劣化"クラスタに引きずり込まれることがある(実測で本文のみのページで
    黒文字と判定されるべき画素の45〜75%が劣化クラスタに誤分類される例を確認)。
    本物のシミ・裏写りは塊状(fill_ratio高い)か、文字ほど暗くない(gray_max以上)
    傾向があるため、この2条件で区別する(makura 200017197_00008_0で連結成分の
    面積・形状を実測して閾値を較正済み: 上位30件はfill_ratio 0.03〜0.54、
    gray平均15〜87に集中)。

    Returns: 2D bool、救済対象の画素
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        degradation_mask.astype(np.uint8), connectivity=8)
    rescued = np.zeros(degradation_mask.shape, dtype=bool)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        fill_ratio = area / max(w * h, 1)
        comp = labels == i
        if gray[comp].mean() < gray_max and fill_ratio < fill_ratio_max:
            rescued |= comp
    return rescued


def _find_deepest_valley(hist: np.ndarray, min_peak_count: int) -> tuple[int, int, int] | None:
    """
    平滑化済みヒストグラム(1D)から、最も高い2つの山(min_peak_count以上)と、
    その間で最も深い谷のビン番号を (valley_bin, peak1_bin, peak2_bin) で返す。
    山が2つ見つからなければNone。
    """
    n = len(hist)
    is_peak = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - 1), min(n, i + 2)
        if hist[i] == hist[lo:hi].max() and hist[i] >= min_peak_count:
            is_peak[i] = True
    peak_idx = np.where(is_peak)[0]
    # 隣接する山は1つにまとめる（プラトー対策）
    merged = []
    for idx in peak_idx:
        if merged and idx - merged[-1] <= 2:
            if hist[idx] > hist[merged[-1]]:
                merged[-1] = idx
        else:
            merged.append(idx)
    if len(merged) < 2:
        return None
    # 最も高い2つの山を採用し、その間の最小値のビンを谷とする
    merged.sort(key=lambda i: -hist[i])
    p1, p2 = sorted(merged[:2])
    valley_rel = int(np.argmin(hist[p1:p2 + 1]))
    return p1 + valley_rel, p1, p2


def _filter_by_proximity_to_text(candidate_mask: np.ndarray, black_text_mask: np.ndarray,
                                 max_gap: int) -> np.ndarray:
    """
    candidate_mask(2D bool)を連結成分ごとに調べ、既存のblack_text_mask(2D bool)から
    max_gap px以内に触れる成分だけを残す。孤立した(既存の黒文字から離れた)成分は
    裏写り由来の疑いが強いとみなして除外する。

    既存の黒文字を膨張(dilate)させた領域と交わる連結成分を「近い」とみなす。
    ストローク全体が1つの連結成分であれば、その一部が黒文字に触れているだけで
    成分全体（先端の薄い部分も含む）が残る。

    Returns: 2D bool
    """
    if not candidate_mask.any():
        return candidate_mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap * 2 + 1, max_gap * 2 + 1))
    near_text_zone = cv2.dilate(black_text_mask.astype(np.uint8), k) > 0

    n_labels, labels = cv2.connectedComponents(candidate_mask.astype(np.uint8), connectivity=8)
    kept = np.zeros(candidate_mask.shape, dtype=bool)
    for i in range(1, n_labels):
        comp = labels == i
        if near_text_zone[comp].any():
            kept |= comp
    return kept


def _rescue_ink_by_lightness(image_bgr: np.ndarray, degradation_mask: np.ndarray,
                             black_text_mask: np.ndarray | None = None,
                             max_gap_from_text: int = 5,
                             valley_depth_ratio_max: float = 0.35,
                             max_low_fraction: float = 0.55,
                             min_pixels: int = 2000) -> np.ndarray:
    """
    劣化クラスタ内のLab L*(明度)のヒストグラムに明確な谷(二峰性)があるかを
    直接検出し、谷より暗い側をStage2 GMMが取りこぼした黒文字とみなして救済する。

    根拠(200018243_00003_2フルページで実測, [[project_csc_rescue_color_feasibility]]の
    「追記2」参照): 正しく検出された黒文字はL*平均6.5、取りこぼされた黒文字は
    L*中央値28で独立した山を持ち、本物の劣化はL*中央値115。色(Lab a*/b*)による
    分離は弱かったが、明度による分離はほぼ無重複だった。

    単純なOtsu二値化+平均差の安全弁では、水染みページ(hnsd001_011)のように
    「単峰だが右に裾を引く」分布まで強引に2群化してしまい、水染み本体を大量に
    黒文字化する誤爆が実際に発生した(実測: 620,804px中373,733pxが新規誤爆)。
    Otsuは分散最小化するだけで二峰性の有無を検証しないため、この安全弁では
    不十分だった。そこで「2つの山の間に密度が実際に落ち込む谷があるか」を
    ヒストグラム上で直接確認する方式に変更した。谷が浅い(単峰に近い)場合は
    救済しない。

    安全弁: (a)有意な山が2つ見つからない、(b)谷の深さが両側の山の低い方の
    valley_depth_ratio_max倍を超える(谷が浅い=単峰寄り)、(c)暗い側の画素が
    過半数(max_low_fraction超)を占める、のいずれかに該当する場合は救済しない。

    追加の安全弁(2026-07-31追記): 上記の谷検出だけでは裏写りの筋の一部(暗い
    箇所)まで拾ってしまうことが200018243_00003_2フルページの実測で判明した
    （新規救済画素のうち、既存黒文字から10px以上離れた孤立画素4.7%が、目視で
    裏写りの筋に沿っていた）。`black_text_mask`を渡すと、`_filter_by_proximity_to_text`で
    既存黒文字からmax_gap_from_text px以内に触れる連結成分だけに絞り込む
    （文字ストロークの掠れた先端は既存黒文字と繋がっているのでほぼ影響を受けず、
    黒文字から孤立した裏写り由来の点だけが除外される）。

    Returns: 2D bool、救済対象の画素
    """
    deg_bool = degradation_mask > 127
    n_deg = int(deg_bool.sum())
    if n_deg < min_pixels:
        return np.zeros(degradation_mask.shape, dtype=bool)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    L_deg = L[deg_bool]

    n_bins = 40
    lo, hi = 0.0, 140.0
    hist, edges = np.histogram(L_deg, bins=n_bins, range=(lo, hi))
    kernel = np.ones(3) / 3.0
    smooth = np.convolve(hist, kernel, mode='same')

    min_peak_count = max(50, int(0.01 * n_deg))
    found = _find_deepest_valley(smooth, min_peak_count)
    if found is None:
        return np.zeros(degradation_mask.shape, dtype=bool)
    valley_bin, peak1_bin, peak2_bin = found
    peak_left, peak_right = smooth[peak1_bin], smooth[peak2_bin]
    valley_depth = smooth[valley_bin]

    if valley_depth > valley_depth_ratio_max * min(peak_left, peak_right):
        return np.zeros(degradation_mask.shape, dtype=bool)

    valley_L = edges[valley_bin + 1]  # ビンの右端をしきい値にする
    low = L_deg < valley_L
    low_fraction = float(low.mean())
    if low_fraction > max_low_fraction or low.sum() < 10:
        return np.zeros(degradation_mask.shape, dtype=bool)

    rescued = np.zeros(degradation_mask.shape, dtype=bool)
    deg_idx = np.where(deg_bool.reshape(-1))[0]
    rescued_flat = rescued.reshape(-1)
    rescued_flat[deg_idx[low]] = True

    if black_text_mask is not None and max_gap_from_text > 0:
        rescued = _filter_by_proximity_to_text(rescued, black_text_mask, max_gap_from_text)

    return rescued


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
    black_dilate_r: int = 0,      # 黒文字マスク膨張半径（文字近傍の劣化誤検出除去、0で無効）
    ink_rescue_gray_max: float = 0,   # 劣化クラスタ連結成分の平均輝度がこれ未満なら救済対象候補（0で無効。
                                      # デフォルトは無効: 水染み・裏写りの線維状パターンを黒文字に誤救済する
                                      # 副作用が2例(hnsd001_011の水染み、200018243_00003_2の裏写り)で
                                      # 再現したため。[[project_csc_v3_rescue_sideeffect]]参照
    ink_rescue_fill_ratio_max: float = 0.5,  # 同、バウンディングボックス充填率がこれ未満なら救済対象候補
    ink_rescue_by_lightness: bool = False,  # 劣化クラスタのL*ヒストグラムに谷があれば暗い群を黒文字に救済（0/Falseで無効）
    lightness_rescue_valley_ratio_max: float = 0.35,  # 谷の深さが両山の低い方の何倍を超えたら単峰とみなし救済しないか
    lightness_rescue_max_low_fraction: float = 0.55,  # 暗い群の画素比率がこれを超えたら救済しない
    lightness_rescue_max_gap_from_text: int = 5,  # 既存黒文字からこのpx以内に触れる連結成分だけ救済（孤立した裏写り点を除外、0で無効）
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
        ink_rescue_gray_max: 劣化クラスタの連結成分のうち、平均輝度がこれ未満かつ
                        ink_rescue_fill_ratio_max未満(細長い線状)の成分を、
                        Stage2が濃淡の連続体を誤って割ったことによる黒文字の
                        見逃しとみなし黒文字側に救済する（0で無効）
        ink_rescue_fill_ratio_max: 同、バウンディングボックス充填率の上限
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

    # Stage2が濃淡の連続体を誤って割ったことによる黒文字見逃しを救済
    # （暗くて細長い=文字ストロークらしい劣化クラスタ連結成分を黒文字に戻す）
    if ink_rescue_gray_max > 0:
        rescued = _rescue_ink_like_degradation(
            image_bgr, degradation_mask, ink_rescue_gray_max, ink_rescue_fill_ratio_max)
        rescued_mask = (rescued.astype(np.uint8)) * 255
        black_text_mask = cv2.bitwise_or(black_text_mask, rescued_mask)
        degradation_mask = cv2.bitwise_and(degradation_mask, cv2.bitwise_not(rescued_mask))

    # 劣化クラスタ自身のL*分布が二峰性なら、暗い方の群を黒文字見逃しとして救済
    # （形状条件を使わないため、裏写りの線状パターンを誤救済しにくい）
    if ink_rescue_by_lightness:
        rescued_l = _rescue_ink_by_lightness(
            image_bgr, degradation_mask,
            black_text_mask=black_text_mask > 127,
            max_gap_from_text=lightness_rescue_max_gap_from_text,
            valley_depth_ratio_max=lightness_rescue_valley_ratio_max,
            max_low_fraction=lightness_rescue_max_low_fraction)
        rescued_l_mask = (rescued_l.astype(np.uint8)) * 255
        black_text_mask = cv2.bitwise_or(black_text_mask, rescued_l_mask)
        degradation_mask = cv2.bitwise_and(degradation_mask, cv2.bitwise_not(rescued_l_mask))

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
