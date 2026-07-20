import torch
from tqdm import tqdm


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def ddnm_steps(x, seq, model, b, H_funcs, y_0, sigma_0, device, eta=0.85, mask=None,
               mask_timesteps=0, final_correction=True,
               missing_pull=0.0, pull_target=None):
    """
    DDNM逆拡散ループ。

    Args:
        x:       初期ノイズ (B, C, H, W)
        seq:     タイムステップ列
        model:   DDPMモデル
        b:       betaスケジュール
        H_funcs: Inpaintingオペレータ
        y_0:     背景参照画像テンソル (B, C, H, W)
        sigma_0: 観測ノイズ水準
        device:  デバイス
        eta:     ノイズスケール（0=決定論的）
        mask:    (B, C, H, W) float、背景=1 文字=0（DDRMと同方向）
                 Noneのときxt*maskをスキップ
        mask_timesteps: t >= この値のステップだけ xt*mask を適用（DDRMのmask_timesteps=50相当）。
                 終盤(t < mask_timesteps)はマスクなしでモデルに仕上げをさせる
        final_correction: Falseで最終出力を未補正のx0予測にする（DDRMの最終ステップと同じ）。
                 背景ピン留めによる継ぎ目・文字境界のアーティファクトをモデルが均して仕上げる
        missing_pull: missing側のx0予測を pull_target にブレンドする強さ(0〜1)。
                 DDNMのrange-space correctionはkept側しか補正しないため、missing側
                 (汚れ領域)はモデルのx0予測をそのまま出すだけで「除去」ではなく
                 「別の色への置換」にしかならない。周辺背景色への収束をソフトに
                 促すことで、汚れを実際に馴染ませる（本家DDRMのetaB相当の効果）。
        pull_target: (B, C, H, W)。missing_pull>0のとき収束させる先の色マップ
                 （周辺背景をinpaintで滑らかに補間したもの）。Noneならy_0を使う。

    Returns:
        xs, x0_preds
    """
    if missing_pull > 0 and pull_target is None:
        pull_target = y_0
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []

        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t      = (torch.ones(n) * i).to(device)
            next_t = (torch.ones(n) * j).to(device)
            at      = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            xt = xs[-1].to(device)

            # DDRM (denoising.py line94) と同じ: 状態そのものをマスクして置き換える。
            # モデル入力だけマスクすると et(マスク済み入力への予測) と xt(非マスク) が
            # 不整合になり x0_hat の文字次元がノイズ化する
            if mask is not None and i >= mask_timesteps:
                xt = xt * mask

            et = model(xt, t)
            if et.size(1) == 6:
                et = et[:, :3]

            # DDRM同様クランプしない: マスク相では文字次元の x0 = -et·√((1-at)/at) が
            # 負方向（墨方向）に増幅される。クランプするとこの効果が消えて文字が
            # 中間グレーに留まる（濃い文字再生成の源泉）
            x0_hat = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # DDNM range-space correction:
            # x0_hat_corrected = x0_hat + H†(H*y_0 - H*x0_hat)
            # inpaintingではH*x = kept ピクセルの抽出
            H_x0_hat = H_funcs.H(x0_hat)
            H_y0     = H_funcs.H(y_0)
            residual = H_y0 - H_x0_hat  # (B, kept)

            # kept_indices はpixel-majorフラット空間のインデックス
            # H(vec) は (B,C,H,W) → permute → pixel-major で kept 抽出
            # 逆変換: pixel-major フラットに戻してから (B,C,H,W) へ
            B, C, pH, pW = x0_hat.shape
            res_pm = torch.zeros(B, C * pH * pW, device=device)  # pixel-major フラット
            res_pm[:, H_funcs.kept_indices] = residual
            # pixel-major (B, H*W, C) → channel-first (B, C, H, W)
            correction = res_pm.reshape(B, pH * pW, C).permute(0, 2, 1).reshape(B, C, pH, pW)
            x0_hat_corrected = x0_hat + correction

            if missing_pull > 0:
                # missing側だけ pull_target 方向にソフトブレンド。kept側はcorrectionで
                # 既にy_0に一致しているため、missing_maskで領域を絞って副作用を防ぐ
                missing_mask_pm = torch.ones(B, C * pH * pW, device=device)
                missing_mask_pm[:, H_funcs.kept_indices] = 0
                missing_mask = missing_mask_pm.reshape(B, pH * pW, C).permute(0, 2, 1).reshape(B, C, pH, pW)
                pulled = (1 - missing_pull) * x0_hat_corrected + missing_pull * pull_target
                x0_hat_corrected = torch.where(missing_mask > 0.5, pulled, x0_hat_corrected)

            x0_preds.append(x0_hat_corrected)

            if j < 0:
                xs.append(x0_hat_corrected if final_correction else x0_hat)
                break

            sigma_next = eta * ((1 - at_next) / (1 - at)).sqrt() * (1 - at / at_next).sqrt()
            c1 = at_next.sqrt()
            c2 = (1 - at_next - sigma_next ** 2).clamp(0).sqrt()

            direction = (xt - at.sqrt() * x0_hat_corrected) / (1 - at).sqrt()
            xt_next = c1 * x0_hat_corrected + c2 * direction + sigma_next * torch.randn_like(xt)
            xs.append(xt_next)

    return xs, x0_preds
