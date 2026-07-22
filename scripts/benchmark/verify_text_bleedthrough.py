"""
裏写り（文字の裏移り: 向かい合う/裏の紙に書かれた文字の墨が透けて見える現象）の
「隣接丁転写」仮説を検証するスクリプト。

npj論文はrecto-verso裏写り除去を「同じ丁の表裏(CRF/MRFベース)」を前提として
未解決課題に挙げているが、本プロジェクトの対象（伝統的な袋綴じ製本）では、
裏写りの発生源が同じ丁の反対面ではなく、本を閉じたときに物理的に接触する
隣接した別の丁である可能性がある。

（前身の verify_bleedthrough_hypothesis.py は「裏写り」をシミ(offset stain)の
隣接丁転写と誤って混同していたため、文字の裏写りには使えない。本スクリプトは
比較対象を文字ストロークに変更した別物。）

手法:
  1. 対象ページの「本来白紙のはずなのにCSCが非背景と判定した領域」
     (04_degradation.png、シミと薄い裏写り文字の両方を含みうる)を
     裏写り候補(ghost)として抽出する。
  2. 候補丁(隣接丁)側は実際の文字ストローク(02_text_black.png)を使う。
     文字はページ全面に分布し密度が高いため、ghost領域のbounding boxを
     パディングした局所窓だけを切り出し、その中で左右反転(紙を裏返すと鏡像)
     ＋局所的な平行移動探索でDice係数を最大化する（全ページ一括の位置合わせでは
     密度差でDiceが意味を持たないため、ghost塊ごとの局所マッチングにしている）。
  3. 「本当に隣接する丁」候補群と、十分離れた「ランダム対照」候補群のDice係数を
     比較する。ただし文字ページは列構造が規則的なため、無関係なページの文字でも
     ある程度マッチしてしまう可能性がある（対照群との比較はこの交絡を測るため）。

使い方:
    # 単一のghost候補を指定して候補ランキングを見る
    conda run -n diffusion python scripts/benchmark/verify_text_bleedthrough.py \
        --book 200010454 --page 5 --side 2 --window 2

    # 本全体を自動スキャンして「白紙+ghost」ページを検出し、集計する
    conda run -n diffusion python scripts/benchmark/verify_text_bleedthrough.py \
        --book 200010454 --auto --window 2 --save_overlays
"""

import argparse
import csv
import os

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_ROOT = os.path.join(REPO_ROOT, 'data', 'split_dataset')
MASK_ROOT = os.path.join(REPO_ROOT, 'data', 'split_dataset_csc_mask_v2')
OUT_BASE = os.path.join(REPO_ROOT, 'tmp_work', 'text_bleedthrough_verify')


def find_split(book: str) -> str:
    for split in ('test', 'val', 'train'):
        if os.path.isdir(os.path.join(MASK_ROOT, split, book)):
            return split
    raise FileNotFoundError(f'マスクが見つかりません: {book}')


def list_pages(book: str, split: str):
    d = os.path.join(MASK_ROOT, split, book)
    pages = []
    for name in os.listdir(d):
        parts = name.split('_')
        pages.append((int(parts[-2]), int(parts[-1])))
    return sorted(pages)


def load_masks(book: str, split: str, page: int, side: int):
    """Returns (degradation_bool, text_black_bool) or None。"""
    stem = f'{book}_{page:05d}_{side}'
    d = os.path.join(MASK_ROOT, split, book, stem)
    deg_p = os.path.join(d, '04_degradation.png')
    txt_p = os.path.join(d, '02_text_black.png')
    if not os.path.exists(deg_p) or not os.path.exists(txt_p):
        return None
    deg = cv2.imread(deg_p, cv2.IMREAD_GRAYSCALE)
    txt = cv2.imread(txt_p, cv2.IMREAD_GRAYSCALE)
    return deg > 127, txt < 127  # txtは反転保存(0=文字)なので<127が文字


def load_image(book: str, split: str, page: int, side: int):
    p = os.path.join(SPLIT_ROOT, split, book, f'{book}_{page:05d}_{side}.jpg')
    return cv2.imread(p)


def extract_ghost_regions(deg_mask: np.ndarray, margin_ratio: float = 0.03,
                          min_area_ratio: float = 0.00005,
                          max_area_ratio: float = 0.01,
                          close_kernel: int = 9) -> np.ndarray:
    """
    04_degradation.pngから、スキャン端(ヴィネット)を除外し、面積が妥当な範囲
    （小さすぎるノイズ／大きすぎる全面劣化を除く）の連結成分だけを残す。

    薄い裏写り文字はCSCの劣化マスク上では1px単位のスペックルとして断片化
    されていることが多い（本文の実例: 200003803_00002_2 で確認済み）ため、
    面積フィルタの前にclose_kernelでモルフォロジークロージングを行い、
    近接するスペックルを文字ストローク程度の塊にまとめてから連結成分解析する。
    これをしないと極小連結成分が大量発生し、局所マッチングが実用的な時間で
    終わらない。
    """
    H, W = deg_mask.shape
    my, mx = int(H * margin_ratio), int(W * margin_ratio)
    interior = np.zeros((H, W), dtype=bool)
    interior[my:H - my, mx:W - mx] = True

    m = (deg_mask & interior).astype(np.uint8) * 255
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = min_area_ratio * H * W
    max_area = max_area_ratio * H * W
    out = np.zeros((H, W), dtype=bool)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            out |= (labels == i)
    return out


def blob_bboxes(mask: np.ndarray, pad_px: int, min_area: int = 1):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask.astype(np.uint8)) * 255, connectivity=8)
    H, W = mask.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        x0, y0 = max(0, x - pad_px), max(0, y - pad_px)
        x1, y1 = min(W, x + w + pad_px), min(H, y + h + pad_px)
        boxes.append((x0, y0, x1, y1, area))
    return boxes


def local_shift_dice(t_crop: np.ndarray, c_crop: np.ndarray,
                     shift_range: int = 40, step: int = 4):
    """
    t_crop, c_cropは同じshapeのbool。c_cropを平行移動して最良Diceを探す。

    cv2.matchTemplate(TM_CCORR)でt_cropを±shift_range分パディングした画像に
    対しc_cropをテンプレートとして相関を取ると、範囲内の全整数シフトの
    「重なり画素数」が1回のFFT相関で一括に求まる。ブロブ数が多い（数百〜）と
    総当たりのwarpAffineループでは実用的な時間で終わらないため、この方式に
    置き換えている（stepは総当たり版の名残で現在は未使用、探索は常に密）。
    分母(t_crop.sum()+c_crop.sum())はシフトによらず一定なので、
    交差画素数の最大化だけで良い。
    """
    if t_crop.sum() < 1 or c_crop.sum() < 1:
        return 0.0, (0, 0)
    t_f = t_crop.astype(np.float32)
    c_f = c_crop.astype(np.float32)
    pad = shift_range
    t_pad = cv2.copyMakeBorder(t_f, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    res = cv2.matchTemplate(t_pad, c_f, cv2.TM_CCORR)
    row, col = np.unravel_index(np.argmax(res), res.shape)
    max_inter = float(res[row, col])
    dy, dx = int(row - pad), int(col - pad)
    denom = t_crop.sum() + c_crop.sum()
    dice = 2 * max_inter / denom if denom > 0 else 0.0
    return dice, (dx, dy)


def match_ghost_to_text(ghost_mask: np.ndarray, cand_text_flipped: np.ndarray,
                        pad_px: int = 100, shift_range: int = 40, step: int = 4):
    """
    ghost_mask内の各連結成分について、cand_text_flipped(候補丁の文字マスクを
    左右反転・target解像度にリサイズ済み)との局所一致度を測る。
    Returns: (area加重平均dice, 各blobの詳細リスト)
    """
    boxes = blob_bboxes(ghost_mask, pad_px)
    if not boxes:
        return 0.0, []
    details = []
    total_area = 0
    weighted = 0.0
    for (x0, y0, x1, y1, area) in boxes:
        t_crop = ghost_mask[y0:y1, x0:x1]
        c_crop = cand_text_flipped[y0:y1, x0:x1]
        dice, shift = local_shift_dice(t_crop, c_crop, shift_range, step)
        details.append(dict(bbox=(x0, y0, x1, y1), area=int(area),
                            dice=dice, shift=shift))
        weighted += dice * area
        total_area += area
    return (weighted / total_area if total_area > 0 else 0.0), details


def gen_neighbor_candidates(pages, page: int, side: int, window: int):
    cands = []
    pageset = set(pages)
    for off in range(-window, window + 1):
        for cs in (1, 2):
            if off == 0 and cs == side:
                continue
            p2 = page + off
            if (p2, cs) in pageset:
                cands.append((p2, cs, off))
    return cands


def gen_control_candidates(pages, page: int, far_thresh: int, n: int, seed: int = 0):
    far = [(p, s) for (p, s) in pages if abs(p - page) > far_thresh]
    if not far:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(far), size=min(n, len(far)), replace=False)
    return [far[i] for i in idx]


def flip_resize_text(text_mask: np.ndarray, shape) -> np.ndarray:
    H, W = shape
    rs = cv2.resize(text_mask.astype(np.uint8) * 255, (W, H),
                    interpolation=cv2.INTER_NEAREST)
    return np.fliplr(rs) > 127


def analyze_one(book: str, split: str, page: int, side: int, window: int,
                far_thresh: int = 5, n_control: int = 5,
                margin_ratio: float = 0.03, min_area_ratio: float = 0.00005,
                max_area_ratio: float = 0.01, pad_px: int = 100,
                shift_range: int = 40, step: int = 4, close_kernel: int = 9):
    pages = list_pages(book, split)
    target = load_masks(book, split, page, side)
    if target is None:
        raise FileNotFoundError(f'マスクなし: {book} {page} {side}')
    deg_t, _ = target
    ghost_mask = extract_ghost_regions(deg_t, margin_ratio, min_area_ratio,
                                       max_area_ratio, close_kernel)
    shape = ghost_mask.shape

    results = []

    def score_candidate(p2, cs, off, kind):
        m = load_masks(book, split, p2, cs)
        if m is None:
            return
        _, txt_c = m
        cand_flipped = flip_resize_text(txt_c, shape)
        score, details = match_ghost_to_text(ghost_mask, cand_flipped,
                                             pad_px, shift_range, step)
        results.append(dict(page=p2, side=cs, offset=off, kind=kind,
                            dice=score, details=details, cand_flipped=cand_flipped))

    for p2, cs, off in gen_neighbor_candidates(pages, page, side, window):
        score_candidate(p2, cs, off, 'neighbor')
    for p2, cs in gen_control_candidates(pages, page, far_thresh, n_control):
        score_candidate(p2, cs, None, 'control')

    results.sort(key=lambda r: -r['dice'])
    return ghost_mask, results


def save_overlay(out_path: str, book: str, split: str, page: int, side: int,
                 ghost_mask: np.ndarray, top: dict):
    img = load_image(book, split, page, side)
    if img is None:
        return
    H, W = ghost_mask.shape
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H))
    overlay = img.copy()

    cnts_t, _ = cv2.findContours(ghost_mask.astype(np.uint8) * 255,
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts_t, -1, (0, 0, 255), 4)  # 対象ghost=赤

    # 各blobを局所シフトで最良位置に合わせた候補文字を貼り付けて可視化
    for d in top['details']:
        x0, y0, x1, y1 = d['bbox']
        dx, dy = d['shift']
        crop = top['cand_flipped'][y0:y1, x0:x1].astype(np.uint8) * 255
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(crop, M, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST)
        cnts_c, _ = cv2.findContours(shifted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_c:
            c[:, 0, 0] += x0
            c[:, 0, 1] += y0
        cv2.drawContours(overlay, cnts_c, -1, (255, 0, 0), 3)  # 候補文字(反転後)=青
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)

    label = f"p{top['page']}_{top['side']} off={top['offset']} dice={top['dice']:.3f}"
    cv2.putText(overlay, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
               (0, 255, 0), 3, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlay)


def print_ranking(book, page, side, results, topk=8):
    print(f'\n=== {book} p{page}_{side} の候補ランキング（上位{min(topk,len(results))}件）===')
    print(f"{'label':16s} {'kind':9s} {'dice':>7s} {'#blobs':>7s}")
    for r in results[:topk]:
        label = f"p{r['page']}_{r['side']}" + (f" (Δ{r['offset']:+d})" if r['offset'] is not None else '')
        print(f"{label:16s} {r['kind']:9s} {r['dice']:7.3f} {len(r['details']):7d}")


def cmd_single(args):
    split = find_split(args.book)
    ghost_mask, results = analyze_one(
        args.book, split, args.page, args.side, args.window,
        args.far_thresh, args.n_control, args.margin_ratio,
        args.min_area_ratio, args.max_area_ratio, args.pad_px,
        args.shift_range, args.step, args.close_kernel)
    print_ranking(args.book, args.page, args.side, results)
    if results:
        out_path = os.path.join(OUT_BASE, args.book,
                                f'p{args.page:05d}_{args.side}_top1.png')
        save_overlay(out_path, args.book, split, args.page, args.side,
                    ghost_mask, results[0])
        print(f'\n可視化を保存: {out_path}')


def find_candidate_ghosts(book, split, text_ratio_max, deg_ratio_min, deg_ratio_max,
                          margin_ratio, min_area_ratio, max_area_ratio, close_kernel):
    out = []
    for page, side in list_pages(book, split):
        m = load_masks(book, split, page, side)
        if m is None:
            continue
        deg, txt = m
        text_ratio = txt.mean()
        ghost = extract_ghost_regions(deg, margin_ratio, min_area_ratio,
                                      max_area_ratio, close_kernel)
        ghost_ratio = ghost.mean()
        if text_ratio <= text_ratio_max and deg_ratio_min <= ghost_ratio <= deg_ratio_max:
            out.append((page, side, text_ratio, ghost_ratio))
    return out


def cmd_auto(args):
    split = find_split(args.book)
    instances = find_candidate_ghosts(args.book, split, args.text_ratio_max,
                                      args.deg_ratio_min, args.deg_ratio_max,
                                      args.margin_ratio, args.min_area_ratio,
                                      args.max_area_ratio, args.close_kernel)
    print(f'{args.book}: 白紙+ghost候補 {len(instances)} 件を検出')

    rows = []
    neighbor_wins = 0
    offset_side_counter = {}
    for page, side, text_ratio, ghost_ratio in instances:
        ghost_mask, results = analyze_one(
            args.book, split, page, side, args.window, args.far_thresh,
            args.n_control, args.margin_ratio, args.min_area_ratio,
            args.max_area_ratio, args.pad_px, args.shift_range, args.step,
            args.close_kernel)
        if not results:
            continue
        best = results[0]
        best_neighbor = next((r for r in results if r['kind'] == 'neighbor'), None)
        best_control = next((r for r in results if r['kind'] == 'control'), None)
        margin = (best_neighbor['dice'] - best_control['dice']
                 if best_neighbor and best_control else float('nan'))

        if best['kind'] == 'neighbor':
            neighbor_wins += 1
            key = (best['offset'], best['side'])
            offset_side_counter[key] = offset_side_counter.get(key, 0) + 1

        rows.append([page, side, f'{text_ratio:.4f}', f'{ghost_ratio:.5f}',
                    best['kind'], best['page'], best['side'], best['offset'],
                    f"{best['dice']:.4f}",
                    f"{best_control['dice']:.4f}" if best_control else '',
                    f'{margin:.4f}' if margin == margin else ''])

        if args.save_overlays:
            out_path = os.path.join(OUT_BASE, args.book, 'overlays',
                                    f'p{page:05d}_{side}_top1.png')
            save_overlay(out_path, args.book, split, page, side, ghost_mask, best)

    n = len(rows)
    if n == 0:
        print('有効なインスタンスがありませんでした。')
        return
    print(f'\n=== {args.book} 集計（{n}件中、最良候補が近傍丁だったもの: {neighbor_wins}件, '
         f'{100*neighbor_wins/n:.1f}%）===')
    print('近傍丁が勝った場合の (offset, side) 分布:')
    for key, cnt in sorted(offset_side_counter.items(), key=lambda kv: -kv[1]):
        print(f'  offset={key[0]:+d}, side={key[1]}: {cnt}件')

    valid_margins = [float(r[10]) for r in rows if r[10] != '']
    if valid_margins:
        print(f'\n近傍丁 dice - ランダム対照 dice の平均差分: {np.mean(valid_margins):.4f}'
             f'（正なら近傍丁の方がランダムより一致度が高い＝仮説を支持。ただし文字ページは'
             f'列構造が規則的なので、対照群でもある程度のDiceが出ること自体は起こりうる）')

    out_csv = os.path.join(OUT_BASE, args.book, 'summary.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['page', 'side', 'text_ratio', 'ghost_ratio', 'best_kind',
                   'best_page', 'best_side', 'best_offset', 'best_dice',
                   'best_control_dice', 'margin'])
        w.writerows(rows)
    print(f'\nCSV保存: {out_csv}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--book', required=True)
    parser.add_argument('--page', type=int, help='単一ghost検証モード: 対象ページ番号')
    parser.add_argument('--side', type=int, choices=[1, 2], help='単一ghost検証モード: 対象side')
    parser.add_argument('--auto', action='store_true', help='本全体を自動スキャンして集計する')
    parser.add_argument('--window', type=int, default=2, help='前後何丁を候補にするか')
    parser.add_argument('--far_thresh', type=int, default=5,
                        help='ランダム対照に使う「十分離れている」丁数の閾値')
    parser.add_argument('--n_control', type=int, default=5, help='ランダム対照の件数')
    parser.add_argument('--text_ratio_max', type=float, default=1.0,
                        help='自動探索: 文字率の上限。裏写りは本文ページの行間にも'
                             '出る(実例: 200003803_00002_2)ため、デフォルトでは'
                             '白紙ページに限定しない')
    parser.add_argument('--deg_ratio_min', type=float, default=0.00002,
                        help='自動探索: ghostありとみなす面積比の下限')
    parser.add_argument('--deg_ratio_max', type=float, default=0.02,
                        help='自動探索: 面積比の上限')
    parser.add_argument('--save_overlays', action='store_true',
                        help='自動探索モードで各インスタンスの可視化も保存する')
    parser.add_argument('--close_kernel', type=int, default=9,
                        help='劣化マスクのスペックルを文字塊にまとめる'
                             'モルフォロジークロージングのカーネルサイズ')
    parser.add_argument('--margin_ratio', type=float, default=0.05,
                        help='マスクから除外するスキャン端/綴じ側の割合')
    parser.add_argument('--min_area_ratio', type=float, default=0.00005,
                        help='ghostとみなす連結成分の最小面積（ページ面積比）')
    parser.add_argument('--max_area_ratio', type=float, default=0.01,
                        help='ghostとみなす連結成分の最大面積（ページ面積比、大きすぎる'
                             '塊はシミ等の別要因として除外）')
    parser.add_argument('--pad_px', type=int, default=100,
                        help='ghost塊のbounding boxに追加するパディング(px)')
    parser.add_argument('--shift_range', type=int, default=40,
                        help='局所位置合わせの探索範囲(px)')
    parser.add_argument('--step', type=int, default=4,
                        help='局所位置合わせの探索刻み幅(px)')
    args = parser.parse_args()

    if args.auto:
        cmd_auto(args)
    else:
        if args.page is None or args.side is None:
            parser.error('--auto を指定しない場合は --page と --side が必須です')
        cmd_single(args)


if __name__ == '__main__':
    main()
