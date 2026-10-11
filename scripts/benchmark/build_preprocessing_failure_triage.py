"""Build side-by-side mask overlays for classifying visible failures by stage."""
import json
from pathlib import Path
import cv2
import numpy as np
from build_npj_v6_comparison import read, sha
from check_paper_margin_defect_ddrm import figure

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'tmp_work/paper_reference_setback_pilot_20261001'
OUT = ROOT / 'tmp_work/preprocessing_failure_stage_triage_20261001'
IDS = ['unseen_thin_crack', 'extra_top_branch', 'extra_second_branch',
       'hole_edge_new', 'new_margin_hole', 'extra_brown_stain']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {r['id']: r for r in json.loads((BASE / 'diagnostics.json').read_text())}
    index = []
    for rid in IDS:
        r = rows[rid]
        x, y, w, h = r['xywh']; cx, cy, cw, ch = r['context_xywh']
        s = (slice(y - cy, y - cy + h), slice(x - cx, x - cx + w))
        original = read(ROOT / r['original_source'])[y:y+h, x:x+w]
        native = BASE / 'native' / rid
        restored = read(native / 'baseline.png')
        mask_context = read(native / 'applied_context.png', True)
        applied = mask_context[s]
        assert original.shape[:2] == restored.shape[:2] == applied.shape, (rid, original.shape, restored.shape, applied.shape, r['xywh'])
        overlay = original.copy()
        colour = np.zeros_like(original); colour[:] = (40, 50, 245)
        overlay[applied] = np.rint(.48 * original[applied] + .52 * colour[applied]).astype(np.uint8)
        contours, _ = cv2.findContours(applied.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 230, 40), 1)
        write = __import__('build_npj_v6_comparison').write
        write(OUT / 'original_masks' / f'{rid}_original.png', original)
        write(OUT / 'original_masks' / f'{rid}_applied.png', applied)
        write(OUT / 'original_masks' / f'{rid}_overlay.png', overlay)
        write(OUT / 'original_masks' / f'{rid}_current.png', restored)
        figure([original, overlay, restored], ['原画像', '赤＝実際の補間範囲・緑＝境界', '現行の補修'],
               OUT / f'{rid}.png', r['title'] + ' · 失敗が生じた段階の切り分け',
               f"ROI={rid}。適用 {int(applied.sum()):,}画素。赤外に残る跡は前処理側候補、赤内の違和感は復元側候補。",
               '赤い範囲は紙面補間の実適用。損傷の正解ラベルではない。現行の補修・保留をそのまま表示。')
        index.append(dict(id=rid, page=r['page'], original_source=r['original_source'],
                          xywh=r['xywh'], context_xywh=r['context_xywh'],
                          applied_pixels=int(applied.sum()), original_sha256=sha(ROOT / r['original_source']),
                          restored_sha256=sha(native / 'baseline.png'), mask_sha256=sha(native / 'applied_context.png')))
    (OUT / 'cases.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    print('Built six original/mask/restoration failure triage sheets.')


if __name__ == '__main__':
    main()
