"""Measure rejected dark fragments near new thin cores without changing masks."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_thin_margin_components as prior
from build_npj_v6_comparison import dump, read, sha
from package_paper_blending_review import check_bundle

ROOT = prior.ROOT
OUT = ROOT/'tmp_work/thin_core_fragment_diagnostic_20261001'


def main():
    check_bundle(prior.OUT); OUT.mkdir(parents=True, exist_ok=True)
    inputs = {}; rows = []
    for source in prior.prior.MARGIN.SOURCES+[prior.new.INPUT]:
        for pg in json.loads((source/'baseline_review.json').read_text())['cases']:
            key = pg['page']; folder = prior.OUT/'pages'/key
            core = read(folder/'new_core.png', True)
            if not core.any(): rows.append(dict(page=key, fragments=[], new_core_pixels=0)); continue
            original = read(ROOT/pg['original']); inputs[pg['original']] = sha(ROOT/pg['original'])
            v6 = {k: read(source/'v6_masks'/key/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
            features, _ = prior.prior.MARGIN.load_features(key, original, v6)
            _, maps, notes = prior.prior.MARGIN.infer(original, v6, features)
            support = maps['dark_support']; margin = maps['top_margin']
            n, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=8)
            from_core = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            peak_map = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
            distance = np.full(n, np.inf); peak = np.zeros(n, np.float32)
            np.minimum.at(distance, labels.ravel(), from_core.ravel()); np.maximum.at(peak, labels.ravel(), peak_map.ravel())
            seeds = np.bincount(labels.ravel(), weights=(features['color_seed']&margin).ravel(), minlength=n)
            touched = np.bincount(labels.ravel(), weights=core.ravel(), minlength=n) > 0
            relaxed = read(folder/'relaxed_core.png', True)
            accepted = np.bincount(labels.ravel(), weights=relaxed.ravel(), minlength=n) > 0
            guard = read(folder/'guard.png', True)
            guarded = np.bincount(labels.ravel(), weights=guard.ravel(), minlength=n)
            components = []
            for i in range(1, n):
                if distance[i] > 24 or accepted[i] or stats[i, 4] < 4: continue
                components.append(dict(id=i, xywh=stats[i, :4].tolist(), area=int(stats[i, 4]),
                    brown_seeds=int(seeds[i]), peak=float(peak[i]), distance_to_new_core=float(distance[i]),
                    guarded_pixels=int(guarded[i]), reason='fails at least one current thin-core gate'))
            rows.append(dict(page=key, new_core_pixels=int(core.sum()), fragments=components))
            print('Nearby fragments:', key, len(components), flush=True)
            del original, v6, features, support, margin, labels, from_core, peak_map; gc.collect()
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    dump(OUT/'diagnostics.json', rows)
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), inputs=inputs,
        frozen_mask_manifest_sha256=sha(prior.OUT/'manifest.json'), mask_changes=False,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))


if __name__ == '__main__': main()
