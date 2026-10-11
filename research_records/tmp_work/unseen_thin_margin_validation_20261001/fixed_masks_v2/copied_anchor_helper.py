def anchored_for_page(page, source, original, thin_folder):
    key = page['page']; previous = thin_folder
    base = read(previous/'edit.png', True); anchor = read(previous/'new_core.png', True)
    domain = read(previous/'domain.png', True); guard = read(previous/'guard.png', True)
    if not anchor.any(): return {r: dict(edit=base.copy(), added=np.zeros_like(base), fragment_core=np.zeros_like(base), notes=[]) for r in RADII}
    v6 = {k: read(source/'v6_masks'/key/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
    features, _ = prior.mask.prior.MARGIN.load_features(key, original, v6)
    _, maps, _ = prior.mask.prior.MARGIN.infer(original, v6, features)
    connected = maps['dark_support']; margin = maps['top_margin']
    n, labels, stats, _ = cv2.connectedComponentsWithStats(connected.astype(np.uint8), connectivity=8)
    seeds = np.bincount(labels.ravel(), weights=(features['color_seed']&margin).ravel(), minlength=n)
    dt = cv2.distanceTransform(connected.astype(np.uint8), cv2.DIST_L2, 5)
    peak = np.zeros(n, np.float32); np.maximum.at(peak, labels.ravel(), dt.ravel())
    away = cv2.distanceTransform((~anchor).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    gap = np.full(n, np.inf); np.minimum.at(gap, labels.ravel(), away.ravel())
    inner = cv2.erode(margin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    boundary = np.bincount(labels.ravel(), weights=(connected&~inner).ravel(), minlength=n)
    results = {}; H, W = base.shape
    for radius in RADII:
        selected = np.zeros(n, bool); notes = []
        for i in range(1, n):
            x, y, w, h, area = map(int, stats[i])
            valid = 16 <= area < 128 and seeds[i] >= 8 and peak[i] >= 2 and 0 < gap[i] <= radius and boundary[i] == 0 and max(w, h) <= .2*min(H, W)
            if valid:
                selected[i] = True
                notes.append(dict(id=i, xywh=[x, y, w, h], area=area, brown_seeds=int(seeds[i]), peak=float(peak[i]), distance=float(gap[i])))
        core = selected[labels]&connected
        halo = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
        added = halo&domain&~guard&(features['light_drop'] >= 2)&~base
        results[radius] = dict(edit=base|added, added=added, fragment_core=core, notes=notes)
    return results
