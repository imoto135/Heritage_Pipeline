def distance_to(mask):
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)

def refine(original, foreground, votes, old_red, old_ink, old_stain, brown, components):
    median = cv2.medianBlur(original, 3)
    filtered_lab = cv2.cvtColor(median.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    filtered_hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    (L, A, B) = (filtered_lab[:, :, 0], filtered_lab[:, :, 1], filtered_lab[:, :, 2])
    (H, S) = (filtered_hsv[:, :, 0], filtered_hsv[:, :, 1])
    robust_seed = (A >= 20) & (A >= 0.95 * np.maximum(B, 0)) & (L >= 20) & (S >= 80) & ((H <= 14) | (H >= 174)) & old_red
    (n, labels, stats, _) = cv2.connectedComponentsWithStats(old_red.astype(np.uint8), connectivity=8)
    seed_counts = np.bincount(labels.ravel(), weights=robust_seed.ravel(), minlength=n)
    raw_light = cv2.cvtColor(original.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)[:, :, 0]
    median_light = np.zeros(n, np.float32)
    component_rows = []
    for i in range(1, n):
        (x, y, cw, ch, area) = map(int, stats[i])
        sl = (slice(y, y + ch), slice(x, x + cw))
        median_light[i] = np.median(raw_light[sl][labels[sl] == i])
        component_rows.append(dict(id=i, xywh=[x, y, cw, ch], area=area, robust_seed_pixels=int(seed_counts[i]), median_lightness=float(median_light[i])))
    seed_supported_ids = seed_counts >= 3
    seed_supported_ids[0] = False
    confirmed_ids = seed_supported_ids & (median_light >= 25.0)
    confirmed_red = confirmed_ids[labels]
    seed_supported_red = seed_supported_ids[labels]
    red_distance = distance_to(confirmed_red)
    minimum_distance = np.full(n, np.inf, np.float32)
    np.minimum.at(minimum_distance, labels.ravel(), red_distance.ravel())
    context_ids = ~confirmed_ids & (median_light >= 25) & (minimum_distance <= 24.0)
    context_ids[0] = False
    contextual_red = context_ids[labels]
    uncertain_color = old_red & ~confirmed_red
    for row in component_rows:
        i = row['id']
        row.update(confirmed_red=bool(confirmed_ids[i]), protected_by_red_context=bool(context_ids[i]), minimum_distance_to_confirmed_red=float(minimum_distance[i]) if np.isfinite(minimum_distance[i]) else None)
    del labels, filtered_lab, filtered_hsv, red_distance
    (n, fg_labels, fg_stats, _) = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    dense_stain_ids = np.zeros(n, bool)
    dense_hold_ids = np.zeros(n, bool)
    for component in components:
        i = component['id']
        assert fg_stats[i].tolist() == component['xywh'] + [component['area']]
        if component['disposition'] == 'stain_candidate':
            dense_stain_ids[i] = True
        else:
            dense_hold_ids[i] = True
    dense_stain = dense_stain_ids[fg_labels]
    dense_hold = dense_hold_ids[fg_labels]
    del fg_labels
    base = foreground & ~dense_stain & ~dense_hold | confirmed_red | contextual_red
    thin_extension = (votes > 0) & (distance_to(base) <= 2.0) & ~dense_stain & ~dense_hold
    protect = base | thin_extension
    stain_candidate = brown | dense_stain
    guard = distance_to(protect) <= 3.0
    hold = uncertain_color | (votes > 0) & (votes < 3) & ~protect & ~dense_stain | dense_hold | stain_candidate & guard & ~protect
    edit_allowed = stain_candidate & ~guard & ~hold
    ink_color_uncertain = protect & uncertain_color
    black = protect & ~confirmed_red & ~ink_color_uncertain
    masks = dict(red=confirmed_red, protect=protect, edit_allowed=edit_allowed, hold=hold, ink_color_uncertain=ink_color_uncertain, stain_candidate=stain_candidate, black=black, robust_seed=robust_seed, seed_supported_red=seed_supported_red, contextual_red=contextual_red, thin_extension=thin_extension, protection_guard=guard, dense_stain=dense_stain, dense_hold=dense_hold, legacy_red=old_red, legacy_ink=old_ink, legacy_stain=old_stain)
    return (masks, component_rows)
