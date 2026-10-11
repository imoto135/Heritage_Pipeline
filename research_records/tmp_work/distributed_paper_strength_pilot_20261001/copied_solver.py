def solve_screened(lab, known, applied, sigma, screening_strength):
    mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
    guide = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), sigma)/np.maximum(mass[:, :, None], 1e-6)
    smooth = prior.reference_lowpass(lab, known)
    strength = np.where(mass >= .05, mass*screening_strength, 0).astype(np.float32)*applied
    fields = {name: np.zeros_like(lab) for name in ['boundary_raw', 'boundary_smooth']}
    used = np.zeros_like(known); notes = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(applied.astype(np.uint8), connectivity=4)
    H, W = applied.shape
    for k in range(1, n):
        component = labels == k; yy, xx = np.nonzero(component); N = len(yy)
        index = np.full(applied.shape, -1, np.int32); index[yy, xx] = np.arange(N)
        coefficient = strength[yy, xx].astype(np.float64)
        diagonal = coefficient.copy()
        rhs = coefficient[:, None]*np.tile(guide[yy, xx], (1, 2))
        rows = []; cols = []; values = []; pairs = 0
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = yy+dy, xx+dx; valid = (ny >= 0)&(ny < H)&(nx >= 0)&(nx < W)
            ids = np.flatnonzero(valid); ny, nx = ny[valid], nx[valid]
            inside = component[ny, nx]; ii = ids[inside]; diagonal[ii] += 1
            rows.extend(ii.tolist()); cols.extend(index[ny[inside], nx[inside]].tolist()); values.extend([-1.]*len(ii))
            boundary = known[ny, nx]; ii = ids[boundary]; diagonal[ii] += 1
            rhs[ii] += np.concatenate([lab[ny[boundary], nx[boundary]], smooth[ny[boundary], nx[boundary]]], axis=1)
            used[ny[boundary], nx[boundary]] = True; pairs += len(ii)
        assert pairs >= 8
        rows.extend(range(N)); cols.extend(range(N)); values.extend(diagonal.tolist())
        matrix = sparse.coo_matrix((values, (rows, cols)), shape=(N, N)).tocsr()
        solution = spsolve(matrix, rhs); residual = float(np.abs(matrix@solution-rhs).max())
        assert np.isfinite(solution).all() and residual < 1e-6
        for j, name in enumerate(fields): fields[name][yy, xx] = solution[:, j*3:j*3+3].astype(np.float32)
        notes.append(dict(id=k, xywh=stats[k, :4].tolist(), area=N, pairs=pairs, residual=residual,
            guide_L_min=float(guide[component, 0].min()), guide_L_max=float(guide[component, 0].max()),
            mass_min=float(mass[component].min()), mass_max=float(mass[component].max())))
    return fields, guide, mass, strength, used, notes
