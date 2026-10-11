"""Continue only nonconverged v4 fits, retaining the same scientific objective."""
from pathlib import Path
import json
import shutil
import time
import os
import numpy as np
from scipy.optimize import minimize
import probe_competing_stain_decomposition as pilot

base = pilot.base
ROOT, SOURCE = pilot.ROOT, pilot.OUT
OUT = ROOT/'tmp_work/ink_stain_competition_v4_refined_20260930'
os.environ['MPLCONFIGDIR'] = str(OUT/'plot_cache')


def continue_solve(p, a, c, limit=4000):
    a, c = a.copy(), c.copy()
    count = a.size
    seed = np.concatenate([a.ravel(), c.ravel()])
    def objective(value):
        loss, ga, gc = p.evaluate(value[:count].reshape(a.shape), value[count:].reshape(c.shape), True)
        return loss, np.concatenate([ga.ravel(), gc.ravel()])
    initial = objective(seed)[0]
    history = [dict(iteration=0, objective=initial)]
    accepted = 0
    def callback(value):
        nonlocal accepted
        accepted += 1
        if accepted % 25 == 0:
            history.append(dict(iteration=accepted, objective=objective(value)[0]))
    result = minimize(objective, seed, jac=True, method='L-BFGS-B',
        bounds=[(0., None)]*len(seed), callback=callback,
        options=dict(gtol=pilot.SETTINGS['kkt_relative_tolerance']*p.norm,
                     ftol=1e-15, maxiter=limit, maxfun=16000, maxls=40, maxcor=10))
    final, gradient = objective(result.x)
    kkt = float(np.max(np.abs(np.where(result.x > 1e-8, gradient, np.minimum(gradient, 0.))))/p.norm)
    history.append(dict(iteration=int(result.nit), objective=final, relative_kkt=kkt))
    assert np.isfinite(result.x).all() and (result.x >= 0).all()
    assert final <= initial+1e-8*max(1., abs(initial))
    assert all(v['objective'] <= u['objective']+1e-8*max(1.,abs(u['objective'])) for u,v in zip(history,history[1:]))
    a = result.x[:count].reshape(a.shape); c = result.x[count:].reshape(c.shape)
    _, b, stain = p.evaluate(a, c)
    residual = p.y-b-(a+stain)[..., None]*p.q
    return b, a, stain, c, dict(converged=kkt <= pilot.SETTINGS['kkt_relative_tolerance'],
        iterations=int(result.nit), relative_kkt=kkt, objective=final,
        residual_rmse=float(np.sqrt(np.mean(residual**2))), history=history,
        initial_objective=initial, solver='L-BFGS-B', solver_success=bool(result.success),
        solver_message=str(result.message), function_evaluations=int(result.nfev))


def main():
    if (OUT/'complete.json').exists():
        raise RuntimeError('Completed continuation is frozen')
    frozen = json.loads((SOURCE/'complete.json').read_text())
    for name, expected in frozen['outputs'].items():
        assert base.sha(SOURCE/name) == expected, name
    primary = json.loads((SOURCE/'manifest.json').read_text())
    for name, expected in primary['dependencies'].items():
        assert base.sha(ROOT/name) == expected, name
    assert base.sha(Path(pilot.__file__)) == primary['source_sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    deps = {str(p.relative_to(ROOT)): base.sha(p) for p in [SOURCE/'complete.json', Path(pilot.__file__), Path(pilot.spatial.__file__)]}
    manifest = dict(settings=primary['settings'], source_sha256=base.sha(Path(__file__)), dependencies=deps,
        primary_source='tmp_work/ink_stain_competition_v4_20260930',
        reason='Some fits reached iteration cap; continue saved A and coefficients without changing objective or tolerance',
        max_additional_iterations=4000, solver="L-BFGS-B; same projected-gradient tolerance", warm_start='saved float32 A and coefficients',
        protocol_saved_before_continuations=True)
    base.dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    for page in sorted({r['page'] for r in json.loads((SOURCE/'diagnostics.json').read_text())}):
        shutil.copytree(SOURCE/page, OUT/page, dirs_exist_ok=True)
    shutil.copyfile(SOURCE/'index.html', OUT/'index.html')
    shutil.copyfile(SOURCE/'numerical_checks.json', OUT/'primary_numerical_checks.json')
    rows = json.loads((SOURCE/'diagnostics.json').read_text())
    continued = 0
    for row in rows:
        page, roi = row['page'], row['roi']; x, y, w, h = row['xywh']
        folder = OUT/page/roi
        reference = pilot.native(np.load(folder/'no_stain.npz')['ink'], w, h)
        original = base.read(folder/'original.png')
        edit = base.read(base.V6/page/'split_direction_edit_allowed.png', True)[y:y+h,x:x+w]
        for name, kernels, cost in pilot.MODELS:
            old_info = row['fits'][name]
            if old_info['converged']:
                continue
            start = time.monotonic()
            fit = np.load(folder/(name+'.npz'))
            p = pilot.Problem(fit['observation'].astype(float), fit['q'], kernels, cost)
            b, a, stain, coeff, info = continue_solve(p, fit['ink'].astype(float), fit['coefficients'].astype(float))
            ink, dirt = pilot.native(a, w, h), pilot.native(stain, w, h)
            old_ink = pilot.native(fit['ink'], w, h)
            info.update(ink_sum=float(ink.sum()), stain_sum=float(dirt.sum()),
                ink_density_mass_ratio_to_no_stain=float(ink.sum()/max(reference.sum(), 1e-12)),
                ink_above_gate=int((ink >= .015).sum()), ink_above_gate_on_v6_edit=int(((ink >= .015)&edit).sum()),
                lost_above_gate_from_no_stain=int(((reference >= .015)&(ink < .015)).sum()),
                old_solver_relative_l1=None, seconds=time.monotonic()-start,
                primary_iterations=old_info['iterations'], primary_relative_kkt=old_info['relative_kkt'],
                relative_l1_change_from_primary=float(np.abs(ink-old_ink).sum()/max(old_ink.sum(), 1e-12)),
                total_iterations=old_info['iterations']+info['iterations'])
            np.savez_compressed(folder/(name+'.npz'), background=b.astype(np.float32), ink=a.astype(np.float32),
                stain=stain.astype(np.float32), coefficients=coeff.astype(np.float32), q=p.q,
                observation=p.y.astype(np.float32))
            for label, arr in [('ink', a), ('stain', stain)]:
                display, _ = pilot.spatial.heatmap(arr, (w+96,h+96))
                base.write(folder/(name+'_'+label+'.png'), display[48:48+h,48:48+w])
            overlay = original.copy(); kept = ink >= .015; lost = (reference >= .015)&~kept
            overlay[kept] = np.rint(.6*original[kept]+.4*np.array([40,180,40])).astype(np.uint8)
            overlay[lost] = np.rint(.6*original[lost]+.4*np.array([200,60,220])).astype(np.uint8)
            base.write(folder/(name+'_change.png'), overlay)
            row['fits'][name] = info; continued += 1
            base.dump(folder/'diagnostics.json', row['fits'])
            print(page, roi, name, 'additional iterations', info['iterations'], 'kkt', round(info['relative_kkt'],7),
                'converged', info['converged'], 'ink mass ratio', round(info['ink_density_mass_ratio_to_no_stain'],3), flush=True)
        base.dump(OUT/'progress.json', dict(continued_fits=continued, last_roi=roi))
    base.dump(OUT/'diagnostics.json', rows)
    for name, expected in frozen['outputs'].items():
        assert base.sha(SOURCE/name) == expected, name
    for name, expected in deps.items():
        assert base.sha(ROOT/name) == expected, name
    assert base.sha(Path(__file__)) == manifest['source_sha256']
    verification = dict(status='passed' if all(v['converged'] for r in rows for v in r['fits'].values()) else 'incomplete_convergence',
        all_solvers_converged=all(v['converged'] for r in rows for v in r['fits'].values()),
        fits=sum(len(r['fits']) for r in rows), continued_fits=continued,
        primary_outputs_unchanged=True, source_and_dependencies_unchanged=True,
        objective_and_tolerance_unchanged=True,
        original_crops_unchanged=all(np.array_equal(base.read(OUT/r['page']/r['roi']/'original.png'),
            base.read(SOURCE/r['page']/r['roi']/'original.png')) for r in rows),
        browser_rendering_tested=False, pixel_GT=False, neural_training=False, edit_permission_generated=False)
    assert verification['original_crops_unchanged']
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ['src','href'] and value and value != 'verification.json':
                    assert (OUT/value).is_file(), value
    Links().feed((OUT/'index.html').read_text())
    verification['local_viewer_links_checked'] = True
    base.dump(OUT/'verification.json', verification)
    files = [p for p in OUT.rglob('*') if p.is_file() and 'plot_cache' not in p.parts and p.name not in ['complete.json','run.log','progress.json']]
    base.dump(OUT/'complete.json', dict(status='complete', outputs={str(p.relative_to(OUT)):base.sha(p) for p in sorted(files)}))
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
