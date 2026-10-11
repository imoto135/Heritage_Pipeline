"""Convex ink/stain competition on frozen development patches, without GT.

The stain and ink deliberately share the same chromatic direction. Separation
therefore tests morphological assumptions rather than an oracle material color.
No density is an edit permission or calibrated probability.
"""
from __future__ import annotations
import html
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import cv2
from scipy.fft import dctn, idctn, rfft2, irfft2, next_fast_len
import probe_spatial_ink_decomposition as spatial

base = spatial.base
ROOT = base.ROOT
PRIOR = spatial.OUT
OUT = ROOT / 'tmp_work/ink_stain_competition_v4_20260930'
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')
MODELS = [
    ('no_stain', [], 0.),
    ('blobs_low', ['blob4', 'blob8'], .001),
    ('creases_low', ['crease0', 'crease45', 'crease90', 'crease135'], .001),
    ('both_low', ['blob4', 'blob8', 'crease0', 'crease45', 'crease90', 'crease135'], .001),
    ('both_high', ['blob4', 'blob8', 'crease0', 'crease45', 'crease90', 'crease135'], .004),
]
SETTINGS = dict(
    models=MODELS, observation='frozen v3 half-resolution chromatic OD patches',
    ink_direction='frozen v3 page median q, same for stain',
    background_lambda=36., ink_graph_lambda=1., ink_sparse=.004,
    stain='nonnegative convolution coefficients, normalized symmetric kernels, zero boundary',
    blob_sigma_analysis=[4., 8.], crease_sigmas_analysis=[16., 2.],
    crease_angles_degrees=[0, 45, 90, 135], kernel_radius_analysis=48,
    max_iterations=3000, kkt_relative_tolerance=5e-5,
    background_solution='exact Neumann graph inverse via orthonormal DCT, eliminated variable',
    candidate_density=.015, display_density_max=.2,
    pixel_GT=False, neural_training=False, edit_permission_generated=False,
    note='Known development patches; material labels are hypotheses, not identifications',
)


class Dictionary:
    def __init__(self, shape, names):
        self.shape = shape
        self.names = names
        self.radius = 48
        self.full = tuple(next_fast_len(n+2*self.radius) for n in shape)
        yy, xx = np.mgrid[-48:49, -48:49]
        kernels = []
        for name in names:
            if name.startswith('blob'):
                s = float(name[4:])
                k = np.exp(-.5*(xx*xx+yy*yy)/s**2)
            else:
                theta = np.deg2rad(float(name[6:]))
                along = xx*np.cos(theta)+yy*np.sin(theta)
                across = -xx*np.sin(theta)+yy*np.cos(theta)
                k = np.exp(-.5*((along/16.)**2+(across/2.)**2))
            kernels.append(k/k.sum())
        self.kernels = np.asarray(kernels)
        self.spectra = rfft2(self.kernels, s=self.full) if names else None

    def crop(self, arr):
        r = self.radius
        return arr[..., r:r+self.shape[0], r:r+self.shape[1]]

    def forward(self, c):
        if not self.names:
            return np.zeros(self.shape)
        spectrum = np.sum(rfft2(c, s=self.full)*self.spectra, axis=0)
        return self.crop(irfft2(spectrum, s=self.full))

    def adjoint(self, value):
        if not self.names:
            return np.zeros((0, *self.shape))
        return self.crop(irfft2(rfft2(value, s=self.full)*self.spectra, s=self.full))


class Problem:
    def __init__(self, y, q, names, stain_cost, geometry=True):
        self.y, self.q = y, q
        self.ws = spatial.weights(y, geometry, 'smooth')
        self.dictionary = Dictionary(y.shape[:2], names)
        self.cost = stain_cost
        h, w = y.shape[:2]
        spectrum = (2.-2.*np.cos(np.pi*np.arange(h)/h))[:, None] + \
                   (2.-2.*np.cos(np.pi*np.arange(w)/w))[None, :]
        self.inverse = 1./(1.+36.*spectrum)
        self.paper_y = np.stack([self.paper(y[..., c]) for c in range(2)], axis=2)
        self.parallel = np.sum(y*q, axis=2)
        self.norm = max(float(np.max(np.abs(y))), .01)
        # Row/column sums of each normalized nonnegative convolution <= 1.
        self.step = 1./(1.+len(names)+8.)

    def paper(self, x):
        return idctn(dctn(x, type=2, norm='ortho')*self.inverse, type=2, norm='ortho')

    def evaluate(self, a, c, gradient=False):
        stain = self.dictionary.forward(c)
        total = a+stain
        b = self.paper_y-self.paper(total)[..., None]*self.q
        u = self.y-total[..., None]*self.q
        r = b-u
        value = .5*np.sum(u*(-r))+spatial.graph_energy(a, self.ws[2], self.ws[3]) + \
                .004*np.sum(self.ws[4]*a)+self.cost*np.sum(c)
        if gradient:
            projected = np.sum(r*self.q, axis=2)
            ga = projected+spatial.graph_laplacian(a, self.ws[2], self.ws[3])+.004*self.ws[4]
            gc = self.dictionary.adjoint(projected)+self.cost
            return float(value), ga, gc
        return float(value), b, stain


def solve(problem, warm_a=None, limit=None):
    p = problem
    a = np.zeros(p.y.shape[:2]) if warm_a is None else warm_a.copy()
    c = np.zeros((len(p.dictionary.names), *a.shape))
    za, zc = a.copy(), c.copy()
    old = p.evaluate(a, c)[0]
    history = [dict(iteration=0, objective=old)]
    momentum = 1.; restarts = 0; converged = False
    for iteration in range(1, (limit or SETTINGS['max_iterations'])+1):
        _, ga, gc = p.evaluate(za, zc, True)
        na, nc = np.maximum(za-p.step*ga, 0.), np.maximum(zc-p.step*gc, 0.)
        current = p.evaluate(na, nc)[0]
        if current > old+1e-10*max(1., abs(old)):
            momentum = 1.; restarts += 1
            _, ga, gc = p.evaluate(a, c, True)
            na, nc = np.maximum(a-p.step*ga, 0.), np.maximum(c-p.step*gc, 0.)
            current = p.evaluate(na, nc)[0]
        assert current <= old+1e-8*max(1., abs(old)), (old, current)
        next_m = (1.+np.sqrt(1.+4.*momentum**2))/2.
        f = (momentum-1.)/next_m
        za, zc = na+f*(na-a), nc+f*(nc-c)
        a, c, old, momentum = na, nc, current, next_m
        if iteration % 25 == 0 or iteration == (limit or SETTINGS['max_iterations']):
            _, ga, gc = p.evaluate(a, c, True)
            ka = np.max(np.abs(np.where(a > 1e-8, ga, np.minimum(ga, 0.))))
            kc = np.max(np.abs(np.where(c > 1e-8, gc, np.minimum(gc, 0.)))) if c.size else 0.
            kkt = float(max(ka, kc)/p.norm)
            history.append(dict(iteration=iteration, objective=current, relative_kkt=kkt))
            if kkt <= SETTINGS['kkt_relative_tolerance']:
                converged = True
                break
    _, b, stain = p.evaluate(a, c)
    r = p.y-b-(a+stain)[..., None]*p.q
    return b, a, stain, c, dict(converged=converged, iterations=iteration,
        relative_kkt=kkt, objective=old, residual_rmse=float(np.sqrt(np.mean(r*r))),
        restarts=restarts, history=history)


def checks():
    rng = np.random.default_rng(20260930)
    shape = (19, 23); y = rng.uniform(.1, .5, (*shape, 2)); q = np.array([.6, .8])
    p = Problem(y, q, MODELS[3][1], .001)
    c = rng.uniform(.01, .1, (6, *shape)); a = rng.uniform(.01, .1, shape)
    test = rng.normal(size=shape)
    left = np.sum(p.dictionary.forward(c)*test)
    right = np.sum(c*p.dictionary.adjoint(test))
    assert np.isclose(left, right, atol=1e-10, rtol=1e-10)
    _, ga, gc = p.evaluate(a, c, True)
    da, dc = rng.normal(size=a.shape), rng.normal(size=c.shape)
    eps = 1e-7
    finite = (p.evaluate(a+eps*da, c+eps*dc)[0]-p.evaluate(a-eps*da, c-eps*dc)[0])/(2*eps)
    analytic = np.sum(ga*da)+np.sum(gc*dc)
    assert abs(finite-analytic)/max(1., abs(analytic)) < 1e-6
    _, b, stain = p.evaluate(a, c)
    residual = b+(a+stain)[..., None]*q-y
    background_gradient = residual+36.*spatial.graph_laplacian(b, p.ws[0], p.ws[1])
    assert np.max(np.abs(background_gradient)) < 1e-11
    direct = .5*np.sum(residual**2)+36.*spatial.graph_energy(b, p.ws[0], p.ws[1]) + \
             spatial.graph_energy(a, p.ws[2], p.ws[3])+.004*np.sum(p.ws[4]*a)+.001*c.sum()
    assert np.isclose(direct, p.evaluate(a, c)[0], atol=1e-10)
    b, aa, ss, cc, info = solve(p)
    assert info['converged'], info
    assert np.isfinite(b).all() and (aa >= 0).all() and (cc >= 0).all()
    return dict(convolution_adjoint=True, gradient_finite_difference=True,
        eliminated_background_stationary=True, eliminated_objective_matches_full=True,
        synthetic_solver_converged=True, density_nonnegative=True)


def native(a, w, h):
    return cv2.resize(a.astype(np.float32), (w+96, h+96), interpolation=cv2.INTER_LINEAR)[48:48+h, 48:48+w]


def main():
    if (OUT/'complete.json').exists():
        raise RuntimeError('Completed experiment is frozen')
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((PRIOR/'complete.json').read_text())
    for name, digest in frozen['outputs'].items():
        assert base.sha(PRIOR/name) == digest, name
    dependencies = json.loads((PRIOR/'manifest.json').read_text())['dependencies']
    dependencies = dict(dependencies)
    for p in [Path(spatial.__file__), PRIOR/'complete.json']:
        dependencies[str(p.relative_to(ROOT))] = base.sha(p)
    for name, digest in dependencies.items():
        assert base.sha(ROOT/name) == digest, name
    manifest = dict(settings=SETTINGS, source_sha256=base.sha(Path(__file__)),
        dependencies=dependencies, protocol_saved_before_outputs=True,
        prior_frozen_output_hashes_checked=len(frozen['outputs']))
    base.dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    base.dump(OUT/'numerical_checks.json', checks())
    print('Numerical checks passed', flush=True)
    rows = []; cards = []
    for previous in json.loads((PRIOR/'diagnostics.json').read_text()):
        page, roi = previous['page'], previous['roi']
        x, y0, w, h = previous['xywh']
        source = PRIOR/page/roi; dest = OUT/page/roi; dest.mkdir(parents=True, exist_ok=True)
        prior = np.load(source/'smooth_geometry_q1.npz')
        y, q = prior['observation'].astype(float), prior['q'].astype(float)
        assert np.isclose(np.linalg.norm(q), 1.)
        original = base.read(source/'original.png'); base.write(dest/'original.png', original)
        edit = base.read(base.V6/page/'split_direction_edit_allowed.png', True)[y0:y0+h, x:x+w]
        entries = {}; reference = None
        for name, kernels, cost in MODELS:
            start = time.monotonic()
            problem = Problem(y, q, kernels, cost)
            b, a, stain, coeff, info = solve(problem, prior['density'].astype(float))
            assert np.isfinite(a).all() and np.isfinite(stain).all() and (a >= 0).all()
            np.savez_compressed(dest/(name+'.npz'), background=b.astype(np.float32),
                ink=a.astype(np.float32), stain=stain.astype(np.float32),
                coefficients=coeff.astype(np.float32), q=q, observation=y.astype(np.float32))
            for label, arr in [('ink', a), ('stain', stain)]:
                display, _ = spatial.heatmap(arr, (w+96, h+96))
                base.write(dest/(name+'_'+label+'.png'), display[48:48+h, 48:48+w])
            ink, dirt = native(a, w, h), native(stain, w, h)
            if reference is None:
                reference = ink
            prior_native = native(prior['density'], w, h)
            info.update(ink_sum=float(ink.sum()), stain_sum=float(dirt.sum()),
                ink_density_mass_ratio_to_no_stain=float(ink.sum()/max(reference.sum(), 1e-12)),
                ink_above_gate=int((ink >= .015).sum()),
                ink_above_gate_on_v6_edit=int(((ink >= .015)&edit).sum()),
                lost_above_gate_from_no_stain=int(((reference >= .015)&(ink < .015)).sum()),
                old_solver_relative_l1=float(np.abs(ink-prior_native).sum()/max(prior_native.sum(), 1e-12)) if name == 'no_stain' else None,
                seconds=time.monotonic()-start)
            overlay = original.copy()
            lost = (reference >= .015)&(ink < .015)
            kept = ink >= .015
            overlay[kept] = np.rint(.6*original[kept]+.4*np.array([40, 180, 40])).astype(np.uint8)
            overlay[lost] = np.rint(.6*original[lost]+.4*np.array([200, 60, 220])).astype(np.uint8)
            base.write(dest/(name+'_change.png'), overlay)
            entries[name] = info
            base.dump(dest/'diagnostics.json', entries)
            print(page, roi, name, 'iterations', info['iterations'], 'kkt', round(info['relative_kkt'], 7),
                'converged', info['converged'], 'ink mass ratio', round(info['ink_density_mass_ratio_to_no_stain'], 3),
                'seconds', round(info['seconds'], 1), flush=True)
        rows.append(dict(page=page, roi=roi, xywh=previous['xywh'], fits=entries,
            v6_editable_pixels=int(edit.sum()), pixel_GT=False))
        rel = str(dest.relative_to(OUT))
        figures = [('原画像', 'original.png'), ('汚れ層なし：筆記候補', 'no_stain_ink.png'),
            ('汚れ層あり：筆記候補', 'both_low_ink.png'), ('汚れ層あり：汚れ候補', 'both_low_stain.png'),
            ('成分割当の変化', 'both_low_change.png')]
        cards.append('<article><h2>'+html.escape(page+' / '+roi)+'</h2><div class="grid">'+''.join(
            '<figure><figcaption>'+label+'</figcaption><a href="'+rel+'/'+file+'"><img src="'+rel+'/'+file+'"></a></figure>'
            for label, file in figures)+'</div><p>'+''.join('<a href="'+rel+'/'+name+'_change.png">'+name+'</a> ' for name, _, _ in MODELS)+'</p></article>')
        base.dump(OUT/'progress.json', dict(status='running', completed_rois=len(rows), total_rois=5))
    base.dump(OUT/'diagnostics.json', rows)
    (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>筆記と汚れの競合分解</title><style>body{font-family:sans-serif;margin:24px;background:#faf8f3}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}figure{margin:0}img{max-width:100%}article{margin:32px 0}p{line-height:1.6}</style><h1>紙・筆記・汚れの競合分解 v4</h1><p>既知5領域、学習なし。解析1/2解像度、色成分の表示範囲0〜0.2。筆記・汚れはモデル内の仮説名で、分類結果ではありません。緑＝筆記候補寄与0.015以上、紫＝汚れ層なしでは候補だったが寄与が下がった場所。どちらも編集許可ではありません。</p><p><a href="manifest.json">条件</a> · <a href="verification.json">検証</a></p>'+''.join(cards)+'</html>')
    for name, digest in dependencies.items():
        assert base.sha(ROOT/name) == digest, name
    assert base.sha(Path(__file__)) == manifest['source_sha256']
    for name, digest in frozen['outputs'].items():
        assert base.sha(PRIOR/name) == digest, name
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ['src', 'href'] and value and value != 'verification.json':
                    assert (OUT/value).is_file(), value
    Links().feed((OUT/'index.html').read_text())
    verification = dict(status='passed' if all(v['converged'] for r in rows for v in r['fits'].values()) else 'incomplete_convergence',
        completed_rois=len(rows), fits=sum(len(r['fits']) for r in rows),
        source_and_dependencies_unchanged=True, prior_outputs_unchanged=True,
        original_crops_unchanged=all(np.array_equal(base.read(OUT/r['page']/r['roi']/'original.png'),
            base.read(PRIOR/r['page']/r['roi']/'original.png')) for r in rows),
        all_solvers_converged=all(v['converged'] for r in rows for v in r['fits'].values()),
        local_viewer_links_checked=True, browser_rendering_tested=False,
        pixel_GT=False, neural_training=False, edit_permission_generated=False)
    assert verification['original_crops_unchanged']
    base.dump(OUT/'verification.json', verification)
    files = [p for p in OUT.rglob('*') if p.is_file() and 'plot_cache' not in p.parts and p.name not in ['complete.json', 'run.log', 'progress.json']]
    base.dump(OUT/'complete.json', dict(status='complete', outputs={str(p.relative_to(OUT)): base.sha(p) for p in sorted(files)}))
    base.dump(OUT/'progress.json', verification)
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
