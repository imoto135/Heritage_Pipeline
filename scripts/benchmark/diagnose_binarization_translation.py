"""No-training translation-equivariance audit; never change adopted inference.

Joint circular shifts preserve all five input-channel values and local labelled
neighbourhoods, but move seams/boundaries. This does not identify position alone.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

import isolate_binarization_causes as parent
from compare_v5_aligned_fade_control import dump, sha, save
from ragan_pilot_support import auxiliaries, make_model, predict
from deterministic_reflection_support import install
from audit_ragan_inference_context import digest_state
from apply_ragan_real_regions import panel

ROOT = parent.ROOT
OUT = ROOT / 'tmp_work/binarization_translation_diagnosis_20261023'
SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
          (32, 0), (-32, 0), (0, 32), (0, -32)]


def read(p):
    return json.loads(Path(p).read_text())


def freeze():
    p = parent.check()
    assert not (OUT / 'plan.json').exists()
    files = {str(Path(__file__).relative_to(ROOT)): sha(__file__)}
    cases = []
    for c in p['context']['cases']:
        k = c['id']
        base = parent.OUT / 'inputs' / k
        a = np.load(base / 'page_page.npy')
        cx, cy = c['center_xy_in_512']
        variants = [(k + '_centered', a, cx, cy)]
        if k == 'S07':
            page = np.asarray(Image.open(ROOT / c['source']).convert('RGB'))
            x, y, _, _ = c['roi']
            variants.append(('S07_actual1920_0', auxiliaries(page[:512, 1920:2432]), x-1920, y))
        core = np.asarray(Image.open(base / 'core.png')) > 0
        paper = np.asarray(Image.open(base / 'stain.png')) > 0
        for name, inp, cx, cy in variants:
            labels = {}
            for key, small in [('core', core), ('paper', paper)]:
                yy, xx = np.nonzero(small)
                assert ((xx+cx >= 0) & (xx+cx < 512) & (yy+cy >= 0) & (yy+cy < 512)).all()
                canvas = np.zeros((512, 512), bool)
                canvas[yy+cy, xx+cx] = True
                labels[key] = canvas
            # Every labelled pixel's square radius32 neighbourhood is preserved
            # exactly, with no seam or boundary inside it, for every intervention.
            selected = labels['core'] | labels['paper']
            halo = ndi.maximum_filter(selected, size=65, mode='constant') > 0
            for dx, dy in SHIFTS:
                shifted = np.roll(inp, (dy, dx), axis=(0, 1))
                sy, sx = np.nonzero(halo)
                assert ((sx+dx >= 0) & (sx+dx < 512) & (sy+dy >= 0) & (sy+dy < 512)).all()
                assert np.array_equal(shifted[sy+dy, sx+dx], inp[sy, sx])
                assert np.array_equal(np.roll(shifted, (-dy, -dx), axis=(0, 1)), inp)
            dest = OUT / 'inputs' / name
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest / 'channels.npy', inp)
            Image.fromarray(inp[:, :, :3]).save(dest / 'original.png')
            for key, label in labels.items():
                Image.fromarray(label.astype(np.uint8)*255).save(dest / (key+'.png'))
            for f in dest.iterdir():
                files[str(f.relative_to(ROOT))] = sha(f)
            yy, xx = np.nonzero(selected)
            box = [max(0, int(xx.min())-48), max(0, int(yy.min())-48),
                   min(512, int(xx.max())+49), min(512, int(yy.max())+49)]
            cases.append(dict(id=name, source_case=k, source=c['source'],
                              core_n=int(core.sum()), paper_n=int(paper.sum()), figure_box=box))
    checkpoints = []
    for n in parent.CONDITIONS:
        cp = parent.OLD / 'runs' / n / 'step_00864.pt'
        expected = read(parent.OLD / 'evaluation' / n / 'results.json')['checkpoint_sha256']
        assert sha(cp) == expected
        checkpoints.append(dict(condition=n, path=str(cp.relative_to(ROOT)), sha256=expected))
    dump(OUT / 'plan.json', dict(
        status='Adaptive audit frozen before translation outputs; prior context/tile results known',
        files=files, checkpoints=checkpoints, cases=cases, shifts_xy=SHIFTS,
        parent_protocol_sha256=sha(parent.OUT / 'protocol.json'),
        fixed='CPU float32, deterministic, 8 threads, eval, threshold0.5, no auxiliary recomputation',
        intervention='Roll all5 channels together, predict, inverse-roll probability to original coordinates',
        checks='Exact radius32 labelled neighbourhoods and channel-value multiset preserved; identity repeat per model; unchanged state hashes',
        readout='Every shift, no best-shift selection. Exact and radius2 core, selected-paper black, pair disagreement, probability differences.',
        limits='Known training sources, provisional partial labels. Remote seam and boundary conditions move. Not pure position, layer identification, heldout accuracy or production recommendation.',
        no_training=True, no_optimizer=True, no_new_annotation=True))
    print('Frozen 108 inference conditions +4 repeat checks; no training', flush=True)


def check():
    p = read(OUT / 'plan.json')
    assert sha(parent.OUT / 'protocol.json') == p['parent_protocol_sha256']
    for f, h in p['files'].items():
        assert sha(ROOT / f) == h, f
    for cp in p['checkpoints']:
        assert sha(ROOT / cp['path']) == cp['sha256']
    return p


def forward(model, inp):
    t = torch.from_numpy(inp.copy()).permute(2, 0, 1)[None].float()/255
    with torch.inference_mode():
        a = predict(model, 'ragan', t)[0, 0].numpy().copy()
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    return a


def run():
    p = check()
    assert not (OUT / 'results.json').exists()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rows, states, figures = [], [], []
    for cp in p['checkpoints']:
        n = cp['condition']
        checkpoint = torch.load(ROOT / cp['path'], map_location='cpu', weights_only=False)
        model = make_model('ragan')
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        install(model)
        model.eval()
        before = digest_state(model)
        repeat = None
        for c in p['cases']:
            start = time.monotonic()
            k = c['id']
            base = OUT / 'inputs' / k
            a = np.load(base / 'channels.npy')
            labels = {key: np.asarray(Image.open(base / (key+'.png'))) > 0 for key in ['core', 'paper']}
            selected = labels['core'] | labels['paper']
            x0, y0, x1, y1 = c['figure_box']
            crop = np.s_[y0:y1, x0:x1]
            items = [('原画像', a[crop][:, :, :3])]
            zero = None
            for dx, dy in p['shifts_xy']:
                shifted = np.roll(a, (dy, dx), axis=(0, 1))
                prediction = np.roll(forward(model, shifted), (-dy, -dx), axis=(0, 1))
                if zero is None:
                    zero = prediction.copy()
                    if repeat is None:
                        repeated = forward(model, a)
                        repeat = bool(np.array_equal(zero, repeated))
                        assert repeat
                dest = OUT / 'predictions' / n / k
                dest.mkdir(parents=True, exist_ok=True)
                stem = f'dx{dx}_dy{dy}'
                pf, mf = dest / (stem+'.npy'), dest / (stem+'.png')
                np.save(pf, prediction)
                save(mf, prediction < .5)
                diff = (prediction < .5) != (zero < .5)
                row = dict(condition=n, case=k, shift_xy=[dx, dy],
                           **parent.score(prediction, labels, .5),
                           selected_disagreement=int((diff & selected).sum()),
                           core_disagreement=int((diff & labels['core']).sum()),
                           paper_disagreement=int((diff & labels['paper']).sum()),
                           selected_probability_max_abs=float(np.abs(prediction-zero)[selected].max()),
                           selected_probability_mean_abs=float(np.abs(prediction-zero)[selected].mean()),
                           probability_file=str(pf.relative_to(ROOT)), probability_sha256=sha(pf),
                           mask_file=str(mf.relative_to(ROOT)), mask_sha256=sha(mf))
                rows.append(row)
                items.append((f"移動({dx},{dy}) 墨{row['core_r2']} 紙黒{row['paper_ink']}",
                              np.where(prediction[crop] < .5, 0, 255).astype(np.uint8)))
            fig = OUT / 'figures' / f'{n}_{k}.png'
            panel(fig, k+' '+n+' / 入力を移動し出力を逆移動', items)
            figures.append(str(fig.relative_to(OUT)))
            print(n, k, [(r['shift_xy'], r['core_r2'], r['paper_ink']) for r in rows[-9:]],
                  f'{time.monotonic()-start:.1f}s', flush=True)
        after = digest_state(model)
        assert before == after
        states.append(dict(condition=n, state_before=before, state_after=after,
                           identity_repeat_bitexact=repeat, checkpoint_sha256=cp['sha256']))
        del model
    dump(OUT / 'results.json', dict(plan_sha256=sha(OUT / 'plan.json'), rows=rows,
                                  unchanged_models=states, figures=figures, no_training=True))


def verify():
    p = check()
    r = read(OUT / 'results.json')
    assert r['plan_sha256'] == sha(OUT / 'plan.json')
    assert len(r['rows']) == 108 and len(r['figures']) == 12
    disk = np.array([[x*x+y*y <= 4 for x in range(-2, 3)] for y in range(-2, 3)])
    for row in r['rows']:
        k = row['case']
        pred = np.asarray(Image.open(ROOT / row['mask_file'])) < 128
        a = np.load(ROOT / row['probability_file'])
        assert sha(ROOT / row['mask_file']) == row['mask_sha256']
        assert sha(ROOT / row['probability_file']) == row['probability_sha256']
        assert np.array_equal(pred, a < .5)
        core = np.asarray(Image.open(OUT / 'inputs' / k / 'core.png')) > 0
        paper = np.asarray(Image.open(OUT / 'inputs' / k / 'paper.png')) > 0
        cov = ndi.convolve(pred.astype(np.uint8), disk.astype(np.uint8), mode='constant') > 0
        expected = dict(core_n=int(core.sum()), core_exact=int(pred[core].sum()),
                        core_r2=int(cov[core].sum()), paper_n=int(paper.sum()),
                        paper_ink=int(pred[paper].sum()), total_ink=int(pred.sum()))
        for key, value in expected.items():
            assert row[key] == value, (row['condition'], k, key)
        zpath = Path(row['probability_file']).parent / 'dx0_dy0.npy'
        z = np.load(ROOT / zpath)
        changed = pred != (z < .5)
        assert row['core_disagreement'] == int(changed[core].sum())
        assert row['paper_disagreement'] == int(changed[paper].sum())
    # Baseline matches the previous centered-context outputs exactly.
    baseline_checks = []
    for row in r['rows']:
        if row['shift_xy'] != [0, 0]:
            continue
        k, n = row['case'], row['condition']
        a = np.load(ROOT / row['probability_file'])
        if k.endswith('_centered'):
            case = next(c for c in read(parent.OUT / 'protocol.json')['context']['cases'] if c['id'] == k[:3])
            cx, cy = case['center_xy_in_512']
            previous = np.load(parent.OUT / 'context' / n / k[:3] / 'page_page.npy')
            assert np.array_equal(a[cy:cy+256, cx:cx+256], previous)
            baseline_checks.append(dict(condition=n, case=k, probabilities_bitexact=True))
        else:
            previous = next(t for t in read(parent.OUT / 'tiles.json')['rows'] if t['condition'] == n and t['case'] == 'S07' and t['variant'] == 'native_aux' and t['tile_xy'] == [1920, 0])
            old = np.load(ROOT / previous['probability_file'])
            assert np.array_equal(a[213:469, 286:512], old[:, :226])
            baseline_checks.append(dict(condition=n, case=k, probabilities_bitexact=True))
    figures = {f: sha(OUT/f) for f in r['figures']}
    dump(OUT / 'verification.json', dict(plan_sha256=sha(OUT/'plan.json'), results_sha256=sha(OUT/'results.json'),
        independent_mask_checks=len(r['rows']), baseline_probability_checks=baseline_checks,
        comparison_figures=figures, unchanged_models=True, identity_repeats=4, no_training=True))
    print('Verified 108 masks,12 baseline exact probability comparisons,12 PNGs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run', 'verify'])
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'verify': verify}[args.action]()
