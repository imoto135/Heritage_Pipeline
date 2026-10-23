"""Frozen-weight interventions on auxiliary channels, context, and GGC means."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
import run_ragan_full_continuation as full
from ragan_pilot_support import auxiliaries, make_model, predict
from compare_v5_aligned_fade_control import coverage, dump, mask, rgb, save, sha
from run_ragan_local_region_control import OUT as PREVIOUS
from run_ragan_stable_exposure_control import OUT as BASELINE

OUT = ROOT / 'tmp_work/ragan_inference_context_20261022'
HARD = '200017197_00214_1_p00'
CONDITIONS = {f'{kind}_order{order}': (BASELINE / f'stable_exposure_order{order}' if kind == 'baseline' else PREVIOUS / f'{kind}_order{order}') / 'step_00864.pt'
              for order in (0, 1) for kind in ('baseline', 'inkmean', 'inkpaper')}
VIEWS = ['crop_local', 'crop_full_otsu', 'crop_full_sobel', 'crop_full_aux',
         'full_global', 'full_patch_local_otsu', 'full_patch_local_sobel', 'full_patch_local_aux']


def cases():
    rows = full.read_rows('train')
    meta = full.metadata()
    result = []
    for patch, (trace_path, trace) in full.reviewed().items():
        index = next(i for i, row in enumerate(rows) if row['patch_id'] == patch and row['damage_family'] == 'targeted_fade' and float(row['damage_strength']) == .8)
        row = rows[index]
        x, y = full.coordinates(index, 13, row, meta)
        result.append(dict(patch=patch, record=row, xy=[x, y], trace=trace, trace_path=trace_path))
    return result


def digest_state(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Reader:
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.forced = None
        self.natural, self.effective = {}, {}
        self.handles = []
        for name, module in model.named_modules():
            if name.endswith('theta_net'):
                self.handles.append(module.register_forward_hook(self.hook(name)))
        assert len(self.handles) == 4

    def hook(self, name):
        def capture(module, args, output):
            value = float(output.mean())
            self.natural[name] = value
            if self.forced is not None:
                value = self.forced[name]
                output = output.new_tensor(value).reshape(1, 1, 1, 1)
            self.effective[name] = value
            return output
        return capture

    def __call__(self, channels, forced=None, contiguous=True):
        self.forced = forced
        self.natural, self.effective = {}, {}
        tensor = torch.from_numpy(channels.copy()).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255
        if contiguous:
            tensor = tensor.contiguous()
        with torch.inference_mode():
            p = predict(self.model, 'ragan', tensor)[0, 0].cpu().numpy()
        self.forced = None
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
        return p, dict(self.natural), dict(self.effective)


def freeze():
    assert not OUT.exists(), 'Preserve existing inference audit'
    OUT.mkdir()
    p = dict(checkpoints={c: dict(path=str(path.relative_to(ROOT)), sha256=sha(path)) for c, path in CONDITIONS.items()},
             no_weight_updates=True, no_GT_or_RGB_edits=True, threshold=.5, train_cases=7, train_trace_n=455,
             primary_hard_patch=HARD, views=VIEWS,
             crop='same epoch13 training window; local auxiliary channels recomputed after crop',
             full_patch='full512; replace only specified auxiliary channels inside the exact training window',
             all_ordinary_forwards_contiguous_NCHW=True,
             extra_hard_diagnostics=['reciprocal GGC means between crop_full_aux and full_global',
                 'same576 canvas with reflect padding, original full512 means fixed; left/top32 versus phase-aligned padding',
                 '256 crop full auxiliary channels; x origins0,4,...32 and fixed y0'],
             phase_diagnostics_are_not_pure_receptive_field_controls=True,
             tiled_followup=dict(conditions=[c for c in CONDITIONS if not c.startswith('inkmean')],
                 primary='native256 stride128 Hann floor.05, recompute auxiliary channels per tile, batch1',
                 source_grid='whole page origin0 and fixed final edge; independent of labels and reviewed coordinates',
                 secondary='same method with interior grid offset16; all boundaries also covered',
                 evaluate=['all7 training strong-fade patches', 'known heldout49 fade', '6 fixed synthetic cases', 'D01 D03 D04 T07 C01'],
                 seams='probability disagreement of overlapping tiles, and global-vs-seam-strip known paper error'),
             known_development=True, provisional_labels=True, independent_initializations=False, formal_accuracy=False)
    dump(OUT / 'protocol.json', p)
    source_hashes = {}
    inputs = []
    for c in cases():
        row = c['record']
        for path in [ROOT / row['input'], ROOT / row['clean'], c['trace_path'], *[ROOT / row['labels'][k] for k in ('target', 'valid')]]:
            source_hashes[str(path.relative_to(ROOT))] = sha(path)
        inputs.append(dict(patch=c['patch'], xy=c['xy'], n=int(c['trace'].sum())))
    dump(OUT / 'source_hashes.json', source_hashes)
    dump(OUT / 'crop_inputs.json', inputs)
    folder = OUT / 'source_snapshot'
    folder.mkdir()
    for filename in [Path(__file__).name, 'ragan_pilot_support.py', 'run_ragan_full_continuation.py']:
        shutil.copy2(ROOT / 'scripts/benchmark' / filename, folder / filename)
    print('Frozen context/auxiliary interventions and label-independent tiled followup', flush=True)


def run(order, device_name):
    torch.set_num_threads(2)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    source_hashes = json.loads((OUT / 'source_hashes.json').read_text())
    assert all(sha(ROOT / name) == digest for name, digest in source_hashes.items())
    result_path = OUT / f'audit_order{order}.json'
    assert not result_path.exists()
    result, input_audit, states, replay = [], [], [], []
    for condition, path in CONDITIONS.items():
        if not condition.endswith(f'order{order}'):
            continue
        assert sha(path) == protocol['checkpoints'][condition]['sha256']
        payload = torch.load(path, map_location='cpu', weights_only=False)
        model = make_model('ragan').to(device)
        model.load_state_dict(payload['model'])
        del payload
        model.eval()
        before = digest_state(model)
        read = Reader(model, device)
        for case in cases():
            patch, row = case['patch'], case['record']
            x, y = case['xy']
            image = rgb(ROOT / row['input'])
            big = auxiliaries(image)
            local = auxiliaries(image[y:y+256, x:x+256])
            fixed = big[y:y+256, x:x+256].copy()
            trace = case['trace'][y:y+256, x:x+256]
            assert trace.sum() == case['trace'].sum()
            arrays = {'crop_local': local, 'crop_full_aux': fixed, 'full_global': big}
            for name, k in [('otsu', 3), ('sobel', 4)]:
                a = local.copy()
                a[:, :, k] = fixed[:, :, k]
                arrays['crop_full_' + name] = a
                a = big.copy()
                a[y:y+256, x:x+256, k] = local[:, :, k]
                arrays['full_patch_local_' + name] = a
            a = big.copy()
            a[y:y+256, x:x+256, 3:] = local[:, :, 3:]
            arrays['full_patch_local_aux'] = a
            predictions, thetas = {}, {}

            def collect(view, channels, forced=None, aligned_xy=None):
                p, natural, effective = read(channels, forced)
                cx, cy = aligned_xy if aligned_xy is not None else ((x, y) if channels.shape[0] > 256 else (0, 0))
                aligned = p[cy:cy+256, cx:cx+256]
                assert aligned.shape == trace.shape
                pred = aligned < .5
                save(OUT / 'masks' / condition / patch / (view + '.png'), pred)
                values = aligned[trace]
                result.append(dict(condition=condition, patch=patch, view=view, input_hw=list(channels.shape[:2]),
                    xy=case['xy'], natural_theta=natural, effective_theta=effective,
                    paper_probability_median=float(np.median(values)), ink_probability_median=float(np.median(1-values)),
                    **coverage(pred, trace)))
                predictions[view], thetas[view] = aligned, effective
                return p

            for view in VIEWS:
                all_p = collect(view, arrays[view])
                if view in ('crop_local', 'full_global'):
                    directory = 'training_native_masks' if view == 'crop_local' else 'training_reviewed_masks'
                    name = patch + ('_single_crop1.png' if view == 'crop_local' else '_targeted_fade.png')
                    cached = mask(PREVIOUS / 'evaluation' / directory / condition / name)
                    fresh = all_p < .5
                    flips = int((cached != fresh).sum())
                    check = dict(condition=condition, patch=patch, view=view, cached_path=str((PREVIOUS/'evaluation'/directory/condition/name).relative_to(ROOT)),
                        cached_sha256=sha(PREVIOUS/'evaluation'/directory/condition/name), changed_pixels=flips,
                        new_trace=coverage(fresh if view == 'crop_local' else fresh[y:y+256,x:x+256], trace),
                        cached_trace=coverage(cached if view == 'crop_local' else cached[y:y+256,x:x+256], trace))
                    if flips:
                        alternative, _, _ = read(arrays[view], contiguous=False)
                        check.update(noncontiguous_changed_vs_cache=int(((alternative < .5) != cached).sum()),
                            layout_probability_max_abs=float(np.max(np.abs(alternative-all_p))),
                            layout_probability_mean_abs=float(np.mean(np.abs(alternative-all_p))))
                    replay.append(check)
            if patch == HARD:
                # Reciprocal interventions keep the five input channels fixed.
                collect('full_global_theta_crop', big, thetas['crop_full_aux'])
                collect('crop_full_aux_theta_full', fixed, thetas['full_global'])
                # Same576 canvas; fixed scalar gates remove their mean as a confound.
                for name, left, top in [('phase_original', 32, 32), ('phase_aligned', 32+(-x)%32, 32+(-y)%32)]:
                    padded = np.pad(big, ((top, 64-top), (left, 64-left), (0, 0)), mode='reflect')
                    collect(name, padded, thetas['full_global'], (x+left, y+top))
                for origin in range(0, 33, 4):
                    subtrace = case['trace'][0:256, origin:origin+256]
                    assert subtrace.sum() == 44 and y == 0
                    p, natural, effective = read(big[0:256, origin:origin+256])
                    pred = p < .5
                    save(OUT / 'masks' / condition / patch / f'origin_x{origin:02d}.png', pred)
                    result.append(dict(condition=condition, patch=patch, view=f'origin_x{origin:02d}', xy=[origin, 0],
                        natural_theta=natural, effective_theta=effective, **coverage(pred, subtrace)))
            if condition == f'baseline_order{order}':
                thresholds = {}
                for name, im in [('full', image), ('crop', image[y:y+256, x:x+256])]:
                    thresholds[name] = float(cv2.threshold(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0])
                input_audit.append(dict(patch=patch, xy=[x, y], otsu_thresholds=thresholds,
                    differences={name: dict(all_pixels=int((fixed[:,:,k] != local[:,:,k]).sum()),
                        trace_pixels=int(((fixed[:,:,k] != local[:,:,k]) & trace).sum())) for name, k in [('otsu', 3), ('sobel', 4)]}))
            print(condition, patch, [(v, next(r['r2'] for r in reversed(result) if r['condition']==condition and r['patch']==patch and r['view']==v)) for v in VIEWS], flush=True)
        after = digest_state(model)
        assert before == after
        states.append(dict(condition=condition, unchanged_state_sha256=before))
        del model, read
        torch.cuda.empty_cache()
    dump(result_path, dict(rows=result, input_audit=input_audit, states=states, cached_replay=replay,
        checkpoint_sha256={c: protocol['checkpoints'][c]['sha256'] for c in CONDITIONS if c.endswith(f'order{order}')},
        protocol_sha256=sha(OUT/'protocol.json'), no_weight_updates=True, formal_accuracy=False))
    print('Audit completed', order, len(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--order', type=int, choices=[0, 1])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    freeze() if args.action == 'freeze' else run(args.order, args.device)
