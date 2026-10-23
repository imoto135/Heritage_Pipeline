"""Matched pilot: replace old brown overlays with audited ink-contact dark stains."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
from run_v5_aligned_fade_control import ROOT as _ROOT, RUNNER, SEED, OLD, crop_xy, trace_centers, sha, dump, rgb
from evaluate_ridge_gate_independent_val import proxies

OUT = ROOT / 'tmp_work/makura_contact_stain_train_control_20261022'
BASE = ROOT / 'tmp_work/makura_v5_aligned_fade_control_20261021'
STRENGTHS = (.25, .50, .75)
KINDS = {'unet': 'unet_v3_v5', 'ragan': 'ragan_public_v3_v5'}


def stain(source, field, strength):
    return np.rint(np.clip(source.astype(np.float32) * np.exp(
        -(field * strength)[..., None] * np.array([2., 2.6, 3.3], np.float32)), 0, 255)).astype(np.uint8)


def make_training_field(core):
    candidates = core.copy()
    candidates[:128] = False
    candidates[384:] = False
    candidates[:, :128] = False
    candidates[:, 384:] = False
    assert candidates.any()
    density = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 65)
    cy, cx = np.unravel_index(np.argmax(np.where(candidates, density, -1)), core.shape)
    yy, xx = np.mgrid[:core.shape[0], :core.shape[1]]
    dx, dy = xx - cx, yy - cy
    angle = .4
    u = (dx * np.cos(angle) + dy * np.sin(angle)) / 110
    v = (-dx * np.sin(angle) + dy * np.cos(angle)) / 75
    theta = np.arctan2(v, u)
    radius = np.sqrt(u*u + v*v) / (1 + .13*np.sin(3*theta + 1.1) + .08*np.sin(7*theta + 2.2))
    field = np.clip((1-radius)/.32 + .5, 0, 1).astype(np.float32)
    field[field < .01] = 0
    return field, (int(cx), int(cy))


def prepare():
    assert not OUT.exists(), OUT
    OUT.mkdir(parents=True)
    baseline = [json.loads(s) for s in (BASE / 'unet_v3_v5/train_manifest.jsonl').read_text().splitlines()]
    patches = {r['patch_id']: r for r in baseline}
    assert len(patches) == 96 and len(baseline) == 864
    new_paths = {}
    audit = []
    for patch, record in sorted(patches.items()):
        source_path = ROOT / record['clean']
        source = rgb(source_path)
        paper, core, remote, _ = proxies(source)
        assert core.any() and remote.any(), patch
        field, anchor = make_training_field(core)
        overlap = field >= .5
        affected = core & overlap
        stain_paper = remote & overlap
        assert affected.sum() >= 20 and stain_paper.sum() >= 200, (patch, affected.sum(), stain_paper.sum())
        for strength in STRENGTHS:
            dest = OUT / 'inputs' / patch / f'contact_stain_{strength:.2f}.png'
            dest.parent.mkdir(parents=True, exist_ok=True)
            damaged = stain(source, field, strength)
            Image.fromarray(damaged).save(dest)
            new_paths[(patch, strength)] = dest
            residual = stain(paper, field, strength)
            contrast = np.max(np.abs(damaged.astype(np.int16) - residual.astype(np.int16)), axis=2)
            audit.append(dict(patch=patch, strength=strength, source_sha256=sha(source_path),
                input_sha256=sha(dest), anchor_xy=anchor, affected_source_core=int(affected.sum()),
                affected_remote_paper=int(stain_paper.sum()),
                median_remaining_rgb_contrast=float(np.median(contrast[affected])),
                source_core_remaining_signal_lt8=int((contrast[affected] < 8).sum()),
                source_proxy_not_ground_truth=True))
    assert len(new_paths) == 288
    conditions = []
    for kind, stem in KINDS.items():
        parent = BASE / stem
        for variant in ('original', 'contact'):
            dest = OUT / f'{kind}_{variant}'
            dest.mkdir()
            rows = [json.loads(s) for s in (parent / 'train_manifest.jsonl').read_text().splitlines()]
            assert len(rows) == 864
            replaced = 0
            for row in rows:
                if variant == 'contact' and row['damage_family'] == 'stain_overlap':
                    key = (row['patch_id'], float(row['damage_strength']))
                    path = new_paths[key]
                    row['input'] = str(path.relative_to(ROOT))
                    row['input_sha256'] = sha(path)
                    row['alpha'] = None
                    row['contact_stain_recipe'] = 'multiplicative_exp_rgb_2_2.6_3.3'
                    replaced += 1
            assert replaced == (288 if variant == 'contact' else 0)
            (dest / 'train_manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            for name in ('val_manifest.jsonl', 'label_protocol.json', 'adapter_verification.json', 'code_hashes.json'):
                (dest / name).write_bytes((parent / name).read_bytes())
            dump(dest / 'condition.json', dict(kind=kind, variant=variant, baseline_manifest=str(parent.relative_to(ROOT)),
                train_manifest_sha256=sha(dest / 'train_manifest.jsonl'), val_manifest_sha256=sha(dest / 'val_manifest.jsonl'),
                steps=432, batch_size=4, seed=SEED, formal_accuracy=False, official_reproduction=False))
            conditions.append(dest.name)
        a = [json.loads(s) for s in (OUT / f'{kind}_original/train_manifest.jsonl').read_text().splitlines()]
        b = [json.loads(s) for s in (OUT / f'{kind}_contact/train_manifest.jsonl').read_text().splitlines()]
        assert all(x['labels'] == y['labels'] for x, y in zip(a, b))
        assert sum(x['input_sha256'] != y['input_sha256'] for x, y in zip(a, b)) == 288
    dump(OUT / 'input_audit.json', dict(rows=audit, source_proxy_not_ground_truth=True))
    dump(OUT / 'protocol.json', dict(conditions=conditions, steps=432, batch_size=4, seed=SEED,
        controlled_difference='288 of 864 stain_overlap training RGB inputs only',
        identical_labels=True, identical_validation=True, identical_crop_schedule=True,
        identical_model_initialization_per_kind=True, train_patches=96,
        train_pages=24, train_groups=24,
        heldout_groups=['200017197_00171', '200017197_00283'],
        heldout_patches=['200017197_00171_1_p00', '200017197_00283_0_p00'],
        heldout_selection='p00 from two not-yet-scored validation groups, fixed before model output inspection',
        eval_strength=.65, eval_thresholds={'unet': [.5, .2], 'ragan': [.5]},
        damage_not_DKDS=True, all_dense_labels_provisional=True, formal_accuracy=False))
    print('Prepared', len(conditions), 'conditions;', len(audit), 'audited contact inputs', flush=True)
    print('Min ink overlap', min(x['affected_source_core'] for x in audit),
          'min paper overlap', min(x['affected_remote_paper'] for x in audit), flush=True)


def train(condition, device, steps=432, validate_every=108, keep_checkpoints=False):
    conf = json.loads((OUT / condition / 'condition.json').read_text())
    assert sha(OUT / condition / 'train_manifest.jsonl') == conf['train_manifest_sha256']
    sys.path.insert(0, str(RUNNER.parent))
    import ragan_pilot_support as support
    support.ROOT = ROOT
    support.REFERENCE = ROOT / 'tmp_work/ra_gan_reference_20261021'
    spec = importlib.util.spec_from_file_location('contact_pilot_runner_' + condition, RUNNER)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    runner.ROOT = ROOT
    runner.OUT = OUT / condition
    centers = trace_centers()
    generic = {r['patch']: r['selected_generic_xy'] for r in json.loads((OLD / 'input_audit.json').read_text())['rows'] if r['strength'] == .8}

    class CropDataset(runner.PilotDataset):
        def __getitem__(self, idx):
            row = self.records[idx]
            if self.split != 'train':
                return super().__getitem__(idx)
            if row['damage_family'] == 'targeted_fade':
                x, y = crop_xy(idx, self.epoch, row, centers, generic)
            elif row['damage_family'] == 'stain_overlap':
                patch = row['patch_id']
                anchor = anchors[patch]
                rng = np.random.default_rng(SEED + self.epoch * 100000 + idx)
                dx, dy = map(int, rng.integers(-16, 17, size=2))
                x = int(np.clip(anchor[0] - 128 + dx, 0, 256))
                y = int(np.clip(anchor[1] - 128 + dy, 0, 256))
            else:
                rng = np.random.default_rng(SEED + self.epoch * 100000 + idx)
                x, y = map(int, rng.integers(0, 257, size=2))
            raw = runner.rgb(ROOT / row['input'])[y:y+256, x:x+256]
            target, valid = self.labels[row['patch_id']]
            image = runner.auxiliaries(raw)
            return (torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255,
                torch.from_numpy(target[y:y+256, x:x+256].copy()).unsqueeze(0).float() / 255,
                torch.from_numpy(valid[y:y+256, x:x+256].copy()).unsqueeze(0).float() / 255)

    anchors = {r['patch']: r['anchor_xy'] for r in json.loads((OUT / 'input_audit.json').read_text())['rows']}
    runner.PilotDataset = CropDataset
    args = argparse.Namespace(model=conf['kind'], steps=steps, batch_size=4, lr=3e-5, validate_every=validate_every, device=device)
    original_save = torch.save
    if keep_checkpoints:
        def save_checkpoint(payload, path, *save_args, **save_kwargs):
            original_save(payload, path, *save_args, **save_kwargs)
            if Path(path).name == 'last.pt':
                snapshot = Path(path).with_name(f'step_{payload["step"]:05d}.pt')
                assert not snapshot.exists(), snapshot
                original_save(payload, snapshot, *save_args, **save_kwargs)
        torch.save = save_checkpoint
    try:
        runner.train(args)
    finally:
        torch.save = original_save


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train'])
    parser.add_argument('--condition', choices=['unet_original', 'unet_contact', 'ragan_original', 'ragan_contact'])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        assert args.condition
        train(args.condition, args.device)
