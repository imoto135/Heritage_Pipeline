"""Same anchored stain field, old brown-alpha compositing, matched RA-GAN run."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
from run_contact_stain_training_control import OUT, STRENGTHS, make_training_field, train
from evaluate_ridge_gate_independent_val import proxies
from compare_v5_aligned_fade_control import sha, dump, rgb

DEST = OUT / 'ragan_brown_aligned'


def prepare():
    assert not DEST.exists()
    DEST.mkdir()
    old = [json.loads(s) for s in (OUT / 'ragan_original/train_manifest.jsonl').read_text().splitlines()]
    dark = [json.loads(s) for s in (OUT / 'ragan_contact/train_manifest.jsonl').read_text().splitlines()]
    assert len(old) == len(dark) == 864
    old_color = {}
    sources = {}
    for row in old:
        if row['damage_family'] != 'stain_overlap' or float(row['damage_strength']) != .75:
            continue
        patch = row['patch_id']
        src = rgb(ROOT / row['clean'])
        image = rgb(ROOT / row['input'])
        alpha = np.asarray(Image.open(ROOT / row['alpha']).convert('L')).astype(np.float32) / 255
        valid = alpha > .3
        assert valid.sum() > 1000
        estimate = (image.astype(np.float32) - src.astype(np.float32) * (1-alpha[..., None])) / np.maximum(alpha[..., None], 1e-6)
        color = np.median(estimate[valid], axis=0)
        assert (color >= 20).all() and (color <= 190).all(), (patch, color)
        old_color[patch] = color
        sources[patch] = src
    assert len(old_color) == 96
    audit = []
    image_paths = {}
    expected_anchors = {r['patch']: r['anchor_xy'] for r in json.loads((OUT / 'input_audit.json').read_text())['rows']}
    for patch, source in sorted(sources.items()):
        _, core, remote, _ = proxies(source)
        field, anchor = make_training_field(core)
        assert list(anchor) == expected_anchors[patch]
        for strength in STRENGTHS:
            alpha = (field * strength)[..., None]
            damaged = np.rint(np.clip(source.astype(np.float32)*(1-alpha) + old_color[patch]*alpha, 0, 255)).astype(np.uint8)
            path = OUT / 'brown_aligned_inputs' / patch / f'stain_{strength:.2f}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(damaged).save(path)
            image_paths[(patch, strength)] = path
            affected = core & (field >= .5)
            stain_paper = remote & (field >= .5)
            audit.append(dict(patch=patch, strength=strength, input_sha256=sha(path),
                recovered_old_brown_rgb=old_color[patch].round(2).tolist(),
                source_core_n=int(affected.sum()), remote_paper_n=int(stain_paper.sum())))
    revised = []
    for row in dark:
        if row['damage_family'] == 'stain_overlap':
            path = image_paths[(row['patch_id'], float(row['damage_strength']))]
            row['input'] = str(path.relative_to(ROOT))
            row['input_sha256'] = sha(path)
            row['contact_stain_recipe'] = 'brown_alpha_same_new_field_old_per_patch_color'
        revised.append(row)
    (DEST / 'train_manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in revised))
    for name in ('val_manifest.jsonl', 'label_protocol.json', 'adapter_verification.json', 'code_hashes.json'):
        shutil.copyfile(OUT / 'ragan_contact' / name, DEST / name)
    dump(DEST / 'condition.json', dict(kind='ragan', variant='brown_aligned',
        train_manifest_sha256=sha(DEST / 'train_manifest.jsonl'),
        val_manifest_sha256=sha(DEST / 'val_manifest.jsonl'), steps=432,
        batch_size=4, seed=20261021, formal_accuracy=False, official_reproduction=False))
    dump(OUT / 'brown_aligned_protocol.json', dict(condition='ragan_brown_aligned',
        old_color='Recover median alpha-overlay RGB from old .75 image/alpha/source per patch',
        same_field_crop_labels_initialization_steps_as_ragan_contact=True,
        only_compositing_color_differs_from_ragan_contact=True,
        new_input_count=288, source_proxy_not_GT=True, formal_accuracy=False))
    dump(OUT / 'brown_aligned_input_audit.json', dict(rows=audit))
    print('Prepared brown aligned', len(audit), 'inputs', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'train'])
    a = p.parse_args()
    if a.action == 'prepare':
        prepare()
    else:
        train('ragan_brown_aligned', 'cuda:0')
