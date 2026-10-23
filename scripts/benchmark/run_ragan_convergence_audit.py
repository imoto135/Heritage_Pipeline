"""Freeze unchanged-input RA-GAN runs to distinguish 2-epoch pilot from convergence."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
import run_contact_stain_training_control as control
from compare_v5_aligned_fade_control import sha, dump

OUT = ROOT / 'tmp_work/ragan_convergence_audit_20261022'
STEPS = 2160


def prepare():
    assert not OUT.exists()
    OUT.mkdir()
    parent = control.OUT
    for name in ('ragan_original', 'ragan_contact'):
        dest = OUT / name
        dest.mkdir()
        for filename in ('train_manifest.jsonl','val_manifest.jsonl','label_protocol.json','adapter_verification.json','code_hashes.json'):
            shutil.copyfile(parent / name / filename, dest / filename)
        conf = json.loads((parent / name / 'condition.json').read_text())
        conf.update(steps=STEPS, epochs=10, parent_checkpoint_sha256=sha(parent / name / 'models/ragan/last.pt'),
            changed_factor='training duration only; restart from identical seed; checkpoints every two epochs')
        dump(dest / 'condition.json', conf)
    shutil.copyfile(parent / 'input_audit.json', OUT / 'input_audit.json')
    dump(OUT / 'protocol.json', dict(conditions=['ragan_original','ragan_contact'],
        seed=20261021, steps=STEPS, epochs=10, batch_size=4, learning_rate=3e-5,
        unchanged_RGB=True, unchanged_labels=True, unchanged_crops=True, unchanged_GAN_loss=True,
        checkpoints=[432,864,1296,1728,2160], preexisting_outputs_not_blind=True,
        fixed_final_checkpoint_primary=True, no_threshold_tuning=True,
        implementation='public-code generator with GA cancellation retained; local GGC adapter',
        purpose='convergence engineering audit; not paper reproduction or model-limit claim',
        formal_accuracy=False))
    print('Frozen two 10-epoch runs', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare','run'])
    p.add_argument('--condition', choices=['ragan_original','ragan_contact'])
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        assert args.condition
        control.OUT = OUT
        control.train(args.condition, args.device, steps=STEPS, validate_every=432, keep_checkpoints=True)
