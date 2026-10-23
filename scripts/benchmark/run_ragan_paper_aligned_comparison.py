"""Controlled continuation: teacher v2/v3 and public/equation GA.

The prior pilot remains immutable. Each condition gets an isolated output dir.
Validation stays on v2 labels for all conditions; v3 changes training labels only.
This is an engineering comparison with provisional labels, not a benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'tmp_work/makura_supervised_smoke_20261021'
V3 = ROOT / 'tmp_work/makura_teacher_v3_20261021'
OUT = ROOT / 'tmp_work/ragan_paper_aligned_comparison_20261021'
RUNNER_PATH = BASE / 'code_snapshot/run_makura_supervised_pilot.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def load_runner():
    if str(RUNNER_PATH.parent) not in sys.path:
        sys.path.insert(0, str(RUNNER_PATH.parent))
    spec = importlib.util.spec_from_file_location('frozen_makura_pilot', RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    assert not OUT.exists(), f'Keep comparison immutable; output already exists: {OUT}'
    code_hashes = json.loads((BASE/'code_hashes.json').read_text())
    assert sha(RUNNER_PATH) == code_hashes['scripts/benchmark/run_makura_supervised_pilot.py']
    source_rows = {}
    for split in ('train', 'val'):
        source_rows[split] = [json.loads(line) for line in (BASE/f'{split}_manifest.jsonl').read_text().splitlines()]
    revised_train = [json.loads(line) for line in (V3/'train_manifest.jsonl').read_text().splitlines()]
    assert [r['id'] for r in source_rows['train']] == [r['id'] for r in revised_train]
    assert all(a['input_sha256'] == b['input_sha256'] for a,b in zip(source_rows['train'], revised_train))
    conditions = {
        'unet_v2': ('unet', 'v2', 'public'),
        'unet_v3': ('unet', 'v3', 'public'),
        'ragan_public_v3': ('ragan', 'v3', 'public'),
        'ragan_paperga_v3': ('ragan', 'v3', 'paper_eq9_12_candidate'),
    }
    for name, (model, teacher, attention) in conditions.items():
        dest = OUT/name
        dest.mkdir(parents=True)
        train_rows = source_rows['train'] if teacher == 'v2' else revised_train
        # One fixed validation target across conditions isolates training-label changes.
        for split, rows in (('train', train_rows), ('val', source_rows['val'])):
            (dest/f'{split}_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        shutil.copy2(BASE/'label_protocol.json', dest/'label_protocol.json')
        shutil.copy2(BASE/'adapter_verification.json', dest/'adapter_verification.json')
        protocol = json.loads((dest/'label_protocol.json').read_text())
        protocol['comparison_training_teacher'] = teacher
        protocol['validation_teacher'] = 'v2 fixed for every condition'
        protocol['attention'] = attention
        protocol['formal_accuracy'] = False
        dump(dest/'label_protocol.json', protocol)
        shutil.copy2(BASE/'code_hashes.json', dest/'code_hashes.json')
        dump(dest/'condition.json', dict(name=name, model=model, training_teacher=teacher,
            validation_teacher='v2 fixed', attention=attention, same_inputs=True,
            same_seed=20261021, source_input_hashes_checked=True,
            labels_are_provisional=True, formal_accuracy=False))
    dump(OUT/'protocol.json', dict(conditions=conditions,
        validation='v2 labels fixed across all runs; same native 256 center crop',
        training='same 1440-input manifest and paired damaged RGB + Otsu + Sobel; only v2/v3 target differs',
        updates=432, batch_size=4, seed=20261021,
        caveat='v3 has source-checked sparse centers in only 3 patches; rest inherits provisional v2 labels',
        attention_candidate='Paper Eq.9-12 attention; I*W output interpretation; not complete paper reproduction',
        evaluation='engineering comparison only; fixed v2 validation still provisional, not independent GT'))
    print('Prepared 4 isolated conditions under', OUT, flush=True)


def run(name, device):
    runner = load_runner()
    import ragan_pilot_support as support
    runner.ROOT = ROOT
    support.ROOT = ROOT
    support.REFERENCE = ROOT / 'tmp_work/ra_gan_reference_20261021'
    dest = OUT/name
    condition = json.loads((dest/'condition.json').read_text())
    assert not (dest/'models'/condition['model']).exists(), 'Condition already run; preserve it.'
    runner.OUT = dest
    if condition['attention'] == 'paper_eq9_12_candidate':
        from ragan_paper_ga import candidate_generator
        base_factory = support.make_model
        support.make_model = lambda kind: base_factory(kind)
        runner.make_model = lambda kind: candidate_generator() if kind in ('ragan','ra_no_gan') else base_factory(kind)
    elif condition['attention'] == 'paper_eq9_12_residual_gate_control':
        from ragan_paper_ga import candidate_generator
        base_factory = support.make_model
        runner.make_model = lambda kind: candidate_generator('residual_gate') if kind in ('ragan','ra_no_gan') else base_factory(kind)
    args = argparse.Namespace(model=condition['model'], steps=432, batch_size=4,
        lr=3e-5, validate_every=108, device=device)
    runner.train(args)
    print('Completed', name, flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','run'])
    parser.add_argument('--condition', choices=['unet_v2','unet_v3','ragan_public_v3','ragan_paperga_v3','ra_nogan_paperga_v3','ra_nogan_residualga_v3'])
    parser.add_argument('--device', default='cuda:0')
    args=parser.parse_args()
    if args.action == 'prepare': prepare()
    else:
        assert args.condition, '--condition required'
        run(args.condition,args.device)


if __name__=='__main__': main()
