"""Fixed-method comparison on previously collected red-positive source pages.

No threshold tuning, neural training, annotation, deletion or restoration.
Frozen refinement arithmetic is extracted from its AST and checked against all
12 saved development pages. Old runs and original modules remain read-only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/red_positive_fixed_check_20260930'
COLLECTION = ROOT / 'tmp_work/red_positive_collection_20260930'
FROZEN = ROOT / 'tmp_work/training_free_protection_v2_20260930'
PREPROCESS = ROOT / 'tmp_work/training_free_preprocess_20260930'
COLOR = ROOT / 'tmp_work/training_free_color_shape_20260930'
NPJ = ROOT / 'tmp_work/npj_preprocess_review_20260930'
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')

import cv2
import numpy as np
cv2.setNumThreads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def read(path, mask=False):
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
    if value is None:
        raise OSError(path)
    return value > 0 if mask else value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if value.dtype == bool:
        value = value.astype(np.uint8) * 255
    assert cv2.imwrite(str(path), value), path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selection():
    return json.loads((COLLECTION / 'selection.json').read_text())


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = selection()
    # Verify input packaging and the frozen code before processing any new page.
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/benchmark/finalize_red_positive_collection.py'),
                    '--verify-only'], check=True, cwd=ROOT)
    prior = json.loads((FROZEN / 'manifest.json').read_text())
    old_color = json.loads((COLOR / 'manifest.json').read_text())
    old_local = json.loads((PREPROCESS / 'manifest.json').read_text())
    npj = json.loads((NPJ / 'manifest.json').read_text())
    hashes = {}
    for relative, expected in npj['source_hashes'].items():
        assert sha(ROOT / relative) == expected
        hashes[relative] = expected
    for name, expected in [('refine_training_free_protection.py', prior['source_sha256']),
                           ('compare_training_free_color_shape.py', old_color['source_sha256']),
                           ('compare_training_free_preprocessing.py', old_local['source_sha256'])]:
        path = ROOT / 'scripts/benchmark' / name
        assert sha(path) == expected
        hashes[str(path.relative_to(ROOT))] = expected
    dependencies = [COLLECTION / 'selection.json', COLLECTION / 'complete.json',
                    FROZEN / 'manifest.json', COLOR / 'manifest.json',
                    PREPROCESS / 'manifest.json', NPJ / 'manifest.json',
                    ROOT / 'scripts/benchmark/prepare_training_free_fixed_check.py']
    hashes.update({str(p.relative_to(ROOT)): sha(p) for p in dependencies})
    manifest = dict(purpose='Fixed conditions on input-only purposive red-positive collection; no pixel GT',
                    cases=selected['cases'], selection_sha256=sha(COLLECTION / 'selection.json'),
                    source_sha256=sha(Path(__file__)), dependencies=hashes,
                    local_settings=old_local['settings'], color_settings=old_color['parameters'],
                    refinement_settings=prior['settings'], ace_settings=npj['ace'], csc_settings=npj['csc'],
                    baseline='Local frozen v1, not established as exact published reproduction',
                    classification_resolution='native; thumbnails/plot scaling separate',
                    method_thresholds_changed=False, neural_training=False, pixel_GT=False,
                    notes='Baseline CSC fits per-image PCA/GMM without labeled training; proposals use classical image operations.',
                    versions=dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__))
    mp = OUT / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise RuntimeError('Different experiment already exists; create a new output version')
    dump(mp, manifest)
    shutil.copyfile(__file__, OUT / 'runner_source_snapshot.py')
    inputs = {r['original']: r['original_sha256'] for r in selected['cases']}
    cases = [dict(id=r['id'], title='・'.join(r['tags']), original=r['original']) for r in selected['cases']]
    rois = {r['id']: [dict(title=s['name'], xywh=s['xywh']) for s in r['rois']] for r in selected['cases']}
    dump(OUT / 'input_review/manifest.json', dict(cases=cases, inputs=inputs,
        selection_manifest_sha256=sha(COLLECTION / 'selection.json'), purpose='Input-selected crops, not pixel GT'))
    dump(OUT / 'input_review/rois.json', rois)
    adapter = load_module('red_fixed_foreground', ROOT / 'scripts/benchmark/prepare_training_free_fixed_check.py')
    reference = PREPROCESS / 'hnsd001_011'
    local, votes, normalized = adapter.foreground_inputs(read(reference / 'original.png'))
    for value, name in [(local, 'sauvola_local/foreground.png'), (votes, 'scale_votes.png'), (normalized, 'normalized.png')]:
        assert np.array_equal(value, cv2.imread(str(reference / name), 0)), name
    dump(OUT / 'foreground_adapter_parity.json', dict(status='passed', page='hnsd001_011', masks_and_normalized_equal=True))
    pilot = OUT / 'pilot'
    dump(pilot / 'manifest.json', dict(settings=old_local['settings'], inputs=inputs,
        frozen_manifest_sha256=sha(PREPROCESS / 'manifest.json'), adapter_sha256=sha(ROOT / 'scripts/benchmark/prepare_training_free_fixed_check.py')))
    for row in selected['cases']:
        dest = pilot / row['id']
        if (dest / 'complete.json').exists():
            for rel, digest in json.loads((dest / 'complete.json').read_text())['outputs'].items():
                assert sha(pilot / rel) == digest
            continue
        assert sha(ROOT / row['original']) == row['original_sha256']
        original = read(ROOT / row['original'])
        local, votes, normalized = adapter.foreground_inputs(original)
        for value, name in [(original, 'original.png'), (local, 'sauvola_local/foreground.png'),
                            (votes, 'scale_votes.png'), (normalized, 'normalized.png')]:
            write(dest / name, value)
        dump(dest / 'complete.json', dict(outputs={str(p.relative_to(pilot)): sha(p) for p in sorted(dest.rglob('*.png'))}))
        print(f'Local input {row["id"]}: {int((local > 0).sum())} foreground candidates', flush=True)
    dump(pilot / 'complete.json', dict(status='complete', outputs={str(p.relative_to(pilot)): sha(p) for p in sorted(pilot.rglob('*.png'))}))


def color():
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/benchmark/compare_training_free_color_shape.py'),
                    '--pilot', str(OUT / 'pilot'), '--review', str(OUT / 'input_review'),
                    '--output', str(OUT / 'color')], check=True, cwd=ROOT)
    actual = json.loads((OUT / 'color/manifest.json').read_text())['parameters']
    assert actual == json.loads((COLOR / 'manifest.json').read_text())['parameters']


def refinement_function():
    """Extract the frozen computational block, changing only file input plumbing."""
    path = ROOT / 'scripts/benchmark/refine_training_free_protection.py'
    expected = json.loads((FROZEN / 'manifest.json').read_text())['source_sha256']
    assert sha(path) == expected
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    distance = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'distance_to')
    loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'case')
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'median')
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'masks')
    nodes = loop.body[start:end+1]
    replacements = 0
    for node in nodes:
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "json.loads((old / 'components.json').read_text())":
            node.iter = ast.Name(id='components', ctx=ast.Load())
            replacements += 1
    assert replacements == 1
    args = 'original foreground votes old_red old_ink old_stain brown components'.split()
    function = ast.FunctionDef(name='refine', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in args],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=nodes + [ast.Return(value=ast.Tuple(elts=[
            ast.Name(id='masks', ctx=ast.Load()), ast.Name(id='component_rows', ctx=ast.Load())], ctx=ast.Load()))], decorator_list=[])
    generated = ast.fix_missing_locations(ast.Module(body=[distance, function], type_ignores=[]))
    adapted = ast.unparse(generated) + '\n'
    path_out = OUT / 'refinement_adapter_snapshot.py'
    path_out.write_text(adapted)
    ns = dict(cv2=cv2, np=np)
    exec(compile(generated, str(path_out), 'exec'), ns)
    return ns['refine']


def refine_inputs(function, original_dir, pilot, stem):
    old = original_dir / stem
    masks, components = function(read(old / 'original.png'), read(pilot / stem / 'sauvola_local/foreground.png', True),
        cv2.imread(str(pilot / stem / 'scale_votes.png'), 0) // 85,
        read(old / 'color_shape/red.png', True), read(old / 'color_shape/ink.png', True),
        read(old / 'color_shape/stain.png', True), read(old / 'brown_field.png', True),
        json.loads((old / 'components.json').read_text()))
    return masks, components


def refine():
    function = refinement_function()
    parity = []
    # This comparison checks executable adaptation, rather than assuming a copied formula is identical.
    for group, pilot in [(COLOR, PREPROCESS),
                         (ROOT / 'tmp_work/training_free_fixed_check_20260930', ROOT / 'tmp_work/training_free_fixed_check_20260930/pilot')]:
        for old in sorted(group.iterdir()):
            if not (old / 'color_shape/red.png').exists():
                continue
            masks, _ = refine_inputs(function, group, pilot, old.name)
            for key, mask in masks.items():
                assert np.array_equal(mask, read(FROZEN / old.name / (key + '.png'), True)), (old.name, key)
            parity.append(dict(page=old.name, masks_equal=len(masks)))
    assert len(parity) == 12
    dump(OUT / 'refinement_adapter_parity.json', dict(status='passed', cases=parity,
        adapted_source_sha256=sha(OUT / 'refinement_adapter_snapshot.py'), frozen_source_sha256=sha(ROOT / 'scripts/benchmark/refine_training_free_protection.py')))
    for row in selection()['cases']:
        stem = row['id']; dest = OUT / 'refined' / stem
        if (dest / 'complete.json').exists():
            for rel, digest in json.loads((dest / 'complete.json').read_text())['outputs'].items():
                assert sha(OUT / rel) == digest
            continue
        started = time.monotonic()
        masks, components = refine_inputs(function, OUT / 'color', OUT / 'pilot', stem)
        assert not (masks['edit_allowed'] & masks['protect']).any()
        assert not (masks['edit_allowed'] & masks['hold']).any()
        assert not (masks['edit_allowed'] & masks['protection_guard']).any()
        assert not (masks['edit_allowed'] & masks['legacy_red']).any()
        assert np.array_equal(masks['protect'], masks['black'] | masks['red'] | masks['ink_color_uncertain'])
        for key, value in masks.items():
            write(dest / (key + '.png'), value)
        dump(dest / 'red_components.json', components)
        dump(dest / 'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in sorted(dest.glob('*')) if p.name != 'complete.json'}))
        print(f'Refined {stem}: red {int(masks["legacy_red"].sum())}->{int(masks["red"].sum())}, '
              f'old red still protected {int((masks["legacy_red"] & masks["protect"]).sum())}, '
              f'edit {int(masks["edit_allowed"].sum())}, {time.monotonic()-started:.1f}s', flush=True)


def baseline():
    ace = load_module('fixed_red_ace', ROOT / 'modules/ddrm_npj/scripts/ace.py')
    csc = load_module('fixed_red_csc', ROOT / 'modules/ddrm_npj/scripts/csc.py')
    for row in selection()['cases']:
        dest = OUT / 'baseline' / row['id']
        done = dest / 'complete.json'
        if done.exists():
            for rel, digest in json.loads(done.read_text())['outputs'].items():
                assert sha(OUT / rel) == digest
            print(f'Baseline {row["id"]}: verified cached outputs', flush=True)
            continue
        original = read(ROOT / row['original'])
        started = time.monotonic()
        ace_path = dest / 'ace.png'
        ace_done = dest / 'ace_complete.json'
        if ace_done.exists():
            assert sha(ace_path) == json.loads(ace_done.read_text())['sha256']
            equalized = read(ace_path)
        else:
            print(f'CPU ACE {row["id"]} {original.shape}: start', flush=True)
            equalized = ace.automatic_color_equalization(original, alpha=5., n_samples=200, seed=0)
            write(ace_path, equalized)
            dump(ace_done, dict(sha256=sha(ace_path), original_sha256=row['original_sha256'],
                               seconds=time.monotonic()-started))
            print(f'CPU ACE {row["id"]}: {time.monotonic()-started:.1f}s', flush=True)
        for mode, image in [('v1_no_ace', original), ('v1_ace', equalized)]:
            mode_done = dest / mode / 'complete.json'
            if mode_done.exists():
                for rel, digest in json.loads(mode_done.read_text())['outputs'].items():
                    assert sha(OUT / rel) == digest
                continue
            t = time.monotonic()
            result = csc.color_space_clustering(image, seed=42, max_samples=50000, border_margin_ratio=.04)
            masks = dict(black=result.black_text_mask, red=result.red_text_mask,
                         stain=result.degradation_mask, background=result.background_mask)
            count = np.zeros(image.shape[:2], np.uint8)
            for key, mask in masks.items():
                assert set(np.unique(mask)).issubset({0, 255})
                count += mask > 0
                write(dest / mode / (key + '.png'), mask)
            assert np.all(count == 1)
            dump(mode_done, dict(seconds=time.monotonic()-t,
                outputs={str(p.relative_to(OUT)): sha(p) for p in sorted((dest / mode).glob('*.png'))}))
            print(f'{row["id"]} {mode}: red {int((masks["red"] > 0).sum())}, {time.monotonic()-t:.1f}s', flush=True)
        dump(done, dict(outputs={str(p.relative_to(OUT)): sha(p) for p in sorted(dest.rglob('*.png'))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'color', 'refine', 'baseline', 'all'], default='all')
    args = parser.parse_args()
    for name, action in [('prepare', prepare), ('color', color), ('refine', refine), ('baseline', baseline)]:
        if args.stage in (name, 'all'):
            action()
