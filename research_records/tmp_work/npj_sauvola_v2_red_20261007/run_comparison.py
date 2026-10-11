"""Hold the saved CSC v2 red mask fixed; compare black masks at native page size."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREV = ROOT / 'tmp_work/npj_contact_research_20261007'
AUDIT = ROOT / 'tmp_work/npj_reproduction_audit_20261007'
sys.path.insert(0, str(PREV))
from common import sauvola

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return np.asarray(Image.open(p).convert('RGB'))

def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2))

def main():
    cases = json.loads((AUDIT / 'protocol.json').read_text())['cases']
    dependencies = {}
    selected = []
    for c in cases:
        page, cid = c['page'], c['id']
        original = ROOT / c['source']['original']
        v2 = ROOT / c['v2_dir']
        sp = ROOT / c['folder'] / 'masks/sauvola' / page
        old = ROOT / c['folder'] / 'output/sauvola' / page / 'result' / (page + '.png')
        baseline = AUDIT / 'outputs/v2_masks' / page / 'result' / (page + '.png')
        assert np.array_equal(read(sp / '02_text_black.png')[:, :, 0] < 127, sauvola(read(original)))
        red_identical = np.array_equal(read(sp / '03_text_red.png'), read(v2 / '03_text_red.png'))
        prior = json.loads((ROOT / c['folder'] / 'logs' / (cid + '_sauvola.json')).read_text())
        prior_input = Path(prior['command'][prior['command'].index('--input_dir') + 1]) / (page + '.png')
        assert np.array_equal(read(prior_input), read(original))
        old_expected = prior.get('sha256', prior.get('output_sha256'))
        assert sha(old) == old_expected
        masks = OUT / 'masks/sauvola_v2_red' / page
        masks.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sp / '02_text_black.png', masks / '02_text_black.png')
        shutil.copyfile(v2 / '03_text_red.png', masks / '03_text_red.png')
        result = old if red_identical else OUT / 'outputs/sauvola_v2_red' / page / 'result' / (page + '.png')
        baseline_log = json.loads((AUDIT / 'logs' / (cid + '_v2_masks.json')).read_text())
        assert sha(baseline) == baseline_log['sha256']
        for p in [original, sp / '02_text_black.png', sp / '03_text_red.png', v2 / '02_text_black.png', v2 / '03_text_red.png', old, baseline, prior_input]:
            dependencies[str(p.relative_to(ROOT))] = sha(p)
        selected.append(dict(id=cid, page=page, original=str(original.relative_to(ROOT)),
            v2_black=str((v2 / '02_text_black.png').relative_to(ROOT)),
            v2_red=str((v2 / '03_text_red.png').relative_to(ROOT)),
            sauvola_black=str((sp / '02_text_black.png').relative_to(ROOT)),
            old_red=str((sp / '03_text_red.png').relative_to(ROOT)),
            v2_result=str(baseline.relative_to(ROOT)), old_sauvola_result=str(old.relative_to(ROOT)),
            fixed_sauvola_result=str(result.relative_to(ROOT)),
            reused_existing=red_identical, full_page_red_masks_identical=red_identical,
            prior_command=prior['command']))
    snap = ROOT / 'tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2'
    checkpoint = ROOT / 'modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'
    paper = ROOT / 'modules/ddrm_npj_v2/background/isemonogatari.png'
    for p in [checkpoint, paper, *snap.rglob('*.py')]:
        dependencies[str(p.relative_to(ROOT))] = sha(p)
    protocol = dict(cases=selected, source_sha256=dependencies, seed=999, timesteps=20, patch=128, overlap=32,
        sauvola=dict(window=61, k=0.2, R=0.5, grayscale='cv2 RGB2GRAY / 255', border='REFLECT_101'),
        native_full_page=True, red_mask='saved CSC v2, identical in the two compared conditions',
        changed_factor='black mask only; saved CSC v2 versus previously frozen Sauvola',
        shared_stepwise_random_draws_guaranteed=False, reading_ground_truth=False,
        authentic_paper_replay=False, created=time.time())
    dump(OUT / 'protocol.json', protocol)
    jobs = []
    for c in selected:
        if c['reused_existing']:
            print('Reuse verified identical-red condition:', c['id'], flush=True)
            continue
        cid, page = c['id'], c['page']
        inp = OUT / 'inputs' / cid
        inp.mkdir(parents=True, exist_ok=True)
        Image.open(ROOT / c['original']).convert('RGB').save(inp / (page + '.png'))
        device = c['prior_command'][c['prior_command'].index('--device') + 1]
        cmd = [sys.executable, '-B', str(snap / 'scripts/pipeline_ddrm.py'),
            '--input_dir', str(inp), '--output_dir', str(OUT / 'outputs/sauvola_v2_red'),
            '--tmp_dir', str(OUT / 'infer_tmp' / cid), '--mask_dir', str(OUT / 'masks/sauvola_v2_red'),
            '--ckpt', str(checkpoint), '--back_ground', str(paper), '--device', device,
            '--patch_size', '128', '--overlap', '32', '--no_wandb']
        cwd = OUT / 'work' / cid
        cwd.mkdir(parents=True, exist_ok=True)
        log = OUT / 'logs' / (cid + '.log')
        log.parent.mkdir(exist_ok=True)
        print('Native full-page inference:', cid, device, flush=True)
        start = time.monotonic()
        with log.open('w') as f:
            process = subprocess.run(cmd, cwd=cwd,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
                stdout=f, stderr=subprocess.STDOUT)
        assert process.returncode == 0, log
        result = ROOT / c['fixed_sauvola_result']
        assert result.exists() and Image.open(result).size == Image.open(ROOT / c['original']).size
        old, new = read(ROOT / c['old_sauvola_result']), read(result)
        old_red = read(ROOT / c['old_red'])[:, :, 0] < 127
        new_red = read(ROOT / c['v2_red'])[:, :, 0] < 127
        changed = np.any(old != new, axis=2)
        stat = dict(id=cid, seconds=time.monotonic()-start, command=cmd, sha256=sha(result),
            changed_pixels=int(changed.sum()), changed_outside_red_union=int((changed & ~(old_red | new_red)).sum()),
            old_red_pixels=int(old_red.sum()), new_red_pixels=int(new_red.sum()))
        dump(OUT / 'logs' / (cid + '.json'), stat)
        jobs.append(stat)
        print('Complete:', stat, flush=True)
    for path, expected in dependencies.items():
        assert sha(ROOT / path) == expected, path
    dump(OUT / 'complete.json', dict(jobs=jobs, sources_unchanged=True,
        output_sha256={c['id']: sha(ROOT / c['fixed_sauvola_result']) for c in selected}))

if __name__ == '__main__':
    main()
