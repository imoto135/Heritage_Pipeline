"""Resolve input confounding for the two preselected nonidentical references."""
import argparse
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from run_npj_literature_models import ROOT, TRIAL, EXTERNAL, SETTINGS, load, prompts, tiled
from build_npj_v6_comparison import read, write, sha, dump
from package_saved_npj_inference_v6_review import check


def main(method, device):
    out = TRIAL/('matched_'+method)
    assert not (out/'manifest.json').exists(), 'Use a new version'
    out.mkdir(parents=True, exist_ok=True)
    check(TRIAL/method)
    frames = [f for f in json.loads((TRIAL/'selection.json').read_text())['frames']
              if not f['input_reference_exact']]
    assert {f['id'] for f in frames} == {'crop_hnsd', 'page_200017197'}
    model, resources = load(method, device)
    pns = prompts() if method == 'docres' else None
    resources += [Path(__file__), Path(__file__).parent/'run_npj_literature_models.py',
                  TRIAL/'selection.json', TRIAL/'selection_complete.json']
    if method == 'docres':
        resources += [EXTERNAL/'repos/DocRes/inference.py', EXTERNAL/'repos/DocRes/utils.py']
    resources += [TRIAL/'inputs'/f['id']/'saved_input.png' for f in frames]
    deps = {str(p.relative_to(ROOT)):sha(p) for p in resources}
    dump(out/'manifest.json', dict(dependencies=deps, settings={**SETTINGS,
        'input':'actual historical saved npj input; matched input control only',
        'method':method, 'device':device}, frames=[f['id'] for f in frames]))
    shutil.copyfile(__file__, out/'source_snapshot.py')
    rows = []
    for f in frames:
        a = read(TRIAL/'inputs'/f['id']/'saved_input.png'); dest = out/f['id']
        a_historical = read(ROOT/f['input'])
        assert np.array_equal(a, a_historical), 'Exact historical input'
        inp = a if method == 'stain' else np.concatenate([a,pns['binarization_promptv2'](a)],axis=2)
        start = time.time(); v,count = tiled(model,inp,method,device)
        if method == 'stain': write(dest/'output.png',np.rint(v*255).astype(np.uint8))
        else:
            write(dest/'output.png', v[:,:,0] >= .5)
            np.save(dest/'background_probability.npy',v[:,:,0])
        rows.append(dict(frame=f['id'],tiles=count,seconds=time.time()-start,
                         historical_input_exact=True,native_shape=list(a.shape)))
        print(method,rows[-1],flush=True)
    for rel,digest in deps.items(): assert sha(ROOT/rel) == digest,rel
    dump(out/'verification.json',dict(historical_inputs_exact=True,finite=True,
        native_coverage_once=True,strict_weights=True,pixel_GT=False,adopted=False))
    dump(out/'diagnostics.json',rows)
    dump(out/'complete.json',dict(outputs={str(p.relative_to(out)):sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['stain','docres'],required=True)
    p.add_argument('--device',required=True);args=p.parse_args();main(args.method,args.device)
