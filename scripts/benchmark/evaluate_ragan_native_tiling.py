"""Native256 tiling with fixed, annotation-independent grids; no training."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
from audit_ragan_inference_context import OUT, PREVIOUS, CONDITIONS, HARD, cases, digest_state
from ragan_pilot_support import make_model, predict, auxiliaries
from compare_v5_aligned_fade_control import coverage, dump, rgb, mask, save, sha
from evaluate_ridge_gate_independent_val import proxies
from evaluate_contact_stain_counterfactual import make_field
import apply_ragan_real_regions as real
import evaluate_classical_partial_labels as partial
import run_ragan_full_continuation as full

CONTACT = ROOT / 'tmp_work/makura_contact_stain_train_control_20261022'
METHODS = {'tiled0': 0, 'tiled16': 16}


def positions(length, offset=0):
    assert length >= 256
    return sorted(set([0, length-256] + list(range(offset, length-255, 128))))


def tiles(page, roi, offset):
    x0, y0, w, h = roi
    return [(x, y) for y in positions(page.shape[0], offset) for x in positions(page.shape[1], offset)
            if x < x0+w and x+256 > x0 and y < y0+h and y+256 > y0]


def tiled_probability(model, page, roi, device, offset):
    x0, y0, w, h = roi
    total = np.zeros((h, w), np.float64)
    second, den = total.copy(), total.copy()
    count, votes = np.zeros((h, w), np.int16), np.zeros((h, w), np.int16)
    seams = np.zeros((h, w), bool)
    axis = np.maximum(np.hanning(256), .05)
    weight = np.outer(axis, axis)
    gx, gy = np.arange(x0, x0+w)[None, :], np.arange(y0, y0+h)[:, None]
    coords = tiles(page, roi, offset)
    for x, y in coords:
        tile = page[y:y+256, x:x+256]
        channels = auxiliaries(tile)
        inp = torch.from_numpy(channels.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device).contiguous() / 255
        with torch.inference_mode():
            p = predict(model, 'ragan', inp)[0, 0].cpu().numpy()
        l, t, r, b = max(x0, x), max(y0, y), min(x0+w, x+256), min(y0+h, y+256)
        local = np.s_[t-y:b-y, l-x:r-x]
        target = np.s_[t-y0:b-y0, l-x0:r-x0]
        ww, pp = weight[local], p[local]
        total[target] += pp * ww
        second[target] += pp.astype(np.float64)**2 * ww
        den[target] += ww
        count[target] += 1
        votes[target] += pp < .5
        for edge in (x, x+256):
            if 0 < edge < page.shape[1]:
                seams |= (np.abs(gx-edge) <= 4) & (gy >= y) & (gy < y+256)
        for edge in (y, y+256):
            if 0 < edge < page.shape[0]:
                seams |= (np.abs(gy-edge) <= 4) & (gx >= x) & (gx < x+256)
    assert np.all(den > 0) and np.all(count > 0)
    mean = total / den
    std = np.sqrt(np.maximum(0, second/den - mean**2))
    overlap = count > 1
    split = (votes > 0) & (votes < count)
    stats = dict(tile_count=len(coords), overlap_n=int(overlap.sum()),
                 disagree_n=int((overlap & split).sum()),
                 weighted_std_mean=float(std[overlap].mean()) if overlap.any() else 0.,
                 weighted_std_max=float(std.max()), seam_definition='within4px of any internal tile edge; overlaps are blended, not hard stitched')
    return mean.astype(np.float32), std, seams, split, stats


def load_cases():
    result = []
    for c in cases():
        row = c['record']
        image, source = rgb(ROOT / row['input']), rgb(ROOT / row['clean'])
        _, _, remote, _ = proxies(source)
        valid, target = full.gray(ROOT / row['labels']['valid']) > 0, full.gray(ROOT / row['labels']['target']) < 128
        result.append(dict(id=c['patch'], family='training_targeted_fade', group='train', page=image, source=source,
            roi=[0, 0, 512, 512], core=c['trace'], paper=remote & valid & ~target,
            files=[row['input'], row['clean'], str(c['trace_path'].relative_to(ROOT)), row['labels']['valid'], row['labels']['target']]))
    prior = json.loads((ROOT / 'tmp_work/ragan_convergence_audit_20261022/evaluation_protocol.json').read_text())
    for patch in prior['synthetic_patches']:
        clean = Path('tmp_work/makura_supervised_pilot_20261021/patches/val') / patch / 'clean.png'
        source = rgb(ROOT / clean)
        paper, core, remote, _ = proxies(source)
        field, _ = make_field(core)
        stain = CONTACT / 'evaluation/inputs' / (patch+'_stain.png')
        images = [('contact_stain', rgb(stain), [str(clean), str(stain.relative_to(ROOT))]), ('clean', source, [str(clean)]),
                  ('uniform_fade_070', np.rint(source.astype(np.float32)*.3 + paper.astype(np.float32)*.7).astype(np.uint8), [str(clean)])]
        for family, image, files in images:
            active = field >= .5 if family == 'contact_stain' else np.ones(core.shape, bool)
            result.append(dict(id=patch+'_'+family, family=family, group='synthetic', page=image, source=source,
                roi=[0, 0, 512, 512], core=core & active, paper=remote & active, files=files))
    patch = prior['known_traced_fade_patch']
    clean = Path('tmp_work/makura_supervised_pilot_20261021/patches/val') / patch / 'clean.png'
    inp = Path('tmp_work/targeted_thin_fade_20261021/cases/alpha_0.70/input.png')
    trace = Path('tmp_work/makura_damage_challenge_20261021/source_thin_review') / (patch+'_high_centers.png')
    source = rgb(ROOT / clean)
    _, _, remote, _ = proxies(source)
    result.append(dict(id='heldout49', family='known_traced_fade_070', group='traced_fade', page=rgb(ROOT/inp), source=source,
        roi=[0, 0, 512, 512], core=full.gray(ROOT/trace)>0, paper=remote, files=list(map(str, [clean, inp, trace]))))
    classic = {c['id']: c for c in json.loads((real.CLASSIC/'protocol.json').read_text())['cases']}
    realcases = [classic[k] for k in ('D01', 'D03', 'D04')] + json.loads((CONTACT/'other_book_transfer/protocol.json').read_text())['cases']
    regions = json.loads((real.PARTIAL/'source_regions.json').read_text())
    ann = {r['id']: r for r in json.loads((real.PARTIAL/'annotations_source_only.json').read_text())['regions']}
    other = json.loads((CONTACT/'other_book_transfer/protocol.json').read_text())
    for c in realcases:
        page = rgb(ROOT/c['source'])
        assert sha(ROOT/c['source']) == c['source_sha256']
        key = c['id']
        x0, y0, w, h = c.get('roi', c.get('crop_xywh'))
        core, paper = np.zeros((h, w), bool), np.zeros((h, w), bool)
        subregions = []
        if key in ('D01', 'D03'):
            for region in [r for r in regions if r['case'] == key]:
                labels, paths = partial.raster(region, ann[region['id']])
                rx, ry, rw, rh = region['roi']
                a = np.s_[ry-y0:ry-y0+rh, rx-x0:rx-x0+rw]
                core[a] |= labels == 1
                paper[a] |= labels == 2
                subregions.append(dict(region=region['id'], xywh=[rx-x0, ry-y0, rw, rh], labels=labels, paths=paths))
        else:
            box = json.loads((CONTACT/'d04_spot_control/protocol.json').read_text())['spot_box_xywh'] if key == 'D04' else other['T07_stain_paper_box_xywh' if key == 'T07' else 'C01_spot_box_xywh']
            rx, ry, rw, rh = box
            paper[ry:ry+rh, rx:rx+rw] = True
        result.append(dict(id=key, family='real', group='real', page=page, source=page[y0:y0+h, x0:x0+w], roi=[x0,y0,w,h],
            core=core, paper=paper, subregions=subregions, files=[c['source'], str((real.PARTIAL/'source_regions.json').relative_to(ROOT)),
                str((real.PARTIAL/'annotations_source_only.json').relative_to(ROOT)), str((CONTACT/'other_book_transfer/protocol.json').relative_to(ROOT)),
                str((CONTACT/'d04_spot_control/protocol.json').relative_to(ROOT))]))
    return result


def cached_path(case, condition):
    folder = PREVIOUS / 'evaluation'
    if case['group'] == 'train':
        return folder / 'training_reviewed_masks' / condition / (case['id']+'_targeted_fade.png')
    if case['group'] == 'traced_fade':
        return folder / 'traced_fade' / (condition+'.png')
    return folder / case['group'] / condition / (case['id']+'.png')


def measure(pred, case):
    core, paper = case['core'], case['paper']
    distance = ndi.distance_transform_edt(~pred)
    return dict(core_n=int(core.sum()), core_exact=int((pred & core).sum()), core_r2=int(((distance <= 2) & core).sum()),
                paper_n=int(paper.sum()), paper_ink=int((pred & paper).sum()), total_ink=int(pred.sum()))


def freeze():
    assert not (OUT/'tiling_protocol.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    allcases = load_cases()
    manifests, sources = [], {}
    for c in allcases:
        for name in c['files']:
            sources[name] = sha(ROOT/name)
        manifests.append(dict(id=c['id'], family=c['family'], group=c['group'], roi=c['roi'], core_n=int(c['core'].sum()), paper_n=int(c['paper'].sum()),
            coordinates={m: tiles(c['page'], c['roi'], offset) for m, offset in METHODS.items()}))
    dump(OUT/'tiling_protocol.json', dict(parent_protocol_sha256=sha(OUT/'protocol.json'), conditions=p['tiled_followup']['conditions'],
        methods=METHODS, cases=manifests, source_hashes=sources, threshold=.5,
        auxiliaries='per256 tile after cropping; no clean or label information', grids_ignore_annotations=True,
        blend='paper probabilities, Hann256 floor.05 outer product', batch_size=1, primary_method='tiled0',
        secondary_grid_not_for_posthoc_selection=True, no_weight_updates=True, formal_accuracy=False))
    shutil.copy2(Path(__file__), OUT/'source_snapshot'/Path(__file__).name)
    print('Frozen tiling:',len(allcases),'cases,',len(p['tiled_followup']['conditions']),'models,2 fixed grids',flush=True)


def run(order, device_name):
    torch.set_num_threads(2)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    p = json.loads((OUT/'tiling_protocol.json').read_text())
    mainp = json.loads((OUT/'protocol.json').read_text())
    assert p['parent_protocol_sha256'] == sha(OUT/'protocol.json')
    assert all(sha(ROOT/name) == h for name,h in p['source_hashes'].items())
    result_path = OUT/f'tiling_order{order}.json'
    assert not result_path.exists()
    allcases = load_cases()
    result, realrows, baseline, states = [], [], [], []
    for condition in p['conditions']:
        if not condition.endswith(f'order{order}'):
            continue
        path = CONDITIONS[condition]
        assert sha(path) == mainp['checkpoints'][condition]['sha256']
        payload = torch.load(path, map_location='cpu', weights_only=False)
        model = make_model('ragan').to(device)
        model.load_state_dict(payload['model'])
        model.eval()
        del payload
        before = digest_state(model)
        for case in allcases:
            key = case['id']
            for method in ['full512_cached', *METHODS]:
                if method == 'full512_cached':
                    source = cached_path(case, condition)
                    pred = mask(source)
                    baseline.append(dict(condition=condition, case=key, original=str(source.relative_to(ROOT)), sha256=sha(source)))
                    pmap, std, seams, split, stats = None, None, None, None, {}
                else:
                    pmap, std, seams, split, stats = tiled_probability(model, case['page'], case['roi'], device, METHODS[method])
                    pred = pmap < .5
                    paper, core = case['paper'], case['core']
                    stats.update(seam_paper_n=int((seams & paper).sum()), seam_paper_ink=int((seams & paper & pred).sum()),
                        other_paper_n=int((~seams & paper).sum()), other_paper_ink=int((~seams & paper & pred).sum()),
                        core_disagree=int((split & core).sum()), paper_disagree=int((split & paper).sum()))
                    save(OUT/'tiling/seam_strips'/condition/method/(key+'.png'), seams)
                    from PIL import Image
                    fp = OUT/'tiling/disagreement'/condition/method/(key+'.png')
                    fp.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(np.rint(np.clip(std*2,0,1)*255).astype(np.uint8)).save(fp)
                    if key in (HARD, 'heldout49'):
                        np.save(fp.with_suffix('.npy'), pmap)
                assert pred.shape == case['core'].shape
                save(OUT/'tiling/masks'/condition/method/(key+'.png'), pred)
                result.append(dict(condition=condition, case=key, group=case['group'], family=case['family'], method=method, **measure(pred,case), **stats))
                if case['group'] == 'real':
                    if case['subregions']:
                        for region in case['subregions']:
                            rx, ry, rw, rh = region['xywh']
                            realrows.append(dict(condition=condition, case=key, method=method, region=region['region'],
                                **partial.measure(pred[ry:ry+rh,rx:rx+rw], region['labels'], region['paths'])))
                    else:
                        realrows.append(dict(condition=condition, case=key, method=method, region='selected_spot_or_stain',
                            nonink_n=int(case['paper'].sum()), nonink_false=int((pred & case['paper']).sum())))
            print(condition,key,[(r['method'],r['core_r2'],r['paper_ink'],r.get('tile_count')) for r in result[-3:]],flush=True)
        after = digest_state(model)
        assert before == after
        states.append(dict(condition=condition, unchanged_state_sha256=before))
        del model
        torch.cuda.empty_cache()
    dump(result_path, dict(rows=result, real_regions=realrows, baseline_cache=baseline, unchanged_states=states,
        tiling_protocol_sha256=sha(OUT/'tiling_protocol.json'), no_weight_updates=True, formal_accuracy=False))
    print('Tiling completed',order,len(result),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--order',type=int,choices=[0,1])
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    freeze() if args.action=='freeze' else run(args.order,args.device)
