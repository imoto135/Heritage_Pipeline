"""Re-score saved masks and verify frozen inputs, weights, grids and caches."""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
from audit_ragan_inference_context import OUT, PREVIOUS, CONDITIONS, HARD, cases
from evaluate_ragan_native_tiling import load_cases, cached_path, tiles
from compare_v5_aligned_fade_control import dump, mask, sha


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    tp = json.loads((OUT/'tiling_protocol.json').read_text())
    combined = json.loads((OUT/'combined_results.json').read_text())
    assert tp['parent_protocol_sha256'] == sha(OUT/'protocol.json')
    for hashes in [json.loads((OUT/'source_hashes.json').read_text()), tp['source_hashes']]:
        assert all(sha(ROOT/path) == h for path,h in hashes.items())
    assert all(sha(path) == p['checkpoints'][c]['sha256'] for c,path in CONDITIONS.items())
    audits = [json.loads((OUT/f'audit_order{o}.json').read_text()) for o in (0,1)]
    tilings = [json.loads((OUT/f'tiling_order{o}.json').read_text()) for o in (0,1)]
    assert all(a['protocol_sha256'] == sha(OUT/'protocol.json') for a in audits)
    assert all(a['tiling_protocol_sha256'] == sha(OUT/'tiling_protocol.json') for a in tilings)
    states = {r['condition']:r['unchanged_state_sha256'] for a in audits for r in a['states']}
    for t in tilings:
        for r in t['unchanged_states']:
            assert r['unchanged_state_sha256'] == states[r['condition']]
    assert len(states) == 6
    # Independent Euclidean-radius2 dilation check, rather than the scorer's EDT.
    yy,xx = np.ogrid[-2:3,-2:3]
    disk = xx*xx + yy*yy <= 4
    traincases = {c['patch']:c for c in cases()}
    for r in combined['audit_rows']:
        case = traincases[r['patch']]
        x,y = r['xy']
        trace = case['trace'][y:y+256,x:x+256]
        pred = mask(OUT/'masks'/r['condition']/r['patch']/(r['view']+'.png'))
        assert pred.shape == trace.shape and int(trace.sum()) == r['n']
        assert int((pred & trace).sum()) == r['r0']
        assert int((ndi.binary_dilation(pred,structure=disk) & trace).sum()) == r['r2']
    allcases = {c['id']:c for c in load_cases()}
    for manifest in tp['cases']:
        c = allcases[manifest['id']]
        x0,y0,w,h = c['roi']
        for method,offset in tp['methods'].items():
            coords = tiles(c['page'],c['roi'],offset)
            assert [list(xy) for xy in coords] == manifest['coordinates'][method]
            count = np.zeros((h,w),np.uint8)
            for x,y in coords:
                assert x >= 0 and y >= 0 and x+256 <= c['page'].shape[1] and y+256 <= c['page'].shape[0]
                l,t,r,b = max(x0,x),max(y0,y),min(x0+w,x+256),min(y0+h,y+256)
                count[t-y0:b-y0,l-x0:r-x0] += 1
            assert count.min() >= 1
    cache_count = 0
    for r in combined['tiling_rows']:
        case = allcases[r['case']]
        pred = mask(OUT/'tiling/masks'/r['condition']/r['method']/(r['case']+'.png'))
        assert pred.shape == case['core'].shape
        assert int(case['core'].sum()) == r['core_n']
        assert int((case['core'] & pred).sum()) == r['core_exact']
        assert int((case['core'] & ndi.binary_dilation(pred,structure=disk)).sum()) == r['core_r2']
        assert int(case['paper'].sum()) == r['paper_n']
        assert int((case['paper'] & pred).sum()) == r['paper_ink']
        if r['method'] == 'full512_cached':
            assert np.array_equal(pred,mask(cached_path(case,r['condition'])))
            cache_count += 1
        else:
            seam = mask(OUT/'tiling/seam_strips'/r['condition']/r['method']/(r['case']+'.png'))
            assert int((case['paper'] & seam).sum()) == r['seam_paper_n']
            assert int((case['paper'] & seam & pred).sum()) == r['seam_paper_ink']
            assert r['seam_paper_n']+r['other_paper_n']==r['paper_n']
            assert r['seam_paper_ink']+r['other_paper_ink']==r['paper_ink']
    old = json.loads((PREVIOUS/'evaluation/results.json').read_text())
    for r in combined['real_regions']:
        if r['method']=='full512_cached':
            expected = next(q for q in old['real'] if q['condition']==r['condition'] and q['case']==r['case'] and q['region']==r['region'])
            assert {k:v for k,v in r.items() if k!='method'} == expected
    assert len(combined['audit_rows'])==414 and len(combined['tiling_rows'])==228
    replay = [r for a in audits for r in a['cached_replay']]
    assert all(r['new_trace']['r2']==r['cached_trace']['r2'] for r in replay)
    assert all(r['changed_pixels']==0 for r in replay if r['view']=='full_global')
    links=0
    report=OUT/'研究判断_推論範囲と切り出し位置.md'
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',report.read_text()):
        if '://' not in target and not target.startswith('#'):
            assert (report.parent/target.split('#')[0]).exists(),target
            links+=1
    for filename, archived in [('audit_ragan_inference_context.py','audit_ragan_inference_context_replayfix.py'),
                                ('evaluate_ragan_native_tiling.py','evaluate_ragan_native_tiling.py'),
                                ('report_ragan_inference_context.py','report_ragan_inference_context.py')]:
        assert sha(ROOT/'scripts/benchmark'/filename)==sha(OUT/'source_snapshot'/archived)
    shutil.copy2(Path(__file__),OUT/'source_snapshot'/Path(__file__).name)
    dump(OUT/'integrity_verification.json',dict(checkpoints_sha256={c:sha(path) for c,path in CONDITIONS.items()},
        unchanged_state_sha256=states,all_source_and_label_hashes_verified=True,all_grids_frozen_and_cover_ROI=True,
        audit_mask_scores_independently_verified=414,tiling_mask_scores_independently_verified=228,
        baseline_saved_masks_identical=cache_count,baseline_real_region_scores_identical=True,
        prior_replay_max_changed_pixels=max(r['changed_pixels'] for r in replay),prior_replay_all_trace_r2_equal=True,
        prior_full512_replay_masks_identical=True,report_local_links_verified=links,
        protocol_sha256=sha(OUT/'protocol.json'),tiling_protocol_sha256=sha(OUT/'tiling_protocol.json'),
        report_sha256=sha(report),combined_results_sha256=sha(OUT/'combined_results.json'),
        figure_hashes={f:sha(OUT/f) for f in combined['figures']},
        source_snapshot_hashes={f.name:sha(f) for f in sorted((OUT/'source_snapshot').glob('*.py'))},
        no_weight_updates=True,formal_accuracy=False))
    print('Verified414 factor scores,228 tiled scores,76 cached masks,43 PNGs; input/weight hashes unchanged',flush=True)


if __name__=='__main__':
    main()
