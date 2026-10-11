"""Recompute original-derived features without the historical page cache."""
import gc
import json
from pathlib import Path
import shutil
import numpy as np
import frozen_margin_preprocessing_candidate as candidate
import probe_full_upper_strong_masks_v2 as fixed
from build_npj_v6_comparison import dump,read,sha

ROOT=fixed.ROOT
OUT=ROOT/'tmp_work/margin_original_feature_recomputation_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    modules=[candidate,candidate.rim.MARGIN,candidate.rim.MARGIN.first,candidate.rim.MARGIN.second]
    deps={str(Path(m.__file__).relative_to(ROOT)):sha(Path(m.__file__)) for m in modules}
    deps[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=deps,pages=25,
        recomputation='second.infer(original,v6), no cached feature loader',pixel_GT=False)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=[]
    for i,desc in enumerate(fixed.pale.prior.descriptors(),1):
        key=desc['page']['page'];original=read(ROOT/desc['page']['original'])
        v6={k:read(desc['source']/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        frozen,_=candidate.rim.MARGIN.load_features(key,original,v6)
        _,fresh,_=candidate.rim.MARGIN.second.infer(original,v6)
        differences={k:int(np.count_nonzero(frozen[k]!=fresh[k])) for k in ['light_drop','color_seed','color_soft','warm_anchor_guard']}
        rows.append(dict(page=key,differences=differences,original_sha256=sha(ROOT/desc['page']['original'])))
        dump(OUT/'progress.json',dict(completed=i,total=25,rows=rows))
        assert not any(differences.values()),rows[-1]
        print('Fresh original features:',i,'/25',key,flush=True)
        del original,v6,frozen,fresh;gc.collect()
    for n,h in deps.items():assert sha(ROOT/n)==h
    dump(OUT/'verification.json',dict(status='passed',pages=25,rows=rows,fresh_features_bitwise_identical=True,
        thresholds_unchanged=True,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:25 pages fresh original features exactly match history',flush=True)


if __name__=='__main__':main()
