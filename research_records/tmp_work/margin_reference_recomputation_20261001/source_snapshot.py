"""Check fresh full-page known-paper gate against historical frozen donors."""
import gc
import json
from pathlib import Path
import shutil
import numpy as np
import probe_full_upper_strong_masks_v2 as fixed
import prepare_chromatic_external_baselines as reference
import probe_paper_tone_boundary as paper
from build_npj_v6_comparison import dump,read,sha

ROOT=fixed.ROOT
OUT=ROOT/'tmp_work/margin_reference_recomputation_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    deps={str(Path(m.__file__).relative_to(ROOT)):sha(Path(m.__file__)) for m in [fixed,reference,paper]}
    deps[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=deps,pages=25,reference='same clean_reference(...)[1] qualified base',pixel_GT=False)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=[]
    for i,desc in enumerate(fixed.pale.prior.descriptors(),1):
        key=desc['page']['page'];original=read(ROOT/desc['page']['original'])
        v6={k:read(desc['source']/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        folder=fixed.OUT/'pages'/key;edit=read(folder/'short_upper/edit.png',True);domain=read(folder/'domain.png',True);guard=read(folder/'guard.png',True)
        protected=v6['protect']|v6['hold']|v6['red']|v6['extra']|guard
        fresh=paper.clean_reference(original,edit,protected&~edit,domain&~guard)[1]
        expected,path=reference.full_reference(desc,edit)
        row=dict(page=key,difference=int(np.count_nonzero(fresh!=expected)),known_pixels=int(fresh.sum()),reference_sha256=sha(path));rows.append(row)
        dump(OUT/'progress.json',dict(completed=i,total=25,rows=rows))
        assert row['difference']==0,row
        print('Fresh reference parity:',i,'/25',key,flush=True)
        del original,v6,edit,domain,guard,protected,fresh,expected;gc.collect()
    for path,h in deps.items():assert sha(ROOT/path)==h
    dump(OUT/'verification.json',dict(status='passed',pages=25,rows=rows,qualified_paper_recomputed_exactly=True,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:25 pages fresh known-paper gate exactly equals frozen references',flush=True)


if __name__=='__main__':main()
