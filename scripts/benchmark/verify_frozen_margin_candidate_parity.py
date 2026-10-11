"""Check reusable composition against every frozen development page."""
import gc
import json
from pathlib import Path
import shutil
import numpy as np
import frozen_margin_preprocessing_candidate as candidate
import probe_full_upper_strong_masks_v2 as fixed
import prepare_chromatic_external_baselines as references
from build_npj_v6_comparison import dump,read,sha

ROOT=fixed.ROOT
OUT=ROOT/'tmp_work/frozen_margin_candidate_parity_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=fixed.pale.prior.descriptors();checks=[]
    deps={str(Path(candidate.__file__).relative_to(ROOT)):sha(Path(candidate.__file__)),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for module in [candidate.rim,candidate.rim.MARGIN,candidate.rim.MARGIN.second,candidate.top,candidate.thin,candidate.pale,candidate.upper,candidate.direction]:
        deps[str(Path(module.__file__).relative_to(ROOT))]=sha(Path(module.__file__))
    for folder in [fixed.OUT,candidate.direction.OUT]:
        for n in ['manifest.json','complete.json']:deps[str((folder/n).relative_to(ROOT))]=sha(folder/n)
    manifest=dict(dependencies=deps,source_sha256=sha(Path(__file__)),settings=candidate.SETTINGS,pages=25,
        purpose='exact array composition parity, not accuracy or independent evaluation')
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');shutil.copyfile(candidate.__file__,OUT/'candidate_snapshot.py')
    for i,desc in enumerate(inputs,1):
        page=desc['page'];key=page['page'];original=read(ROOT/page['original'])
        v6={k:read(desc['source']/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        features,_=candidate.rim.MARGIN.load_features(key,original,v6)
        result,notes=candidate.infer(original,v6,features)
        expected={'edit':fixed.OUT/'pages'/key/'short_upper/edit.png','strong_core':fixed.OUT/'pages'/key/'short_upper/strong_core.png',
                  'pale_edit':fixed.OUT/'pages'/key/'base_edit.png','core':fixed.OUT/'pages'/key/'core.png',
                  'domain':fixed.OUT/'pages'/key/'domain.png','guard':fixed.OUT/'pages'/key/'guard.png'}
        counts={}
        for n,path in expected.items():
            value=read(path,True);difference=int(np.count_nonzero(value!=result[n]));counts[n]=difference
        record=dict(page=key,mask_differences=counts,original_sha256=sha(ROOT/page['original']))
        checks.append(record);dump(OUT/'progress.json',dict(completed=i,total=25,checks=checks))
        assert not any(counts.values()),record
        known,_=references.full_reference(desc,result['edit'])
        labels,direction_notes,held,retained=candidate.infer_hold(original,result,known)
        assert np.array_equal(held,read(candidate.direction.OUT/'pages'/key/'provisional_matches.png',True))
        assert direction_notes==json.loads((candidate.direction.OUT/'pages'/key/'components.json').read_text())
        record.update(held_pixels=int(held.sum()),directions_exact=True)
        print('Reusable candidate parity:',i,'/25',key,flush=True)
        del original,v6,features,result,labels,known,held,retained;gc.collect()
    for path,h in deps.items():assert sha(ROOT/path)==h
    dump(OUT/'verification.json',dict(status='passed',pages=25,checks=checks,mask_and_direction_bitwise_parity=True,
        no_page_or_ROI_gating=True,pixel_GT=False,features='same frozen original-image features; fresh recomputation not asserted'))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete: reusable composition parity on25 development pages',flush=True)


if __name__=='__main__':main()
