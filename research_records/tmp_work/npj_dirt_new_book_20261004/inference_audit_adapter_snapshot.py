"""Audit five finished model controls and one copied historical result.

The generic transfer verifier assumed every case had a new inference input.
The model subprocess completed successfully; only that final audit failed on
the intentionally input-less saved seal control. Never rerun inference here.
"""
import json
from pathlib import Path
import shutil

import numpy as np

from build_npj_v6_comparison import ROOT,read,sha,dump
from validate_npj_dirt_new_book import OUT,FROZEN


def main():
    assert not (OUT/'inference_complete.json').exists(),'Already audited'
    selection=json.loads((OUT/'selection.json').read_text())
    for rel,digest in selection['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    assert '[INFO] 完了。' in (OUT/'inference.log').read_text()
    source=json.loads((FROZEN/'complete.json').read_text())
    for rel,digest in source['outputs'].items():assert sha(FROZEN/rel)==digest,rel
    assert len(list((OUT/'input').glob('*.png')))==5
    command=json.loads((OUT/'inference_command.json').read_text())
    frozen=json.loads((FROZEN/'manifest.json').read_text())
    for rel in ['modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth',
                'modules/ddrm_npj_v2/background/isemonogatari.png']:
        assert sha(ROOT/rel)==frozen['dependencies'][rel]
    products={};cases=[]
    for c in selection['cases']:
        p=OUT/'npj_output'/c['id']/'result'/(c['id']+'.png')
        a=read(p);original=read(ROOT/c['original']);x,y,w,h=c['xywh']
        assert a.shape==(h,w,3)
        if c.get('saved_full'):
            assert np.array_equal(a,read(ROOT/c['saved_full'])[y:y+h,x:x+w])
            assert not (OUT/'input'/(c['id']+'.png')).exists()
            kind='Actual historical saved npj crop; model not run'
        else:
            raw=read(OUT/'input'/(c['id']+'.png'))
            assert np.array_equal(raw,original[y:y+h,x:x+w])
            kind='New frozen-code crop control'
        products[str(p.relative_to(OUT))]=sha(p)
        cases.append(dict(id=c['id'],kind=kind,native_dimensions_verified=True))
    shutil.copyfile(__file__,OUT/'inference_audit_adapter_snapshot.py')
    dump(OUT/'inference_complete.json',dict(outputs=products,cases=cases,
         newly_inferred_controls=5,historical_saved_controls=1,
         source_bundle_sha256=sha(FROZEN/'complete.json'),inference_command_sha256=sha(OUT/'inference_command.json'),
         no_inference_rerun=True,model_log_terminal_success=True,
         correction='Generic verifier expected an input for the actual saved seal; do not fabricate that input',
         adapter_sha256=sha(Path(__file__))))
    print('5 completed new crop outputs and 1 historical saved crop verified; inference not repeated',flush=True)


if __name__=='__main__':main()
