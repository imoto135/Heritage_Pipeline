"""Run only changed masks, reusing exact baseline outputs with provenance."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import run_ddrm_pilot as driver

BASE=Path(__file__).resolve().parent
ROOT=driver.ROOT
OUT=BASE/'evidence_gated_howe_20261010'

def main():
    p=json.loads((OUT/'protocol.json').read_text())
    audit=json.loads((OUT/'complete.json').read_text())
    prior=json.loads((BASE/'ddrm_pilot/protocol.json').read_text())
    for file,key in [(driver.PIPELINE,'code_sha256'),(driver.CHECKPOINT,'checkpoint_sha256'),(driver.BACKGROUND,'background_sha256')]:
        assert driver.digest(file)==prior['ddrm'][key]
    for row in audit['results']:
        for rel,expected in row['dependencies'].items():assert driver.digest(ROOT/rel)==expected,rel
    save=OUT/'inputs';results=[];driver.OUT=OUT
    hashes={driver.relative(q):driver.digest(q) for q in (OUT/'masks/automatic').glob('*/*.png')}
    driver.save_json(OUT/'ddrm_protocol.json',dict(ddrm=prior['ddrm'],mask_hashes=hashes,
        rule=p['rule'],development_only=True,multiclass_routing_map_is_ddrm_input=False,
        csc_degradation_used_in_black_mask_selection=True,shared_stepwise_noise_guaranteed=False,
        source_sha256=driver.digest(Path(__file__))))
    for c,row in zip(p['cases'],audit['results']):
        assert c['id']==row['id'];page=c['page'];origin=BASE/c['origin']
        olddir=origin/'masks/howe'/page;newdir=OUT/'masks/automatic'/page
        for name in ['02_text_black.png','03_text_red.png']:
            assert Image.open(olddir/name).size==Image.open(newdir/name).size==tuple(c['input_size'])
        assert np.array_equal(np.asarray(Image.open(olddir/'03_text_red.png')),np.asarray(Image.open(newdir/'03_text_red.png')))
        same=np.array_equal(np.asarray(Image.open(olddir/'02_text_black.png')),np.asarray(Image.open(newdir/'02_text_black.png')))
        baseline=origin/'outputs/howe'/page/'result'/f'{page}.png'
        assert baseline.exists(),baseline
        dest=OUT/'outputs/automatic'/page/'result'/f'{page}.png'
        if same:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to(baseline)
            assert driver.digest(dest)==driver.digest(baseline)
            result=dict(id=c['id'],method='automatic',reused=True,output=driver.relative(dest),sha256=driver.digest(dest),
                        reason='Identical black/red inputs with unchanged frozen inference settings')
        else:
            inputdir=save/page;inputdir.mkdir(parents=True,exist_ok=True);src=ROOT/c['source'];link=inputdir/src.name
            if not link.exists():link.symlink_to(src)
            result=driver.run(c,'automatic',0)
            assert '[ERROR]' not in (OUT/'logs'/f"{c['id']}_automatic.log").read_text(errors='replace')
        result.update(baseline=driver.relative(baseline),baseline_sha256=driver.digest(baseline),removed_pixels=row['removed_pixels'])
        results.append(result)
    for rel,expected in hashes.items():assert driver.digest(ROOT/rel)==expected,rel
    driver.save_json(OUT/'ddrm_complete.json',dict(results=results,new=sum(not r['reused'] for r in results),
        reused=sum(r['reused'] for r in results)))
    print('Verified DDRM outputs',len(results),flush=True)

if __name__=='__main__':main()
