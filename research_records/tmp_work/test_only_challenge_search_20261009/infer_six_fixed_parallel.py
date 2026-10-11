"""Resolve the six-case audit, parallelizing only required new inference jobs."""
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import json
from pathlib import Path
import review_six_fixed as v

def main():
    proto=json.loads((v.OUT/'protocol.json').read_text());audit=json.loads((v.OUT/'mask_audit.json').read_text())
    expected={(r['id'],r['method']):r for r in json.loads((v.PRIOR/'complete.json').read_text())['results']}
    v.driver.OUT=v.OUT;results=[];pending=[]
    for c in proto['cases']:
        row=next(r for r in audit if r['id']==c['id'])
        for method in v.METHODS:
            if not row['methods'][method]['same_as_previous_input']:
                pending.append((c,method));continue
            previous=expected[c['id'],method];src=v.ROOT/previous['output']
            assert v.driver.digest(src)==previous['sha256'] and Image.open(src).size==tuple(c['input_size'])
            log=v.PRIOR/'logs'/(c['id']+'_'+method+'.log');assert '[ERROR]' not in log.read_text(errors='replace')
            dest=v.OUT/'outputs'/method/c['page']/'result'/(c['page']+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to(src)
            results.append(dict(id=c['id'],method=method,reused=True,
                reason='Pixel-identical black/red, unchanged source, model, code, background and sampling settings',
                output=v.driver.relative(dest),sha256=v.driver.digest(dest),previous_output=previous['output'],previous_log_sha256=v.driver.digest(log)))
    v.driver.save_json(v.OUT/'inference_plan.json',dict(new_jobs=[dict(id=c['id'],method=m) for c,m in pending],
        reused_outputs=len(results),runner_sha256=v.driver.digest(Path(__file__)),
        preparation_script_sha256=v.driver.digest(Path(v.__file__)),same_stepwise_noise_between_black_masks_guaranteed=False))
    def group(jobs,gpu):
        out=[]
        for c,method in jobs:
            r=v.driver.run(c,method,gpu)
            assert '[ERROR]' not in (v.OUT/'logs'/(c['id']+'_'+method+'.log')).read_text(errors='replace');out.append(r)
        return out
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(group,pending[i::2],i) for i in range(2)]
        for f in futures:results.extend(f.result())
    assert len(results)==18
    v.driver.save_json(v.OUT/'inference_audit.json',dict(results=results,new=len(pending),reused=18-len(pending)))
    print('Verified 18 outputs: new',len(pending),'reused',18-len(pending),flush=True)

if __name__=='__main__':main()
