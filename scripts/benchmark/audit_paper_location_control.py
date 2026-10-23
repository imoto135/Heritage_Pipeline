"""Auxiliary readout: numeric rerun variation and additional-paper loss history."""
import json,csv
import numpy as np
from compare_v5_aligned_fade_control import dump,sha,mask
import run_paper_location_control as tr
OUT=tr.OUT;ROOT=tr.ROOT
def main():
    tr.check();result=json.loads((OUT/'results.json').read_text());old=json.loads((tr.NEAR/'results.json').read_text());rows=[];losses=[];summary=[]
    for kind in ['unet','ragan']:
      for seed in [0,1]:
        initial=[]
        for arm in tr.ARMS:
            history=json.loads((OUT/'runs'/f'{kind}_seed{seed}_{arm}/loss_components.json').read_text())
            losses.append(dict(kind=kind,seed=seed,arm=arm,initial=float(history[0]['extra_paper_unweighted']),mean_first216=float(np.mean([h['extra_paper_unweighted'] for h in history[:216]])),mean_last216=float(np.mean([h['extra_paper_unweighted'] for h in history[-216:]]))))
            initial.append({k:history[0][k] for k in ['main','background','mixed_paper','mixed_ink_unweighted']})
        assert initial[0]==initial[1]
        prior={ (r['case'],r['grid']):r for r in old['rows'] if r['kind']==kind and r['seed']==seed and r['arm']=='with_near'}
        current=[r for r in result['rows'] if r['kind']==kind and r['seed']==seed and r['arm']=='near'];assert len(prior)==len(current)==78
        for r in current:
            key=(r['case'],r['grid']);p=prior[key];f=OUT/'evaluation'/r['condition']/'masks'/r['grid']/f"{r['case']}.png";g=tr.NEAR/'evaluation'/p['condition']/'masks'/p['grid']/f"{p['case']}.png";a=mask(f);b=mask(g);assert a.shape==b.shape and sha(f)==r['mask_sha256'] and sha(g)==p['mask_sha256']
            rows.append(dict(kind=kind,seed=seed,case=r['case'],grid=r['grid'],pixel_changes=int((a!=b).sum()),pixels=int(a.size),old_core_r2=p['core_r2'],new_core_r2=r['core_r2'],old_paper_ink=p['paper_ink'],new_paper_ink=r['paper_ink'],old_sha256=sha(g),new_sha256=sha(f)))
        rr=[r for r in rows if r['kind']==kind and r['seed']==seed and r['grid']=='full512' and r['case'] in ['R05','R06']]
        summary.append([kind,seed,sum(r['old_core_r2'] for r in rr),sum(r['new_core_r2'] for r in rr),sum(r['old_paper_ink'] for r in rr),sum(r['new_paper_ink'] for r in rr)])
    dump(OUT/'repeat_and_loss_audit.json',dict(repeated_near_rows=rows,repeated_near_summary=summary,extra_paper_losses=losses,interpretation='Same mathematical near objective and identical teacher masks repeated across sessions, but floating point training trajectories are not guaranteed identical. Not extra initialization seeds, not a confidence interval. Primary contrast remains contemporaneous near/far.',formal_accuracy=False))
    with (OUT/'追加紙損失の推移.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(losses[0]));w.writeheader();w.writerows(losses)
    print('Audited312 repeated-near readouts and8 additional-paper loss histories',flush=True)
if __name__=='__main__':main()
