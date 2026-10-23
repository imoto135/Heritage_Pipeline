"""Complete the exploratory BCE-formulation x exposure 2x2, without new knobs."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_ragan_stable_bce_control as stable
from run_ragan_fine_retention_control import OUT as FREQUENCY
from compare_v5_aligned_fade_control import sha,dump
OUT=ROOT/'tmp_work/ragan_stable_exposure_control_20261022'
BASE=stable.OUT
CONDITIONS={f'stable_exposure_order{n}':dict(gan=False,batch_order_seed=20261021+n,order=n) for n in (0,1)}

def freeze():
 assert not OUT.exists();OUT.mkdir()
 for name in ['source_hashes.json','crop_schedule.json','fine_exposure_audit.json','evaluation_protocol.json','loss_checks.json']:shutil.copy2(BASE/name,OUT/name)
 p=json.loads((BASE/'protocol.json').read_text());p.update(conditions=CONDITIONS,
  parent_stable_protocol_sha256=sha(BASE/'protocol.json'),frequency_protocol_sha256=sha(FREQUENCY/'protocol.json'),
  changed_factor='strong fade20% schedule versus stable BCE normal exposure; unchanged stable loss',
  exposure_target_draws_per_epoch=173,exploratory_2x2_completion=True,
  one_factor_comparator=str(BASE.relative_to(ROOT)),
  other_cells='previous original BCE normal + original BCE20%, both two orders')
 dump(OUT/'protocol.json',p);(OUT/'batch_schedules').mkdir()
 for n in (0,1):shutil.copy2(FREQUENCY/'batch_schedules'/f'exposure20_order{n}.json',OUT/'batch_schedules'/f'stable_exposure_order{n}.json')
 snapshot=OUT/'source_snapshot';snapshot.mkdir()
 for name in [Path(__file__).name,'run_ragan_stable_bce_control.py','run_ragan_full_continuation.py','ragan_pilot_support.py','evaluate_ragan_fine_retention_control.py','evaluate_ragan_joint_continuation.py']:
  shutil.copy2(ROOT/'scripts/benchmark'/name,snapshot/name)
 print('Frozen stable BCE + original20% schedules; two unchanged order controls',flush=True)

def train(condition,device):
 stable.OUT=OUT;stable.CONDITIONS=CONDITIONS;stable.train(condition,device)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','train']);p.add_argument('--condition',choices=list(CONDITIONS));p.add_argument('--device',default='cuda:0');a=p.parse_args()
 if a.action=='freeze':freeze()
 else:train(a.condition,a.device)
