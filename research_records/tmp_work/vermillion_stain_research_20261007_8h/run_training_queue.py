"""Sequential GPU-local experiments, with immutable source snapshots and logs."""
import argparse
from pathlib import Path
import json
import subprocess
import shutil
import sys
from datetime import datetime, timezone

OUT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--names',nargs='+',required=True)
    p.add_argument('--steps',type=int,default=12000);a=p.parse_args()
    queue=OUT/'training_queues';queue.mkdir(exist_ok=True)
    key=a.device.replace(':','_');record=dict(device=a.device,names=a.names,steps=a.steps,
        started_utc=datetime.now(timezone.utc).isoformat(),completed=[],failed=[],status='running')
    for name in a.names:
        dest=OUT/'learned_trials'/f'{name}_seed413';dest.mkdir(parents=True,exist_ok=True)
        snap=dest/'source';snap.mkdir(exist_ok=True)
        for f in ['train_layer_selector.py','synthetic_layers.py','cases_frozen.json']:
            shutil.copyfile(OUT/f,snap/f)
        record['current']=name;(queue/(key+'.json')).write_text(json.dumps(record,indent=2))
        cmd=[sys.executable,str(OUT/'train_layer_selector.py'),'--name',name,'--device',a.device,'--steps',str(a.steps)]
        print('Starting',name,a.device,flush=True)
        with (dest/'training.log').open('w') as log:
            run=subprocess.run(cmd,cwd=OUT.parents[1],stdout=log,stderr=subprocess.STDOUT)
        if run.returncode:record['failed'].append(dict(name=name,code=run.returncode))
        else:record['completed'].append(name)
        print('Finished',name,'code',run.returncode,flush=True)
        record['current']=None;(queue/(key+'.json')).write_text(json.dumps(record,indent=2))
    record['status']='completed_with_failures' if record['failed'] else 'completed'
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    (queue/(key+'.json')).write_text(json.dumps(record,indent=2))

if __name__=='__main__':main()
