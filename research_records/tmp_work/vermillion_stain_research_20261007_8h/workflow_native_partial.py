"""Durable dependent GPU lanes; retain logs and terminate on an actual failure."""
import argparse,json,subprocess,time
from pathlib import Path
from synthetic_layers import OUT

PY='/home/imoto/miniconda3/envs/diffusion/bin/python'
BOOKS=['200020019','200021086','200021763']
def run(script,args):
    argv=[PY,'-u',str(OUT/script),*args];print('RUN',json.dumps(argv),flush=True);subprocess.run(argv,check=True)

def wait_completed(name,pid):
    folder=OUT/'native_partial_trials'/f'{name}_seed715';start=time.monotonic()
    while not (folder/'complete.json').exists():
        stat=Path(f'/proc/{pid}/stat')
        assert stat.exists() and stat.read_text().split()[2]!='Z',f'Prerequisite process terminal without complete: {name} {pid}'
        assert time.monotonic()-start<1800,'Prerequisite timeout; do not restart automatically'
        time.sleep(10)
    assert json.loads((folder/'complete.json').read_text())['steps']==8000

def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=['0','1'],required=True);p.add_argument('--dense-pid',type=int,required=True);p.add_argument('--control-pid',type=int,required=True);a=p.parse_args()
    device='cuda:'+a.lane;wait_completed('dense_native',a.dense_pid)
    if a.lane=='0':
        name='dense_native_brown';run('train_native_partial.py',['--name',name,'--device',device]);run('predict_native_partial.py',['--names',name,'--device',device])
    else:
        wait_completed('sparse_control',a.control_pid);name='dense_native'
        run('predict_native_partial.py',['--names','sparse_control','dense_native','--device',device])
    for book in BOOKS:
        run('train_native_partial.py',['--name',name,'--exclude-dev-book',book,'--device',device])
        run('predict_native_partial.py',['--names',name,'--exclude-dev-book',book,'--device',device])
    methods=['partial_'+name+'_exclude'+b for b in BOOKS]
    run('score_sparse_review.py',['--methods',*methods,'--suffix','leave_book_out_partial_'+name])
    run('score_paper_regions.py',['--methods',*methods,'--suffix','leave_book_out_partial_'+name])
    run('render_actual_red_overlay.py',['--methods','saved_csc_v2_red',*methods,*['region_spatial_forest_exclude'+b for b in BOOKS],'--suffix','leave_book_out_partial_'+name])
    (OUT/'runtime_logs'/('native_partial_lane'+a.lane+'_complete.json')).write_text(json.dumps({'name':name,'books':BOOKS,'device':device},indent=2))

if __name__=='__main__':main()
