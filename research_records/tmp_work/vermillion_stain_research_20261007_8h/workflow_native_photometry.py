"""Follow the verified brown lane with the native-photometry ablation."""
import argparse,json,time
from pathlib import Path
from synthetic_layers import OUT
from workflow_native_partial import BOOKS,run

def main():
    p=argparse.ArgumentParser();p.add_argument('--preceding-pid',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();start=time.monotonic()
    while not (OUT/'runtime_logs/native_partial_lane0_complete.json').exists():
        stat=Path(f'/proc/{a.preceding_pid}/stat')
        assert stat.exists() and stat.read_text().split()[2]!='Z','Brown lane terminal without completion'
        assert time.monotonic()-start<7200,'Brown lane wait exceeded two hours'
        time.sleep(10)
    name='dense_native_photometry';run('train_native_photometry.py',['--name',name,'--device',a.device]);run('predict_native_partial.py',['--names',name,'--device',a.device])
    for book in BOOKS:
        run('train_native_photometry.py',['--name',name,'--exclude-dev-book',book,'--device',a.device])
        run('predict_native_partial.py',['--names',name,'--exclude-dev-book',book,'--device',a.device])
    methods=['partial_'+name+'_exclude'+b for b in BOOKS]
    run('score_sparse_review.py',['--methods',*methods,'--suffix','leave_book_out_partial_'+name])
    run('score_paper_regions.py',['--methods',*methods,'--suffix','leave_book_out_partial_'+name])
    run('render_actual_red_overlay.py',['--methods','saved_csc_v2_red',*methods,*['region_spatial_forest_exclude'+b for b in BOOKS],'--suffix','leave_book_out_partial_'+name])
    (OUT/'runtime_logs/native_photometry_complete.json').write_text(json.dumps(dict(name=name,books=BOOKS,device=a.device),indent=2))

if __name__=='__main__':main()
