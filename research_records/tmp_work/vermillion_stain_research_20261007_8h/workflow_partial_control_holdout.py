"""Matched point-only holdout control follows the completed dense lane."""
import argparse,json,time
from pathlib import Path
from synthetic_layers import OUT
from workflow_native_partial import BOOKS,run

def main():
    p=argparse.ArgumentParser();p.add_argument('--preceding-pid',type=int,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args();start=time.monotonic()
    prerequisite=OUT/'runtime_logs/native_partial_lane1_complete.json'
    while not prerequisite.exists():
        stat=Path(f'/proc/{a.preceding_pid}/stat')
        assert stat.exists() and stat.read_text().split()[2]!='Z','Preceding lane ended without completing; stop rather than rerun'
        assert time.monotonic()-start<7200,'Preceding lane wait exceeded two hours'
        time.sleep(10)
    for book in BOOKS:
        run('train_native_partial.py',['--name','sparse_control','--exclude-dev-book',book,'--device',a.device])
        run('predict_native_partial.py',['--names','sparse_control','--exclude-dev-book',book,'--device',a.device])
    methods=['partial_sparse_control_exclude'+b for b in BOOKS]
    run('score_sparse_review.py',['--methods',*methods,'--suffix','leave_book_out_partial_sparse_control'])
    run('score_paper_regions.py',['--methods',*methods,'--suffix','leave_book_out_partial_sparse_control'])
    run('render_actual_red_overlay.py',['--methods','saved_csc_v2_red',*methods,*['region_spatial_forest_exclude'+b for b in BOOKS],'--suffix','leave_book_out_partial_sparse_control'])
    (OUT/'runtime_logs/native_partial_control_holdout_complete.json').write_text(json.dumps(dict(name='sparse_control',books=BOOKS,device=a.device),indent=2))

if __name__=='__main__':main()
