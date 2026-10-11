"""One final dev-only causal ablation, no independent checkpoint selection."""
import json,time,subprocess,sys,datetime
from synthetic_layers import OUT,ROOT,file_hash
while not (OUT/'runtime_logs/stain_pair_control_complete.json').exists():time.sleep(15)
def run(f,*args):
 cmd=[sys.executable,'-u',str(OUT/f),*args];print('RUN',cmd,flush=True);subprocess.run(cmd,check=True,cwd=OUT)
for book in [None,'200020019','200021086','200021763']:
 args=['--exclude-dev-book',book] if book else []
 run('train_native_relative_only.py','--name','dense_native_relative_only','--device','cuda:1',*args)
 run('predict_relative_only.py','--device','cuda:1',*args)
 if not book:
  # Weight resource registration follows design registration and precedes independent prediction/review.
  parent=OUT/'independent_evaluation_protocol_v2.json';pr=json.loads(parent.read_text());pr['version']=3;pr['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();pr['predecessor_sha256']=file_hash(parent);ck=OUT/'native_partial_trials/dense_native_relative_only_seed715/last.pth';pr['model_resources']['partial_dense_native_relative_only']=dict(path=str(ck.relative_to(ROOT)),sha256=file_hash(ck),step=8000,relative_only=True,loader='relative_only_selector.py');pr['relative_only_design_sha256']=file_hash(OUT/'relative_only_ablation_registered.json');q=OUT/'independent_evaluation_protocol_v3.json';assert not q.exists();q.write_text(json.dumps(pr,ensure_ascii=False,indent=2));run('predict_relative_only.py','--device','cuda:1','--independent')
methods=['partial_dense_native_relative_only_exclude'+b for b in ['200020019','200021086','200021763']]
run('score_sparse_review.py','--methods',*methods,'--suffix','leave_book_out_partial_dense_native_relative_only')
run('score_paper_regions.py','--methods',*methods,'--suffix','leave_book_out_partial_dense_native_relative_only')
run('render_actual_red_overlay.py','--methods','saved_csc_v2_red',*methods,'--suffix','leave_book_out_partial_dense_native_relative_only')
(OUT/'runtime_logs/relative_only_complete.json').write_text(json.dumps(dict(method='dense_native_relative_only',folds=4)))
