from common import *
import sys,subprocess,os,time,shutil,argparse
from concurrent.futures import ThreadPoolExecutor
from full_page import SNAP,CKPT,PAPER
def prepare(cohort):
 out=HERE/'full_followup'/cohort;out.mkdir(parents=True,exist_ok=True)
 if (out/'protocol.json').exists():return json.loads((out/'protocol.json').read_text())
 selected=[]
 if cohort=='B':
  for c in cases():
   if c['id'] in ['T00','T01','T02','T03']:continue
   selected.append(dict(id=c['id'],page=c['page'],source=c['source'],detail_page_xywh=c['page_detail_xywh'],role=c['group']))
  for c in json.loads((HERE/'new_cases/selection.json').read_text())['cases']:
   cx,cy,_,_=c['crop_xywh'];x,y,w,h=c['detail_xywh']
   selected.append(dict(id=c['id'],page=c['page'],source=c['source'],detail_page_xywh=[cx+x,cy+y,w,h],role=c['reason']))
 else:
  selected=json.loads((HERE/'other_book_screen/final/cohort_C.json').read_text())['selected']
 protocol=dict(selected=selected,variants=['baseline','sauvola'],source_sha256=sha(__file__),created=time.time(),
  native_full_page=True,output_review_before_selection=False,config='frozen local NPJ-derived seed999 20steps patch128 overlap32; unchanged red mask',
  rationale='page-scope controls for crop artifacts, faint strokes and automatic local-correction coverage',
  source_code={str(p.relative_to(ROOT)):sha(p) for p in SNAP.rglob('*.py')},checkpoint_sha256=sha(CKPT),paper_sha256=sha(PAPER),
  material_GT=False,reading_evaluation=False)
 for c in selected:
  c['source_sha256']={k:sha(ROOT/c['source'][k]) for k in ['original','npj_black','npj_red']}
  im=Image.open(ROOT/c['source']['original']);c['image_size']=list(im.size)
 dump(out/'protocol.json',protocol);return protocol
def main(cohort):
 protocol=prepare(cohort);out=HERE/'full_followup'/cohort
 wait_for=HERE/'full_page/complete.json' if cohort=='B' else HERE/'full_followup/B/complete.json'
 print('full followup frozen',cohort,'cases',len(protocol['selected']),'waiting for preceding cohort',flush=True)
 while not wait_for.exists():time.sleep(5)
 def group(gpu):
  stats=[]
  for c in protocol['selected']:
   variant=protocol['variants'][gpu];id=c['id'];page=c['page'];a=read(ROOT/c['source']['original'])
   inp=out/'inputs'/id;inp.mkdir(parents=True,exist_ok=True);ip=inp/(page+'.png')
   if not ip.exists():write(ip,a)
   md=out/'masks'/variant/page;md.mkdir(parents=True,exist_ok=True)
   if variant=='baseline':shutil.copyfile(ROOT/c['source']['npj_black'],md/'02_text_black.png')
   else:write(md/'02_text_black.png',np.repeat((~sauvola(a))[:,:,None]*255,3,axis=2))
   shutil.copyfile(ROOT/c['source']['npj_red'],md/'03_text_red.png')
   dest=out/'output'/variant;result=dest/page/'result'/(page+'.png');log=out/'logs'/f'{id}_{variant}.log';log.parent.mkdir(exist_ok=True)
   cmd=[sys.executable,'-B',str(SNAP/'scripts/pipeline_ddrm.py'),'--input_dir',str(inp),'--output_dir',str(dest),
    '--tmp_dir',str(out/'infer_tmp'/f'{id}_{variant}'),'--mask_dir',str(out/'masks'/variant),
    '--ckpt',str(CKPT),'--back_ground',str(PAPER),'--device',f'cuda:{gpu}','--patch_size','128','--overlap','32','--no_wandb']
   cwd=out/'work'/f'{id}_{variant}';cwd.mkdir(parents=True,exist_ok=True);start=time.monotonic();print('followup start',cohort,id,variant,flush=True)
   if not result.exists():
    with log.open('w') as f:p=subprocess.run(cmd,cwd=cwd,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
    assert p.returncode==0,(id,variant,log)
   assert Image.open(result).size==Image.open(ip).size
   value=dict(id=id,page=page,variant=variant,seconds=time.monotonic()-start,output=str(result.relative_to(ROOT)),sha256=sha(result),command=cmd)
   dump(out/'logs'/f'{id}_{variant}.json',value);stats.append(value);print('followup complete',cohort,id,variant,round(value['seconds'],1),flush=True)
   shutil.rmtree(out/'infer_tmp'/f'{id}_{variant}')
  return stats
 with ThreadPoolExecutor(max_workers=2) as pool:stats=sum(pool.map(group,[0,1]),[])
 for path,h in protocol['source_code'].items():assert sha(ROOT/path)==h
 for c in protocol['selected']:
  for k,h in c['source_sha256'].items():assert sha(ROOT/c['source'][k])==h
 dump(out/'complete.json',dict(stats=stats,sources_unchanged=True,cohort=cohort))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['B','C'],required=True);a=p.parse_args();main(a.cohort)
