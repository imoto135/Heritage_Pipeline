from common import *
import sys,os,subprocess,time,shutil,argparse
from concurrent.futures import ThreadPoolExecutor
SNAP=ROOT/'tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2'
CKPT=ROOT/'modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'
PAPER=ROOT/'modules/ddrm_npj_v2/background/isemonogatari.png'
def prepare():
 out=HERE/'full_page';out.mkdir(exist_ok=True)
 if (out/'prepared.json').exists():return json.loads((out/'prepared.json').read_text())
 selected=[c for c in cases() if c['id'] in ['T00','T01','T02','T03']]
 protocol=dict(selected=selected,variants=['baseline','sauvola'],
  crop=False,resize=False,mask_variant='saved CSC v1 versus global native Sauvola w61 k.2',
  red_mask='unchanged saved source in both variants',seed=999,timesteps=20,patch=128,overlap=32,
  note='local frozen NPJ-derived implementation, not a complete paper replay',
  baseline_source_code={str(p.relative_to(ROOT)):sha(p) for p in SNAP.rglob('*.py')},
  ckpt_sha256=sha(CKPT),paper_sha256=sha(PAPER),source_sha256=sha(__file__),created=time.time())
 dump(out/'protocol.json',protocol)
 for c in selected:
  id=c['id'];a=read(ROOT/c['source']['original']);write(out/'inputs'/id/(c['page']+'.png'),a)
  for variant in ['baseline','sauvola']:
   md=out/'masks'/variant/c['page'];md.mkdir(parents=True,exist_ok=True)
   if variant=='baseline':shutil.copyfile(ROOT/c['source']['npj_black'],md/'02_text_black.png')
   else:write(md/'02_text_black.png',np.repeat((~sauvola(a))[:,:,None]*255,3,axis=2))
   shutil.copyfile(ROOT/c['source']['npj_red'],md/'03_text_red.png')
   if variant=='baseline':assert sha(md/'02_text_black.png')==c['source_sha256']['npj_black']
 dump(out/'prepared.json',protocol);return protocol
def run():
 protocol=prepare();out=HERE/'full_page';jobs=[(c,v) for c in protocol['selected'] for v in protocol['variants']]
 def group(gpu):
  done=[]
  for c,v in jobs[gpu::2]:
   id=c['id'];directory=out/'output'/v;result=directory/c['page']/'result'/(c['page']+'.png')
   log=out/'logs'/f'{id}_{v}.log';log.parent.mkdir(exist_ok=True)
   cmd=[sys.executable,'-B',str(SNAP/'scripts/pipeline_ddrm.py'),'--input_dir',str(out/'inputs'/id),
    '--output_dir',str(directory),'--tmp_dir',str(out/'infer_tmp'/f'{id}_{v}'),
    '--mask_dir',str(out/'masks'/v),'--ckpt',str(CKPT),'--back_ground',str(PAPER),
    '--device',f'cuda:{gpu}','--patch_size','128','--overlap','32','--no_wandb']
   start=time.monotonic();print('full page start',id,v,'gpu',gpu,flush=True)
   if not result.exists():
    cwd=out/'work'/f'{id}_{v}';cwd.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with log.open('w') as f:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
    assert r.returncode==0,(id,v,log)
   assert Image.open(result).size==Image.open(ROOT/c['source']['original']).size
   stat=dict(id=id,variant=v,seconds=time.monotonic()-start,output=str(result.relative_to(ROOT)),output_sha256=sha(result),command=cmd)
   dump(out/'logs'/f'{id}_{v}.json',stat);done.append(stat);print('full page complete',id,v,round(stat['seconds'],1),flush=True)
  return done
 with ThreadPoolExecutor(max_workers=2) as pool:results=sum(pool.map(group,[0,1]),[])
 for p,h in protocol['baseline_source_code'].items():assert sha(ROOT/p)==h
 assert sha(CKPT)==protocol['ckpt_sha256'] and sha(PAPER)==protocol['paper_sha256']
 dump(out/'complete.json',dict(results=results,sources_unchanged=True,full_native=True))
if __name__=='__main__':run()
