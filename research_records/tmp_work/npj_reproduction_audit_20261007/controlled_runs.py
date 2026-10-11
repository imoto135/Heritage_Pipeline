"""Read-only baseline provenance audit and native full-page factor controls.

These are local factor controls, not an authenticated replay of the 2025 paper.
"""
from pathlib import Path
import sys,os,json,subprocess,time,shutil
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
PREV=ROOT/'tmp_work/npj_contact_research_20261007'
sys.path.insert(0,str(PREV))
from common import read,write,dump,sha
from render_full import records
from full_page import SNAP,CKPT,PAPER
import numpy as np
from PIL import Image
OUT=Path(__file__).resolve().parent
CASES=[c for c in records() if c['id'] in ['T01','N01','N02']]
MODEL_CHAR=ROOT/'modules/stain_removal/model/model_char.pth'
MODEL_PAGE=ROOT/'modules/stain_removal/model/model_page.pth'
JOBS=[(id,v) for id in ['T01','N01'] for v in ['local_char_weight','local_page_weight','v2_masks']]+[('N02','v2_masks'),('N02','saved_ACE_input')]
def main():
 lookup={c['id']:c for c in CASES};frozen={str(p.relative_to(ROOT)):sha(p) for p in SNAP.rglob('*.py')}
 deps={str(p.relative_to(ROOT)):sha(p) for p in [CKPT,PAPER,MODEL_CHAR,MODEL_PAGE]}
 selected=[]
 for c in CASES:
  md=ROOT/c['source']['npj_black'];v2=Path(str(md.parent).replace('split_dataset_csc_mask/','split_dataset_csc_mask_v2/'))
  c=dict(c,folder=str(c['folder'].relative_to(ROOT)),source=dict(c['source']),v2_dir=str(v2.relative_to(ROOT)))
  for p in [ROOT/c['source']['original'],md,ROOT/c['source']['npj_red'],md.parent/'00_input.png',v2/'02_text_black.png',v2/'03_text_red.png']:deps[str(p.relative_to(ROOT))]=sha(p)
  selected.append(c)
 dump(OUT/'protocol.json',dict(created=time.time(),cases=selected,jobs=JOBS,source_code=frozen,dependencies=deps,
  baseline='Existing verified full-page output from previous run, epoch350 + v1 masks + isemonogatari background + overlap32',
  variants={'local_char_weight':'Only checkpoint changes to model/model_char.pth; provenance to 2025 paper unverified',
   'local_page_weight':'Only checkpoint changes to model/model_page.pth; original local main_high Args uses model_type=page; paper provenance unverified',
   'v2_masks':'Only black/red masks change to stored CSC v2; this is a later local change, not the paper method',
   'saved_ACE_input':'Only input pixels change to saved classifier ACE input. Same shape/masks/checkpoint/seed/background; raw RGB is not a direct black-text observation in this implementation'},
  seed=999,timesteps=20,patch=128,overlap=32,authentic_paper_replay=False,ground_truth=False))
 def group(gpu):
  done=[]
  for id,variant in JOBS[gpu::2]:
   c=lookup[id];page=c['page'];md=ROOT/c['source']['npj_black'];inp=OUT/'inputs'/variant/id;inp.mkdir(parents=True,exist_ok=True)
   source=md.parent/'00_input.png' if variant=='saved_ACE_input' else ROOT/c['source']['original']
   write(inp/(page+'.png'),read(source))
   masks=OUT/'masks'/variant/page;masks.mkdir(parents=True,exist_ok=True)
   masksource=Path(str(md.parent).replace('split_dataset_csc_mask/','split_dataset_csc_mask_v2/')) if variant=='v2_masks' else md.parent
   for name in ['02_text_black.png','03_text_red.png']:shutil.copyfile(masksource/name,masks/name)
   weight=MODEL_CHAR if variant=='local_char_weight' else MODEL_PAGE if variant=='local_page_weight' else CKPT
   cmd=[sys.executable,'-B',str(SNAP/'scripts/pipeline_ddrm.py'),'--input_dir',str(inp),'--output_dir',str(OUT/'outputs'/variant),
    '--tmp_dir',str(OUT/'infer_tmp'/f'{id}_{variant}'),'--mask_dir',str(OUT/'masks'/variant),'--ckpt',str(weight),
    '--back_ground',str(PAPER),'--device',f'cuda:{gpu}','--patch_size','128','--overlap','32','--no_wandb']
   result=OUT/'outputs'/variant/page/'result'/(page+'.png');log=OUT/'logs'/f'{id}_{variant}.log';log.parent.mkdir(exist_ok=True)
   cwd=OUT/'work'/f'{id}_{variant}';cwd.mkdir(parents=True,exist_ok=True);start=time.monotonic();print('audit start',id,variant,gpu,flush=True)
   with log.open('w') as f:p=subprocess.run(cmd,cwd=cwd,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
   assert p.returncode==0,(id,variant,log)
   assert Image.open(result).size==Image.open(source).size
   baseline=c['folder']/'output/baseline'/page/'result'/(page+'.png');a,b=read(baseline),read(result);red=read(md.parent/'03_text_red.png')[:,:,0]<127
   stat=dict(id=id,variant=variant,seconds=time.monotonic()-start,command=cmd,output=str(result.relative_to(ROOT)),sha256=sha(result),changed_from_previous=int(np.any(a!=b,axis=2).sum()),changed_outside_previous_red=int((np.any(a!=b,axis=2)&~red).sum()))
   dump(OUT/'logs'/f'{id}_{variant}.json',stat);done.append(stat);shutil.rmtree(OUT/'infer_tmp'/f'{id}_{variant}',ignore_errors=True)
   print('audit complete',id,variant,round(stat['seconds'],1),stat['changed_from_previous'],stat['changed_outside_previous_red'],flush=True)
  return done
 with ThreadPoolExecutor(max_workers=2) as pool:results=sum(pool.map(group,[0,1]),[])
 for p,h in {**frozen,**deps}.items():assert sha(ROOT/p)==h,p
 dump(OUT/'complete.json',dict(results=results,source_dependencies_unchanged=True,authentic_paper_replay=False))
if __name__=='__main__':main()
