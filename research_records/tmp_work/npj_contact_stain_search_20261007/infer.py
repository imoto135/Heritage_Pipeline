from pathlib import Path
import json,hashlib,subprocess,sys,os,shutil,time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SNAP=ROOT/'tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2'
CKPT=ROOT/'modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'
PAPER=ROOT/'modules/ddrm_npj_v2/background/isemonogatari.png'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows={r['id']:r for r in json.loads((HERE/'screening.json').read_text())['screened_candidates']}
selection=[('T01','P82','本文を広く覆う茶色いシミ'),('T02','P68','本文と小さい文字に重なる2つのシミ'),
 ('T03','P76','本文と小さい文字の上の茶色いシミ'),('T04','P81','本文の細い線に接する茶色いシミ'),
 ('T05','P70','本文途中の薄い小さなシミ'),('T06','P84','本文に接する薄い小さなシミ'),
 ('T07','P51','別資料・本文の一画に接する茶色いシミ'),('C01','P36','本文との重なりが不確かな粒状の汚れ候補')]
if not (HERE/'inference_selection.json').exists():
 chosen=[]
 for idx,(id,pid,reason) in enumerate(selection):
  r=dict(rows[pid]);r.update(test_id=id,selection_reason=reason,selected_before_inference=True)
  inputdir=HERE/'inputs'/id;inputdir.mkdir(parents=True,exist_ok=True)
  x,y,w,h=r['crop_xywh'];box=(x,y,x+w,y+h)
  im=Image.open(ROOT/r['original']).convert('RGB');crop=im.crop(box)
  ip=inputdir/(r['page']+'.png');crop.save(ip)
  assert np.array_equal(np.asarray(crop),np.asarray(Image.open(ip).convert('RGB')))
  maskout=HERE/'masks'/r['page'];maskout.mkdir(parents=True,exist_ok=True)
  for key,name in [('npj_black','02_text_black.png'),('npj_red','03_text_red.png')]:
   m=Image.open(ROOT/r[key]).convert('L');assert m.size==im.size
   m.crop(box).save(maskout/name)
   r[name+'_sha256']=sha(maskout/name)
   assert np.array_equal(np.asarray(m.crop(box)),np.asarray(Image.open(maskout/name)))
  r['actual_input']=str(ip.relative_to(ROOT));r['input_sha256']=sha(ip)
  r['output']=str((HERE/'output'/r['page']/'result'/(r['page']+'.png')).relative_to(ROOT))
  r['source_mask_sha256']={key:sha(ROOT/r[key]) for key in ('npj_black','npj_red')}
  chosen.append(r)
 manifest=dict(cases=chosen,method='NPJ-derived frozen Oct2 local v2 code with saved CSC v1 black/red masks',
  historical_paper_replay=False,full_page=False,manual_mask_override=False,original_pixel_edit=False,
  config=dict(seed=999,timesteps=20,patch_size=128,overlap=32,masking=True,mask_timesteps=50,shared_noise=False),
  source_snapshot=str(SNAP.relative_to(ROOT)),source_code_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(SNAP.rglob('*.py'))},
  checkpoint=str(CKPT.relative_to(ROOT)),checkpoint_sha256=sha(CKPT),background=str(PAPER.relative_to(ROOT)),background_sha256=sha(PAPER),
  original_material_GT=False,selection_protocol='Original + saved black mask reviewed before baseline inference; no output-based threshold tuning',
  reference_case='hnsd007_042 existing Oct4 control, no rerun')
 (HERE/'inference_selection.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
else:manifest=json.loads((HERE/'inference_selection.json').read_text())
(HERE/'logs').mkdir(exist_ok=True)
def run_group(gpu):
 results=[]
 for r in manifest['cases'][gpu::2]:
  id=r['test_id'];output=ROOT/r['output'];cwd=HERE/'work'/id;cwd.mkdir(parents=True,exist_ok=True)
  command=[sys.executable,'-B',str(SNAP/'scripts/pipeline_ddrm.py'),'--input_dir',str(HERE/'inputs'/id),
   '--output_dir',str(HERE/'output'),'--tmp_dir',str(HERE/'inference_tmp'/id),'--mask_dir',str(HERE/'masks'),
   '--ckpt',str(CKPT),'--back_ground',str(PAPER),'--device',f'cuda:{gpu}','--patch_size','128','--overlap','32','--no_wandb']
  print('Inference start',id,r['page'],'gpu',gpu,flush=True);start=time.monotonic()
  if not output.exists():
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR=str(HERE/'plot_cache'))
   with (HERE/'logs'/f'{id}.log').open('w') as log:
    result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
   assert result.returncode==0,(id,result.returncode)
  assert output.exists(),id
  assert Image.open(output).size==tuple(r['crop_xywh'][2:]),id
  log=(HERE/'logs'/f'{id}.log').read_text(errors='replace')
  assert 'ERROR' not in log and 'Otsuにフォールバック' not in log and '推論結果なし' not in log,id
  seconds=round(time.monotonic()-start,2)
  result=dict(id=id,page=r['page'],output=r['output'],sha256=sha(output),shape=list(Image.open(output).size),command=command,seconds=seconds)
  (HERE/'logs'/f'{id}_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));results.append(result)
  print('Inference complete',id,r['page'],seconds,'seconds',flush=True)
 return results
with ThreadPoolExecutor(max_workers=2) as pool:
 groups=list(pool.map(run_group,[0,1]))
for p,h in manifest['source_code_hashes'].items():assert sha(ROOT/p)==h
assert sha(CKPT)==manifest['checkpoint_sha256'] and sha(PAPER)==manifest['background_sha256']
for r in manifest['cases']:
 assert sha(ROOT/r['original'])==r['original_sha256']
 for key,h in r['source_mask_sha256'].items():assert sha(ROOT/r[key])==h
(HERE/'inference_verified.json').write_text(json.dumps(dict(cases=sum(groups,[]),dependencies_unchanged=True),ensure_ascii=False,indent=2))
print('All baselines verified',len(manifest['cases']),flush=True)
