"""Run public pretrained models on fixed original frames, native resolution.

Author model architectures and weights unchanged. Custom 512px native tiling
with 64px halo, disjoint 384px cores; not exact full-page benchmark inference.
No ROI labels or npj pixels enter model inference. No mask-based composition.
"""
import argparse,ast,json,importlib.util,os,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EXTERNAL=ROOT/'tmp_work/npj_literature_models_20261002'
for key,value in [('PYTHONDONTWRITEBYTECODE','1'),('TORCH_HOME',str(EXTERNAL/'cache/torch')),('XDG_CACHE_HOME',str(EXTERNAL/'cache')),('MPLCONFIGDIR',str(EXTERNAL/'cache/mpl'))]:os.environ[key]=value
sys.path.insert(0,str(EXTERNAL/'python_deps'))
import cv2,numpy as np,torch
from build_npj_v6_comparison import read,write,sha,dump
TRIAL=ROOT/'tmp_work/npj_literature_trial_20261002'
SETTINGS=dict(native_resolution=True,resize=False,tile=512,halo=64,core=384,
 padding='reflect101 raw model input+prompt on frame boundary',blend='disjoint cores, halo discarded',
 precision='float32',tf32=False,model_architecture_changed=False,weights_changed=False,
 input='aligned reference original only, not saved npj or masks',
 output='raw external method comparison, not adopted or pasted onto saved npj',pixel_GT=False,
 decoding=dict(stain='first return element, clamp [0,1], RGB to BGR, round*255',
               docres='softmax first 2 channels, argmax=1 is white background as author code'),
 provenance=dict(stain='Author GitHub release Weights/StainDoc.pth, not mark/seal weights',
 docres='Community mirror DaVinciCode/doctra-docres-main, hash agrees with second mirror; original OneDrive requires auth; author-copy identity not independently verified'))

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prompts():
 # Extract only actual author pure preprocessing functions; bypass unrelated
 # dewarping import and checkpoint which are not needed for binarization.
 util=EXTERNAL/'repos/DocRes/utils.py';src=EXTERNAL/'repos/DocRes/inference.py'
 names={'SauvolaModBinarization'}
 tree=ast.parse(util.read_text());ns={'np':np,'cv2':cv2}
 from skimage.filters import threshold_sauvola
 ns['threshold_sauvola']=threshold_sauvola
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(util),'exec'),ns)
 import types
 pns={'np':np,'cv2':cv2,'utils':types.SimpleNamespace(SauvolaModBinarization=ns['SauvolaModBinarization'])}
 tree=ast.parse(src.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'binarization_promptv2','appearance_prompt'}]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),pns)
 return pns

def load(method,device):
 torch.manual_seed(999);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False
 if method=='stain':
  path=EXTERNAL/'repos/StainRestorer/models/StainRestorer.py';weight=EXTERNAL/'weights/StainDoc.pth'
  m=module(path,'stain_official').StainRestorer()
  state=torch.load(weight,map_location='cpu',weights_only=True)['state_dict']
 else:
  path=EXTERNAL/'repos/DocRes/models/restormer_arch.py';weight=EXTERNAL/'weights/docres_mirror.pkl'
  m=module(path,'docres_official').Restormer(inp_channels=6,out_channels=3,dim=48,num_blocks=[2,3,3,4],num_refinement_blocks=4,heads=[1,2,4,8],ffn_expansion_factor=2.66,bias=False,LayerNorm_type='WithBias',dual_pixel_task=True)
  state=torch.load(weight,map_location='cpu',weights_only=True)['model_state']
 assert m.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
 m=m.to(device).eval();return m,[path,weight]

def model_call(model,array,method,device):
 # Explicit channel semantics match each author loader.
 a=array[:,:,::-1].copy() if method=='stain' else array
 t=torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).to(device,dtype=torch.float32)/255
 with torch.inference_mode():
  p=model(t)
  if method=='stain':p=p[0] if isinstance(p,(tuple,list)) else p;p=p.clamp(0,1)
  else:p=torch.softmax(p[:,:2],dim=1)[:,1:2]
  assert torch.isfinite(p).all()
  value=p[0].permute(1,2,0).cpu().numpy()
 if method=='stain':value=value[:,:,::-1].copy()
 return value

def tiled(model,a,method,device):
 h,w=a.shape[:2];core=384;halo=64
 pad=cv2.copyMakeBorder(a,halo,halo+core,halo,halo+core,cv2.BORDER_REFLECT_101)
 channels=3 if method=='stain' else 1
 result=np.empty((h,w,channels),np.float32);coverage=np.zeros((h,w),np.uint8)
 count=0
 for y in range(0,h,core):
  for x in range(0,w,core):
   bh,bw=min(core,h-y),min(core,w-x);v=model_call(model,pad[y:y+512,x:x+512],method,device)
   result[y:y+bh,x:x+bw]=v[halo:halo+bh,halo:halo+bw];coverage[y:y+bh,x:x+bw]+=1;count+=1
 assert (coverage==1).all();return result,count

def main(method,device):
 out=TRIAL/method;assert not (out/'manifest.json').exists(),'Use a new version'
 out.mkdir(parents=True,exist_ok=True);selection=json.loads((TRIAL/'selection.json').read_text())
 model,resources=load(method,device);pns=prompts() if method=='docres' else None
 deps={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),TRIAL/'selection.json',TRIAL/'selection_complete.json',*resources]}
 if method=='docres':
  for p in [EXTERNAL/'repos/DocRes/inference.py',EXTERNAL/'repos/DocRes/utils.py']:deps[str(p.relative_to(ROOT))]=sha(p)
 settings={**SETTINGS,'method':method,'device':device,'torch':torch.__version__}
 dump(out/'manifest.json',dict(settings=settings,dependencies=deps,source_sha=sha(Path(__file__))))
 shutil.copyfile(__file__,out/'source_snapshot.py');rows=[]
 for c in selection['frames']:
  dest=out/c['id'];a=read(TRIAL/'inputs'/c['id']/'original.png');start=time.time()
  if method=='docres':
   # Author prompt computed once on entire native frame, not separately per tile.
   prompt=pns['binarization_promptv2'](a)
   write(dest/'prompt.png',prompt);inp=np.concatenate([a,prompt],axis=2)
  else:inp=a
  v,count=tiled(model,inp,method,device)
  if method=='stain':write(dest/'output.png',np.rint(v*255).astype(np.uint8))
  else:
   np.save(dest/'background_probability.npy',v[:,:,0]);write(dest/'output.png',v[:,:,0]>=.5)
   write(dest/'ink_candidate.png',v[:,:,0]<.5)
  stat=dict(frame=c['id'],tiles=count,elapsed_seconds=time.time()-start,shape=list(a.shape),finite=True,native_pixels=True)
  # Small-frame full forward validates dimensions/channel/padding and measures
  # tile-scope sensitivity on the failure example, without tuning parameters.
  if c['id']=='crop_hnsd':
   h,w=inp.shape[:2];ph=(-h)%32;pw=(-w)%32
   whole=cv2.copyMakeBorder(inp,0,ph,0,pw,cv2.BORDER_REFLECT_101)
   full=model_call(model,whole,method,device)[:h,:w]
   if method=='stain':write(dest/'whole_frame_output.png',np.rint(full*255).astype(np.uint8))
   else:write(dest/'whole_frame_output.png',full[:,:,0]>=.5);np.save(dest/'whole_frame_background_probability.npy',full[:,:,0])
   stat['whole_vs_tiles_mean_float_difference']=float(np.abs(full-v).mean())
  dump(dest/'diagnostics.json',stat);rows.append(stat);print(method,json.dumps(stat),flush=True)
 for rel,digest in deps.items():assert sha(ROOT/rel)==digest,rel
 dump(out/'diagnostics.json',rows);dump(out/'verification.json',dict(strict_all_keys_loaded=True,finite_outputs=True,all_native_pixels_covered_once=True,original_only=True,NPJ_untouched=True,pixel_GT=False,adopted=False))
 dump(out/'complete.json',dict(outputs={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['stain','docres'],required=True);p.add_argument('--device',required=True);a=p.parse_args();main(a.method,a.device)
