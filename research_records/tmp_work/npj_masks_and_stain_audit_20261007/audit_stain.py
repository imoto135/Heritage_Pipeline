from pathlib import Path
import sys,os,json,importlib.util,hashlib,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
EX=ROOT/'tmp_work/npj_literature_models_20261002';REPO=OUT/'official/StainRestorer';PREV=ROOT/'tmp_work/npj_contact_research_20261007'
os.environ['TORCH_HOME']=str(OUT/'cache/torch');os.environ['XDG_CACHE_HOME']=str(OUT/'cache');os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(OUT/'python_deps'));sys.path.insert(0,str(EX/'python_deps'));sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import cv2,numpy as np,torch,torchvision
from PIL import Image,ImageDraw,ImageFont
from torchvision.utils import save_image
import run_npj_literature_models as adapter

def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def read(p):return np.asarray(Image.open(p).convert('RGB')).copy()
def save(p,a):p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.clip(np.rint(a),0,255).astype('uint8')).save(p)
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(a,b,mask=None):
 if mask is None:mask=np.ones(a.shape[:2],bool)
 delta=b-a
 return dict(raw_RGB_delta_255=(delta[mask].mean(0)*255).tolist(),green_balance_delta_255=float((delta[:,:,1]-.5*(delta[:,:,0]+delta[:,:,2]))[mask].mean()*255),outside_0_1_fraction=float(((b<0)|(b>1)).mean()),input_range=[float(a.min()),float(a.max())],output_range=[float(b.min()),float(b.max())])

def main():
 torch.set_num_threads(4);torch.manual_seed(3407);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model_module=module(REPO/'models/StainRestorer.py','stain_fresh_official');dataset_module=module(REPO/'data/dataset_RGB.py','loader_fresh_official')
 weight=OUT/'official/StainDoc_fresh.pth';model=model_module.StainRestorer();state=torch.load(weight,map_location='cpu',weights_only=True)['state_dict'];load=model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);assert not load.missing_keys and not load.unexpected_keys;model=model.cuda(0).eval()
 deps=[REPO/'models/StainRestorer.py',REPO/'data/dataset_RGB.py',weight,ROOT/'scripts/benchmark/run_npj_literature_models.py',Path(__file__)]
 rows={c['id']:c for c in json.loads((PREV/'real_models/protocol.json').read_text())['selected']};frozen={str(p.relative_to(ROOT)):sha(p) for p in deps};result=[]
 dump(OUT/'stain_protocol.json',dict(dependencies=frozen,selected=['T01','N01','N02','F41'],fresh_author_weights=True,actual_author_DataLoaderVal=True,loader_package='albumentations1.3.1 local isolated dependency; uses Resize linear only; author current environment lists2.0.8',torch=torch.__version__,torchvision=torchvision.__version__,official_head='model(inp)[0], as author train.py validation',paired_benchmark=False,dummy_target='input copied only to satisfy paired loader; not a clean GT; no metrics against it'))
 for id in ['T01','N01','N02','F41']:
  c=rows[id];a=read(PREV/'real_models'/id/'original.png');folder=OUT/'stain'/id;folder.mkdir(parents=True,exist_ok=True);data=folder/'loader_data'
  for name in ['input','target']:save(data/name/(id+'.png'),a)
  ds=dataset_module.DataLoaderVal(str(data),dict(w=256,h=256,ori=False));inp,_,_=ds[0];manual=torch.from_numpy(cv2.resize(a,(256,256),interpolation=cv2.INTER_LINEAR).transpose(2,0,1).copy()).float()/255;assert torch.equal(inp,manual)
  start=time.monotonic()
  with torch.inference_mode():heads=model(inp[None].cuda())
  raw=heads[0][0].permute(1,2,0).cpu().numpy();np.save(folder/'official256_raw_RGB.npy',raw);save(folder/'official256_input.png',inp.permute(1,2,0).numpy()*255);save_image(heads[0],folder/'official256_final.png');save_image(heads[1],folder/'official256_coarse.png');assert torch.isfinite(heads[0]).all()
  small=cv2.resize(a,(256,256),interpolation=cv2.INTER_LINEAR);wrapped=adapter.model_call(model,small[:,:,::-1].copy(),'stain','cuda:0')[:,:,::-1].copy();expected=np.clip(raw,0,1);err=float(np.abs(wrapped-expected).max())
  wrapper_tensor=torch.from_numpy(small.transpose(2,0,1).copy())[None].cuda().float()/255
  tensor_err=float((wrapper_tensor-inp[None].cuda()).abs().max());assert tensor_err<=1e-7
  print('adapter diagnostic',id,'float max',err,'mean',float(np.abs(wrapped-expected).mean()),'input max',tensor_err,flush=True)
  manual_saved=np.clip(np.rint(expected*255),0,255).astype('uint8');save_diff=int(np.abs(manual_saved.astype(int)-read(folder/'official256_final.png').astype(int)).max());assert save_diff<=1
  direct=dataset_module.DataLoaderVal(str(data),dict(w=256,h=256,ori=True))[0][0];assert np.array_equal(np.rint(direct.permute(1,2,0).numpy()*255).astype('uint8'),a)
  with torch.inference_mode():native_heads=model(direct[None].cuda())
  native=native_heads[0][0].permute(1,2,0).cpu().numpy();np.save(folder/'native_whole_raw_RGB.npy',native);save_image(native_heads[0],folder/'native_whole_final.png');save_image(native_heads[1],folder/'native_whole_coarse.png');assert torch.isfinite(native_heads[0]).all()
  tiled,_=adapter.tiled(model,a[:,:,::-1].copy(),'stain','cuda:0');rgb=tiled[:,:,::-1].copy();save(folder/'native_tiles_final.png',rgb*255);old=read(PREV/'real_models'/id/'stain.png');fresh=read(folder/'native_tiles_final.png');old_difference=int(np.abs(old.astype(int)-fresh.astype(int)).max());changed=int(np.any(old!=fresh,axis=2).sum());assert old_difference<=1
  cpu_err=None
  if id=='N01':
   cm=model_module.StainRestorer().eval();cm.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
   with torch.inference_mode():cp=cm(inp[None])[0][0].permute(1,2,0).numpy()
   cpu_err=float(np.abs(cp-raw).max());assert cpu_err<1e-4;del cm
  v=dict(id=id,page=c['page'],frame_xywh=c['frame_xywh'],seconds=time.monotonic()-start,official_loader_manual_tensor_equal=True,adapter_vs_direct_float_max=err,adapter_vs_direct_float_mean=float(np.abs(wrapped-expected).mean()),adapter_CPU_GPU_input_float_max=tensor_err,save_image_vs_manual_quantization_max_255=save_diff,fresh_tiles_vs_previous_max_255=old_difference,fresh_tiles_vs_previous_changed_pixels=changed,CPU_GPU_max=cpu_err,official256=stats(inp.permute(1,2,0).numpy(),raw),native_whole=stats(a.astype('float32')/255,native),tiles=stats(a.astype('float32')/255,rgb),whole_vs_tiles_mean_255=float(np.abs(np.clip(native,0,1)-rgb).mean()*255))
  result.append(v);dump(OUT/'stain_results.json',result);print(json.dumps(v,ensure_ascii=False),flush=True)
 # Own controlled documents: checks behaviour, not the published StainDoc benchmark.
 controls=[]
 for name,color in [('white',[245,245,245]),('cream',[235,224,200]),('aged',[210,184,141])]:
  im=Image.new('RGB',(256,256),tuple(color));d=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
  for y in [35,73,111,149,187]:d.text((22,y),'Document ABC 0123',font=font,fill=(35,35,35))
  clean=np.asarray(im).copy();yy,xx=np.mgrid[:256,:256];field=.25*np.exp(-((xx-135)**2+(yy-113)**2)/(2*46**2));dirty=np.rint(clean*(1-field[:,:,None]*np.array([.25,.6,1.0])[None,None,:])).astype('uint8')
  folder=OUT/'controls'/name
  for condition,a in [('clean',clean),('stained',dirty)]:
   t=torch.from_numpy(a.transpose(2,0,1).copy()).float()[None].cuda()/255
   with torch.inference_mode():raw=model(t)[0][0].permute(1,2,0).cpu().numpy()
   save(folder/(condition+'_input.png'),a);save(folder/(condition+'_output.png'),np.clip(raw,0,1)*255);paper=np.all(clean==np.array(color),axis=2);controls.append(dict(paper=name,condition=condition,paper_stats=stats(a.astype('float32')/255,raw,paper),RGB_MAE_to_known_clean=float(np.abs(np.clip(raw,0,1)-clean/255).mean()),input_RGB_MAE_to_known_clean=float(np.abs(a.astype(float)/255-clean/255).mean()),official_dataset=False))
 dump(OUT/'controlled_document_results.json',controls)
 for p,h in frozen.items():assert sha(ROOT/p)==h
 dump(OUT/'stain_complete.json',dict(dependencies_unchanged=True,strict_load=True,actual_loader_used=True,all_four_previous_tiles_reproduced_within_1_255=True,adapter_identity_checked=True,official_benchmark=False,outputs={str(p.relative_to(OUT)):sha(p) for p in (OUT/'stain').rglob('*') if p.is_file() and p.suffix in ['.png','.npy']}))
if __name__=='__main__':main()
