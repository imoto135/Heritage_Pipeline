from common import *
import sys,time,os,argparse
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_npj_literature_models as ext
def main(method,device):
 model,deps=ext.load(method,device);pns=ext.prompts() if method=='docres' else None
 dest=HERE/'external'/method;dest.mkdir(parents=True,exist_ok=True)
 manifest=dict(method=method,device=device,settings=ext.SETTINGS,
  dependencies={str(p.relative_to(ROOT)):sha(p) for p in deps},source_sha256=sha(__file__),
  selection_sha256=sha(SOURCE/'benchmark_candidates.json'),created=time.time(),
  changes='new frozen real cases; stain native tiles and 256px resized whole-frame both reported',
  official_benchmark_reproduction=False)
 dump(dest/'manifest.json',manifest);stats=[]
 for c in cases():
  id=c['id'];rgb=read(SOURCE/'native'/id/'original_context.png');a=rgb[:,:,::-1].copy()
  (dest/id).mkdir(parents=True,exist_ok=True)
  inp=np.concatenate([a,pns['binarization_promptv2'](a)],axis=2) if pns else a
  start=time.monotonic();v,n=ext.tiled(model,inp,method,device)
  if method=='stain':
   write(dest/id/'native.png',v[:,:,::-1]*255)
   small=cv2.resize(a,(256,256),interpolation=cv2.INTER_AREA)
   rv=ext.model_call(model,small,method,device)
   write(dest/id/'resized256.png',cv2.resize(rv[:,:,::-1],rgb.shape[1::-1],interpolation=cv2.INTER_CUBIC)*255)
  else:
   np.save(dest/id/'background_probability.npy',v[:,:,0]);write(dest/id/'native.png',np.repeat((v>=.5)*255,3,axis=2))
  stat=dict(id=id,seconds=time.monotonic()-start,tiles=n,output_sha256=sha(dest/id/'native.png'))
  stats.append(stat);dump(dest/'progress.json',stats);print(method,id,round(stat['seconds'],2),flush=True)
 for path,h in manifest['dependencies'].items():assert sha(ROOT/path)==h
 dump(dest/'complete.json',dict(stats=stats,dependencies_unchanged=True,original_only=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['stain','docres'],required=True);p.add_argument('--device',required=True);a=p.parse_args();main(a.method,a.device)
