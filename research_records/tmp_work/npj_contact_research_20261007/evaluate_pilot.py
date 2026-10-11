from common import *
from learned_pilot import Pilot,contamination,load_data,batch
import torch,time,argparse,collections
def load(mode,device):
 ck=HERE/'learned'/mode/'best.pt';state=torch.load(ck,map_location=device,weights_only=True);model=Pilot(mode).to(device)
 model.load_state_dict(state['state']);model.eval();return model,state,ck
def metrics(target,dirty,value,mask):
 t=target.detach().cpu().permute(0,2,3,1).numpy();d=dirty.detach().cpu().permute(0,2,3,1).numpy()
 v=value.detach().cpu().permute(0,2,3,1).numpy();m=mask.detach().cpu().numpy()[:,0]>.05;rs=[]
 for a,b,c,mm in zip(t,d,v,m):
  err=np.abs(c-a);de=np.abs(b-a);mse=float(np.mean((c-a)**2))
  clean_ink=sauvola(a*255);out_ink=sauvola(c*255)
  rs.append(dict(mae=float(err.mean()),input_mae=float(de.mean()),psnr=float(-10*np.log10(max(mse,1e-12))),
   stain_mae=float(err[mm].mean()) if mm.any() else None,
   intact_mae=float(err[~mm].mean()) if (~mm).any() else None,
   gradient_mae=float((np.abs(np.diff(c,axis=0)-np.diff(a,axis=0)).mean()+np.abs(np.diff(c,axis=1)-np.diff(a,axis=1)).mean())/2),
   proxy_ink_recall=float((out_ink&clean_ink).sum()/max(1,clean_ink.sum())),
   proxy_ink_precision=float((out_ink&clean_ink).sum()/max(1,out_ink.sum())),
   proxy_note='Sauvola on source as operational reference; not annotated strokes or reading accuracy'))
 return rs
def real(model,mode,device):
 out=HERE/'learned_real'/mode;stats=[]
 for c in cases():
  a=read(SOURCE/'native'/c['id']/'original_context.png');h,w=a.shape[:2];halo=32;core=192
  padded=cv2.copyMakeBorder(a,halo,halo+core,halo,halo+core,cv2.BORDER_REFLECT_101);v=np.empty_like(a,dtype='float32');field=np.empty_like(v)
  with torch.inference_mode():
   for y in range(0,h,core):
    for x in range(0,w,core):
     t=torch.from_numpy(padded[y:y+256,x:x+256].copy()).permute(2,0,1)[None].float().to(device)/255
     p,z=model(t);p=p[0].permute(1,2,0).cpu().numpy();z=z[0].permute(1,2,0).cpu().numpy()
     bh,bw=min(core,h-y),min(core,w-x);v[y:y+bh,x:x+bw]=p[halo:halo+bh,halo:halo+bw];field[y:y+bh,x:x+bw]=z[halo:halo+bh,halo:halo+bw]
  write(out/c['id']/'output.png',v*255);np.save(out/c['id']/'field.npy',field)
  stats.append(dict(id=c['id'],mean_difference=float(np.abs(v-a/255).mean()),field_min=float(field.min()),field_max=float(field.max())))
 dump(out/'stats.json',stats)
def main(mode,device):
 torch.set_num_threads(2);model,state,ck=load(mode,device);arr,meta=load_data()
 inds=[m['index'] for m in meta if m['split']=='test'];out=HERE/'synthetic_eval'/mode
 protocol=dict(mode=mode,checkpoint_sha256=sha(ck),selected_step=state['step'],source_sha256=sha(__file__),
  test_indices=inds,test_books=sorted({m['book'] for m in meta if m['split']=='test'}),
  domains=['identity','id','shape','ring','crease','sharp','color','gray'],seed=100707,
  ground_truth='original before artificial stain; intrinsic degradation not removed',
  test_used_for_checkpoint_selection=False,created=time.time())
 dump(out/'protocol.json',protocol);lookup={m['index']:m for m in meta};records=[];start=time.monotonic()
 with torch.inference_mode():
  for domain in protocol['domains']:
   gen=torch.Generator(device=device).manual_seed(protocol['seed']);seen=set()
   for j in range(0,len(inds),12):
    ii=inds[j:j+12];target=batch(arr,ii).to(device)
    if domain=='identity':dirty=target.clone();mask=torch.zeros_like(target[:,:1]);od=target[:,:1]*0
    else:dirty,od,mask=contamination(target,gen,domain,identity=False)
    output,field=model(dirty);ms=metrics(target,dirty,output,mask)
    for k,(index,m) in enumerate(zip(ii,ms)):
     book=lookup[index]['book'];records.append(dict(domain=domain,index=index,book=book,page=lookup[index]['page'],**m))
     if book not in seen:
      seen.add(book);folder=out/'examples'/domain/book
      arrays=[target[k],dirty[k],output[k]]
      for name,t in zip(['original','dirty','output'],arrays):write(folder/(name+'.png'),t.cpu().permute(1,2,0).numpy()*255)
   print('synthetic eval',mode,domain,'patches',len(inds),round(time.monotonic()-start,1),flush=True)
 real(model,mode,device)
 dump(out/'records.json',records);summary=[]
 for domain in protocol['domains']:
  rr=[r for r in records if r['domain']==domain]
  summary.append(dict(domain=domain,count=len(rr),**{k:float(np.mean([r[k] for r in rr if r[k] is not None])) for k in ['mae','input_mae','psnr','stain_mae','intact_mae','gradient_mae','proxy_ink_recall','proxy_ink_precision'] if any(r[k] is not None for r in rr)}))
 dump(out/'summary.json',summary);dump(out/'complete.json',dict(seconds=time.monotonic()-start,records=len(records),real_cases=len(cases()),checkpoint_unchanged=sha(ck)==protocol['checkpoint_sha256']))
 print('pilot eval complete',mode,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['gain','residual'],required=True);p.add_argument('--device',required=True);a=p.parse_args();main(a.mode,a.device)
