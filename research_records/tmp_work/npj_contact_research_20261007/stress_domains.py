from common import *
from learned_pilot import Pilot,load_data,batch,contamination
from evaluate_pilot import metrics
import torch,time
DOMAINS=['chromatic_field','micro_speckles','opaque_overlay']
def corrupt(t,gen,kind):
 B,C,H,W=t.shape;dev=t.device;rand=lambda *s:torch.rand(s,device=dev,generator=gen)
 dirty,od,mask=contamination(t,gen,'id',identity=False)
 if kind=='chromatic_field':
  xx=torch.linspace(0,1,W,device=dev)[None,None,None,:];od=od.mean(1,keepdim=True)*torch.cat([.3+xx,.7+xx*.2,1.3-xx],1);return t*torch.exp(-od),od.mean(1,keepdim=True)
 if kind=='opaque_overlay':
  alpha=1-torch.exp(-od.mean(1,keepdim=True));color=t.new_tensor([.65,.46,.29])[None,:,None,None];return t*(1-alpha)+color*alpha,alpha
 yy,xx=torch.meshgrid(torch.arange(H,device=dev),torch.arange(W,device=dev),indexing='ij')
 cx=rand(B,24,1,1)*(W-20)+10;cy=rand(B,24,1,1)*(H-20)+10;radius=rand(B,24,1,1)*2+1
 field=torch.exp(-((xx-cx)**2+(yy-cy)**2)/(2*radius**2)).sum(1,keepdim=True).clamp(0,1)*.5
 q=t.new_tensor([.5,.8,1.2])[None,:,None,None];return t*torch.exp(-field*q),field
def main():
 device='cuda:0';torch.set_num_threads(2);arr,meta=load_data();inds=[m['index'] for m in meta if m['split']=='test'];lookup={m['index']:m for m in meta};out=HERE/'stress_eval';models={};deps={}
 for mode in ['residual','gain_od_regression','gain_augmented']:
  ck=HERE/'learned'/mode/'best.pt';model=Pilot('residual' if mode=='residual' else 'gain').to(device);model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['state']);models[mode]=model.eval();deps[mode]=sha(ck)
 protocol=json.loads((out/'protocol.json').read_text());protocol['checkpoints']=deps;dump(out/'protocol_used.json',protocol);rows=[]
 with torch.inference_mode():
  for domain in DOMAINS:
   gen=torch.Generator(device=device).manual_seed(92341);seen=set()
   for j in range(0,len(inds),12):
    ii=inds[j:j+12];target=batch(arr,ii).to(device);dirty,mask=corrupt(target,gen,domain);results={'input':dirty}
    for mode,model in models.items():results[mode]=model(dirty)[0]
    for mode,value in results.items():
     ms=metrics(target,dirty,value,mask)
     for k,(index,m) in enumerate(zip(ii,ms)):
      lower=float((dirty[k]-target[k]).clamp(min=0).mean());rows.append(dict(method=mode,domain=domain,index=index,book=lookup[index]['book'],nonnegative_gain_MAE_lower_bound=lower,**m))
    if not seen:
     seen.add(True);folder=out/'examples'/domain
     write(folder/'target.png',target[0].cpu().permute(1,2,0).numpy()*255)
     for mode,t in results.items():write(folder/(mode+'.png'),t[0].cpu().permute(1,2,0).numpy()*255)
   print('stress domain complete',domain,flush=True)
 dump(out/'records.json',rows);dump(out/'complete.json',dict(records=len(rows),domains=DOMAINS,checkpoint_sha256=deps,source_sha256=sha(__file__)))
if __name__=='__main__':
 out=HERE/'stress_eval';dump(out/'protocol.json',dict(created=time.time(),domains=DOMAINS,source_sha256=sha(__file__),seed=92341,
  prescribed_before_augmented_weights_review=True,OOD_not_used_for_training_or_checkpoint_selection=True,
  limits='new synthetic mechanisms on existing held-out source groups; no true clean historical images or reading GT',
  formulae=dict(chromatic_field='spatial RGB optical-density variation',micro_speckles='24 Gaussian radius1..3px multiplicative spots',opaque_overlay='alpha mixture with brown RGB[.65,.46,.29], can brighten ink and darken paper'),
  scope='test the assumptions of nonnegative smooth optical-density correction, not a claim of real physical stain modeling'))
 while not (HERE/'real_augmented/complete.json').exists():time.sleep(5)
 main()
