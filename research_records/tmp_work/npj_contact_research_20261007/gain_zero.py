from learned_pilot import *
class GainZero(Pilot):
 def __init__(self):
  super().__init__('gain');nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x):
  a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(F.avg_pool2d(c,2))
  up=lambda t,r:F.interpolate(t,size=r.shape[-2:],mode='bilinear',align_corners=False)
  u=self.u(torch.cat([up(d,c),c],1));v=self.v(torch.cat([up(u,b),b],1));w=self.w(torch.cat([up(v,a),a],1));z=self.head(w).float()
  # Exactly identity at initialization, nonzero derivative at zero.
  od=F.interpolate(F.avg_pool2d((torch.tanh(z)*.9).clamp(min=0),8),size=x.shape[-2:],mode='bilinear',align_corners=False)
  return (x.float()*torch.exp(od)).clamp(0,1),od
def main(device):
 torch.set_num_threads(2);torch.manual_seed(20261007);torch.backends.cudnn.benchmark=True
 model=GainZero().to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
 arr,meta=load_data();inds=[m['index'] for m in meta if m['split']=='train'];rng=np.random.default_rng(888)
 gen=torch.Generator(device=device).manual_seed(999);out=HERE/'learned/gain_zero';out.mkdir(parents=True,exist_ok=True)
 steps=8000;manifest=dict(mode='gain_zero',steps=steps,source_sha256=sha(__file__),parent_source_sha256=sha(HERE/'learned_pilot.py'),
  excluded_books=EXCLUDED_BOOKS,data_sha256=sha(HERE/'synthetic/patches.npy'),
  changes='exactly identity initialization: zero head, tanh/clamp positive OD, float32 gain; no auxiliary OD loss',
  hypothesis='nonzero sigmoid gain initially changes the majority intact pixels, driving the head into saturation before localization is learned',
  hypothesis_not_proven=True,loss='same RGB L1 + .25 gradient L1 as original residual/gain runs',
  test_data_used=False,real_images_used=False,created=time.time())
 dump(out/'manifest.json',manifest);history=[];best=float('inf');start=time.monotonic();scaler=torch.cuda.amp.GradScaler()
 for step in range(1,steps+1):
  target=batch(arr,rng.choice(inds,12,replace=True)).to(device)
  if rng.random()<.5:target=target.flip(-1)
  dirty,od,mask=contamination(target,gen,identity=True);opt.zero_grad(set_to_none=True)
  for p in opt.param_groups:p['lr']=2e-5+1.8e-4*.5*(1+np.cos(np.pi*step/steps))
  with torch.cuda.amp.autocast():
   output,field=model(dirty);dx,dy=grad(output);tx,ty=grad(target)
   loss=(output-target).abs().mean()+.25*((dx-tx).abs().mean()+(dy-ty).abs().mean())
  scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
  scaler.step(opt);scaler.update()
  if step%250==0:
   stat=dict(step=step,loss=float(loss),seconds=round(time.monotonic()-start,1),field_mean=float(field.mean()))
   if step%1000==0:
    stat['dev_mae']=evaluate_dev(model,arr,meta,device)
    if stat['dev_mae']<best:
     best=stat['dev_mae'];torch.save(dict(mode='gain_zero',step=step,state=model.state_dict(),dev_mae=best),out/'best.pt')
   history.append(stat);dump(out/'history.json',history);print('gain_zero',json.dumps(stat),flush=True)
 torch.save(dict(mode='gain_zero',step=steps,state=model.state_dict()),out/'last.pt')
 dump(out/'complete.json',dict(steps=steps,best_dev_mae=best,seconds=time.monotonic()-start,best_sha256=sha(out/'best.pt'),source_sha256=sha(__file__)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',required=True);a=p.parse_args();main(a.device)
