from learned_pilot import *
def main(device):
 torch.set_num_threads(2);torch.manual_seed(20261007);torch.backends.cudnn.benchmark=True
 model=Pilot('gain').to(device);nn.init.normal_(model.head.weight,std=.001);nn.init.constant_(model.head.bias,-3)
 opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);arr,meta=load_data()
 inds=[m['index'] for m in meta if m['split']=='train'];rng=np.random.default_rng(888);gen=torch.Generator(device=device).manual_seed(999)
 out=HERE/'learned/gain_od_regression';steps=8000
 manifest=dict(mode='gain_od_regression',steps=steps,source_sha256=sha(__file__),parent_source_sha256=sha(HERE/'learned_pilot.py'),data_sha256=sha(HERE/'synthetic/patches.npy'),
  loss='10 * MSE(known synthetic optical density) + .05 RGB L1 + .01 gradient L1',
  rationale='original L1 restoration drives global gain to zero before localization; squared field regression supplies a dense synthetic supervision signal',
  changed_loss=True,not_equal_loss_ablation=True,synthetic_OD_GT_only=True,real_images_used=False,test_used_for_selection=False,
  book_sets={s:sorted({m['book'] for m in meta if m['split']==s}) for s in ['train','dev','test']},created=time.time())
 dump(out/'manifest.json',manifest);best=float('inf');history=[];start=time.monotonic();scaler=torch.cuda.amp.GradScaler()
 for step in range(1,steps+1):
  target=batch(arr,rng.choice(inds,12,replace=True)).to(device)
  if rng.random()<.5:target=target.flip(-1)
  dirty,truth,_=contamination(target,gen,identity=True);opt.zero_grad(set_to_none=True)
  for p in opt.param_groups:p['lr']=2e-5+1.8e-4*.5*(1+np.cos(np.pi*step/steps))
  with torch.cuda.amp.autocast():
   output,field=model(dirty);dx,dy=grad(output);tx,ty=grad(target)
   loss=10*(field.float()-truth).square().mean()+.05*(output-target).abs().mean()+.01*((dx-tx).abs().mean()+(dy-ty).abs().mean())
  scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update()
  if step%250==0:
   stat=dict(step=step,loss=float(loss),field_mean=float(field.mean()),truth_mean=float(truth.mean()),seconds=round(time.monotonic()-start,1))
   if step%1000==0:
    stat['dev_mae']=evaluate_dev(model,arr,meta,device)
    if stat['dev_mae']<best:
     best=stat['dev_mae'];torch.save(dict(mode='gain_od_regression',step=step,state=model.state_dict(),dev_mae=best),out/'best.pt')
   history.append(stat);dump(out/'history.json',history);print('gain_od_regression',json.dumps(stat),flush=True)
 torch.save(dict(mode='gain_od_regression',step=steps,state=model.state_dict()),out/'last.pt')
 dump(out/'complete.json',dict(steps=steps,best_dev_mae=best,seconds=time.monotonic()-start,best_sha256=sha(out/'best.pt'),source_sha256=sha(__file__)))
if __name__=='__main__':main('cuda:1')
