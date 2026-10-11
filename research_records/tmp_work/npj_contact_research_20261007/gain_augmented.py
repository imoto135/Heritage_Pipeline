from learned_pilot import *
from controlled_strokes import curves
FAMILIES=['id','shape','ring','crease','sharp','color','gray']
def validate(model,arr,meta,device):
 model.eval();inds=[m['index'] for m in meta if m['split']=='dev'][::6];values=[]
 with torch.inference_mode():
  for family in FAMILIES:
   gen=torch.Generator(device=device).manual_seed(76767)
   for j in range(0,len(inds),12):
    target=batch(arr,inds[j:j+12]).to(device);dirty,_,_=contamination(target,gen,family,identity=True);p,_=model(dirty);values.append(float((p-target).abs().mean()))
 model.train();return float(np.mean(values))
def main(device):
 torch.set_num_threads(2);torch.manual_seed(20261007);torch.backends.cudnn.benchmark=True
 model=Pilot('gain').to(device);nn.init.normal_(model.head.weight,std=.001);nn.init.constant_(model.head.bias,-3)
 opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);arr,meta=load_data()
 inds=[m['index'] for m in meta if m['split']=='train'];rng=np.random.default_rng(889);gen=torch.Generator(device=device).manual_seed(1000)
 line_masks=torch.from_numpy(np.array([curves(seed) for seed in range(50,82)],dtype='float32'))[:,None].to(device)
 out=HERE/'learned/gain_augmented';steps=8000
 manifest=dict(mode='gain_augmented',steps=steps,source_sha256=sha(__file__),parent_source_sha256=sha(HERE/'learned_pilot.py'),data_sha256=sha(HERE/'synthetic/patches.npy'),
  loss='10 synthetic OD MSE + .05 RGB L1 + .01 gradient L1, same as gain_od_regression',
  changes=['seven synthetic stain families','random mild clean-paper RGB scaling','additional low-contrast thin gray/brown procedural lines before stain'],
  families=FAMILIES,identity_fraction=.25,real_data_used=False,test_used_for_checkpoint_selection=False,
  exploratory_design_after_old_synthetic_results=True,new_independent_generalization_claim=False,one_combined_change_not_factorial_ablation=True,
  book_sets={s:sorted({m['book'] for m in meta if m['split']==s}) for s in ['train','dev','test']},created=time.time())
 dump(out/'manifest.json',manifest);best=float('inf');history=[];start=time.monotonic();scaler=torch.cuda.amp.GradScaler()
 for step in range(1,steps+1):
  target=batch(arr,rng.choice(inds,12,replace=True)).to(device)
  if rng.random()<.5:target=target.flip(-1)
  factors=torch.rand((12,3,1,1),device=device,generator=gen)*.2+.9;target=(target*factors).clamp(0,1)
  if rng.random()<.5:
   alpha=line_masks[torch.from_numpy(rng.integers(0,len(line_masks),12)).to(device)]
   if rng.random()<.5:alpha=alpha.transpose(-1,-2)
   q=torch.rand((12,3,1,1),device=device,generator=gen)*.5+.5;contrast=torch.rand((12,1,1,1),device=device,generator=gen)*.3+.04
   target=target*(1-alpha*q*contrast)
  family=FAMILIES[int(rng.integers(0,len(FAMILIES)))];dirty,truth,_=contamination(target,gen,family,identity=True);opt.zero_grad(set_to_none=True)
  for p in opt.param_groups:p['lr']=2e-5+1.8e-4*.5*(1+np.cos(np.pi*step/steps))
  with torch.cuda.amp.autocast():
   output,field=model(dirty);dx,dy=grad(output);tx,ty=grad(target)
   loss=10*(field.float()-truth).square().mean()+.05*(output-target).abs().mean()+.01*((dx-tx).abs().mean()+(dy-ty).abs().mean())
  scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update()
  if step%250==0:
   stat=dict(step=step,family=family,loss=float(loss),field_mean=float(field.mean()),seconds=round(time.monotonic()-start,1))
   if step%1000==0:
    stat['dev_mae']=validate(model,arr,meta,device)
    if stat['dev_mae']<best:
     best=stat['dev_mae'];torch.save(dict(mode='gain_augmented',step=step,state=model.state_dict(),dev_mae=best),out/'best.pt')
   history.append(stat);dump(out/'history.json',history);print('gain_augmented',json.dumps(stat),flush=True)
 torch.save(dict(mode='gain_augmented',step=steps,state=model.state_dict()),out/'last.pt')
 dump(out/'complete.json',dict(steps=steps,best_dev_mae=best,seconds=time.monotonic()-start,best_sha256=sha(out/'best.pt'),source_sha256=sha(__file__)))
if __name__=='__main__':main('cuda:0')
