from gain_zero import *
import copy
torch.set_num_threads(2);torch.manual_seed(20261007)
arr,meta=load_data();inds=[m['index'] for m in meta if m['split']=='train'];rng=np.random.default_rng(888)
target=batch(arr,rng.choice(inds,12,replace=True));gen=torch.Generator().manual_seed(999);dirty,truth,_=contamination(target,gen,identity=True)
rows=[]
for mode in ['rgb_only','rgb_gradient']:
 torch.manual_seed(20261007);model=GainZero();opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
 for step in range(15):
  opt.zero_grad();output,od=model(dirty);loss=(output-target).abs().mean()
  if mode=='rgb_gradient':
   dx,dy=grad(output);tx,ty=grad(target);loss=loss+.25*((dx-tx).abs().mean()+(dy-ty).abs().mean())
  loss.backward();rows.append(dict(mode=mode,step=step,loss=float(loss.detach()),field_mean=float(od.detach().mean()),bias=model.head.bias.detach().tolist(),head_bias_grad=model.head.bias.grad.tolist()))
  torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
dump(HERE/'learned/gain_diagnosis.json',dict(rows=rows,device='CPU float32',batch_fixed=True,source_sha256=sha(__file__),note='diagnostic only; fixed batch is not generalization evaluation'))
for mode in ['rgb_only','rgb_gradient']:
 print(mode,[r for r in rows if r['mode']==mode and r['step'] in [0,1,2,14]])
