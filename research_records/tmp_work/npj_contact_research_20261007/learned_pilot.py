from common import *
import torch
from torch import nn
from torch.nn import functional as F
import argparse,time,os
os.environ['MPLCONFIGDIR']=str(HERE/'plot_cache')
EXCLUDED_BOOKS=['100249371','200019865','100249537','100249416','200014740','200003076']
class Block(nn.Module):
 def __init__(self,a,b):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())
 def forward(self,x):return self.net(x)
class Pilot(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.a=Block(3,24);self.b=Block(24,48);self.c=Block(48,96);self.d=Block(96,144)
  self.u=Block(144+96,96);self.v=Block(96+48,48);self.w=Block(48+24,24);self.head=nn.Conv2d(24,3,1)
  nn.init.zeros_(self.head.weight);nn.init.constant_(self.head.bias,-5 if mode=='gain' else 0)
 def forward(self,x):
  a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(F.avg_pool2d(c,2))
  up=lambda t,r:F.interpolate(t,size=r.shape[-2:],mode='bilinear',align_corners=False)
  u=self.u(torch.cat([up(d,c),c],1));v=self.v(torch.cat([up(u,b),b],1));w=self.w(torch.cat([up(v,a),a],1));z=self.head(w)
  if self.mode=='gain':
   od=F.interpolate(F.avg_pool2d(torch.sigmoid(z)*.9,8),size=x.shape[-2:],mode='bilinear',align_corners=False)
   return (x*torch.exp(od)).clamp(0,1),od
  return (x+torch.tanh(z)*.6).clamp(0,1),z
def contamination(clean,generator,kind='id',identity=True):
 B,C,H,W=clean.shape;dev=clean.device
 rand=lambda *s:torch.rand(s,device=dev,generator=generator)
 yy,xx=torch.meshgrid(torch.linspace(-1,1,H,device=dev),torch.linspace(-1,1,W,device=dev),indexing='ij');x=xx[None,None];y=yy[None,None]
 cx=(rand(B,2,1,1)*1.3-.65);cy=(rand(B,2,1,1)*1.3-.65)
 rx=rand(B,2,1,1)*.6+.15;ry=rand(B,2,1,1)*.6+.15
 th=rand(B,2,1,1)*np.pi;dx=x-cx;dy=y-cy
 rr=((dx*th.cos()+dy*th.sin())/rx)**2+((-dx*th.sin()+dy*th.cos())/ry)**2
 shape=torch.sigmoid((1-rr)*8).amax(1,keepdim=True)
 if kind=='shape':
  ripple=F.interpolate(rand(B,1,8,8),size=(H,W),mode='bicubic',align_corners=False)
  shape=torch.sigmoid((shape*.8+ripple*.5-.55)*18)
 elif kind=='ring':shape=torch.exp(-((rr[:,0:1]-.65)/.23)**2)
 elif kind=='crease':shape=torch.exp(-((dx[:,0:1]+dy[:,0:1]*.5)/.08)**2)
 elif kind=='sharp':shape=(rr.min(1,keepdim=True).values<1).float()
 amp=rand(B,1,1,1)*.55+.12
 if kind=='color':q=rand(B,3,1,1)*.9+.3
 elif kind=='gray':q=torch.ones(B,3,1,1,device=dev)
 else:q=torch.cat([rand(B,1,1,1)*.2+.35,rand(B,1,1,1)*.2+.7,rand(B,1,1,1)*.15+1.],1)
 if identity:amp=amp*(rand(B,1,1,1)>.25)
 od=shape*amp*q
 return clean*torch.exp(-od),od,shape*(amp>0)
def load_data():
 arr=np.load(HERE/'synthetic/patches.npy',mmap_mode='r');meta=json.loads((HERE/'synthetic/patches.json').read_text())
 good=[m for m in meta if m['book'] not in EXCLUDED_BOOKS]
 return arr,good
def grad(x):return x[:,:,:,1:]-x[:,:,:,:-1],x[:,:,1:,:]-x[:,:,:-1,:]
def batch(arr,inds):return torch.from_numpy(np.asarray(arr[inds]).copy()).permute(0,3,1,2).float()/255
def evaluate_dev(model,arr,meta,device):
 model.eval();gen=torch.Generator(device=device).manual_seed(76767);rs=[]
 inds=[m['index'] for m in meta if m['split']=='dev'][::3]
 with torch.inference_mode():
  for i in range(0,len(inds),12):
   target=batch(arr,inds[i:i+12]).to(device);dirty,od,mask=contamination(target,gen,identity=True)
   output,_=model(dirty);rs.append(float((output-target).abs().mean()))
 model.train();return float(np.mean(rs))
def train(mode,device,steps=8000):
 torch.set_num_threads(2);torch.manual_seed(20261007);torch.backends.cudnn.benchmark=True
 torch.backends.cuda.matmul.allow_tf32=True
 model=Pilot(mode).to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
 arr,meta=load_data();inds=[m['index'] for m in meta if m['split']=='train'];rng=np.random.default_rng(888)
 gen=torch.Generator(device=device).manual_seed(999);out=HERE/'learned'/mode;out.mkdir(parents=True,exist_ok=True)
 manifest=dict(mode=mode,steps=steps,batch=12,optimizer='AdamW 2e-4 cosine to2e-5',source_sha256=sha(__file__),
  data_sha256=sha(HERE/'synthetic/patches.npy'),excluded_books=EXCLUDED_BOOKS,
  train_patches=len(inds),dev_patches=sum(m['split']=='dev' for m in meta),test_patches=sum(m['split']=='test' for m in meta),
  book_sets={s:sorted({m['book'] for m in meta if m['split']==s}) for s in ['train','dev','test']},
  identity_fraction=.25,loss='L1 + .25 gradient L1; same for both modes',
  gain='nonnegative RGB optical density /8 spatial grid, applied multiplicatively to observed pixels',
  residual='unrestricted signed RGB residual, no gain constraint',
  real_data='no real contact-stain input is used for training or checkpoint selection',created=time.time())
 dump(out/'manifest.json',manifest);history=[];best=float('inf');start=time.monotonic()
 scaler=torch.cuda.amp.GradScaler()
 for step in range(1,steps+1):
  ix=rng.choice(inds,12,replace=True);target=batch(arr,ix).to(device)
  if rng.random()<.5:target=target.flip(-1)
  dirty,od,mask=contamination(target,gen,identity=True);opt.zero_grad(set_to_none=True)
  for p in opt.param_groups:p['lr']=2e-5+1.8e-4*.5*(1+np.cos(np.pi*step/steps))
  with torch.cuda.amp.autocast():
   output,_=model(dirty);dx,dy=grad(output);tx,ty=grad(target)
   loss=(output-target).abs().mean()+.25*((dx-tx).abs().mean()+(dy-ty).abs().mean())
  scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
  scaler.step(opt);scaler.update()
  if step%250==0:
   stat=dict(step=step,loss=float(loss),seconds=round(time.monotonic()-start,1))
   if step%1000==0:
    stat['dev_mae']=evaluate_dev(model,arr,meta,device)
    if stat['dev_mae']<best:
     best=stat['dev_mae'];torch.save(dict(mode=mode,step=step,state=model.state_dict(),dev_mae=best),out/'best.pt')
    torch.save(dict(mode=mode,step=step,state=model.state_dict()),out/'last.pt')
   history.append(stat);dump(out/'history.json',history);print(mode,json.dumps(stat),flush=True)
 dump(out/'complete.json',dict(steps=steps,best_dev_mae=best,seconds=time.monotonic()-start,best_sha256=sha(out/'best.pt'),source_sha256=sha(__file__)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['gain','residual'],required=True);p.add_argument('--device',required=True);p.add_argument('--steps',type=int,default=8000);a=p.parse_args();train(a.mode,a.device,a.steps)
