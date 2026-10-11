from common import *
from learned_pilot import Pilot,load_data,batch,contamination
import torch,time,sys
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_npj_literature_models as ext
def curves(seed):
 rng=np.random.default_rng(seed);canvas=np.zeros((1024,1024),np.uint8)
 for x in [55,95,140,185]:
  points=[]
  for y in np.linspace(25,230,100):points.append([int((x+14*np.sin(y/35+rng.random()*.03))*4),int(y*4)])
  cv2.polylines(canvas,[np.array(points)],False,255,thickness=5,lineType=cv2.LINE_AA)
 return cv2.resize(canvas,(256,256),interpolation=cv2.INTER_AREA)/255
def main():
 cv2.setNumThreads(2);device='cuda:1';arr,meta=load_data();bybook={}
 for m in meta:
  if m['split']=='test' and m['book'] not in bybook:bybook[m['book']]=m
 design=[]
 for book,m in sorted(bybook.items()):
  for color in ['gray','brown']:
   for contrast in [.06,.12,.25,.5]:
    for domain in ['identity','id']:design.append(dict(book=book,index=m['index'],color=color,contrast=contrast,domain=domain))
 out=HERE/'controlled_strokes';dump(out/'protocol.json',dict(created=time.time(),design=design,source_sha256=sha(__file__),
  ground_truth='procedural 1.25px curved lines, antialias coverage>.25, over Gaussian-smoothed test-source texture; not real characters',
  claim_limit='controlled line visibility and paired model response; not historic handwriting, material classification or reading accuracy',
  train_or_checkpoint_selection=False,methods=['input','sauvola','docres','residual','gain_od_regression','stain']))
 targets=[];inputs=[];fields=[];alphas=[]
 for i,r in enumerate(design):
  paper=cv2.GaussianBlur(np.asarray(arr[r['index']]).astype('float32')/255,(0,0),35);alpha=curves(77+r['index'])
  q=np.array([1.,1.,1.]) if r['color']=='gray' else np.array([.5,.75,1.])
  writing=paper*(1-alpha[:,:,None]*r['contrast']*q);pair=torch.from_numpy(np.array([paper,writing],dtype='float32')).permute(0,3,1,2).to(device)
  if r['domain']=='identity':dirty=pair.clone();od=torch.zeros_like(pair)
  else:
   generator=torch.Generator(device=device).manual_seed(98123+r['index']);_,od,_=contamination(pair[:1],generator,identity=False);dirty=pair*torch.exp(-od)
  targets.append(pair.cpu().permute(0,2,3,1).numpy());inputs.append(dirty.cpu().permute(0,2,3,1).numpy());alphas.append(alpha)
 models={}
 for mode in ['residual','gain_od_regression']:
  model=Pilot('gain' if mode=='gain_od_regression' else mode).to(device);model.load_state_dict(torch.load(HERE/'learned'/mode/'best.pt',map_location=device,weights_only=True)['state']);models[mode]=model.eval()
 sm,deps=ext.load('stain',device);models['stain']=sm;dm,ddeps=ext.load('docres',device);pns=ext.prompts();stats=[]
 exemplars={};start=time.monotonic()
 for i,r in enumerate(design):
  value=inputs[i];target=targets[i];alpha=alphas[i];support=alpha>.25
  results={'input':value};binary={}
  results['sauvola']=np.array([ink_render(sauvola(a*255).astype('float32'))/255 for a in value])
  binary['sauvola']=np.array([sauvola(a*255) for a in value])
  with torch.inference_mode():
   t=torch.from_numpy(value).permute(0,3,1,2).to(device)
   for mode,model in models.items():
    z=model(t);z=z[0] if isinstance(z,(tuple,list)) else z;results[mode]=z.clamp(0,1).cpu().permute(0,2,3,1).numpy()
   db=[]
   for a in value:
    bgr=np.clip(np.rint(a[:,:,::-1]*255),0,255).astype('uint8');prompt=pns['binarization_promptv2'](bgr)
    prob=ext.model_call(dm,np.concatenate([bgr,prompt],axis=2),'docres',device)[:,:,0];db.append(prob<.5)
   binary['docres']=np.array(db);results['docres']=np.array([ink_render(b.astype('float32'))/255 for b in db])
  for mode,v in results.items():
   record=dict(r,method=mode,mae=float(np.abs(v[1]-target[1]).mean()),stroke_mae=float(np.abs(v[1]-target[1])[support].mean()))
   if mode in binary:
    b=binary[mode][1];record.update(known_line_recall=float(b[support].mean()),background_false_positive=float(b[~support].mean()))
   else:record['paired_contrast_response']=float((v[0]-v[1])[support].mean()/max(1e-9,(target[0]-target[1])[support].mean()))
   stats.append(record)
  if r['book']==sorted(bybook)[0] and r['color']=='gray':
   label=f"c{int(r['contrast']*100):02d}_{r['domain']}";folder=out/'examples'/label
   for mode,v in results.items():write(folder/(mode+'.png'),v[1]*255)
   write(folder/'target.png',target[1]*255);write(folder/'line_GT.png',support*255)
   card(folder/'comparison.png',[target[1]*255,results['input'][1]*255,results['sauvola'][1]*255,results['docres'][1]*255,results['residual'][1]*255,results['gain_od_regression'][1]*255],
    ['人工線の正解','入力','Sauvola','DocRes','RGB差分学習','濃度教師＋明るさ制約'],f"人工線 コントラスト{r['contrast']:.0%} {r['domain']}",
    '線の形は手続き生成、実際の古文書文字ではない。1.25px線／表示原寸。\n一般化・読解改善の証明には使わない。既存条件を固定して適用。',size=256)
  if (i+1)%16==0:print('controlled strokes',i+1,len(design),round(time.monotonic()-start,1),flush=True)
 dump(out/'records.json',stats);dump(out/'complete.json',dict(cases=len(design),records=len(stats),seconds=time.monotonic()-start,source_sha256=sha(__file__)))
if __name__=='__main__':
 while not (HERE/'synthetic_eval/gain_od_regression/complete.json').exists():time.sleep(5)
 main()
