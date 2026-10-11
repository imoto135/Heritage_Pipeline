from evaluate_pilot import *
import sys,os
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_npj_literature_models as ext
def main(device):
 model,deps=ext.load('stain',device);torch.set_num_threads(2);arr,meta=load_data();lookup={m['index']:m for m in meta}
 inds=[m['index'] for m in meta if m['split']=='test'];out=HERE/'synthetic_eval/stain'
 protocol=json.loads((HERE/'synthetic_eval/residual/protocol.json').read_text());protocol.update(mode='stain',checkpoint_sha256=sha(deps[-1]),source_sha256=sha(__file__),settings='author architecture and pretrained weights, native256 RGB input, no resizing/tiling, batch size12 for exact input RNG matching',official_benchmark_reproduction=False,dependencies={str(p.relative_to(ROOT)):sha(p) for p in deps})
 dump(out/'protocol.json',protocol);records=[];start=time.monotonic()
 with torch.inference_mode():
  for domain in protocol['domains']:
   gen=torch.Generator(device=device).manual_seed(protocol['seed']);seen=set()
   for j in range(0,len(inds),12):
    ii=inds[j:j+12];target=batch(arr,ii).to(device)
    if domain=='identity':dirty=target.clone();mask=torch.zeros_like(target[:,:1])
    else:dirty,_,mask=contamination(target,gen,domain,identity=False)
    output=model(dirty);output=output[0] if isinstance(output,(tuple,list)) else output;output=output.clamp(0,1)
    for k,(index,m) in enumerate(zip(ii,metrics(target,dirty,output,mask))):
     book=lookup[index]['book'];records.append(dict(domain=domain,index=index,book=book,page=lookup[index]['page'],**m))
     if book not in seen:
      seen.add(book)
      for name,t in zip(['original','dirty','output'],[target[k],dirty[k],output[k]]):write(out/'examples'/domain/book/(name+'.png'),t.cpu().permute(1,2,0).numpy()*255)
   print('stain synthetic',domain,round(time.monotonic()-start,1),flush=True)
 dump(out/'records.json',records);summary=[]
 for domain in protocol['domains']:
  rr=[r for r in records if r['domain']==domain]
  summary.append(dict(domain=domain,count=len(rr),**{k:float(np.mean([r[k] for r in rr if r[k] is not None])) for k in ['mae','input_mae','psnr','stain_mae','intact_mae','gradient_mae','proxy_ink_recall','proxy_ink_precision'] if any(r[k] is not None for r in rr)}))
 dump(out/'summary.json',summary)
 for p in deps:assert sha(p)==protocol['dependencies'][str(p.relative_to(ROOT))]
 ref=json.loads((HERE/'synthetic_eval/residual/records.json').read_text())
 assert len(ref)==len(records)
 for a,b in zip(ref,records):assert a['index']==b['index'] and a['domain']==b['domain'] and abs(a['input_mae']-b['input_mae'])<1e-8
 dump(out/'complete.json',dict(records=len(records),input_metric_matching=True,seconds=time.monotonic()-start,dependencies_unchanged=True))
if __name__=='__main__':main('cuda:0')
