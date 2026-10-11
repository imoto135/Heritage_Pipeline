from controlled_strokes import *
from learned_pilot import Pilot
device='cuda:0';torch.set_num_threads(2);cv2.setNumThreads(2);arr,_=load_data();out=HERE/'controlled_strokes_augmented';base=json.loads((HERE/'controlled_strokes/protocol.json').read_text());design=base['design']
ck=HERE/'learned/gain_augmented/best.pt';model=Pilot('gain').to(device);model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['state']);model.eval();rows=[]
dump(out/'protocol.json',dict(created=time.time(),base_protocol_sha256=sha(HERE/'controlled_strokes/protocol.json'),source_sha256=sha(__file__),checkpoint_sha256=sha(ck),exploratory=True,design=design,training_includes_procedural_lines=True,real_character_GT=False))
with torch.inference_mode():
 for r in design:
  paper=cv2.GaussianBlur(np.asarray(arr[r['index']]).astype('float32')/255,(0,0),35);alpha=curves(77+r['index']);q=np.array([1.,1.,1.]) if r['color']=='gray' else np.array([.5,.75,1.])
  writing=paper*(1-alpha[:,:,None]*r['contrast']*q);pair=torch.from_numpy(np.array([paper,writing],dtype='float32')).permute(0,3,1,2).to(device)
  if r['domain']=='identity':dirty=pair.clone()
  else:
   gen=torch.Generator(device=device).manual_seed(98123+r['index']);_,od,_=contamination(pair[:1],gen,identity=False);dirty=pair*torch.exp(-od)
  v=model(dirty)[0].cpu().permute(0,2,3,1).numpy();target=pair.cpu().permute(0,2,3,1).numpy();support=alpha>.25
  rows.append(dict(r,method='gain_augmented',mae=float(np.abs(v[1]-target[1]).mean()),stroke_mae=float(np.abs(v[1]-target[1])[support].mean()),paired_contrast_response=float((v[0]-v[1])[support].mean()/max(1e-9,(target[0]-target[1])[support].mean())),input_mae=float((dirty[1]-pair[1]).abs().mean())))
  if r['book']==sorted({r['book'] for r in design})[0] and r['color']=='gray':
   label=f"c{int(r['contrast']*100):02d}_{r['domain']}";input_array=dirty[1].cpu().permute(1,2,0).numpy()
   expected=read(HERE/'controlled_strokes/examples'/label/'input.png')
   assert np.array_equal(np.clip(np.rint(input_array*255),0,255).astype('uint8'),expected)
   write(out/'examples'/(label+'.png'),v[1]*255)
ref=json.loads((HERE/'controlled_strokes/records.json').read_text());lookup={(r['book'],r['color'],r['contrast'],r['domain']):r for r in ref if r['method']=='input'}
delta=[dict(book=r['book'],color=r['color'],contrast=r['contrast'],domain=r['domain'],delta=r['input_mae']-lookup[(r['book'],r['color'],r['contrast'],r['domain'])]['mae']) for r in rows]
dump(out/'input_matching_diagnostic.json',delta);print('max input MAE delta',max(abs(r['delta']) for r in delta),flush=True)
for r in delta:assert abs(r['delta'])<1e-7
dump(out/'records.json',rows);dump(out/'complete.json',dict(cases=len(rows),input_matching=True,eight_exemplar_input_PNGs_pixel_equal=True,max_metric_delta=max(abs(r['delta']) for r in delta),tolerance=1e-7,note='GPU torch versus NumPy float32 reduction differences; exemplar pixels checked exactly'));print('augmented controlled strokes',len(rows))
