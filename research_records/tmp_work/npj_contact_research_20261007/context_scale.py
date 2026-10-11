from common import *
from learned_pilot import Pilot
from apply_real_models import selection
import torch,time
def infer_field(model,a,device):
 h,w=a.shape[:2];factor=256/max(h,w);ww,hh=round(w*factor),round(h*factor);small=cv2.resize(a,(ww,hh),interpolation=cv2.INTER_AREA)
 left=(256-ww)//2;top=(256-hh)//2;pad=cv2.copyMakeBorder(small,top,256-hh-top,left,256-ww-left,cv2.BORDER_REFLECT_101)
 with torch.inference_mode():
  t=torch.from_numpy(pad.copy()).permute(2,0,1)[None].float().to(device)/255;_,od=model(t);od=od[0].cpu().permute(1,2,0).numpy()[top:top+hh,left:left+ww]
 return cv2.resize(od,(w,h),interpolation=cv2.INTER_LINEAR)
def main():
 cv2.setNumThreads(2);torch.set_num_threads(2);device='cuda:0';rows=selection();out=HERE/'context_scale';models={};deps={}
 for mode in ['gain_od_regression','gain_augmented']:
  ck=HERE/'learned'/mode/'best.pt';m=Pilot('gain').to(device);m.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['state']);models[mode]=m.eval();deps[mode]=sha(ck)
 dump(out/'protocol.json',dict(created=time.time(),source_sha256=sha(__file__),checkpoint_sha256=deps,
  settings='same weights, compare native256 tiles, whole-frame resized256, whole-page maxdimension256 with reflect padding; only OD field is upsampled and applied to native observed RGB',
  change='context and effective spatial scale, not RGB resampling of restored text',
  all_conditions_fixed=True,selection=rows,diagnostic=True,reading_GT=False,material_GT=False));stats=[]
 for c in rows:
  id=c['id'];x,y,w,h=c['frame_xywh'];page=read(ROOT/c['source']['original']);a=page[y:y+h,x:x+w];folder=out/id;values={}
  for mode,model in models.items():
   native=read(HERE/('real_augmented' if mode=='gain_augmented' else 'real_models')/id/('output.png' if mode=='gain_augmented' else 'gain_od_regression.png'));values[mode+'_native']=native
   for scope,input in [('frame',a),('page',page)]:
    od=infer_field(model,input,device)
    if scope=='page':od=od[y:y+h,x:x+w]
    v=np.clip(np.rint(a.astype('float32')*np.exp(od)),0,255).astype('uint8');write(folder/(mode+'_'+scope+'.png'),v);np.save(folder/(mode+'_'+scope+'_field.npy'),od);values[mode+'_'+scope]=v
    stats.append(dict(id=id,mode=mode,scope=scope,field_mean=float(od.mean()),changed_fraction=float(np.any(v!=a,axis=2).mean())))
  xx,yy,ww,hh=c['detail_xywh'];sl=np.s_[yy:yy+hh,xx:xx+ww]
  for mode in models:
   card(folder/(mode+'_comparison.png'),[v[sl] for v in [a,values[mode+'_native'],values[mode+'_frame'],values[mode+'_page']]],
    ['原画像','原寸256タイル','フレーム全体から推定','ページ全体から推定'],id+' '+c['page']+' '+mode,
    '同じ学習済み重み。入力の文脈・空間スケールだけを変更した診断。\n濃度場だけを拡大し、原寸の原画像に乗算。正常領域保持や読解改善は未評価。',size=440)
  print('context scale',id,flush=True)
 dump(out/'complete.json',dict(records=stats,source_sha256=sha(__file__)))
if __name__=='__main__':main()
