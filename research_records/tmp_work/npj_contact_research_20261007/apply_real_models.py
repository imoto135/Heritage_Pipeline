from common import *
from learned_pilot import Pilot
import torch,sys,time
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_npj_literature_models as ext
def selection():
 rows=[]
 for c in cases():rows.append(dict(id=c['id'],page=c['page'],source=c['source'],frame_xywh=c['crop_page_xywh'] if 'crop_page_xywh' in c else c['crop_xywh'],detail_xywh=c['detail_xywh']))
 for c in json.loads((HERE/'new_cases/selection.json').read_text())['cases']:rows.append(dict(id=c['id'],page=c['page'],source=c['source'],frame_xywh=c['crop_xywh'],detail_xywh=c['detail_xywh']))
 for c in json.loads((HERE/'other_book_screen/final/cohort_C.json').read_text())['selected']:rows.append(dict(id=c['id'],page=c['page'],source=c['source'],frame_xywh=c['detail_page_xywh'],detail_xywh=[0,0,768,768]))
 return rows
def infer_pilot(model,a,device):
 h,w=a.shape[:2];halo=32;core=192;pad=cv2.copyMakeBorder(a,halo,halo+core,halo,halo+core,cv2.BORDER_REFLECT_101);out=np.empty_like(a);od=np.empty_like(a,dtype='float32')
 with torch.inference_mode():
  for y in range(0,h,core):
   for x in range(0,w,core):
    t=torch.from_numpy(pad[y:y+256,x:x+256].copy()).permute(2,0,1)[None].float().to(device)/255;p,z=model(t)
    bh,bw=min(core,h-y),min(core,w-x);v=p[0].permute(1,2,0).cpu().numpy();zz=z[0].permute(1,2,0).cpu().numpy()
    out[y:y+bh,x:x+bw]=np.clip(np.rint(v[halo:halo+bh,halo:halo+bw]*255),0,255).astype('uint8');od[y:y+bh,x:x+bw]=zz[halo:halo+bh,halo:halo+bw]
 return out,od
def main():
 cv2.setNumThreads(2);device='cuda:1';models={};deps=[]
 for mode in ['residual','gain_od_regression']:
  ck=HERE/'learned'/mode/'best.pt';model=Pilot('gain' if mode=='gain_od_regression' else mode).to(device);model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['state']);models[mode]=model.eval();deps.append(ck)
 sm,sd=ext.load('stain',device);dm,dd=ext.load('docres',device);pns=ext.prompts();rows=selection();out=HERE/'real_models'
 dump(out/'protocol.json',dict(created=time.time(),source_sha256=sha(__file__),selected=rows,dependencies={str(p.relative_to(ROOT)):sha(p) for p in deps+sd+dd},source_only=True,all_conditions_fixed_before_final_C_outputs=True,ROI_used_only_for_frame_selection=True,native_resolution=True,source_group_generalization_evaluation=False))
 stats=[]
 for c in rows:
  x,y,w,h=c['frame_xywh'];a=read(ROOT/c['source']['original'])[y:y+h,x:x+w];folder=out/c['id'];write(folder/'original.png',a);values={}
  for mode,model in models.items():
   value,field=infer_pilot(model,a,device);write(folder/(mode+'.png'),value);np.save(folder/(mode+'_field.npy'),field);values[mode]=value
  bgr=a[:,:,::-1].copy();v,_=ext.tiled(sm,bgr,'stain',device);values['stain']=v[:,:,::-1]*255;write(folder/'stain.png',values['stain'])
  prompt=pns['binarization_promptv2'](bgr);v,_=ext.tiled(dm,np.concatenate([bgr,prompt],axis=2),'docres',device);np.save(folder/'docres_probability.npy',v[:,:,0]);values['docres']=ink_render((v[:,:,0]<.5).astype('float32'));write(folder/'docres.png',values['docres'])
  values['sauvola']=ink_render(sauvola(a).astype('float32'));write(folder/'sauvola.png',values['sauvola'])
  xx,yy,ww,hh=c['detail_xywh'];sl=np.s_[yy:yy+hh,xx:xx+ww]
  card(folder/'comparison.png',[a[sl],values['sauvola'][sl],values['docres'][sl],values['residual'][sl],values['gain_od_regression'][sl],values['stain'][sl]],
   ['原画像','Sauvola','DocRes','合成RGB差分','合成濃度＋明るさ制約','StainRestorer'],c['id']+' '+c['page'],
   '原画像のみから推論、全例固定条件。学習・チェックポイント選択には実例・テストを使っていない。\n画像は原寸フレーム処理後の表示切り出し。読解・材質・筆画保持の正解未注釈。',size=420)
  stats.append(dict(id=c['id'],frame=c['frame_xywh'],output_difference={m:float(np.abs(np.asarray(v,dtype='float32')-a).mean()/255) for m,v in values.items()}));print('real models',c['id'],flush=True)
 dump(out/'stats.json',stats);protocol=json.loads((out/'protocol.json').read_text())
 for p,h in protocol['dependencies'].items():assert sha(ROOT/p)==h
 dump(out/'complete.json',dict(cases=len(rows),dependencies_unchanged=True))
if __name__=='__main__':
 while not (HERE/'controlled_strokes/complete.json').exists():time.sleep(5)
 main()
