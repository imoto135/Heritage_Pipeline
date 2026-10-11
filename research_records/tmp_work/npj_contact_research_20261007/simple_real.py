from common import *
import time
def main():
 cv2.setNumThreads(2)
 config=dict(sauvola=[(31,.15),(61,.2),(101,.2),(151,.2)],morphology=[(8,.06),(15,.08),(25,.1)],
  same_parameters_all_cases=True,training=False,GT=False,original_only=True,
  render='fixed neutral paper times continuous darkness; display contrast is not a pixel GT')
 dump(HERE/'simple_manifest.json',dict(config=config,source_sha256=sha(__file__),selection_sha256=sha(SOURCE/'benchmark_candidates.json'),created=time.time()))
 for c in cases():
  id=c['id'];a=read(SOURCE/'native'/id/'original_context.png');b=read(SOURCE/'native'/id/'npj_context.png')
  out=HERE/'simple'/id;write(out/'original.png',a);write(out/'npj.png',b)
  arrays=[a,b];labels=['原画像','NPJ系基準']
  for win,k in config['sauvola']:
   v=ink_render(sauvola(a,win,k).astype('float32'));name=f'sauvola_w{win}_k{k}'
   write(out/(name+'.png'),v)
   if win in [61,151]:arrays.append(v);labels.append(f'Sauvola w={win}')
  for rad,t in config['morphology']:
   d,p=morph_contrast(a,rad);alpha=np.clip((d-t)/.3,0,1);v=ink_render(alpha);name=f'morph_r{rad}_t{t}'
   write(out/(name+'.png'),v);np.save(out/(name+'_contrast.npy'),d)
   if rad==15:arrays.append(v);labels.append('局所背景との差')
  x,y,w,h=c['detail_xywh'];arr=[v[y:y+h,x:x+w] for v in arrays]
  card(HERE/'simple_comparisons'/f'{id}.png',arr,labels,id+' '+c['page'],
   '全例共通条件。二値化／背景差分は原画像だけを使用。\n文字の正解・読解改善は未評価。原画像とNPJ系基準は無補正。',size=440)
 print('simple real cases complete',len(cases()),flush=True)
if __name__=='__main__':main()
