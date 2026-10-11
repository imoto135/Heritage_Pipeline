from common import *
from render_full import records
import time
def homomorphic(a,sigma):
 x=np.log(np.maximum(a.astype('float32')/255,.01));smooth=cv2.GaussianBlur(x,(0,0),sigma)
 reference=np.quantile(x.reshape(-1,3),.8,axis=0)
 return np.exp(x-smooth+reference)*255
def main():
 cv2.setNumThreads(2);stats=[]
 for c in records():
  x,y,w,h=c['detail_page_xywh'];a=read(ROOT/c['source']['original']);dx=max(0,x-128);dy=max(0,y-128)
  frame=a[dy:min(a.shape[0],y+h+128),dx:min(a.shape[1],x+w+128)];dest=HERE/'frequency'/c['id'];arrays=[frame]
  for sigma in [12,24,48]:
   v=homomorphic(frame,sigma);write(dest/f'log_gaussian_{sigma}.png',v);arrays.append(v)
  sl=np.s_[y-dy:y-dy+h,x-dx:x-dx+w]
  card(dest/'comparison.png',[v[sl] for v in arrays],['原画像','log 背景 σ=12','log 背景 σ=24','log 背景 σ=48'],c['id']+' '+c['page'],
   '低周波の背景を除いて同一フレームの明るい色へ戻す。全例同じ規則。\n表示ROIは既存選択、処理フレームは128px余白付き。紙色・汚れ・文字の正解は未注釈。',size=440)
  stats.append(dict(id=c['id'],frame_page_xywh=[dx,dy,frame.shape[1],frame.shape[0]],display_page_xywh=c['detail_page_xywh']))
 dump(HERE/'frequency/protocol.json',dict(records=stats,source_sha256=sha(__file__),sigmas=[12,24,48],created=time.time(),role='scale-separation baseline, not a newly proposed method',formula='exp(log(I) - Gaussian(log(I)) + channelwise80percentile(log(I)))'))
 print('frequency baselines',len(stats),flush=True)
if __name__=='__main__':main()
