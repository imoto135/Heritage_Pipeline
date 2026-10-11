from common import *
import time
def detect(a,b,window=61,k=.2,radius=5,minarea=150):
 observed=sauvola(a,window,k);black=gray(b)<.25
 disagreement=(black&~observed).astype('uint8')
 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
 broad=cv2.morphologyEx(disagreement,cv2.MORPH_OPEN,kernel)
 n,lab,stats,_=cv2.connectedComponentsWithStats(broad,8)
 mask=np.zeros(black.shape,'uint8');components=[]
 for j in range(1,n):
  x,y,w,h,area=map(int,stats[j])
  if area<minarea:continue
  mask[lab==j]=1;components.append(dict(xywh=[x,y,w,h],seed_area=area))
 mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
 # Fill holes so observed strokes inside a stain receive the same local style.
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(mask,contours,-1,1,cv2.FILLED)
 return mask.astype(bool),components
def paper_field(b):
 valid=(gray(b)>.6).astype('float32');weight=cv2.GaussianBlur(valid,(0,0),24)
 numerator=cv2.GaussianBlur(b.astype('float32')*valid[:,:,None],(0,0),24)
 fallback=np.median(b[valid>.5],axis=0) if valid.any() else np.array([235,224,200])
 return np.where((weight>.002)[:,:,None],numerator/np.maximum(weight[:,:,None],.002),fallback)
def combine(a,b,mask,ink):
 paper=paper_field(b);local=paper*(1-ink[:,:,None]);out=b.copy()
 out[mask]=np.clip(np.rint(local[mask]),0,255).astype('uint8');return out
def main():
 cv2.setNumThreads(2);stats=[]
 config=dict(window=61,k=.2,opening_radius=5,minarea=150,dilation=12,closing=15,
  original_plus_npj=True,automatic_edit_mask=True,ROI_used_for_inference=False,
  heuristic_stain_detection=True,stain_material_GT=False,parameters_fixed_across_cases=True)
 dump(HERE/'selective_manifest.json',dict(config=config,created=time.time(),source_sha256=sha(__file__)))
 for c in cases():
  id=c['id'];a=read(SOURCE/'native'/id/'original_context.png');b=read(SOURCE/'native'/id/'npj_context.png')
  mask,components=detect(a,b);out=HERE/'selective'/id
  write(out/'edit_mask.png',mask*255);overlay=a.copy();overlay[mask]=np.rint(.65*overlay[mask]+.35*np.array([255,60,40]))
  write(out/'mask_overlay.png',overlay)
  for name,ink in [('sauvola',sauvola(a).astype('float32')),('docres_hard',(np.load(HERE/'external/docres'/id/'background_probability.npy')<.5).astype('float32')),
    ('docres_soft',np.clip((1-np.load(HERE/'external/docres'/id/'background_probability.npy')-.1)/.8,0,1))]:
   value=combine(a,b,mask,ink);write(out/(name+'.png'),value)
   changed=np.any(value!=b,axis=2);assert not (changed&~mask).any()
   stats.append(dict(id=id,method=name,mask_pixels=int(mask.sum()),changed_pixels=int(changed.sum()),outside_changed=0,components=components))
  x,y,w,h=c['detail_xywh'];arrays=[a,b,overlay,read(out/'sauvola.png'),read(out/'docres_hard.png')]
  card(HERE/'selective_comparisons'/f'{id}.png',[v[y:y+h,x:x+w] for v in arrays],
   ['原画像','NPJ系基準','自動処理候補範囲','局所Sauvola反映','局所DocRes反映'],id+' '+c['page'],
   '原画像とNPJ系基準の不一致から範囲を推定。全例同じ条件、表示ROIは処理に不使用。\nマスク外の出力はNPJ系基準と画素一致。材質・筆画保持・読解の正解は未確定。',size=420)
 dump(HERE/'selective_stats.json',stats);print('selective complete',len(stats),flush=True)
if __name__=='__main__':main()
