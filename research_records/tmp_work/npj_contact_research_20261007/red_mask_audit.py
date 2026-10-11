from common import *
from render_full import records
import time
def retained(a,mask):
 hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV);absolute=((hsv[:,:,0]<=12)|(hsv[:,:,0]>=168))&(hsv[:,:,1]>180)
 n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
 support=np.bincount(labels.ravel(),weights=absolute.ravel(),minlength=n)/np.maximum(stats[:,cv2.CC_STAT_AREA],1)
 keep=support>=.05;keep[0]=False
 return keep[labels],absolute
def main():
 cv2.setNumThreads(2);rows=[]
 for c in records():
  a=read(ROOT/c['source']['original']);p=ROOT/c['source']['npj_red'];red=np.asarray(Image.open(p).convert('L'))<127;keep,absolute=retained(a,red)
  h=sha(p);r=dict(id=c['id'],page=c['page'],stored_red_pixels=int(red.sum()),retained_pixels=int(keep.sum()),rejected_pixels=int((red&~keep).sum()),absolute_red_support=int((red&absolute).sum()),source_sha256=h)
  bp=c['folder']/'output/baseline'/c['page']/'result'/(c['page']+'.png')
  if bp.exists():
   b=read(bp);r['original_equal_on_stored_red']=bool(np.array_equal(a[red],b[red]))
  if c['id'] in ['N02','N03','N04','F41']:
   dest=HERE/'red_audit'/c['id'];write(dest/'stored_red_mask.png',red*255);write(dest/'filtered_red_mask.png',keep*255)
   overlay=a.copy();overlay[red]=np.rint(.65*a[red]+.35*np.array([255,0,0]));filtered=a.copy();filtered[keep]=np.rint(.65*a[keep]+.35*np.array([255,0,0]))
   x,y,w,hh=c['detail_page_xywh'];card(dest/'comparison.png',[im[y:y+hh,x:x+w] for im in [a,overlay,filtered]],['原画像','保存済み赤マスク範囲','既存コードの絶対色確認後'],c['id']+' '+c['page'],
    '赤マスクは原画像を最後に貼り戻す範囲。赤インクの正解注釈ではない。\n既存CSCコードと同じHSV規則・成分支持率0.05。新しい提案手法ではない。',size=500)
  assert sha(p)==h;rows.append(r)
 dump(HERE/'red_audit/summary.json',dict(created=time.time(),records=rows,rule='existing csc.py HSV hue<=12 or>=168, saturation>180; connected component support>=.05',source_sha256=sha(__file__),material_GT=False,ink_color_GT=False))
 print('red audit',len(rows),'pages',[(r['id'],r['stored_red_pixels'],r['retained_pixels']) for r in rows if r['id'] in ['N02','N03','N04','F41']])
if __name__=='__main__':main()
