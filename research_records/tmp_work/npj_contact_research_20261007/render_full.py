from common import *
from selective import detect,combine
import argparse,time
def records():
 out=[]
 for c in cases():
  if c['id'] in ['T00','T01','T02','T03']:
   out.append(dict(id=c['id'],page=c['page'],source=c['source'],detail_page_xywh=c['page_detail_xywh'],folder=HERE/'full_page',role=c['group']))
 for cohort in ['B','C']:
  f=HERE/'full_followup'/cohort/'protocol.json'
  if f.exists():
   for c in json.loads(f.read_text())['selected']:out.append(dict(c,folder=HERE/'full_followup'/cohort))
 return out
def main():
 cv2.setNumThreads(2);stats=[]
 for c in records():
  id=c['id'];page=c['page'];folder=c['folder'];bp=folder/'output/baseline'/page/'result'/(page+'.png');sp=folder/'output/sauvola'/page/'result'/(page+'.png')
  if not bp.exists() or not sp.exists():continue
  dest=HERE/'full_comparisons';a=read(ROOT/c['source']['original']);b=read(bp);s=read(sp)
  assert a.shape==b.shape==s.shape
  mask,components=detect(a,b);local=combine(a,b,mask,sauvola(a).astype('float32'))
  changed=np.any(local!=b,axis=2);assert not (changed&~mask).any()
  write(HERE/'full_local'/id/'sauvola.png',local);write(HERE/'full_local'/id/'edit_mask.png',mask*255)
  x,y,w,h=c['detail_page_xywh'];arrays=[a,b,s,local]
  card(dest/(id+'.png'),[v[y:y+h,x:x+w] for v in arrays],
   ['原画像','全ページNPJ系基準','全ページSauvolaマスク＋DDRM','全ページ自動範囲＋局所二値化'],id+' '+page,
   '全ページを原寸処理。切り出しは表示のためだけ。原ページROI '+str(c['detail_page_xywh'])+'\n同じDDRMコード・重み・乱数初期値。文字／材質／読解の正解は未注釈。',size=480)
  stats.append(dict(id=id,page=page,role=c['role'],shape=list(a.shape),mask_pixels=int(mask.sum()),changed_pixels=int(changed.sum()),outside_changed=0,
   components=len(components),baseline_sha256=sha(bp),sauvola_ddrm_sha256=sha(sp),comparison=str((dest/(id+'.png')).relative_to(HERE))))
 dump(HERE/'full_comparison_stats.json',stats);print('full comparisons rendered',len(stats),flush=True)
if __name__=='__main__':main()
