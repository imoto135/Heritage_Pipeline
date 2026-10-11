from common import *
from render_full import records
from selective import detect,combine
cv2.setNumThreads(2);old=json.loads((HERE/'full_comparison_stats.json').read_text());cache={r['id']:r for r in old};stats=[];made=0
for c in records():
 id=c['id'];page=c['page'];folder=c['folder'];bp=folder/'output/baseline'/page/'result'/(page+'.png');sp=folder/'output/sauvola'/page/'result'/(page+'.png')
 if not bp.exists() or not sp.exists():continue
 bh,sh=sha(bp),sha(sp);prior=cache.get(id)
 if prior and prior['baseline_sha256']==bh and prior['sauvola_ddrm_sha256']==sh and (HERE/'full_comparisons'/(id+'.png')).exists():stats.append(prior);continue
 a=read(ROOT/c['source']['original']);b=read(bp);s=read(sp);assert a.shape==b.shape==s.shape
 mask,components=detect(a,b);local=combine(a,b,mask,sauvola(a).astype('float32'));changed=np.any(local!=b,axis=2);assert not (changed&~mask).any()
 write(HERE/'full_local'/id/'sauvola.png',local);write(HERE/'full_local'/id/'edit_mask.png',mask*255)
 x,y,w,h=c['detail_page_xywh'];card(HERE/'full_comparisons'/(id+'.png'),[v[y:y+h,x:x+w] for v in [a,b,s,local]],
  ['原画像','全ページNPJ系基準','全ページSauvolaマスク＋DDRM','全ページ自動範囲＋局所二値化'],id+' '+page,
  '全ページを原寸処理。切り出しは表示のためだけ。原ページROI '+str(c['detail_page_xywh'])+'\n同じDDRMコード・重み・乱数初期値。文字／材質／読解の正解は未注釈。',size=480)
 stats.append(dict(id=id,page=page,role=c['role'],shape=list(a.shape),mask_pixels=int(mask.sum()),changed_pixels=int(changed.sum()),outside_changed=0,
  components=len(components),baseline_sha256=bh,sauvola_ddrm_sha256=sh,comparison=str((HERE/'full_comparisons'/(id+'.png')).relative_to(HERE)),renderer_sha256=sha(__file__)));made+=1
dump(HERE/'full_comparison_stats.json',stats);print('incremental full comparisons',len(stats),'new',made,flush=True)
if len(stats)==len(records())==25:dump(HERE/'full_render_complete.json',dict(cases=25,source_sha256=sha(__file__),stats_sha256=sha(HERE/'full_comparison_stats.json')))
