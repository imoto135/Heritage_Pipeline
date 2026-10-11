from common import *
from render_full import records
from apply_real_models import selection
from full_page import SNAP,CKPT,PAPER
import time,re,argparse
def main(final=False):
 cv2.setNumThreads(2);result=dict(created=time.time(),source_sha256=sha(__file__),checks={});checks=result['checks'];assert (HERE/'full_followup/C/complete.json').exists()
 manifests=[json.loads((HERE/f).read_text()) for f in ['full_page/protocol.json','full_followup/B/protocol.json','full_followup/C/protocol.json']]
 for m in manifests:
  hashes=m.get('source_code',m.get('baseline_source_code',{}))
  for p,h in hashes.items():assert sha(ROOT/p)==h,p
  assert sha(CKPT)==m.get('checkpoint_sha256',m.get('ckpt_sha256'))
  assert sha(PAPER)==m['paper_sha256']
 checks['frozen_DDRM_code_weights_paper_unchanged']=True
 for f in ['external/stain/manifest.json','external/docres/manifest.json','real_models/protocol.json']:
  for p,h in json.loads((HERE/f).read_text())['dependencies'].items():assert sha(ROOT/p)==h,p
 checks['external_models_and_local_selected_checkpoints_unchanged']=True
 stats=json.loads((HERE/'full_comparison_stats.json').read_text());assert len(stats)==25;lookup={r['id']:r for r in stats};pages=[]
 for c in records():
  a=read(ROOT/c['source']['original']);folder=c['folder'];page=c['page'];id=c['id']
  for variant in ['baseline','sauvola']:
   op=folder/'output'/variant/page/'result'/(page+'.png');b=read(op);assert b.shape==a.shape
   if folder==HERE/'full_page':ip=folder/'inputs'/id/(page+'.png')
   else:ip=folder/'inputs'/variant/id/(page+'.png')
   assert np.array_equal(read(ip),a),(id,variant,'input mismatch')
   md=folder/'masks'/variant/page/'03_text_red.png';red=np.asarray(Image.open(md).convert('L'))<127;assert np.array_equal(b[red],a[red])
   if 'source_sha256' in c:
    for k,h in c['source_sha256'].items():assert sha(ROOT/c['source'][k])==h
  bp=folder/'output/baseline'/page/'result'/(page+'.png');sp=folder/'output/sauvola'/page/'result'/(page+'.png')
  assert sha(bp)==lookup[id]['baseline_sha256'];assert sha(sp)==lookup[id]['sauvola_ddrm_sha256']
  b=read(bp);local=read(HERE/'full_local'/id/'sauvola.png');mask=read(HERE/'full_local'/id/'edit_mask.png')[:,:,0]>127
  outside=int((np.any(local!=b,axis=2)&~mask).sum());assert outside==0
  pages.append(dict(id=id,page=page,shape=list(a.shape),original_sha256=sha(ROOT/c['source']['original']),outside_changed=outside))
 checks['all50_full_page_outputs_native_dimensions_exact_inputs_and_red_overlays']=True;checks['25_local_outputs_zero_change_outside_edit_mask']=True;result['pages']=pages
 for c in selection():
  x,y,w,h=c['frame_xywh'];a=read(ROOT/c['source']['original'])[y:y+h,x:x+w];assert np.array_equal(a,read(HERE/'real_models'/c['id']/'original.png'))
 checks['all25_model_frames_exact_source_crops']=True
 bad=[];logs=list((HERE/'full_page/logs').glob('*.log'))+list((HERE/'full_followup/B/logs').glob('*.log'))+list((HERE/'full_followup/C/logs').glob('*.log'))
 for p in logs:
  content=p.read_text(errors='replace')
  for pattern in ['Traceback','推論結果なし','CSCマスクが見つかりません','背景参照画像が読めません']:
   if pattern in content:bad.append([str(p),pattern])
 assert not bad,bad;checks['full_page_logs_no_failure_missing_patch_or_fallback']=len(logs)
 rs=json.loads((HERE/'synthetic_eval/residual/records.json').read_text())
 for mode in ['gain','gain_od_regression','gain_augmented','stain']:
  rr=json.loads((HERE/'synthetic_eval'/mode/'records.json').read_text());assert len(rr)==len(rs)==1840
  for a,b in zip(rs,rr):assert a['index']==b['index'] and a['domain']==b['domain'] and abs(a['input_mae']-b['input_mae'])<1e-8
 checks['all_synthetic_methods_matching_inputs']=True
 for mode in ['residual','gain','gain_od_regression','gain_augmented']:
  m=json.loads((HERE/'learned'/mode/'manifest.json').read_text());sets={k:set(v) for k,v in m['book_sets'].items()};assert not (sets['train']&sets['dev'] or sets['train']&sets['test'] or sets['dev']&sets['test'])
 checks['small_CNN_source_groups_disjoint']=True
 m=json.loads((HERE/'learned/gain_augmented/manifest.json').read_text());used=set(m['book_sets']['train']+m['book_sets']['dev']);final_groups={c['page'].split('_')[0] for c in selection() if c['id'].startswith('F')};assert not used&final_groups
 checks['final10_real_pages_8_groups_outside_small_CNN_train_dev']=len(final_groups)
 rendering=json.loads((HERE/'evidence/rendering.json').read_text());n=0
 for f in rendering['figures']:
  canvas=read(HERE/f['figure'])
  for p in f['panels']:
   assert sha(ROOT/p['source'])==p['source_sha256'];x,y,w,h=p['xywh'];raw=read(ROOT/p['source'])[y:y+h,x:x+w];assert np.array_equal(raw,read(HERE/p['native_copy']))
   scale=p['integer_scale'];expected=np.repeat(np.repeat(raw,scale,axis=0),scale,axis=1);px,py,pw,ph=p['panel_canvas_xywh'];assert np.array_equal(canvas[py:py+ph,px:px+pw],expected);n+=1
 checks['evidence_panels_exact_native_or_integer_scaling']=n
 packet=json.loads((HERE/'reading_packet/管理者用_割付と出典.json').read_text());assert len(packet['key'])==45
 for p in packet['key']:
  assert sha(ROOT/p['source'])==p['source_sha256'];x,y,w,h=p['source_roi'];assert np.array_equal(read(ROOT/p['source'])[y:y+h,x:x+w],read(HERE/'reading_packet'/p['image']))
 checks['45_reading_preparation_images_exact_native']=True;checks['reading_GT_or_responses_created']=False
 old=json.loads((HERE/'other_book_screen/interim/visual_review.json').read_text())['review'];new=json.loads((HERE/'other_book_screen/final/visual_review.json').read_text())['review'];assert len(old)==26 and len(new)==68;assert not {r['page'] for r in old}&{r['page'] for r in new}
 checks['94_original_visual_reviews_nonoverlapping']=True
 mechanism=json.loads((HERE/'red_filter_study/overlay_mechanism_verification.json').read_text());assert all(r['outside_removed_red_changed']==0 and r['saved_red_original_exact'] for r in mechanism)
 checks['red_overlay_only_ablation_zero_change_outside_rejected_red']=True
 required=['simple_manifest.json','selective_manifest.json','external/stain/complete.json','external/docres/complete.json','learned/residual/complete.json','learned/gain/complete.json',
  'learned/gain_supervised/complete.json','learned/gain_zero/complete.json','learned/gain_od_regression/complete.json','learned/gain_augmented/complete.json',
  'synthetic_eval/gain_augmented/complete.json','synthetic_eval/stain/complete.json','controlled_strokes/complete.json','controlled_strokes_augmented/complete.json','stress_eval/complete.json','real_models/complete.json','real_augmented/complete.json','context_scale/complete.json','red_filter_study/complete.json']
 assert all((HERE/p).exists() for p in required);checks['all_planned_learning_and_diagnostic_jobs_complete']=len(required)
 if final:
  report=(HERE/'2時間作業報告.md').read_text();missing=[]
  for path in re.findall(r'\]\(([^)]+)\)',report):
   if path.startswith(('https://','http://')):continue
   if not (HERE/path).exists():missing.append(path)
  assert not missing,missing;checks['report_local_links_resolve']=True
  bundle=json.loads((HERE/'bundle_verification.json').read_text());assert sha(HERE/'接触シミ_2時間検証_資料.zip')==bundle['zip_sha256'];checks['bundle_integrity_verified']=bundle['integrity']
 result['passed']=True;dump(HERE/('final_verification.json' if final else 'verification.json'),result);print('research verification passed',len(checks),'checks',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();main(a.final)
