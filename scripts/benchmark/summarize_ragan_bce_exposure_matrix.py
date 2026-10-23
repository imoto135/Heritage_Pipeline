"""Aggregate the exploratory 2x2 and render original + all four cells per order."""
import json,sys,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_ragan_full_continuation as full
from run_ragan_fine_retention_control import OUT as FINE
from run_ragan_stable_bce_control import OUT as STABLE
from run_ragan_stable_exposure_control import OUT,CONDITIONS
from compare_v5_aligned_fade_control import sha,dump,rgb,mask
import apply_ragan_real_regions as real
import diagnose_ragan_discriminator as diag
CELLS=[('従来BCE・通常',FINE,'baseline_order'),('従来BCE・提示20%',FINE,'exposure20_order'),('安定BCE・通常',STABLE,'stable_bce_order'),('安定BCE・提示20%',OUT,'stable_exposure_order')]
def load(path):return json.loads(path.read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 data={};rows=[];primary=[]
 for label,folder,prefix in CELLS:
  for step in (432,864):
   dest=folder/('evaluation' if step==864 else 'evaluation_00432');data[(label,step)]=(load(dest/'results.json'),load(dest/'training_reviewed_results.json'))
  native=load(folder/'evaluation/training_native_results.json')
  for order in (0,1):
   c=prefix+str(order);a,tr=data[(label,864)];fade=next(r for r in a['traced_fade'] if r['condition']==c)
   def rm(case,reg,key):return next(r[key] for r in a['real'] if r['condition']==c and r['case']==case and r['region']==reg)
   def sm(family,key):return sum(r[key] for r in a['synthetic'] if r['condition']==c and r['family']==family)
   r=dict(label=label,order=order,condition=c,source=str(folder.relative_to(ROOT)),
    training_native_r0=sum(r['r0'] for r in native['rows'] if r['condition']==c and r['view']=='actual_batch4'),
    training_native_r2=sum(r['r2'] for r in native['rows'] if r['condition']==c and r['view']=='actual_batch4'),
    training512_r2=sum(r['r2'] for r in tr['rows'] if r['condition']==c and r['family']=='targeted_fade'),
    fade_r0=fade['r0'],fade_r2=fade['r2'],D01=rm('D01','thin','ink_r2'),D03upper=rm('D03','contact_upper','ink_r2'),
    stain_core=sm('contact_stain','core_r2'),stain_paper=sm('contact_stain','paper_ink'),
    D04=rm('D04','selected_spot_or_stain','nonink_false'),T07=rm('T07','selected_spot_or_stain','nonink_false'),C01=rm('C01','selected_spot_or_stain','nonink_false'),clean_core=sm('clean','core_r2'))
   primary.append(r);rows.append([label,order+1,r['training_native_r2'],f"{r['fade_r0']} / {r['fade_r2']}",r['D01'],r['stain_core'],r['stain_paper'],r['D04']])
 figures=[];classic={c['id']:c for c in load(real.CLASSIC/'protocol.json')['cases']};contact=ROOT/'tmp_work/makura_contact_stain_train_control_20261022'
 cases=[classic[k] for k in ['D01','D03','D04']]+load(contact/'other_book_transfer/protocol.json')['cases']
 def render(key,title,items,order):
  path=OUT/'matrix_figures'/f'order{order+1}'/(key+'.png');real.panel(path,title,items);figures.append(str(path.relative_to(OUT)))
 for order in (0,1):
  for case in cases:
   key=case['id'];x,y,w,h=case.get('roi',case.get('crop_xywh'));source=rgb(ROOT/case['source'])[y:y+h,x:x+w]
   hp=real.CLASSIC/'crops'/key/'howe_fixed.png'
   if not hp.exists():hp=contact/'other_book_transfer/masks'/key/'howe_fixed.png'
   items=[('原画像',source),('Howe固定',np.where(mask(hp),0,255).astype(np.uint8)),('再開前10周',np.where(mask(FINE/'evaluation/real/initial'/(key+'.png')),0,255).astype(np.uint8))]
   for label,folder,prefix in CELLS:items.append((label,np.where(mask(folder/'evaluation/real'/(prefix+str(order))/(key+'.png')),0,255).astype(np.uint8)))
   render(key+'_原本と4条件',key+f' / 損失数値計算×提示頻度 / 順序{order+1}',items,order)
  folder=ROOT/'tmp_work/targeted_thin_fade_20261021/cases/alpha_0.70'
  items=[('退色前原本',rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val/200017197_00172_0_p00/clean.png')),('退色入力 .70',rgb(folder/'input.png')),('Howe固定',np.where(mask(folder/'howe_fixed.png'),0,255).astype(np.uint8)),('再開前10周',np.where(mask(FINE/'evaluation/traced_fade/initial.png'),0,255).astype(np.uint8))]
  for label,folder,prefix in CELLS:items.append((label,np.where(mask(folder/'evaluation/traced_fade'/(prefix+str(order)+'.png')),0,255).astype(np.uint8)))
  render('退色文字_原本と4条件','学習外の既知細線49点 / 損失×提示頻度',items,order)
  for patch,(_,trace) in full.reviewed().items():
   records=full.read_rows('train');idx=next(i for i,r in enumerate(records) if r['patch_id']==patch and r['damage_family']=='targeted_fade' and float(r['damage_strength'])==.8);r=records[idx];x,y=full.coordinates(idx,13,r,full.metadata());t=full.gray(ROOT/r['labels']['target'])[y:y+256,x:x+256].copy();v=full.gray(ROOT/r['labels']['valid'])[y:y+256,x:x+256]>0;t[~v]=128
   items=[('原本の学習切り出し',rgb(ROOT/r['clean'])[y:y+256,x:x+256]),('強退色.80入力',rgb(ROOT/r['input'])[y:y+256,x:x+256]),('暫定教師 / 不明は灰色',t)]
   for label,folder,prefix in CELLS:items.append((label,np.where(mask(folder/'evaluation/training_native_masks'/(prefix+str(order))/(patch+'_actual_batch4.png')),0,255).astype(np.uint8)))
   render(patch+'_256学習窓と4条件',patch+' / 各条件で提示した4枚組の最終推論',items,order)
 # All midpoints reported, with the final checkpoint kept primary.
 middle=[]
 for label,folder,prefix in CELLS:
  for n in (0,1):
   c=prefix+str(n);values=[]
   for step in [432,864]:
    a,tr=data[(label,step)];f=next(r['r2'] for r in a['traced_fade'] if r['condition']==c)
    values.append((sum(r['r2'] for r in tr['rows'] if r['condition']==c and r['family']=='targeted_fade'),f))
   middle.append([label,n+1,f'{values[0][0]} → {values[1][0]}',f'{values[0][1]} → {values[1][1]}'])
 text=['# BCE数値計算と強退色提示頻度の2×2対照',
 '既存の教師・RGB・生成器構造を変えず、GANなしで「従来／安定BCE」と「通常／強退色20%提示」を比較した。各組合せに同じ10周G/Adamからの2つのバッチ順を使用し、864更新後の14周を主評価とする。数値飽和の発見後に段階的に追加した探索的対照で、全実験を最初から事前登録したものではない。',
 '## 判明した損失実装の問題',
 'ローカル試験用損失では、白へ飽和した真の墨画素に直接の教師あり勾配が届かなくなる。強退色の残存失敗44点すべてでp=1.0、従来の直接勾配0を確認した。生成器の最終出力p=σ(a)σ(b)を変えず、等価logit a+b−logsumexp(0,a,b)からBCEWithLogitsを計算し、BCE経路だけを修正した。L1/Dice/Jaccardと公開forwardは維持した。これはRA-GAN論文本来の能力の問題とは区別する。',
 '[飽和画素の監査](../ragan_fine_retention_control_20261022/fine_saturation_audit.json)、[数式・損失・勾配と不明マスクの性質確認](loss_checks.json)、[BCEだけの対照](../ragan_stable_bce_control_20261022/BCE飽和の修正と対照結果.md)。',
 '4枚だけの反復学習では、従来損失でも648更新で強退色76/76点を拾った。直接勾配0でも共有重みの他画素からの更新で回復し得るため、飽和だけを全失敗の唯一の原因にしない。この診断は固定4枚への当てはまりで、汎化や採用モデルの性能ではない。[局所診断](../ragan_fine_retention_control_20261022/hard_fine_fit_probe/results.json)。',
 '## 14周の全結果',
 '墨芯r2は出力黒画素から2画素以内。学習455点は強退色.80、学習外49点は別入力の.70。学習外49の欄はr0/r2を併記する。合成の墨芯・紙は原本からの代理、実画像墨芯は部分的な暫定注釈で、専門家GTの正式精度ではない。',
 table(['損失・提示','順序','学習455・256窓r2','学習外49 r0/r2','D01/225','汚れ下墨/596','汚れ内紙の黒/18105','D04黒/5100'],rows),
 '再開前10周：学習256窓220/455、学習外48/49、D01 195/225、合成汚れ下墨567/596・紙黒1161/18105、D04黒2668/5100。Howe固定：学習外42/49、D01 181/225、合成汚れ下墨412/596・紙黒0/18105、D04黒0。',
 '## 保全と別本の選択汚れ部',
 table(['損失・提示','順序','無損傷墨芯/5500','D03上側/124','T07黒/4500','C01黒/10800'],[[r['label'],r['order']+1,r['clean_core'],r['D03upper'],r['T07'],r['C01']] for r in primary]),
 '## 12周→14周',table(['損失・提示','順序','学習455の512推論r2','学習外49r2'],middle),
 '## 原本付き4条件PNG', '\n'.join(f'- [{Path(f).name}・{Path(f).parent.name}]({f})' for f in figures),
 '## 制約と再現',
 '20%提示は元の各周864位置のうち非対象166位置を強退色対象に替え、対象提示を7→173回にする。他画像の提示と損傷族の比率も変わる。RGB・教師・切り出し式を変える調整ではないが、純粋な損失変更とは分ける。8倍墨芯損失は別に試し、結果が順序依存だった。[8倍損失・提示頻度の全対照](../ragan_fine_retention_control_20261022/細線の損失と提示頻度の対照結果.md)。',
 '単一本の暫定教師、既知の開発例、共通Gに対するバッチ順感度の2試行であり、独立初期化反復・盲検別本テストではない。公開版GAの注意相殺は残る。以前の教師ありモデル結果は従来の損失に依存し、学習配分やGANの能力だけへ帰属させない。',
 '[この追加セルの事前固定プロトコル](protocol.json)、[主評価全数値](evaluation/results.json)、[12周全数値](evaluation_00432/results.json)、[2×2集計](matrix_results.json)、[検証](integrity_verification.json)。',
 ]
 (OUT/'損失と提示頻度の2x2比較.md').write_text('\n\n'.join(text)+'\n');dump(OUT/'matrix_results.json',dict(rows=primary,figures=figures,formal_accuracy=False))
 # New cell integrity and cross-cell input/order identity.
 p=load(OUT/'protocol.json');assert sha(diag.CHECKPOINT)==p['checkpoint_sha256'];ck=[]
 for c,conf in CONDITIONS.items():
  assert sha(OUT/'batch_schedules'/(c+'.json'))==sha(FINE/'batch_schedules'/f"exposure20_order{conf['order']}.json")
  assert load(OUT/c/'completion.json')['final_optimizer_steps']==[3024]
  for step in [216,432,648,864]:
   path=OUT/c/f'step_{step:05d}.pt';q=torch.load(path,map_location='cpu',weights_only=False);assert q['step']==2160+step and q['discriminator'] is None
   assert {int(s['step']) for s in q['optimizer']['state'].values() if 'step' in s}=={2160+step};ck.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
 for file in ['source_hashes.json','crop_schedule.json']:assert sha(OUT/file)==sha(STABLE/file)==sha(FINE/file)
 for file in ['run_ragan_stable_exposure_control.py','run_ragan_stable_bce_control.py','run_ragan_full_continuation.py','ragan_pilot_support.py','evaluate_ragan_fine_retention_control.py','evaluate_ragan_stable_exposure_control.py','evaluate_ragan_joint_continuation.py']:
  assert sha(ROOT/'scripts/benchmark'/file)==sha(OUT/'source_snapshot'/file)
 for file in ['results.json','training_reviewed_results.json','training_native_results.json']:
  if file=='training_native_results.json':dirs=['evaluation']
  else:dirs=['evaluation','evaluation_00432']
  for directory in dirs:figures+=load(OUT/directory/file)['figures']
 shutil.copy2(__file__,OUT/'source_snapshot'/Path(__file__).name)
 dump(OUT/'integrity_verification.json',dict(checkpoints=ck,initial_G_and_Adam_common=True,frequency_schedules_identical_to_previous20percent=True,
  RGB_labels_and_crop_schedule_identical_across_matrix=True,only_BCE_path_changed=True,final_optimizer_step=3024,
  protocol_sha256=sha(OUT/'protocol.json'),matrix_results_sha256=sha(OUT/'matrix_results.json'),figure_hashes={f:sha(OUT/f) for f in figures},formal_accuracy=False))
 print('Verified',len(ck),'checkpoints;',len(figures),'figure entries');print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
