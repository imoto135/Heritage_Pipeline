"""Verify and summarize the BCE-only numerical control."""
import json,sys,shutil
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_stable_bce_control import OUT,PARENT,CONDITIONS
from compare_v5_aligned_fade_control import sha,dump
import diagnose_ragan_discriminator as diag
LABELS={'initial':'再開前10周','baseline_order0':'従来BCE 順序1','baseline_order1':'従来BCE 順序2','stable_bce_order0':'安定BCE 順序1','stable_bce_order1':'安定BCE 順序2'}
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 a=json.loads((OUT/'evaluation/results.json').read_text());train=json.loads((OUT/'evaluation/training_reviewed_results.json').read_text());native=json.loads((OUT/'evaluation/training_native_results.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
 def metric(c,case,reg,key):return next(r[key] for r in a['real'] if r['condition']==c and r['case']==case and r['region']==reg)
 def count(c,f,key):return sum(r[key] for r in a['synthetic'] if r['condition']==c and r['family']==f)
 def thin(c):return next(r for r in a['traced_fade'] if r['condition']==c)
 def fit(c):return sum(r['r2'] for r in train['rows'] if r['condition']==c and r['family']=='targeted_fade')
 def nat(c):return sum(r['r2'] for r in native['rows'] if r['condition']==c and r['view']=='actual_batch4')
 rows=[[LABELS[c],nat(c),fit(c),f"{thin(c)['r0']} / {thin(c)['r2']}",metric(c,'D01','thin','ink_r2'),metric(c,'D03','contact_upper','ink_r2'),count(c,'contact_stain','core_r2'),count(c,'contact_stain','paper_ink'),metric(c,'D04','selected_spot_or_stain','nonink_false')] for c in LABELS]
 text=['# 試験用BCEの飽和時の勾配修正',
 'ローカルの教師あり損失は、出力確率を[1e-6,1-1e-6]へclampしてからBCE・L1・重なり損失を計算していた。誤って白確率1.0になった墨では、その画素自身の損失勾配がclampとsigmoidの飽和によりゼロになる。レビュー対象の強退色44点すべてでこの状態を確認した。これは公開RA-GAN論文の能力限界ではなく、今回の試験用損失・計算経路の問題である。',
 '## 修正した範囲',
 '公開生成器の最後のGateConvはp=σ(a)σ(b)を出す。その厳密な等価logitは a+b−logsumexp(0,a,b)。この活性化前値をforward hookで取得し、BCEWithLogitsでBCEだけを計算した。生成器の公開forward、重み構造、入力、教師、切り出し、バッチ順、Adam初期状態、学習率、更新数、L1/Dice/Jaccard、推論閾値は変えない。モデルの出力pにBCEを掛けるだけでは、sigmoidが丸めで1.0になった時に勾配を失うため、この計算経路を使う。',
 '非飽和時の確率・損失・勾配一致、飽和した誤りへの非ゼロ勾配、不明領域の教師値不変性・勾配ゼロを検証した。単なる表示精度の変更ではなく、従来clampで打ち切っていたBCE値と勾配を安定に計算する修正である。[性質のチェック](loss_checks.json)。',
 '同じ10周G/Adamから864更新。2通りの元の通常学習順をそのまま使い、GANなし・重み1・追加の提示頻度変更なしで比較した。従来BCEの通常学習出力は前回検証済みキャッシュを再利用する。14周を主評価、12周を途中評価に固定した。',
 '## 14周の結果',
 '墨芯は2画素以内被覆。学習455点は強退色.80、学習外49点は別入力の.70であり、直接の比で汎化ギャップを算出しない。',
 table(['条件','学習455・256','学習455・512','学習外49 r0/r2','D01/225','D03上側/124','汚れ下墨/596','汚れ内紙の黒/18105','D04黒/5100'],rows),
 '## その他の保全',
 table(['条件','無損傷墨芯/5500','退色2例墨芯/5500','T07黒/4500','C01黒/10800'],[[LABELS[c],count(c,'clean','core_r2'),count(c,'uniform_fade_070','core_r2'),metric(c,'T07','selected_spot_or_stain','nonink_false'),metric(c,'C01','selected_spot_or_stain','nonink_false')] for c in LABELS]),
 '## 比較PNG',
 '\n'.join(f'- [{Path(f).name}]({f})' for f in a['figures']),
 '学習時256画素：\n'+'\n'.join(f'- [{Path(f).name}]({f})' for f in native['figures']),
 '## 解釈の制約',
 '以前の提示頻度・損失重み・GANの比較には、損失の飽和による直接勾配ゼロが混在していた。既存の数値と重みは書き換えず、試験用損失に依存する結果として残す。損傷調整やGANの能力限界だけに帰属させない。修正だけで画質が改善するかは上の対照で評価し、勾配が非ゼロになったことを復元精度の改善とは扱わない。',
 '既知の開発例、暫定教師、単一本の学習であり、正式精度や未見別本への汎化を証明しない。2通りは共通学習済みGのバッチ順感度で、独立モデル初期化ではない。公開版GAの相殺も残るため論文完全再現ではない。',
 '[事前固定プロトコル](protocol.json)、[主評価全数値](evaluation/results.json)、[12周全数値](evaluation_00432/results.json)、[検証](integrity_verification.json)。',
 ]
 (OUT/'BCE飽和の修正と対照結果.md').write_text('\n\n'.join(text)+'\n')
 assert sha(diag.CHECKPOINT)==p['checkpoint_sha256'] and sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
 ck=[]
 for c,conf in CONDITIONS.items():
  complete=json.loads((OUT/c/'completion.json').read_text());assert complete['final_optimizer_steps']==[3024]
  assert sha(OUT/'batch_schedules'/(c+'.json'))==sha(PARENT/'batch_schedules'/f"no_gan_order{conf['order']}.json")
  for step in p['saved_steps']:
   path=OUT/c/f'step_{step:05d}.pt';q=torch.load(path,map_location='cpu',weights_only=False)
   assert q['step']==2160+step and q['discriminator'] is None and {int(s['step']) for s in q['optimizer']['state'].values() if 'step' in s}=={2160+step}
   ck.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
 for name in ['run_ragan_stable_bce_control.py','run_ragan_full_continuation.py','ragan_pilot_support.py','evaluate_ragan_fine_retention_control.py','evaluate_ragan_stable_bce_control.py','evaluate_ragan_joint_continuation.py']:
  assert sha(ROOT/'scripts/benchmark'/name)==sha(OUT/'source_snapshot'/name)
 for c in LABELS:
  for view in ['actual_batch4','single_crop1']:assert sum(r['n'] for r in native['rows'] if r['condition']==c and r['view']==view)==455
 figures=[]
 for directory in ['evaluation','evaluation_00432']:
  for file in ['results.json','training_reviewed_results.json']:figures+=json.loads((OUT/directory/file).read_text())['figures']
  current=json.loads((OUT/directory/'results.json').read_text());old=json.loads((PARENT/directory/'results.json').read_text());assert current['baseline_cache_results_sha256']==sha(PARENT/directory/'results.json')
  for oldc,newc in [('initial','initial'),('no_gan_order0','baseline_order0'),('no_gan_order1','baseline_order1')]:
   for kind in ['real','synthetic','traced_fade']:assert [dict(r,condition=newc) for r in old[kind] if r['condition']==oldc]==[r for r in current[kind] if r['condition']==newc]
 figures+=native['figures'];shutil.copy2(__file__,OUT/'source_snapshot'/Path(__file__).name)
 dump(OUT/'integrity_verification.json',dict(new_checkpoints=ck,common_initial_G_and_Adam=True,initial_optimizer_step=2160,final_optimizer_step=3024,
  schedules_equal_original_noGAN=True,non_BCE_terms_unchanged=True,public_forward_unchanged=True,loss_properties_verified=json.loads((OUT/'loss_checks.json').read_text()),baseline_cache_verified=True,
  source_hashes_checked_by_each_training_run=len(json.loads((OUT/'source_hashes.json').read_text())),figure_hashes={f:sha(OUT/f) for f in figures},formal_accuracy=False))
 print('Verified',len(ck),'checkpoints',len(figures),'PNGs');print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
