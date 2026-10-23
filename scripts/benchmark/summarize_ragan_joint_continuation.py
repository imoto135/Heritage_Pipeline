"""Write evidence tables and integrity checks for the two joint-training diagnostics."""
import json,sys,shutil
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from compare_v5_aligned_fade_control import dump,sha
from evaluate_ragan_joint_continuation import LABELS
import diagnose_ragan_discriminator as diag

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def summarize(out):
 p=json.loads((out/'protocol.json').read_text());a=json.loads((out/'evaluation/results.json').read_text())
 histories={c:json.loads((out/c/'history.json').read_text()) for c in p['conditions']}
 native={'initial':next(iter(histories.values()))[0]['validation']['total'],**{c:h[-1]['validation']['total'] for c,h in histories.items()}}
 sections=['# 同じ生成器からのG/D同時追加学習',
 '判定は432追加更新の最終重み、二値化閾値0.5で固定。8ページの合成評価と既知実画像5例は学習に入れない。ただし全て既知の開発例であり、未見テスト・正式精度ではない。',
 '## 条件',
 '10周学習済みの同じ生成器・Adam状態（2160更新）を継承し、432更新。生成器lr=3e-5、勾配clip=1、GAN係数=.1。識別器は同じ2160時点の重みから、全GAN条件とも新しいAdamで開始した。従来GANはsigmoid＋scalar hinge/lr1e-6、BCE条件はsigmoid BCE/lr1e-4。BCE同士で変えたのは負例分布だけ。各バッチは4枚、通常BCEは現在のG出力4枚、改変付きは現在G・汚れ追加・墨消去・黒量保存の配置変更を1枚ずつ。GのGAN損失には常に現在G出力4枚を使う。',
 '学習入力：'+p['training']+'。学習24ページから1パッチずつ切り出した256画素の固定画像を反復する診断で、従来の全96パッチ/864入力の本学習とは異なる。無損傷/退色の再提示ありの場合、各位置の入力種を(step+batch内位置)%3で固定した。',
 '公開コード由来の試験実装であり、論文完全再現ではない。既知の注意相殺・バッチ依存の制約は前回監査と同じ。単一seed・同機種GPU2枚での実行で、GPU演算の完全決定性は保証せず、小差の優劣を統計的に主張しない。',
 '## 固定256画素の評価8枚',
 '原本の暫定ラベルとの一致。紙領域は汚れ範囲内の確定扱い画素、墨は同じ範囲内。教師には古典手法由来の暫定マスクを含む。',
 table(['条件','汚れ内の墨保持（多いほど良い）','汚れ内紙の誤黒化（少ないほど良い）','暫定F1'],[[LABELS[c],f"{r['overlap_ink_kept']}/{r['overlap_ink_n']}",f"{r['stain_paper_ink']}/{r['stain_paper_n']}",f"{r['ink_f1_to_provisional']:.6f}"] for c,r in native.items()]),
 '## 512画素の合成評価・退色への波及',
 '墨芯骨格と明紙は原本から算出した代理指標で、専門家GTではない。r2は出力黒画素から2画素以内の被覆。',
 table(['条件','接触汚れ 墨芯r2','接触汚れ 紙の誤黒化','無損傷 墨芯r2','退色.70 墨芯r2','別パッチ既知細線r2'],[[LABELS[c]]+[f"{sum(r[k] for r in a['synthetic'] if r['condition']==c and r['family']==f)}/{sum(r[n] for r in a['synthetic'] if r['condition']==c and r['family']==f)}" for f,k,n in [('contact_stain','core_r2','core_n'),('contact_stain','paper_ink','paper_n'),('clean','core_r2','core_n'),('uniform_fade_070','core_r2','core_n')]]+[f"{next(r for r in a['traced_fade'] if r['condition']==c)['r2']}/49"] for c in native]),
 '## 実画像',
 'D01/D03は原本だけを見て付けた部分的な暫定墨芯。D03の非文字注釈は中央の汚れ全体を覆わないため、非文字誤黒化0を「汚れ除去完了」と解釈しない。D04/T07/C01は固定小領域内の黒画素数であり、物質・隠れた文字の厳密な真値を保証しない。',
 table(['条件','D01 墨芯r2/225','D03上 墨芯r2/124','D04斑点 黒/5100','T07選択部 黒/4500','C01選択部 黒/10800'],[[LABELS[c]]+[str(next(r for r in a['real'] if r['condition']==c and r['case']==case and r['region']==reg)[key]) for case,reg,key in [('D01','thin','ink_r2'),('D03','contact_upper','ink_r2'),('D04','selected_spot_or_stain','nonink_false'),('T07','selected_spot_or_stain','nonink_false'),('C01','selected_spot_or_stain','nonink_false')]] for c in native]),
 'Howe固定の既知基準：D01 181/225、D03上89/124、D04黒0、T07黒0、C01黒7035。今回の追加学習で変更していない。',
 '## 全途中結果',
 '中間重みを選ばず最終を主評価とする。以下は固定合成8枚の途中結果を省略せず記録する。',
 table(['条件','追加更新','暫定F1','汚れ内墨保持','汚れ内紙の誤黒化'],[[LABELS[c],r['step'],f"{r['validation']['total']['ink_f1_to_provisional']:.6f}",r['validation']['total']['overlap_ink_kept'],r['validation']['total']['stain_paper_ink']] for c,h in histories.items() for r in h]),
 '## 原本付き比較画像',
 '\n'.join(f'- [{Path(f).name}]({f})' for f in a['figures']),
 '## 再現・照合',
 '[事前固定プロトコル](protocol.json)、[評価プロトコル](evaluation_protocol.json)、[全数値](evaluation/results.json)、[検証](integrity_verification.json)。',
 ]
 (out/'GとDの同時追加学習.md').write_text('\n\n'.join(sections)+'\n')
 assert sha(diag.CHECKPOINT)==p['checkpoint_sha256']
 schedules=[];checkpoints=[]
 for c in p['conditions']:
  completion=json.loads((out/c/'completion.json').read_text());assert completion['additional_steps']==432
  assert completion['initial_optimizer_steps']==[2160] and completion['final_optimizer_steps']==[2592]
  schedules.append(completion['schedule_sha256'])
  for step in (216,432):
   path=out/c/f'step_{step:05d}.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
   assert payload['additional_steps']==step and payload['step']==2160+step and payload['condition']==c
   checkpoints.append(dict(path=str(path.relative_to(out)),sha256=sha(path)))
 assert len(set(schedules))==1
 baselines=[h[0]['validation']['total'] for h in histories.values()];assert all(b==baselines[0] for b in baselines)
 index=json.loads((diag.OUT/'index.json').read_text())['rows'];train={r['group'] for r in index if r['split']=='train'};val={r['group'] for r in index if r['split']=='val'};assert train.isdisjoint(val)
 for r in index:assert sha(diag.OUT/r['cache'])==r['cache_sha256']
 for f in a['figures']:assert (out/f).exists()
 snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for f in ('run_ragan_joint_continuation.py','evaluate_ragan_joint_continuation.py','summarize_ragan_joint_continuation.py','ragan_pilot_support.py'):
  shutil.copy2(ROOT/'scripts/benchmark'/f,snapshot/f)
 dump(out/'integrity_verification.json',dict(common_generator_checkpoint_unchanged=True,
  all_initial_native_readouts_equal=True,common_G_optimizer_restored_step=2160,final_G_optimizer_step=2592,
  all_training_index_schedules_equal=True,train_val_page_groups_disjoint=True,cache_hashes_checked=len(index),
  checkpoint_hashes=checkpoints,figure_hashes={f:sha(out/f) for f in a['figures']},
  protocol_sha256=sha(out/'protocol.json'),evaluation_protocol_sha256=sha(out/'evaluation_protocol.json'),
  labels_provisional=True,one_seed=True,formal_accuracy=False))
 print(out.name,'verified',len(checkpoints),'checkpoints',len(a['figures']),'figures')

if __name__=='__main__':
 for name in ('ragan_joint_continuation_20261022','ragan_joint_replay_20261022'):
  out=ROOT/'tmp_work'/name
  if (out/'evaluation/results.json').exists():summarize(out)
