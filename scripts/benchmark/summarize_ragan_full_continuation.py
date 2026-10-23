"""Summarize paired full-composition continuations without selecting checkpoints."""
import json,sys,shutil,re
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_full_continuation import OUT,PARENT,CONDITIONS
from compare_v5_aligned_fade_control import sha,dump
from evaluate_ragan_full_continuation import LABELS
import diagnose_ragan_discriminator as diag

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def totals(a,condition,family,key):return sum(r[key] for r in a['synthetic'] if r['condition']==condition and r['family']==family)
def metric(a,c,case,region,key):return next(r[key] for r in a['real'] if r['condition']==c and r['case']==case and r['region']==region)
def thin(a,c):return next(r['r2'] for r in a['traced_fade'] if r['condition']==c)
def main():
 p=json.loads((OUT/'protocol.json').read_text());a=json.loads((OUT/'evaluation/results.json').read_text());b=json.loads((OUT/'evaluation_00432/results.json').read_text())
 train=json.loads((OUT/'evaluation/training_reviewed_results.json').read_text())
 histories={c:json.loads((OUT/c/'history.json').read_text()) for c in CONDITIONS};conditions=['initial',*CONDITIONS]
 labels={'initial':'再開前10周',**LABELS}
 rows=[]
 for c in conditions:
  rows.append([labels[c],f'{thin(a,c)}/49',f"{metric(a,c,'D01','thin','ink_r2')}/225",f"{metric(a,c,'D03','contact_upper','ink_r2')}/124",
   f"{totals(a,c,'contact_stain','core_r2')}/596",f"{totals(a,c,'contact_stain','paper_ink')}/18105",
   metric(a,c,'D04','selected_spot_or_stain','nonink_false'),metric(a,c,'T07','selected_spot_or_stain','nonink_false'),metric(a,c,'C01','selected_spot_or_stain','nonink_false')])
 paired=[]
 for n in (0,1):
  no=f'no_gan_order{n}';gan=f'reference_gan_order{n}'
  paired.append([n+1,thin(a,gan)-thin(a,no),metric(a,gan,'D01','thin','ink_r2')-metric(a,no,'D01','thin','ink_r2'),
   metric(a,gan,'D03','contact_upper','ink_r2')-metric(a,no,'D03','contact_upper','ink_r2'),
   totals(a,gan,'contact_stain','paper_ink')-totals(a,no,'contact_stain','paper_ink'),
   metric(a,gan,'D04','selected_spot_or_stain','nonink_false')-metric(a,no,'D04','selected_spot_or_stain','nonink_false')])
 text=['# 元の96パッチ構成でのGAN対照',
 '同じ10周生成器とAdam状態から4周分（864更新）を追加し、GANなしと従来GANを2通りのバッチ順で対比較した。主評価は14周時点の最終重み、閾値0.5。途中12周も全条件を別途報告し、良い重みを事後選択しない。',
 '## 学習入力と細線の露出',
 '元の24ページ群・96パッチ・864入力を変更せず使用した。無損傷96、退色288、暗い乗算汚れ288、混合損傷192。教師は損傷前原本由来の暫定v5で、原本で確認した可視墨芯を含むが専門家GTではない。',
 '既存レビュー7パッチ455点は全点が有効な墨教師で、3強度×4周の退色切り出し84枚にも全点が入った。合計5460回の点露出を事前検証した。全強度で輝度差1以上、強度.80では全455点で差10以上。弱損傷の一部は差4〜9であり、その事実を記録してRGBは変えなかった。',
 '[細線露出の事前監査](fine_exposure_audit.json)、[汚れと墨の同時切り出し監査](contact_crop_audit.json)、[弱損傷の監査メモ](prefreeze_audit_note.json)、[原本・損傷・教師PNG一覧](training_exposure_figures/index.json)。',
 '## 比較条件',
 '全4条件で同じG重み・G Adamの2160更新時点を復元。GAN条件ではD重みとD Adamも復元した。G lr3e-5、D lr1e-6、GAN係数.1、clip1、FP32を維持。従来のsigmoid＋scalar-mean hingeのDと、BCEのG損失をそのまま使う。BCE改変負例案や新しい損失は追加していない。',
 '切り出しは元のepoch/index式をepoch10〜13で継続。バッチ順は新しく事前固定したepoch別順列で、同じ順序番号のGAN有無では完全に共通。元のDataLoader状態を保存していなかったため、中断前の正確なシャッフル列の続きとはしない。2つの順序番号は共通の学習済み初期重みに対するバッチ順感度の確認であり、独立したモデル初期化seedによる再現実験ではない。',
 '今回のfull-data学習は元のDataLoader同様の連続NCHWバッチを使う。前回の固定24切り出し試験とは、入力構成・切り出し範囲・バッチレイアウト・D状態の扱いが異なる。その前後差を細線露出だけの一因子効果とは解釈しない。今回の同じ順序番号内でGANの有無を比較する。',
 '公開実装のGA(y,y)による注意相殺と局所GGCアダプタは前回と同じで、論文本来の領域適応注意の効果を示す比較ではない。学習なしのHoweとの課題別比較は維持するが、これは公開コード由来の試験実装でありRA-GAN論文完全再現ではない。',
 '## 14周時点の主評価',
 '墨芯は出力黒画素から2画素以内の被覆で多いほど良い。紙の誤黒化・選択部の黒画素は少ないほど良い。ただし合成の墨芯と紙は原本から算出した代理、実画像の墨芯は部分的な暫定注釈である。D04/T07/C01の固定領域は厳密な物質分類や隠れた文字の真値を保証しない。',
 table(['条件','退色細線','D01細線','D03上側','合成汚れ下の墨','合成汚れ内の紙を黒化','D04黒/5100','T07黒/4500','C01黒/10800'],rows),
 'Howe固定の既知基準：退色42/49、D01 181/225、D03上側89/124、合成接触汚れの墨412/596・紙誤黒化0/18105、D04黒0、T07黒0、C01黒7035。各画像の原本付きPNGに固定Howeを並べた。',
 '## 同じバッチ順内でのGAN追加の差',
 'GANあり−なし。墨芯の正差は増加、誤黒化の正差は悪化。点数の小差だけで優位性や有意差を主張しない。',
 table(['順序','退色細線差','D01差','D03上側差','合成汚れ紙の誤黒化差','D04黒の差'],paired),
 '## 12周から14周の変化',
 table(['条件','退色細線 12周→14周','D01 12周→14周','合成汚れ紙の黒 12周→14周'],[[labels[c],f'{thin(b,c)} → {thin(a,c)}',f"{metric(b,c,'D01','thin','ink_r2')} → {metric(a,c,'D01','thin','ink_r2')}",f"{totals(b,c,'contact_stain','paper_ink')} → {totals(a,c,'contact_stain','paper_ink')}"] for c in CONDITIONS]),
 '## 学習側の確認済み細線',
 'ここは学習への当てはまりであり、汎化性能として数えない。455点は強退色.80と無損傷をnative512で評価した。',
 table(['条件','無損傷 r0 / r2','学習退色.80 r0 / r2'],[[labels[c]]+[f"{sum(r['r0'] for r in train['rows'] if r['condition']==c and r['family']==f)}/455 / {sum(r['r2'] for r in train['rows'] if r['condition']==c and r['family']==f)}/455" for f in ('identity','targeted_fade')] for c in conditions]),
 '## その他の保全確認',
 table(['条件','無損傷の墨芯r2/5500','退色.70の墨芯r2/5500'],[[labels[c],totals(a,c,'clean','core_r2'),totals(a,c,'uniform_fade_070','core_r2')] for c in conditions]),
 '固定256画素の暫定F1は補助診断にのみ使った。注目する学習外細線49点のうち、従来評価窓で有効画素になるのは17点であり、このF1だけでは文字保持を判断できない。',
 table(['条件','追加更新','補助 暫定F1','汚れ内墨保持/9677','汚れ内紙誤黒化/189765'],[[labels[c],r['step'],f"{r['validation']['total']['ink_f1_to_provisional']:.6f}",r['validation']['total']['overlap_ink_kept'],r['validation']['total']['stain_paper_ink']] for c,h in histories.items() for r in h]),
 '## 比較PNG',
 '\n'.join(f'- [{Path(f).name}]({f})' for f in a['figures']),
 '学習側PNG：\n'+'\n'.join(f'- [{Path(f).name}]({f})' for f in train['figures']),
 '## 評価の独立性と再現情報',
 '原本の学習/検証ページ群は非重複で、評価49点・D01/D03の注釈は学習に入れなかった。ただし評価例は繰り返し使用している既知の開発例であり、D01はDKDS Testing-Rと同一ページでもある。別本T07/C01も未見の盲検テストとは呼ばない。専門家による密な正解、独立した別本の保留評価、独立初期化を含む複数試行は未実施で、正式精度や一般化の優位は主張しない。',
 '[事前学習プロトコル](protocol.json)、[事前評価プロトコル](evaluation_protocol.json)、[全数値・14周](evaluation/results.json)、[全数値・12周](evaluation_00432/results.json)、[学習側細線結果](evaluation/training_reviewed_results.json)、[検証結果](integrity_verification.json)。',
 ]
 (OUT/'96パッチでの追加学習とGAN対照.md').write_text('\n\n'.join(text)+'\n')
 # Integrity: all files, labels, common start and planned update counts.
 assert sha(diag.CHECKPOINT)==p['checkpoint_sha256']
 assert sha(PARENT/'train_manifest.jsonl')==p['train_manifest_sha256']
 files=json.loads((OUT/'source_hashes.json').read_text())
 for path,digest in files.items():assert sha(ROOT/path)==digest
 baselines=[h[0]['validation']['total'] for h in histories.values()];assert all(r==baselines[0] for r in baselines)
 checkpoints=[]
 for c in CONDITIONS:
  completion=json.loads((OUT/c/'completion.json').read_text());assert completion['additional_steps']==864 and completion['final_optimizer_steps']==[3024]
  for step in p['saved_steps']:
   path=OUT/c/f'step_{step:05d}.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
   assert payload['additional_steps']==step and payload['step']==2160+step
   assert {int(s['step']) for s in payload['optimizer']['state'].values() if 'step' in s}=={2160+step}
   assert all(g['lr']==3e-5 for g in payload['optimizer']['param_groups'])
   if CONDITIONS[c]['gan']:
    assert {int(s['step']) for s in payload['d_optimizer']['state'].values() if 'step' in s}=={2160+step}
    assert all(g['lr']==1e-6 for g in payload['d_optimizer']['param_groups'])
   else:assert payload['discriminator'] is None and payload['d_optimizer'] is None
   checkpoints.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
 for n in (0,1):assert sha(OUT/'batch_schedules'/f'no_gan_order{n}.json')==sha(OUT/'batch_schedules'/f'reference_gan_order{n}.json')
 exposure=json.loads((OUT/'fine_exposure_audit.json').read_text());assert exposure['repeated_point_exposures']==5460
 for row in exposure['rows']:assert row['in_crop']==row['n']==row['valid_ink']==row['changed_ge1']
 for row in exposure['rows']:
  if float(row['strength'])==.8:assert row['changed_ge10']==row['n']
 allfigures=[]
 for directory in ('evaluation','evaluation_00432'):
  for filename in ('results.json','training_reviewed_results.json'):
   allfigures+=json.loads((OUT/directory/filename).read_text())['figures']
 for f in allfigures:assert (OUT/f).exists()
 for filename in ('run_ragan_full_continuation.py','evaluate_ragan_full_continuation.py','evaluate_ragan_joint_continuation.py','ragan_pilot_support.py'):
  assert sha(ROOT/'scripts/benchmark'/filename)==sha(OUT/'source_snapshot'/filename)
 shutil.copy2(__file__,OUT/'source_snapshot'/Path(__file__).name)
 dump(OUT/'integrity_verification.json',dict(source_files_hashed=len(files),checkpoint_unchanged=True,initial_readouts_equal=True,
  shared_G_optimizer_restored_step=2160,final_G_optimizer_step=3024,D_optimizer_preserved=True,paired_batch_order_equal=True,
  two_order_seeds_not_independent_initializations=True,reviewed_trace_unique_points=455,reviewed_point_exposures_per_run=5460,
  planned_checkpoints=checkpoints,figure_hashes={f:sha(OUT/f) for f in allfigures},
  protocol_sha256=sha(OUT/'protocol.json'),evaluation_protocol_sha256=sha(OUT/'evaluation_protocol.json'),formal_accuracy=False))
 print('Verified',len(checkpoints),'checkpoints',len(allfigures),'comparison PNGs;',len(files),'input/label/source hashes')
 print(json.dumps(dict(main_table=rows,paired_GAN_minus_noGAN=paired),ensure_ascii=False))
if __name__=='__main__':main()
