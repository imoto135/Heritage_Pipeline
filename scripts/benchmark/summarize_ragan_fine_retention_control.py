"""Evidence tables and integrity checks for no-GAN fine-retention controls."""
import json,sys,shutil,re
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_fine_retention_control import OUT,CONDITIONS,BASELINES
from evaluate_ragan_fine_retention_control import LABELS,ORDER
from compare_v5_aligned_fade_control import sha,dump
import run_ragan_full_continuation as full
import diagnose_ragan_discriminator as diag

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def total(a,c,f,k):return sum(r[k] for r in a['synthetic'] if r['condition']==c and r['family']==f)
def realmetric(a,c,case,reg,key):return next(r[key] for r in a['real'] if r['condition']==c and r['case']==case and r['region']==reg)
def thin(a,c):return next(r for r in a['traced_fade'] if r['condition']==c)
def fit(rows,c,f,k):return sum(r[k] for r in rows if r['condition']==c and r.get('family')==f)
def native(rows,c,view,k):return sum(r[k] for r in rows if r['condition']==c and r['view']==view)

def main():
 p=json.loads((OUT/'protocol.json').read_text());a=json.loads((OUT/'evaluation/results.json').read_text());b=json.loads((OUT/'evaluation_00432/results.json').read_text())
 train=json.loads((OUT/'evaluation/training_reviewed_results.json').read_text());nat=json.loads((OUT/'evaluation/training_native_results.json').read_text())
 schedule=json.loads((OUT/'sampling_audit.json').read_text());audit=json.loads((OUT/'core_mask_audit.json').read_text())
 mainrows=[]
 for c in ORDER:
  mainrows.append([LABELS[c],f"{native(nat['rows'],c,'actual_batch4','r0')} / {native(nat['rows'],c,'actual_batch4','r2')}",
   f"{fit(train['rows'],c,'targeted_fade','r0')} / {fit(train['rows'],c,'targeted_fade','r2')}",
   f"{thin(a,c)['r0']} / {thin(a,c)['r2']}",realmetric(a,c,'D01','thin','ink_r2'),realmetric(a,c,'D03','contact_upper','ink_r2'),total(a,c,'contact_stain','core_r2'),total(a,c,'contact_stain','paper_ink')])
 text=['# GANなし：細線の損失重みと提示頻度の対照',
 '元の学習RGB・教師・切り出し式を固定し、同じ10周G重みとAdam状態から4周（864更新）追加した。通常学習2条件は前回の検証済み出力を再利用し、墨芯重み8倍／強退色提示20%をそれぞれ2通りのバッチ順で追加した。14周を主評価、12周を途中評価に固定し、閾値は0.5で変更しない。全条件GANなし。',
 '## 変更した一因子',
 '墨芯8倍：既存の原本確認7パッチ・1558画素の有効墨芯のみ、教師ありL1/BCE/墨Dice/墨Jaccardの重みを8にする。その他の有効画素は1、不明は0。どの損傷族でも同じ原本の墨芯に適用する。入力順と切り出しは同じ順序番号の通常学習と完全に同じ。教師の値や有効マスク自体は変更しない。重みを各積に1度だけ掛け、Dice/Jaccardで重みを二乗しない。',
 '提示20%：強退色.80のレビュー7画像の提示を各周7回から173回（173/864≒20%）へ増やす。通常順列の非対象166位置を置換し、その他の位置・教師・切り出し・損失は維持する。更新数を同じにしたため他画像の提示は減る。これは頻度変更に伴う入力分布の変更を含み、同じ全864入力を毎周1度ずつ見る条件ではない。',
 '倍率8と提示20%は以前の432更新試験から継承した探索設定で、今回の結果を見て最適化していない。前回の短期結果と、今回の十分学習済み共通初期状態からの継続を分けて解釈する。',
 '[事前固定プロトコル](protocol.json)、[重みの性質チェック](weighted_loss_checks.json)、[元の墨芯マスク監査](core_mask_audit.json)、[実際の提示数と損傷族内訳](sampling_audit.json)。',
 '## 14周の主評価',
 'r0は墨芯の座標がそのまま黒、r2は出力黒画素から2画素以内。455点は学習側の強退色.80、49点は別ページの既知退色.70であり、直接比から汎化ギャップを算出しない。合成汚れの墨芯・紙は原本由来の代理で、正式GTではない。',
 table(['条件','学習455・256窓 r0/r2','学習455・512 r0/r2','学習外49 r0/r2','D01墨芯/225','D03上側/124','合成汚れ下墨/596','汚れ内紙の黒/18105'],mainrows),
 '## 汚れと無損傷の保全',
 table(['条件','D04黒/5100','T07黒/4500','C01黒/10800','無損傷墨芯/5500','退色2例墨芯/5500','学習無損傷墨芯/455'],[[LABELS[c],*[realmetric(a,c,case,'selected_spot_or_stain','nonink_false') for case in ['D04','T07','C01']],total(a,c,'clean','core_r2'),total(a,c,'uniform_fade_070','core_r2'),fit(train['rows'],c,'identity','r2')] for c in ORDER]),
 'Howe固定の既知基準：学習外退色42/49、D01 181/225、D03上側89/124、合成汚れ下墨412/596・紙誤黒化0/18105、D04黒0、T07黒0、C01黒7035。選択した実汚れ部の黒画素数は厳密な物質分類や隠れた文字の真値を保証しない。',
 '## 途中12周から14周',
 table(['条件','学習退色512 r2 12→14周','学習外49 r2 12→14周','D01 r2 12→14周','合成汚れ紙の黒 12→14周'],[[LABELS[c],f"{fit(json.loads((OUT/'evaluation_00432/training_reviewed_results.json').read_text())['rows'],c,'targeted_fade','r2')} → {fit(train['rows'],c,'targeted_fade','r2')}",f"{thin(b,c)['r2']} → {thin(a,c)['r2']}",f"{realmetric(b,c,'D01','thin','ink_r2')} → {realmetric(a,c,'D01','thin','ink_r2')}",f"{total(b,c,'contact_stain','paper_ink')} → {total(a,c,'contact_stain','paper_ink')}"] for c in ORDER if c!='initial']),
 '## 学習した7パッチ別の保持',
 '強退色.80、実際に最後に提示したepoch13の256画素窓と4枚組で最終重みを推論。提示頻度条件は同じ対象を複数回見るため、各対象の最終出現を事前規則として使った。通常・8倍条件の対象は各周1回である。全455点が窓と有効墨教師に含まれる。',
 table(['パッチ','点数',*[LABELS[c] for c in ORDER]],[[patch,next(r['n'] for r in nat['rows'] if r['patch']==patch),*[next(r['r2'] for r in nat['rows'] if r['patch']==patch and r['condition']==c and r['view']=='actual_batch4') for c in ORDER]] for patch in sorted({r['patch'] for r in nat['rows']})]),
 '## 原本付きPNG',
 '\n'.join(f'- [{Path(f).name}]({f})' for f in a['figures']),
 '学習時の256窓：\n'+'\n'.join(f'- [{Path(f).name}]({f})' for f in nat['figures']),
 '学習側512：\n'+'\n'.join(f'- [{Path(f).name}]({f})' for f in train['figures']),
 '## 制約',
 '2通りは共通の学習済みGに対するバッチ順の感度確認で、独立初期化のモデル反復ではない。公開版RA生成器の注意相殺とGGCのバッチ全体平均は残る。この対照は教師あり学習の診断であり、RA-GAN論文完全再現や領域適応注意の効果を示すものではない。',
 '学習は単一本Makuraの24ページ群で、教師は大半が古典手法合意由来・一部だけ原本確認の暫定版。学習外49点は教師評価の不明画素を含むため、全体F1を保持指標の代わりにしない。D01は既知開発例かつDKDS Testing-R掲載ページと同一で、別本T07/C01もすでに検討に使っている。正式精度、未見別本への汎化優位、文字の意味や字形を学習した証明とはしない。',
 '[主評価の全数値](evaluation/results.json)、[12周の全数値](evaluation_00432/results.json)、[学習側512](evaluation/training_reviewed_results.json)、[学習時256窓](evaluation/training_native_results.json)、[検証結果](integrity_verification.json)。',
 ]
 (OUT/'細線の損失と提示頻度の対照結果.md').write_text('\n\n'.join(text)+'\n')
 assert sha(diag.CHECKPOINT)==p['checkpoint_sha256'] and sha(full.OUT/'protocol.json')==p['parent_protocol_sha256']
 for path,digest in json.loads((OUT/'source_hashes.json').read_text()).items():assert sha(ROOT/path)==digest
 for r in audit:assert sha(ROOT/r['path'])==r['sha256']
 assert all(json.loads((OUT/'weighted_loss_checks.json').read_text()).values())
 hashes=[];histories=[]
 for c in CONDITIONS:
  conf=CONDITIONS[c];completion=json.loads((OUT/c/'completion.json').read_text());assert completion['final_optimizer_steps']==[3024]
  histories.append(json.loads((OUT/c/'history.json').read_text())[0]['validation']['total'])
  for step in p['saved_steps']:
   path=OUT/c/f'step_{step:05d}.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
   assert payload['step']==2160+step and payload['additional_steps']==step and payload['condition']==c
   assert payload['discriminator'] is None and {int(s['step']) for s in payload['optimizer']['state'].values() if 'step' in s}=={2160+step}
   assert all(g['lr']==3e-5 for g in payload['optimizer']['param_groups']);hashes.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
  if conf['factor']=='core8':assert sha(OUT/'batch_schedules'/(c+'.json'))==sha(full.OUT/'batch_schedules'/f"no_gan_order{conf['order']}.json")
 assert all(h==histories[0] for h in histories)
 for r in schedule:
  assert r['target_draws']==(173 if r['condition'].startswith('exposure20') else 7)
  assert r['changed_positions']==(166 if r['condition'].startswith('exposure20') else 0)
 for c in ORDER:
  for view in ['actual_batch4','single_crop1']:
   rows=[r for r in nat['rows'] if r['condition']==c and r['view']==view];assert len(rows)==7 and sum(r['n'] for r in rows)==455
  assert fit(train['rows'],c,'targeted_fade','n')==455
 # Verify cached output provenance against the previous run's integrity record.
 parent_integrity=json.loads((full.OUT/'integrity_verification.json').read_text());parent_hashes={r['path']:r['sha256'] for r in parent_integrity['planned_checkpoints']}
 for directory in ['evaluation','evaluation_00432']:
  current=json.loads((OUT/directory/'results.json').read_text());old=json.loads((full.OUT/directory/'results.json').read_text())
  assert current['baseline_cache_results_sha256']==sha(full.OUT/directory/'results.json')
  for oldc,newc in [('initial','initial'),('no_gan_order0','baseline_order0'),('no_gan_order1','baseline_order1')]:
   for kind in ['real','synthetic','traced_fade']:
    previous=[dict(r,condition=newc) for r in old[kind] if r['condition']==oldc];cached=[r for r in current[kind] if r['condition']==newc];assert previous==cached
   if oldc!='initial':
    step=864 if directory=='evaluation' else 432;expected=parent_hashes[oldc+f'/step_{step:05d}.pt'];assert current['checkpoint_sha256'][newc]==expected
 for filename in ['run_ragan_fine_retention_control.py','evaluate_ragan_fine_retention_control.py','run_ragan_full_continuation.py','ragan_pilot_support.py','evaluate_ragan_joint_continuation.py']:
  assert sha(ROOT/'scripts/benchmark'/filename)==sha(OUT/'source_snapshot'/filename)
 shutil.copy2(__file__,OUT/'source_snapshot'/Path(__file__).name)
 figs=[]
 for directory in ['evaluation','evaluation_00432']:
  for name in ['results.json','training_reviewed_results.json']:
   figs+=json.loads((OUT/directory/name).read_text())['figures']
 figs+=nat['figures']
 for f in figs:assert (OUT/f).exists()
 dump(OUT/'integrity_verification.json',dict(source_hashes_checked=len(json.loads((OUT/'source_hashes.json').read_text())),initial_G_checkpoint_unchanged=True,
  G_optimizer_restored_step=2160,final_step=3024,no_GAN=True,new_checkpoints=hashes,weighted_loss_properties_checked=True,
  positive_core_pixels=1558,core8_schedules_equal_baseline=True,exposure_target_draws_per_epoch=173,baseline_cache_provenance_verified=True,
  native_training_trace_total455=True,initial_readouts_identical=True,protocol_sha256=sha(OUT/'protocol.json'),
  figure_hashes={f:sha(OUT/f) for f in figs},formal_accuracy=False))
 print('Verified',len(hashes),'new checkpoints and',len(figs),'comparison PNGs')
 print(json.dumps(mainrows,ensure_ascii=False))
if __name__=='__main__':main()
