"""Region-loss readout, baseline cache provenance, and source/checkpoint audits."""
import json,sys,shutil
from pathlib import Path
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_local_region_control import OUT,CONDITIONS,PARENT
from ragan_local_region_support import region_masks
import run_ragan_full_continuation as full
from compare_v5_aligned_fade_control import dump,sha
import diagnose_ragan_discriminator as diag
LABELS={'initial':'再開前10周','baseline_order0':'補助項なし 順序1','baseline_order1':'補助項なし 順序2',
 **{f'{kind}_order{n}':label+f' 順序{n+1}' for n in (0,1) for kind,label in [('inkmean','墨芯平均BCE追加'),('inkpaper','墨芯＋近傍紙BCE追加')]}}

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 a=json.loads((OUT/'evaluation/results.json').read_text());b=json.loads((OUT/'evaluation_00432/results.json').read_text());tr=json.loads((OUT/'evaluation/training_reviewed_results.json').read_text());nat=json.loads((OUT/'evaluation/training_native_results.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
 def rm(data,c,case,reg,key):return next(r[key] for r in data['real'] if r['condition']==c and r['case']==case and r['region']==reg)
 def sm(data,c,f,key):return sum(r[key] for r in data['synthetic'] if r['condition']==c and r['family']==f)
 def thin(data,c,key='r2'):return next(r[key] for r in data['traced_fade'] if r['condition']==c)
 def fit(rows,c,f,key):return sum(r[key] for r in rows if r['condition']==c and r.get('family')==f)
 def ns(c,view='actual_batch4',key='r2'):return sum(r[key] for r in nat['rows'] if r['condition']==c and r['view']==view)
 def hard(c):return next(r['r2'] for r in nat['rows'] if r['condition']==c and r['view']=='actual_batch4' and r['patch']=='200017197_00214_1_p00')
 rows=[]
 for c,label in LABELS.items():rows.append([label,hard(c),ns(c),f"{thin(a,c,'r0')} / {thin(a,c)}",rm(a,c,'D01','thin','ink_r2'),sm(a,c,'contact_stain','core_r2'),sm(a,c,'contact_stain','paper_ink'),rm(a,c,'D04','selected_spot_or_stain','nonink_false')])
 components=[]
 for c in CONDITIONS:
  xs=json.loads((OUT/c/'loss_components.json').read_text());active=[x for x in xs if x['active_regions']]
  components.append([LABELS[c],len(active),sum(x['active_regions'] for x in xs),*[round(sum(x[k] for x in active)/len(active),5) for k in ['global_loss','normalized_ink_BCE','normalized_paper_BCE']]])
 text=['# 残存細線44点：局所平均BCEによる対照',
 '修正BCE＋強退色20%提示を基準とし、RGB・教師・切り出し・提示順・共通10周G/Adamを固定して4周追加した。GANなし。各対象の墨芯BCEを領域内で平均して補助項にする条件と、それに同じ係数で近傍明紙BCEを足す条件を2つのバッチ順で比較した。14周の最終重みを主評価に固定し、12周も全条件を報告する。',
 '## 事前監査',
 '残存44点と幅を含む124点は有効墨教師で、切り出しから漏れていない。最後の2つの対象バッチでは修正BCEの墨芯勾配は非ゼロで、墨芯と残りの損失の勾配余弦は+.67/+.15だった。この2バッチに関して、欠落を勾配消失や単純な逆向き勾配だけでは説明できない。Adamや他バッチの影響、有限回の反復をこの監査だけで分離しない。[全監査](audit/所見.md)。',
 '## 補助項の範囲',
 '対象は既存7パッチの強退色.80画像のみ。墨芯は原本確認済み有効ink1558画素で、教師値は変更しない。1対象ごとの平均BCEを取り、複数対象が同じバッチなら対象間でも平均し、係数1で元の教師あり損失に加える。',
 '紙を加える条件では、墨芯bbox+16pxの内側で、元から有効paperであり、かつ損傷前原本の明紙proxyに該当する画素の平均BCEを係数1で加える。紙proxyは背景との差<=5、暗部から8px以上離れる等の既存規則で、専門家GTではない。判別不能な筆画縁を新しい紙教師に変えない。元の全体損失は全条件共通で維持する。',
 '両条件の墨芯係数は同じ1であり、2つ目が変えるのは近傍紙項の追加だけ。各周173/864対象、4周692対象提示を共通にした。各対象で有効紙proxyは最低35画素。学習配分は同じだが、124点の墨芯平均は監査対象バッチで全体BCE中の係数を約2000倍にする強い介入で、以前の8倍重みと同じ実験ではない。',
 '[事前プロトコル](protocol.json)、[領域露出数](region_exposure_audit.json)、[補助マスク](region_mask_audit.json)、[符号と除外画素のチェック](loss_checks.json)。',
 '## 14周の主評価',
 '墨芯は2px以内被覆、学習外49はr0/r2を併記。44点と455点は学習強退色.80で、49点は別ページの.70入力。学習側の記憶と転移を分ける。',
 table(['条件','残存44点','学習455点256窓','学習外49 r0/r2','D01/225','合成汚れ下墨/596','合成汚れ内紙黒/18105','D04黒/5100'],rows),
 '## 回復の範囲と筆画の太り',
 '残存44点の回復は学習と同じ256画素の窓に限られる。1枚だけの256推論でも全4条件が44/44点を覆う一方、512推論では墨芯のみ12/1点、墨芯＋紙5/0点に低下する。512での学習455点総被覆も墨芯のみ423/412、墨芯＋紙416/408で、256窓の455/455、454/455とは異なる。入力範囲・再計算されるOtsu/Sobel・空間集約が一緒に変わるため、単独原因をこの比較から断定しない。',
 '原本付き256比較では、墨芯のみ順序1で回復した細線が局所的に太い黒い塊になる。紙項を足しても線幅の一致は保証できない。2px以内の被覆は線幅・連結・字形の正しさを測る指標ではない。',
 '[同じ窓の256/512比較・順序1](evaluation/figures/残存44点_推論範囲比較_順序1.png)、[順序2](evaluation/figures/残存44点_推論範囲比較_順序2.png)、[キャッシュ出力の再集計](evaluation/context_comparison.json)。',
 '## 無損傷・別本・接触汚れ',
 table(['条件','無損傷墨芯/5500','退色2例墨芯/5500','学習無損傷/455','D03上側/124','T07黒/4500','C01黒/10800'],[[label,sm(a,c,'clean','core_r2'),sm(a,c,'uniform_fade_070','core_r2'),fit(tr['rows'],c,'identity','r2'),rm(a,c,'D03','contact_upper','ink_r2'),rm(a,c,'T07','selected_spot_or_stain','nonink_false'),rm(a,c,'C01','selected_spot_or_stain','nonink_false')] for c,label in LABELS.items()]),
 'Howe固定：学習外42/49、D01 181/225、D03上側89/124、合成接触汚れ墨412/596・紙黒0/18105、D04黒0、T07黒0、C01黒7035。実汚れ部の黒画素数は厳密な材質・隠れた文字の真値を保証しない。',
 '## 12周から14周',
 table(['条件','学習455・512 r2 12→14','学習外49 r2 12→14','合成汚れ紙の黒 12→14'],[[LABELS[c],f"{fit(json.loads((OUT/'evaluation_00432/training_reviewed_results.json').read_text())['rows'],c,'targeted_fade','r2')} → {fit(tr['rows'],c,'targeted_fade','r2')}",f'{thin(b,c)} → {thin(a,c)}',f"{sm(b,c,'contact_stain','paper_ink')} → {sm(a,c,'contact_stain','paper_ink')}"] for c in LABELS if c!='initial']),
 '## 対象を含むバッチの平均損失',
 'activeは補助項を持つ対象を1つ以上含むバッチ。紙項は墨芯のみ条件でも診断値を記録したが、その条件の更新には使っていない。平均値から勾配の寄与は直接算出しない。',
 table(['条件','activeバッチ/864','active対象数','元の損失','墨芯BCE平均','紙BCE平均'],components),
 '## 原本付きPNG', '\n'.join(f'- [{Path(f).name}]({f})' for f in a['figures']),
 '学習256窓：\n'+'\n'.join(f'- [{Path(f).name}]({f})' for f in nat['figures']),
 '## 制約',
 '基準は前回の修正BCE＋20%の出力をそのまま再利用し、条件名だけを内部変換して同じ読み出し器に渡した。baseline_cache_adapterのprovenance.jsonに元のファイルとハッシュを保存した。元の基準出力、教師、RGBは変更していない。',
 '既知開発例・単一本の暫定教師・共通Gに対する2つのバッチ順の感度確認で、独立初期化反復でも正式精度でもない。公開GAの注意相殺は残る。原本の紙proxyを使った補助項は専門家が新規に付けた密な正解ではない。改善点数だけから文字の意味や字形一般を学習したとは結論しない。',
 '[主評価全数値](evaluation/results.json)、[12周全数値](evaluation_00432/results.json)、[学習256窓](evaluation/training_native_results.json)、[全更新の損失成分](inkpaper_order0/loss_components.json)、[検証](integrity_verification.json)。',
 ]
 (OUT/'局所平均BCEの対照結果.md').write_text('\n\n'.join(text)+'\n')
 # Required provenance, masks and optimizer continuity.
 assert sha(diag.CHECKPOINT)==p['checkpoint_sha256'] and sha(PARENT/'protocol.json')==p['parent_local_baseline_protocol_sha256'];checkpoints=[]
 for c,conf in CONDITIONS.items():
  assert sha(OUT/'batch_schedules'/(c+'.json'))==sha(PARENT/'batch_schedules'/f"stable_exposure_order{conf['order']}.json")
  assert json.loads((OUT/c/'completion.json').read_text())['final_optimizer_steps']==[3024]
  xs=json.loads((OUT/c/'loss_components.json').read_text());assert len(xs)==864 and sum(x['active_regions'] for x in xs)==692
  for x in xs:
   expected=x['global_loss']+x['normalized_ink_BCE']+(x['normalized_paper_BCE'] if conf['kind']=='inkpaper' else 0)
   assert abs(expected-x['total'])<=1e-4*max(1,abs(expected))
  for step in [216,432,648,864]:
   path=OUT/c/f'step_{step:05d}.pt';payload=torch.load(path,map_location='cpu',weights_only=False);assert payload['step']==2160+step and payload['discriminator'] is None
   assert {int(s['step']) for s in payload['optimizer']['state'].values() if 'step' in s}=={2160+step};checkpoints.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
 for directory,step in [('evaluation',864),('evaluation_00432',432)]:
  current=json.loads((OUT/directory/'results.json').read_text());original=json.loads((PARENT/directory/'results.json').read_text());provenance=json.loads((OUT/'baseline_cache_adapter'/f'step_{step:05d}/provenance.json').read_text())
  assert provenance['original_results_sha256']==sha(PARENT/directory/'results.json')
  for old,new in [('initial','initial'),('stable_exposure_order0','baseline_order0'),('stable_exposure_order1','baseline_order1')]:
   assert current['checkpoint_sha256'][new]==original['checkpoint_sha256'][old]
   for kind in ['real','synthetic','traced_fade']:assert [dict(r,condition=new) for r in original[kind] if r['condition']==old]==[r for r in current[kind] if r['condition']==new]
 for old,new in [('initial','initial'),('stable_exposure_order0','baseline_order0'),('stable_exposure_order1','baseline_order1')]:
  original=json.loads((PARENT/'evaluation/training_native_results.json').read_text())
  for view in ['actual_batch4','single_crop1']:assert sum(r['r2'] for r in nat['rows'] if r['condition']==new and r['view']==view)==sum(r['r2'] for r in original['rows'] if r['condition']==old and r['view']==view)
 regions=region_masks(full.read_rows('train'))
 for row in json.loads((OUT/'region_mask_audit.json').read_text()):
  for key in ['core','paper','known_paper','roi']:
   path=OUT/'region_masks'/(row['patch']+'_'+key+'.png');assert sha(path)==row['mask_hashes'][key] and np.array_equal(full.gray(path)>0,regions[row['patch']][key])
 for c in LABELS:
  for view in ['actual_batch4','single_crop1']:assert sum(r['n'] for r in nat['rows'] if r['condition']==c and r['view']==view)==455
 for filename in ['run_ragan_local_region_control.py','ragan_local_region_support.py','run_ragan_full_continuation.py','run_ragan_stable_bce_control.py','ragan_pilot_support.py','evaluate_ragan_fine_retention_control.py','evaluate_ragan_local_region_control.py','evaluate_ragan_joint_continuation.py']:
  assert sha(ROOT/'scripts/benchmark'/filename)==sha(OUT/'source_snapshot'/filename)
 for filename in [Path(__file__).name,'evaluate_ridge_gate_independent_val.py']:shutil.copy2(ROOT/'scripts/benchmark'/filename,OUT/'source_snapshot'/filename)
 figs=[]
 for directory in ['evaluation','evaluation_00432']:
  for filename in ['results.json','training_reviewed_results.json']:figs+=json.loads((OUT/directory/filename).read_text())['figures']
 figs+=nat['figures'];dump(OUT/'integrity_verification.json',dict(checkpoints=checkpoints,common_initial_G_and_Adam=True,initial_optimizer_step=2160,final_optimizer_step=3024,
  schedules_equal_stable20percent_baseline=True,source_masks_recomputed_equal_frozen=True,active_target_examples_per_run=692,baseline_cache_verified_against_actual_stable20percent=True,
  loss_components_reconstruct_objective=True,no_label_edits=True,no_RGB_edits=True,protocol_sha256=sha(OUT/'protocol.json'),figure_hashes={f:sha(OUT/f) for f in figs},formal_accuracy=False))
 print('Verified',len(checkpoints),'checkpoints',len(figs),'PNGs');print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
