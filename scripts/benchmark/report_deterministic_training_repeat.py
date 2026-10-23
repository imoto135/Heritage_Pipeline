"""Verify identical learned state/readouts; compare with prior context without selecting."""
import json,csv
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import evaluate_deterministic_training_repeat as ev
import run_deterministic_training_repeat as tr
import report_background_diversity_control as refs
from compare_v5_aligned_fade_control import dump,sha,mask
from apply_ragan_real_regions import panel
ROOT=tr.ROOT;OUT=tr.OUT
def main():
    ev.check();tv=json.loads((OUT/'full_training_repeat_verification.json').read_text());assert tv['model_tensors_equal'] and tv['optimizer_tensors_equal'] and tv['history_equal']
    cases=ev.old.cases();by={c['id']:c for c in cases};rows=[];fits=[];verified=[];checks=[];figures=[];yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    def verify(r,path,core,paper):
        a=mask(path);v=dict(core_n=int(core.sum()),paper_n=int(paper.sum()),core_exact=int((a&core).sum()),core_r2=int((ndi.binary_dilation(a,structure=disk)&core).sum()),paper_ink=int((a&paper).sum()),total_ink=int(a.sum()));assert all(r[k]==v[k] for k in v);assert sha(path)==r['mask_sha256'];verified.append(str(path.relative_to(ROOT)))
    for arm in tr.ARMS:
        name='ragan_seed1_'+arm;q=json.loads((OUT/'evaluation'/name/'results.json').read_text());assert q['evaluation_plan_sha256']==sha(OUT/'evaluation_plan.json');rows+=q['rows'];fits+=q['teacher_fit']
        for r in q['rows']:
            c=by[r['case']];verify(r,OUT/'evaluation'/name/'masks'/r['grid']/f"{c['id']}.png",c['core'],c['paper'])
        for r in q['teacher_fit']:
            lab=np.asarray(Image.open(tr.OLD/'labels'/f"{r['case']}.png"));verify(r,OUT/'evaluation'/name/'new_teacher_fit'/f"{r['case']}.png",lab==1,lab==2)
    for c in cases:
      for grid in ev.GRIDS:
        paths=[OUT/'evaluation'/f'ragan_seed1_{arm}'/'masks'/grid/f"{c['id']}.png" for arm in tr.ARMS];assert sha(paths[0])==sha(paths[1]);checks.append(dict(case=c['id'],grid=grid,bitwise_equal=True,sha256=sha(paths[0])))
    for r in [r for r in fits if r['condition'].endswith('repeat0')]:
        paths=[OUT/'evaluation'/f'ragan_seed1_{arm}'/'new_teacher_fit'/f"{r['case']}.png" for arm in tr.ARMS];assert sha(paths[0])==sha(paths[1])
    def get(key,arm='repeat0',grid='full512'):return next(r for r in rows if r['case']==key and r['condition']=='ragan_seed1_'+arm and r['grid']==grid)
    focus=['R05','R06','D04','heldout49','200017197_00103_1_p00','200017197_00214_1_p00']
    for key in focus:
        c=by[key];x,y,w,h=c['roi'];ref=refs.reference_path(c);items=[('同じ入力原画像',c['page'][y:y+h,x:x+w]),('損傷前原画像' if c['group']=='train' else 'Howe固定',c['source'] if c['group']=='train' else np.asarray(Image.open(ref)))]
        for label,root,name in [('通常演算：近傍',tr.PARENT,'ragan_seed1_near'),('通常演算：遠方',tr.PARENT,'ragan_seed1_far'),('厳密再現：1回目',OUT,'ragan_seed1_repeat0'),('厳密再現：2回目',OUT,'ragan_seed1_repeat1')]:
            r=get(key,name.rsplit('_',1)[-1]) if root==OUT else next(r for r in json.loads((tr.PARENT/'evaluation'/name/'results.json').read_text())['rows'] if r['case']==key and r['grid']=='full512')
            items.append((label+f" 墨{r['core_r2']}/{r['core_n']} 紙{r['paper_ink']}",np.asarray(Image.open(root/'evaluation'/name/'masks/full512'/f'{key}.png'))))
        target=OUT/'figures'/f'{key}_再現性と文字保持.png';panel(target,key+' / 同じ開始重みを反復・独立な初期化2種ではない',items);figures.append(str(target.relative_to(OUT)))
    summary=[]
    for grid in ev.GRIDS:
        rs=[r for r in rows if r['condition']=='ragan_seed1_repeat0' and r['grid']==grid];faint=[r for r in rs if r['case'] in ['R05','R06']];summary.append([grid,sum(r['core_r2'] for r in rs if r['group']=='train'),get('heldout49',grid=grid)['core_r2'],sum(r['core_r2'] for r in faint),sum(r['paper_ink'] for r in faint),get('R03',grid=grid)['paper_ink'],get('R08',grid=grid)['paper_ink'],get('D04',grid=grid)['paper_ink']])
    lines=['# 厳密な演算設定で同じ864更新を2回繰り返す','',
      '**最終モデル・Adam状態・864更新の損失履歴が一致し、全26領域×3推論条件の出力78組と学習8窓の出力も一致した。** 同じ初期化・同じ近傍教師を2回実行した再現性検査であり、独立な初期化2種の精度実験ではない。','',
      '公開由来G・初期化2の14周モデルとAdamから864更新。CUBLAS_WORKSPACE_CONFIG=:4096:8、厳密な決定的演算を有効にし、cuDNN benchmarkとTF32を無効にした。CUDAで決定的な逆伝播を利用できなかった反射パディングを、スライス・反転・連結で同じ前向き値を作る実装に置き換えた。8か所のConv2dを対象とし、パラメータ構成は維持した。','',
      '置き換え前後の公開G全体の前向き出力は、原画像256/512の両方でビット一致を確認した。CPU倍精度のパディング勾配も照合した。演算精度・加算順が以前と異なるため、以前の精度との小差を単一のアルゴリズム改善とは解釈しない。','',
      refs.table(['格子','旧学習墨芯455','退色49','薄字1158','薄字紙黒/6857','R03紙黒','R08紙黒','D04紙黒'],summary),'',
      '墨芯被覆は半径2であり、字形・認識精度ではない。今回の再現性一致は、この固定条件と同じ構成のGPUで実測した範囲の結果で、任意環境・モデル性能・GAN効果を保証しない。','',
      '[訓練状態の照合](full_training_repeat_verification.json)、[推論・独立再採点の照合](verification.json)、[凍結条件](protocol.json)、[評価計画](evaluation_plan.json)。','',
      '\n'.join(f'- [{f}]({f})' for f in figures),'']
    (OUT/'再現性の検証結果.md').write_text('\n'.join(lines));assert len(rows)==156 and len(fits)==16 and len(checks)==78 and len(verified)==172
    dump(OUT/'results.json',dict(rows=rows,teacher_fit=fits,summary=summary,figures=figures,independent_seeds=False,formal_accuracy=False))
    dump(OUT/'verification.json',dict(independent_rescores=verified,matched_eval_pairs=checks,matched_teacher_pairs=8,figures={f:sha(OUT/f) for f in figures},model_optimizer_history_equal=True))
    shutil_path=OUT/'source_snapshot'/Path(__file__).name;shutil_path.write_text(Path(__file__).read_text())
    print('Verified2 identical864-update runs;78 eval mask pairs+8 fit pairs exact;172 rescored masks',flush=True)
if __name__=='__main__':main()
