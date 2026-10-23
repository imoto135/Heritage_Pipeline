"""Exploratory score-order diagnostic after fixed readout; never choose a threshold."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'tmp_work/dark_nontext_teacher_control_20261023'
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import json,csv
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate_dark_nontext_teacher_control as ev
import report_background_diversity_control as refs
from compare_v5_aligned_fade_control import mask,sha,dump

def main():
    ev.check();q=json.loads((OUT/'results.json').read_text());cases={c['id']:c for c in ev.old.cases()};yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    howe_core=sum(int((ndi.binary_dilation(mask(refs.reference_path(cases[k])),structure=disk)&cases[k]['core']).sum()) for k in ['R05','R06']);howe_fp=int((mask(refs.reference_path(cases['D04']))&cases['D04']['paper']).sum());records=[];files={};curves=[]
    fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained');colors=['#1368ac','#d95f02','#42a78a','#993e96']
    for color,(seed,arm) in zip(colors,[(0,'clear'),(0,'stain'),(1,'clear'),(1,'stain')]):
        name=f'ragan_seed{seed}_{arm}';fg=[]
        for key in ['R05','R06','D04']:
            f=OUT/'evaluation'/name/'probabilities/full512'/f'{key}.npy';p=np.load(f);files[str(f.relative_to(ROOT))]=sha(f)
            if key!='D04':fg.extend(ndi.minimum_filter(p,footprint=disk,mode='constant',cval=np.inf)[cases[key]['core']].astype(np.float64).tolist())
            else:bg=p[cases[key]['paper']].astype(np.float64)
        fg=np.sort(np.asarray(fg,np.float64));bg=np.sort(bg);assert len(fg)==1158 and len(bg)==5100
        events=np.unique(np.r_[0.,.5,1.,np.nextafter(np.unique(np.r_[fg,bg]),np.inf)]);events=events[(events>=0)&(events<=1)]
        covers=np.searchsorted(fg,events,side='left');fps=np.searchsorted(bg,events,side='left');actual_fg=int(np.searchsorted(fg,.5,side='left'));actual_fp=int(np.searchsorted(bg,.5,side='left'))
        primary=[r for r in q['rows'] if r['condition']==name and r['grid']=='full512'];assert actual_fg==sum(r['core_r2'] for r in primary if r['case'] in ['R05','R06']) and actual_fp==next(r['paper_ink'] for r in primary if r['case']=='D04')
        boundary=float(fg[howe_core-1]);necessary_fp=int(np.searchsorted(bg,boundary,side='right'));reachable=boundary<1
        rec=dict(condition=name,seed=seed,arm=arm,fixed_threshold=.5,faint_core_r2_at_fixed=actual_fg,D04_paper_black_at_fixed=actual_fp,howe_faint_target=howe_core,target_reachable_by_global_threshold=reachable,necessary_D04_black_at_target=necessary_fp if reachable else None,score_boundary_for_target=boundary)
        records.append(rec);curves.append(dict(condition=name,thresholds=events.tolist(),core_r2=covers.tolist(),D04_black=fps.tolist()))
        ax.plot(fps,covers,color=color,label=f'{arm}, init {seed+1}',linewidth=1.7);ax.scatter([actual_fp],[actual_fg],color=color,edgecolors='white',s=50,zorder=4)
    ax.scatter([howe_fp],[howe_core],marker='*',s=180,color='black',label='Howe fixed',zorder=5);ax.axhline(howe_core,color='grey',linewidth=.8,linestyle='--');ax.set(xlabel='D04 selected nontext black pixels / 5100 (lower is better)',ylabel='R05+R06 radius-2 ink-core coverage / 1158 (higher is better)',xlim=(-80,5180),ylim=(-20,1180),title='Global threshold score ordering; dots mark fixed threshold 0.5');ax.grid(alpha=.2);ax.legend(loc='lower right');fig.savefig(OUT/'閾値の分離限界.png',dpi=180);plt.close(fig)
    dump(OUT/'score_separation_diagnostic.json',dict(status='Exploratory post-readout score ordering, not preregistered performance or threshold tuning. No model/label/mask changes. Fixed512 only.',rows=records,howe_faint_core=howe_core,howe_D04_paper_black=howe_fp,source_probability_files=files,results_sha256=sha(OUT/'results.json'),script_sha256=sha(__file__),limits='Partial labels only, no denseGT/F1. Bounds concern the current scalar paper scores under ONE global threshold across these images, not all algorithms or per-image adaptive methods.',curves=curves))
    table=refs.table(['条件','固定.5薄字/1158','固定.5 D04黒/5100',f'薄字{howe_core}点を残す際のD04黒の下限'],[[r['condition'],r['faint_core_r2_at_fixed'],r['D04_paper_black_at_fixed'],r['necessary_D04_black_at_target']] for r in records])
    (OUT/'補助診断_閾値の分離.md').write_text('''# 固定出力のスコア順序を調べる補助診断

主評価後の探索的診断であり、閾値の採用・学習・注釈の変更は行っていない。保存した512の紙確率を使い、全画像に同じ閾値を適用した場合の到達範囲を確認した。

半径2の墨芯被覆は、固定された円盤内の最小紙確率が閾値未満なら被覆される。この最小値1158点とD04非文字の紙確率5100点を並べ、Howeと同じ薄字被覆を得る際に必ず残るD04の黒化を計数した。同じスコア値の同順位も含める。未知画素はGTへ変更していない。

'''+table+'''

[曲線PNG](閾値の分離限界.png)。点は主評価の閾値.5、星は固定Howe。曲線は同じ閾値を両画像群へ適用した場合の部分指標であり、正解の線幅・F1・認識精度ではない。

この診断は現在の確率出力の順位の限界を示すもので、学習済みモデル一般や画像ごとに閾値を変える手法全体の限界ではない。閾値を評価例ごとに選んで改善したとは扱わない。
''')
    print('Score ordering diagnostic:',records,flush=True)
if __name__=='__main__':main()
