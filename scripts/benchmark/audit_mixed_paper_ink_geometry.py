"""Post-readout geometry diagnostic, never new GT or a parameter search."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import report_mixed_paper_ink_control as report
import evaluate_mixed_paper_ink_control as ev
from compare_v5_aligned_fade_control import dump,sha,mask
OUT=report.OUT
def main():
    ev.check();results=json.loads((OUT/'results.json').read_text());cases={c['id']:c for c in ev.old.cases() if c['group']=='reviewed_real' and not c.get('secondary',False)}
    plan=OUT/'geometry_diagnostic_plan.json'
    if not plan.exists():
        dump(plan,dict(utc=datetime.now(timezone.utc).isoformat(),status='Exploratory geometry audit after visual inspection of outputs; not preregistered accuracy or GT',
          cases=sorted(cases),all_kinds_seeds_arms_grids=True,mask_results_sha256=sha(OUT/'results.json'),
          quantities='Foreground EDT at fixed source-traced native points; median radius over allpoints incl0; median over foreground-covered points with count; fixed source-path radius8 corridor black occupancy and global black area',
          interpretation='Geometric inflation/merging warning, not true stroke width or boundary accuracy. Corridor pixels are unknown, never relabeled as nonink. Same source paths for all outputs; Howe is reference, not widthGT.',
          no_training=True,no_threshold_changes=True,no_annotation_changes=True))
    p=json.loads(plan.read_text());assert p['mask_results_sha256']==sha(OUT/'results.json')
    output=[];pathmasks={}
    for key,c in cases.items():
        for path in c['paths']:
            trace=np.zeros(c['core'].shape,bool);xy=path['xy'];trace[xy[:,1],xy[:,0]]=True;corridor=ndi.distance_transform_edt(~trace)<=8
            pathmasks[(key,path['id'])]=(trace,corridor)
    candidates=[]
    for r in results['rows']:
        if r['case'] not in cases:continue
        c=cases[r['case']];target=report.path_of(c,r['kind'],r['seed'],r['arm'],r['grid']);candidates.append((r['condition'],r['case'],r['grid'],target))
    for key,c in cases.items():candidates.append(('howe_fixed',key,'native_reference',report.previous.reference_path(c)))
    files={}
    for condition,key,grid,target in candidates:
        pred=mask(target);inside=ndi.distance_transform_edt(pred);files[str(target.relative_to(report.ROOT))]=sha(target)
        for path in cases[key]['paths']:
            trace,corridor=pathmasks[(key,path['id'])];v=inside[trace];covered=v>0
            output.append(dict(condition=condition,case=key,grid=grid,path=path['id'],trace_n=int(trace.sum()),trace_exact=int(covered.sum()),
              radius_median_all=float(np.median(v)),radius_median_on_covered=float(np.median(v[covered])) if covered.any() else None,radius_p90_all=float(np.percentile(v,90)),
              corridor8_n=int(corridor.sum()),corridor8_black=int((pred&corridor).sum()),roi_black=int(pred.sum()),mask_sha256=sha(target)))
    focus=[r for r in output if r['case']=='R06' and r['path']=='center_weak_vertical' and (r['grid']=='full512' and r['condition'].startswith('ragan') or r['condition']=='howe_fixed')]
    lines=['# 筆画の太りを確認する補助診断','', '出力の原寸確認後に計画した探索的診断。墨芯被覆と遠い紙の注釈だけでは、筆画周囲の過大黒化を見逃すため、全6領域・全条件・全格子について同じ原本由来経路で計測した。重み・閾値・注釈は変えていない。','',
      '「内部距離」は墨芯点から予測の白までのEuclidean距離。全点の中央値では白の点を0とし、黒に覆われた点数も併記する。太りの補助量であり、真の線幅ではない。原画像の経路から8画素以内の帯の黒化も数えるが、この帯は不明画素のままで、非文字GTへ変換しない。','',
      '## R06の細い縦線：公開Gの512主評価','',report.previous.table(['条件','墨芯厳密/経路点','内部距離中央値px','半径8帯の黒画素/帯画素','ROI全黒画素'],[[r['condition'],f"{r['trace_exact']}/{r['trace_n']}",f"{r['radius_median_all']:.2f}",f"{r['corridor8_black']}/{r['corridor8_n']}",r['roi_black']] for r in focus]),'',
      'Howeの線幅も正解GTではない。数値の増加だけで過太りを断定せず、原寸PNGと合わせる。今回の紙＋墨芯条件には、細い筆画が太い帯になり、近接した文字の塊がつながる例がある。中心被覆の回復を、字形復元の成功と同一視しない。','',
      '[R06原画像付き比較](figures/R06_ragan_seed0_full512.png)、[R05](figures/R05_ragan_seed0_full512.png)、[R08](figures/R08_ragan_seed0_full512.png)。','']
    (OUT/'補助診断_筆画の太り.md').write_text('\n'.join(lines))
    dump(OUT/'geometry_diagnostic.json',dict(rows=output,focus=focus,source_geometry='Unchanged original partial-label paths',formal_accuracy=False))
    dump(OUT/'geometry_verification.json',dict(plan_sha256=sha(plan),results_sha256=sha(OUT/'results.json'),prediction_files=files,rows=len(output),all_source_paths_shared=True,no_relabeling_of_unknown_pixels=True))
    print('Audited',len(output),'path geometries across',len(candidates),'masks',flush=True)
if __name__=='__main__':main()
