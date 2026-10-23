"""Paired source-near-paper control: partial labels and geometry together."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import torch
import evaluate_near_paper_teacher_control as ev
import run_near_paper_teacher_control as tr
import report_background_diversity_control as refs
from report_mixed_paper_ink_control import state_sha
from apply_ragan_real_regions import panel
from compare_v5_aligned_fade_control import dump,sha,rgb,mask
ROOT=tr.ROOT;OUT=tr.OUT
LABEL={'without_near':'周囲紙なし','with_near':'周囲紙あり'}
def path_of(c,kind,seed,arm,grid):return OUT/'evaluation'/f'{kind}_seed{seed}_{arm}'/'masks'/grid/f"{c['id']}.png"
def main():
    p=ev.check();protocol=tr.check();gp=json.loads((OUT/'geometry_plan.json').read_text());assert gp['report_script_sha256']==sha(Path(__file__));cases=ev.old.cases();by={c['id']:c for c in cases};cs=json.loads((tr.OLD/'selected_sources.json').read_text())['cases'];rows=[];fits=[];verified=[];paired=[];figures=[];geometry=[]
    yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    def verify(r,target,core,paper):
        pred=mask(target);assert pred.shape==core.shape and sha(target)==r['mask_sha256']
        vals=dict(core_n=int(core.sum()),paper_n=int(paper.sum()),core_exact=int((pred&core).sum()),core_r2=int((ndi.binary_dilation(pred,structure=disk)&core).sum()),paper_ink=int((pred&paper).sum()),total_ink=int(pred.sum()))
        for k,v in vals.items():assert r[k]==v,(r['condition'],r['case'],k,r[k],v)
        verified.append(dict(path=str(target.relative_to(ROOT)),sha256=sha(target)))
        return pred
    def geom(condition,c,grid,pred):
        inside=ndi.distance_transform_edt(pred)
        for path in c['paths']:
            trace=np.zeros(pred.shape,bool);xy=path['xy'];trace[xy[:,1],xy[:,0]]=True;corr=ndi.distance_transform_edt(~trace)<=8;v=inside[trace]
            geometry.append(dict(condition=condition,case=c['id'],grid=grid,path=path['id'],trace_n=int(trace.sum()),trace_exact=int((v>0).sum()),radius_median_all=float(np.median(v)),radius_p90_all=float(np.percentile(v,90)),corridor8_black=int((pred&corr).sum()),corridor8_n=int(corr.sum()),roi_black=int(pred.sum())))
    for kind in ['unet','ragan']:
      for seed in [0,1]:
        first=[];start=torch.load(tr.base.old.OUT/f'{kind}_seed{seed}'/'step_03024.pt',map_location='cpu',weights_only=False);start_sha=state_sha(start['model']);del start
        for arm in tr.ARMS:
            name=f'{kind}_seed{seed}_{arm}';dest=OUT/'evaluation'/name;q=json.loads((dest/'results.json').read_text());assert q['evaluation_plan_sha256']==sha(OUT/'evaluation_plan.json');rows+=q['rows'];fits+=q['teacher_fit']
            completion=json.loads((OUT/'runs'/name/'completion.json').read_text());history=json.loads((OUT/'runs'/name/'loss_components.json').read_text());assert len(history)==864 and sum(h['active'] for h in history)==692
            assert completion['initial_state_sha256']==start_sha and completion['initial_optimizer_steps']==[3024] and completion['final_optimizer_steps']==[3888]
            assert completion['mixed_ink_exposures']==125172 and completion['mixed_paper_exposures']==1590624 and completion['near_paper_exposures']==(574020 if arm=='with_near' else 0)
            first.append({k:history[0][k] for k in ['main','background','mixed_paper','mixed_ink_unweighted','near_paper_unweighted']})
            assert all(np.isfinite(h[k]) for h in history for k in ['main','background','mixed_paper','mixed_ink_unweighted','near_paper_unweighted','gradient_norm'])
            cp=OUT/'runs'/name/'step_00864.pt';assert q['checkpoint_sha256']==sha(cp);saved=torch.load(cp,map_location='cpu',weights_only=False)
            assert state_sha(saved['model'])==q['unchanged_state_sha256'] and saved['step']==3888 and saved['additional_steps']==864 and saved['protocol_sha256']==sha(OUT/'protocol.json') and saved['discriminator'] is None
            assert {int(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s}=={3888};del saved
        assert first[0]==first[1],(kind,seed,first);paired.append(dict(kind=kind,seed=seed,identical_first_unweighted_losses=first[0]))
    for r in rows:
        c=by[r['case']];pred=verify(r,path_of(c,r['kind'],r['seed'],r['arm'],r['grid']),c['core'],c['paper'])
        if c['group']=='reviewed_real' and not c.get('secondary',False):geom(r['condition'],c,r['grid'],pred)
    for r in fits:
        lab=np.asarray(Image.open(tr.OLD/'labels'/f"{r['case']}.png"));target=OUT/'evaluation'/r['condition']/'new_teacher_fit'/f"{r['case']}.png";pred=verify(r,target,lab==1,lab==2);near=np.asarray(Image.open(OUT/'near_paper'/f"{r['case']}.png"))>0;assert r['near_paper_n']==int(near.sum()) and r['near_paper_ink']==int((pred&near).sum())
    for c in cases:
        if c['group']=='reviewed_real' and not c.get('secondary',False):geom('howe_fixed',c,'native_reference',mask(refs.reference_path(c)))
    def get(kind,seed,arm,key,grid='full512'):return next(r for r in rows if (r['kind'],r['seed'],r['arm'],r['case'],r['grid'])==(kind,seed,arm,key,grid))
    summary=[];fit_table=[];sensitivity=[]
    for kind in ['unet','ragan']:
      for seed in [0,1]:
       for arm in tr.ARMS:
        name=f'{kind}_seed{seed}_{arm}';rs=[r for r in rows if r['condition']==name and r['grid']=='full512'];faint=[r for r in rs if r['case'] in ['R05','R06']]
        summary.append([name,sum(r['core_r2'] for r in rs if r['group']=='train'),get(kind,seed,arm,'heldout49')['core_r2'],sum(r['core_r2'] for r in faint),sum(r['paper_ink'] for r in faint),get(kind,seed,arm,'R03')['paper_ink'],get(kind,seed,arm,'R08')['paper_ink'],get(kind,seed,arm,'D04')['paper_ink']])
        fs=[r for r in fits if r['condition']==name];fit_table.append([name,sum(r['core_exact'] for r in fs),sum(r['core_r2'] for r in fs),sum(r['paper_ink'] for r in fs),sum(r['near_paper_ink'] for r in fs)])
        for grid in ev.GRIDS:
            rr=[get(kind,seed,arm,c,grid) for c in ['R05','R06']];sensitivity.append([name,grid,sum(r['core_r2'] for r in rr),sum(r['paper_ink'] for r in rr),get(kind,seed,arm,'R08',grid)['paper_ink']])
    # Native image panels, same scales and placement, no resize of result masks.
    focus=[c for c in cases if c['group']=='reviewed_real' or c['id'] in ['D04','heldout49']]
    for c in focus:
        x,y,w,h=c['roi'];original=c['page'][y:y+h,x:x+w];ref=refs.reference_path(c)
        for kind in ['unet','ragan']:
          for seed in [0,1]:
            for grid in ['full512','tiled0']:
                items=[('原画像',original),('Howe固定',np.asarray(Image.open(ref))) ]
                for arm in tr.ARMS:
                    r=get(kind,seed,arm,c['id'],grid);items.append(((LABEL[arm] if w>=512 else ("追加なし" if arm=="without_near" else "追加あり"))+f" 墨{r['core_r2']}/{r['core_n']} 紙{r['paper_ink']}",np.asarray(Image.open(path_of(c,kind,seed,arm,grid)))))
                items += [('14周基準',np.asarray(Image.open(refs.initial_path(c,kind,seed,grid)))),('前回：別冊紙のみ',np.asarray(Image.open(tr.base.OUT/'evaluation'/f'{kind}_seed{seed}_multibook'/'masks'/grid/f"{c['id']}.png")))]
                target=OUT/'figures'/f"{c['id']}_{kind}_seed{seed}_{grid}.png";panel(target,c['id']+' '+kind+f' 初期化{seed+1} '+grid+' / 周囲紙BCEだけ0対1',items);figures.append(str(target.relative_to(OUT)))
    for c in cs:
        raw=rgb(ROOT/c['crop']);near=np.asarray(Image.open(OUT/'near_paper'/f"{c['id']}.png"))>0;over=raw.copy();over[near]=[255,0,190]
        for seed in [0,1]:
            items=[('学習用原画像',raw),('紫=追加紙教師',over)]
            for kind in ['unet','ragan']:
              for arm in tr.ARMS:
                name=f'{kind}_seed{seed}_{arm}';r=next(r for r in fits if r['condition']==name and r['case']==c['id']);label=('U' if kind=='unet' else 'G')+(' 周囲紙なし' if arm=='without_near' else ' 周囲紙あり')
                items.append((label+f" 墨{r['core_r2']}/{r['core_n']}",np.asarray(Image.open(OUT/'evaluation'/name/'new_teacher_fit'/f"{c['id']}.png"))))
            target=OUT/'training_figures'/f"{c['id']}_seed{seed}.png";panel(target,c['id']+' 学習側適合・汎化ではない',items);figures.append(str(target.relative_to(OUT)))
    rerun=[]
    old_results=json.loads((tr.OLD/'results.json').read_text())['rows']
    for kind in ['unet','ragan']:
      for seed in [0,1]:
        a=[r for r in rows if r['kind']==kind and r['seed']==seed and r['arm']=='without_near' and r['grid']=='full512' and r['case'] in ['R05','R06']]
        b=[r for r in old_results if r['kind']==kind and r['seed']==seed and r['arm']=='paper_ink' and r['grid']=='full512' and r['case'] in ['R05','R06']]
        assert len(a)==len(b)==2
        rerun.append([kind,seed,sum(r['core_r2'] for r in b),sum(r['core_r2'] for r in a),sum(r['paper_ink'] for r in b),sum(r['paper_ink'] for r in a)])
    focusgeo=[r for r in geometry if r['case']=='R06' and r['path']=='center_weak_vertical' and (r['grid']=='full512' or r['condition']=='howe_fixed')]
    lines=['# 筆画周囲の紙教師を加える対照','',
      '同じ8枚の文字付きRGB・1159墨芯・14728既存紙画素に対し、原画像で濃い筆画から離れた5315画素を紙の暫定教師として追加した。追加紙BCEの係数0/1だけを変更し、U-Netと公開由来G、各2初期化の8条件をすべて再実行した。共通14周のモデル・Adamから864更新、閾値.5。GAN・新しい損傷は追加していない。','',
      '追加教師は局所濃度に基づく部分的な代理教師で、専門家による境界GTではない。原画像と旧注釈だけから作成し、8枚の重ね合わせを学習前に確認した。B03は14画素と少なく、全周を覆う制約でもない。過去の出力を見て設計した開発実験であり、盲検テストとはしない。','',
      '[条件](protocol.json)、[教師の生成条件](label_preparation.json)、[固定評価](evaluation_plan.json)。','',
      '## 主評価：512・同じ26領域','',refs.table(['条件','旧墨芯455','退色49','別冊薄字1158','薄字紙黒/6857','R03紙黒','R08紙黒','D04紙黒'],summary),'',
      '墨芯保持は半径2画素の被覆。字形や文字認識の精度ではない。紙の黒化と下の太りの補助量・原寸PNGも合わせて判断する。','',
      '## R06の固定縦線：太りの補助量','',refs.table(['条件','墨芯厳密/点数','内部距離中央値px','半径8帯の黒画素','ROI全黒'],[[r['condition'],f"{r['trace_exact']}/{r['trace_n']}",f"{r['radius_median_all']:.2f}",f"{r['corridor8_black']}/{r['corridor8_n']}",r['roi_black']] for r in focusgeo]),'',
      '内部距離は固定墨芯点から予測の白までの距離（白点は0）。真の線幅や境界精度ではない。帯内の未注釈画素は不明のままであり、非文字GTへ置き換えていない。Howeも幅の正解ではない。','',
      '## 学習側8窓の適合','',refs.table(['条件','墨芯厳密/1159','墨芯r2/1159','旧紙黒/14728','追加紙黒/5315'],fit_table),'','学習側の適合を示す表であり、汎化性能ではない。','',
      '## 格子の感度確認','',refs.table(['条件','格子','別冊薄字墨芯/1158','薄字紙黒','R08紙黒'],sensitivity),'',
      '512主評価・256格子0/16をすべて報告し、良い格子や初期化を選ばない。既知の開発例と暫定部分注釈であり、密なF1やRA-GAN原論文の再現精度は主張しない。','',
      '## 前回の数学的に同じ目的関数との再実行差','',refs.table(['生成器','初期化','前回薄字墨芯','今回なし墨芯','前回薄字紙黒','今回なし紙黒'],rerun),'','今回の係数0条件は新しい損失項を0倍で足す計算グラフで再実行した。同じ目的関数でも前回と完全な数値軌道の一致は保証しない。前回の結果との微小な差を新しい教師の効果として数えず、今回の同一実装内の0/1対照を主比較にする。','',
      '## 原画像付き比較PNG','', '\n'.join(f'- [{f}]({f})' for f in figures),'', '[全指標CSV](全条件の部分指標.csv)、[全経路の形状補助指標CSV](筆画形状の補助指標.csv)、[検証](verification.json)、[研究判断](研究判断_筆画周囲の紙教師.md)。','']
    (OUT/'比較結果_筆画周囲の紙教師.md').write_text('\n'.join(lines))
    assert len(rows)==624 and len(fits)==64
    dump(OUT/'results.json',dict(rows=rows,teacher_fit=fits,summary=summary,teacher_summary=fit_table,geometry=geometry,geometry_focus=focusgeo,rerun_comparison=rerun,figures=figures,formal_accuracy=False))
    fields=['condition','kind','seed','arm','case','group','family','secondary','grid','core_n','core_exact','core_r2','paper_n','paper_ink','total_ink','mask_sha256']
    with (OUT/'全条件の部分指標.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    with (OUT/'筆画形状の補助指標.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(geometry[0]));writer.writeheader();writer.writerows(geometry)
    dump(OUT/'verification.json',dict(protocol_sha256=sha(OUT/'protocol.json'),evaluation_plan_sha256=sha(OUT/'evaluation_plan.json'),paired_initial_losses=paired,independent_disk_rescores=verified,geometry_records=len(geometry),figures={f:sha(OUT/f) for f in figures},formal_accuracy=False))
    (OUT/'source_snapshot').mkdir(exist_ok=True);shutil.copy2(__file__,OUT/'source_snapshot'/Path(__file__).name)
    print('Reported',len(rows),'readouts,',len(fits),'fits,',len(geometry),'geometry records;',len(figures),'PNGs',flush=True)
if __name__=='__main__':
    from pathlib import Path
    main()
