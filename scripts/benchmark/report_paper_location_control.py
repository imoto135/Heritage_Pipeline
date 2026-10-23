"""Paired source-near-paper control: partial labels and geometry together."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import torch
import evaluate_paper_location_control as ev
import run_paper_location_control as tr
import report_background_diversity_control as refs
from report_mixed_paper_ink_control import state_sha
from apply_ragan_real_regions import panel
from compare_v5_aligned_fade_control import dump,sha,rgb,mask
ROOT=tr.ROOT;OUT=tr.OUT
LABEL={'near':'近傍紙','far':'遠方紙'}
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
            assert completion['mixed_ink_exposures']==125172 and completion['mixed_paper_exposures']==1590624 and completion['extra_paper_exposures']==574020
            first.append({k:history[0][k] for k in ['main','background','mixed_paper','mixed_ink_unweighted']})
            assert all(np.isfinite(h[k]) for h in history for k in ['main','background','mixed_paper','mixed_ink_unweighted','extra_paper_unweighted','gradient_norm'])
            cp=OUT/'runs'/name/'step_00864.pt';assert q['checkpoint_sha256']==sha(cp);saved=torch.load(cp,map_location='cpu',weights_only=False)
            assert state_sha(saved['model'])==q['unchanged_state_sha256'] and saved['step']==3888 and saved['additional_steps']==864 and saved['protocol_sha256']==sha(OUT/'protocol.json') and saved['discriminator'] is None
            assert {int(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s}=={3888};del saved
        assert first[0]==first[1],(kind,seed,first);paired.append(dict(kind=kind,seed=seed,identical_first_unweighted_losses=first[0]))
    for r in rows:
        c=by[r['case']];pred=verify(r,path_of(c,r['kind'],r['seed'],r['arm'],r['grid']),c['core'],c['paper'])
        if c['group']=='reviewed_real' and not c.get('secondary',False):geom(r['condition'],c,r['grid'],pred)
    for r in fits:
        lab=np.asarray(Image.open(tr.OLD/'labels'/f"{r['case']}.png"));target=OUT/'evaluation'/r['condition']/'new_teacher_fit'/f"{r['case']}.png";pred=verify(r,target,lab==1,lab==2)
        for loc in tr.ARMS:
            paper=np.asarray(Image.open(OUT/'paper_masks'/loc/f"{r['case']}.png"))>0;assert r[loc+'_paper_n']==int(paper.sum()) and r[loc+'_paper_ink']==int((pred&paper).sum())
    for c in cases:
        if c['group']=='reviewed_real' and not c.get('secondary',False):geom('howe_fixed',c,'native_reference',mask(refs.reference_path(c)))
    def get(kind,seed,arm,key,grid='full512'):return next(r for r in rows if (r['kind'],r['seed'],r['arm'],r['case'],r['grid'])==(kind,seed,arm,key,grid))
    summary=[];fit_table=[];sensitivity=[]
    for kind in ['unet','ragan']:
      for seed in [0,1]:
       for arm in tr.ARMS:
        name=f'{kind}_seed{seed}_{arm}';rs=[r for r in rows if r['condition']==name and r['grid']=='full512'];faint=[r for r in rs if r['case'] in ['R05','R06']]
        summary.append([name,sum(r['core_r2'] for r in rs if r['group']=='train'),get(kind,seed,arm,'heldout49')['core_r2'],sum(r['core_r2'] for r in faint),sum(r['paper_ink'] for r in faint),get(kind,seed,arm,'R03')['paper_ink'],get(kind,seed,arm,'R08')['paper_ink'],get(kind,seed,arm,'D04')['paper_ink']])
        fs=[r for r in fits if r['condition']==name];fit_table.append([name,sum(r['core_exact'] for r in fs),sum(r['core_r2'] for r in fs),sum(r['paper_ink'] for r in fs),sum(r['near_paper_ink'] for r in fs),sum(r['far_paper_ink'] for r in fs)])
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
                    r=get(kind,seed,arm,c['id'],grid);items.append((LABEL[arm]+f" 墨{r['core_r2']}/{r['core_n']} 紙{r['paper_ink']}",np.asarray(Image.open(path_of(c,kind,seed,arm,grid)))))
                items += [('参考：追加紙なし',np.asarray(Image.open(tr.NEAR/'evaluation'/f'{kind}_seed{seed}_without_near'/'masks'/grid/f"{c['id']}.png"))),('14周基準',np.asarray(Image.open(refs.initial_path(c,kind,seed,grid))))]
                target=OUT/'figures'/f"{c['id']}_{kind}_seed{seed}_{grid}.png";panel(target,c['id']+' '+kind+f' 初期化{seed+1} '+grid+' / 同画素数・同係数：近傍紙対遠方紙',items);figures.append(str(target.relative_to(OUT)))
    for c in cs:
        raw=rgb(ROOT/c['crop']);near=np.asarray(Image.open(OUT/'paper_masks/near'/f"{c['id']}.png"))>0;far=np.asarray(Image.open(OUT/'paper_masks/far'/f"{c['id']}.png"))>0;over=raw.copy();over[near]=[255,0,190];over[far]=[0,175,230]
        for seed in [0,1]:
            items=[('学習用原画像',raw),('紫=近傍 水色=遠方',over)]
            for kind in ['unet','ragan']:
              for arm in tr.ARMS:
                name=f'{kind}_seed{seed}_{arm}';r=next(r for r in fits if r['condition']==name and r['case']==c['id']);label=('U' if kind=='unet' else 'G')+(' 近傍紙' if arm=='near' else ' 遠方紙')
                items.append((label+f" 墨{r['core_r2']}/{r['core_n']}",np.asarray(Image.open(OUT/'evaluation'/name/'new_teacher_fit'/f"{c['id']}.png"))))
            target=OUT/'training_figures'/f"{c['id']}_seed{seed}.png";panel(target,c['id']+' 学習側適合・汎化ではない',items);figures.append(str(target.relative_to(OUT)))
    focusgeo=[r for r in geometry if r['case']=='R06' and r['path']=='center_weak_vertical' and (r['grid']=='full512' or r['condition']=='howe_fixed')]
    # Previous zero-addition outputs are clearly context references, not this paired contrast.
    context=[];cached_files={}
    for kind in ['unet','ragan']:
      for seed in [0,1]:
        old=json.loads((tr.NEAR/'evaluation'/f'{kind}_seed{seed}_without_near/results.json').read_text());rs=[r for r in old['rows'] if r['grid']=='full512'];faint=[r for r in rs if r['case'] in ['R05','R06']]
        context.append([kind,seed,sum(r['core_r2'] for r in faint),sum(r['paper_ink'] for r in faint)])
        for c in focus:
          for grid in ['full512','tiled0']:
            f=tr.NEAR/'evaluation'/f'{kind}_seed{seed}_without_near'/'masks'/grid/f"{c['id']}.png";cached_files[str(f.relative_to(ROOT))]=sha(f)
    lines=['# 紙教師の位置を揃えた増量対照：近傍と遠方','',
      '近傍と遠方の追加紙教師を同じ8原画像から各5315画素選び、画像ごとの画素数・局所濃度差の整数層・明るさ32階調の層別度数を一致させた。両条件の追加紙BCE係数は1。同じ14周モデルとAdamから864更新を追加し、U-Net・公開由来G各2初期化の8条件をすべて再実行した。','',
      '近傍は既存墨芯から20画素以内、遠方は全手作業墨芯から40画素以上離れた領域。遠方が全未注釈文字から40画素以上離れているとは主張しない。両者とも原画像の濃度に基づく暫定紙教師で、専門家GTではない。近傍の塊状領域と遠方の散在画素という形状差も含む空間配置の比較である。','',
      '既存の墨芯1159点・紙14728画素・別冊背景16窓・主入力・順番・切り出し・更新数は共通。遠方教師は既存紙を重複加算せず、新しい画素から選んだ。新しいRGB・損傷・GANは追加していない。','',
      '[凍結条件](protocol.json)、[教師生成](label_preparation.json)、[評価](evaluation_plan.json)、[形状補助計画](geometry_plan.json)、[研究判断](研究判断_紙教師の位置.md)。','',
      '## 固定512主評価','',refs.table(['条件','旧墨芯455','既知退色49','別冊薄字1158','薄字紙黒/6857','R03紙黒','R08紙黒','D04紙黒'],summary),'',
      '墨芯保持は半径2画素の被覆であり、字形・認識精度ではない。密なGTがないためF1や文字形状の正しさは主張しない。','',
      '## R06の固定細線：太りの補助量','',refs.table(['条件','墨芯厳密/点数','内部距離中央値px','半径8帯内黒/帯画素','ROI全黒'],[[r['condition'],f"{r['trace_exact']}/{r['trace_n']}",f"{r['radius_median_all']:.2f}",f"{r['corridor8_black']}/{r['corridor8_n']}",r['roi_black']] for r in focusgeo]),'',
      '内部距離は固定墨芯から予測の白までの距離（白点は0）で、真の線幅ではない。帯の不明画素は不明のまま。Howeも線幅GTではない。','',
      '## 学習側：両位置の紙を共通に評価','',refs.table(['条件','墨芯厳密/1159','墨芯r2/1159','既存紙黒/14728','近傍紙黒/5315','遠方紙黒/5315'],fit_table),'',
      '学習側適合は汎化を保証しない。近傍・遠方とも両条件の出力に対して同じマスクで評価した。','',
      '## 格子の感度','',refs.table(['条件','格子','薄字墨芯/1158','薄字紙黒','R08紙黒'],sensitivity),'',
      '主評価512、256格子0/16をすべて報告し、条件ごとに選ばない。','',
      '## 文脈参照：前回の追加紙なし','',refs.table(['生成器','初期化','薄字墨芯/1158','薄字紙黒'],context),'',
      'この追加なし条件は前回の保存結果で、今回の近傍対遠方が主対照。計算上の軌道差があるため、過去の追加なしとの小差を位置の効果とは数えない。','',
      '## 原画像付き比較PNG','', '\n'.join(f'- [{f}]({f})' for f in figures),'',
      '[全指標CSV](全条件の部分指標.csv)、[全経路の形状補助指標CSV](筆画形状の補助指標.csv)、[検証](verification.json)。','',
      '既知開発例・暫定部分注釈・2初期化。R02参考、R07無効を維持する。GAN効果・RA-GAN/DKDS再現の検証ではない。','']
    (OUT/'比較結果_紙教師の位置.md').write_text('\n'.join(lines))
    assert len(rows)==624 and len(fits)==64 and len(geometry)==500 and len(figures)==88
    dump(OUT/'results.json',dict(rows=rows,teacher_fit=fits,summary=summary,teacher_summary=fit_table,geometry=geometry,geometry_focus=focusgeo,figures=figures,formal_accuracy=False))
    fields=['condition','kind','seed','arm','case','group','family','secondary','grid','core_n','core_exact','core_r2','paper_n','paper_ink','total_ink','mask_sha256']
    for name,records,keys in [('全条件の部分指標.csv',rows,fields),('筆画形状の補助指標.csv',geometry,list(geometry[0]))]:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    dump(OUT/'verification.json',dict(protocol_sha256=sha(OUT/'protocol.json'),evaluation_plan_sha256=sha(OUT/'evaluation_plan.json'),geometry_plan_sha256=sha(OUT/'geometry_plan.json'),paired_initial_common_losses=paired,independent_disk_rescores=verified,geometry_records=len(geometry),cached_context_files=cached_files,figures={f:sha(OUT/f) for f in figures},formal_accuracy=False))
    shutil.copy2(__file__,OUT/'source_snapshot'/Path(__file__).name)
    print('Reported624 readouts,64 fits,500 geometry records,88 PNGs',flush=True)
if __name__=='__main__':main()
