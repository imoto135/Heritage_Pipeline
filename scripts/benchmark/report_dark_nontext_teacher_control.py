"""Report same-input clear/stain supervision, retention and source-context fit."""
import json,csv,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import evaluate_dark_nontext_teacher_control as ev
import evaluate_dark_nontext_source_context as ctx
import evaluate_dark_nontext_text_guards as guards
import run_dark_nontext_teacher_control as tr
import report_background_diversity_control as refs
from report_mixed_paper_ink_control import state_sha
from compare_v5_aligned_fade_control import rgb,mask,dump,sha
from apply_ragan_real_regions import panel
ROOT=tr.ROOT;OUT=tr.OUT;LABEL={'clear':'明るい紙教師','stain':'汚れ教師'}
def path_of(c,seed,arm,grid):return OUT/'evaluation'/f'ragan_seed{seed}_{arm}'/'masks'/grid/f"{c['id']}.png"
def main():
    ev.check();ctx.check();guards.check();plan=json.loads((OUT/'report_plan.json').read_text());assert plan['script_sha256']==sha(__file__)
    cases=ev.old.cases();by={c['id']:c for c in cases};cs=json.loads((OUT/'selected_sources.json').read_text())['cases'];mixed=json.loads((tr.OLD/'selected_sources.json').read_text())['cases'];bynew={c['id']:c for c in cs};rows=[];fits=[];contexts=[];verified=[];paired=[];geometry=[];figures=[]
    grow=json.loads((OUT/'text_guard_results.json').read_text())['rows'];ann=json.loads((OUT/'text_guard_annotations.json').read_text())['rows'];gcore={r['id']:np.asarray(Image.open(ROOT/r['mask_file']))>0 for r in ann};yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    def guard(name,key,grid):return next(r for r in grow if (r['condition'],r['case'],r['grid'])==(name,key,grid))
    def verify(r,f,core,paper):
        a=mask(f);v=dict(core_n=int(core.sum()),paper_n=int(paper.sum()),core_exact=int((a&core).sum()),core_r2=int((ndi.binary_dilation(a,structure=disk)&core).sum()),paper_ink=int((a&paper).sum()),total_ink=int(a.sum()))
        assert a.shape==core.shape and sha(f)==r['mask_sha256'];assert all(r[k]==v[k] for k in v),(r['condition'],r['case'],v)
        verified.append(dict(path=str(f.relative_to(ROOT)),sha256=sha(f)));return a
    def geom(name,c,grid,a):
        inside=ndi.distance_transform_edt(a)
        for p in c['paths']:
            trace=np.zeros(a.shape,bool);xy=p['xy'];trace[xy[:,1],xy[:,0]]=True;band=ndi.distance_transform_edt(~trace)<=8;v=inside[trace]
            geometry.append(dict(condition=name,case=c['id'],grid=grid,path=p['id'],trace_n=int(trace.sum()),trace_exact=int((v>0).sum()),radius_median_all=float(np.median(v)),radius_p90_all=float(np.percentile(v,90)),corridor8_n=int(band.sum()),corridor8_black=int((a&band).sum()),roi_black=int(a.sum())))
    for seed in [0,1]:
        q0=torch.load(tr.base.old.OUT/f'ragan_seed{seed}/step_03024.pt',map_location='cpu',weights_only=False);initial=state_sha(q0['model']);del q0;first=[]
        for arm in tr.ARMS:
            name=f'ragan_seed{seed}_{arm}';dest=OUT/'evaluation'/name;q=json.loads((dest/'results.json').read_text());cq=json.loads((OUT/'source_context'/name/'results.json').read_text());rows+=q['rows'];fits+=q['teacher_fit'];contexts+=cq['rows']
            assert q['evaluation_plan_sha256']==sha(OUT/'evaluation_plan.json') and cq['source_context_plan_sha256']==sha(OUT/'source_context_plan.json')
            done=json.loads((OUT/'runs'/name/'completion.json').read_text());hs=json.loads((OUT/'runs'/name/'loss_components.json').read_text());assert len(hs)==864 and sum(h['active'] for h in hs)==692
            assert done['initial_state_sha256']==initial and done['initial_optimizer_steps']==[3024] and done['final_optimizer_steps']==[3888]
            for k,n in [('mixed_ink_exposures',125172),('mixed_paper_exposures',1590624),('extra_paper_exposures',574020),('spot_paper_exposures',497664)]:assert done[k]==n,(name,k)
            assert done['main_plan_sha256']==sha(tr.base.OUT/'plans'/f'main_seed{seed}.json') and done['background_plan_sha256']==sha(tr.base.OUT/'plans'/f'background_seed{seed}.json') and done['mixed_plan_sha256']==sha(tr.OLD/'plans'/f'mixed_seed{seed}.json') and done['spot_plan_sha256']==sha(OUT/'plans'/f'spot_seed{seed}.json')
            keys=['main','background','mixed_paper','mixed_ink_unweighted','extra_paper_unweighted'];first.append({k:hs[0][k] for k in keys});assert all(np.isfinite(h[k]) for h in hs for k in keys+['spot_paper_loss','gradient_norm'])
            ck=OUT/'runs'/name/'step_00864.pt';assert q['checkpoint_sha256']==cq['checkpoint_sha256']==sha(ck);saved=torch.load(ck,map_location='cpu',weights_only=False)
            assert saved['protocol_sha256']==sha(OUT/'protocol.json') and saved['step']==3888 and saved['discriminator'] is None and state_sha(saved['model'])==q['unchanged_state_sha256']==cq['unchanged_state_sha256'];assert {int(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s}=={3888};del saved
        assert first[0]==first[1],(seed,first);paired.append(dict(seed=seed,identical_common_first_losses=first[0]))
    for r in rows:
        c=by[r['case']];a=verify(r,path_of(c,r['seed'],r['arm'],r['grid']),c['core'],c['paper'])
        if c['group']=='reviewed_real' and not c.get('secondary',False):geom(r['condition'],c,r['grid'],a)
        if c['id'] in ['D04','R03','R05','R06']:
            prob=np.load(OUT/'evaluation'/r['condition']/'probabilities'/r['grid']/f"{c['id']}.npy");assert np.array_equal(a,prob<.5)
    for r in fits:
        dest=OUT/'evaluation'/r['condition']
        if r['fit_group']=='mixed':
            lab=np.asarray(Image.open(tr.OLD/'labels'/f"{r['case']}.png"));a=verify(r,dest/'mixed_teacher_fit'/f"{r['case']}.png",lab==1,lab==2);near=np.asarray(Image.open(tr.NEAR/'near_paper'/f"{r['case']}.png"))>0;assert r['near_paper_ink']==np.count_nonzero(a[near])
        else:
            masks={arm:np.asarray(Image.open(OUT/'paper_masks'/arm/f"{r['case']}.png"))>0 for arm in tr.ARMS};a=verify(r,dest/'stain_teacher_fit'/f"{r['case']}.png",np.zeros((256,256),bool),masks['stain'])
            for arm,m in masks.items():assert r[arm+'_paper_n']==576 and r[arm+'_paper_ink']==np.count_nonzero(a[m])
    for r in contexts:
        f=OUT/'source_context'/r['condition']/'masks'/r['grid']/f"{r['case']}.png";a=mask(f);assert sha(f)==r['mask_sha256'] and int(a.sum())==r['total_ink'];assert r['paper_n_each']==576
        for arm in tr.ARMS:
            m=np.asarray(Image.open(OUT/'paper_masks'/arm/f"{r['case']}.png"))>0;assert r[arm+'_paper_ink']==np.count_nonzero(a[m])
        verified.append(dict(path=str(f.relative_to(ROOT)),sha256=sha(f)))
    for c in cases:
        if c['group']=='reviewed_real' and not c.get('secondary',False):geom('howe_fixed',c,'native_reference',mask(refs.reference_path(c)))
    def get(seed,arm,key,grid='full512'):return next(r for r in rows if (r['seed'],r['arm'],r['case'],r['grid'])==(seed,arm,key,grid))
    retention=[];stains=[];fit_table=[];context_table=[];sensitivity=[]
    for seed in [0,1]:
      for arm in tr.ARMS:
        name=f'ragan_seed{seed}_{arm}';rr=[r for r in rows if r['condition']==name and r['grid']=='full512'];thin=[r for r in rr if r['case'] in ['R05','R06']]
        retention.append([name,sum(r['core_r2'] for r in rr if r['group']=='train'),get(seed,arm,'heldout49')['core_r2'],sum(r['core_r2'] for r in thin),sum(r['paper_ink'] for r in thin),guard(name,'D04','full512')['core_r2']])
        stains.append([name,get(seed,arm,'D04')['paper_ink'],get(seed,arm,'R01')['paper_ink'],get(seed,arm,'R03')['core_r2'],get(seed,arm,'R03')['paper_ink'],get(seed,arm,'R04')['core_r2'],get(seed,arm,'R04')['paper_ink'],get(seed,arm,'R08')['paper_ink']])
        fs=[r for r in fits if r['condition']==name];fit_table.append([name,sum(r['core_r2'] for r in fs),sum(r['paper_ink'] for r in fs if r['fit_group']=='mixed'),sum(r['near_paper_ink'] for r in fs if r['fit_group']=='mixed'),sum(r['clear_paper_ink'] for r in fs if r['fit_group']=='stain'),sum(r['stain_paper_ink'] for r in fs if r['fit_group']=='stain')])
        for grid in ['native256_fit',*ev.GRIDS]:
            rs=[r for r in fits if r['condition']==name and r['fit_group']=='stain'] if grid=='native256_fit' else [r for r in contexts if r['condition']==name and r['grid']==grid]
            context_table.append([name,grid,sum(r['clear_paper_ink'] for r in rs),sum(r['stain_paper_ink'] for r in rs),guard(name,'S03',grid)['core_r2']])
        for grid in ev.GRIDS:
            rr=[get(seed,arm,k,grid) for k in ['R05','R06']];sensitivity.append([name,grid,sum(r['core_r2'] for r in rows if r['condition']==name and r['grid']==grid and r['group']=='train'),sum(r['core_r2'] for r in rr),sum(r['paper_ink'] for r in rr),get(seed,arm,'D04',grid)['paper_ink'],guard(name,'D04',grid)['core_r2']])
    focus=[c for c in cases if c['group']=='reviewed_real' or c['id'] in ['D04','heldout49','200017197_00103_1_p00','200017197_00214_1_p00']]
    for c in focus:
        x,y,w,h=c['roi'];raw=c['page'][y:y+h,x:x+w];ref=refs.reference_path(c);annotation=raw.copy();viscore=gcore['D04'] if c['id']=='D04' else c['core'];annotation[viscore]=[0,90,255];annotation[c['paper']]=np.rint(.5*raw[c['paper']]+.5*np.array([0,190,90])).astype(np.uint8)
        for seed in [0,1]:
          for grid in ['full512','tiled0']:
            items=[('同じ入力原画像',raw),('損傷前原画像' if ref is None else 'Howe固定',c['source'] if ref is None else np.asarray(Image.open(ref)))];preds={}
            for arm in tr.ARMS:
                r=get(seed,arm,c['id'],grid);preds[arm]=mask(path_of(c,seed,arm,grid));ink=guard(r['condition'],'D04',grid) if c['id']=='D04' else r;prefix='補助墨' if c['id']=='D04' else '墨'
                items.append((LABEL[arm]+f" {prefix}{ink['core_r2']}/{ink['core_n']} 紙{r['paper_ink']}",np.asarray(Image.open(path_of(c,seed,arm,grid)))))
            diff=raw.copy();diff[preds['clear']&~preds['stain']]=[255,0,60];diff[~preds['clear']&preds['stain']]=[0,100,255];items += [('差分 赤=黒の消失 青=追加',diff),('部分注釈 青=墨 緑=非文字',annotation)]
            target=OUT/'figures'/f"{c['id']}_seed{seed}_{grid}.png";panel(target,c['id']+f' 初期化{seed+1} {grid} / 同じ画像で明るい紙対汚れを教師化',items);figures.append(str(target.relative_to(OUT)))
    for c in cs:
        raw=rgb(ROOT/c['crop']);over=raw.copy()
        for arm,color in [('clear',[0,190,90]),('stain',[255,0,180])]:over[np.asarray(Image.open(OUT/'paper_masks'/arm/f"{c['id']}.png"))>0]=color
        for seed in [0,1]:
            items=[('学習原画像',raw),('緑=明るい紙 紫=汚れ',over)]
            for grid in ['native256_fit','full512']:
              for arm in tr.ARMS:
                name=f'ragan_seed{seed}_{arm}';rs=[r for r in fits if r['condition']==name and r['case']==c['id'] and r['fit_group']=='stain'] if grid=='native256_fit' else [r for r in contexts if r['condition']==name and r['case']==c['id'] and r['grid']==grid];r=rs[0];f=OUT/'evaluation'/name/'stain_teacher_fit'/f"{c['id']}.png" if grid=='native256_fit' else OUT/'source_context'/name/'masks'/grid/f"{c['id']}.png"
                items.append((('窓256 ' if grid=='native256_fit' else '頁512 ')+('紙教師' if arm=='clear' else '汚れ教師')+f" 汚{r['stain_paper_ink']}",np.asarray(Image.open(f))))
            target=OUT/'source_figures'/f"{c['id']}_seed{seed}.png";panel(target,c['id']+f' 初期化{seed+1} / 学習側・文脈の診断、別冊評価ではない',items);figures.append(str(target.relative_to(OUT)))
    for c in mixed:
        raw=rgb(ROOT/c['crop']);lab=np.asarray(Image.open(tr.OLD/'labels'/f"{c['id']}.png"));over=raw.copy();over[lab==1]=[0,90,255];over[lab==2]=[0,190,90];items=[('共通の学習原画像',raw),('共通部分教師 青=墨 緑=紙',over)]
        for seed in [0,1]:
          for arm in tr.ARMS:
            name=f'ragan_seed{seed}_{arm}';r=next(r for r in fits if r['condition']==name and r['case']==c['id'] and r['fit_group']=='mixed');items.append((f"初期{seed+1} "+('紙教師' if arm=='clear' else '汚れ教師')+f" 墨{r['core_r2']}/{r['core_n']}",np.asarray(Image.open(OUT/'evaluation'/name/'mixed_teacher_fit'/f"{c['id']}.png"))))
        target=OUT/'mixed_figures'/f"{c['id']}.png";panel(target,c['id']+' 共通教師への適合・別冊評価ではない',items);figures.append(str(target.relative_to(OUT)))
    geo=[r for r in geometry if r['case']=='R06' and r['path']=='center_weak_vertical' and (r['grid']=='full512' or r['condition']=='howe_fixed')]
    lines=['# 明るい紙と濃い非文字を教える同入力対照','',
      '公開由来生成器の2初期化について、同じ8枚の追加RGBと、同数・同形状の紙教師を使い、明るい紙か孤立汚れかだけを変えた。各24×24、合計4608画素、BCE係数1。6冊8ページの余白から原画像だけで選び、全8教師の重ね合わせを確認した。S08は薄い着色汚れの対照も兼ねる。','',
      '共通14周モデルとAdamから864更新。元の主入力、16背景窓、8文字付き窓と墨芯・紙・近傍紙損失は共通。追加画像は両条件で同じ別forward。厳密な再現設定・反射パディングの等価実装を両条件に使った。GANや新しい合成損傷は追加していない。','',
      '**D04を含む評価用の画像・冊は新しい教師に使っていない。** 固定26例は既知の開発例で、盲検テストではない。追加負例もCodexによる暫定注釈で、専門家GTではない。色・濃度・模様・教師の位置は同時に変わるため、単一の物理的な汚れ特徴の効果とはしない。','',
      '[研究判断](研究判断_濃い非文字教師.md)、[凍結条件](protocol.json)、[評価](evaluation_plan.json)、[文脈診断](source_context_plan.json)、[文字保持の補助計画](text_guard_plan.json)。','',
      '## 512主評価：文字を残せるか','',refs.table(['条件','既存墨芯455','退色49','別冊薄字1158','薄字紙黒/6857','D04補助墨525'],retention),'',
      'D04補助墨は新しく原画像から固定した保持経路で、既存26例のラベルを変更したものではない。学習に使っていない。墨芯被覆は半径2であり、境界・認識の精度ではない。','',
      '## 512主評価：汚れと誤黒化','',refs.table(['条件','D04斑点黒/5100','R01斑点黒/2756','R03墨/276','R03紙黒/4955','R04墨/342','R04紙黒/5516','R08紙黒/8735'],stains),'',
      'HoweのD04斑点黒は0。D04で文字ごと消す出力を成功としない。','',
      '## 追加教師の適合とページ文脈','',refs.table(['条件','入力文脈','明るい紙黒/4608','汚れ黒/4608','S03補助墨/115'],context_table),'',
      '学習窓256は教師への適合、残りは同じ訓練原本のページ上での処理。別冊への一般化ではない。S03の文字経路も診断だけに使い、教師へ追加していない。窓で消えてもページで戻る汚れを、この表とPNGで区別する。','',
      '## 共通教師と追加教師の学習側適合','',refs.table(['条件','墨芯r2/1159','既存紙黒/14728','共通近傍紙黒/5315','追加明るい紙黒/4608','追加汚れ黒/4608'],fit_table),'',
      '## R06固定細線の太りの補助量','',refs.table(['条件','厳密墨/272','内部距離中央値px','半径8帯内黒','ROI全黒'],[[r['condition'],r['trace_exact'],round(r['radius_median_all'],2),r['corridor8_black'],r['roi_black']] for r in geo]),'',
      '内部距離と帯内面積は真の線幅・境界GTではない。帯の未知画素は不明のまま。','',
      '## 固定格子の感度','',refs.table(['条件','格子','既存墨芯455','別冊薄字1158','薄字紙黒','D04斑点黒','D04補助墨525'],sensitivity),'',
      '512主評価、256格子0/16をすべて報告し、良い初期化・格子・中間チェックポイントを選ばない。','',
      '## 原画像付きPNG','', '\n'.join(f'- [{f}]({f})' for f in figures),'',
      '[全指標CSV](全条件の部分指標.csv)、[文脈診断CSV](学習原画像の文脈別指標.csv)、[形状補助量CSV](筆画形状の補助指標.csv)、[文字保持の補助指標](text_guard_results.json)、[検証](verification.json)。','']
    (OUT/'比較結果_濃い非文字教師.md').write_text('\n'.join(lines));assert len(rows)==312 and len(fits)==64 and len(contexts)==96 and len(geometry)==260 and len(verified)==472 and len(figures)==68
    dump(OUT/'results.json',dict(rows=rows,teacher_fit=fits,source_context=contexts,geometry=geometry,retention_summary=retention,stain_summary=stains,fit_summary=fit_table,context_summary=context_table,figures=figures,formal_accuracy=False))
    fields=['condition','kind','seed','arm','case','group','family','secondary','grid','core_n','core_exact','core_r2','paper_n','paper_ink','total_ink','mask_sha256']
    for name,records,keys in [('全条件の部分指標.csv',rows,fields),('学習原画像の文脈別指標.csv',contexts,list(contexts[0])),('筆画形状の補助指標.csv',geometry,list(geometry[0]))]:
        with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(records)
    dump(OUT/'verification.json',dict(paired_initial_common_losses=paired,independent_mask_rescores=verified,guard_rows=29,geometry_rows=260,figures={f:sha(OUT/f) for f in figures},protocol_sha256=sha(OUT/'protocol.json'),evaluation_plan_sha256=sha(OUT/'evaluation_plan.json'),source_context_plan_sha256=sha(OUT/'source_context_plan.json'),guard_plan_sha256=sha(OUT/'text_guard_plan.json'),report_plan_sha256=sha(OUT/'report_plan.json'),formal_accuracy=False))
    shutil.copy2(__file__,OUT/'source_snapshot'/Path(__file__).name);print('Reported312 eval+64 fit+96 context masks,29 guard scores,260 geometry records,68 PNGs',flush=True)
if __name__=='__main__':main()
