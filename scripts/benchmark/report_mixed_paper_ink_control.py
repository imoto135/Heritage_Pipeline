"""Paired ink-supervision report, cached context references, independent rescoring."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import run_mixed_paper_ink_control as tr
import evaluate_mixed_paper_ink_control as ev
import report_background_diversity_control as previous
from compare_v5_aligned_fade_control import dump,sha,rgb,mask
from apply_ragan_real_regions import panel
ROOT=tr.ROOT;OUT=tr.OUT;BASE=tr.base.OUT
LABELS={'initial':'14周終了','continue':'前回：通常継続','multibook':'前回：別冊紙追加','paper_only':'同じ字付き窓：紙のみ','paper_ink':'同じ字付き窓：紙＋墨芯'}
def state_sha(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def path_of(c,kind,seed,arm,grid):
    if arm=='initial':return previous.initial_path(c,kind,seed,grid)
    root=OUT if arm in tr.ARMS else BASE
    return root/'evaluation'/f'{kind}_seed{seed}_{arm}'/'masks'/grid/f"{c['id']}.png"
def main():
    ev.check();cases=ev.old.cases();bycase={c['id']:c for c in cases};cs=json.loads((OUT/'selected_sources.json').read_text())['cases'];bynew={c['id']:c for c in cs}
    rows=[r for r in json.loads((BASE/'results.json').read_text())['rows'] if r['arm'] in ['initial','continue','multibook']];fit=[];training=[];initial_files={};verifications=[];figures=[];shared=[]
    yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    def verify(r,path,core,paper):
        pred=mask(path);assert pred.shape==core.shape
        values=dict(core_n=int(core.sum()),paper_n=int(paper.sum()),core_exact=int((pred&core).sum()),core_r2=int((ndi.binary_dilation(pred,structure=disk)&core).sum()),paper_ink=int((pred&paper).sum()),total_ink=int(pred.sum()))
        for k,v in values.items():assert r[k]==v,(r['condition'],r['case'],r.get('grid'),k,r[k],v)
        assert sha(path)==r['mask_sha256'];verifications.append(dict(condition=r['condition'],case=r['case'],grid=r.get('grid','teacher_fit256'),sha256=sha(path),disk_rescore=True))
    for kind in ['unet','ragan']:
      for seed in [0,1]:
        payload=torch.load(tr.base.old.OUT/f'{kind}_seed{seed}'/'step_03024.pt',map_location='cpu',weights_only=False);initial=state_sha(payload['model']);del payload
        first=[]
        for arm in tr.ARMS:
            name=f'{kind}_seed{seed}_{arm}';dest=OUT/'evaluation'/name;result=json.loads((dest/'results.json').read_text());assert result['evaluation_plan_sha256']==sha(OUT/'evaluation_plan.json')
            rows+=result['rows'];fit+=result['teacher_fit'];completion=json.loads((OUT/'runs'/name/'completion.json').read_text());history=json.loads((OUT/'runs'/name/'loss_components.json').read_text())
            first.append({k:history[0][k] for k in ['main','background','mixed_paper','mixed_ink_unweighted']})
            assert completion['initial_state_sha256']==initial and completion['initial_optimizer_steps']==[3024] and completion['final_optimizer_steps']==[3888]
            assert len(history)==864 and sum(r['active'] for r in history)==692 and completion['strong_fade_draws']==692
            assert completion['mixed_ink_exposures']==(1159*108 if arm=='paper_ink' else 0) and completion['mixed_paper_exposures']==14728*108
            assert completion['mixed_draws']==completion['background_draws']==864
            assert completion['main_plan_sha256']==sha(BASE/'plans'/f'main_seed{seed}.json') and completion['background_plan_sha256']==sha(BASE/'plans'/f'background_seed{seed}.json') and completion['mixed_plan_sha256']==sha(OUT/'plans'/f'mixed_seed{seed}.json')
            assert all(np.isfinite(r[k]) for r in history for k in ['main','background','mixed_paper','mixed_ink_unweighted','gradient_norm'])
            checkpoint=OUT/'runs'/name/'step_00864.pt';assert result['checkpoint_sha256']==sha(checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['protocol_sha256']==sha(OUT/'protocol.json') and saved['step']==3888 and saved['discriminator'] is None
            assert state_sha(saved['model'])==result['unchanged_state_sha256'];assert {int(q['step']) for q in saved['optimizer']['state'].values() if 'step' in q}=={3888};del saved
            training.append(completion)
        assert first[0]==first[1],(kind,seed,first);shared.append(dict(kind=kind,seed=seed,identical_first_unweighted_losses=first[0]))
    for r in rows:
        c=bycase[r['case']];path=path_of(c,r['kind'],r['seed'],r['arm'],r['grid']);verify(r,path,c['core'],c['paper'])
        if r['arm'] not in tr.ARMS:initial_files[str(path.relative_to(ROOT))]=sha(path)
    for r in fit:
        c=bynew[r['case']];lab=np.asarray(Image.open(OUT/'labels'/f"{c['id']}.png"));verify(r,OUT/'evaluation'/r['condition']/'new_teacher_fit'/f"{c['id']}.png",lab==1,lab==2)
    def row(kind,seed,arm,key,grid='full512'):return next(r for r in rows if (r['kind'],r['seed'],r['arm'],r['case'],r['grid'])==(kind,seed,arm,key,grid))
    def total(kind,seed,arm,key,group=None,family=None,grid='full512'):
        return sum(r[key] for r in rows if r['kind']==kind and r['seed']==seed and r['arm']==arm and r['grid']==grid and (group is None or r['group']==group) and (family is None or r['family']==family) and not r['secondary'])
    focus=[c for c in cases if c['group']=='reviewed_real' or c['id'] in ['D04','heldout49','200017197_00214_1_p00'] or c['group']=='synthetic' and c['family']=='contact_stain']
    for c in focus:
        key=c['id'];x,y,w,h=c['roi'];original=c['page'][y:y+h,x:x+w];hp=previous.reference_path(c)
        for kind in ['unet','ragan']:
          for seed in [0,1]:
            for grid in ['full512','tiled0']:
                items=[('原画像・同じ損傷入力',original),('Howe固定' if hp is not None else '損傷前原本',np.where(mask(hp),0,255).astype(np.uint8) if hp is not None else c['source'])]
                for arm in ['multibook','paper_only','paper_ink','continue']:
                    r=row(kind,seed,arm,key,grid);items.append((LABELS[arm]+f" 墨{r['core_r2']}/{r['core_n']} 非文字{r['paper_ink']}",np.where(mask(path_of(c,kind,seed,arm,grid)),0,255).astype(np.uint8)))
                target=OUT/'figures'/f'{key}_{kind}_seed{seed}_{grid}.png';panel(target,key+' '+kind+f' 初期化{seed+1} '+grid+' / 同入力・墨芯教師だけ変更',items);figures.append(str(target.relative_to(OUT)))
    for c in cs:
        raw=rgb(ROOT/c['crop']);lab=np.asarray(Image.open(OUT/'labels'/f"{c['id']}.png"));over=raw.copy();over[lab==1]=[0,100,255];over[lab==2]=np.rint(.45*raw[lab==2]+.55*np.array([0,200,80])).astype(np.uint8)
        for seed in [0,1]:
            items=[('訓練用原画像',raw),('暫定部分教師 青=墨 緑=紙',over)]
            for kind in ['unet','ragan']:
                for arm in tr.ARMS:
                    name=f'{kind}_seed{seed}_{arm}';r=next(r for r in fit if r['condition']==name and r['case']==c['id']);label=('U' if kind=='unet' else 'G')+(' 紙のみ' if arm=='paper_only' else ' 紙＋墨');items.append((label+f" {r['core_r2']}/{r['core_n']} 紙黒{r['paper_ink']}",np.where(mask(OUT/'evaluation'/name/'new_teacher_fit'/f"{c['id']}.png"),0,255).astype(np.uint8)))
            target=OUT/'training_figures'/f"{c['id']}_seed{seed}_学習側.png";panel(target,c['id']+f' 初期化{seed+1} / 学習側の適合・汎化ではない',items);figures.append(str(target.relative_to(OUT)))
    text=[];paper=[];fits=[];sensitivity=[]
    for kind in ['unet','ragan']:
      for seed in [0,1]:
        for arm in ['initial','continue','multibook',*tr.ARMS]:
            name=('U-Net' if kind=='unet' else '公開G')+f' 初期化{seed+1} '+LABELS[arm];a=row(kind,seed,arm,'R03')
            text.append([name,total(kind,seed,arm,'core_r2',group='train'),row(kind,seed,arm,'heldout49')['core_r2'],total(kind,seed,arm,'core_r2',group='reviewed_real',family='faint_stroke'),total(kind,seed,arm,'paper_ink',group='reviewed_real',family='faint_stroke')])
            paper.append([name,row(kind,seed,arm,'R08')['paper_ink'],a['core_r2'],a['paper_ink'],row(kind,seed,arm,'R01')['paper_ink'],row(kind,seed,arm,'D04')['paper_ink'],total(kind,seed,arm,'paper_ink',group='synthetic',family='contact_stain')])
            if arm in tr.ARMS:
                rs=[r for r in fit if r['condition']==f'{kind}_seed{seed}_{arm}'];fits.append([name,sum(r['core_exact'] for r in rs),sum(r['core_r2'] for r in rs),sum(r['paper_ink'] for r in rs)])
                for grid in ev.GRIDS:
                    a=row(kind,seed,arm,'R03',grid);sensitivity.append([name,grid,total(kind,seed,arm,'core_r2',group='train',grid=grid),total(kind,seed,arm,'core_r2',group='reviewed_real',family='faint_stroke',grid=grid),total(kind,seed,arm,'paper_ink',group='reviewed_real',family='faint_stroke',grid=grid),row(kind,seed,arm,'R08',grid)['paper_ink'],a['core_r2'],a['paper_ink']])
    lines=['# 同じ入力で紙のみ／紙と墨芯を教える対照','',
      '**判定：中心被覆は回復したが、筆画が大きく太る例があり、字形復元の成功とはしない。** [研究判断](研究判断_墨芯教師の追加.md)、[筆画の太りの補助診断](補助診断_筆画の太り.md)と原寸PNGを併せて確認する。','',
      'U-Net・公開G各2初期化について、14周終了の共通重みとAdamから4周864更新を追加した。両条件は元の主入力、別冊の紙16窓、追加の文字付き8窓、紙教師、順番、切り出し、損失係数を同じにし、文字付き窓の墨芯BCEだけを係数0/1で変えた。主評価は追加864更新・閾値.5。GANや新しい損傷は使わない。','',
      '文字付き窓は訓練splitの8冊の別ページから原画像だけで選び、見える墨芯1159点と紙14728画素を部分教師にした。紙は両条件で同量、墨芯は片方のみ。同じ画像を両条件へ与えるため、新しい文字付き入力を加えた効果と墨芯教師の効果を混ぜない。前回の通常継続・別冊紙追加・14周基準は文脈参照であり、今回の同入力対照そのものではない。','',
      '訓練注釈はCodexによる暫定ラベル。自動候補の裏移り・紙繊維・罫線を除外し、原画像と注釈を全8例で確認した。曖昧なB08末尾を除き、手作業経路のうち原画像の局所濃度差12未満の12画素を不明へ除外した。RGBや欠落文字の補完は行わない。評価6冊は教師に使わず、評価注釈も変更していない。','',
      '[凍結条件](protocol.json)、[固定評価](evaluation_plan.json)、[訓練注釈前半](最終注釈確認_00.png)、[後半](最終注釈確認_01.png)。','',
      '## 固定512主評価：墨芯の保持','',previous.table(['条件','学習455点','既知退色49点','別冊薄字1158点','同じ薄字の非文字黒/6857'],text),'',
      '## 固定512主評価：背景と汚れ','',previous.table(['条件','R08紙黒/8735','R03墨/276','R03非文字黒/4955','R01斑点黒/2756','D04斑点黒/5100','合成紙黒/18105'],paper),'',
      '保持は半径2画素の墨芯被覆、非文字は選択内部の黒化画素。密なGT・認識精度ではなく、汚れごと黒くした保持や文字を消した清浄化を成功としない。','',
      '## 新しい教師8窓の学習側適合','',previous.table(['条件','墨芯厳密/1159','墨芯r2/1159','選択紙黒/14728'],fits),'',
      'この表は訓練側の適合であり、汎化性能ではない。太い筆画の対照も含む。教師点数だけでなく別冊薄字と紙の結果を併せて読む。','',
      '## 固定256格子の感度確認','',previous.table(['条件','格子','旧学習455点','別冊薄字1158点','薄字の非文字黒','R08紙黒','R03墨','R03非文字黒'],sensitivity),'',
      '256/stride128・Hann・固定格子0/16を全条件で報告し、例ごとに選ばない。512主評価は維持する。','',
      '## 原画像付きPNG','', '\n'.join(f'- [{Path(f).name}]({f})' for f in figures),'',
      '[全条件CSV](全条件の部分指標.csv)、[検証記録](verification.json)、[研究判断](研究判断_墨芯教師の追加.md)。','',
      '既知の開発例・暫定部分注釈・2初期化の制約を維持する。R07無効、R02参考の扱いも維持した。公開由来生成器の教師あり対照であり、RA-GAN/DKDSの原論文再現やGAN効果の実験ではない。','']
    (OUT/'比較結果_紙と墨芯の教師.md').write_text('\n'.join(lines));assert len(rows)==1560 and len(fit)==64 and len(training)==8
    dump(OUT/'results.json',dict(rows=rows,teacher_fit=fit,text_summary=text,paper_summary=paper,teacher_summary=fits,sensitivity_summary=sensitivity,figures=figures,formal_accuracy=False))
    fields=['condition','kind','seed','arm','case','group','family','secondary','grid','core_n','core_exact','core_r2','paper_n','paper_ink','total_ink','mask_sha256']
    with (OUT/'全条件の部分指標.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    dump(OUT/'verification.json',dict(train_protocol_sha256=sha(OUT/'protocol.json'),evaluation_plan_sha256=sha(OUT/'evaluation_plan.json'),paired_training=training,identical_first_unweighted_losses=shared,
      independent_mask_rescores=verifications,new_eval_masks=624,new_teacher_fit_masks=64,cached_reference_masks=936,cached_reference_files=initial_files,figures={f:sha(OUT/f) for f in figures},formal_accuracy=False))
    shutil.copy2(Path(__file__),OUT/'source_snapshot'/Path(__file__).name);print('Verified8 paired trainings,1624 masks;',len(figures),'PNGs',flush=True)
if __name__=='__main__':main()
