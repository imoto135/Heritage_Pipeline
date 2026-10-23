"""Re-score cached and new masks; original-image comparisons and paired summaries."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import evaluate_background_diversity_control as ev
import run_background_diversity_control as tr
from report_matched_supervised_generators import howe_path
from compare_v5_aligned_fade_control import dump,sha,rgb,mask
from apply_ragan_real_regions import panel
ROOT=tr.ROOT;OUT=tr.OUT;OLD=tr.old.OUT;REAL=ROOT/'tmp_work/frozen_real_transfer_20261022'
LABELS={'initial':'追加前・14周終了','continue':'通常継続','samebook':'Makura背景追加','multibook':'別冊背景追加'}
def initial_path(c,kind,seed,grid):
    name=f'{kind}_seed{seed}';key=c['id']
    if c['group']=='reviewed_real':
        if grid=='full512':return REAL/'additional_control/masks'/f'{name}.png' if key=='R08' else REAL/'evaluation'/name/'masks'/f'{key}.png'
        return REAL/'native256_followup/masks'/name/grid/f'{key}.png'
    return OLD/'evaluation_03024'/name/('masks' if grid=='full512' else f'tiling/{grid}')/f'{key}.png'
def reference_path(c):
    if c['group']=='reviewed_real':return REAL/'additional_control/masks/howe_fixed.png' if c['id']=='R08' else REAL/'evaluation/howe_fixed/masks'/f"{c['id']}.png"
    return howe_path(c)
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def state_digest(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    p=ev.check();cc=ev.cases();case_by={c['id']:c for c in cc};rows=[];verification=[];initial_files={};paired=[];figures=[];classical=[]
    yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    def verify(row,path):
        c=case_by[row['case']];pred=mask(path);assert pred.shape==c['core'].shape
        v=dict(core_n=int(c['core'].sum()),core_exact=int((pred&c['core']).sum()),core_r2=int((ndi.binary_dilation(pred,structure=disk)&c['core']).sum()),paper_n=int(c['paper'].sum()),paper_ink=int((pred&c['paper']).sum()),total_ink=int(pred.sum()))
        for k,n in v.items():assert row[k]==n,(row['condition'],row['case'],row['grid'],k,row[k],n)
        assert sha(path)==row['mask_sha256']
        verification.append(dict(condition=row['condition'],case=row['case'],grid=row['grid'],mask_sha256=sha(path),independent_disk_rescore=True))
    for kind in ['unet','ragan']:
      for seed in [0,1]:
        start=torch.load(OLD/f'{kind}_seed{seed}'/'step_03024.pt',map_location='cpu',weights_only=False);digest=state_digest(start['model']);del start
        first_losses=[json.loads((OUT/'runs'/f'{kind}_seed{seed}_{arm}'/'loss_components.json').read_text())[0]['main'] for arm in tr.ARMS]
        assert len(set(first_losses))==1,(kind,seed,first_losses)
        for arm in ['initial',*tr.ARMS]:
            name=f'{kind}_seed{seed}_{arm}'
            if arm=='initial':
                rs=[]
                for c in cc:
                  for grid in ev.GRIDS:
                    target=initial_path(c,kind,seed,grid);pred=mask(target)
                    r=dict(condition=name,kind=kind,seed=seed,arm=arm,case=c['id'],group=c['group'],family=c['family'],secondary=c.get('secondary',False),grid=grid,mask_sha256=sha(target),**ev.metrics(pred,c));rs.append(r);initial_files[str(target.relative_to(ROOT))]=sha(target)
            else:
                dest=OUT/'evaluation'/name;result=json.loads((dest/'results.json').read_text());assert result['evaluation_plan_sha256']==sha(OUT/'evaluation_plan.json');rs=result['rows']
                comp=json.loads((OUT/'runs'/name/'completion.json').read_text());losses=json.loads((OUT/'runs'/name/'loss_components.json').read_text())
                assert comp['initial_state_sha256']==digest and comp['initial_optimizer_steps']==[3024] and comp['final_optimizer_steps']==[3888]
                assert comp['strong_fade_draws']==692 and sum(r['active'] for r in losses)==692 and len(losses)==864
                assert comp['main_plan_sha256']==sha(OUT/'plans'/f'main_seed{seed}.json')
                assert comp['auxiliary_draws']==(0 if arm=='continue' else 864) and comp['paper_pixels']==(0 if arm=='continue' else 864*128*128)
                checkpoint=OUT/'runs'/name/'step_00864.pt';assert result['checkpoint_sha256']==sha(checkpoint)
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['step']==3888 and saved['additional_steps']==864 and saved['protocol_sha256']==sha(OUT/'protocol.json')
                assert sorted({int(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s})==[3888];del saved
                paired.append(comp)
            for r in rs:
                path=initial_path(case_by[r['case']],kind,seed,r['grid']) if arm=='initial' else OUT/'evaluation'/name/'masks'/r['grid']/f"{r['case']}.png"
                verify(r,path)
            rows+=rs
    for c in cc:
        path=reference_path(c)
        if path is not None:
            classical.append(dict(case=c['id'],method='howe_fixed',path=str(path.relative_to(ROOT)),sha256=sha(path),**ev.metrics(mask(path),c)))
    def row(kind,seed,arm,key,grid='full512'):return next(r for r in rows if (r['kind'],r['seed'],r['arm'],r['case'],r['grid'])==(kind,seed,arm,key,grid))
    def total(kind,seed,arm,field,group=None,family=None,grid='full512'):
        return sum(r[field] for r in rows if r['kind']==kind and r['seed']==seed and r['arm']==arm and r['grid']==grid and (group is None or r['group']==group) and (family is None or r['family']==family) and not r['secondary'])
    # Full-panel references for all7 reviewed examples and representative inherited cases.
    focus=[c for c in cc if c['group']=='reviewed_real' or c['id'] in ['D04','heldout49','200017197_00214_1_p00'] or c['group']=='synthetic' and c['family']=='contact_stain']
    for c in focus:
      key=c['id'];x,y,w,h=c['roi'];original=c['page'][y:y+h,x:x+w];hp=reference_path(c)
      for kind in ['unet','ragan']:
       for seed in [0,1]:
        for grid in ['full512','tiled0']:
            items=[('原画像・同じ損傷入力',original)]
            if hp is not None:items.append(('Howe固定',np.where(mask(hp),0,255).astype(np.uint8)))
            else:items.append(('損傷前原本',c['source']))
            for arm in ['initial',*tr.ARMS]:
                r=row(kind,seed,arm,key,grid);target=initial_path(c,kind,seed,grid) if arm=='initial' else OUT/'evaluation'/f'{kind}_seed{seed}_{arm}'/'masks'/grid/f'{key}.png'
                items.append((LABELS[arm]+f" 墨{r['core_r2']}/{r['core_n']} 背景黒{r['paper_ink']}",np.where(mask(target),0,255).astype(np.uint8)))
            target=OUT/'figures'/f'{key}_{kind}_seed{seed}_{grid}.png';panel(target,key+' '+kind+f' 初期化{seed+1} '+grid+' / 共通14周から4周追加',items);figures.append(str(target.relative_to(OUT)))
    text=[];paper=[];tiling=[]
    for kind in ['unet','ragan']:
      for seed in [0,1]:
       for arm in ['initial',*tr.ARMS]:
        name=('U-Net' if kind=='unet' else '公開G')+f' 初期化{seed+1} '+LABELS[arm]
        a=row(kind,seed,arm,'R03');b=row(kind,seed,arm,'R08')
        text.append([name,total(kind,seed,arm,'core_r2',group='train'),row(kind,seed,arm,'200017197_00214_1_p00')['core_r2'],row(kind,seed,arm,'heldout49')['core_r2'],total(kind,seed,arm,'core_r2',group='reviewed_real',family='faint_stroke'),total(kind,seed,arm,'paper_ink',group='reviewed_real',family='faint_stroke')])
        paper.append([name,total(kind,seed,arm,'core_r2',group='synthetic',family='contact_stain'),total(kind,seed,arm,'paper_ink',group='synthetic',family='contact_stain'),row(kind,seed,arm,'D04')['paper_ink'],row(kind,seed,arm,'R01')['paper_ink'],a['core_r2'],a['paper_ink'],b['paper_ink']])
        for grid in ev.GRIDS:
            a=row(kind,seed,arm,'R03',grid);tiling.append([name,grid,total(kind,seed,arm,'core_r2',group='train',grid=grid),total(kind,seed,arm,'core_r2',group='reviewed_real',family='faint_stroke',grid=grid),a['core_r2'],a['paper_ink'],row(kind,seed,arm,'R08',grid)['paper_ink'],row(kind,seed,arm,'D04',grid)['paper_ink']])
    lines=['# 背景の多様性を増やす対照実験','',
      '共通の14周重み・Adamから4周（864更新）追加し、通常継続／Makura背景追加／別冊背景追加を比較した。U-Net・公開G各2初期化の全12条件。元の文字・損傷・教師・3456提示・強退色692提示は同じ。追加背景は両条件とも16領域を各54回、中央128×128画素の紙だけに係数1のBCEを与え、周囲の文字は教師にしない。主評価は追加864更新・閾値.5。良い初期化や中間重みへの差し替えは行わない。','',
      '別冊背景は訓練splitの8冊で、最近の評価6冊と異なる。Makura背景は既存の訓練ページ群のみ。全32背景を原画像で確認して凍結した。暫定の紙教師であり、追加の文字教師・合成損傷・GANは使っていない。冊を変えると色・模様・周囲文脈も変わるため、本数のみの因果実験ではない。','',
      '[凍結条件](protocol.json)、[入力の原画像一覧](multibook_原画像候補.png)、[Makura対照](samebook_原画像候補.png)、[固定評価計画](evaluation_plan.json)。','',
      '墨は半径2画素の中心線被覆。紙・非文字は選択領域内の黒化画素。暫定部分注釈・原本由来の代理ラベルを含み、文書全体のF値や文字認識精度ではない。今回の実画像は既に見た開発群。R07は無効のまま除外し、R02は参考群のまま主集計に混ぜない。','',
      '## 文字保持：固定512主評価','',table(['条件','学習墨/455','対象/44','既知退色/49','別冊の薄い墨/1158','同じ薄字の非文字黒/6857'],text),'',
      '## 汚れと背景：固定512主評価','',table(['条件','合成汚れ墨/596','合成紙黒/18105','D04斑点黒/5100','R01斑点黒/2756','R03接触墨/276','R03非文字黒/4955','R08紙黒/8735'],paper),'',
      '保持の増加を汚れごとの黒化と取り違えない。背景黒化の低下だけで文字が消えたモデルを採用しない。','',
      '## 固定格子256の感度確認','',table(['条件','格子','学習墨/455','別冊薄字/1158','R03墨/276','R03非文字黒/4955','R08紙黒/8735','D04黒/5100'],tiling),'',
      '256はstride128・Hann混合・固定格子0/16。両格子を全条件で報告し、例ごとに選ばない。512を主評価のまま維持する。','',
      '## 原画像と手法を並べたPNG','', '\n'.join(f'- [{Path(name).name}]({name})' for name in figures),'',
      '## 検証','', '保存マスクを独立の半径2画素disk膨張で再採点した。3条件の初期重み・Adam開始3024・終了3888・共通切り出し計画・強退色提示692・追加紙の等量提示を照合した。PNGは原寸の貼り込み画素が一致することを確認した。数値一致は暫定教師の意味的正しさを保証しない。','', '[検証記録](verification.json)。','']
    lines+=['[全26例・全条件・全格子のCSV](全条件の部分指標.csv)、[研究判断](研究判断_背景追加と文字保持.md)。','']
    (OUT/'比較結果_背景多様性.md').write_text('\n'.join(lines))
    dump(OUT/'results.json',dict(rows=rows,classical=classical,figures=figures,text_summary=text,paper_summary=paper,tiling_summary=tiling,formal_accuracy=False))
    fields=['condition','kind','seed','arm','case','group','family','secondary','grid','core_n','core_exact','core_r2','paper_n','paper_ink','total_ink','mask_sha256']
    with (OUT/'全条件の部分指標.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    assert len(rows)==1248 and len(paired)==12
    dump(OUT/'verification.json',dict(train_protocol_sha256=sha(OUT/'protocol.json'),evaluation_plan_sha256=sha(OUT/'evaluation_plan.json'),paired_training_checks=paired,
      independent_mask_rescores=verification,new_masks=936,cached_initial_masks=312,source_and_input_hashes_checked=True,cached_initial_files=initial_files,
      figures={name:sha(OUT/name) for name in figures},formal_accuracy=False))
    shutil.copy2(Path(__file__),OUT/'source_snapshot'/Path(__file__).name)
    print('Verified12 trainings and1248 masks; wrote',len(figures),'PNGs',flush=True)
if __name__=='__main__':main()
