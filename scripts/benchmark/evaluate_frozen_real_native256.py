"""Exploratory frozen followup: prior fixed native256 grids on all reviewed cases."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import frozen_real_transfer as f
from evaluate_ragan_native_tiling import tiled_probability,tiles,METHODS
from evaluate_matched_supervised_generators import ThreeInput
D=f.OUT/'native256_followup'
EXTRA=f.OUT/'additional_control'
def sources():
    p=f.load();extra=json.loads((EXTRA/'protocol.json').read_text())
    return [c for c in p['cases'] if c['id']!='R07']+[extra['case']]
def label(c):
    if c['id']=='R08':a=json.loads((EXTRA/'annotation.json').read_text())
    else:a=next(r for r in json.loads((f.OUT/'annotations_source_only.json').read_text())['regions'] if r['id']==c['id'])
    return f.partial.raster(c,a)
def freeze():
    assert not (D/'protocol.json').exists();cases=sources();coords=[]
    files=[Path(__file__),f.ROOT/'scripts/benchmark/evaluate_ragan_native_tiling.py',f.ROOT/'scripts/benchmark/evaluate_matched_supervised_generators.py',f.OUT/'protocol.json',EXTRA/'protocol.json',f.OUT/'annotation_review.json']
    for c in cases:
        page=f.rgb(f.ROOT/c['source']);assert f.sha(f.ROOT/c['source'])==c['source_sha256'];files.append(f.ROOT/c['source'])
        coords.append(dict(case=c['id'],coordinates={m:tiles(page,c['roi'],off) for m,off in METHODS.items()}))
    f.dump(D/'protocol.json',dict(utc=datetime.now(timezone.utc).isoformat(),cases=cases,coordinates=coords,
      files={str(p.relative_to(f.ROOT)):f.sha(p) for p in files},conditions=f.NAMES[1:],grids=METHODS,
      primary='tiled0',secondary='tiled16; report all results, never choose per-case grid',
      status='Exploratory followup planned after full512 real-transfer readout; prior tiling algorithm unchanged',
      no_weight_updates=True,no_label_changes=True,invalid_R07_excluded=True,conditional_R02_retained_separately=True,
      inference='native256, stride128, page-fixed grid independent of annotation, final edges included, per-tile RGB/Otsu/Sobel, contiguous NCHW batch1, Hann floor.05 probability average<.5',
      interpretation='Changes inference dimension, context and auxiliary generation scope together; not a pure causal isolation of normalization or architecture.'))
def check():
    p=json.loads((D/'protocol.json').read_text());f.load()
    for n,h in p['files'].items():assert f.sha(f.ROOT/n)==h
    return p
def run(seed,device_name):
    p=check();dest=D/f'seed{seed}.json';assert not dest.exists()
    torch.set_num_threads(2);device=torch.device(device_name);torch.cuda.set_device(device);rows=[];states=[]
    for kind in ['unet','ragan']:
        name=f'{kind}_seed{seed}';model=f.make_model(kind).to(device);payload=torch.load(f.MODELS/name/'step_03024.pt',map_location='cpu',weights_only=False);model.load_state_dict(payload['model']);del payload;model.eval();before=f.digest_state(model);adapter=ThreeInput(model,kind).eval()
        for c in p['cases']:
            page=f.rgb(f.ROOT/c['source']);labels,paths=label(c)
            for grid,off in p['grids'].items():
                mean,std,seams,split,stats=tiled_probability(adapter,page,c['roi'],device,off);pred=mean<.5;assert np.isfinite(mean).all()
                target=D/'masks'/name/grid/f"{c['id']}.png";f.save(target,pred)
                rows.append(dict(case=c['id'],method=name,grid=grid,family=c['family'],book=c['book'],secondary=c['id']=='R02',prediction_sha256=f.sha(target),**stats,**f.partial.measure(pred,labels,paths)))
            print(name,c['id'],[(r['grid'],r['ink_r2'],r['nonink_false']) for r in rows[-2:]],flush=True)
        assert before==f.digest_state(model);states.append(dict(method=name,unchanged_state_sha256=before));del adapter,model;torch.cuda.empty_cache()
    f.dump(dest,dict(protocol_sha256=f.sha(D/'protocol.json'),rows=rows,states=states))
def report():
    p=check();rows=[]
    for s in [0,1]:
        q=json.loads((D/f'seed{s}.json').read_text());assert q['protocol_sha256']==f.sha(D/'protocol.json');rows+=q['rows']
    baseline=json.loads((f.OUT/'reviewed_results.json').read_text())['rows'];verified=[];figures=[]
    for r in rows:
        pred=np.asarray(Image.open(D/'masks'/r['method']/r['grid']/f"{r['case']}.png"))==0;c=next(c for c in p['cases'] if c['id']==r['case']);labels,_=label(c)
        dist=ndi.distance_transform_edt(~pred) if pred.any() else np.full(pred.shape,np.inf)
        assert int((pred&(labels==1)).sum())==r['ink_exact'] and int(((dist<=2)&(labels==1)).sum())==r['ink_r2'] and int((pred&(labels==2)).sum())==r['nonink_false']
        mp=D/'masks'/r['method']/r['grid']/f"{r['case']}.png";assert f.sha(mp)==r['prediction_sha256'];verified.append(dict(case=r['case'],method=r['method'],grid=r['grid'],sha256=f.sha(mp)))
    lines=['# 同じ重みによる256分割の追試','', '512比較を見た後に計画した探索的追試。以前から固定していた256/stride128の格子0を主条件、格子16を感度確認に用い、重み・閾値・注釈は変更しなかった。全7領域・4重み・両格子を報告し、例ごとの格子選択はしない。R02は参考、R07は無効判定のため対象外。','', '墨はr2の中心線被覆、非文字は選択内部の黒化画素。推論寸法・周辺文脈・補助画像の計算範囲を一緒に変えるため、単独要因の因果分離ではない。','']
    for c in p['cases']:
        key=c['id'];lines+=['## '+key+' '+c['family'],'','| 手法 | 512墨 | 256格子0墨 | 格子16墨 | 512非文字黒 | 格子0非文字黒 | 格子16非文字黒 |','|---|---:|---:|---:|---:|---:|---:|']
        orig=f.rgb(EXTRA/'R08_original.png' if key=='R08' else f.OUT/'inputs'/f'{key}_original.png')
        howe_path=EXTRA/'masks/howe_fixed.png' if key=='R08' else f.OUT/'evaluation/howe_fixed/masks'/f'{key}.png'
        for name in p['conditions']:
            b=next(r for r in baseline if r['case']==key and r['method']==name);g=[next(r for r in rows if r['case']==key and r['method']==name and r['grid']==grid) for grid in ['tiled0','tiled16']]
            lines.append(f"| {f.TITLES[name]} | {b['ink_r2']}/{b['ink_n']} | {g[0]['ink_r2']}/{g[0]['ink_n']} | {g[1]['ink_r2']}/{g[1]['ink_n']} | {b['nonink_false']}/{b['nonink_n']} | {g[0]['nonink_false']}/{g[0]['nonink_n']} | {g[1]['nonink_false']}/{g[1]['nonink_n']} |")
            base_path=EXTRA/'masks'/f'{name}.png' if key=='R08' else f.OUT/'evaluation'/name/'masks'/f'{key}.png'
            items=[('原画像',orig),('Howe 固定',np.asarray(Image.open(howe_path))),('既存512 '+f.TITLES[name],np.asarray(Image.open(base_path)))]
            for r in g:items.append((r['grid']+f" 墨{r['ink_r2']}/{r['ink_n']} 非文字{r['nonink_false']}",np.asarray(Image.open(D/'masks'/name/r['grid']/f'{key}.png'))))
            target=D/'figures'/f'{key}_{name}_推論寸法比較.png';f.real.panel(target,key+' '+f.TITLES[name]+' / 原画像・固定条件',items);figures.append(str(target.relative_to(D)));lines+=['',f"[{f.TITLES[name]}の原寸比較](figures/{key}_{name}_推論寸法比較.png)",'']
    lines+=['','過去の自動スクリーニング済み資料・Codex暫定部分注釈という制約は元の比較と同じ。中心線被覆は線幅・文字形状・認識精度の完全な評価ではない。','']
    (D/'比較結果_256分割追試.md').write_text('\n'.join(lines))
    assert len(rows)==56
    f.dump(D/'results.json',dict(rows=rows,figures=figures,formal_accuracy=False))
    f.dump(D/'verification.json',dict(protocol_sha256=f.sha(D/'protocol.json'),independent_mask_rescore=verified,source_files_checked=len(p['files']),unchanged_model_states=True,figures={name:f.sha(D/name) for name in figures}))
    print('Verified56 readouts and28 original+method figures',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','report']);p.add_argument('--seed',type=int,choices=[0,1]);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.seed,a.device) if a.action=='run' else globals()[a.action]()
