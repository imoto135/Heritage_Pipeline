"""Source-only partial-label freeze, then fixed-checkpoint real-page readout."""
import argparse, importlib.util, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from compare_v5_aligned_fade_control import sha,dump,rgb,save
from ragan_pilot_support import make_model
from audit_ragan_inference_context import digest_state
from evaluate_contact_stain_counterfactual import howe_mask
import evaluate_classical_partial_labels as partial
import apply_ragan_real_regions as real
OUT=ROOT/'tmp_work/frozen_real_transfer_20261022'
MODELS=ROOT/'tmp_work/matched_supervised_generators_20261022'
NAMES=['howe_fixed','unet_seed0','unet_seed1','ragan_seed0','ragan_seed1']
TITLES={'howe_fixed':'Howe 固定','unet_seed0':'U-Net 初期化1','unet_seed1':'U-Net 初期化2','ragan_seed0':'公開G 初期化1','ragan_seed1':'公開G 初期化2'}
def load():
    p=json.loads((OUT/'protocol.json').read_text())
    for name,h in p['frozen_files'].items():assert sha(ROOT/name)==h,name
    return p
def freeze():
    assert not (OUT/'protocol.json').exists()
    cases=json.loads((OUT/'selected_sources.json').read_text())['cases']
    annotations={r['id']:r for r in json.loads((OUT/'annotations_source_only.json').read_text())['regions']}
    eligible={r['page'] for r in json.loads((OUT/'refined_exposure_audit_v2.json').read_text())['eligible']}
    counts=[]
    for c in cases:
        assert c['page'] in eligible and c['book']!='200017197'
        assert sha(ROOT/c['source'])==c['source_sha256']
        x,y,w,h=c['roi'];a=rgb(ROOT/c['source'])[y:y+h,x:x+w]
        assert np.array_equal(a,rgb(OUT/'inputs'/f"{c['id']}_original.png"))
        labels,paths=partial.raster(c,annotations[c['id']]);f=OUT/'labels'/f"{c['id']}.png";f.parent.mkdir(exist_ok=True);Image.fromarray(labels).save(f)
        overlay=a.copy();overlay[labels==1]=[0,100,255];overlay[labels==2]=np.rint(.45*a[labels==2]+.55*np.array([0,200,80])).astype(np.uint8)
        real.panel(OUT/'annotations'/f"{c['id']}_原画像と部分注釈.png",c['id']+' 青=見える墨芯 緑=選択非文字 他=採点対象外', [('原画像',a),('Codex暫定注釈・専門家未確認',overlay)])
        counts.append(dict(case=c['id'],ink_n=int((labels==1).sum()),nonink_n=int((labels==2).sum()),paths=len(paths)))
    files=[OUT/'selected_sources.json',OUT/'annotations_source_only.json',OUT/'refined_exposure_audit_v2.json']
    files+=list((OUT/'labels').glob('*.png'))+list((OUT/'inputs').glob('*_original.png'))
    files += [MODELS/c/'step_03024.pt' for c in NAMES if c!='howe_fixed']
    source_names=['frozen_real_transfer.py','screen_frozen_real_transfer.py','apply_ragan_real_regions.py','evaluate_classical_partial_labels.py','ragan_pilot_support.py','evaluate_contact_stain_counterfactual.py']
    files += [ROOT/'scripts/benchmark'/n for n in source_names]
    files += [ROOT/'tmp_work/howe_diagnosis_20261008/experiment.py',ROOT/'tmp_work/howe_diagnosis_20261008/sweep_cli',ROOT/'tmp_work/classical_deep_review_20261008/run.py']
    for f in files:
        if f.suffix=='.py':
            d=OUT/'source_snapshot'/f.name;d.parent.mkdir(exist_ok=True);shutil.copy2(f,d)
    dump(OUT/'protocol.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),cases=cases,counts=counts,
      frozen_files={str(f.relative_to(ROOT)):sha(f) for f in files},
      methods=NAMES,checkpoint_step=3024,all_seeds_reported=True,no_checkpoint_or_threshold_selection=True,no_training=True,
      neural_inference='Previous fixed native512 whole-page grid, stride384 plus final edges, per-tile RGB/Otsu/Sobel, batch1, Hann floor.05, paper probability<.5',
      howe_inference='Full native page, gray RGB2GRAY/255, thi.4, tlo.1, sigma.6, C160, existing graph cut and postprocessing; crop after full-page processing',
      annotations='Codex source-only manual visible-ink centerlines and selected nontext interiors; unknown excluded; expert-unconfirmed partial labels, no denseGT',
      metrics='Exact and Euclidean-radius2 centerline coverage; nontext foreground count without tolerance, separately. No page F1, precision or character accuracy.',
      secondary_cases=['R02'],secondary_reason='Dark mark crossing printed frame is visibly nonletter, but physical origin and desired preservation unconfirmed. Conditional diagnostic only; exclude from primary pooled metrics.',
      provenance='New pages relative to existing inspected comparison cases in local audit; all pages had automatic source screening and CSC preprocessing. All books differ from training Makura. Books themselves are not new to project. Not a fully blind independent benchmark.',
      sampling='Source-only random overview16 + brown-ranked overview16 + per-book top2 overview22 + contact-ranked source crops20; purposeful challenge selection, not representative sampling',
      limitations=['Sparse provisional labels','Two initializations are not confidence intervals','No large charcoal-like new spot found; retain known D04 as separate development diagnostic','This is supervised public-derived generator, not GAN effect, DKDS or RA-GAN reproduction']))
    print('Frozen',counts,flush=True)
def run(method,device_name):
    p=load();assert method in NAMES
    dest=OUT/'evaluation'/method;assert not (dest/'results.json').exists()
    annotations={r['id']:r for r in json.loads((OUT/'annotations_source_only.json').read_text())['regions']}
    torch.set_num_threads(2);model=None;before=None
    if method!='howe_fixed':
        kind=method.split('_')[0];device=torch.device(device_name);torch.cuda.set_device(device)
        ck=MODELS/method/'step_03024.pt';payload=torch.load(ck,map_location='cpu',weights_only=False)
        assert payload['kind']==kind and payload['step']==3024
        model=make_model(kind).to(device);model.load_state_dict(payload['model']);del payload;model.eval();before=digest_state(model)
    else:
        spec=importlib.util.spec_from_file_location('frozen_transfer_howe',ROOT/'tmp_work/howe_diagnosis_20261008/experiment.py');howe=importlib.util.module_from_spec(spec);spec.loader.exec_module(howe)
    rows=[]
    for c in p['cases']:
        page=rgb(ROOT/c['source']);x,y,w,h=c['roi']
        if method=='howe_fixed':
            pred_page=howe_mask(page,dest/'scratch'/c['id'],'howe',howe);pred=pred_page[y:y+h,x:x+w]
            save(dest/'full_pages'/f"{c['id']}.png",pred_page)
        else:
            prob=real.infer_roi(model,kind,page,c['roi'],device)
            assert np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
            pred=prob<.5
        f=dest/'masks'/f"{c['id']}.png";save(f,pred)
        labels,paths=partial.raster(c,annotations[c['id']])
        assert np.array_equal(labels,np.asarray(Image.open(OUT/'labels'/f"{c['id']}.png")))
        row=dict(case=c['id'],book=c['book'],family=c['family'],method=method,secondary=c['id'] in p['secondary_cases'],prediction_sha256=sha(f),**partial.measure(pred,labels,paths));rows.append(row)
        print(method,c['id'],'r0/r2',row['ink_exact'],row['ink_r2'],'/',row['ink_n'],'nontext',row['nonink_false'],'/',row['nonink_n'],flush=True)
    if model is not None:assert before==digest_state(model)
    dump(dest/'results.json',dict(method=method,protocol_sha256=sha(OUT/'protocol.json'),started_after_freeze=True,completed_utc=datetime.now(timezone.utc).isoformat(),unchanged_model_state_sha256=before,rows=rows,formal_accuracy=False))
def report():
    p=load();rows=[]
    for m in NAMES:
        r=json.loads((OUT/'evaluation'/m/'results.json').read_text());assert r['protocol_sha256']==sha(OUT/'protocol.json');rows.extend(r['rows'])
    figures=[]
    for c in p['cases']:
        items=[('原画像',rgb(OUT/'inputs'/f"{c['id']}_original.png"))]
        for m in NAMES:
            row=next(r for r in rows if r['case']==c['id'] and r['method']==m)
            a=np.asarray(Image.open(OUT/'evaluation'/m/'masks'/f"{c['id']}.png"))
            items.append((TITLES[m]+f" 墨{row['ink_r2']}/{row['ink_n']} 汚{row['nonink_false']}/{row['nonink_n']}",a))
        f=OUT/'figures'/f"{c['id']}_原画像と固定手法比較.png";real.panel(f,c['id']+' '+c['family']+' 暫定部分注釈 r2/非文字残存',items);figures.append(str(f.relative_to(OUT)))
    lines=['# 新しい実画像での固定重み比較','', '全4重みの学習は行わず、14周時点を固定した。原画像だけで7領域の部分注釈を作成・凍結してから推論した。R02は黒い非文字付着物の由来が未確認のため、参考群として主集計から外す。','', '各行の墨は「厳密一致r0 / 半径2画素r2 / 注釈点数」、非文字は「黒化画素 / 選択画素」。文字を汚れごと黒くした結果を保持成功だけで評価しない。','']
    for c in p['cases']:
        lines+=['## '+c['id']+' '+c['family']+' '+c['page'],'', '| 手法 | 墨 r0 / r2 / 全点 | 非文字の黒化 / 全点 |','|---|---:|---:|']
        for m in NAMES:
            r=next(r for r in rows if r['case']==c['id'] and r['method']==m)
            lines.append(f"| {TITLES[m]} | {r['ink_exact']} / {r['ink_r2']} / {r['ink_n']} | {r['nonink_false']} / {r['nonink_n']} |")
        lines+=['',f"[原寸比較PNG](figures/{c['id']}_原画像と固定手法比較.png)、[原画像と部分注釈](annotations/{c['id']}_原画像と部分注釈.png)。",'']
    lines+=['## 条件別の主集計（R02を除く）','', '| 課題 | 手法 | 墨 r0 / r2 / 全点 | 非文字の黒化 / 全点 |','|---|---|---:|---:|']
    groups={g:[] for g in ['isolated_spot','contact_stain','faint_stroke','clean_control']}
    for g in groups:
        for m in NAMES:
            rs=[r for r in rows if r['family']==g and r['method']==m and not r['secondary']]
            sums={k:sum(r[k] for r in rs) for k in ['ink_exact','ink_r2','ink_n','nonink_false','nonink_n']}
            groups[g].append(dict(method=m,**sums))
            lines.append(f"| {g} | {TITLES[m]} | {sums['ink_exact']} / {sums['ink_r2']} / {sums['ink_n']} | {sums['nonink_false']} / {sums['nonink_n']} |")
    lines+=['','Codexによる暫定の疎な注釈であり、専門家GT・文書全体の精度・認識率ではない。過去の自動スクリーニングとCSC処理があり、全本が完全に未見ではない。既知の比較・目視記録ページは監査範囲内で除いた。ページ選定は課題探索で、無作為な母集団評価ではない。大きな炭状斑点は今回見つからず、既知D04を別の開発例として残す。','']
    (OUT/'比較結果_新しい実画像.md').write_text('\n'.join(lines))
    dump(OUT/'combined_results.json',dict(rows=rows,groups=groups,figures=figures,formal_accuracy=False))
    print('Saved7 original+5method PNGs',flush=True)
def verify():
    p=load();results=json.loads((OUT/'combined_results.json').read_text());checks=[]
    for r in results['rows']:
        pred=np.asarray(Image.open(OUT/'evaluation'/r['method']/'masks'/f"{r['case']}.png"))==0
        label=np.asarray(Image.open(OUT/'labels'/f"{r['case']}.png"));ink=label==1;non=label==2
        assert sha(OUT/'evaluation'/r['method']/'masks'/f"{r['case']}.png")==r['prediction_sha256']
        distance=ndi.distance_transform_edt(~pred) if pred.any() else np.full(pred.shape,np.inf)
        assert int((pred&ink).sum())==r['ink_exact']
        assert int(((distance<=2)&ink).sum())==r['ink_r2']
        assert int((pred&non).sum())==r['nonink_false']
        checks.append(dict(case=r['case'],method=r['method'],mask_sha256=r['prediction_sha256'],independent_EDT_r2=True))
    assert len(checks)==35
    dump(OUT/'verification.json',dict(protocol_sha256=sha(OUT/'protocol.json'),frozen_files_checked=len(p['frozen_files']),independently_rescored=checks,figures={f:sha(OUT/f) for f in results['figures']},annotation_counts=p['counts'],weights_unchanged=True,formal_accuracy=False))
    print('Verified35 masks and frozen sources/checkpoints/annotations',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run','report','verify']);a.add_argument('--method',choices=NAMES);a.add_argument('--device',default='cuda:0');args=a.parse_args()
    run(args.method,args.device) if args.action=='run' else globals()[args.action]()
