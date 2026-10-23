"""Native-pixel pilot figures and explicitly provisional diagnostic scores."""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from threadpoolctl import threadpool_limits
import torch

from ragan_pilot_support import ROOT, auxiliaries, make_model, predict
from run_makura_supervised_pilot import OUT, SOURCE, dump, sha, rel, rgb, gray, rows

CLASSIC = ROOT/'tmp_work/classical_limit_audit_20261021'
PARTIAL = ROOT/'tmp_work/classical_partial_labels_20261021'
KINDS = ('unet','ra_no_gan','ragan')
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def save_mask(path,ink):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.where(ink,0,255).astype(np.uint8)).save(path)


def freeze():
    assert not (OUT/'comparison_protocol.json').exists()
    all_rows=[r for split in ('train','val','pilot_eval') for r in rows(split)]
    selected=[]
    # Match the source-only, fixed review queue; never select by model output.
    for split,patch in [('val','200017197_00172_0_p00'),('pilot_eval','200017197_00149_1_p00')]:
        for family,strength in [('stain_overlap',.75),('fade',.8),('mixed',.65)]:
            selected.append(next(r for r in all_rows if r['patch_id']==patch and
                r['damage_family']==family and r['damage_strength']==strength))
    selected.append(next(r for r in all_rows if r['patch_id']=='200017197_00341_1_p00' and
                         r['damage_family']=='mixed' and r['damage_strength']==.65))
    cases=json.loads((CLASSIC/'protocol.json').read_text())['cases']
    real=[c for c in cases if c['id'] in ('D01','D02','D03','D04','T01','L07')]
    assert len(real)==6
    dump(OUT/'comparison_protocol.json',dict(synthetic=selected,real=real,
        selection='Fixed prior review source patches, fixed damage strengths, and pre-existing real development cases',
        thresholds='Neural paper probability < .5 means black ink; no threshold tuning',
        checkpoints='Last, equal 432 steps, no real-example model selection',
        synthetic_context='All methods receive native 512x512 damaged input',
        real_context='Classical cached native full-page output. Neural native 512 overlapping tiles from full original; stride 384, feather average.',
        proxy_scores='Only consistency to partial provisional labels; not formal accuracy',
        known_development=True,expert_ground_truth=False))
    print('Frozen 7 synthetic and 6 known real development comparisons',flush=True)


def baselines():
    protocol=json.loads((OUT/'comparison_protocol.json').read_text())
    audit=load_module(ROOT/'scripts/benchmark/audit_classical_limits.py','pilot_classics')
    howe=load_module(ROOT/'tmp_work/howe_diagnosis_20261008/experiment.py','pilot_howe_comparison')
    csc=load_module(ROOT/'modules/ddrm_npj_v2/scripts/csc.py','pilot_csc_comparison')
    csc._filter_red_candidate_by_component=audit.component_filter
    cv2.setNumThreads(1)
    results=[]
    for row in protocol['synthetic']:
        start=time.monotonic(); folder=OUT/'synthetic_predictions'/row['id']
        dest=folder/'classical_metadata.json'
        if dest.exists():results.append(json.loads(dest.read_text()));continue
        assert sha(ROOT/row['input'])==row['input_sha256']
        image=rgb(ROOT/row['input']); g=howe.deep.base.gray(image)
        sauvola=howe.deep.base.sauvola(g,61,.2)
        terms,_=howe.terms(g,thi=.25)
        [(pre,metadata)]=howe.sweep(terms,[640],OUT/'scratch'/row['id'],'baseline')
        mask,_=howe.post(pre)
        save_mask(folder/'sauvola.png',sauvola);save_mask(folder/'howe.png',mask)
        bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR);ace=audit.ace_chunked(bgr)
        success,jpeg=cv2.imencode('.jpg',ace,[cv2.IMWRITE_JPEG_QUALITY,95]);assert success
        ace=cv2.imdecode(jpeg,cv2.IMREAD_COLOR)
        with threadpool_limits(limits=1):
            csc_mask=csc.color_space_clustering(ace).black_text_mask>0
        save_mask(folder/'csc.png',csc_mask)
        result=dict(id=row['id'],input_sha256=row['input_sha256'],seconds=time.monotonic()-start,
            sauvola={'window':61,'k':.2},howe={'thi':.25,'c':640,'canny':'local skimage approximation'},
            csc='Local v2 default, seeded ACE alpha5,200 samples, JPEG95',
            hashes={p.name:sha(p) for p in sorted(folder.glob('*.png'))})
        dump(dest,result);results.append(result);print('Synthetic baselines',row['id'],round(result['seconds'],1),flush=True)
    dump(OUT/'synthetic_baseline_verification.json',results)


def tensor(rgb_image,device):
    a=auxiliaries(rgb_image)
    return torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255


def tile_origins(length):
    assert length>=512
    return sorted(set(list(range(0,length-511,384))+[length-512]))


def infer_roi(model,kind,image,roi,device):
    x0,y0,w,h=roi; total=np.zeros((h,w),np.float64);den=np.zeros((h,w),np.float64)
    weights=np.maximum(np.hanning(512),.05)
    weights=np.outer(weights,weights)
    count=0
    for y in tile_origins(image.shape[0]):
        for x in tile_origins(image.shape[1]):
            left=max(x0,x);top=max(y0,y);right=min(x0+w,x+512);bottom=min(y0+h,y+512)
            if left>=right or top>=bottom:continue
            q=image[y:y+512,x:x+512]
            p=predict(model,kind,tensor(q,device))[0,0].cpu().numpy()
            qweight=weights[top-y:bottom-y,left-x:right-x]
            total[top-y0:bottom-y0,left-x0:right-x0]+=p[top-y:bottom-y,left-x:right-x]*qweight
            den[top-y0:bottom-y0,left-x0:right-x0]+=qweight
            count+=1
    assert den.min()>0
    return (total/den).astype(np.float32),count


def inference(args):
    torch.set_num_threads(2);device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.30,device)
    protocol=json.loads((OUT/'comparison_protocol.json').read_text())
    records=[]
    for kind in KINDS:
        checkpoint=OUT/'models'/kind/'last.pt'
        completed=json.loads((OUT/'models'/kind/'completion.json').read_text());assert completed['steps']==432
        payload=torch.load(checkpoint,map_location='cpu')
        model=make_model(kind);model.load_state_dict(payload['model']);model.to(device).eval()
        digest=sha(checkpoint)
        with torch.inference_mode():
            for row in protocol['synthetic']:
                folder=OUT/'synthetic_predictions'/row['id']
                p=predict(model,kind,tensor(rgb(ROOT/row['input']),device))[0,0].cpu().numpy()
                assert p.shape==(512,512) and np.isfinite(p).all()
                np.save(folder/(kind+'_paper_probability.npy'),p)
                save_mask(folder/(kind+'.png'),p<.5)
                records.append(dict(case=row['id'],kind=kind,checkpoint_sha256=digest,
                    prediction_sha256=sha(folder/(kind+'.png')),view='synthetic native 512'))
            for case in protocol['real']:
                assert sha(ROOT/case['source'])==case['source_sha256']
                image=rgb(ROOT/case['source']);p,ntiles=infer_roi(model,kind,image,case['roi'],device)
                folder=OUT/'real_predictions'/case['id'];folder.mkdir(parents=True,exist_ok=True)
                np.save(folder/(kind+'_paper_probability.npy'),p)
                save_mask(folder/(kind+'.png'),p<.5)
                records.append(dict(case=case['id'],kind=kind,checkpoint_sha256=digest,
                    prediction_sha256=sha(folder/(kind+'.png')),tiles=ntiles,view='real native page tiles'))
                print('Native inference',kind,case['id'],ntiles,'tiles',flush=True)
        del model,payload
        if device.type=='cuda':torch.cuda.empty_cache()
    dump(OUT/'inference_verification.json',records)


def panels(path,title,images,labels,columns=3,note=''):
    h,w=images[0].shape[:2];assert all(a.shape==(h,w,3) for a in images)
    gap=18;head=140;labelh=38;nr=(len(images)+columns-1)//columns
    canvas=Image.new('RGB',(columns*(w+gap)+gap,head+nr*(h+labelh+gap)+72),'#faf9f5')
    draw=ImageDraw.Draw(canvas)
    def text(xy,s,size=18):draw.text(xy,s,font=ImageFont.truetype(FONT,size),fill='#203442')
    text((18,10),title,25)
    text((18,49),'暫定教師による短期学習の診断。正式な精度・優劣の判定ではない。')
    text((18,77),'原画素・縮小なし。Howeは近似Cannyによるローカル移植。',16)
    text((18,103),'RA公開実装ではGA注意分岐が相殺。注意機構の有効性を検証した比較ではない。',16)
    positions=[]
    for i,(a,label) in enumerate(zip(images,labels)):
        x=gap+(i%columns)*(w+gap);y=head+(i//columns)*(h+labelh+gap)
        text((x,y),label);canvas.paste(Image.fromarray(a),(x,y+labelh));positions.append([x,y+labelh,w,h])
    text((18,canvas.height-55),note,16)
    text((18,canvas.height-30),'ニューラル出力の閾値は固定0.5。原本は損傷合成前の画像。',16)
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    actual=rgb(path)
    for (x,y,ww,hh),a in zip(positions,images):assert np.array_equal(actual[y:y+hh,x:x+ww],a)
    return dict(path=rel(path),sha256=sha(path),panels=positions,exact_pixels=True)


def render():
    protocol=json.loads((OUT/'comparison_protocol.json').read_text());figures=[]
    for row in protocol['synthetic']:
        folder=OUT/'synthetic_predictions'/row['id']
        target=rgb(ROOT/row['labels']['target']);valid=gray(ROOT/row['labels']['valid'])>0
        target[~valid]=[235,40,150]
        images=[rgb(ROOT/row['clean']),rgb(ROOT/row['input']),target]
        images += [rgb(folder/(kind+'.png')) for kind in ('csc','sauvola','howe')+KINDS]
        labels=['原本（合成前）','損傷入力','暫定教師：桃は学習対象外',
            'CSC v2 + ACE','Sauvola w61 / k.20','Howe c640 / thi.25',
            'U-Net（共通5ch入力）','RA生成器（GA相殺・GANなし）','RA-GAN公開版（GA相殺あり）']
        figures.append(panels(OUT/'figures'/(row['id']+'_合成比較.png'),row['split']+' / '+row['id'],images,labels,
            note='同じ入力・教師と432更新。細線の輪郭は未確認領域が多く、数値の一致だけで保持精度を判断しない。'))
    for case in protocol['real']:
        x,y,w,h=case['roi'];image=rgb(ROOT/case['source'])[y:y+h,x:x+w]
        folder=CLASSIC/'crops'/case['id'];neural=OUT/'real_predictions'/case['id']
        images=[image]+[rgb(folder/(key+'.png')) for key in ('csc','sauvola_w61_k0.20','howe_fixed','howe_t025_c20')]
        images += [rgb(neural/(kind+'.png')) for kind in KINDS]
        figures.append(panels(OUT/'figures'/(case['id']+'_実損傷比較.png'),case['id']+' / '+case['title'],images,
            ['原画像','CSC v2 + ACE','Sauvola w61 / k.20','Howe固定 c160 / thi.4',
             'Howe調整 c640 / thi.25','U-Net','RA生成器（GA相殺・GANなし）','RA-GAN公開版（GA相殺あり）'],columns=4,
            note='既知の開発例。古典法は全ページ処理、学習法は同じ全ページから原解像度の重複タイルを処理。'))
    dump(OUT/'figure_verification.json',figures)
    print('Rendered and verified',len(figures),'PNGs',flush=True)


def score_partial():
    scoring=load_module(ROOT/'scripts/benchmark/evaluate_classical_partial_labels.py','pilot_partial_scores')
    protocol=json.loads((OUT/'comparison_protocol.json').read_text())
    cases={c['id']:c for c in protocol['real']}
    regions=json.loads((PARTIAL/'source_regions.json').read_text())
    annotations={r['id']:r for r in json.loads((PARTIAL/'annotations_source_only.json').read_text())['regions']}
    result=[]
    for region in regions:
        labels,paths=scoring.raster(region,annotations[region['id']])
        case=cases[region['case']];x,y,w,h=region['roi'];bx,by,_,_=case['roi']
        inputs={kind:OUT/'real_predictions'/case['id']/(kind+'.png') for kind in KINDS}
        inputs.update({key:CLASSIC/'crops'/case['id']/(key+'.png') for key in
            ('csc','sauvola_w61_k0.20','howe_fixed','howe_t025_c20')})
        for kind,path in inputs.items():
            mask=gray(path)[y-by:y-by+h,x-bx:x-bx+w]==0
            assert mask.shape==labels.shape
            result.append(dict(region=region['id'],kind=kind,
                prediction_sha256=sha(path),formal_accuracy=False,**scoring.measure(mask,labels,paths)))
    dump(OUT/'real_partial_diagnostics.json',dict(
        status='Existing source-traced Codex provisional partial labels; known development, not expert/blinded/dense GT',
        annotation_sha256=sha(PARTIAL/'annotations_source_only.json'),
        metrics='Ink centerline coverage at multiple tolerances; selected nonink residuals separately',rows=result))
    synthetic=[]
    for row in protocol['synthetic']:
        target=gray(ROOT/row['labels']['target'])==0
        valid=gray(ROOT/row['labels']['valid'])>0
        folder=OUT/'synthetic_predictions'/row['id']
        for kind in ('csc','sauvola','howe')+KINDS:
            pred=gray(folder/(kind+'.png'))==0
            tp=int((pred&target&valid).sum());fp=int((pred&~target&valid).sum());fn=int((~pred&target&valid).sum())
            synthetic.append(dict(id=row['id'],kind=kind,split=row['split'],family=row['damage_family'],
                ink_f1_to_provisional=2*tp/max(1,2*tp+fp+fn),tp=tp,fp=fp,fn=fn,
                valid_pixels=int(valid.sum()),formal_accuracy=False))
    dump(OUT/'synthetic_consistency.json',dict(
        warning='Consistency to unreviewed three-method agreement regions, not expert GT or formal accuracy',rows=synthetic))
    print('Partial diagnostics',len(result),'method-region pairs',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','baselines','infer','render','score'])
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='baselines':baselines()
    elif args.action=='infer':inference(args)
    elif args.action=='render':render()
    else:score_partial()
