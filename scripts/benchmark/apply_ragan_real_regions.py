"""Apply the aligned pilot checkpoints to preselected real damage ROIs."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'tmp_work/ragan_paper_aligned_comparison_20261021'
BASE=ROOT/'tmp_work/makura_supervised_smoke_20261021'
CLASSIC=ROOT/'tmp_work/classical_limit_audit_20261021'
PARTIAL=ROOT/'tmp_work/classical_partial_labels_20261021'
OUT=EXP/'real_application'
sys.path.insert(0,str(ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import ragan_pilot_support as support
support.ROOT=ROOT;support.REFERENCE=ROOT/'tmp_work/ra_gan_reference_20261021'
from ragan_pilot_support import auxiliaries,predict
from ragan_paper_ga import candidate_generator

def read_rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def read_mask(p):return np.asarray(Image.open(p).convert('L'))==0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def load_model(name,device):
    if name=='unet_v3': condition='unet_v3';key='unet';model=support.make_model('unet');kind='unet'
    elif name=='ragan_public_v3':condition='ragan_public_v3';key='ragan';model=support.make_model('ragan');kind='ragan'
    elif name=='ragan_paperga_v3':condition='ragan_paperga_v3';key='ragan';model=candidate_generator();kind='ragan'
    elif name=='ra_nogan_residualga_v3':condition=name;key='ra_no_gan';model=candidate_generator('residual_gate');kind='ragan'
    else:raise ValueError(name)
    checkpoint=EXP/condition/'models'/key/'last.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model']);model.to(device).eval()
    return model,kind,sha(checkpoint)

def origins(length):
    assert length>=512
    return sorted(set(list(range(0,length-511,384))+[length-512]))

def infer_roi(model,kind,page,roi,device):
    x0,y0,w,h=roi;total=np.zeros((h,w),np.float64);den=np.zeros((h,w),np.float64)
    weight=np.maximum(np.hanning(512),.05);weight=np.outer(weight,weight)
    for y in origins(page.shape[0]):
        for x in origins(page.shape[1]):
            l=max(x0,x);t=max(y0,y);r=min(x0+w,x+512);b=min(y0+h,y+512)
            if l>=r or t>=b:continue
            tile=page[y:y+512,x:x+512]
            q=auxiliaries(tile)
            inp=torch.from_numpy(q.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
            with torch.inference_mode():p=predict(model,kind,inp)[0,0].cpu().numpy()
            ww=weight[t-y:b-y,l-x:r-x]
            total[t-y0:b-y0,l-x0:r-x0]+=p[t-y:b-y,l-x:r-x]*ww
            den[t-y0:b-y0,l-x0:r-x0]+=ww
    assert np.all(den>0)
    return (total/den).astype(np.float32)

def panel(path,title,items):
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    w,h=items[0][1].shape[1],items[0][1].shape[0]
    cols=3;gap=14;head=52;labelh=32;rows=(len(items)+cols-1)//cols
    canvas=Image.new('RGB',(cols*(w+gap)+gap,head+rows*(h+labelh+gap)+gap),'#faf9f5');d=ImageDraw.Draw(canvas)
    d.text((gap,8),title+' / 原寸・補間なし',font=font,fill='#203442')
    boxes=[]
    for i,(label,mask_or_rgb) in enumerate(items):
        x=gap+(i%cols)*(w+gap);y=head+(i//cols)*(h+labelh+gap)
        d.text((x,y),label,font=font,fill='#203442')
        a=mask_or_rgb
        if a.ndim==2:a=np.repeat(a[:,:,None],3,2)
        canvas.paste(Image.fromarray(a.astype(np.uint8)),(x,y+labelh));boxes.append((x,y+labelh))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    saved=read_rgb(path)
    for (x,y),(_,a) in zip(boxes,items):
        if a.ndim==2:a=np.repeat(a[:,:,None],3,2)
        assert np.array_equal(saved[y:y+h,x:x+w],a.astype(np.uint8))

def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.3,device)
    cases=json.loads((CLASSIC/'protocol.json').read_text())['cases'];cases=[c for c in cases if c['id'] in ('D01','D02','D03','D04','T01','L07')]
    assert len(cases)==6
    names=['unet_v3','ragan_public_v3','ragan_paperga_v3','ra_nogan_residualga_v3']
    models={n:load_model(n,device) for n in names}
    scoring_spec=importlib.util.spec_from_file_location('source_partial_scoring',ROOT/'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring=importlib.util.module_from_spec(scoring_spec);sys.modules[scoring_spec.name]=scoring;scoring_spec.loader.exec_module(scoring)
    regions=json.loads((PARTIAL/'source_regions.json').read_text())
    annotations={r['id']:r for r in json.loads((PARTIAL/'annotations_source_only.json').read_text())['regions']}
    all_metrics=[];case_records=[];font_note='Development examples; partial source-only annotations, not expert or blind GT.'
    for case in cases:
        source=ROOT/case['source'];assert sha(source)==case['source_sha256']
        page=read_rgb(source);x,y,w,h=case['roi'];raw=page[y:y+h,x:x+w]
        outputs={}
        for name,(model,kind,cksha) in models.items():
            prob=infer_roi(model,kind,page,case['roi'],device);pred=prob<.5
            outputs[name]=pred
            f=OUT/'predictions'/case['id']/(name+'.png');f.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.where(pred,0,255).astype(np.uint8)).save(f)
            case_records.append(dict(case=case['id'],model=name,checkpoint_sha256=cksha,prediction_sha256=sha(f),
                ink_pixels=int(pred.sum()),paper_probability_median=float(np.median(prob))))
        base=CLASSIC/'crops'/case['id']
        classic_keys=['csc','sauvola_w61_k0.20','howe_fixed','howe_t025_c20']
        items=[('原画像',raw)]
        for key in classic_keys:
            p=base/(key+'.png');assert p.exists(),p
            outputs[key]=read_mask(p);items.append(({'csc':'CSC v2','sauvola_w61_k0.20':'Sauvola 61/.20','howe_fixed':'Howe固定','howe_t025_c20':'Howe調整B'}[key],np.where(outputs[key],0,255).astype(np.uint8)))
        for name in names:items.append((name,np.where(outputs[name],0,255).astype(np.uint8)))
        panel(OUT/'figures'/(case['id']+'_実損傷.png'),case['id']+' '+case['title'],items)
        case_regions=[r for r in regions if r['case']==case['id']]
        for region in case_regions:
            labels,paths=scoring.raster(region,annotations[region['id']]);rx,ry,rw,rh=region['roi']
            for name,pred in outputs.items():
                q=pred[ry-y:ry-y+rh,rx-x:rx-x+rw]
                assert q.shape==labels.shape
                all_metrics.append(dict(region=region['id'],model=name,**scoring.measure(q,labels,paths)))
    dump(OUT/'real_inference_verification.json',dict(cases=case_records,rows=all_metrics,
        role=font_note,threshold='neural paper probability <0.5',native_page_context='512 tiles, stride 384, feather average',
        classical_context='previously saved native full-page parameter outputs',formal_accuracy=False))
    print('Completed real-page regions:',len(cases),'cases;',len(all_metrics),'partial-label rows',flush=True)
    for row in all_metrics:
        if row['model'] in ('howe_fixed','howe_t025_c20','ragan_public_v3','ra_nogan_residualga_v3','unet_v3'):
            print(row['region'],row['model'],f"ink r2 {row['ink_r2']}/{row['ink_n']}",f"paper FP {row['nonink_false']}/{row['nonink_n']}",flush=True)

if __name__=='__main__':main()
