"""Render fixed validation examples for the isolated RA-GAN continuation."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'tmp_work/ragan_paper_aligned_comparison_20261021'
BASE=ROOT/'tmp_work/makura_supervised_smoke_20261021'
FIG=EXP/'figures'; OUT=EXP/'inference'
sys.path.insert(0,str(BASE/'code_snapshot'))
import ragan_pilot_support as support
support.ROOT=ROOT;support.REFERENCE=ROOT/'tmp_work/ra_gan_reference_20261021'
from ragan_pilot_support import auxiliaries, predict
from ragan_paper_ga import candidate_generator

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gray(p):return np.asarray(Image.open(p).convert('L'))
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def image_tensor(a,device):
    q=auxiliaries(a)
    return torch.from_numpy(q.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255

def main():
    torch.set_num_threads(2);device=torch.device('cuda:0')
    torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.3,device)
    rows=[json.loads(x) for x in (BASE/'val_manifest.jsonl').read_text().splitlines()]
    chosen=[]
    # Same fixed validation source/damage identity in each model comparison.
    for patch,family,strength in [('200017197_00172_0_p00','mixed',.65),('200017197_00171_1_p00','fade',.8),
                                  ('200017197_00170_0_p00','stain_overlap',.75)]:
        row=next(r for r in rows if r['patch_id']==patch and r['damage_family']==family and abs(r['damage_strength']-strength)<1e-6)
        chosen.append(row)
    models={}
    for name,condition,arch,model_key in [('unet_v2','unet_v2','unet','unet'),('unet_v3','unet_v3','unet','unet'),
                                ('ragan_public_v3','ragan_public_v3','public','ragan'),
                                ('ragan_paperga_v3','ragan_paperga_v3','paper','ragan'),
                                ('ra_nogan_paperga_v3','ra_nogan_paperga_v3','paper','ra_no_gan'),
                                ('ra_nogan_residualga_v3','ra_nogan_residualga_v3','residual','ra_no_gan')]:
        c=EXP/condition;ckpt=c/'models'/model_key/'last.pt'
        model=support.make_model('unet') if arch=='unet' else (candidate_generator('residual_gate') if arch=='residual' else (candidate_generator() if arch=='paper' else support.make_model('ragan')))
        payload=torch.load(ckpt,map_location='cpu',weights_only=False);model.load_state_dict(payload['model']);model.to(device).eval()
        models[name]=(model,'unet' if arch=='unet' else 'ragan',sha(ckpt))
    records=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    for row in chosen:
        a=rgb(ROOT/row['input']);target=gray(ROOT/row['labels']['target']);valid=gray(ROOT/row['labels']['valid'])>0
        samples=[('原本',rgb(ROOT/row['clean'])),('損傷入力',a),('v2暫定教師',np.repeat(target[:,:,None],3,2))]
        gt=target<128;figdir=FIG/row['id'];figdir.mkdir(parents=True,exist_ok=True)
        for name,(model,kind,ckhash) in models.items():
            with torch.inference_mode():p=predict(model,kind,image_tensor(a,device))[0,0].cpu().numpy()
            mask=p<.5; pred=np.where(mask,0,255).astype(np.uint8)
            Image.fromarray(pred).save(figdir/(name+'.png'))
            tp=int((mask&gt&valid).sum());fp=int((mask&~gt&valid).sum());fn=int((~mask&gt&valid).sum())
            records.append(dict(case=row['id'],model=name,checkpoint_sha256=ckhash,
                p_mean=float(p.mean()),p_median=float(np.median(p)),p_min=float(p.min()),p_max=float(p.max()),
                ink_f1_to_provisional=2*tp/max(1,2*tp+fp+fn),tp=tp,fp=fp,fn=fn,formal_accuracy=False))
            samples.append((name,pred))
        # Preserve native 512 pixels; images are displayed 2x nearest for comfortable inspection.
        scale=2;w=h=512;pad=16;head=48;lab=34;cols=2;rowsn=(len(samples)+1)//2
        canvas=Image.new('RGB',(cols*(w*scale+pad)+pad,head+rowsn*(h*scale+lab+pad)+pad),'#faf9f5');d=ImageDraw.Draw(canvas)
        d.text((pad,8),row['id']+' / '+row['damage_family'],font=font,fill='#223344')
        for i,(label,im) in enumerate(samples):
            x=pad+(i%cols)*(w*scale+pad);y=head+(i//cols)*(h*scale+lab+pad)
            d.text((x,y),label,font=font,fill='#223344')
            if im.ndim==2:im=np.repeat(im[:,:,None],3,2)
            tile=Image.fromarray(im).resize((w*scale,h*scale),Image.Resampling.NEAREST)
            canvas.paste(tile,(x,y+lab))
        out=FIG/(row['id']+'_v2検証.png');canvas.save(out)
    dump(EXP/'validation_prediction_diagnostics.json',dict(
        warning='Known validation pages and provisional labels; consistency only, not formal accuracy',rows=records,
        candidates='Paper GA candidate final feature I*W is an interpretation of underspecified paper output; failure cannot establish published method performance'))
    print('Rendered',len(chosen),'fixed native validation panels')
    for r in records:print(r['case'],r['model'],'F1',round(r['ink_f1_to_provisional'],5),'p median',round(r['p_median'],4))

if __name__=='__main__':main()
