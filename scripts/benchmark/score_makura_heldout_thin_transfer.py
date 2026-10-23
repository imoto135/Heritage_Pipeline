"""Check source-traced held-out thin strokes under clean and severe synthetic fade."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'tmp_work/ragan_paper_aligned_comparison_20261021'
STRESS=ROOT/'tmp_work/makura_damage_challenge_20261021'
PILOT=ROOT/'tmp_work/makura_supervised_pilot_20261021'
OUT=EXP/'heldout_thin_transfer'
PATCH='200017197_00172_0_p00'
sys.path.insert(0,str(ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import ragan_pilot_support as support
support.ROOT=ROOT;support.REFERENCE=ROOT/'tmp_work/ra_gan_reference_20261021'
from ragan_pilot_support import auxiliaries,predict
from ragan_paper_ga import candidate_generator
from evaluate_classical_partial_labels import ordered_line

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def gray(p):return np.asarray(Image.open(p).convert('L'))
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def get_model(name,device):
    spec={
      'unet_v2':('unet_v2','unet',lambda:support.make_model('unet'),'unet'),
      'unet_v3':('unet_v3','unet',lambda:support.make_model('unet'),'unet'),
      'ragan_public_v3':('ragan_public_v3','ragan',lambda:support.make_model('ragan'),'ragan'),
      'ragan_paperga_v3':('ragan_paperga_v3','ragan',lambda:candidate_generator(),'ragan'),
      'ra_nogan_residualga_v3':('ra_nogan_residualga_v3','ra_no_gan',lambda:candidate_generator('residual_gate'),'ragan'),
    }
    condition,key,factory,kind=spec[name];model=factory();path=EXP/condition/'models'/key/'last.pt'
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);model.to(device).eval()
    return model,kind,sha(path)

def neural_mask(model,kind,input_rgb,device):
    a=auxiliaries(input_rgb);x=torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
    with torch.inference_mode():p=predict(model,kind,x)[0,0].cpu().numpy()
    return p<.5

def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.25,device)
    source_map={x['patch_id']:x for x in json.loads((STRESS/'protocol.json').read_text())['selections']}
    source=source_map[PATCH];clean=rgb(ROOT/source['original']);assert clean.shape==(512,512,3)
    annotations=json.loads((STRESS/'thin_annotations_source_only.json').read_text())['paths']
    paths=[p for p in annotations if p['patch_id']==PATCH and p['confidence']=='high']
    point_records=[];path_points={}
    for path in paths:
        points=ordered_line(path['xy']);path_points[path['id']]=points
        point_records.extend([(path['id'],int(x),int(y)) for x,y in points])
    # Center pixels are sparse confirmed traces, not a dense character mask.
    assert len(point_records)>0
    cases={}
    all_results=json.loads((STRESS/'all_results.json').read_text())
    for family,strength,label in [('identity',0.0,'clean_identity'),('thin_fade_grain',0.90,'thin_fade_grain_0.90')]:
        r=next(x for x in all_results if x['patch_id']==PATCH and x['family']==family and abs(x['strength']-strength)<1e-8)
        cases[label]=(ROOT/r['folder'],r)
    methods=['csc_v2','sauvola_w61_k0.20','howe_t0.250_c640.000','unet_v2','unet_v3','ragan_public_v3','ragan_paperga_v3','ra_nogan_residualga_v3']
    models={m:get_model(m,device) for m in methods if m.startswith(('unet','ragan','ra_nogan'))}
    results=[];figures=[]
    labels={'csc_v2':'CSC v2','sauvola_w61_k0.20':'Sauvola 61/.20','howe_t0.250_c640.000':'Howe c640 / thi.25',
      'unet_v2':'U-Net v2教師','unet_v3':'U-Net v3教師','ragan_public_v3':'RA-GAN公開GA',
      'ragan_paperga_v3':'論文式 GA: I×W','ra_nogan_residualga_v3':'残差GA: I×(1+W)'}
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
    for case,(folder,record) in cases.items():
        inp=rgb(folder/'input.png');assert inp.shape==(512,512,3)
        masks={}
        for name in ('csc_v2','sauvola_w61_k0.20','howe_t0.250_c640.000'):
            masks[name]=gray(folder/(name+'.png'))==0
        for name,(model,kind,checkpoint_hash) in models.items():masks[name]=neural_mask(model,kind,inp,device)
        for method,mask in masks.items():
            perpath=[];allpts=np.asarray([(x,y) for _,x,y in point_records],dtype=int);xx,yy=allpts[:,0],allpts[:,1]
            from scipy.ndimage import distance_transform_edt
            dist=distance_transform_edt(~mask)
            for pid,points in path_points.items():
                x,y=points[:,0],points[:,1];perpath.append(dict(path=pid,n=len(points),hit_r0=int(mask[y,x].sum()),
                    hit_r1=int((dist[y,x]<=1).sum()),hit_r2=int((dist[y,x]<=2).sum()),
                    max_missed_run_r2=_max_missed_run(dist[y,x]>2)))
            results.append(dict(case=case,method=method,n=len(allpts),hit_r0=int(mask[yy,xx].sum()),
                hit_r1=int((dist[yy,xx]<=1).sum()),hit_r2=int((dist[yy,xx]<=2).sum()),paths=perpath,
                input_sha256=sha(folder/'input.png'),method_sha256=sha(folder/(method+'.png')) if method.startswith(('csc','sauvola','howe')) else None,
                checkpoint_sha256=models[method][2] if method in models else None,formal_accuracy=False))
        items=[('原本（教師元）',clean)]
        items.append(('損傷入力',inp))
        for method in methods:items.append((labels[method],np.where(masks[method],0,255).astype(np.uint8)))
        canvas=Image.new('RGB',(3*(512+14)+14,52+4*(512+34+14)+14),'#faf9f5');d=ImageDraw.Draw(canvas)
        d.text((14,8),PATCH+' / '+case+' / 薄線中心追跡',font=font,fill='#203442')
        positions=[]
        for i,(name,a) in enumerate(items):
            x=14+(i%3)*(512+14);y=52+(i//3)*(512+34+14);d.text((x,y),name,font=font,fill='#203442')
            if a.ndim==2:a=np.repeat(a[:,:,None],3,2)
            canvas.paste(Image.fromarray(a.astype(np.uint8)),(x,y+34));positions.append((x,y+34,a))
        dest=OUT/'figures'/(case+'.png');dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest)
        checked=rgb(dest)
        for x,y,a in positions:assert np.array_equal(checked[y:y+512,x:x+512],a)
        figures.append(dict(case=case,path=str(dest.relative_to(OUT)),sha256=sha(dest),shape=list(checked.shape)))
    dump(OUT/'thin_center_diagnostics.json',dict(status='Source-confirmed partial centerline diagnostics; not dense or expert GT',
      annotation_sha256=sha(STRESS/'thin_annotations_source_only.json'),source_sha256=source['original_sha256'],
      high_confidence_center_pixels=len(point_records),holdout='00172 patch not used for gradient updates; its v3 centers are validation-only',
      selection='known development data; not blinded',tolerance='Euclidean 0/1/2 pixels',rows=results,figures=figures))
    print('High-confidence held-out thin centers:',len(point_records),flush=True)
    for row in results:print(row['case'],row['method'],row['hit_r0'],row['hit_r1'],row['hit_r2'],flush=True)

def _max_missed_run(values):
    best=run=0
    for v in values:run=run+1 if v else 0;best=max(best,run)
    return best

if __name__=='__main__':main()
