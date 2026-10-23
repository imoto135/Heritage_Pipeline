"""Source-trace targeted fade diagnostic, separate from training and formal tests."""
from __future__ import annotations
import hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
STRESS=ROOT/'tmp_work/makura_damage_challenge_20261021'
OUT=ROOT/'tmp_work/targeted_thin_fade_20261021'
PATCH='200017197_00172_0_p00'
LEVELS=(0.,.45,.70,.85)
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from score_makura_heldout_thin_transfer import get_model,neural_mask

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def rgb(path):return np.asarray(Image.open(path).convert('RGB'))
def gray(a):return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def save_mask(path,pred):Image.fromarray(np.where(pred,0,255).astype(np.uint8)).save(path)

def audit_prior(annotations,records):
    output=[]
    for patch in ['200017197_00103_1_p00','200017197_00214_1_p00',PATCH]:
        points=np.asarray(Image.open(STRESS/'source_thin_review'/(patch+'_high_centers.png'))) >0
        assert points.any();y,x=np.where(points)
        baseline=next(r for r in records if r['patch_id']==patch and r['family']=='identity')
        source=rgb(ROOT/baseline['folder']/'input.png');sg=gray(source).astype(np.int16)
        for family in ('irregular_stain','carbon_spot','water_rim','thin_fade_grain'):
            r=next(v for v in records if v['patch_id']==patch and v['family']==family and v['strength']==.9)
            folder=ROOT/r['folder'];dam=rgb(folder/'input.png');dg=gray(dam).astype(np.int16)
            field=np.asarray(Image.open(folder/'damage_field.png'))/255
            delta=dg[y,x]-sg[y,x]
            output.append(dict(patch=patch,family=family,strength=.9,n=len(x),
                field_gt_point1=int((field[y,x]>.1).sum()),luma_change_ge2=int((np.abs(delta)>=2).sum()),
                luma_change_ge10=int((np.abs(delta)>=10).sum()),median_luma_change=float(np.median(delta)),
                input_sha256=r['input_sha256']))
    dump(OUT/'prior_damage_overlap_audit.json',dict(status='Input-only diagnostic at prior source-traced centers',rows=output,
        annotation_sha256=sha(STRESS/'thin_annotations_source_only.json')))
    return output

def metrics(mask,points):
    y,x=np.where(points);dist=distance_transform_edt(~mask)
    return dict(n=len(x),r0=int(mask[y,x].sum()),r1=int((dist[y,x]<=1).sum()),r2=int((dist[y,x]<=2).sum()),ink_pixels=int(mask.sum()))

def montage(path,title,items,detail=False):
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
    gap=15;head=48;labelh=30;cols=3;rows=(len(items)+cols-1)//cols
    if detail:
        # Fixed by source traces, before inspecting current method output.
        boxes=[(232,108,287,184),(344,207,406,263)]
        # Enlarge each crop uniformly, then pad it; stroke geometry must not be stretched.
        cropw,croph=248,304
        tilew,tileh=2*cropw+12,croph
        def convert(a):
            crops=[]
            for x0,y0,x1,y1 in boxes:
                q=a[y0:y1,x0:x1]
                if q.ndim==2:q=np.repeat(q[:,:,None],3,2)
                q=Image.fromarray(q.astype(np.uint8));q=q.resize((q.width*4,q.height*4),Image.Resampling.NEAREST);crops.append(q)
            t=Image.new('RGB',(tilew,tileh),'white')
            for i,q in enumerate(crops):
                t.paste(q,(i*(cropw+12)+(cropw-q.width)//2,(croph-q.height)//2))
            return t
    else:
        tilew=tileh=512
        def convert(a):
            if a.ndim==2:a=np.repeat(a[:,:,None],3,2)
            return Image.fromarray(a.astype(np.uint8))
    canvas=Image.new('RGB',(cols*(tilew+gap)+gap,head+rows*(tileh+labelh+gap)+gap),'#faf9f5');d=ImageDraw.Draw(canvas)
    d.text((gap,8),title,font=font,fill='#203442')
    for i,(label,a) in enumerate(items):
        x=gap+(i%cols)*(tilew+gap);y=head+(i//cols)*(tileh+labelh+gap)
        d.text((x,y),label,font=font,fill='#203442');canvas.paste(convert(a),(x,y+labelh))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def run():
    assert not OUT.exists(),'Frozen diagnostic already exists; preserve it.'
    OUT.mkdir(parents=True)
    records=json.loads((STRESS/'all_results.json').read_text())
    ann_path=STRESS/'thin_annotations_source_only.json'
    ann=json.loads(ann_path.read_text());prior=audit_prior(ann,records)
    select=next(p for p in json.loads((STRESS/'protocol.json').read_text())['selections'] if p['patch_id']==PATCH)
    source=rgb(ROOT/select['original']);assert sha(ROOT/select['original'])==select['original_sha256']
    points=np.asarray(Image.open(STRESS/'source_thin_review'/(PATCH+'_high_centers.png')))>0
    assert int(points.sum())==49
    paper=cv2.morphologyEx(source,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    sg=gray(source).astype(np.int16);pg=gray(paper).astype(np.int16)
    distance=distance_transform_edt(~points)
    field=np.exp(-.5*(distance/12.)**2).astype(np.float32);field[field<.005]=0
    assert np.all(field[points]==1)
    Image.fromarray(np.rint(field*255).astype(np.uint8)).save(OUT/'source_trace_field.png')
    classic=module(ROOT/'scripts/benchmark/audit_classical_limits.py','targeted_classic')
    howe=module(ROOT/'tmp_work/howe_diagnosis_20261008/experiment.py','targeted_howe')
    csc=module(ROOT/'modules/ddrm_npj_v2/scripts/csc.py','targeted_csc')
    csc._filter_red_candidate_by_component=classic.component_filter
    import torch
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.25,device)
    names=['unet_v2','unet_v3','ragan_public_v3','ra_nogan_residualga_v3']
    models={n:get_model(n,device) for n in names}
    all_rows=[];inputs=[];figures=[]
    for strength in LEVELS:
        key=f'alpha_{strength:.2f}';folder=OUT/'cases'/key;folder.mkdir(parents=True)
        q=field*strength
        damaged=np.rint(np.clip(source.astype(np.float32)*(1-q[...,None])+paper.astype(np.float32)*q[...,None],0,255)).astype(np.uint8)
        Image.fromarray(damaged).save(folder/'input.png')
        dgray=gray(damaged).astype(np.int16);y,x=np.where(points);delta=dgray[y,x]-sg[y,x]
        # The paper estimate is only a local reference, not truth or human readability.
        remaining=pg[y,x]-dgray[y,x]
        overlap=dict(level=strength,n=len(x),changed_ge2=int((np.abs(delta)>=2).sum()),changed_ge10=int((np.abs(delta)>=10).sum()),
            median_change=float(np.median(delta)),median_original_contrast=float(np.median(pg[y,x]-sg[y,x])),
            median_remaining_contrast=float(np.median(remaining)),min_remaining_contrast=int(remaining.min()),
            source_sha256=select['original_sha256'],input_sha256=sha(folder/'input.png'))
        inputs.append(overlap)
        g=howe.deep.base.gray(damaged)
        masks={'sauvola_w61_k0.20':howe.deep.base.sauvola(g,61,.2)}
        for thi,weight,name in [(.25,640,'howe_t025_c640'),(.4,160,'howe_fixed')]:
            terms,_=howe.terms(g,thi=thi)
            [(pre,metadata)]=howe.sweep(terms,[weight],OUT/'scratch'/key,name)
            pred,_=howe.post(pre);masks[name]=pred
        ace=classic.ace_chunked(cv2.cvtColor(damaged,cv2.COLOR_RGB2BGR))
        ok,jpeg=cv2.imencode('.jpg',ace,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
        ace=cv2.imdecode(jpeg,cv2.IMREAD_COLOR)
        with threadpool_limits(limits=1):masks['csc_v2']=csc.color_space_clustering(ace).black_text_mask>0
        for n,(model,kind,cksha) in models.items():masks[n]=neural_mask(model,kind,damaged,device)
        for name,mask in masks.items():
            assert mask.shape==(512,512)
            path=folder/(name+'.png');save_mask(path,mask)
            all_rows.append(dict(level=strength,method=name,mask_sha256=sha(path),**metrics(mask,points)))
        display=[('損傷前原本',source),('診断入力',damaged)]
        labels={'csc_v2':'CSC v2','sauvola_w61_k0.20':'Sauvola 61/.20','howe_t025_c640':'Howe c640 / thi.25',
            'howe_fixed':'Howe固定','unet_v2':'U-Net v2教師','unet_v3':'U-Net v3教師',
            'ragan_public_v3':'RA-GAN公開GA','ra_nogan_residualga_v3':'残差GA'}
        for name in ['csc_v2','sauvola_w61_k0.20','howe_t025_c640','howe_fixed']+names:
            display.append((labels[name],np.where(masks[name],0,255).astype(np.uint8)))
        if strength>0:
            for detail in (False,True):
                path=OUT/'figures'/(key+('_details' if detail else '_full')+'.png')
                montage(path,'確認済み薄線49点を狙った退色 / '+key,display,detail)
                figures.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path),detail=detail,level=strength))
        print(key,'center changed>=10',overlap['changed_ge10'],'median remaining contrast',overlap['median_remaining_contrast'],flush=True)
    dump(OUT/'protocol.json',dict(source_patch=PATCH,source_split='val',center_annotation_sha256=sha(ann_path),center_pixels=49,
        construction='Gaussian field distance to source-traced centers, sigma12 native pixels; RGB blend to morphology-close41 paper estimate',
        levels=LEVELS,source_annotation_used_to_position_damage=True,training_data=False,formal_evaluation=False,
        selected_after_previous_results=True,warning='Not DKDS, not blind or naturally damaged; source-only counterfactual diagnostic',
        neural='Last 432-step checkpoints, fixed 0.5 threshold',classical='Same local Sauvola/Howe/CSC adapters as previous stress suite; Howe skimage Canny approximation'))
    dump(OUT/'results.json',dict(prior_overlap=prior,inputs=inputs,rows=all_rows,figures=figures,
        source_original_sha256=select['original_sha256'],annotation_sha256=sha(ann_path),formal_accuracy=False))
    print('Completed',len(inputs),'input levels',len(all_rows),'method-level masks',flush=True)
    for r in all_rows:
        if r['level'] in (.7,.85):print(r['level'],r['method'],f"r0={r['r0']}/49 r2={r['r2']}/49",flush=True)

if __name__=='__main__':run()
