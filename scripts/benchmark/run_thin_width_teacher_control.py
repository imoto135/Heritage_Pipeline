"""Versioned v4 partial teacher and matched v3/v4 targeted-fade comparison.

Only two training patch labels change: source-visible core pixels around pre-
existing thin center traces become valid ink. Validation and input RGB are fixed.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp_work/targeted_fade_crop_control_20261021'
WIDTH=ROOT/'tmp_work/makura_thin_width_candidate_20261021'
OUT=ROOT/'tmp_work/makura_teacher_v4_width_20261021'
PATCHES=('200017197_00103_1_p00','200017197_00214_1_p00')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def gray(p):return np.asarray(Image.open(p).convert('L'))
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))

def prepare():
    assert not OUT.exists(),f'Preserve existing version: {OUT}'
    OUT.mkdir(parents=True)
    before=[json.loads(x) for x in (BASE/'unet_v3_targeted_fade/train_manifest.jsonl').read_text().splitlines()]
    labels={};audit=[];figures=[]
    for patch in PATCHES:
        old=next(r['labels'] for r in before if r['patch_id']==patch)
        target=gray(ROOT/old['target']);valid=gray(ROOT/old['valid'])>0
        core_path=WIDTH/'masks'/(patch+'_core.png');core=gray(core_path)>0
        assert core.shape==target.shape==(512,512)
        assert not np.any(core&valid&(target!=0))
        newtarget=target.copy();newvalid=valid.copy()
        newtarget[core]=0;newvalid[core]=True
        folder=OUT/'labels'/patch;folder.mkdir(parents=True)
        tp=folder/'paper_target.png';vp=folder/'valid.png'
        Image.fromarray(newtarget).save(tp)
        Image.fromarray(newvalid.astype(np.uint8)*255).save(vp)
        info=dict(old,target=str(tp.relative_to(ROOT)),valid=str(vp.relative_to(ROOT)),
            target_sha256=sha(tp),valid_sha256=sha(vp),previous_target=old['target'],
            previous_valid=old['valid'],core_candidate_sha256=sha(core_path),
            source_review='Conservative source-visible core near confirmed centerline; automatic candidate visually reviewed, not expert GT',
            formal_gt=False)
        labels[patch]=info
        new=int((core&~valid).sum())
        assert new>0
        audit.append(dict(patch=patch,core_n=int(core.sum()),old_valid_ink=int((core&valid&(target==0)).sum()),
            newly_valid_ink=new,old_valid_paper=int((core&valid&(target==255)).sum()),
            target_pixels_changed=int((target!=newtarget).sum()),valid_pixels_changed=int((valid!=newvalid).sum()),
            source_core_sha256=sha(core_path),old_target_sha256=old['target_sha256'],
            old_valid_sha256=old['valid_sha256'],new_target_sha256=sha(tp),new_valid_sha256=sha(vp)))
        source=rgb(ROOT/next(r['clean'] for r in before if r['patch_id']==patch))
        display=np.repeat(newtarget[...,None],3,axis=2);display[~newvalid]=[225,70,155]
        beforedisplay=np.repeat(target[...,None],3,axis=2);beforedisplay[~valid]=[225,70,155]
        overlay=source.copy();overlay[core&~valid]=[0,170,255];overlay[core&valid]=[255,30,50]
        imgs=[source,beforedisplay,overlay,display]
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
        canvas=Image.new('RGB',(4*512+5*12,512+42+24),'#faf9f5');d=ImageDraw.Draw(canvas)
        for i,(im,name) in enumerate(zip(imgs,('原本','v3 暫定教師','青=追加墨芯 / 赤=既存中心','v4 暫定教師'))):
            x=12+i*(512+12);d.text((x,8),name,font=font,fill='#203442');canvas.paste(Image.fromarray(im),(x,42))
        f=OUT/'figures'/(patch+'_教師v3_v4.png');f.parent.mkdir(parents=True,exist_ok=True);canvas.save(f)
        figures.append(dict(path=str(f.relative_to(OUT)),sha256=sha(f)))
    for model,cond in [('unet','unet_v3'),('ragan','ragan_public_v3')]:
        src=BASE/(cond+'_targeted_fade');dst=OUT/(cond+'_v4teacher');dst.mkdir()
        revised=[]
        for line in (src/'train_manifest.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['patch_id'] in labels:r['labels']=labels[r['patch_id']]
            r['teacher_version']='v4 source-visible core in two train patches; remaining provisional v3'
            revised.append(r)
        (dst/'train_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in revised))
        for name in ('val_manifest.jsonl','adapter_verification.json','code_hashes.json'):
            shutil.copy2(src/name,dst/name)
        protocol=json.loads((src/'label_protocol.json').read_text())
        protocol['comparison_training_teacher']='v4 source-visible core in two training patches'
        protocol['validation_teacher']='v2 fixed, exactly as paired v3 baseline'
        protocol['core_candidate_protocol_sha256']=sha(WIDTH/'protocol.json')
        protocol['formal_accuracy']=False
        dump(dst/'label_protocol.json',protocol)
        dump(dst/'condition.json',dict(model=model,training_teacher='v4 width candidate',
            paired_baseline=str(src.relative_to(ROOT)),same_train_input_rgb=True,same_crop_schedule=True,
            same_validation=True,seed=20261021,steps=432,batch_size=4,formal_accuracy=False))
        original=[json.loads(x) for x in (src/'train_manifest.jsonl').read_text().splitlines()]
        assert len(revised)==len(original)==864
        assert all(a['input_sha256']==b['input_sha256'] for a,b in zip(revised,original))
        assert sum(a['labels']['target_sha256']!=b['labels']['target_sha256'] for a,b in zip(revised,original))==18
        for row in revised:
            assert sha(ROOT/row['labels']['target'])==row['labels']['target_sha256']
            assert sha(ROOT/row['labels']['valid'])==row['labels']['valid_sha256']
    dump(OUT/'protocol.json',dict(purpose='Isolated v3/v4 teacher comparison with source-targeted fading',
        changed_training_patches=PATCHES,changed_training_rows=18,additional_valid_ink=sum(a['newly_valid_ink'] for a in audit),
        source_core_protocol_sha256=sha(WIDTH/'protocol.json'),
        validation='fixed v2 provisional labels and original RGB manifest',
        training='same source-targeted RGB and anchor-aware crop schedule as v3 baseline',
        seed=20261021,steps=432,batch_size=4,formal_accuracy=False,
        no_validation_trace_in_training=True,expert_verified=False))
    dump(OUT/'label_audit.json',dict(rows=audit,figures=figures))
    print('Prepared v4 partial teacher, newly valid ink',sum(a['newly_valid_ink'] for a in audit),flush=True)

def run(condition,device):
    script=ROOT/'scripts/benchmark/run_targeted_fade_crop_control.py'
    spec=importlib.util.spec_from_file_location('matched_crop_control',script)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.OUT=OUT
    module.run(condition,device)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--condition',choices=['unet_v3_v4teacher','ragan_public_v3_v4teacher'])
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        assert a.condition
        run(a.condition,a.device)
