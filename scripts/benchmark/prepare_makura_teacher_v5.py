"""Add five reviewed source-visible thin cores to a separate provisional teacher.

Earlier v4 labels, synthetic inputs, and validation labels are untouched.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp_work/makura_teacher_v4_width_20261021'
CORES=ROOT/'tmp_work/makura_additional_thin_cores_20261021'
OUT=ROOT/'tmp_work/makura_teacher_v5_diverse_20261021'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def gray(p):return np.asarray(Image.open(p).convert('L'))
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))

def main():
    assert not OUT.exists(),f'Keep old teacher immutable: {OUT}'
    OUT.mkdir(parents=True)
    original=[json.loads(x) for x in (BASE/'unet_v3_v4teacher/train_manifest.jsonl').read_text().splitlines()]
    audit=json.loads((CORES/'audit.json').read_text())['rows']
    assert len(audit)==5 and len({r['group'] for r in audit})==5
    revised={};changes=[];figures=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    for row in audit:
        patch=row['patch'];prev=next(r['labels'] for r in original if r['patch_id']==patch)
        t=gray(ROOT/prev['target']);v=gray(ROOT/prev['valid'])>0
        core=gray(CORES/row['masks']['core']['path'])>0
        corridor=gray(CORES/row['masks']['corridor']['path'])>0
        assert np.all(core<=corridor)
        newt=t.copy();newv=v.copy();newv[corridor]=False;newt[core]=0;newv[core]=True
        assert np.array_equal(t[~core],newt[~core])
        assert np.array_equal(v[~corridor],newv[~corridor])
        folder=OUT/'labels'/patch;folder.mkdir(parents=True)
        tp=folder/'paper_target.png';vp=folder/'valid.png'
        Image.fromarray(newt).save(tp);Image.fromarray(newv.astype(np.uint8)*255).save(vp)
        info=dict(prev,target=str(tp.relative_to(ROOT)),valid=str(vp.relative_to(ROOT)),
            target_sha256=sha(tp),valid_sha256=sha(vp),previous_target=prev['target'],previous_valid=prev['valid'],
            source_core_sha256=row['masks']['core']['sha256'],
            source_review='Five new page groups: source-visible conservative thin cores; remaining width unknown; not expert GT',
            formal_gt=False)
        revised[patch]=info
        changes.append(dict(patch=patch,group=row['group'],trace_n=row['trace_n'],core_n=int(core.sum()),
            previous_valid_ink_in_core=int((core&v&(t==0)).sum()),
            previous_valid_paper_in_core=int((core&v&(t==255)).sum()),
            previous_unknown_in_core=int((core&~v).sum()),
            newly_valid_ink=int((core&(~v|(t!=0))).sum()),
            corridor_n=int(corridor.sum()),new_unknown_in_corridor=int((corridor&~newv).sum()),
            target_pixels_changed=int((t!=newt).sum()),valid_pixels_changed=int((v!=newv).sum()),
            old_target_sha256=prev['target_sha256'],old_valid_sha256=prev['valid_sha256'],
            new_target_sha256=sha(tp),new_valid_sha256=sha(vp)))
        source=rgb(ROOT/next(r['clean'] for r in original if r['patch_id']==patch))
        old_display=np.repeat(t[...,None],3,axis=2);old_display[~v]=[225,70,155]
        new_display=np.repeat(newt[...,None],3,axis=2);new_display[~newv]=[225,70,155]
        overlay=source.copy();overlay[core&~v]=[0,170,255];overlay[core&v]=[255,30,50]
        canvas=Image.new('RGB',(4*512+5*12,512+42+24),'#faf9f5');d=ImageDraw.Draw(canvas)
        for i,(im,name) in enumerate(zip((source,old_display,overlay,new_display),
            ('原本','旧v4暫定教師','青=追加墨芯 / 赤=既存文字','新v5暫定教師'))):
            X=12+i*(512+12);d.text((X,8),name,font=font,fill='#203442');canvas.paste(Image.fromarray(im),(X,42))
        f=OUT/'figures'/(patch+'_教師v4_v5.png');f.parent.mkdir(parents=True,exist_ok=True);canvas.save(f)
        figures.append(dict(path=str(f.relative_to(OUT)),sha256=sha(f)))
    for model,cond in [('unet','unet_v3'),('ragan','ragan_public_v3')]:
        base=BASE/(cond+'_v4teacher');dest=OUT/(cond+'_v5teacher');dest.mkdir()
        records=[]
        for line in (base/'train_manifest.jsonl').read_text().splitlines():
            record=json.loads(line)
            if record['patch_id'] in revised:record['labels']=revised[record['patch_id']]
            record['teacher_version']='v5 source-visible cores on 7 train groups; rest provisional v3'
            records.append(record)
        (dest/'train_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        for name in ('val_manifest.jsonl','adapter_verification.json','code_hashes.json'):
            (dest/name).write_bytes((base/name).read_bytes())
        protocol=json.loads((base/'label_protocol.json').read_text())
        protocol['comparison_training_teacher']='v5: five additional source-visible cores plus prior two v4 groups'
        protocol['additional_core_protocol_sha256']=sha(CORES/'protocol.json')
        protocol['validation_teacher']='v2 fixed';protocol['formal_accuracy']=False
        dump(dest/'label_protocol.json',protocol)
        dump(dest/'condition.json',dict(model=model,training_teacher='v5 provisional 7 groups',
            source_condition=str(base.relative_to(ROOT)),same_inputs_as_v4=True,
            same_validation=True,formal_accuracy=False))
        assert len(records)==864
        for record in records:
            assert sha(ROOT/record['labels']['target'])==record['labels']['target_sha256']
            assert sha(ROOT/record['labels']['valid'])==record['labels']['valid_sha256']
    dump(OUT/'protocol.json',dict(parent=str(BASE.relative_to(ROOT)),
        source_core_protocol_sha256=sha(CORES/'protocol.json'),added_groups=5,total_source_reviewed_train_groups=7,
        added_patches=5,changed_train_rows=45,prior_two_v4_patches_unchanged=True,
        validation_v2_unchanged=True,train_inputs_unchanged=True,
        purpose='Versioned provisional teacher candidate; no automatic claim of trainability or formal GT',
        expert_verified=False,formal_accuracy=False))
    dump(OUT/'label_audit.json',dict(rows=changes,figures=figures))
    print('Prepared v5: five additional train groups, newly valid ink',sum(r['newly_valid_ink'] for r in changes),flush=True)
    for r in changes:print(r['patch'],'new ink',r['newly_valid_ink'],'old paper corrected',r['previous_valid_paper_in_core'],flush=True)

if __name__=='__main__':main()
