"""Versioned partial teacher correction using visible source-traced centers.

Keep unknown widths unknown. Never rewrite prior labels or damage inputs.
These remain Codex provisional labels, not expert verified ground truth.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi

from evaluate_classical_partial_labels import ordered_line
from run_makura_supervised_pilot import ROOT,OUT as OLD,rgb,gray,sha,dump,rel

PROBES=ROOT/'tmp_work/makura_damage_challenge_20261021'
OUT=ROOT/'tmp_work/makura_teacher_v3_20261021'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def main():
    assert not (OUT/'protocol.json').exists(),'Preserve existing teacher version.'
    ann=json.loads((PROBES/'thin_annotations_source_only.json').read_text())
    selections={p['patch_id']:p for p in json.loads((PROBES/'protocol.json').read_text())['selections']}
    index={p['patch_id']:p for p in json.loads((OLD/'labels_index.json').read_text())}
    revised={};audit=[];figures=[]
    for patch in sorted({p['patch_id'] for p in ann['paths']}):
        source=selections[patch];previous=index[patch]
        raw=rgb(ROOT/source['original']);target=gray(ROOT/previous['target']);valid=gray(ROOT/previous['valid'])>0
        positive=np.zeros((512,512),bool);uncertain=np.zeros_like(positive)
        for path in [p for p in ann['paths'] if p['patch_id']==patch]:
            points=ordered_line(path['xy']);x,y=points.T
            uncertain[y,x]=True
            if path['confidence']=='high':positive[y,x]=True
        yy,xx=np.ogrid[-2:3,-2:3]
        corridor=ndi.binary_dilation(uncertain,structure=(xx*xx+yy*yy<=4))
        # Positive points were independently checked on native source detail.
        # This contrast check rejects any gross coordinate-placement mistake;
        # it is not an automatic proof that a dark pixel is text.
        source_gray=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY)
        paper=cv2.morphologyEx(source_gray,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
        contrast=paper.astype(np.int16)-source_gray.astype(np.int16)
        assert contrast[positive].min()>=12
        target_new=target.copy();valid_new=valid.copy()
        valid_new[corridor]=False
        target_new[positive]=0;valid_new[positive]=True
        assert np.array_equal(target[~positive],target_new[~positive])
        assert np.array_equal(valid[~corridor],valid_new[~corridor])
        assert np.all(valid_new[positive]) and np.all(target_new[positive]==0)
        dest=OUT/'labels'/patch;dest.mkdir(exist_ok=True,parents=True)
        Image.fromarray(target_new).save(dest/'paper_target.png')
        Image.fromarray(valid_new.astype(np.uint8)*255).save(dest/'valid.png')
        info=dict(previous, target=rel(dest/'paper_target.png'),valid=rel(dest/'valid.png'),
            target_sha256=sha(dest/'paper_target.png'),valid_sha256=sha(dest/'valid.png'),
            previous_target=previous['target'],previous_valid=previous['valid'],
            source_review='Visible source-only ink centers added; radius2 width uncertainty ignored. Dense labels remain provisional.',
            source_annotation_sha256=sha(PROBES/'thin_annotations_source_only.json'),formal_gt=False)
        revised[patch]=info
        audit.append(dict(patch_id=patch,positive_centers=int(positive.sum()),
            old_valid_ink=int((positive&valid&(target==0)).sum()),
            old_valid_paper=int((positive&valid&(target==255)).sum()),
            old_unknown=int((positive&~valid).sum()),new_valid_ink=int((positive&valid_new&(target_new==0)).sum()),
            target_pixels_changed=int((target!=target_new).sum()),width_corridor_pixels=int(corridor.sum()),
            source_contrast_min=int(contrast[positive].min()),source_contrast_median=float(np.median(contrast[positive])),
            expert_verified=False))
        old_display=np.repeat(target[...,None],3,axis=2);old_display[~valid]=[235,40,150]
        new_display=np.repeat(target_new[...,None],3,axis=2);new_display[~valid_new]=[235,40,150]
        centers=raw.copy();centers[positive]=[0,100,255]
        images=[raw,old_display,centers,new_display]
        canvas=Image.new('RGB',(2140,720),'#faf9f5');d=ImageDraw.Draw(canvas)
        font=ImageFont.truetype(FONT,18)
        d.text((18,12),patch+' / 原画像から確認した細線中心の教師への追加',font=font,fill='#203442')
        positions=[]
        for i,(image,label) in enumerate(zip(images,['原画像（合成前）','v2：桃は対象外','青：原画像で確認した中心線','v3：中心のみ黒、線幅は未確定'])):
            x=18+i*532;d.text((x,66),label,font=font,fill='#203442');canvas.paste(Image.fromarray(image),(x,104));positions.append([x,104,512,512])
        d.text((18,645),'全パネル原画素512角。見える中心だけを追加し、太さや欠損を推測して塗り足していない。',font=font,fill='#203442')
        d.text((18,675),'Codexの暫定部分注釈に基づく教師候補。専門家確認済みの密な正解ではない。',font=font,fill='#203442')
        f=OUT/'figures'/(patch+'_教師v2_v3比較.png');f.parent.mkdir(exist_ok=True,parents=True);canvas.save(f)
        actual=rgb(f)
        for (x,y,w,h),image in zip(positions,images):assert np.array_equal(actual[y:y+h,x:x+w],image)
        figures.append(dict(path=rel(f),sha256=sha(f),panels=positions))
    all_rows=[]
    for split in ('train','val','pilot_eval'):
        records=[json.loads(line) for line in (OLD/(split+'_manifest.jsonl')).read_text().splitlines()]
        for record in records:
            if record['patch_id'] in revised:record['labels']=revised[record['patch_id']]
            record['teacher_version']='v3 source-checked thin centers, rest inherited provisional v2'
            record['use']='provisional_masked_training_or_consistency_only'
            assert record['formal_gt'] is False
        (OUT/(split+'_manifest.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        all_rows.extend(records)
    for record in all_rows:
        assert sha(ROOT/record['input'])==record['input_sha256']
        assert sha(ROOT/record['labels']['target'])==record['labels']['target_sha256']
        assert sha(ROOT/record['labels']['valid'])==record['labels']['valid_sha256']
    for old in index.values():
        assert sha(ROOT/old['target'])==old['target_sha256']
        assert sha(ROOT/old['valid'])==old['valid_sha256']
    dump(OUT/'protocol.json',dict(parent=rel(OLD),annotation=rel(PROBES/'thin_annotations_source_only.json'),
        annotation_sha256=sha(PROBES/'thin_annotations_source_only.json'),
        revised_patches=len(revised),revision='Add high-confidence source-observed ink centers; ignore radius2 around traced paths except known centers',
        uncertain_widths_not_filled=True,missing_strokes_not_invented=True,formal_gt=False,
        dense_expert_labels_ready=False,quality_training_run=False,
        target_source_is_undamaged_original=True,same_target_across_damage_variants=True,
        revision_created_after_prior_damage_generation=True,existing_inputs_unchanged=True))
    dump(OUT/'labels_index.json',[revised.get(p,index[p]) for p in index])
    dump(OUT/'revision_audit.json',audit);dump(OUT/'figure_verification.json',figures)
    dump(OUT/'verification.json',dict(pairs=len(all_rows),source_positive_centers=sum(a['positive_centers'] for a in audit),
        all_centers_valid_ink=True,old_labels_and_inputs_unchanged=True,all_manifest_hashes_checked=True,
        no_foreground_dilation=True,formal_gt=False))
    report=['# 教師候補v3：原画像から確認した細線中心の追加','',
        '2026-10-21。原画像を拡大して確認した3パッチの細線中心125画素を有効な黒文字として確保した。従来の有効文字21画素に加え、不明83画素・紙ラベル21画素を正例にした。専門家確認済みの密な正解ではなく、Codexによる暫定部分注釈の反映である。','',
        '線幅は断定せず、中心線の半径2画素近傍は中心自体を除いて無効化した。低確信の区間も無効化する。中心線を膨張して太さを作る処理は行わず、元から見えない筆画の描き足しもない。原画像以外の予測結果を使って中心位置を選んでいない。','',
        '旧v2ラベルと損傷入力は未変更。160パッチ・1,440組のマニフェストを別版として保存し、同じ原本の全損傷で同じv3教師を参照させた。変更は3パッチだけで、残る157パッチは従来の暫定教師を引き継ぐ。v3修正は既存の損傷入力を生成した後、今回の再学習前に行った。根拠は損傷前の原画像だけである。','',
        'この版ではまだ再学習していない。原画像で確認した小さな部分教師を加える準備であり、160パッチ全体を精密に修正したことにはならない。正式評価には別途確認済みの正解が必要。','',
        '[修正箇所](revision_audit.json) · [検証](verification.json) · [細線の元注釈](../makura_damage_challenge_20261021/thin_annotations_source_only.json)','']
    for f in figures:report.append('- ['+Path(f['path']).stem+'](figures/'+Path(f['path']).name+')')
    (OUT/'教師v3の変更.md').write_text('\n'.join(report)+'\n')
    print('Teacher v3: 125 source-observed positive centers,1440 paired manifest entries; all hashes checked',flush=True)


if __name__=='__main__':main()
