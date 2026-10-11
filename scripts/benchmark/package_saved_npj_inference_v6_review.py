"""Auditable PNG comparisons of actual saved npj inference and a v6 candidate."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_npj_v6_comparison import ROOT, FONT_PATH, sha, dump, read, write

PILOT = ROOT / 'tmp_work/saved_npj_inference_v6_pilot_20261002'
OUT = ROOT / 'tmp_work/saved_npj_inference_v6_review_20261002'
NOTES = ROOT / 'tmp_work/saved_npj_inference_v6_review_notes_20261002.md'


def check(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    for rel,digest in manifest['dependencies'].items(): assert sha(ROOT / rel)==digest,rel
    for rel,digest in json.loads((folder / 'complete.json').read_text())['outputs'].items():
        assert sha(folder / rel)==digest,rel
    return manifest


def figure(images, labels, path, title, footers, overview=False):
    h,w = images[0].shape[:2]
    assert all(a.shape==(h,w,3) for a in images)
    if overview:
        ratio=min(540/w,900/h,1)
        dw,dh=round(w*ratio),round(h*ratio)
        thumbs=[cv2.resize(a,(dw,dh),interpolation=cv2.INTER_AREA) for a in images]
        scale_note='全体像は閲覧用縮小。拡大比較は原寸の整数倍率。'
    else:
        scale=min(4,max(1,min(660//w,1300//h)))
        dw,dh=w*scale,h*scale
        thumbs=[np.repeat(np.repeat(a,scale,axis=0),scale,axis=1) for a in images]
        scale_note=f'同位置を{scale}倍・最近傍表示。色補正なし。'
    margin,gap,top=24,24,120
    canvas=Image.new('RGB',(len(images)*dw+(len(images)-1)*gap+margin*2,dh+270),'#faf8f2')
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT_PATH,25);small=ImageFont.truetype(FONT_PATH,20)
    draw.text((margin,12),title,font=font,fill='black')
    draw.text((margin,49),scale_note,font=small,fill='black')
    for i,(a,label) in enumerate(zip(thumbs,labels)):
        x=margin+i*(dw+gap)
        draw.text((x,85),label,font=small,fill='black')
        canvas.paste(Image.fromarray(a[...,::-1]),(x,top))
    for i,note in enumerate(footers):
        draw.text((margin,dh+136+i*32),note,font=small,fill='black')
    canvas.save(path)
    saved=np.asarray(Image.open(path).convert('RGB'))
    for i,a in enumerate(thumbs):
        x=margin+i*(dw+gap)
        assert np.array_equal(saved[top:top+dh,x:x+dw],a[...,::-1])
    return dict(integer_nearest=not overview,display_size=[dw,dh],panels_exact=True)


def main(render_only):
    assert not (OUT/'complete.json').exists(),'Review already frozen'
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=check(PILOT)
    specs={r['id']:r for r in manifest['frames']}
    frames={};records=[]
    for fid,spec in specs.items():
        native=PILOT/'native'/fid
        arrays={name:read(native/(name+'.png')) for name in ['original','saved_npj','candidate','candidate_raw']}
        masks={name:read(native/(name+'.png'),True) for name in ['edit','extra','hold','red','protect']}
        x,y,w,h=spec['frame_xywh']
        original=read(ROOT/spec['original'])[y:y+h,x:x+w]
        assert np.array_equal(original,arrays['original'])
        assert np.array_equal(arrays['saved_npj'],read(ROOT/spec['saved_npj']))
        assert sha(ROOT/spec['saved_npj'])==manifest['dependencies'][spec['saved_npj']]
        expected=arrays['original'].copy();expected[masks['edit']]=arrays['candidate_raw'][masks['edit']]
        assert np.array_equal(expected,arrays['candidate'])
        assert not (np.any(arrays['candidate']!=original,axis=2)&~masks['edit']).any()
        for name in ['hold','red','protect']:
            assert not (masks['edit']&masks[name]).any()
            assert np.array_equal(arrays['candidate'][masks[name]],original[masks[name]])
        frames[fid]=(arrays,masks)
        figure([arrays[k] for k in ['original','saved_npj','candidate']],
               ['原画像',spec['baseline_label'],'新候補：固定v6・汚れ編集のみ'],
               OUT/('overview_'+fid+'.png'),spec['page']+'：実際の保存npj出力との比較',
               ['npjは保存推論結果をそのまま使用。新候補は文字再生成を行わず原筆画を保持する。',
                '復元方針を含む方式の比較。マスクだけの効果・同一ノイズの比較ではありません。',
                '黒さ・読みやすさ、字形の欠け、汚れの残りを分けて確認する。'],overview=True)
    for spec in manifest['regions']:
        fid=spec['frame'];arrays,masks=frames[fid];x,y,w,h=spec['xywh'];s=np.s_[y:y+h,x:x+w]
        cuts={name:a[s] for name,a in arrays.items()}
        assert all(a.shape==(h,w,3) for a in cuts.values())
        native=OUT/'native'/spec['id']
        for name,a in cuts.items():write(native/(name+'.png'),a)
        for name,a in masks.items():write(native/(name+'.png'),a[s])
        changed=int(np.any(cuts['candidate']!=cuts['original'],axis=2).sum())
        roles={k:int(v[s].sum()) for k,v in masks.items()}
        if roles['edit']==0:assert np.array_equal(cuts['candidate'],cuts['original'])
        rendering=figure([cuts[k] for k in ['original','saved_npj','candidate']],
                        ['原画像',specs[fid]['baseline_label'],'新候補：固定v6・汚れ編集のみ'],
                        OUT/(spec['id']+'.png'),spec['title']+' / '+specs[fid]['page'],
                        ['中央は実際の保存npj出力。npjの劣化マスク穴埋め診断で代用していません。',
                         f'新候補の編集 {roles["edit"]:,}・実変更 {changed:,}画素。原形保持と文字再生成の差も含む。',
                         '画素数は筆画GTや精度ではない。原筆画を保つだけで汚れも残す場合がある。'])
        ox,oy,_,_=specs[fid]['frame_xywh']
        records.append(dict(**spec,page=specs[fid]['page'],page_xywh=[ox+x,oy+y,w,h],
                            candidate_mask_pixels=roles,candidate_changed_pixels=changed,
                            candidate_exact_original=bool(np.array_equal(cuts['candidate'],cuts['original'])),
                            baseline_uses_real_saved_inference=True,**rendering))
    fid='upper_crop';arrays,masks=frames[fid]
    other=read(PILOT/'native'/fid/'saved_npj_other_reference.png')
    figure([arrays['original'],arrays['saved_npj'],other],
           ['原画像','保存npj：旧v1・selfref','関連保存出力：旧v1・extref'],
           OUT/'09_saved_reference_variants.png','背景参照の異なる保存版を別に表示',
           ['右列は同名実験の別保存版。全実行条件と入力の直接対応は未確認。',
            '背景参照や推論段階が異なる版を一つのnpj基準へ混ぜない。',
            '新候補の比較基準は中央のselfref版として固定。'],overview=True)
    if render_only:
        dump(OUT/'draft_diagnostics.json',records)
        print('Rendered 11 draft PNGs; review delivery not frozen.',flush=True)
        return
    assert NOTES.exists(),'Write visual findings before freezing the review'
    report='''# 保存済みの実際のnpj推論結果と固定v6編集候補

2026-10-02。従来npjを再設計・再推論せず、以前の実際の保存出力を固定基準にした。新候補側だけを推論した。原画像・保存npj・新候補を同位置で並べ、8拡大PNG、2全体像、参照違いの関連保存版1PNGを作成した。

## 比較の読み方

中央列は保存された文字再生成を含むnpj出力。前の劣化マスク穴埋め診断ではない。右列は固定v6の編集範囲だけDDRMを採用し、文字・朱・保留・編集外は原画素を保持する候補。文字再生成はしない。

これは最終処理方式の比較で、同じ復元器に対するマスクだけの効果ではない。背景参照・文字再生成・合成の差も含む。保存npjとのノイズ一致・実行ソースの完全再現は未確認。黒さや読みやすさの改善と、原筆画の形を保持することを別に評価する。新候補の原画像一致だけを復元精度や全体の優位としない。

## 保存基準と入力

200010454_00002_2は過去のmask_ablation_200010454_selfref/output_v1を使い、保存inputは元ページ[550,0,550,500]と全画素一致。保存黒・朱マスクも保存旧v1の同位置と完全一致した。extref版は関連版として別表示し、同一の基準へ混ぜない。

200018243_00003_2はfull_page_compare/ddrm_v1true_outを使い、同じ実験の保存inputが元ページ全体と全画素一致。過去のログも保存。v1trueは保存ディレクトリの版名で、実行時の全ソース・全条件を再現した保証ではない。入力と版・段階の分かる範囲を記録して実画像を再利用する。

2ページは既知開発例で、拡大位置は原画像・保存npjを見て提示用に選んだ。8ROIは独立評価や8冊の結果ではなく、重なる切り出しもある。図を含むページは使わない。別のhnsd保存cropは現在の原画像と平均絶対差26.9/255があったため、今回の同入力比較へ混ぜなかった。

## 新候補の固定条件

元解像度、固定split_direction v6編集マスク、ckpt_epoch0350、20段、観測ノイズsigma0=.05、seed999からページ・座標別乱数、128pxパッチ・32px重なり、反射パディング、元の保存入力と同じ推論範囲。マスク追加・しきい値調整・ACE/CSC再計算・新学習・余白拡張条件・紙色補正はない。新候補の途中の文字用ノイズ抑制は使わない。詳細は試作manifest.json。

## 目視所見

'''+NOTES.read_text()+'''

## 拡大画像と確認範囲

'''
    for r in records:
        report+=f'- [{r["title"]}]({r["id"]}.png)：新候補の編集 {r["candidate_mask_pixels"]["edit"]:,}、変更 {r["candidate_changed_pixels"]:,}画素。\n'
    report+='''
## 検証

保存npj出力の全画素・入力切り出し・マスク資格・ソースと重み・凍結条件のハッシュを確認。新候補は同マスク・同ノイズで再推論し画素一致、SVD対応と乱数対応を検証。保存生出力と編集マスクから最終合成を独立に再構成して完全一致。編集外と朱・保護・保留は原画像一致。比較PNGの全パネルは元画像の整数反復または明示した全体縮小と一致。ZIP全11PNGのCRCとバイト、全出力ハッシュを確認。人による筆画注釈・GT・全文書の優越は未確認。

[全11PNG ZIP](comparison_pngs.zip)、[検証記録](verification.json)、[全体：上部切り出し](overview_upper_crop.png)、[全体：本文ページ](overview_full_page.png)。
'''
    (OUT/'report.md').write_text(report)
    dependencies={str((PILOT/n).relative_to(ROOT)):sha(PILOT/n) for n in ['manifest.json','complete.json','verification.json','source_snapshot.py']}
    dependencies[str(NOTES.relative_to(ROOT))]=sha(NOTES)
    dependencies[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    dump(OUT/'manifest.json',dict(dependencies=dependencies,source_sha256=sha(Path(__file__)),primary_artifact='PNG',
         region_selection='developmental illustration, not independent assessment'))
    dump(OUT/'verification.json',dict(status='passed',records=records,actual_saved_npj=True,npj_rerun=False,
         full_frame_input_and_outputs_verified=True,independent_composite_verified=True,
         no_outside_edit_changes=True,panel_pixel_verification=True,pixel_GT=False,
         historical_matched_conditions=False))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    archive=OUT/'comparison_pngs.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')):z.write(p,p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==11
        import hashlib
        for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==sha(OUT/name)
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('Verified and packaged 11 PNGs with actual saved npj outputs.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true')
    main(parser.parse_args().render_only)
