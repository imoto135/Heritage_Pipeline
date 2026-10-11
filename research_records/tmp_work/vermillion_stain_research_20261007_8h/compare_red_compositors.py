"""Fixed selectors, raw overlay versus background-relative color compositing.

This is a normalization control, not pigment recovery or a novelty claim.
The operation may remove observed tones if the paper field contains broad ink.
"""
import argparse
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,local_paper,file_hash
from render_selector_diagnostics import mask_path,LABELS
from render_actual_red_overlay import ZOOMS

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def paper_relative_overlay(original,base,mask):
    src_paper=local_paper(original);dst_paper=local_paper(base)
    # RGB ratio is observational normalization; gamma encoded JPEG is not
    # assumed to be a calibrated spectral/physical transmission measurement.
    relative=(original.astype(np.float32)+1)/(src_paper+1)
    normalized=np.clip(np.rint((dst_paper+1)*relative-1),0,255).astype(np.uint8)
    out=base.copy();out[mask]=normalized[mask]
    return out,src_paper,dst_paper

def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['saved_csc_v2_red','spatial_forest','region_spatial_forest'])
    p.add_argument('--roles',nargs='+',default=['development']);p.add_argument('--suffix',default='development_fixed');a=p.parse_args()
    dest=OUT/'red_compositor_controls'/a.suffix;dest.mkdir(parents=True,exist_ok=True)
    cases=json.loads((OUT/'cases_frozen.json').read_text())['cases'];font=ImageFont.truetype(FONT,17);small=ImageFont.truetype(FONT,14)
    records=[]
    for c in cases:
        if c['role'] not in a.roles or c['case_id'] not in ZOOMS:continue
        done=OUT/'sauvola_ddrm_base'/c['page']/'complete.json'
        if not done.exists():continue
        r=json.loads(done.read_text());assert file_hash(r['result'])==r['result_sha256']
        original=np.asarray(Image.open(ROOT/c['original']).convert('RGB'));base=np.asarray(Image.open(r['result']).convert('RGB'))
        available=[m for m in a.methods if mask_path(m,c['page']).exists()]+['all_pixels_normalization_control']
        xx,yy,_,_=c['ROI'];x,y,w,h=ZOOMS[c['case_id']];sl=np.s_[yy+y:yy+y+h,xx+x:xx+x+w]
        cw=max(w,300);rowh=h+42;sheet=Image.new('RGB',(4*cw,100+len(available)*rowh+70),'#faf9f5');d=ImageDraw.Draw(sheet)
        d.text((10,8),c['name']+'／朱マスクを固定した貼り戻し比較',font=font,fill='#20313a')
        d.text((10,36),'紙色との差を新しい紙の上へ移す。RGB値そのものの保持・化学的顔料復元ではない。',font=small,fill='#53636d')
        for j,label in enumerate(['原画像','朱の候補マスク','原画素の貼り戻し','紙色との差を貼り戻す']):d.text((j*cw+4,70),label,font=font,fill='#20313a')
        for idx,name in enumerate(available):
            if name=='all_pixels_normalization_control':mask=np.ones(base.shape[:2],bool);label='全画素の紙色補正（単純対照）'
            else:mask=np.asarray(Image.open(mask_path(name,c['page'])).convert('L'))>127;label=LABELS[name]
            raw=base.copy();raw[mask]=original[mask];normalized,sp,dp=paper_relative_overlay(original,base,mask)
            assert np.array_equal(normalized[~mask],base[~mask])
            variant=dest/c['page']/name;variant.mkdir(parents=True,exist_ok=True)
            output=variant/'paper_relative_result.png'
            if not output.exists():Image.fromarray(normalized).save(output)
            else:assert np.array_equal(np.asarray(Image.open(output)),normalized)
            yp=100+idx*rowh;d.text((10,yp+3),label,font=font,fill='#20313a')
            panels=[original[sl],np.repeat(mask[sl][:,:,None]*255,3,2).astype(np.uint8),raw[sl],normalized[sl]]
            for j,v in enumerate(panels):sheet.paste(Image.fromarray(v),(j*cw,yp+34))
            records.append(dict(case=c['case_id'],page=c['page'],selector=name,base_sha256=r['result_sha256'],
                output_sha256=file_hash(output),changed_outside_mask=0,
                normalized_ROI_vs_raw_mean_abs_change=float(np.abs(normalized[sl].astype(float)-raw[sl]).mean()),
                source_paper_estimate='input-derived closing31 at half resolution + Gaussian2, clipped not below source',
                preservation='relative RGB attenuation, not exact original RGB or expert reading',
                actual_black_DDRM_base=True,novelty_claim=False))
        d.text((10,100+len(available)*rowh+10),'広い筆記を紙色として推定すると消失する。単純な紙色補正だけで済む場合は分類器の貢献に数えない。',font=small,fill='#53636d')
        sheet.save(dest/(c['case_id']+'_マスク固定_貼り戻し方式.png'))
    (dest/'records.json').write_text(json.dumps(dict(records=records,source_sha256=file_hash(__file__),
        operation='dst_paper*(src_pixel/src_paper), epsilon1 and rounding on gamma-encoded RGB',
        no_expert_physical_or_reading_GT=True),ensure_ascii=False,indent=2))
    print('Compositor controls:',len(records),flush=True)

if __name__=='__main__':main()
