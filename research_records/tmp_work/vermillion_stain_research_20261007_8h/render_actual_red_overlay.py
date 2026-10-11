"""Original, native red mask, and genuine restoration using shared DDRM base."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash
from render_selector_diagnostics import mask_path,LABELS

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
ZOOMS={
 'dev_10':[10,740,490,450],
 'dev_08':[110,270,430,330],
 'dev_06':[30,90,330,320],
 'validation_00':[0,30,250,305],
 'supplemental_known_book_08':[45,35,150,180],
 'supplemental_known_book_09':[0,0,100,500],
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['saved_csc_v2_red','relative_overlap','spatial_forest'])
    p.add_argument('--roles',nargs='+',default=['development']);p.add_argument('--suffix',default='first_development');a=p.parse_args()
    cases=json.loads((OUT/'cases_frozen.json').read_text())['cases'];dest=OUT/'actual_restoration_comparisons'/a.suffix;dest.mkdir(parents=True,exist_ok=True)
    records=[];font=ImageFont.truetype(FONT,18);small=ImageFont.truetype(FONT,15)
    for case in cases:
        if case['role'] not in a.roles or case['case_id'] not in ZOOMS:continue
        done=OUT/'sauvola_ddrm_base'/case['page']/'complete.json'
        if not done.exists():continue
        rec=json.loads(done.read_text());assert file_hash(rec['result'])==rec['result_sha256']
        original=np.asarray(Image.open(ROOT/case['original']).convert('RGB'))
        base=np.asarray(Image.open(rec['result']).convert('RGB'));assert base.shape==original.shape
        xx,yy,ww,hh=case['ROI'];x,y,w,h=ZOOMS[case['case_id']];sx=xx+x;sy=yy+y;sl=np.s_[sy:sy+h,sx:sx+w]
        available=[m for m in a.methods if mask_path(m,case['page']).exists()]+['preserve_everything','preserve_nothing']
        cw=max(340,w);header=90;rowh=h+48;footer=76
        sheet=Image.new('RGB',(3*cw,header+len(available)*rowh+footer),'#faf9f5');d=ImageDraw.Draw(sheet)
        d.text((10,8),case['name']+'／同じ黒修復に朱候補だけを貼り戻す',font=font,fill='#20313a')
        d.text((10,34),'Sauvola黒＋DDRMを一度だけ計算。白＝元画素の貼り戻し。原ページ座標 '+str([sx,sy,w,h]),font=small,fill='#52616c')
        for j,label in enumerate(['原画像','朱の候補マスク','修復後の画像']):d.text((j*cw+8,64),label,font=font,fill='#20313a')
        for i,method in enumerate(available):
            if method=='preserve_everything':mask=np.ones(base.shape[:2],bool);label='全て貼り戻す（除去なし）'
            elif method=='preserve_nothing':mask=np.zeros(base.shape[:2],bool);label='朱を貼り戻さない'
            else:mask=np.asarray(Image.open(mask_path(method,case['page'])).convert('L'))>127;label=LABELS[method]
            out=base.copy();out[mask]=original[mask]
            changed=np.any(out!=base,2)
            assert not (changed&~mask).any()
            assert np.array_equal(out[mask],original[mask])
            method_dir=dest/case['page']/method;method_dir.mkdir(parents=True,exist_ok=True)
            # Only one native page per method; comparison crops refer to these exact pixels.
            full=method_dir/'result.png'
            if not full.exists():Image.fromarray(out).save(full)
            else:assert np.array_equal(np.asarray(Image.open(full)),out)
            yo=header+i*rowh;d.text((10,yo+4),label,font=font,fill='#20313a')
            panels=[original[sl],np.repeat(mask[sl][:,:,None]*255,3,2).astype(np.uint8),out[sl]]
            for j,array in enumerate(panels):sheet.paste(Image.fromarray(array),(j*cw,yo+38))
            records.append(dict(case=case['case_id'],method=method,page=case['page'],ROI_original=[sx,sy,w,h],
                base_sha256=rec['result_sha256'],output_sha256=file_hash(full),selected_ROI_pixels=int(mask[sl].sum()),
                changed_outside_selected_mask=0,selected_pixels_exact_original=True,
                genuine_frozen_black_DDRM=True,reading_GT=False))
        d.text((10,header+len(available)*rowh+12),'全て貼り戻す対照は原画像そのもの。朱が残っても、シミを除去できたことにはならない。',font=small,fill='#52616c')
        d.text((10,header+len(available)*rowh+36),'実物の読解・シミ・顔料の正解は未確定。学習試作の誤保持も表示する。',font=small,fill='#52616c')
        sheet.save(dest/(case['case_id']+'_原画像_マスク_修復.png'))
    (dest/'verification.json').write_text(json.dumps(dict(records=records,source_sha256=file_hash(__file__),
        red_overlay_changed_factor_only=True,full_pixel_or_reading_GT=False),ensure_ascii=False,indent=2))
    print('Actual restoration comparison variants:',len(records),flush=True)

if __name__=='__main__':main()
