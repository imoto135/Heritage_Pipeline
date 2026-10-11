"""Known added fields on real development pages; no clean-material claim."""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash

def field(shape,w,h,rng):
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    if shape=='narrow_fold':
        x=w*.6;slope=.08;scale=max(2,w*.012)
        return np.exp(-((xx-x-slope*(yy-h*.5))/scale)**2/2).astype(np.float32)
    v=np.zeros((h,w),np.float32)
    for dx,dy,rx,ry in [(.33,.3,.48,.28),(.68,.66,.4,.35)]:
        v=np.maximum(v,np.exp(-2*(((xx-w*dx)/(w*rx))**2+((yy-h*dy)/(h*ry))**2)))
    return v

def main():
    dest=OUT/'counterfactual_stain_challenge';protocol=json.loads((dest/'protocol_frozen.json').read_text())
    cases={r['case_id']:r for r in json.loads((OUT/'cases_frozen.json').read_text())['cases']};records=[]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    for cid in protocol['source_cases']:
        c=cases[cid];assert file_hash(ROOT/c['original'])==c['source_sha256'];rgb=np.asarray(Image.open(ROOT/c['original']).convert('RGB'));x,y,w,h=c['ROI']
        previews=[]
        for shape in protocol['shapes']:
            alpha=field(shape,w,h,np.random.default_rng(protocol['seed']))
            for color,direction in protocol['directions'].items():
                for opacity in protocol['opacity']:
                    rid=f'{cid}_{shape}_{color}_{round(opacity*100):02d}';d=dest/'inputs'/rid;d.mkdir(parents=True,exist_ok=True)
                    od=alpha[:,:,None]*opacity*np.asarray(direction,np.float32);altered=rgb.copy()
                    altered[y:y+h,x:x+w]=np.clip(np.rint(rgb[y:y+h,x:x+w].astype(np.float32)*np.exp(-od)),0,255).astype(np.uint8)
                    assert np.array_equal(altered[:y],rgb[:y]);Image.fromarray(altered).save(d/'input.png')
                    np.save(d/'added_alpha_ROI.npy',alpha);Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(d/'added_alpha_ROI_preview.png')
                    rec=dict(id=rid,case=cid,page=c['page'],book=c['book'],original=c['original'],original_sha256=c['source_sha256'],
                        ROI=c['ROI'],shape=shape,direction_name=color,direction=direction,opacity=opacity,
                        input=str((d/'input.png').relative_to(OUT)),input_sha256=file_hash(d/'input.png'),
                        alpha=str((d/'added_alpha_ROI.npy').relative_to(OUT)),alpha_sha256=file_hash(d/'added_alpha_ROI.npy'),
                        supervision='Frozen tentative original points; known added attenuation only',original_is_not_clean_GT=True)
                    records.append(rec);previews.append((rid,altered[y:y+h,x:x+w]))
        scale=min(1,650/h);cw=max(270,int(w*scale));hh=int(h*scale)
        sheet=Image.new('RGB',(cw*3,(hh+55)*3+70),'#faf9f5');d=ImageDraw.Draw(sheet);d.text((8,8),cid+' 原画像＋既知の追加場／候補評価前',font=font,fill='#20313a')
        entries=[('original',rgb[y:y+h,x:x+w])]+previews
        for i,(label,a) in enumerate(entries):
            xx=i%3*cw;yy=55+i//3*(hh+55);d.text((xx+4,yy),label.replace(cid+'_',''),font=font,fill='#20313a');sheet.paste(Image.fromarray(a).resize((int(w*scale),hh),Image.Resampling.LANCZOS),(xx,yy+28))
        sheet.save(dest/(cid+'_原画像と追加場.png'))
    (dest/'inputs_frozen.json').write_text(json.dumps(dict(records=records,protocol_sha256=file_hash(dest/'protocol_frozen.json'),script_sha256=file_hash(__file__),
        no_real_stain_or_reading_GT=True,original_pages_all_development=True),ensure_ascii=False,indent=2));print('Counterfactual development inputs',len(records),flush=True)

if __name__=='__main__':main()
