"""Show actual added fields and masks; never call these masks restorations."""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash
from render_selector_diagnostics import LABELS

def main():
    root=OUT/'counterfactual_stain_challenge';dest=root/'diagnostic_comparisons';dest.mkdir(exist_ok=True)
    rows=json.loads((root/'inputs_frozen.json').read_text())['records'];summary=[]
    methods=['csc_v2_red','region_spatial_forest','texture_paper_texture','texture_paper_texture_sparse','docres_foreground_chroma']
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',13)
    for r in rows:
        for name in methods:
            f=root/'predictions'/name/r['id']/'complete.json'
            if not f.exists():continue
            rec=json.loads(f.read_text());groups=rec['diagnostic']['points'];neg=rec['diagnostic']['selected_paper_region_added_support']
            summary.append(dict(id=r['id'],case=r['case'],method=name,shape=r['shape'],opacity=r['opacity'],direction=r['direction_name'],
                red_points=groups['observed_red_mark'],paper_points=groups['paper_or_brown_nonmark'],
                eligible_added_field_paper_pixels=sum(x['added_support_pixels'] for x in neg),
                before_selected_paper_pixels=sum(x['before_selected'] for x in neg),after_selected_paper_pixels=sum(x['after_selected'] for x in neg)))
        if r['case']!='dev_08' or r['opacity']!=.3 or r['direction_name']!='red_brown':continue
        x,y,w,h=r['ROI'];original=np.asarray(Image.open(ROOT/r['original']).convert('RGB'));altered=np.asarray(Image.open(OUT/r['input']).convert('RGB'))
        # Include the handwritten-like red field and lower paper/fold support.
        scale=min(1,600/h);cw=max(250,int(w*scale));hh=int(h*scale);panels=[('原画像',original[y:y+h,x:x+w]),('既知の茶色を追加',altered[y:y+h,x:x+w])]
        for name in methods:
            f=root/'predictions'/name/r['id']/'mark_mask.png'
            if not f.exists():continue
            mask=np.asarray(Image.open(f).convert('L'))[y:y+h,x:x+w];panels.append((LABELS.get(name,'DocRes二値化＋紙色差'),np.repeat(mask[:,:,None],3,2)))
        nr=(len(panels)+2)//3;sheet=Image.new('RGB',(cw*3,nr*(hh+55)+100),'#faf9f5');d=ImageDraw.Draw(sheet)
        d.text((10,8),'既知の追加場への反応／開発資料だけの診断',font=font,fill='#20313a')
        d.text((10,32),'白＝保持候補。朱と同じ色に近い折れを、朱と誤認するか。',font=small,fill='#536470')
        for i,(label,a) in enumerate(panels):
            xx=i%3*cw;yy=58+i//3*(hh+55);d.text((xx+5,yy),label,font=small,fill='#20313a')
            im=Image.fromarray(a);im=im.resize((int(w*scale),hh),Image.Resampling.NEAREST if i>=2 else Image.Resampling.LANCZOS);sheet.paste(im,(xx,yy+27))
        d.text((10,sheet.height-35),'本物のシミ・顔料の正解ではない。学習済み点を含む診断。マスクは修復画像ではない。',font=small,fill='#536470')
        sheet.save(dest/(r['id']+'_追加場とマスク.png'))
    (dest/'summary.json').write_text(json.dumps(dict(records=summary,source_sha256=file_hash(__file__),complete=all(len([r for r in summary if r['method']==m])==24 for m in methods),
        evaluation='Development known digital field only, not independent actual stain restoration accuracy'),ensure_ascii=False,indent=2))
    print('Counterfactual summaries',len(summary),flush=True)

if __name__=='__main__':main()
