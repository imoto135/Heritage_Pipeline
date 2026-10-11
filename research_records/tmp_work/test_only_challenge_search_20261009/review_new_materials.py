"""Package all transfer candidates, and exploratory ink/spot close-ups."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import validate_new_materials as trial
import run_ddrm_pilot as driver

OUT = trial.OUT
DETAILS = {
    'V01': [('本文の曲線', [420,1690,150,210])],
    'V02': [('本文の連続した曲線', [1460,1570,200,300]),
            ('本文の輪郭', [1620,1640,180,290])],
    'V03': [('本文の漢字', [840,2100,200,330]),
            ('本文の曲線', [1230,1100,230,270]),
            ('除去されなかった余白の点', [1630,220,110,100])],
    'V04': [('上部の小印字', [0,850,200,380]),
            ('下部の小印字', [0,2130,200,430]),
            ('本文の記名', [1280,2230,230,330])],
}


def readable_headers(image, titles, cid, panel_width):
    draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,image.width,57),fill='#f4f2ec')
    for i,title in enumerate(titles):
        text=cid+' '+title;size=24
        while size>10:
            font=ImageFont.truetype(trial.FONT,size)
            if draw.textbbox((0,0),text,font=font)[2]<=panel_width:break
            size-=1
        draw.text((12+i*(panel_width+12),12),text,font=font,fill='black')


def main():
    proto=json.loads((OUT/'protocol.json').read_text())
    complete=json.loads((OUT/'ddrm_complete.json').read_text())
    diag=json.loads((OUT/'mask_diagnostics.json').read_text())
    assert all(driver.digest(trial.ROOT/r['output'])==r['sha256'] for r in complete['results'])
    rows=[];overview=[];candidate_panels=[]
    for case,d in zip(proto['cases'],diag):
        p=case['page'];cid=case['id']
        raw=np.asarray(Image.open(trial.ROOT/case['source']).convert('RGB'))
        outputs=[np.asarray(Image.open(OUT/'outputs'/m/p/'result'/(p+'.png')).convert('RGB'))
                 for m in ['csc','howe','isolated']]
        masks=[np.asarray(Image.open(OUT/'masks'/m/p/'02_text_black.png'))<127
               for m in ['csc','howe','isolated']]
        removed=masks[1]&~masks[2]
        assert not (masks[2]&~masks[1]).any()
        page_panel=trial.panel(OUT/'full_figures'/(cid+'_全ページ比較.png'),[raw]+outputs,
            ['原画像','CSC → DDRM','Howe → DDRM','孤立点条件 → DDRM'],[0,0,raw.shape[1],raw.shape[0]],cid)
        page_panel.thumbnail((1800,1400),Image.Resampling.LANCZOS)
        page_panel.save(OUT/'full_figures'/(cid+'_全ページpreview.png'))
        overview.append(page_panel)
        for label,roi in DETAILS[cid]:
            x,y,w,h=roi;sl=np.s_[y:y+h,x:x+w]
            panel=trial.panel(OUT/'detail_figures'/(cid+'_'+label+'.png'),[raw]+outputs,
                ['原画像','CSC','Howe','孤立点条件'],roi,cid)
            readable_headers(panel,['原画像','CSC','Howe','点除去後'],cid,w)
            panel.save(OUT/'detail_figures'/(cid+'_'+label+'.png'))
            rows.append(dict(id=cid,label=label,xywh=roi,
                selection='exploratory after-mask review; not preselected accuracy samples',
                csc_mask_pixels=int(masks[0][sl].sum()),howe_mask_pixels=int(masks[1][sl].sum()),
                removed_mask_pixels=int(removed[sl].sum()),
                changed_howe_candidate_output_pixels=int(np.any(outputs[1][sl]!=outputs[2][sl],axis=2).sum()),
                dark_pixels_csc=int((outputs[0][sl].mean(2)<=128).sum()),
                dark_pixels_howe=int((outputs[1][sl].mean(2)<=128).sum()),
                dark_pixels_candidate=int((outputs[2][sl].mean(2)<=128).sum()),
                figure=driver.relative(OUT/'detail_figures'/(cid+'_'+label+'.png'))))
        for i,c in enumerate(d['removed_components']):
            component=Image.open(OUT/'ddrm_component_figures'/f'{cid}_{i:03d}.png').convert('RGB')
            readable_headers(component,['原画像','CSC','Howe','点除去後'],cid,220)
            component.save(OUT/'ddrm_component_figures'/f'{cid}_{i:03d}.png')
            candidate_panels.append(component)
    def stack(panels,path):
        width=max(p.width for p in panels);height=sum(p.height for p in panels)+16*(len(panels)-1)
        sheet=Image.new('RGB',(width,height),'#ddd');y=0
        for p in panels:sheet.paste(p,(0,y));y+=p.height+16
        sheet.save(path)
    stack(overview,OUT/'別資料4ページ_全ページ一覧.png')
    stack(candidate_panels,OUT/'全5除去候補_DDRM比較.png')
    driver.save_json(OUT/'review_verification.json',dict(details=rows,
        all_outputs_hash_verified=True,script_sha256=driver.digest(Path(__file__)),
        interpretation='Pixel counts are descriptive; neither dense GT nor accuracy rates'))
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
