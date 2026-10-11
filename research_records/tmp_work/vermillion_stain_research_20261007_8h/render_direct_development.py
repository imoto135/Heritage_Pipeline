"""Compare fixed direct literature outputs on development ROIs only."""
import argparse,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash
from render_actual_red_overlay import ZOOMS,FONT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--global-controls',action='store_true');args=parser.parse_args()
    dest=OUT/'direct_literature_review'/('development_global_controls' if args.global_controls else 'development_all');dest.mkdir(parents=True,exist_ok=True)
    methods={'stain':'StainRestorer・通常汚れ重み','docres_appearance_author':'DocRes・著者の全体寸法',
             'docres_appearance_native':'DocRes・原寸タイル','docres_binarization':'DocRes・二値化（色は保存しない）'}
    if args.global_controls:methods.update(stain_global256='Stain・全体256（拡大表示）',stain_global512='Stain・全体512（拡大表示）')
    records=[];font=ImageFont.truetype(FONT,17);small=ImageFont.truetype(FONT,14)
    for c in json.loads((OUT/'cases_frozen.json').read_text())['cases']:
        if c['case_id'] not in ['dev_06','dev_08','dev_10']:continue
        xx,yy,_,_=c['ROI'];x,y,w,h=ZOOMS[c['case_id']];sx,sy=xx+x,yy+y
        original=np.asarray(Image.open(ROOT/c['original']).convert('RGB'))
        panels=[('原画像',original)]
        for m,label in methods.items():
            folder=OUT/'direct_literature_baselines'/m/c['page'];p=folder/'result.png'
            assert (folder/'complete.json').exists()
            if m=='docres_binarization':
                a=np.asarray(Image.open(folder/'ink_mask.png').convert('L'));rgb=np.repeat((255-a)[:,:,None],3,2)
            else:rgb=np.asarray(Image.open(p).convert('RGB'))
            assert rgb.shape==original.shape;panels.append((label,rgb))
            records.append({'case':c['case_id'],'method':m,'result_sha256':file_hash(p) if p.exists() else None,'ROI':[sx,sy,w,h]})
        cw=max(w,360);nr=(len(panels)+2)//3;sheet=Image.new('RGB',(cw*3,nr*(h+65)+105),'#faf9f5');d=ImageDraw.Draw(sheet)
        d.text((10,8),c['name']+'／同じ原画像からの汎用修復',font=font,fill='#20313a')
        d.text((10,35),'色補正・色保存・二値化を区別。独立した読解の正解は未確定。',font=small,fill='#536470')
        for i,(label,a) in enumerate(panels):
            px=i%3*cw;py=65+i//3*(h+65);d.text((px+8,py+4),label,font=font,fill='#20313a')
            sheet.paste(Image.fromarray(a[sy:sy+h,sx:sx+w]),(px,py+32))
        d.text((10,sheet.height-35),'DocRes重みはミラー。著者寸法方式のFP16量子化差を記録済み。原寸の同一箇所。',font=small,fill='#536470')
        sheet.save(dest/(c['case_id']+'_原画像と汎用修復.png'))
    (dest/'verification.json').write_text(json.dumps({'records':records,'source_sha256':file_hash(__file__),'expert_reading_GT':False},ensure_ascii=False,indent=2))
    print('Development direct comparisons:',len(records),flush=True)

if __name__=='__main__':main()
