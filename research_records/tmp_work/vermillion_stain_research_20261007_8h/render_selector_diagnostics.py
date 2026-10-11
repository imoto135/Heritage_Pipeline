"""PNG original/mask comparisons. Do not label mask visualizations restoration."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
LABELS=dict(csc_v1_red='CPU ACE＋CSC v1',csc_v2_red='CPU ACE＋CSC v2',saved_csc_v1_red='保存済みCSC v1',saved_csc_v2_red='保存済みCSC v2',hsv_loose='HSVを緩める',relative_chroma='紙色との差だけ',relative_ridge='紙色との差＋線応答',
    relative_pointwise='学習・画素分類',relative_overlap='学習・紙色＋空間',rgb_separated='学習・色分布を分離',rgb_overlap='学習・色分布を重ねる',
    relative_pair_aug='学習・色を変えた対',relative_pair_consistent='学習・対の一致も要求',
    color_logistic='暫定点・色の線形分類',color_forest='暫定点・色の森分類',spatial_logistic='暫定点・色と周囲／線形',spatial_forest='暫定点・色と周囲／森分類',
    region_color_logistic='紙領域も学習／色・線形',region_color_forest='紙領域も学習／色・森',region_spatial_logistic='紙領域も学習／空間・線形',region_spatial_forest='紙領域も学習／空間・森')
LABELS.update(texture_continue_control='学習・追加学習のみ',texture_paper_texture='学習・実物紙の粒を追加',texture_paper_texture_sparse='学習・紙の粒と暫定点')
LABELS.update(partial_sparse_control='追加試作・点のみの対照',partial_dense_native='追加試作・点と紙領域',partial_dense_native_brown='追加試作・点と紙領域＋茶色')
LABELS['partial_dense_native_photometry']='追加試作・点と紙領域＋撮影色'
LABELS['partial_dense_native_relative_only']='追加試作・相対色のみ'
for _book in ['200021763','200021086','200020019']:
    LABELS['texture_paper_texture_sparse_exclude'+_book]='当該冊を外した学習・'+_book
    LABELS['region_spatial_forest_exclude'+_book]='当該冊なし・色と周囲の森分類'
    for _name in ['sparse_control','dense_native','dense_native_brown']:
        LABELS['partial_'+_name+'_exclude'+_book]='当該冊なし・'+{'sparse_control':'点対照','dense_native':'紙領域','dense_native_brown':'紙領域＋茶色'}[_name]
    LABELS['partial_dense_native_photometry_exclude'+_book]='当該冊なし・紙領域＋撮影色'
    LABELS['partial_dense_native_relative_only_exclude'+_book]='当該冊なし・相対色のみ'

def mask_path(method,page):
    if method in ['csc_v1_red','csc_v2_red','saved_csc_v1_red','saved_csc_v2_red','hsv_loose','relative_chroma','relative_ridge']:
        return OUT/'fullpage_baselines'/page/(method+'.png')
    return OUT/'native_predictions'/method/page/'mark_mask.png'

def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['csc_v2_red','hsv_loose','relative_ridge','relative_overlap'])
    p.add_argument('--roles',nargs='+',default=['development']);p.add_argument('--suffix',default='initial')
    a=p.parse_args();dest=OUT/'selector_diagnostics'/a.suffix;dest.mkdir(parents=True,exist_ok=True)
    records=json.loads((OUT/'cases_frozen.json').read_text())['cases'];font=ImageFont.truetype(FONT,18);small=ImageFont.truetype(FONT,15)
    metrics=[]
    for r in records:
        if r['role'] not in a.roles:continue
        paths=[mask_path(m,r['page']) for m in a.methods]
        if not all(p.exists() for p in paths):continue
        x,y,w,h=r['ROI'];rgb=np.asarray(Image.open(OUT/'case_inputs'/(r['case_id']+'.png')).convert('RGB'))
        masks=[np.asarray(Image.open(p).convert('L'))[y:y+h,x:x+w]>127 for p in paths]
        target_w=max(w,260);header=94;footer=66
        sheet=Image.new('RGB',((len(masks)+1)*target_w,header+h+footer),'#faf9f5');d=ImageDraw.Draw(sheet)
        d.text((12,8),r['name']+'  '+r['page'],font=font,fill='#20323b')
        d.text((12,33),'白＝朱の貼り戻し候補。修復画像ではなく、原寸マスクの比較。',font=small,fill='#536470')
        d.text((12,64),'原画像',font=font,fill='#20323b');sheet.paste(Image.fromarray(rgb),(0,header))
        counts={}
        for j,(name,mask) in enumerate(zip(a.methods,masks),1):
            xx=j*target_w;d.text((xx+4,64),LABELS[name],font=font,fill='#20323b')
            shown=np.repeat(mask[:,:,None]*255,3,2).astype(np.uint8);sheet.paste(Image.fromarray(shown),(xx,header))
            counts[name]=int(mask.sum());d.text((xx+4,header+h+10),f'候補 {counts[name]:,}画素',font=small,fill='#536470')
        d.text((12,header+h+37),'候補数は保存精度・除去精度ではない。画像の粒や茶色の場を拾う失敗も含む。',font=small,fill='#536470')
        sheet.save(dest/(r['case_id']+'_原画像とマスク.png'))
        metrics.append(dict(case=r['case_id'],role=r['role'],ROI=r['ROI'],pixel_count=w*h,counts=counts,
            candidate_count_is_not_accuracy=True))
    (dest/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    print('Rendered mask diagnostic PNGs:',len(metrics),flush=True)

if __name__=='__main__':main()
