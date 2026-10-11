"""Added mark layers on original-only tentative paper-negative textures.

Preservation supervision is synthetic; real regions only say no visible red mark.
Background patches are not expert clean-material/stain ground truth.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,LayerDataset,input_features,file_hash

class PaperTextureLayers(LayerDataset):
    def __init__(self,*args,texture=True,**kwargs):
        super().__init__(*args,**kwargs);self.texture=texture
        info=json.loads((OUT/'original_paper_negative_regions/regions_frozen.json').read_text())
        heldout=set(json.loads((OUT/'cases_frozen.json').read_text())['validation_books'])
        self.textures=[];self.texture_rows=info['regions']
        for r in self.texture_rows:
            assert r['book'] not in heldout
            x,y,w,h=r['xywh_original'];a=np.asarray(Image.open(ROOT/r['original']).convert('RGB'))[y:y+h,x:x+w]
            self.textures.append(a.copy())
    def sample(self,index,variant=0):
        rgb,target,clean,meta=super().sample(index,variant)
        if not self.texture:return rgb,target,clean,meta
        rng=np.random.default_rng(self.seed+index*104729)
        oldpaper=self.papers[int(rng.integers(len(self.papers)))].astype(np.float32)
        rng=np.random.default_rng(self.seed+index*104729+82333)
        texture_index=int(rng.integers(len(self.textures)));tex=self.textures[texture_index]
        # Reflection extends a small audited texture without synthesizing lettering.
        tiled=cv2.copyMakeBorder(tex,256,256,256,256,cv2.BORDER_REFLECT_101)
        y=int(rng.integers(0,tiled.shape[0]-256));x=int(rng.integers(0,tiled.shape[1]-256));paper=tiled[y:y+256,x:x+256].astype(np.float32)
        if rng.random()<.5:paper=paper[:,::-1]
        if rng.random()<.5:paper=paper[::-1]
        gains=rng.uniform(.9,1.1,3);brightness=float(rng.uniform(.9,1.08))
        paper=np.clip(paper*gains*brightness,15,255)
        recolored=np.clip(np.rint((rgb.astype(np.float32)+1)/(oldpaper+1)*(paper+1)-1),0,255).astype(np.uint8)
        clean_new=np.clip(np.rint((clean.astype(np.float32)+1)/(oldpaper+1)*(paper+1)-1),0,255).astype(np.uint8)
        meta.update(paper_texture_region_index=texture_index,
            paper_negative_source_book=self.texture_rows[texture_index]['book'],
            real_paper_texture=True,real_clean_stain_GT=False)
        return recolored,target,clean_new,meta

def preview():
    dest=OUT/'paper_texture_synthesis';dest.mkdir(exist_ok=True)
    ds=PaperTextureLayers('train',24,mode='relative',training='overlap',seed=20261013)
    sheet=Image.new('RGB',(4*256,6*292),'#faf9f5');d=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
    infos=[]
    for i in range(12):
        rgb,y,clean,meta=ds.sample(i);infos.append(meta);xx=(i%2)*512;yy=(i//2)*292
        sheet.paste(Image.fromarray(rgb),(xx,yy));sheet.paste(Image.fromarray(clean),(xx+256,yy))
        d.text((xx+3,yy+258),f'{i} 追加層入力／追加シミなし',font=font,fill='#20313a')
        d.text((xx+3,yy+274),meta['paper_negative_source_book']+' 紙の粒を含む',font=font,fill='#20313a')
    sheet.save(dest/'実物紙の粒と合成層.png')
    (dest/'protocol.json').write_text(json.dumps(dict(preview_info=infos,
        paper_regions_sha256=file_hash(OUT/'original_paper_negative_regions/regions_frozen.json'),
        script_sha256=file_hash(__file__),semantic_supervision='Known added mark support; underlying paper has tentative no-red labels only',
        no_real_positive_mark_GT=True,no_definitive_real_stain_GT=True,
        augmentation='Reflected native paper texture + small channel and brightness changes',
        validation_paper_books_excluded=True),ensure_ascii=False,indent=2))
    print('Paper-texture synthesis preview prepared',flush=True)

if __name__=='__main__':preview()
