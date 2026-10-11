"""Create original / local npj ACE+CSC / frozen v6 comparison PNGs.

Classification stays at native resolution. The chunked CPU ACE arithmetic is
checked pixel-for-pixel against a saved frozen CPU output before new inputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/npj_vs_v6_comparison_20261001'
V6 = ROOT / 'tmp_work/warm_ink_v6_other_books_fixed_20261001'
NPJ = ROOT / 'tmp_work/npj_preprocess_review_20260930'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '4'

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)
COLORS = dict(blue=(225,100,38), red=(69,52,228), edit=(23,178,235), hold=(176,65,155))
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def read(path, mask=False):
    value = cv2.imread(str(path),0 if mask else 1)
    if value is None:
        raise OSError(path)
    return value > 0 if mask else value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if value.dtype == bool:
        value = value.astype(np.uint8)*255
    assert cv2.imwrite(str(path),value),path


def load_module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def chunked_ace(image, alpha=5., n_samples=200, seed=0, block_pixels=32768):
    """Same random samples, float32 row sums and global stretch as frozen CPU ACE."""
    h,w = image.shape[:2]; count = h*w
    rng = np.random.default_rng(seed)
    samples = rng.integers(0,count,size=n_samples)
    sy = (samples//w).astype(np.float32)
    sx = (samples%w).astype(np.float32)
    flat = (image.astype(np.float32)/255.).reshape(count,3)
    result = np.zeros((count,3),np.float32)
    sample_values = flat[samples]
    for start in range(0,count,block_pixels):
        stop = min(start+block_pixels,count)
        positions = np.arange(start,stop)
        dy = (positions//w).astype(np.float32)[:,None]-sy[None,:]
        dx = (positions%w).astype(np.float32)[:,None]-sx[None,:]
        dist = np.sqrt(dy**2+dx**2)
        dist = np.maximum(dist,1.)
        inv = 1./dist
        denominator = inv.sum(axis=1)
        for c in range(3):
            diff = flat[start:stop,c,None]-sample_values[None,:,c]
            clipped = np.clip(alpha*diff,-1.,1.)
            result[start:stop,c] = (clipped*inv).sum(axis=1)/denominator
    lo,hi = result.min(),result.max()
    normalized = (result-lo)/(hi-lo) if hi-lo>1e-6 else flat.copy()
    return (np.clip(normalized,0.,1.)*255).astype(np.uint8).reshape(h,w,3)


def prepare_manifest():
    OUT.mkdir(parents=True,exist_ok=True)
    prior = json.loads((NPJ/'manifest.json').read_text())
    v6 = json.loads((V6/'manifest.json').read_text())
    assert json.loads((V6/'verification.json').read_text())['status']=='passed'
    deps = {str(p.relative_to(ROOT)):sha(p) for p in [NPJ/'manifest.json',V6/'manifest.json',V6/'complete.json',V6/'verification.json']}
    for relative in ['modules/ddrm_npj/scripts/ace.py','modules/ddrm_npj/scripts/csc.py']:
        assert sha(ROOT/relative)==prior['source_hashes'][relative],relative
        deps[relative]=sha(ROOT/relative)
    products = json.loads((V6/'complete.json').read_text())['outputs']
    for row in v6['cases']:
        assert sha(ROOT/row['original'])==row['original_sha256']
        deps[row['original']]=row['original_sha256']
        for name in ['v2/red.png','v2/protect.png','v6_edit_allowed.png','v6_hold.png']:
            relative=row['id']+'/'+name
            assert sha(V6/relative)==products[relative],relative
            deps[str((V6/relative).relative_to(ROOT))]=products[relative]
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=deps,cases=v6['cases'],rois=v6['rois'],
                  npj_baseline='Local frozen ddrm_npj CPU ACE + v1 CSC; exact published reproduction not established',
                  ace=dict(alpha=5.,n_samples=200,seed=0,arithmetic='native resolution; pixel-row chunking only'),
                  csc=dict(seed=42,max_samples=50000,border_margin_ratio=.04,red_a_min=133.),
                  v6_variant='frozen split_direction',restoration=False,pixel_GT=False,thresholds_changed=False,
                  display='Masks classified at native resolution; composited onto the identical original image')
    path=OUT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('Changed comparison; use a new output directory')
    dump(path,manifest);shutil.copyfile(__file__,OUT/'source_snapshot.py')
    return manifest


def baseline(manifest):
    csc=load_module('comparison_frozen_npj_csc',ROOT/'modules/ddrm_npj/scripts/csc.py')
    parity_path=OUT/'ace_parity.json'
    if not parity_path.exists():
        stem='hnsd001_011'
        native=read(ROOT/'data/train/hnsd00000/hnsd001_011.jpg')
        expected=read(NPJ/stem/'ace.png')
        saved=json.loads((NPJ/stem/'complete.json').read_text())
        assert sha(NPJ/stem/'ace.png')==saved['outputs'][stem+'/ace.png']
        actual=chunked_ace(native)
        assert np.array_equal(actual,expected),'Chunked ACE must match frozen CPU ACE pixels'
        dump(parity_path,dict(status='passed',page=stem,native_pixels_equal=True,original_sha256=sha(ROOT/'data/train/hnsd00000/hnsd001_011.jpg'),
                             expected_ace_sha256=sha(NPJ/stem/'ace.png'),block_pixels=32768))
        print('Chunked CPU ACE: native pixel parity passed',flush=True)
        del actual,native,expected;gc.collect()
    else:
        assert json.loads(parity_path.read_text())['status']=='passed'
    for index,row in enumerate(manifest['cases'],1):
        dest=OUT/'baseline'/row['id'];done=dest/'complete.json'
        if done.exists():
            for rel,digest in json.loads(done.read_text())['outputs'].items():
                assert sha(dest/rel)==digest,rel
            print('Cached npj baseline verified:',row['id'],flush=True)
            continue
        original=read(ROOT/row['original']);start=time.monotonic()
        print('CPU ACE start:',row['id'],flush=True)
        ace_done=dest/'ace_complete.json'
        if ace_done.exists():
            assert sha(dest/'ace.png')==json.loads(ace_done.read_text())['sha256']
            equalized=read(dest/'ace.png')
        else:
            equalized=chunked_ace(original)
            write(dest/'ace.png',equalized)
            dump(ace_done,dict(sha256=sha(dest/'ace.png'),original_sha256=row['original_sha256']))
        print(f'CPU ACE finished: {row["id"]}, {time.monotonic()-start:.1f}s; CSC start',flush=True)
        result=csc.color_space_clustering(equalized,**manifest['csc'])
        masks=dict(black=result.black_text_mask,red=result.red_text_mask,stain=result.degradation_mask,background=result.background_mask)
        total=np.zeros(original.shape[:2],np.uint8)
        for key,value in masks.items():
            assert set(np.unique(value)).issubset({0,255})
            total+=value>0;write(dest/(key+'.png'),value)
        assert np.all(total==1)
        dump(done,dict(outputs={p.name:sha(p) for p in sorted(dest.glob('*.png'))},seconds=time.monotonic()-start))
        dump(OUT/'progress.json',dict(stage='npj_baseline',completed_pages=index,total_pages=6))
        print(f'npj complete: {row["id"]}, {time.monotonic()-start:.1f}s',flush=True)
        del original,equalized,result,masks,total;gc.collect()


def overlay(original,masks):
    result=original.copy()
    for key,mask in masks:
        result[mask]=np.round(.45*original[mask]+.55*np.array(COLORS[key])).astype(np.uint8)
    return result


def figure(images,labels,path,title,note,max_width=900):
    width,height=images[0].shape[1],images[0].shape[0]
    scale=min(max_width/width,1250/height)
    if width*scale<420:
        scale=420/width
    dw,dh=max(1,round(width*scale)),max(1,round(height*scale))
    gap=24;margin=28;header=90;caption=64;footer=76
    canvas=Image.new('RGB',(3*dw+2*gap+2*margin,header+caption+dh+footer),(250,248,243))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT_PATH,28);small=ImageFont.truetype(FONT_PATH,22)
    draw.text((margin,14),title,font=font,fill=(25,25,25))
    draw.text((margin,54),'同じ原画像・同じ領域へのマスク重ね合わせ',font=small,fill=(70,70,70))
    for i,(value,label) in enumerate(zip(images,labels)):
        x=margin+i*(dw+gap)
        draw.text((x,header),label,font=font,fill=(25,25,25))
        thumb=cv2.resize(value,(dw,dh),interpolation=cv2.INTER_NEAREST) if (dw,dh)!=(width,height) else value
        canvas.paste(Image.fromarray(cv2.cvtColor(thumb,cv2.COLOR_BGR2RGB)),(x,header+caption))
    draw.text((margin,header+caption+dh+16),note,font=ImageFont.truetype(FONT_PATH,20),fill=(55,55,55))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def render(manifest):
    cards=[];metrics=[];all_images=[]
    for row in manifest['cases']:
        stem=row['id'];original=read(ROOT/row['original']);h,w=original.shape[:2]
        npj={k:read(OUT/'baseline'/stem/(k+'.png'),True) for k in ['black','red','stain','background']}
        protect=read(V6/stem/'v2/protect.png',True);red=read(V6/stem/'v2/red.png',True)
        edit=read(V6/stem/'v6_edit_allowed.png',True);hold=read(V6/stem/'v6_hold.png',True)
        npj_overlay=overlay(original,[('edit',npj['stain']),('blue',npj['black']),('red',npj['red'])])
        v6_overlay=overlay(original,[('hold',hold),('edit',edit),('blue',protect),('red',red)])
        npj_edit=overlay(original,[('edit',npj['stain'])]);v6_edit=overlay(original,[('edit',edit)])
        sections=[]
        boxes=[('whole','ページ全体',[0,0,w,h])]+[(f'roi_{i:02d}',roi['title'],roi['xywh']) for i,roi in enumerate(manifest['rois'][stem],1)]
        for key,title,box in boxes:
            x,y,rw,rh=box;sl=(slice(y,y+rh),slice(x,x+rw))
            folder=OUT/'images'/stem
            for name,value in [('original',original),('npj_classes',npj_overlay),('v6_decisions',v6_overlay),('npj_edit',npj_edit),('v6_edit',v6_edit)]:
                write(folder/(key+'_'+name+'_native.png'),value[sl])
            rel=f'images/{stem}/{key}_compare.png'
            figure([original[sl],npj_overlay[sl],v6_overlay[sl]],['原画像','npj段階：ACE＋CSC','v6：保護・保留・編集候補'],OUT/rel,
                   stem+' / '+title,'青＝墨候補／保護　赤＝朱候補　黄＝劣化候補／編集候補　紫＝v6の保留')
            edit_rel=f'images/{stem}/{key}_edit_compare.png'
            figure([original[sl],npj_edit[sl],v6_edit[sl]],['原画像','npj：劣化・シミ候補（黄）','v6：編集候補（黄）'],OUT/edit_rel,
                   stem+' / '+title,'黄は候補マスク。前処理の比較であり、復元後の画像ではありません。')
            all_images.extend([rel,edit_rel])
            metrics.append(dict(page=stem,region=key,title=title,xywh=box,npj_black_pixels=int(npj['black'][sl].sum()),
                                npj_red_pixels=int(npj['red'][sl].sum()),npj_stain_pixels=int(npj['stain'][sl].sum()),
                                v6_protect_pixels=int(protect[sl].sum()),v6_hold_pixels=int(hold[sl].sum()),v6_edit_pixels=int(edit[sl].sum()),
                                compare=rel,edit_compare=edit_rel))
            sections.append(f'<section><h3>{html.escape(title)} · {box}</h3><a href="{rel}"><img src="{rel}" loading="lazy" alt="原画像・npj・v6の比較"></a><details><summary>黄色の候補だけで比較</summary><a href="{edit_rel}"><img src="{edit_rel}" loading="lazy" alt="劣化候補と編集候補の比較"></a></details><p>原寸：<a href="images/{stem}/{key}_original_native.png">原画像</a> / <a href="images/{stem}/{key}_npj_classes_native.png">npj分類</a> / <a href="images/{stem}/{key}_v6_decisions_native.png">v6判断</a> / <a href="images/{stem}/{key}_npj_edit_native.png">npj劣化候補</a> / <a href="images/{stem}/{key}_v6_edit_native.png">v6編集候補</a></p></section>')
        cards.append(f'<article id="{stem}"><h2>{stem}</h2>{"".join(sections)}</article>')
        print('Comparison images rendered:',stem,flush=True)
        del original,npj,protect,red,edit,hold,npj_overlay,v6_overlay,npj_edit,v6_edit;gc.collect()
    dump(OUT/'metrics.json',metrics)
    dump(OUT/'image_list.json',all_images)
    toc=' · '.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in manifest['cases'])
    (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>npjとv6の比較</title><style>body{font-family:system-ui,sans-serif;margin:24px;background:#faf8f3;line-height:1.7}img{width:100%;height:auto}article{margin-top:48px}section{margin:32px 0}summary{cursor:pointer}</style><h1>原画像・npj段階・v6</h1><p>6ページと21か所の拡大。左＝原画像、中央＝手元のnpj方式（CPU ACE＋CSC）、右＝固定したv6。青＝墨候補／保護、赤＝朱候補、黄＝劣化候補／編集候補、紫＝v6の保留。同じ原画像に重ねて位置を比較します。v6では保護と保留が重なる場合、保護を表示します。</p><p>画像をクリックするとPNGを開きます。黄色だけの比較も展開できます。前処理マスクの比較で、復元後の画像や画素GTではありません。<a href="report.md">比較条件</a> · <a href="verification.json">検証</a></p><nav>'+toc+'</nav>'+''.join(cards)+'</html>\n')
    (OUT/'report.md').write_text('''# npj段階とv6の比較画像

2026-10-01。直近の別冊6冊・6ページと、入力から選定済みの21矩形に対して、原画像／npj段階／固定v6の3列比較を作成した。[画像一覧](index.html)。分類全体と黄色の候補だけの比較があり、各画像をPNGで保存した。原寸の重ね画像も保存している。

npj段階は凍結済みのローカル `modules/ddrm_npj` のCPU ACE＋CSCであり、原著者の出力・原論文の完全再現を確認した結果とは区別する。ACEはalpha=5、n_samples=200、seed=0。CSCはseed=42、max_samples=50000、border_margin_ratio=0.04、red_a_min=133。元解像度で処理した。

CPU ACEは同じサンプル・演算・全ページでの伸張を画素行のチャンク単位で行い、既存hnsdページの保存済みCPU ACEと原寸全画素の一致を検証してから新しい6枚へ適用した。v6の条件と出力マスクは前段の固定確認から再利用した。

3列の比較は同じ原画像へそれぞれのマスクを重ねたもので、ACE後の色の変化と分類結果の違いを混同しない。ACE画像はbaselineディレクトリに別保存してある。青はnpjの墨候補、v6では文字等の保護（色未確定を含む）。赤はそれぞれの朱候補。黄はnpjの劣化・シミ候補、v6の編集候補。両者は同じ定義のクラスではない。紫はv6の保留で、保護と重なる位置では保護を優先して表示する。表示はマスクの重なりを変更しない。

ページ全体のPNGは閲覧用に縮小し、拡大図は原寸または最大表示幅に合わせて表示する。分類と原寸PNGは元解像度。JPEG再圧縮や原画像の変更、拡散復元、ニューラル学習、画素GT作成は行っていない。候補の面積は精度や誤消去率ではない。
''')


def verify(manifest):
    assert sha(Path(__file__))==manifest['source_sha256']
    for rel,digest in manifest['dependencies'].items():
        assert sha(ROOT/rel)==digest,rel
    assert json.loads((OUT/'ace_parity.json').read_text())['status']=='passed'
    for row in manifest['cases']:
        stem=row['id'];original=read(ROOT/row['original']);dest=OUT/'baseline'/stem
        for rel,digest in json.loads((dest/'complete.json').read_text())['outputs'].items():
            assert sha(dest/rel)==digest,rel
        masks={k:read(dest/(k+'.png'),True) for k in ['black','red','stain','background']}
        assert np.all(sum(v.astype(np.uint8) for v in masks.values())==1)
        for record in [r for r in json.loads((OUT/'metrics.json').read_text()) if r['page']==stem]:
            x,y,w,h=record['xywh'];sl=(slice(y,y+h),slice(x,x+w));p=OUT/'images'/stem/(record['region']+'_original_native.png')
            assert np.array_equal(read(p),original[sl])
            expected_np=overlay(original[sl],[('edit',masks['stain'][sl]),('blue',masks['black'][sl]),('red',masks['red'][sl])])
            assert np.array_equal(read(OUT/'images'/stem/(record['region']+'_npj_classes_native.png')),expected_np)
            v6_masks=[('hold',read(V6/stem/'v6_hold.png',True)[sl]),('edit',read(V6/stem/'v6_edit_allowed.png',True)[sl]),
                      ('blue',read(V6/stem/'v2/protect.png',True)[sl]),('red',read(V6/stem/'v2/red.png',True)[sl])]
            assert np.array_equal(read(OUT/'images'/stem/(record['region']+'_v6_decisions_native.png')),overlay(original[sl],v6_masks))
            assert np.array_equal(read(OUT/'images'/stem/(record['region']+'_npj_edit_native.png')),overlay(original[sl],[('edit',masks['stain'][sl])]))
            assert np.array_equal(read(OUT/'images'/stem/(record['region']+'_v6_edit_native.png')),overlay(original[sl],[v6_masks[1]]))
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['href','src'] and value and not value.startswith('#'):
                    assert (OUT/value).is_file(),value
    Links().feed((OUT/'index.html').read_text())
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['complete.json','verification.json','progress.json','run.log']]
    dump(OUT/'complete.json',dict(status='complete',outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(files)}))
    dump(OUT/'verification.json',dict(status='passed',pages=6,input_selected_rois=21,comparison_pngs=54,native_panel_pngs=135,
                                      source_and_input_hashes_checked=True,chunked_ace_pixel_parity=True,
                                      original_crops_and_overlay_pixels_checked=True,v6_masks_unchanged=True,
                                      viewer_links_checked=True,pixel_GT=False,restoration=False,browser_rendering_tested=False))
    dump(OUT/'progress.json',dict(status='verified',completed_pages=6,total_pages=6))
    print('Comparison verified: 6 pages, 21 crops, 54 comparison PNGs',flush=True)


def main():
    manifest=prepare_manifest()
    baseline(manifest)
    render(manifest)
    dump(OUT/'verification.json',dict(status='pending'))
    verify(manifest)


if __name__=='__main__':
    main()
