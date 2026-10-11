"""Review authentic saved npj images without inference or image repair.

The six historical frames are a developmental sample, not a corpus survey.
Render unaltered input/output comparisons; freeze only after visual notes exist.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/npj_need_assessment_20261002'
SELECTION = ROOT / 'tmp_work/npj_literature_trial_20261002/selection.json'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
cv2.setNumThreads(1)


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def read(p):
    a = cv2.imread(str(p))
    if a is None:
        raise OSError(p)
    return a


def panels(images, labels, path, title, note, overview=False):
    h, w = images[0].shape[:2]
    assert all(a.shape == (h, w, 3) for a in images)
    if overview:
        ratio = min(540/w, 900/h, 1)
        dw, dh = round(w*ratio), round(h*ratio)
        shown = [cv2.resize(a, (dw, dh), interpolation=cv2.INTER_AREA) for a in images]
    else:
        scale = min(8, max(1, min(680//w, 900//h)))
        dw, dh = w*scale, h*scale
        shown = [np.repeat(np.repeat(a, scale, 0), scale, 1) for a in images]
    font, small = ImageFont.truetype(FONT, 23), ImageFont.truetype(FONT, 18)
    width = 48+len(images)*dw+(len(images)-1)*24
    wrapped = []
    for line in note:
        current = ''
        for char in line:
            if current and small.getlength(current+char) > width-48:
                wrapped.append(current); current = char
            else:
                current += char
        if current: wrapped.append(current)
    canvas = Image.new('RGB', (width, dh+max(230, 165+27*len(wrapped))), '#faf8f2')
    d = ImageDraw.Draw(canvas)
    d.text((24, 12), title, font=font, fill='black')
    d.text((24, 47), '閲覧用縮小・色補正なし' if overview else f'原画素の{scale}倍整数反復・色補正なし', font=small, fill='black')
    for i, (a, label) in enumerate(zip(shown, labels)):
        x = 24+i*(dw+24)
        d.text((x, 79), label, font=small, fill='black')
        canvas.paste(Image.fromarray(a[:, :, ::-1]), (x, 112))
    for i, line in enumerate(wrapped):
        d.text((24, dh+126+i*27), line, font=small, fill='black')
    canvas.save(path)
    stored = np.asarray(Image.open(path).convert('RGB'))
    for i, a in enumerate(shown):
        x = 24+i*(dw+24)
        assert np.array_equal(stored[112:112+dh, x:x+dw], a[:, :, ::-1])
    return dict(file=path.name, panels_exact=True, overview=overview, display_size=[dw, dh])


def load_frames():
    selection = json.loads(SELECTION.read_text())
    deps = {str(SELECTION.relative_to(ROOT)):sha(SELECTION), str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    frames = {}
    for f in selection['frames']:
        for key in ['original', 'input', 'saved_npj']:
            p = ROOT/f[key]
            digest = sha(p)
            assert digest == selection['dependencies'][f[key]], f[key]
            deps[f[key]] = digest
        original = read(ROOT/f['original'])
        if f['box']:
            x, y, w, h = f['box']
            original = original[y:y+h, x:x+w]
        saved_input, npj = read(ROOT/f['input']), read(ROOT/f['saved_npj'])
        assert original.shape == saved_input.shape == npj.shape
        assert bool(np.array_equal(original, saved_input)) == f['input_reference_exact']
        frames[f['id']] = (f, original, saved_input, npj)
    return frames, deps


def inventory():
    records = []
    for root in ['tmp_work', 'results', 'modules']:
        for p in sorted((ROOT/root).rglob('result/*.png')):
            rel = str(p.relative_to(ROOT))
            records.append(dict(path=rel, page=p.stem,
                family=p.parts[len(ROOT.parts)+1],
                note='保存ファイルの所在のみ。通常npj・実資料・最終工程の認定ではない'))
    with (OUT/'saved_result_inventory.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'page', 'family', 'note'])
        writer.writeheader(); writer.writerows(records)
    return dict(result_named_png_files=len(records), distinct_stems=len({r['page'] for r in records}),
                note='variant, synthetic, crop and other restoration methods included; not npj page coverage')


def render():
    assert not (OUT/'complete.json').exists(), 'Frozen assessment'
    OUT.mkdir(parents=True, exist_ok=True)
    frames, deps = load_frames()
    rendered, grid = [], []
    for fid, (f, original, saved_input, npj) in frames.items():
        exact = f['input_reference_exact']
        images = [original, npj] if exact else [original, saved_input, npj]
        labels = ['原画像＝実推論入力', '保存npj'] if exact else ['参照原画像', '実推論入力', '保存npj']
        note = ['追加処理・再推論なし。文字の正解や読解性は未評価。',
                '入力は原画像と一致。' if exact else '入力の濃淡・斑点が参照原画像と異なる。入力と出力で効果を判断。']
        rendered.append(panels(images, labels, OUT/f'overview_{fid}.png', f['page']+' / '+f['baseline_label'], note, True))
        for region in f['regions']:
            x,y,w,h = region['xywh']; s = np.s_[y:y+h,x:x+w]
            assert original[s].shape == (h,w,3)
            rendered.append(panels([a[s] for a in images], labels, OUT/(region['id']+'.png'),
                region['title']+' / '+f['page'], note))
        H, W = saved_input.shape[:2]
        # Uniform full-frame coverage, irrespective of masks or failure labels.
        for y in range(0, H, 800):
            for x in range(0, W, 800):
                w, h = min(800,W-x), min(800,H-y)
                pair = [a[y:y+h,x:x+w] for a in [saved_input,npj]]
                grid.append(dict(frame=fid, page=f['page'], xywh=[x,y,w,h], pair=pair))
    # Small sheets are screening only; suspected features need native crops.
    font, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 16)
    tile_records = []
    for start in range(0,len(grid),4):
        canvas = Image.new('RGB',(1520,950),'#faf8f2');d=ImageDraw.Draw(canvas)
        d.text((16,8),'全範囲走査：左＝実推論入力、右＝保存npj（縮小・色補正なし）',font=font,fill='black')
        checks=[]
        for i, tile in enumerate(grid[start:start+4]):
            ox,oy=(i%2)*760,44+(i//2)*450
            d.text((ox+12,oy),f'{tile["frame"]} / {tile["xywh"]}',font=small,fill='black')
            for j,a in enumerate(tile['pair']):
                h,w=a.shape[:2]; ratio=min(350/w,400/h,1); dw,dh=round(w*ratio),round(h*ratio)
                display=cv2.resize(a,(dw,dh),interpolation=cv2.INTER_AREA)[:,:,::-1]
                px,py=ox+12+j*370,oy+30
                canvas.paste(Image.fromarray(display),(px,py));checks.append((px,py,display))
            tile_records.append({k:v for k,v in tile.items() if k!='pair'}|dict(sheet=f'scan_{start//4+1:02d}.png'))
        p=OUT/f'scan_{start//4+1:02d}.png';canvas.save(p)
        stored=np.asarray(Image.open(p).convert('RGB'))
        for x,y,a in checks:
            h,w=a.shape[:2];assert np.array_equal(stored[y:y+h,x:x+w],a)
        rendered.append(dict(file=p.name,panels_exact=True,overview=True))
    inv=inventory()
    dump(OUT/'manifest.json', dict(frames=[v[0] for v in frames.values()],dependencies=deps,
        source_sha256=sha(__file__), scope='6 historical frames, 5 known pages; not representative or unseen',
        inference=False,postprocess=False,GT=False,illustrations=False, result_inventory=inv))
    dump(OUT/'rendering.json',dict(figures=rendered,scan_tiles=tile_records,
        fixed_rois=20,full_frames=6,unique_pages=5))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    print(json.dumps(dict(figures=len(rendered),scan_tiles=len(grid),inventory=inv),ensure_ascii=False))


def extra(fid, box, name, title):
    assert not (OUT/'complete.json').exists(), 'Frozen assessment'
    frames,_=load_frames();f,original,saved_input,npj=frames[fid]
    x,y,w,h=box;s=np.s_[y:y+h,x:x+w]
    assert original[s].shape==(h,w,3)
    exact=f['input_reference_exact']
    images=[original,npj] if exact else [original,saved_input,npj]
    labels=['原画像＝実推論入力','保存npj'] if exact else ['参照原画像','実推論入力','保存npj']
    record=panels([a[s] for a in images],labels,OUT/(name+'.png'),title,
        ['追加処理なし。今回の目視で選んだ確認範囲。読解性・材質の正解は未評価。'])
    path=OUT/'additional_regions.json';records=json.loads(path.read_text()) if path.exists() else []
    records = [r for r in records if r['file'] != record['file']]
    records.append(dict(frame=fid,xywh=box,title=title,**record));dump(path,records)


def versions():
    assert not (OUT/'complete.json').exists(), 'Frozen assessment'
    frames,_=load_frames()
    f,original,saved_input,npj=frames['known_200010454_v2']
    assert np.array_equal(original,saved_input)
    images=[original,npj];labels=['原画像＝実推論入力','既存版 ddrm_out']
    deps={}
    for name in ['v2fix','v3fix']:
        p=ROOT/f'tmp_work/speed_test_single/ddrm_out_{name}/200010454_00002_2/result/200010454_00002_2.png'
        log=ROOT/f'tmp_work/speed_test_single/ddrm_run_{name}.log'
        text=log.read_text()
        assert '--input_dir /home/imoto/Heritage_Pipeline/tmp_work/speed_test_single/input' in text
        assert '--mask_dir /home/imoto/Heritage_Pipeline/data/split_dataset_csc_mask_v2/test/200010454' in text
        a=read(p);assert a.shape==original.shape
        images.append(a);labels.append('別保存版 '+name)
        for source in [p,log]:deps[str(source.relative_to(ROOT))]=sha(source)
    records=[]
    for name,box,title in [
        ('24_saved_versions_faint',[1230,500,280,650],'同ページの薄い主線：保存版ごとの比較'),
        ('25_saved_versions_stain',[730,230,180,140],'同ページの茶色い斑点：保存版ごとの比較')]:
        x,y,w,h=box;s=np.s_[y:y+h,x:x+w]
        records.append(dict(xywh=box,**panels([a[s] for a in images],labels,OUT/(name+'.png'),title,
            ['全て既存の保存出力。v2fix／v3fixは保存名で、v6や今回の新手法ではない。',
             '同じ入力を使った記録がある。過去の全ソース・乱数・合成条件の完全再現は未確認。'])))
    old,old_original,old_input,old_npj=frames['old_upper_crop']
    x,y,w,h=old['box'];assert np.array_equal(original[y:y+h,x:x+w],old_input)
    box=[180,230,180,140];x,y,w,h=box;s=np.s_[y:y+h,x:x+w]
    images=[old_original[s],old_npj[s],npj[230:370,730:910]]
    records.append(dict(xywh=box,**panels(images,
        ['原画像（両保存入力と一致）','旧v1/selfref・切り出し','CSC v2・全ページから切り出し'],
        OUT/'26_stain_crop_vs_page.png','同じ斑点：旧切り出し版と全ページ版',
        ['同じページの同じ原画素。推論範囲・背景参照等の違いを含む。マスク単独の比較ではない。'])))
    manifest=json.loads((OUT/'manifest.json').read_text());manifest['dependencies'].update(deps)
    dump(OUT/'manifest.json',manifest);dump(OUT/'version_comparisons.json',records)


def finish():
    assert not (OUT/'complete.json').exists(), 'Frozen assessment'
    manifest=json.loads((OUT/'manifest.json').read_text())
    for rel,digest in manifest['dependencies'].items(): assert sha(ROOT/rel)==digest,rel
    assert (OUT/'report.md').exists() and (OUT/'case_review.csv').exists(), 'Write visual findings first'
    figs=sorted(OUT.glob('*.png'))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in figs:z.write(p,p.name)
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert z.testzip() is None
        for p in figs:assert z.read(p.name)==p.read_bytes()
    dump(OUT/'verification.json',dict(status='passed',input_output_hashes_unchanged=True,
        displayed_panels_exact=True,frames=6,unique_pages=5,figures=len(figs),
        comparisons_not_new_inference=True,GT=False,prevalence_established=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='complete.json'}))
    print('Frozen assessment:',len(figs),'PNGs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['render','extra','versions','finish'])
    parser.add_argument('--frame');parser.add_argument('--box',type=int,nargs=4)
    parser.add_argument('--name');parser.add_argument('--title')
    args=parser.parse_args()
    if args.stage=='render':render()
    elif args.stage=='extra':extra(args.frame,args.box,args.name,args.title)
    elif args.stage=='versions':versions()
    else:finish()
