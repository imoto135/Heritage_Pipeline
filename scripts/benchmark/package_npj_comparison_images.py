"""Package existing native comparisons as annotated PNG sheets and a ZIP.

Presentation only: classification masks and original results are unchanged.
"""
import json
from pathlib import Path
import zipfile
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import FONT_PATH,sha,dump

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tmp_work/npj_failure_collection_20261001'
OUT=ROOT/'tmp_work/npj_failure_comparison_pngs_20261001'
GROUPS={
 '01_stain_comparison.png':['dense_stain','brown_as_red','brown_false_red_new'],
 '02_stroke_comparison.png':['thin_black_water','thin_black_loop','warm_stroke'],
 '03_remaining_problems.png':['dense_stain_new','hole_edge_new'],
 '04_controls.png':['thin_black_text','control_black','control_text_new'],
}


def board(paths,labels,destination,title,max_width=1964):
    font=ImageFont.truetype(FONT_PATH,28)
    small=ImageFont.truetype(FONT_PATH,23)
    resized=[]
    for path in paths:
        with Image.open(path) as source:
            width=min(max_width,source.width)
            resized.append(source.convert('RGB').resize((width,round(source.height*width/source.width)),Image.Resampling.LANCZOS))
    width=max(im.width for im in resized)
    canvas=Image.new('RGB',(width,100+sum(im.height+84 for im in resized)),(250,248,243))
    draw=ImageDraw.Draw(canvas);draw.text((28,18),title,font=font,fill=(20,20,20))
    draw.text((28,58),'原画像 / 保存済みnpj / 固定v6。分類マスクの比較（復元画像ではありません）。',font=small,fill=(55,55,55))
    top=100
    for im,label in zip(resized,labels):
        canvas.paste(im,(0,top));top+=im.height
        draw.text((28,top+18),label,font=small,fill=(35,35,35));top+=84
    destination.parent.mkdir(parents=True,exist_ok=True);canvas.save(destination)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    findings={r['id']:r for r in json.loads((SOURCE/'v6_visual_review.json').read_text())['findings']}
    titles=['汚れの誤判定：改善が見える例','薄い筆画・赤褐色筆画の比較','残っている問題：本文に接する汚れと穴の縁','通常本文の対照']
    for (filename,ids),title in zip(GROUPS.items(),titles):
        board([SOURCE/'comparisons'/(id+'.png') for id in ids],
              [findings[id]['verdict_label']+'：'+findings[id]['title'] for id in ids],OUT/filename,title)
    archive=OUT/'npj_v6_comparison_images.zip'
    sources=list((SOURCE/'comparisons').glob('*.png'))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as zipped:
        for file in sorted(OUT.glob('*.png')):zipped.write(file,'overview/'+file.name)
        for file in sorted(sources):zipped.write(file,'individual/'+file.name)
    dump(OUT/'manifest.json',dict(source_sha256=sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        outputs={p.name:sha(p) for p in OUT.glob('*.png')},individual_pngs=19,overview_pngs=4,
        archive_sha256=sha(archive),presentation_only=True,original_classification_unchanged=True))
    (OUT/'README.md').write_text('''# npjとv6の比較画像

利用者指定に従い、PNGを主な比較成果物としてまとめた。

- `01_stain_comparison.png`：汚れの誤判定で改善が見える3例
- `02_stroke_comparison.png`：薄い黒筆画2例と赤褐色の筆画1例
- `03_remaining_problems.png`：本文と接する汚れ、穴・裂けの誤保護
- `04_controls.png`：通常本文の対照3例
- `npj_v6_comparison_images.zip`：上記4枚と個別比較19枚（全ページ8枚＋切り出し11枚）

すべて原画像・保存済みnpj v1・固定v6の順。同じ原画像に分類マスクを重ねた比較で、復元結果ではない。分類結果は前回と同一。問題8箇所の暫定目視は改善4・部分改善3・未解決1。全資料の精度ではない。
''')
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None and len(zipped.namelist())==23
    print('Packaged: 4 PNG sheets + 19 individual PNGs; ZIP verified',flush=True)


if __name__=='__main__':main()
