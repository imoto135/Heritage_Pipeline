"""Prepare original-only boundary review candidates; never invent dense GT."""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
from compare_v5_aligned_fade_control import rgb,sha,dump
import prepare_near_paper_teacher as old
ROOT=old.ROOT;OUT=ROOT/'tmp_work/source_boundary_review_candidates_20261023'
SELECT={'B01':'long_curve','B02':'middle_fine','B03':'thin_join','B04':'strong_descender','B05':'middle_fine_join','B06':'lower_kana','B07':'left_weak_descender','B08':'middle_fine_kana'}
def main():
    assert not OUT.exists();OUT.mkdir();cs=json.loads((old.OLD/'selected_sources.json').read_text())['cases'];aa={a['id']:a for a in json.loads((old.OLD/'annotations_source_only.json').read_text())['regions']};records=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    for c in cs:
        path=next(p for p in aa[c['id']]['ink_paths'] if p['id']==SELECT[c['id']]);xy=np.asarray(path['xy']);lo=np.maximum(xy.min(axis=0)-16,0);hi=np.minimum(xy.max(axis=0)+17,256);x,y=map(int,lo);w,h=map(int,hi-lo)
        raw=rgb(ROOT/c['crop']);lab=np.asarray(Image.open(old.OLD/'labels'/f"{c['id']}.png"));crop=raw[y:y+h,x:x+w];partial=lab[y:y+h,x:x+w];dest=OUT/c['id'];dest.mkdir()
        Image.fromarray(raw).save(dest/'source_context.png');Image.fromarray(crop).save(dest/'source_native.png');Image.fromarray(crop).resize((w*4,h*4),Image.Resampling.NEAREST).save(dest/'source_nearest4.png');Image.fromarray(partial).save(dest/'initial_partial_labels.png')
        overlay=crop.copy();overlay[partial==1]=[0,90,255];overlay[partial==2]=[0,180,90]
        # Native pixel comparisons are checked after export.
        canvas=Image.new('RGB',(max(560,8*w+20),4*h+55),'white');d=ImageDraw.Draw(canvas);d.text((0,0),c['id']+' 原画像のみ / 旧部分教師・青=墨 緑=紙（4倍最近傍）',font=font,fill='black');canvas.paste(Image.fromarray(crop).resize((w*4,h*4),Image.Resampling.NEAREST),(0,45));canvas.paste(Image.fromarray(overlay).resize((w*4,h*4),Image.Resampling.NEAREST),(w*4+20,45));canvas.save(dest/'source_and_existing_partial.png')
        assert np.array_equal(rgb(dest/'source_native.png'),crop)
        records.append(dict(id=c['id'],path_id=path['id'],book=c['book'],source=c['source'],source_sha256=c['source_sha256'],parent_crop=c['crop'],roi_in_parent=[x,y,w,h],roi_in_source=[c['roi'][0]+x,c['roi'][1]+y,w,h],source_native_sha256=sha(dest/'source_native.png'),initial_partial_sha256=sha(dest/'initial_partial_labels.png'),review_completed=False,use_for_training=False))
    dump(OUT/'candidates.json',dict(status='Original-only review package. No new dense labels created, no training/eval annotations changed. These8 are training-source candidates, not validation or goldGT.',label_encoding={'0':'unknown, leave unresolved boundaries/fibers/ghosts unknown','1':'existing provisional source-supported ink core','2':'existing provisional paper interior'},candidates=records))
    (OUT/'README.md').write_text('''# 輪郭を含む面教師を整えるための原画像候補

訓練用8冊から、既存墨芯の周辺を原寸で切り出した。新しい面教師はまだ作っておらず、この候補一式は学習・評価に使用していない。

各Bxxには原画像全256窓、原寸小領域、最近傍4倍、既存部分注釈との並列PNG、旧部分ラベルを保存した。初期ラベルは0不明・1暫定墨芯・2暫定紙。**未確認の輪郭は0不明のまま**である。

確認する内容は、見える筆画の内部、確実な紙、どちらとも判断できない境界・裏移り・紙繊維の区分。原画像で見えないつながりを補完せず、別の手法の出力から正解を作らない。曖昧な縁は不明の帯として残し、墨と紙の面を確実な範囲だけで付ける。文字全体が小領域からはみ出す場合はsource_context.pngで確認し、切れた筆画を欠損と誤認しない。

B04は濃い筆画の対照であり、薄字の例だけにはしない。これらは訓練候補で、評価の輪郭GTを兼用しない。専門家による確認済みGTではない。

[candidates.json](candidates.json)に原本のハッシュと座標を記録した。

'''+ '\n'.join(f"- [{c['id']} 原画像と既存部分教師]({c['id']}/source_and_existing_partial.png)" for c in records)+'\n')
    print('Packaged8 source-only boundary review candidates; no new GT or training use',flush=True)
if __name__=='__main__':main()
