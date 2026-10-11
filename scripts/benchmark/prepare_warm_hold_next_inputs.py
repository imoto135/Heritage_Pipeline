"""Reserve six other-book inputs from original-only screening; no method run."""
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
cv2.setNumThreads(1)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/warm_ink_v6_next_fixed_inputs_20260930'
V6=ROOT/'tmp_work/warm_ink_hold_v6_20260930'
COLLECTION=ROOT/'tmp_work/red_positive_collection_20260930'
SEED=20260930


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((V6/'candidate_freeze.json').read_text())
    excluded=set(freeze['development_book_ids'])
    assert len(excluded)==20 and freeze['variant']=='split_direction'
    parent=json.loads((V6/'complete.json').read_text())
    assert parent['outputs']['candidate_freeze.json']==sha(V6/'candidate_freeze.json')
    candidates=json.loads((COLLECTION/'candidate_pages.json').read_text());best={}
    for r in candidates:
        if r['book'] in excluded:continue
        if r['book'] not in best or (r['saturated_pixels'],-r['index'])>(best[r['book']]['saturated_pixels'],-best[r['book']]['index']):best[r['book']]=r
    strata=dict(high=[r for r in best.values() if r['saturated_pixels']>=50000],
                medium=[r for r in best.values() if 5000<=r['saturated_pixels']<50000],
                low=[r for r in best.values() if r['saturated_pixels']<5000])
    rng=np.random.default_rng(SEED);selected=[]
    for label,rows in strata.items():
        rows=sorted(rows,key=lambda r:r['book']);assert len(rows)>=2
        for i in rng.choice(len(rows),2,replace=False):selected.append(dict(rows[int(i)],screening_stratum=label))
    protocol=dict(seed=SEED,exclude_book_ids=sorted(excluded),one_page_per_book=True,
                  per_book_page='Largest original-only saturated count in saved 100-candidate pool; tie earlier index',
                  strata='Original reduced-image saturated pixel count: high>=50000, medium5000..49999, low<5000',
                  allocation='Two books per stratum, sorted book order and numpy.default_rng(seed), without replacement',
                  remaining_candidate_books=len(best),available_stratum_books={k:len(v) for k,v in strata.items()},
                  caveat='Purposeful color-screened queue, not representative accuracy sample; count is not ink GT',
                  previous_exposure='Saved input-only candidates may have appeared on contact sheets; excluded books have no outputs in this development series',
                  method_applied=False,pixel_GT=False,training_performed=False)
    dependencies={str(p.relative_to(ROOT)):sha(p) for p in [V6/'candidate_freeze.json',V6/'manifest.json',V6/'complete.json',COLLECTION/'candidate_pages.json',COLLECTION/'screening_manifest.json']}
    allocation=dict(source_sha256=sha(Path(__file__)),protocol=protocol,dependencies=dependencies,cases=selected)
    p=OUT/'allocation.json'
    if p.exists() and json.loads(p.read_text())!=allocation:raise RuntimeError('Changed input allocation; use new output version')
    # Allocation written before native decoding or looking at the selected images.
    dump(p,allocation);shutil.copyfile(__file__,OUT/'source_snapshot.py')
    contact=np.full((3*930,2*680,3),255,np.uint8);full=[];links=[]
    for i,row in enumerate(selected):
        original=ROOT/row['original'];assert original.parent.name==row['book']
        image=cv2.imread(str(original));assert image is not None
        h,w=image.shape[:2];scale=min(640/w,820/h)
        thumb=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
        path=OUT/'thumbnails'/(row['id']+'.png');path.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(path),thumb)
        ry,rx=divmod(i,2);y=ry*930;x=rx*680
        cv2.putText(contact,row['id'],(x+12,y+26),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
        cv2.putText(contact,f'{row["screening_stratum"]}, {w}x{h}, sat={row["saturated_pixels"]}',(x+12,y+51),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        contact[y+76:y+76+thumb.shape[0],x+20:x+20+thumb.shape[1]]=thumb
        record=dict(row,original_sha256=sha(original),width=w,height=h,thumbnail=str(path.relative_to(OUT)));full.append(record)
        links.append(f'<figure><figcaption>{row["id"]} · {row["screening_stratum"]} · {w}×{h}</figcaption><a href="{record["thumbnail"]}"><img src="{record["thumbnail"]}" alt="原画像の縮小表示"></a></figure>')
    assert len({r['book'] for r in full})==6 and not ({r['book'] for r in full}&excluded)
    assert cv2.imwrite(str(OUT/'contact_sheet.png'),contact)
    dump(OUT/'selection.json',dict(purpose='Input-only queue reserved after v6 candidate freeze, before applying method',protocol=protocol,cases=full,dependencies=dependencies))
    (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>v6固定確認の入力候補</title><style>body{font-family:system-ui,sans-serif;margin:24px;line-height:1.7}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}img{width:100%;height:auto}figure{margin:0}</style><h1>v6固定確認の入力候補</h1><p>開発20冊を除いた6冊・6ページ。方法出力を見る前に原画像だけの色スクリーニングで確保した。色の量による目的抽出で、代表標本・画素GT・朱の正解ラベルではない。新しいページへの方法適用はまだ実施していない。</p><p><a href="selection.json">選定と原画像のハッシュ</a> · <a href="contact_sheet.png">一覧画像</a> · <a href="../warm_ink_hold_v6_20260930/candidate_freeze.json">固定した候補</a> · <a href="verification.json">検証</a></p><div class="grid">'+''.join(links)+'</div></html>\n')
    table='\n'.join(f'| {r["id"]} | {r["book"]} | {r["screening_stratum"]} | {r["width"]}×{r["height"]} |' for r in full)
    (OUT/'report.md').write_text(f'''# 次の固定確認に使う入力候補

v5の完了・分析・v6の修正を経て、[v6の経路別方式](../warm_ink_hold_v6_20260930/candidate_freeze.json)を固定した。既知21ページ・20冊を除いた冊から、方法出力を見る前に6冊・6ページを確保した。

原画像だけで作った過去の100候補から、残る17冊の候補を色の量の3層に分け、固定乱数{SEED}で各2冊を選んだ。各冊は元の飽和色画素数が最大の候補1ページ。色の量はインクの正解数ではなく、表紙・大きな色面・茶色い縁などが含まれ得る。以前の入力一覧への露出もあり、完全に未見の画像とは呼ばない。

| ページ | 冊 | 入力色量の層 | 原寸 |
|---|---|---|---|
{table}

冊の非重複、割当後のID不変、原画像ハッシュ、縮小表示の元画素からの一致を確認する。方法の適用・ニューラル学習・手動画素GTはまだ実施していない。次は入力の目視所見を記録し、選定を方法出力に合わせて変えず、固定した条件でv2/v4/v5/v6を確認する。出力を見て調整する場合は開発へ移し、最終確認には残さない。
''')
    dump(OUT/'verification.json',dict(status='pending'))
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['complete.json','verification.json','run.log']]
    complete={str(p.relative_to(OUT)):sha(p) for p in sorted(files)};dump(OUT/'complete.json',dict(status='complete',outputs=complete))
    for rel,h in dependencies.items():assert sha(ROOT/rel)==h,rel
    for rel,h in complete.items():assert sha(OUT/rel)==h,rel
    for expected,row in zip(selected,full):
        assert expected['id']==row['id'] and expected['book']==row['book']
        src=ROOT/row['original'];assert sha(src)==row['original_sha256']
        native=cv2.imread(str(src));scale=min(640/native.shape[1],820/native.shape[0])
        thumbnail=cv2.resize(native,(max(1,round(native.shape[1]*scale)),max(1,round(native.shape[0]*scale))),interpolation=cv2.INTER_AREA)
        assert np.array_equal(thumbnail,cv2.imread(str(OUT/row['thumbnail'])))
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src'] and v:assert (OUT/v).is_file(),v
    Links().feed((OUT/'index.html').read_text())
    dump(OUT/'verification.json',dict(status='passed',books=6,pages=6,development_book_overlap=[],
             allocation_preserved=True,original_hashes_and_thumbnails_checked=True,source_and_dependencies_hashed=True,
             output_hashes_checked=len(complete),viewer_links_checked=True,browser_rendering_tested=False,
             method_applied=False,pixel_GT=False,neural_training=False,independent_accuracy_result=False))
    print(json.dumps(dict(status='passed',selected_ids=[r['id'] for r in full]),indent=2),flush=True)


if __name__=='__main__':main()
