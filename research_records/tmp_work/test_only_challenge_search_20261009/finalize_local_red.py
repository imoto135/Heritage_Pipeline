"""Package full-native red experiments, source-inclusive PNGs and provenance."""
from pathlib import Path
import json,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import validate_local_red as v

OUT=v.OUT
REG=OUT/'regional_refinement'
METHOD='grow5_region'

def main():
    proto=json.loads((REG/'protocol.json').read_text());rows=json.loads((REG/'mask_diagnostics.json').read_text())
    assert v.driver.digest(v.old.V2)==proto['frozen_csc_sha256']
    results=json.loads((REG/('ddrm_results_'+METHOD+'.json')).read_text())
    audits=json.loads((REG/('ddrm_audit_'+METHOD+'.json')).read_text())
    assert len(results)==len(audits)==4
    assert all(not r['log_errors'] and r['red_source_copy_mismatch_pixels']==0 for r in audits)
    dependencies={}
    for c in proto['cases']:
        assert v.driver.digest(v.ROOT/c['source'])==c['source_sha256']
        dependencies[c['source']]=c['source_sha256']
    cases=[next(c for c in proto['cases'] if c['id']==r['id']) for r in results]
    cases.sort(key=lambda c:['F04','R01','R02','R03'].index(c['id']))
    font=ImageFont.truetype(v.review.FONT,24)
    sheet=Image.new('RGB',(1800,4*570+65),'#f4f2ec');draw=ImageDraw.Draw(sheet)
    draw.text((12,10),'固定Howe/DDRM：原画像／現行CSC赤／局所赤＋領域支持。縮小一覧。',font=font,fill='black')
    whole=Image.new('RGB',(1800,4*940),'#f4f2ec');wd=ImageDraw.Draw(whole)
    for i,c in enumerate(cases):
        p=c['page'];cid=c['id'];r=next(r for r in results if r['id']==cid)
        assert v.driver.digest(v.ROOT/r['output'])==r['sha256']
        new=np.asarray(Image.open(v.ROOT/r['output']).convert('RGB'))
        raw=np.asarray(Image.open(v.ROOT/c['source']).convert('RGB'))
        prevbase=v.BASE/('fixed_holdout_validation' if cid=='F04' else 'red_retention_validation')
        oldpath=prevbase/'outputs/howe'/p/'result'/(p+'.png')
        previous=np.asarray(Image.open(oldpath).convert('RGB'));dependencies[v.driver.relative(oldpath)]=v.driver.digest(oldpath)
        assert new.shape==raw.shape
        redpath=REG/'masks'/('howe_'+METHOD)/p/'03_text_red.png';red=v.review.load(redpath)
        assert not np.any(new[red]!=raw[red])
        for label,folder in [('parent',OUT),('regional',REG)]:
            folderpath=folder/'ddrm_figures'/(cid+'_局所赤_DDRM比較.png')
            if label=='regional':
                im=Image.open(folderpath).convert('RGB');im.thumbnail((1780,555));sheet.paste(im,(10,65+570*i))
        for j,(arr,title) in enumerate(zip([raw,previous,new],['原画像','現行CSC赤','局所赤＋領域支持'])):
            tile=Image.fromarray(arr);tile.thumbnail((580,870));whole.paste(tile,(j*600+10,i*940+55));wd.text((j*600+10,i*940+10),cid+' '+title,font=font,fill='black')
    sheet.save(OUT/'4朱印_局所赤改善_DDRM比較.png')
    whole.save(OUT/'4朱印_全ページ_DDRM比較.png')
    # Cases with red=0 have exactly unchanged mask inputs; inference is not repeated.
    unchanged=[]
    for c in proto['cases']:
        row=next(r for r in rows if r['id']==c['id'])
        if row['red_page_pixels']:continue
        p=c['page'];saved=v.ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p/'03_text_red.png'
        newpath=REG/'red_masks'/METHOD/p/'03_text_red.png'
        assert np.array_equal(v.review.load(saved),v.review.load(newpath))
        if c['id']=='R04':black=v.BASE/'red_retention_validation/masks/howe'/p/'02_text_black.png'
        else:black=v.BASE/'fixed_holdout_validation/masks/howe'/p/'02_text_black.png'
        mask=v.review.load(black);assert np.array_equal(mask,mask&~v.review.load(newpath))
        unchanged.append(dict(id=c['id'],red_pixels=0,black_unchanged=True,black_sha256=v.driver.digest(black),
                              source_sha256=c['source_sha256'],note='Mask identity audit only; no new inference'))
    v.driver.save_json(OUT/'6非朱印例_入力同一確認.json',unchanged)
    assert len(unchanged)==6
    fields=['id','page','saved_red_page','local_red_page','regional_red_page','regional_red_roi','brown_binding_red_pixels']
    original=json.loads((OUT/'mask_diagnostics.json').read_text())
    with (OUT/'局所赤_10例比較.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            before=next(o for o in original if o['id']==r['id'])
            writer.writerow(dict(id=r['id'],page=r['page'],saved_red_page=before['saved_pixels'],
                local_red_page=before['variants']['grow5']['page_pixels'],regional_red_page=r['red_page_pixels'],
                regional_red_roi=r['red_roi_pixels'],brown_binding_red_pixels=r['controls'].get('brown_binding','')))
    table='\n'.join(f"| {r['id']} | {next(o for o in original if o['id']==r['id'])['saved_pixels']:,} | {r['red_page_pixels']:,} | {r['red_roi_pixels']:,} |" for r in rows)
    report=f'''# CSC赤候補の局所色補正と領域支持

Howe文字判定（c=160、Canny sigma=.6/low=.1/high=.4）を固定し、CSCの赤候補を原画像の局所的な色で補正した。既に観察したtest由来10ページを使った開発実験。独立最終テスト・正解マスクによる精度評価ではない。

## 規則

1. CSCv2の平均aゲートを外した赤候補を出発点とする。ACE画像でのHSV連結成分支持率.05とclose3/open1は前回と同じ。
2. 原画像のLab a値と51×51メディアンの差を使う。周囲よりaが8以上高く、色相・彩度・RGBの赤色条件を満たす候補画素を強い赤の種とする。
3. CSC赤候補から5画素以内で、a差2以上の赤色支持を持つ画素へ広げる。連結成分に8画素以上の強い種と5%以上の種支持を要求する。close3後も元の弱い赤支持との積を取るため、形態処理だけで中立の墨を赤扱いしない。
4. 第1比較でF03に173画素、R04に695画素の小さな候補が生じたため、追加の開発段階で「半径16の膨張でつながる強い赤の種が128画素以上ある領域」を要求した。全ページに同じ規則を適用。ページID・ROIで採否を切り替えない。

色条件はOpenCV HSVのH<=15またはH>=165、S>=25、R−G>=12、R−2G+B>=6。これらは本開発用の候補設定であり、論文からそのまま採用した数値ではない。局所窓・距離は原寸画素単位。

## マスク確認

表はページ全体／確認範囲内で赤として選択した画素数。範囲内に墨や紙地も含むため、朱印の検出率とは呼ばない。

| 例 | 現行赤・全ページ | 改善候補・全ページ | 改善候補・確認範囲 |
|---|---:|---:|---:|
{table}

F04の現行赤0画素から35,331画素を拾い、朱印の外枠と内部の線を保護できる範囲が広がった。R02の茶色い綴じ側範囲（x70,y150,w200,h2800）で赤扱いされる画素は、現行101,567→局所補正1,961→領域支持0となった。これはこの範囲の誤分類監査であり、全資料に対する誤検出率ではない。

R04の淡い赤像は前ページ朱印の裏移り候補で、表側の朱印としての正解ではない。領域支持後はR04全ページとF03の傷付近を含む追加5非朱印ページで赤マスク0。これら6ページは既存Howe黒＋赤入力と画素単位で同じになり、新しい推論は繰り返していない。

R03のblack_text確認範囲には198画素の赤が残る。この範囲は本文だけの正解領域ではなく、朱印との重なり・色のある点も含む。矩形内の赤画素をすべて墨への誤分類と見なさず、比較画像で確認する。

## 推論と検証

第1段階の局所補正は朱印4ページで原寸DDRMを実行した。領域支持の追加後はF04で黒・赤マスクが第1段階と完全に同一のため結果を再利用し、R01〜R03は原寸で再推論した。最終比較は4ページ（新規3＋完全同一条件の再利用1）。

DDRM epoch350、seed999、20 steps、patch128、overlap32、共通初期ノイズを維持。黒文字マスクは固定Howe出力から新しい赤画素だけを除外する。R01〜R03では前回のgraph_output.binから同じパディングと後処理でHoweを復元し、元の赤マスクで優先処理した黒が前回保存黒と全画素一致することを確認した。Howeの閾値・エネルギーや孤立点除去は変更していない。

赤は推論後に原画像RGBをコピーする仕様。今回の改善は朱印の検出・保持であり、退色や欠損の復元ではない。新しい黒マスクが旧条件と異なるため全ステップの乱数一致は保証しない。出力原寸・赤画素の原画像との完全一致・エラーログなしを検証した。従来のCSCマスク・凍結実装・既存評価成果物は変更していない。

## 目視結果と採否

F04は黒化していた朱印の外枠・内部の赤い線を広く保持できるようになった。ただし内部には黒い点・線が残る。前回から使った朱印確認矩形[x1780,y2600,w240,h400]でmaxRGB<90の画素を数えると、原画像0、現行22,124、平均aゲート除外5,121、局所赤＋領域支持1,338。これは確認矩形の暗画素集計であり、正解マスクを用いた黒混入率ではない。

R02は朱印を保持し、茶色い綴じ側を赤として原画像からコピーしていた長い帯が消えた。綴じ側の注記と本文は比較画像で確認する。全ページの汚れや紙の傷がすべて直ったとは主張しない。

R01の薄く損傷した二つの朱印では、現行より拾える薄い線が増える一方、局所色の条件で既存の赤保持画素も落ち、一部の黒い点・線が増えた。赤の選択画素総数が増えても、画質が一律に改善したとは判断できない。この悪化を理由に、現行赤マスクを一律にこの候補へ置き換えない。

## 採用範囲

局所赤＋領域支持は開発用候補にとどめ、共通既定値には採用しない。次は、十分な赤のまとまりがある領域では既存CSCの赤保持画素を残し、同じ領域に局所色による取りこぼし補完を加える方式を比較する。R02の茶色い綴じ側のような孤立した赤候補は領域支持で除く。この方式は今回の最終4ページの推論にはまだ適用していない。

文字・劣化・背景の全分類器を置き換えたものではなく、CSC赤候補の補正と、DDRM入力の黒／赤優先処理の実験である。今回の観察に合わせて追加条件を作ったので、改善案を固定し、未使用の表側朱印・小さな朱印・裏移り・茶色い紙地で確認してから共通設定として扱う。
'''
    (OUT/'検討結果.md').write_text(report,encoding='utf-8')
    files=[Path(__file__),v.BASE/'validate_local_red.py',v.BASE/'refine_local_red_regions.py',OUT/'protocol.json',REG/'protocol.json',
        OUT/'検討結果.md',OUT/'局所赤_10例比較.csv',OUT/'4朱印_局所赤改善_DDRM比較.png',REG/('ddrm_audit_'+METHOD+'.json')]
    manifest={v.driver.relative(p):v.driver.digest(p) for p in files}
    v.driver.save_json(OUT/'final_review.json',dict(dependencies=dependencies,artifact_sha256=manifest,
        final_outputs=4,new_final_inferences=3,reused_final_inferences=1,unchanged_mask_examples=6,final_test=False))
    print('Finalized four full-native results and ten mask audits',flush=True)

if __name__=='__main__':main()
