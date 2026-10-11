"""Final review of accepted CSC red preservation plus local gap coverage."""
from pathlib import Path
import json,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import validate_local_red as v
import preserve_accepted_csc_red as preserve

OUT=preserve.OUT
METHOD=preserve.METHOD

def main():
    proto=json.loads((OUT/'protocol.json').read_text());rows=json.loads((OUT/'mask_diagnostics.json').read_text())
    results=json.loads((OUT/('ddrm_results_'+METHOD+'.json')).read_text())
    audits=json.loads((OUT/('ddrm_audit_'+METHOD+'.json')).read_text())
    assert len(results)==len(audits)==4
    assert v.driver.digest(v.old.V2)==proto['frozen_csc_sha256']
    assert all(not a['log_errors'] and a['red_source_copy_mismatch_pixels']==0 for a in audits)
    baseline_rows=json.loads((v.OUT/'mask_diagnostics.json').read_text())
    font=ImageFont.truetype(v.review.FONT,24)
    sheet=Image.new('RGB',(2100,4*570+65),'#f4f2ec');draw=ImageDraw.Draw(sheet)
    draw.text((12,10),'原画像／現行CSC赤／局所赤のみ／既存CSC赤保持＋局所補完。固定Howe/DDRM。',font=font,fill='black')
    whole=Image.new('RGB',(1800,4*940),'#f4f2ec');wd=ImageDraw.Draw(whole)
    for i,cid in enumerate(['F04','R01','R02','R03']):
        c=next(c for c in proto['cases'] if c['id']==cid);p=c['page'];r=next(r for r in results if r['id']==cid)
        raw=np.asarray(Image.open(v.ROOT/c['source']).convert('RGB'))
        prevbase=v.BASE/('fixed_holdout_validation' if cid=='F04' else 'red_retention_validation')
        old=np.asarray(Image.open(prevbase/'outputs/howe'/p/'result'/(p+'.png')).convert('RGB'))
        exclusive=np.asarray(Image.open(preserve.PARENT/'outputs/howe_grow5_region'/p/'result'/(p+'.png')).convert('RGB'))
        new=np.asarray(Image.open(v.ROOT/r['output']).convert('RGB'));assert raw.shape==new.shape
        redpath=OUT/'masks'/('howe_'+METHOD)/p/'03_text_red.png';red=v.review.load(redpath)
        assert not np.any(new[red]!=raw[red]);assert v.driver.digest(v.ROOT/r['output'])==r['sha256']
        fig=v.review.figure(OUT/'figures'/(cid+'_既存赤保持_DDRM比較.png'),[raw,old,exclusive,new],
            ['原画像','現行CSC赤','局所赤のみ','既存CSC赤保持＋局所補完'],c['roi_xywh'],cid)
        fig.thumbnail((2080,555));sheet.paste(fig,(10,65+570*i))
        for j,(arr,title) in enumerate(zip([raw,old,new],['原画像','現行CSC赤','既存CSC赤保持＋局所補完'])):
            tile=Image.fromarray(arr);tile.thumbnail((580,870));whole.paste(tile,(j*600+10,i*940+55));wd.text((j*600+10,i*940+10),cid+' '+title,font=font,fill='black')
        for k,roi in c['controls_xywh'].items():
            v.review.figure(OUT/'controls'/(cid+'_'+k+'_DDRM比較.png'),[raw,old,new],['原画像','現行CSC赤','既存CSC赤保持＋局所補完'],roi,cid)
    sheet.save(OUT/'4朱印_既存赤保持と局所補完_DDRM比較.png')
    whole.save(OUT/'4朱印_全ページ比較.png')
    unchanged=[]
    for c in proto['cases']:
        assert v.driver.digest(v.ROOT/c['source'])==c['source_sha256']
        r=next(r for r in rows if r['id']==c['id'])
        maskpath=OUT/'red_masks'/METHOD/c['page']/'03_text_red.png';assert v.driver.digest(maskpath)==r['red_sha256']
        if r['red_page_pixels']:continue
        p=c['page'];saved=v.ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p/'03_text_red.png'
        assert np.array_equal(v.review.load(saved),v.review.load(maskpath))
        bdir='red_retention_validation' if c['id']=='R04' else 'fixed_holdout_validation'
        blackpath=v.BASE/bdir/'masks/howe'/p/'02_text_black.png';black=v.review.load(blackpath)
        assert np.array_equal(black,black&~v.review.load(maskpath))
        unchanged.append(dict(id=c['id'],red_pixels=0,black_sha256=v.driver.digest(blackpath),mask_inputs_identical=True,no_new_inference=True))
    assert len(unchanged)==6;v.driver.save_json(OUT/'6例_入力マスク同一確認.json',unchanged)
    with (OUT/'10例_赤保持比較.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','page','saved_red_page','final_red_page','final_red_roi','preserved_old_roi','excluded_old_roi','brown_binding_red'])
        for r in rows:
            old=next(b for b in baseline_rows if b['id']==r['id'])
            writer.writerow([r['id'],r['page'],old['saved_pixels'],r['red_page_pixels'],r['red_roi_pixels'],r['preserved_old_roi_pixels'],r['excluded_old_roi_pixels'],r['controls'].get('brown_binding','')])
    table='\n'.join(f"| {r['id']} | {next(b for b in baseline_rows if b['id']==r['id'])['saved_pixels']:,} | {r['red_page_pixels']:,} | {r['excluded_old_roi_pixels']:,} |" for r in rows)
    reviewpath=OUT/'visual_review.json'
    visual=json.loads(reviewpath.read_text()) if reviewpath.exists() else []
    findings='\n\n'.join(f"{r['id']}：{r['finding']}" for r in visual if r.get('completed'))
    report=f'''# 既存CSC赤の保持と局所補完：開発比較

Howe文字判定を固定し、CSC赤の検出漏れと茶色い領域の誤検出を調べた。既に観察したtest由来10ページを使う開発実験であり、独立した最終テストではない。

## 最終候補の規則

CSCv2の平均aゲートを外した赤候補を手がかりとする。原画像のLab a値から51×51メディアンのa値を引き、差8以上の強い赤を種とする。色条件はHSV H<=15またはH>=165、S>=25、R−G>=12、R−2G+B>=6。CSC候補から5画素以内でa差2以上の弱い赤へ補完する。連結成分に8画素以上の種と5%以上の種支持を要求する。close3後も元の弱い赤支持の範囲を維持する。

種を半径16で膨張してつながる領域のうち、128画素以上の強い種があるものを受け入れる。その領域では **元の保存CSC赤を保持し、局所補完した赤を加える**。領域外の赤候補は保持しない。全ページで同じ規則を使い、ページIDや確認ROIによる分岐は行わない。数値は本開発用の設定で、論文由来の既定値ではない。

最初の局所色条件だけでは薄い朱印R01の既存赤も落ち、黒の増加を観察したため、既存赤を残す追加条件を作った。領域支持もF03の傷付近・R04の裏移り候補を観察して追加した条件である。この経緯のため、10ページを未見評価として扱わない。

## マスク確認

| 例 | 元の赤・全ページ | 最終候補の赤・全ページ | 確認範囲内で除外された元の赤 |
|---|---:|---:|---:|
{table}

確認範囲には紙地や墨も含まれ、画素数は正解朱印の検出率ではない。R01では色条件だけの段階で落とした4,548画素の元の赤のうち4,434画素を再保持し、範囲内で除外される元の赤は114画素。R02の朱印確認範囲内では元の赤を全画素保持、R03では15画素を除外した。これは保存マスクとの一致で、元のマスクが完全な正解であることを意味しない。

茶色い綴じ側R02（x70,y150,w200,h2800）の赤選択は101,567→局所色補正1,961→領域支持＋既存赤保持0となった。R04の淡い赤印状の像は前ページ朱印の裏移り候補で、表側の正解朱印として数えない。R04と追加5非朱印ページは全ページで赤0、元のHowe黒＋赤の入力と全画素一致している。

## 原寸DDRM

最終候補を朱印4ページで比較した。F04は前段階と黒・赤が完全に同一なので原寸出力を再利用し、R01〜R03は再推論した。最終4結果は新規3＋同一条件再利用1。赤画素と原画像RGBの完全一致、出力の原寸、エラーログなしを検証した。

Howe c=160、Canny sigma=.6/low=.1/high=.4、DDRM epoch350・seed999・20 steps・patch128・overlap32・共通初期ノイズを固定。赤マスクに含まれる画素だけ固定Howe黒から除外する。黒マスクが変わるため、旧条件と全ステップの乱数まで一致するとは保証しない。Howeは前回のgraph_output.binから同じ後処理で復元し、元の赤優先後の黒が前回保存黒と完全一致することを検証済み。

赤は推論後に原画像RGBをコピーする。今回扱うのは検出・保持であり、退色や欠損した朱印を復元する実験ではない。文字・劣化・背景のCSC分類器全体を置き換えるものでもない。元のCSC保存マスクと凍結実装は変更していない。

## 目視評価と次の条件

{findings}

朱印内の黒や、薄い朱印の未検出部分が完全に解消したとは主張しない。最終候補を次の開発比較の設定として固定し、未使用の表側朱印・小さな朱印・退色朱印・茶色い紙地・裏移りで転用を確認する。全データの既定値はまだ変更していない。
'''
    (OUT/'検討結果.md').write_text(report,encoding='utf-8')
    files=[Path(__file__),v.BASE/'validate_local_red.py',v.BASE/'preserve_accepted_csc_red.py',OUT/'protocol.json',OUT/'検討結果.md',OUT/'10例_赤保持比較.csv',OUT/'4朱印_既存赤保持と局所補完_DDRM比較.png']
    if reviewpath.exists():files.append(reviewpath)
    v.driver.save_json(OUT/'final_review.json',dict(artifact_sha256={v.driver.relative(p):v.driver.digest(p) for p in files},visual_reviews_completed=sum(bool(r.get('completed')) for r in visual),
        output_sha256={r['output']:r['sha256'] for r in results},final_outputs=4,new_final_inferences=3,reused=1,unchanged_cases=6,final_test=False))
    print('Prepared final four-page comparison and ten-page mask audit',flush=True)

if __name__=='__main__':main()
