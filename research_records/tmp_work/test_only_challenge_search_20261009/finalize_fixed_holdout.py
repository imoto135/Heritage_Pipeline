"""Write review and manuscript table only after all fixed-condition outputs exist."""
from pathlib import Path
import csv
import json
import re
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import validate_fixed_holdout as v


def main():
    out=v.OUT;proto=json.loads((out/'protocol.json').read_text())
    complete=json.loads((out/'ddrm_complete.json').read_text())
    diag=json.loads((out/'mask_diagnostics.json').read_text())
    review=json.loads((out/'visual_review.json').read_text())
    assert len(complete['results'])==18 and len(review['cases'])==6
    assert all(c['review_complete'] for c in review['cases'])
    v.verify_inputs()
    for row in json.loads((out/'ddrm_runtime_manifest.json').read_text())['files']:
        assert v.driver.digest(v.ROOT/row['path'])==row['sha256']
    for r in complete['results']:
        path=v.ROOT/r['output'];assert v.driver.digest(path)==r['sha256']
        case=next(c for c in proto['cases'] if c['id']==r['id'])
        assert list(Image.open(path).size)==case['input_size']
    # A source-inclusive whole-page overview complements the native ROI crops.
    overview=Image.new('RGB',(2040,950*6),'#eee')
    draw=ImageDraw.Draw(overview)
    overview_font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',23)
    for i,case in enumerate(proto['cases']):
        page=case['page']
        paths=[v.ROOT/case['source']]+[out/'outputs'/method/page/'result'/(page+'.png') for method in v.METHODS]
        for j,(path,label) in enumerate(zip(paths,['原画像','CSC黒＋CSC赤','Howe黒＋CSC赤','孤立点条件＋CSC赤'])):
            im=Image.open(path).convert('RGB');im.thumbnail((490,880),Image.Resampling.LANCZOS)
            x=j*510;y=i*950;overview.paste(im,(x+(510-im.width)//2,y+60))
            draw.text((x+8,y+8),case['id']+' '+label,font=overview_font,fill='black')
    overview.save(out/'全ページ6例_縮小一覧.png')
    # Representative export: identical processing/scale across each row.
    rep=Image.new('RGB',(2472,2400),'#f4f2ec');draw=ImageDraw.Draw(rep)
    rep_font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',30)
    rep_meta=[]
    for cid,scale,y0,label in [('F02',1,0,'(a) 本文の筆画保持'),('F04',2,1100,'(b) 赤候補がない朱印')]:
        case=next(c for c in proto['cases'] if c['id']==cid);page=case['page'];x,y,w,h=case['roi_xywh']
        paths=[v.ROOT/case['source']]+[out/'outputs'/m/page/'result'/(page+'.png') for m in ['csc','howe']]
        for i,(path,title) in enumerate(zip(paths,['原画像','CSC黒＋CSC赤','Howe黒＋CSC赤'])):
            tile=Image.open(path).convert('RGB').crop((x,y,x+w,y+h))
            if scale!=1:tile=tile.resize((w*scale,h*scale),Image.Resampling.NEAREST)
            px=12+i*824;rep.paste(tile,(px,y0+58));draw.text((px,y0+8),label+' / '+title,font=rep_font,fill='black')
        rep_meta.append(dict(id=cid,roi_xywh=case['roi_xywh'],scale=scale,interpolation='nearest if enlarged',
            note='All three tiles within a row have the same crop and scale. ROI and case were selected before new masks/inference.'))
    rep.save(out/'論文用_代表比較_本文と朱印.png',dpi=(300,300))
    v.driver.save_json(out/'representative_figure.json',dict(rows=rep_meta,dpi=300,
        selection='Representative successes/limits selected after reviewing completed outputs; not additional independent test cases'))
    count=sum(len(r['removed_components']) for r in diag)
    pixels=sum(r['removed_pixels'] for r in diag)
    assert count==64 and pixels==996
    # Preserve every native crop. A large contact sheet provides navigation.
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    component_figures=[]
    for case,row in zip(proto['cases'],diag):
        for i,comp in enumerate(row['removed_components']):
            path=out/'component_ddrm_figures'/f"{case['id']}_{i:03d}.png"
            im=Image.open(path).convert('RGB')
            component_figures.append((f"{case['id']} comp {i} {comp['xywh']}",im))
    tile_w=max(im.width for _,im in component_figures)+12
    tile_h=max(im.height for _,im in component_figures)+34
    board=Image.new('RGB',(tile_w*2,tile_h*((count+1)//2)),'#ddd');draw=ImageDraw.Draw(board)
    for i,(label,im) in enumerate(component_figures):
        x=i%2*tile_w;y=i//2*tile_h;board.paste(im,(x,y+30));draw.text((x+5,y+3),label,font=font,fill='black')
    board.save(out/'全64除去候補_DDRM比較.png')
    rows=[]
    for case,d,a,r in zip(proto['cases'],diag,complete['audits'],review['cases']):
        assert case['id']==d['id']==a['id']==r['id']
        rows.append(dict(id=case['id'],page=case['page'],book=case['page'].split('_')[0],
            source_review=case['reason'],howe_vs_csc=r['howe_vs_csc'],remaining_limit=r['remaining_limit'],
            isolated_review=r['isolated_review'],removed_components=len(d['removed_components']),
            removed_mask_pixels=d['removed_pixels'],saved_red_pixels=d['red_pixels'],
            final_test=False,dense_ground_truth=False,figure='ddrm_figures/'+case['id']+'_DDRM比較.png'))
    with (out/'論文用_事例比較表.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    new=sum(not r['reused'] for r in complete['results'])
    reused=len(complete['results'])-new
    table='\n'.join(f"| [{r['id']}]({r['figure']}) | {r['source_review']} | {r['howe_vs_csc']} | {r['remaining_limit']} | {r['removed_components']} / {r['removed_mask_pixels']} |" for r in rows)
    red=json.loads((out/'red_recheck/diagnostic.json').read_text())
    trace=red['intermediate_trace'];mean_a=trace['stage2_mean_a'][trace['red_cluster']]
    small=json.loads((out/'F06_small_text_mask_review.json').read_text())
    assert all(r['removed_mask_pixels']==0 for r in small['regions'])
    dark_before=sum(c['baseline_dark_pixels'] for a in complete['audits'] for c in a['removed_components'])
    dark_after=sum(c['candidate_dark_pixels'] for a in complete['audits'] for c in a['removed_components'])
    report=f'''# 追加資料6ページ：固定条件でのDDRM比較

2026-10-09。**Howeを黒文字の基本候補にする方針は維持する。** カラー資料ではCSCの筆画の断片化を補う例があり、白黒資料では両方式が近い例もある。孤立点除去は追加条件として扱う。今回の朱印1例は保存CSC赤が0画素で、CSC条件ではほぼ消失、Howe条件では印影が黒く残るため、赤の検出漏れを独立した課題として記録する。

[追加6ページの比較PNG](追加6ページ_固定条件DDRM比較.png)は左から原画像、CSC黒＋CSC赤、Howe黒＋CSC赤、Howe黒に固定の孤立点除去を加えた条件。原寸ROIを比較した。[全ページの縮小一覧](全ページ6例_縮小一覧.png)、[論文用の事例比較表CSV](論文用_事例比較表.csv)、[本文と朱印の代表比較PNG](論文用_代表比較_本文と朱印.png)も保存した。代表比較の朱印行は全条件とも同じ2倍表示で、出力自体は原寸のまま。

## 資料と評価範囲

4資料・6ページはすべて`data/test`由来で、分割後testとバイト一致し、資料IDはtrain/valに存在しないことを確認した。[分割確認](split_membership.json)。4資料は、このワークフローの先行する全ページCSC／Howe比較（旧6例・別資料4例・赤保持4例）には含まれていない。ただし、2つの白黒資料では別ページのマスクのみの検討歴がある。

原画像の一覧とスクリーニング値は既存で、対象ページには過去のOtsu推論ファイルもある。今回の選定でそれらのOtsu出力は見ていないが、**完全未使用の独立最終testではなく、固定条件による追加資料検証**とする。ディレクトリ名にholdoutを含むが、[protocol.json](protocol.json)では`final_test=false`を明記した。資料の選択、主ROIと確認矩形は新しいHoweマスクとDDRM出力を見る前に固定した。出力を見てページを除外したり条件を変更したりしていない。画素単位の正解ラベルがないのでPrecision/Recall、PSNR等の最終性能値は報告しない。

## 結果

| 例 | 確認対象 | HoweとCSCの比較 | 残る限界 | 孤立点除去：成分 / 画素 |
|---|---|---|---|---:|
{table}

青灰色の像はF02の隣接ページ原画像を鏡像でも確認した。[比較](F02_隣接原画像_ROI比較.png)。精密な位置合わせや画素正解はないため、対応する背面筆画を定量確定した評価とは扱わない。F03の上部の枝状像も、原因の確定や紙の欠損復元を評価したものではない。

## 孤立点除去の位置づけ

固定した規則が選んだのは全ページで計{count}成分、{pixels:,}マスク画素。黒マスクをDDRMへ渡す前の処理で、DDRM本体の変更ではない。[全除去候補の比較PNG](全64除去候補_DDRM比較.png)と各原寸切り出しを保存し、選択した成分を一部だけ紹介する形にはしなかった。

除去成分を囲む各最小矩形で、出力の暗画素（RGB平均128以下）の合計は{dark_before:,}から{dark_after:,}に変わった。これは局所出力の記述であり、成分矩形内の全暗画素を汚れの正解と見なした値ではない。

F06の日付と署名を含む二つの文字範囲では、孤立点条件によるマスク変更がともに0画素だった。[小文字の確認記録](F06_small_text_mask_review.json)。初めの広い確認矩形には75画素の除去候補が入るが、これは日付の下にある独立点であり、汚れか記号かは未確定。矩形内の変更数を文字欠落数と同一視せず、除去成功とも断定しない。ほかの本文確認矩形にも極小点が入る。目視で主な本文や日付・署名の字形の明らかな欠落は見られないが、句点等を含めた点や独立記号の正解注釈はなく、すべての除去を正解とは断定しない。

F02は孤立点によるマスク変更が0で、Howe出力を全ページの黒・赤マスク一致確認付きで再利用した。変更のある条件間では、共有初期ノイズだけで全段の画素別乱数は一致しない。背景の微小なRGB差や広い確認矩形の変化数を、文字の改善・欠落として採点しない。[出力・局所計測](ddrm_complete.json)。

## 新しい朱印例の検出漏れ

F04 `200021063_00002_2`には、原画像で読める向きの朱印がある。前後ページの鏡像の同位置に対応する朱印は見当たらず、先行する旧06の薄い鏡像とは状況が異なる。[隣接ページ比較](F04_隣接原画像_ROI比較.png)。今回は表側朱印と見られる例として扱うが、実物による確認はしていない。

全6ページの保存赤マスクは0画素であり、今回の出力チェックで「赤マスク内は原画像と一致」となっても、赤を保持した根拠にはならない。F04ではCSC黒が朱印をほぼ拾わず、Howe黒は朱印の形を拾う。赤合成で守る画素がないため、最終出力はそれぞれ朱印の消失／黒化となる。[朱印の比較](ddrm_figures/F04_DDRM比較.png)。以前の赤保持試験で赤候補内の原画像RGBが完全に戻せた事実とは区別する。

保存ACE画像を凍結CSC実装・既定値で別途再計算し、保存黒・赤マスクとともに全画素一致を確認した。赤候補クラスタの平均a値は**{mean_a:.5f}**で、下限133を下回り、HSV支持による確認に入る前に赤候補が落ちた。これはOpenCVの8bit Lab表現のa値である。[診断記録](red_recheck/diagnostic.json)、[保存赤と再計算赤の比較](red_recheck/F04_保存赤と再計算赤.png)。この診断で作ったマスクは今回の推論に使用していない。132.65という値を見て閾値を下げる調整は行わなかった。

## 固定条件と検証

Howeは既存のローカル実装を使い、c=160、Canny sigma=0.6、低閾値比0.1、高閾値比0.4で固定した。作者の基本エネルギーの移植であり、Cannyの実装と自動パラメータ探索は作者のMATLAB実装と同一ではない。孤立点は既存`isolated_csc_v2`の成分形状・CSC黒支持・距離・近接保護規則を変更せず適用し、旧6ページのマスクを全画素再現する確認も行った。

DDRMは凍結コード、epoch350、伊勢物語背景、seed999、20段、原寸128pxパッチ、重なり32、共有初期ノイズで固定。保存CSC赤は同じものを全条件に使い、両方式の黒から赤候補を除いて赤を優先した。今回の赤はすべて0なので、この優先処理で変わった画素はない。現行DDRMが直接使う分類マスクは黒と赤で、CSC背景・劣化の全クラスをDDRMに入力した評価ではない。

新規推論{new}出力＋同一マスクによる再利用{reused}出力、計18条件が完了。全出力の原寸サイズ・ハッシュ、新規ログにエラーなし、入力・マスク・凍結実装が不変であることを確認した。[実装スナップショット](implementation_manifest.json)、[DDRM実行コードと環境](ddrm_runtime_manifest.json)、[マスク診断](mask_diagnostics.json)。`00_input.png`は保存ACE画像なので原JPEGとの画素不一致は想定どおりであり、全6ページのACE生成まで独立再現したわけではない。[入力確認](saved_csc_source_check.json)。

## 今後の検討と論文での扱い

1. **黒文字はHowe固定を基本候補として進める。** 白黒資料で差が小さい例も含め、文字保持と不要像残存を分けて記述する。紙面損傷の枝状像を黒く残すことは修復成功とは扱わない。
2. **孤立点除去は追加条件として残す。** 影響は今回も局所的で、句読点や独立記号の注釈を増やして採用範囲を判断する。基本条件へ自動的には組み込まない。
3. **朱印は細かな黒点より検出漏れを優先する。** 赤候補をページ全体のクラスタ平均で一括棄却する条件と、局所の朱色支持の関係を開発資料側で検討する。茶色い染み・綴じや赤い裏移りを一緒に保持しないよう別々に評価する。
4. 朱印の色と残存線を安定して保持できた後に、薄れを見やすくする処理と欠けた印影の推定を別実験にする。現段階の合成結果を朱印修復の成功とは記載しない。
5. 今回までの閲覧・調整に使ったページは開発／追加検証として論文に明示する。最終性能評価には未確認の資料単位の評価集合と正解注釈を用意し、この6例を見た変更を同じ6例の独立test成績として報告しない。データの再分割は今回行っていない。
'''
    (out/'検討結果.md').write_text(report,encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)',report):assert (out/link).exists(),link
    v.driver.save_json(out/'final_review.json',dict(checked_outputs=18,new_outputs=new,reused_outputs=reused,
        removed_components=count,removed_mask_pixels=pixels,dark_pixels_component_boxes_before=dark_before,
        dark_pixels_component_boxes_after=dark_after,final_test=False,
        report_sha256=v.driver.digest(out/'検討結果.md'),csv_sha256=v.driver.digest(out/'論文用_事例比較表.csv'),
        visual_review_sha256=v.driver.digest(out/'visual_review.json'),finalizer_sha256=v.driver.digest(Path(__file__))))
    print('18 outputs, all removal candidates, CSV and report verified',flush=True)


if __name__=='__main__':main()
