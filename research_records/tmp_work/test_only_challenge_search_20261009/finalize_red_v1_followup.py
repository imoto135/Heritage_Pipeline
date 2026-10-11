"""Package the v1/v2 diagnostic and exploratory red/DDRM comparisons."""
from pathlib import Path
import json
import csv
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import validate_red_retention as review
import run_ddrm_pilot as driver

BASE=Path(__file__).resolve().parent
OUT=BASE/'red_detection_followup'

def main():
    records=json.loads((OUT/'diagnostics.json').read_text())
    font=ImageFont.truetype(review.FONT,24)
    sheet=Image.new('RGB',(2100,len(records)*460+70),'#f4f2ec');draw=ImageDraw.Draw(sheet)
    draw.text((15,8),'赤マスク比較：緑＝赤として保持する画素。縮小一覧／正解マスクによる精度評価ではない',font=font,fill='black')
    for row,rec in enumerate(records):
        p=rec['page'];raw=np.asarray(Image.open(driver.ROOT/'data/test'/p.split('_')[0]/(p+'.jpg')).convert('RGB'))
        masks=[review.load(driver.ROOT/f'data/{d}/test'/p.split('_')[0]/p/'03_text_red.png') for d in ['split_dataset_csc_mask','split_dataset_csc_mask_v2']]
        masks.append(review.load(OUT/p/'red_without_global_gate.png'))
        arrays=[raw]
        for m in masks:
            a=raw.copy();a[m]=[20,235,20];arrays.append(a)
        native=review.figure(OUT/p/(rec['id']+'_赤保持範囲比較.png'),arrays,
            ['原画像','CSCv1赤（緑）','CSCv2赤（緑）','v2平均aゲート除外（緑）'],rec['roi_xywh'],rec['id'])
        native.thumbnail((2080,435));sheet.paste(native,(10,65+row*460))
        classpath=OUT/p/(rec['id']+'_CSC分類と赤候補.png')
        im=Image.open(classpath).convert('RGB');legend=Image.new('RGB',(im.width,im.height+45),'#f4f2ec');legend.paste(im,(0,0))
        ImageDraw.Draw(legend).text((12,im.height+8),'分類の色：黒＝黒文字／赤＝赤文字／青＝劣化／薄黄色＝背景。緑＝改善候補の赤マスク',font=font,fill='black')
        legend.save(OUT/p/(rec['id']+'_CSC分類_凡例付き.png'))
        if rec['id']=='R02':
            review.figure(OUT/'R02_茶色い綴じ側_赤保持比較.png',arrays,
                ['原画像','CSCv1赤（緑）','CSCv2赤（緑）','v2平均aゲート除外（緑）'],[70,150,200,2800],'R02')
    sheet.save(OUT/'5例_CSCv1_v2_赤検出比較.png')
    fields=['id','page','v1_red_roi','v2_red_roi','gate_removed_red_roi','v1_red_page','v2_red_page','gate_removed_red_page']
    with (OUT/'朱印検出比較.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            writer.writerow(dict(id=r['id'],page=r['page'],v1_red_roi=r['saved']['v1']['roi_pixels']['red'],
                v2_red_roi=r['saved']['v2']['roi_pixels']['red'],gate_removed_red_roi=r['candidate']['kept_roi_pixels'],
                v1_red_page=r['saved']['v1']['red_page_pixels'],v2_red_page=r['saved']['v2']['red_page_pixels'],
                gate_removed_red_page=r['candidate']['kept_page_pixels']))
    f04=records[0];table='\n'.join(f"| {r['id']} | {r['saved']['v1']['roi_pixels']['red']:,} | {r['saved']['v2']['roi_pixels']['red']:,} | {r['candidate']['kept_roi_pixels']:,} |" for r in records)
    report=f'''# CSCv1/v2の朱印判定確認と赤ゲートの比較

今回のF04（200021063_00002_2）の朱印は、CSCv1では赤として検出されていた。v1の保存ACEを現行v1コードの既定値で再計算し、保存済み赤マスク・黒マスクとも差分0画素で再現した。v1/v2の保存ACE入力も画素単位で一致した。

## 判定の違い

- v1：第1段階で背景・黒文字・劣化の3群へ分け、劣化群内を色で2群へ分ける。F04の赤候補群の平均aは150.58995で、下限133を通過した。
- v2：第1段階で背景・非背景へ分け、非背景群を9チャンネルの色特徴で3群へ分ける。F04の赤候補群の平均aは132.65013で、下限133を通過しなかった。その群は劣化に残り、朱印の薄い部分には背景分類もある。保存赤・黒マスクは凍結実装で再現済み。
- 数値はOpenCVの8bit Lab a値で、中立値は128。物理的なa*値をそのまま示すものではない。同じ下限でもクラスタの作り方が違うため、平均aの値は直接比較できる赤さの尺度ではない。
- v2の赤マスクが空のままHowe文字マスクを使うと、Howeが拾った朱印の形が黒文字扱いでDDRMへ渡る。CSC黒マスクだけの場合は朱印の線が保護対象に入らず消える。DDRMに劣化クラスラベルを直接渡しているわけではなく、使用するのは黒・赤マスクである。

## 同じ規則を5例に適用

改善候補はv2のページ全体の平均a下限だけを外し、既存の連結成分ごとのHSV赤支持率（0.05以上）とclose=3/open=1を維持した。ページ別の条件分岐・閾値変更は行っていない。Howe・DDRMの条件は固定。

表は各確認範囲内で赤として選択された画素数。範囲には紙地・墨も含むため、朱印の検出率・精度ではない。

| 例 | 保存v1 | 保存v2 | v2平均aゲート除外 |
|---|---:|---:|---:|
{table}

R01〜R03は前回確認した表側の朱印。R02は茶色い綴じ側の誤検出も別画像で確認した。R04の淡い赤印状の像は、前ページの強い朱印と鏡像で似ているため裏移りの可能性がある。R04で赤が増えることを改善と数えない。

F04の候補は朱印の線を新たに拾う。ただし、ページ全体の平均aを外すだけでは、R02で既に生じている茶色い綴じ側の赤誤検出を修正しない。正式な共通設定として採用する前に、局所的な赤色判定の改善が必要。

R01〜R03ではゲート除外候補の赤マスクは保存v2と全ページで同一だった。R02の茶色い綴じ側範囲に残る赤候補は101,567画素。R04は確認範囲に144画素、全ページに21,615画素の赤候補が増えたため、朱印付近だけを見る評価では十分でない。v1はR04の淡い赤像も拾っており、表側の朱印だけを選別する規則としては未検証である。

## DDRM出力

F04について、Howe＋現行CSC赤、Howe＋平均aゲート除外赤、Howe＋CSCv1赤を原画像と並べた。追加2条件は原寸全ページを推論し、比較時だけ朱印付近を切り出した。新しい赤マスクと重なるHowe黒画素は削除してから推論している。

epoch350、seed999、20 steps、patch128、overlap32、共通初期ノイズを維持。黒マスクが変わるので、推論の全ステップの乱数まで同一とは保証しない。赤マスク画素は推論後に原画像RGBをコピーする仕様であり、欠損・退色した朱印を復元する実験ではない。出力寸法・赤画素の原画像との一致・エラーログを検証。

目視では、平均aゲート除外によって赤い線が戻ったが、赤として拾えずHowe黒に残った線・縁の黒混入も残った。CSCv1赤への差し替えも黒混入を解消せず、紙地を含む赤候補の保持により粒状の質感が目立った。いずれも今回の比較で正式採用とはしない。

実行時間を短縮するため、R02〜R04の診断では連結成分の赤支持率集計をbincountで一括計算した。色基準・連結性・閾値は同一であり、元の処理との決定的なサンプル比較、および各ページの保存赤マスクとの全画素一致を検証した。F04・R01は元の集計処理で実行済み。凍結CSCファイルは変更していない。

## 次の検討

Howeの文字判定は固定したまま、赤候補ごとの局所的な色支持を調べる。F04の表側朱印を保持し、R02の茶色い汚れを拾わず、R04の裏移り候補を無条件に保護しない条件を比較する。CSCv1への全面切り替えは行わない。

これは既に観察したtest由来ページを使った開発用の比較であり、独立した最終テストではない。source_red_proxyは紙色の影響を受ける補助集計で、正解朱印画素として扱わない。元の固定条件評価・保存CSCマスク・凍結コードは変更していない。
'''
    (OUT/'検討結果.md').write_text(report,encoding='utf-8')
    manifest={str(p.relative_to(driver.ROOT)):driver.digest(p) for p in [Path(__file__),BASE/'check_red_v1_followup.py',BASE/'run_red_gate_followup.py',OUT/'protocol.json',OUT/'diagnostics.json',OUT/'検討結果.md',OUT/'朱印検出比較.csv']}
    driver.save_json(OUT/'review_manifest.json',manifest)
    print('Rendered 5-case audit and report',flush=True)

if __name__=='__main__':main()
