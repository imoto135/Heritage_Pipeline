"""Review all short-gap candidates, fixed ROIs, and colour-preserving ablation."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import ROOT, FONT_PATH, read, write, sha, dump
from package_saved_npj_inference_v6_review import check
from package_saved_npj_role_separated_review import figure

PILOT=ROOT/'tmp_work/npj_docres_gap_gate_pilot_20261002'
OUT=ROOT/'tmp_work/npj_docres_gap_gate_review_20261002'
NOTES=ROOT/'tmp_work/npj_docres_gap_gate_notes_20261002.md'


def paint(a,mask,color):
    out=a.copy();out[mask]=np.rint(.5*a[mask]+.5*np.array(color)).astype(np.uint8);return out


def main(render_only):
    assert not (OUT/'complete.json').exists(), 'Frozen review'
    OUT.mkdir(parents=True,exist_ok=True);manifest=check(PILOT)
    records=[];gaps=[];figures=[];selected=[]
    for spec in manifest['frames']:
        fid=spec['id'];src=PILOT/'native'/fid
        a={k:read(src/(k+'.png')) for k in ['saved_input','saved_npj','ungated_output','gate_05_output','gate_09_output']}
        m={k:read(src/(k+'_mask.png'),True) for k in ['ungated','gate_05','gate_09','docres_ink','old_protection_disagreement','red','hold','extra','protect']}
        assert np.array_equal(a['saved_input'],read(ROOT/spec['input']))
        assert np.array_equal(a['saved_npj'],read(ROOT/spec['saved_npj']))
        for key in ['ungated','gate_05','gate_09']:
            assert np.array_equal(a[key+'_output'][~m[key]],a['saved_npj'][~m[key]])
            assert not (m[key]&(m['red']|m['hold']|m['extra'])).any()
        assert not (m['gate_09']&~m['gate_05']).any()
        diag=json.loads((src/'diagnostics.json').read_text())
        keys=['saved_input','saved_npj','ungated_output','gate_09_output']
        labels=['実保存入力・原画像','実保存npj','選別なし・同じ補完','DocRes選別0.9＋補完']
        footers=['全列は同じ保存入力に由来。新モデルの色出力は一切使わない。',
                 '補完は原画像で観測できる短い接続部だけ。紙面・朱を処理する方式ではない。',
                 '更新外・朱候補・保留は保存npj一致。既知資料の試作で筆画GTは未確認。']
        name='overview_'+fid+'.png';figure([a[k] for k in keys],labels,OUT/name,
            spec['page']+'：同じ補完でDocRes選別の差を比較',footers,overview=True);figures.append(name)
        overlays=[a['saved_input'],paint(a['saved_input'],m['protect']|m['hold'],[240,100,35]),
                  paint(a['saved_input'],m['docres_ink'],[240,100,35]),
                  paint(a['saved_input'],m['old_protection_disagreement'],[40,190,245])]
        for r in spec['regions']:
            x,y,w,h=r['xywh'];s=np.s_[y:y+h,x:x+w];name=r['id']+'.png'
            figure([a[k][s] for k in keys],labels,OUT/name,r['title'],footers);figures.append(name)
            name='roles_'+r['id']+'.png'
            figure([v[s] for v in overlays],['実保存入力・原画像','旧保護＋保留・青','DocResインク・青','不一致候補・黄'],OUT/name,
                r['title']+'：マスクの不一致は編集せず表示',
                ['黄＝旧保護／保留とDocRes高信頼背景の不一致。文字ではない正解を意味しない。',
                 '朱・追加暖色保留は黄候補から除外。原寸の同じ実保存入力を使う。',
                 '不一致を汚れと決めず、保護範囲全体の置換や背景全域の編集はしない。']);figures.append(name)
            records.append(dict(frame=fid,**r,mask_pixels={k:int(v[s].sum()) for k,v in m.items()},
                changed_pixels={k:int(np.any(v[s]!=a['saved_npj'][s],axis=2).sum()) for k,v in a.items() if k.endswith('_output')}))
        # Every old-heuristic accepted component, including rejected/ambiguous ones.
        for c in diag['components']:
            gx,gy,gw,gh=c['xywh'];h,w=a['saved_npj'].shape[:2]
            x=max(0,gx-22);y=max(0,gy-22);x2=min(w,gx+gw+22);y2=min(h,gy+gh+22)
            s=np.s_[y:y2,x:x2];name=f'gaps/gap_{len(gaps)+1:03d}_{fid}_{gx}_{gy}.png'
            (OUT/'gaps').mkdir(exist_ok=True)
            probability=c['docres_mean_ink_probability']
            figure([a[k][s] for k in keys],labels,OUT/name,
                f'{fid} / [{gx},{gy}] / {"採択" if c["selected_09"] else "見送り"}',
                [f'旧候補 {c["pixels"]}画素。成分内平均インク確率 {probability:.5f}。',
                 f'0.5選別={c["selected_05"]} / 0.9選別={c["selected_09"]}。表示範囲は候補の周囲22px。',
                 '同じ輝度倍率の補完。つなぐべき筆画か、筆の離れか、裏写りかは目視で別判断。'])
            figures.append(name);gaps.append(dict(frame=fid,component=c,file=name,context_xywh=[x,y,x2-x,y2-y]))
            if c['selected_09']:
                # Inspection sheets use fixed integer scaling; primary PNGs remain native-pixel panels.
                thumbs=[Image.fromarray(v[s][:,:,::-1]).resize(((x2-x)*3,(y2-y)*3),Image.Resampling.NEAREST) for v in [a[k] for k in keys]]
                selected.append((fid,c,thumbs))
    # All selected candidates also in compact sheets, without selecting only good examples.
    sheets=[];font=ImageFont.truetype(FONT_PATH,14)
    for start in range(0,len(selected),12):
        rows=selected[start:start+12];panelw=max(t.width for _,_,ts in rows for t in ts)
        heights=[max(t.height for t in ts)+32 for _,_,ts in rows]
        canvas=Image.new('RGB',(4*(panelw+12)+12,sum(heights)+40),'#faf8f2');d=ImageDraw.Draw(canvas)
        d.text((12,6),'全0.9採択候補 / 原画像 | 保存npj | 選別なし | DocRes選別あり',font=font,fill='black')
        y=35
        for (fid,c,ts),rowh in zip(rows,heights):
            d.text((12,y),f'{fid} {c["xywh"]} ink={c["docres_mean_ink_probability"]:.4f}',font=font,fill='black')
            for i,t in enumerate(ts):canvas.paste(t,(12+i*(panelw+12),y+22))
            y+=rowh
        name=f'inspection_selected_{start//12+1:02d}.png';canvas.save(OUT/name);figures.append(name);sheets.append(name)
    expected=6+40+len(gaps)+len(sheets)
    assert len(records)==20 and len(figures)==expected
    dump(OUT/'diagnostics.json',dict(regions=records,gaps=gaps,figures=figures,
        selected_09=len(selected),selected_05=sum(g['component']['selected_05'] for g in gaps),
        count_note='candidate counts not precision or document success rate',pixel_GT=False))
    if render_only:print(f'Rendered {expected} PNGs; all {len(gaps)} candidates, {len(selected)} selected.');return
    assert NOTES.exists(),'Inspect candidates and write notes before freezing'
    report='''# 保存npjの色を保ち、DocResで短い補完候補を選別

2026-10-02。「検証して」に従い、DocResの二値化を補助情報として使う小さな対照試作を実行した。実保存npj・朱印・紙面を土台にし、StainRestorerやDocResの色画像は復元へ使わない。6既知フレーム・5ページ・20固定ROI。前回の候補確率を確認した開発資料で、未知の別冊評価や人の筆画GT評価ではない。

## 結果と目視所見

'''+NOTES.read_text()+'''

## 条件をそろえた比較

前の凍結済み短い接続候補検出器をそのまま使用。原画像の局所Lab L支持8以上、保存npj側の端の局所コントラスト12以上、L損失8超、成分2〜64画素・最長辺12px以下・異なる既存端2つ以上への直接接触等で候補を作る。既存protect内に限定し、hold・朱・追加暖色保留・edit・外縁8pxを除く。旧2フレームの候補17／29画素は前回と完全一致する。

選別なし／成分内平均DocResインク確率0.5以上／0.9以上を比較。0.9を主表示し、0.5は感度対照。成分内平均であり全画素が高信頼という条件ではない。DocResによる新しい領域追加は行わない。元のprotect全体を縮小・置換する方式ではない。

補完器は全条件共通：実保存入力のgrayを17px形態学的closingした局所紙面grayで割り、1以下の同じスカラー倍率を保存npjのBGR全チャネルへ掛ける。最後だけ8bit丸め。以前の各BGR別倍率による補完と区別し、今回は選別の有無以外をそろえる。観測濃淡を短い切れ目へ使う試作で、学習済み筆画生成や文字の正解再現ではない。

前回の参照原画像と実保存入力が異なる2フレームには、保存入力を渡した凍結DocRes対照の確率を使う。全6フレームで入力・確率の由来を合わせた。全保存フレームで検出し、ROI座標は表示だけに使用。新しいモデル推論・学習・DDRM・npj再推論はない。凍結検出器は内部で以前の紙面補修も計算するが、その紙面出力は使用せず、紙面を変更するマスクは今回作っていない。

## 固定ROIの比較

'''
    for r in records:
        report+=f'- [{r["title"]}]({r["id"]}.png)／[保護候補の不一致](roles_{r["id"]}.png)：0.9選別後の更新 {r["mask_pixels"]["gate_09"]}画素。\n'
    report+='\n## 全候補・補助図\n\n'
    for fid in [s['id'] for s in manifest['frames']]:report+=f'- [{fid} 全体](overview_{fid}.png)\n'
    for name in sheets:report+=f'- [0.9採択候補の全件一覧]({name})\n'
    report+=f'\n全旧候補 {len(gaps)}件を採択／見送りに関係なくPNGで提示。0.5条件 {sum(g["component"]["selected_05"] for g in gaps)}件、0.9条件 {len(selected)}件。これらは精度・成功率ではない。\n\n'
    for g in gaps:
        c=g['component'];report+=f'- [{g["frame"]} {c["xywh"]}]({g["file"]})：{c["pixels"]}画素、平均確率 {c["docres_mean_ink_probability"]:.5f}、0.9採択={c["selected_09"]}\n'
    report+='''
## 検証と限界

保存入力とnpj、依存ソース、DocRes出力のハッシュ、候補の前回一致、選別候補の包含、全条件同じ補完値、更新外／朱／保留の保存npj一致、原寸比較パネルの全画素一致、ZIPのCRCと全バイト一致を確認した。紙色・印影の保持は凍結したマスクと更新外の条件に限り、全資料で意味的に正しい保護を保証しない。既存マスクに含まれない朱や筆画を自動的に保護できる証明でもない。

候補平均確率は校正済みの筆画精度ではない。候補数の削減だけでは筆画保持や汚れ除去の改善にならない。人の筆画／筆点／損傷注釈、未使用冊評価、汚れ側の自然な色復元は未完了。自動採用は保留。

[全比較PNG ZIP](comparison_pngs.zip)、[数値診断](diagnostics.json)、[検証](verification.json)、[固定試作条件](../npj_docres_gap_gate_pilot_20261002/manifest.json)。
'''
    (OUT/'report.md').write_text(report)
    deps={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),NOTES,
        PILOT/'manifest.json',PILOT/'complete.json',
        ROOT/'tmp_work/npj_docres_gap_gate_independent_audit_20261002/manifest.json',
        ROOT/'tmp_work/npj_docres_gap_gate_independent_audit_20261002/complete.json',
        Path(__file__).parent/'package_saved_npj_role_separated_review.py',
        Path(__file__).parent/'build_npj_v6_comparison.py']}
    dump(OUT/'manifest.json',dict(dependencies=deps,figures=figures,pixel_GT=False,adopted=False))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in figures:z.write(OUT/name,name)
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert z.testzip() is None and len(z.namelist())==expected
        for name in z.namelist():assert z.read(name)==(OUT/name).read_bytes()
    dump(OUT/'verification.json',dict(status='passed',whole_frame_detection=True,
        historical_input_and_npj_exact=True,all_candidates_shown=True,
        figures=expected,panels_pixel_exact=True,zip_CRC_bytes_exact=True,
        outside_updates_red_hold_extra_exact=True,pixel_GT=False,adopted=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print(f'Frozen {expected} PNGs, {len(gaps)} candidates, no automatic adoption.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true')
    main(parser.parse_args().render_only)
