"""Summarize the frozen development audit without claiming dense GT accuracy."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("limit_audit",HERE/"audit_classical_limits.py")
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def detail(c):
    xywh=c.get("detail")
    if not xywh:
        return None
    keys=["original","csc","sauvola_w61_k0.20","howe_fixed","howe_auto","howe_t0375_c10"]
    w,h=xywh[2:];scale=2;pw,ph=w*scale,h*scale;gap=20;head=95;label=42
    canvas=Image.new("RGB",(3*pw+4*gap,head+2*(ph+label+gap)+70),"#faf9f5")
    d=ImageDraw.Draw(canvas)
    def text(pos,s,size=18):d.text(pos,s,font=ImageFont.truetype(a.FONT,size),fill="#203442")
    text((15,10),c["id"]+" "+c["title"]+" / 詳細・整数2倍",26)
    text((15,50),c["page"]+f" / xywh={xywh} / 全ページ処理後の同位置",17)
    placements=[]
    for i,key in enumerate(keys):
        if key=="original":
            full=np.asarray(Image.open(a.ROOT/c["source"]).convert("RGB"))
        else:full=a.binary(a.load(a.OUT/"masks"/c["page"]/(key+".png")))
        tile=np.repeat(np.repeat(a.crop(full,xywh),scale,0),scale,1)
        x=gap+(i%3)*(pw+gap);y=head+(i//3)*(ph+label+gap)
        text((x,y),a.TITLES[key],18);canvas.paste(Image.fromarray(tile),(x,y+label))
        placements.append(dict(key=key,canvas_xywh=[x,y+label,pw,ph],expected=tile))
    text((15,canvas.height-53),"既知の開発例。点の保持と線の連続性は別。候補Aは既知2領域から選択。",16)
    text((15,canvas.height-29),"CSCはACE前処理あり。Howeは近似Canny。表示の平滑化・輪郭強調なし。",16)
    path=a.OUT/"figures"/(c["id"]+"_詳細2倍.png");canvas.save(path)
    saved=np.asarray(Image.open(path))
    for p in placements:
        assert np.array_equal(a.crop(saved,p["canvas_xywh"]),p.pop("expected"))
    return dict(path=str(path.relative_to(a.ROOT)),sha256=a.digest(path),panels=placements)


def envelope(c):
    """An optimistic per-pixel envelope, not a deployable parameter setting."""
    arrays={"原画像":np.asarray(Image.open(a.OUT/"originals"/(c["id"]+".png")).convert("RGB")),
            "Howe固定":a.binary(a.load(a.OUT/"crops"/c["id"]/"howe_fixed.png"))}
    sources={}
    for family in ("howe","sauvola"):
        paths=[p for p in sorted((a.OUT/"crops"/c["id"]).glob(family+"*.png")) if p.stem!="howe_auto"]
        assert len(paths)==(100 if family=="howe" else 9)
        union=np.logical_or.reduce([a.load(p) for p in paths])
        title=f"{family} 全{len(paths)}条件の和集合"
        arrays[title]=a.binary(union)
        a.save(a.OUT/"envelopes"/c["id"]/(family+"_union.png"),union)
        sources[family]={str(p.relative_to(a.ROOT)):a.digest(p) for p in paths}
    w,h=c["roi"][2:];gap=20;head=105;label=40
    canvas=Image.new("RGB",(2*w+3*gap,head+2*(h+label+gap)+90),"#faf9f5")
    d=ImageDraw.Draw(canvas)
    def text(pos,s,size=18):d.text(pos,s,font=ImageFont.truetype(a.FONT,size),fill="#203442")
    text((15,12),c["id"]+" 設定を変えたときの検出範囲",25)
    text((15,52),"和集合：いずれかの設定で黒なら黒。単一設定の出力ではない。",17)
    panels=[]
    for i,(title,tile) in enumerate(arrays.items()):
        x=gap+(i%2)*(w+gap);y=head+(i//2)*(h+label+gap)
        text((x,y),title,18);canvas.paste(Image.fromarray(tile),(x,y+label))
        panels.append(dict(key=title,canvas_xywh=[x,y+label,w,h]))
    text((15,canvas.height-70),"和集合の白は今回の全候補で未検出。そこが文字かどうかは原画像・注釈で確認。",15)
    text((15,canvas.height-45),"和集合の黒は汚れも含む。文字保持・精度の上限値を保証する図ではない。",15)
    text((15,canvas.height-23),"既知の開発例・原寸。Howeは近似Cannyのローカル移植。",15)
    dest=a.OUT/"figures"/(c["id"]+"_探索範囲の和集合.png");canvas.save(dest)
    saved=np.asarray(Image.open(dest))
    for p in panels:assert np.array_equal(a.crop(saved,p["canvas_xywh"]),arrays[p["key"]])
    return dict(path=str(dest.relative_to(a.ROOT)),sha256=a.digest(dest),panels=panels,sources=sources)


def spatial_changes(proto):
    """Locate full-page changes; distinguish scanner margins geometrically."""
    rows=[]
    for page in sorted({c["page"] for c in proto["cases"]}):
        fixed=a.load(a.OUT/"masks"/page/"howe_fixed.png")
        h,w=fixed.shape;mh,mw=int(h*.04),int(w*.04)
        inside=np.zeros_like(fixed);inside[mh:h-mh,mw:w-mw]=True
        for key in a.JOINT:
            mask=a.load(a.OUT/"masks"/page/(key+".png"))
            extra=mask&~fixed;lost=fixed&~mask
            n,labels,stats,_=cv2.connectedComponentsWithStats(extra.astype(np.uint8),connectivity=8)
            top=np.argsort(stats[1:,4])[-5:][::-1]+1
            rows.append(dict(page=page,key=key,added=int(extra.sum()),removed=int(lost.sum()),
                added_inner_rectangle=int((extra&inside).sum()),removed_inner_rectangle=int((lost&inside).sum()),
                added_border4pct=int((extra&~inside).sum()),removed_border4pct=int((lost&~inside).sum()),
                largest_added_components_xywh_area=[stats[i].tolist() for i in top],
                caveat="Geometric border split only; not an ink/noise label"))
    a.dump(a.OUT/"spatial_changes.json",rows)
    figures=[]
    for page in ("200010454_00002_2","200010454_00003_2"):
        c=next(c for c in proto["cases"] if c["page"]==page)
        raw=np.asarray(Image.open(a.ROOT/c["source"]).convert("RGB"))
        fixed=a.load(a.OUT/"masks"/page/"howe_fixed.png")
        cand=a.load(a.OUT/"masks"/page/"howe_t0375_c10.png")
        roi=[0,950,260,520] if page.endswith("00002_2") else [0,500,260,520]
        arrays=[raw,a.binary(fixed),a.binary(cand)]
        w,h=roi[2:];gap=20;head=96;lh=42
        canvas=Image.new("RGB",(3*w+4*gap,head+lh+h+85),"#faf9f5");d=ImageDraw.Draw(canvas)
        font=ImageFont.truetype(a.FONT,18)
        d.text((15,12),"候補Aの全ページ副作用：左端の影",font=ImageFont.truetype(a.FONT,24),fill="#203442")
        d.text((15,52),page+f" / xywh={roi}",font=font,fill="#203442")
        panels=[]
        for i,(title,full) in enumerate(zip(["原画像","Howe固定","候補A c113.1 / thi=.375"],arrays)):
            x=gap+i*(w+gap);y=head+lh;tile=a.crop(full,roi)
            d.text((x,head),title,font=font,fill="#203442");canvas.paste(Image.fromarray(tile),(x,y))
            panels.append(dict(title=title,canvas_xywh=[x,y,w,h],expected=tile))
        d.text((15,canvas.height-65),"筆画の確認点を保っていても、ページ端では黒像が増える。",font=font,fill="#203442")
        d.text((15,canvas.height-37),"出力差を見て選んだ診断範囲。文字欠落・精度の正解ではない。",font=font,fill="#203442")
        path=a.OUT/"figures"/(page+"_左端の副作用.png");canvas.save(path)
        saved=np.asarray(Image.open(path))
        for p in panels:assert np.array_equal(a.crop(saved,p["canvas_xywh"]),p.pop("expected"))
        figures.append(dict(path=str(path.relative_to(a.ROOT)),sha256=a.digest(path),panels=panels,roi=roi))
    a.dump(a.OUT/"spatial_change_verification.json",figures)


def main():
    proto=json.loads((a.OUT/"protocol.json").read_text())
    scores=json.loads((a.OUT/"scores.json").read_text())
    index={(r["case"],r["key"]):r for r in scores}
    # Mask file hashes allow a direct check against the historical masks even
    # though the originals were removed during cleanup.
    old=json.loads((a.DIAG/"evaluation/scores.json").read_text())["mask_sha256"]
    parity=[]
    for path,sha in old.items():
        oldpath=Path(path)
        if oldpath.stem not in ("sauvola","fixed","auto"):continue
        key={"sauvola":"sauvola_w61_k0.20","fixed":"howe_fixed","auto":"howe_auto"}[oldpath.stem]
        new=a.OUT/"masks"/oldpath.parent.name/(key+".png")
        actual=a.digest(new)
        assert actual==sha, (path,actual,sha)
        parity.append(dict(old=path,new=str(new.relative_to(a.ROOT)),sha256=sha))
    a.dump(a.OUT/"historical_mask_parity.json",parity)
    # Annotate the neighborhood definition explicitly. The generation score uses
    # a 5x5 square; old evaluation used a Euclidean radius-2 disk. Compute both.
    circle=[]
    for r in scores:
        c=next(c for c in proto["cases"] if c["id"]==r["case"])
        probe=next(p for p in c["probes"] if p["id"]==r["probe"])
        m=a.load(a.ROOT/next(v["path"] for v in json.loads((a.OUT/"results"/(c["page"]+".json")).read_text())["rows"] if v["case"]==c["id"] and v["key"]==r["key"]))
        ox,oy,_,_=c["roi"]
        n=0
        for x,y in probe["points"]:
            x-=ox;y-=oy
            yy,xx=np.ogrid[max(0,y-2):min(m.shape[0],y+3),max(0,x-2):min(m.shape[1],x+3)]
            disk=(xx-x)**2+(yy-y)**2 <= 4
            n+=int((m[max(0,y-2):y+3,max(0,x-2):x+3]&disk).any())
        circle.append(dict(case=c["id"],key=r["key"],points=r["ink_total"],
                           exact=r["ink_hit"],square_r2=r["ink_hit_r2"],euclidean_r2=n))
    a.dump(a.OUT/"point_neighborhood_audit.json",circle)
    # Same parameter key on both previously known pages, never independent best
    # for each of the two images.
    shared=[]
    controls=["D01","N01","T01","N05"]
    for key in sorted({r["key"] for r in scores if r["case"]=="D03" and r["key"].startswith("howe_t")}):
        contact=index[("D03",key)];spot=index[("D04",key)]
        losses={}
        gains={}
        for cid in controls:
            fixed=np.asarray(index[(cid,"howe_fixed")]["hit_vector"])
            candidate=np.asarray(index[(cid,key)]["hit_vector"])
            losses[cid]=int((fixed&~candidate).sum())
            gains[cid]=int((candidate&~fixed).sum())
        shared.append(dict(key=key,contact_points=contact["ink_hit"],contact_connected=contact.get("connected"),
                           spot_black=spot["nonink_black"],control_point_losses=losses,control_point_gains=gains,
                           known_two_pass=contact["ink_hit"]>=9 and contact.get("connected") and spot["nonink_black"]==0,
                           controls_no_fixed_point_lost=sum(losses.values())==0))
    a.dump(a.OUT/"joint_candidate_audit.json",shared)
    details=[v for c in proto["cases"] if (v:=detail(c))]
    a.dump(a.OUT/"detail_verification.json",details)
    envelopes=[envelope(c) for c in proto["cases"] if c["id"] in ("D01","D02","D03","N05")]
    a.dump(a.OUT/"envelope_verification.json",envelopes)
    spatial_changes(proto)
    summary=[]
    keys=["csc","sauvola_w61_k0.20","howe_fixed","howe_auto","howe_t0375_c10","howe_t025_c20"]
    for c in proto["cases"]:
        if not c["probes"]:continue
        summary.append(dict(case=c["id"],title=c["title"],methods={key:index[(c["id"],key)] for key in keys}))
    a.dump(a.OUT/"fixed_summary.json",summary)
    write_report(proto,summary,shared,parity)
    print("Historical masks match:",len(parity),"; shared candidates:",sum(r["known_two_pass"] for r in shared),flush=True)
    for r in shared:
        if r["known_two_pass"]:print(r,flush=True)


def write_report(proto,summary,shared,parity):
    keys=["csc","sauvola_w61_k0.20","howe_fixed","howe_auto","howe_t0375_c10","howe_t025_c20"]
    lines=["# 古典的手法の限界調査：再計算とパラメータ診断", "", "2026-10-21。既知の開発用9領域・8ページ・4資料群。教師あり学習・DDRMは今回実行していない。", "",
           "[比較一覧PNG](比較一覧.png) · [評価方針](評価方針.md) · [探索表CSV](scores.csv) · [設定](protocol.json)", "",
           "## 確認点と非文字領域の結果", "",
           "点数は探索的な筆画確認点の保持数。専門家による密な正解マスクの精度ではない。斑点の行は非文字矩形に残った黒画素数で、少ない方がよい。", "",
           "| 対象 | CSC v2 | Sauvola既定 | Howe固定 | Howe自動 | Howe候補A | Howe候補B |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in summary:
        row=[]
        for key in keys:
            r=s["methods"][key]
            row.append(f'{r["ink_hit"]}/{r["ink_total"]}' if r["ink_total"] else str(r["nonink_black"]))
        lines.append("| "+s["case"]+" "+s["title"]+" | "+" | ".join(row)+" |")
    lines += ["", "候補A：Canny上側閾値比.375・c約113.1。候補B：上側閾値比.25・c640。両者とも以前の接触部と別画像の斑点から選ばれた設定であり、今回の全例を使った最適値ではない。", "",
              "D01の正確な点位置の成績を、そのまま欠落率とは解釈しない。近傍2画素まで許すと結果が変わる。[円形近傍と5×5近傍の比較](point_neighborhood_audit.json)を併記した。線の連続性は別に原画像で確認する。", "",
              "## 既知2領域の両立候補を再確認", ""]
    for r in shared:
        if r["known_two_pass"]:
            lines.append(f'- `{r["key"]}`：接触部{r["contact_points"]}/10点、限定通路内で端点接続、孤立斑点の矩形で黒0画素。固定Howeで残っていた対照点の喪失は{sum(r["control_point_losses"].values())}点（内訳{r["control_point_losses"]}）。')
    lines += ["", "点での喪失がなくても、その他の筆画が無傷という証明にはならない。全ページの前景変化を[追加・削除画素数](whole_page_changes.json)に保存した。これは文字の損傷数ではない。", "",
              "実際に候補Aは200010454_00002_2で80,750画素、00003_2で126,852画素を追加した。最大の追加成分はそれぞれ80,418画素、126,534画素で、左端のスキャン影に位置する。[左端の比較](figures/200010454_00003_2_左端の副作用.png)で確認できる。全ページの差分が大きいことを、そのまま本文の誤分類増加とは解釈しない。[外周4%と内側の集計](spatial_changes.json)も併記した。候補Aを全ページの既定値として採用する判断は保留する。", "",
              "## 探索の範囲と再現確認", "",
              "- Howe：各ページ100固定条件（99探索＋既定）とAlg3自動選択。Sauvola：各ページ9条件。CSC：既定のみ。", 
              f'- 過去の3ページのSauvola既定・Howe固定・自動の計{len(parity)}マスクは、保存されていたSHA-256と一致した。[記録](historical_mask_parity.json)。',
              "- CSCは削除済み保存結果を再利用せず、現行v2コードで再計算。ACEは既存式と同じ処理をメモリ分割し、既定JPEG保存相当の読み戻しを含めた。小画像で元のACEと全画素一致を確認済み。", 
              "- 比較PNGの原寸パネルは元の切り出しと全画素一致。詳細画像は整数2倍。[原寸検証](figure_verification.json)・[詳細検証](detail_verification.json)。", 
              "- [探索範囲内の非劣解](frontiers.json)と[文字保持・非文字残存の図](文字保持と非文字残存.png)は開発プローブに対する診断。領域別最良値は正解を参照した選択なので、独立テストの数値として使わない。", "",
              "[薄い筆画の探索和集合](figures/D01_探索範囲の和集合.png)、[水染みの探索和集合](figures/D02_探索範囲の和集合.png)も作成。Howe100条件・Sauvola9条件のいずれかで黒なら黒とし、設定変更で検出し得る範囲を可視化した。これは単一設定の結果ではなく汚れも残すため、精度の上限や改善手法とは主張しない。", "",
              "## 解釈上の限界と次の作業", "",
              "この表だけで教師あり学習の必要性を確定できない。特にD02の水染み、L03の接触染み、L07の裏写りには信頼できる密な正解がなく、正式な文字保持と汚れ除去の両立を採点できていない。", "",
              "Howe原実装とのCanny一致も未確認であるため、ここでの失敗はローカル移植・探索範囲内の挙動として扱う。", "",
              "次は、原画像で確認できる細い筆画と非文字の境界を小範囲だけ注釈し、固定条件・個別最良条件を同じラベルで比較する。調整可能な失敗と調整しても残る失敗が整理できた段階で、確認済みの文字正解に損傷を付与する小規模な教師あり比較へ進む。", "",
              "原画像のみの[確認用画像](originals/)と[注釈状態](annotation_review.json)を用意した。不明部分は不明のまま保持し、白背景や推定文字で正解を埋めない。"]
    (a.OUT/"調査報告.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":main()
