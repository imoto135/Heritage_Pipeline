"""Source-traced partial labels on known development images, never expert GT.

prepare: render annotation overlays without reading any algorithm mask.
score: evaluate frozen annotations against existing native-page parameter crops.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"tmp_work/classical_partial_labels_20261021"
BASE=ROOT/"tmp_work/classical_limit_audit_20261021"
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
PRIMARY=["csc","sauvola_w61_k0.20","howe_fixed","howe_auto","howe_t0375_c10","howe_t025_c20"]
TITLES={"csc":"CSC v2", "sauvola_w61_k0.20":"Sauvola既定", "howe_fixed":"Howe固定", "howe_auto":"Howe自動",
        "howe_t0375_c10":"Howe候補A", "howe_t025_c20":"Howe候補B"}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def read(p):return np.asarray(Image.open(p).convert("RGB"))


def crop(a,box):
    x,y,w,h=box
    result=a[y:y+h,x:x+w]
    assert result.shape[:2]==(h,w)
    return result


def disk(r):
    y,x=np.ogrid[-r:r+1,-r:r+1]
    return x*x+y*y<=r*r


def ordered_line(vertices):
    # Consecutive native pixels with repeated consecutive vertices removed.
    points=[]
    for a,b in zip(vertices[:-1],vertices[1:]):
        a=np.asarray(a);b=np.asarray(b)
        n=int(np.abs(b-a).max())
        for v in np.rint(np.linspace(a,b,n+1)).astype(int):
            q=tuple(v)
            if not points or points[-1]!=q:points.append(q)
    return np.asarray(points,dtype=int)


def raster(region,annotation,include_moderate=False):
    w,h=region["roi"][2:]
    labels=np.zeros((h,w),np.uint8)
    paths=[]
    for path in annotation["ink_paths"]:
        if path["confidence"]!="high" and not include_moderate:continue
        points=ordered_line(path["xy"])
        assert ((points>=0).all() and (points[:,0]<w).all() and (points[:,1]<h).all())
        labels[points[:,1],points[:,0]]=1
        paths.append(dict(id=path["id"],xy=points))
    for poly in annotation["nonink_polygons"]:
        if poly["confidence"]!="high" and not include_moderate:continue
        m=np.zeros((h,w),np.uint8)
        cv2.fillPoly(m,[np.asarray(poly["xy"],np.int32)],1)
        # A one-pixel inset avoids judging manually placed polygon boundaries.
        m=ndi.binary_erosion(m,structure=disk(1))
        assert not np.any((labels==1)&m),(region["id"],poly["id"])
        labels[m]=2
    return labels,paths


def prepare():
    source_path=OUT/"annotations_source_only.json"
    annotations=json.loads(source_path.read_text())
    regions=json.loads((OUT/"source_regions.json").read_text())
    by_id={r["id"]:r for r in annotations["regions"]}
    counts=[];panels=[]
    for region in regions:
        assert sha(ROOT/region["source"])==region["source_sha256"]
        raw=crop(read(ROOT/region["source"]),region["roi"])
        assert np.array_equal(raw,read(OUT/(region["id"]+"_original.png")))
        high,paths=raster(region,by_id[region["id"]])
        full,_=raster(region,by_id[region["id"]],True)
        folder=OUT/"labels";folder.mkdir(exist_ok=True)
        Image.fromarray(high).save(folder/(region["id"]+"_high.png"))
        Image.fromarray(full).save(folder/(region["id"]+"_with_moderate.png"))
        overlay=raw.copy()
        overlay[high==1]=[0,105,245]
        overlay[high==2]=((raw[high==2].astype(float)*.45)+np.array([0,190,90])*.55).astype(np.uint8)
        overlay[(high==0)&(full==1)]=[255,135,0]
        overlay[(high==0)&(full==2)]=[245,180,40]
        scale=4;w,h=raw.shape[1],raw.shape[0];gap=20;head=110
        canvas=Image.new("RGB",(2*w*scale+3*gap,h*scale+head+85),"#faf9f5");d=ImageDraw.Draw(canvas)
        def text(pos,s,size=18):d.text(pos,s,font=ImageFont.truetype(FONT,size),fill="#203442")
        text((15,12),region["id"]+"：原画像と暫定の部分注釈",25)
        text((15,50),"青=筆画中心 / 緑=非文字内部 / 橙=確信が低く主評価から除外",17)
        for i,tile in enumerate([raw,overlay]):
            tile=np.repeat(np.repeat(tile,scale,0),scale,1)
            x=gap+i*(w*scale+gap);canvas.paste(Image.fromarray(tile),(x,head))
            panels.append(dict(id=region["id"],panel=i,shape=list(tile.shape)))
        text((15,canvas.height-65),"無着色は不明・評価対象外。Codexによる暫定注釈。専門家未確認。",16)
        text((15,canvas.height-37),"原画像だけで作図したが、既知の開発例であり盲検評価ではない。",16)
        canvas.save(OUT/(region["id"]+"_注釈確認.png"))
        counts.append(dict(id=region["id"],ink=int((high==1).sum()),nonink=int((high==2).sum()),
                           unknown=int((high==0).sum()),total=int(high.size),paths=len(paths)))
    protocol=dict(region_source_sha256=sha(OUT/"source_regions.json"),annotation_sha256=sha(source_path),
        annotation_status=annotations["status"],annotation_revision=annotations.get("revision_note"),freeze="Frozen before current scoring; historical method outputs and the first provisional scoring had already been viewed.",
        counts=counts,labels={"0":"unknown, excluded","1":"visible ink centerline","2":"selected nonink interior"},
        strict_metric="Ink-center foreground coverage and nonink foreground rate separately; no dense page F1 or precision.",
        tolerance="Euclidean radius 0, 1, 2; radii 3 and 4 as extra annotation-position sensitivity checks; no tolerance on nonink interiors",
        path_metric="Max run of missed traced native pixels and endpoint connectivity within radius-5 corridor; not character correctness",
        sensitivity="Repeat with moderate-confidence labels included; do not select labels using results.",
        population="Five crops from four previously known pages and three source book groups; no independent test.")
    dump(OUT/"protocol.json",protocol)
    dump(OUT/"label_hashes.json",{str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/"labels").glob("*.png"))})
    print("Frozen labels",counts,flush=True)


def max_run(values):
    best=run=0
    for v in values:
        run=run+1 if v else 0
        best=max(best,run)
    return best


def measure(pred,labels,paths):
    positive=labels==1;negative=labels==2
    metrics=dict(ink_n=int(positive.sum()),nonink_n=int(negative.sum()),ink_exact=int((pred&positive).sum()),
                 nonink_false=int((pred&negative).sum()))
    dilated={r:ndi.binary_dilation(pred,structure=disk(r)) if r else pred for r in range(5)}
    for r in (1,2,3,4):metrics[f"ink_r{r}"]=int((dilated[r]&positive).sum())
    line_metrics=[]
    for path in paths:
        points=path["xy"]
        line=np.zeros_like(pred);line[points[:,1],points[:,0]]=True
        corridor=ndi.binary_dilation(line,structure=disk(5))
        comp,_=ndi.label(pred&corridor,np.ones((3,3)))
        def endpoint(p):
            x,y=p;yy,xx=np.ogrid[:pred.shape[0],:pred.shape[1]]
            return set(np.unique(comp[(xx-x)**2+(yy-y)**2<=4]))-{0}
        row=dict(id=path["id"],n=len(points),connected=bool(endpoint(points[0])&endpoint(points[-1])))
        for r in (0,1,2,3,4):
            hit=dilated[r][points[:,1],points[:,0]]
            row[f"hit_r{r}"]=int(hit.sum());row[f"max_missed_run_r{r}"]=int(max_run(~hit))
        line_metrics.append(row)
    metrics["paths"]=line_metrics
    return metrics


def nondominated(rows):
    # Radius-2 positive coverage versus unrelaxed nonink residuals.
    return [r for r in rows if not any(s["ink_r2"]>=r["ink_r2"] and s["nonink_false"]<=r["nonink_false"] and
             (s["ink_r2"]>r["ink_r2"] or s["nonink_false"]<r["nonink_false"]) for s in rows)]


def score():
    protocol=json.loads((OUT/"protocol.json").read_text())
    assert protocol["annotation_sha256"]==sha(OUT/"annotations_source_only.json")
    assert protocol["region_source_sha256"]==sha(OUT/"source_regions.json")
    for rel,digest in json.loads((OUT/"label_hashes.json").read_text()).items():assert sha(OUT/rel)==digest
    regions=json.loads((OUT/"source_regions.json").read_text())
    annotations={r["id"]:r for r in json.loads((OUT/"annotations_source_only.json").read_text())["regions"]}
    base_cases={c["id"]:c for c in json.loads((BASE/"protocol.json").read_text())["cases"]}
    all_rows=[];frontiers=[];verification=[];envelopes=[]
    for region in regions:
        base=base_cases[region["case"]];x,y,w,h=region["roi"];bx,by,_,_=base["roi"]
        for mode in ("high","with_moderate"):
            labels,paths=raster(region,annotations[region["id"]],mode=="with_moderate")
            saved=np.asarray(Image.open(OUT/"labels"/(region["id"]+"_"+mode+".png")))
            assert np.array_equal(labels,saved)
            rows=[];family_masks={"howe":[],"sauvola":[],"csc":[]}
            for p in sorted((BASE/"crops"/region["case"]).glob("*.png")):
                pred=crop(np.asarray(Image.open(p).convert("L"))==0,[x-bx,y-by,w,h])
                rows.append(dict(region=region["id"],mode=mode,key=p.stem,prediction_sha256=sha(p),**measure(pred,labels,paths)))
                if p.stem!="howe_auto":
                    family=next(f for f in family_masks if p.stem.startswith(f))
                    family_masks[family].append(pred)
            for family,masks in family_masks.items():
                union=np.logical_or.reduce(masks)
                envelopes.append(dict(region=region["id"],mode=mode,family=family,candidates=len(masks),**measure(union,labels,paths)))
            combined=np.logical_or.reduce([m for masks in family_masks.values() for m in masks])
            envelopes.append(dict(region=region["id"],mode=mode,family="combined",candidates=sum(len(m) for m in family_masks.values()),**measure(combined,labels,paths)))
            if mode=="high":
                Image.fromarray(np.where(combined,0,255).astype(np.uint8)).save(OUT/(region["id"]+"_all_candidates_union.png"))
            all_rows.extend(rows)
            for family in ("howe","sauvola","csc"):
                rr=[r for r in rows if r["key"].startswith(family) and r["key"]!="howe_auto"]
                frontiers.append(dict(region=region["id"],mode=mode,family=family,candidates=len(rr),
                    pareto=nondominated(rr),oracle=min(rr,key=lambda r:(-r["ink_r2"],r["nonink_false"],-r["ink_exact"]))))
        figures(region,labels=None)
    dump(OUT/"scores.json",all_rows);dump(OUT/"frontiers.json",frontiers);dump(OUT/"envelope_scores.json",envelopes)
    with (OUT/"scores.csv").open("w") as f:
        fields=["region","mode","key","ink_n","ink_exact","ink_r1","ink_r2","ink_r3","ink_r4","nonink_n","nonink_false"]
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");wr.writeheader();wr.writerows(all_rows)
    report(regions,all_rows,frontiers)
    failure_atlas(regions,all_rows,frontiers,envelopes)
    plot(regions,all_rows)
    dump(OUT/"scoring_verification.json",dict(annotation_hash_unchanged=True,unknown_pixels_excluded=True,
         modes=["high","with_moderate"],candidate_rows=len(all_rows),dense_gt_claim=False,
         code_sha256=sha(Path(__file__))))
    print("Scored",len(all_rows),"candidate/annotation-mode rows",flush=True)
    for r in all_rows:
        if r["mode"]=="high" and r["key"] in PRIMARY:
            print(r["region"],r["key"],f'{r["ink_r2"]}/{r["ink_n"]}',f'nonink {r["nonink_false"]}/{r["nonink_n"]}',
                  [(p["id"],p["max_missed_run_r2"],p["connected"]) for p in r["paths"]],flush=True)


def failure_atlas(regions,rows,frontiers,envelopes):
    verification=[]
    for region in regions:
        if region["id"] not in ("thin","water"):continue
        rid=region["id"];raw=read(OUT/(rid+"_original.png"))
        labels=np.asarray(Image.open(OUT/"labels"/(rid+"_high.png")))
        combined=np.asarray(Image.open(OUT/(rid+"_all_candidates_union.png")))==0
        pred_near=ndi.binary_dilation(combined,structure=disk(2))
        overlay=raw.copy();overlay[(labels==1)&pred_near]=[0,105,245];overlay[(labels==1)&~pred_near]=[240,35,65]
        arr=[raw];titles=["原画像"]
        base=next(c for c in json.loads((BASE/"protocol.json").read_text())["cases"] if c["id"]==region["case"])
        x,y,w,h=region["roi"];bx,by,_,_=base["roi"]
        for family in ("howe","sauvola"):
            f=next(f for f in frontiers if f["region"]==rid and f["mode"]=="high" and f["family"]==family)
            key=f["oracle"]["key"]
            m=crop(np.asarray(Image.open(BASE/"crops"/region["case"]/(key+".png")).convert("L")),[x-bx,y-by,w,h])
            arr.append(np.repeat(m[...,None],3,2));titles.append(family+" 領域別最良")
        arr.append(overlay);titles.append("青=検出範囲内 / 赤=全候補で未検出")
        scale=3;pw,ph=w*scale,h*scale;gap=20;head=100;lh=44
        canvas=Image.new("RGB",(2*pw+3*gap,head+2*(ph+lh+gap)+90),"#faf9f5");d=ImageDraw.Draw(canvas)
        def text(pos,s,size=17):d.text(pos,s,font=ImageFont.truetype(FONT,size),fill="#203442")
        text((15,10),rid+"：調整後にも残る筆画の確認",25)
        text((15,49),"Howe100条件 + Sauvola9条件 + CSC既定の検出範囲と照合。",17)
        panels=[]
        for i,(tile,title) in enumerate(zip(arr,titles)):
            tile=np.repeat(np.repeat(tile,scale,0),scale,1)
            px=gap+(i%2)*(pw+gap);py=head+(i//2)*(ph+lh+gap)
            text((px,py),title,16);canvas.paste(Image.fromarray(tile),(px,py+lh))
            panels.append(dict(title=title,xywh=[px,py+lh,pw,ph],expected=tile))
        text((15,canvas.height-70),"赤：暫定筆画中心から半径2画素に、全110候補のどれにも黒がない。",15)
        text((15,canvas.height-45),"既知の開発例・専門家未確認。未注釈領域の文字や汚れは判定しない。",15)
        text((15,canvas.height-22),"領域別最良は診断用。全候補の和集合は単一設定の出力ではない。",15)
        path=OUT/(rid+"_調整後の残存筆画.png");canvas.save(path);saved=read(path)
        for p in panels:assert np.array_equal(crop(saved,p["xywh"]),p.pop("expected"))
        verification.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),panels=panels))
    dump(OUT/"failure_atlas_verification.json",verification)


def figures(region,labels=None):
    cid=region["case"];base=next(c for c in json.loads((BASE/"protocol.json").read_text())["cases"] if c["id"]==cid)
    x,y,w,h=region["roi"];bx,by,_,_=base["roi"]
    raw=read(OUT/(region["id"]+"_original.png"))
    # Larger integer scales for the small annotation ROIs.
    scale=3;pw,ph=w*scale,h*scale;gap=20;head=100;lh=42
    canvas=Image.new("RGB",(3*pw+4*gap,head+2*(ph+lh+gap)+75),"#faf9f5");d=ImageDraw.Draw(canvas)
    def text(pos,s,size=18):d.text(pos,s,font=ImageFont.truetype(FONT,size),fill="#203442")
    text((15,10),region["id"]+" / 原画像と固定条件・調整候補",25)
    text((15,49),region["page"]+f" / xywh={region['roi']} / 全ページ処理後、整数3倍",17)
    keys=["original","csc","sauvola_w61_k0.20","howe_fixed","howe_t0375_c10","howe_t025_c20"]
    panels=[]
    for i,key in enumerate(keys):
        if key=="original":tile=raw
        else:
            m=crop(np.asarray(Image.open(BASE/"crops"/cid/(key+".png")).convert("L")),[x-bx,y-by,w,h])
            tile=np.repeat(m[...,None],3,2)
        tile=np.repeat(np.repeat(tile,scale,0),scale,1)
        px=gap+(i%3)*(pw+gap);py=head+(i//3)*(ph+lh+gap)
        text((px,py),TITLES.get(key,"原画像"),19);canvas.paste(Image.fromarray(tile),(px,py+lh))
        panels.append(dict(key=key,xywh=[px,py+lh,pw,ph],expected=tile))
    text((15,canvas.height-57),"点線は筆画の切断を含む。暫定の部分注釈で評価。専門家の密な正解ではない。",16)
    text((15,canvas.height-30),"不明は評価から除外。HoweはCanny近似。CSCはACE前処理を含む。",16)
    path=OUT/(region["id"]+"_比較.png");canvas.save(path);saved=read(path)
    for p in panels:assert np.array_equal(crop(saved,p["xywh"]),p.pop("expected"))
    dump(OUT/(region["id"]+"_panel_verification.json"),dict(sha256=sha(path),panels=panels))


def report(regions,rows,frontiers):
    index={(r["region"],r["mode"],r["key"]):r for r in rows}
    lines=["# 原画像からの暫定部分注釈による限界診断","", "2026-10-21。既知の開発用5領域。Codexによる暫定注釈で、専門家の確認・密な画素正解・独立テストではない。", "",
        "[注釈の定義](annotations_source_only.json) · [座標](source_regions.json) · [全スコアCSV](scores.csv) · [非劣解と領域別最良](frontiers.json)", "",
        "青で指定した観測筆画の中心線と、緑で指定した非文字の内部のみを採点。その他は不明として除外した。確信が低い注釈は主集計から外し、感度分析で追加した。元の手法出力は以前に見たことがあるため、盲検評価ではない。", "",
        "初回採点後、waterの右側に置いた線を原画像だけで再点検したところ、連続した観測筆画と確認できなかったため全手法から同じように除外した。出力上の別の線へは移動していない。[初回注釈と採点](history_before_source_reaudit/)を保存している。現在のwaterは左側の確認可能な曲線だけを主集計する。", "",
        "## 主集計：筆画中心から半径2画素内の黒画素の有無", "",
        "| 領域 | CSC v2 | Sauvola既定 | Howe固定 | Howe自動 | 候補A | 候補B |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for region in regions:
        s=[]
        for key in PRIMARY:
            r=index[(region["id"],"high",key)];s.append(f'{100*r["ink_r2"]/r["ink_n"]:.1f}%')
        lines.append("| "+region["id"]+" | "+" | ".join(s)+" |")
    lines += ["", "表は部分注釈の中心線近傍保持率であり、文字全体の再現率・適合率・F値ではない。非文字への誤検出と線の接続を別に見る。[保持と非文字残存の図](保持と非文字残存.png)。", "",
              "## 診断用の領域別最良条件", "",
              "| 領域 | 手法 | 半径2画素の保持 | 非文字残存 | 選択設定 |", "| --- | --- | ---: | ---: | --- |"]
    for f in frontiers:
        if f["mode"]!="high":continue
        r=f["oracle"]
        lines.append(f'| {f["region"]} | {f["family"]} | {r["ink_r2"]}/{r["ink_n"]} | {r["nonink_false"]}/{r["nonink_n"]} | {r["key"]} |')
    lines += ["", "領域別最良は保持を最大化し、同率なら非文字残存を最小化した診断用の選択。各領域で別の設定を選んでおり、運用可能な自動手法の結果ではない。CSCは既定のみで、探索予算も同じではない。", "",
              "線の接続・最大の未検出連続区間は[scores.json](scores.json)のpathsに保存した。近傍に黒があっても、汚れに沿う誤接続や字形の誤りを否定しない。", "",
              "## 原画像と注釈の確認", ""]
    for region in regions:
        lines.append(f'- {region["id"]}：[注釈確認]({region["id"]}_注釈確認.png)・[手法比較]({region["id"]}_比較.png)。')
    lines += ["", "## 判断の限界", "",
              "この診断は、以前の点だけの確認より長い筆画を採点するが、原画像の専門的な正解を確定したものではない。位置ずれ対策として半径0/1/2を併記し、不確かな線や染みの追加による感度分析も保存した。", "",
              "裏写り領域では、由来が未確認の青灰色像を非文字として塗っていない。したがって、bleedの数値は表文字保持と紙地の確認であって裏写り除去率ではない。", "",
              "学習を導入する根拠は、実用上残る筆画欠落や非文字残存を特定することにある。古典法全般の不可能性や、RA-GANが文字形状を学べば必ず改善することは、この実験からは主張しない。"]
    (OUT/"部分注釈の調査報告.md").write_text("\n".join(lines)+"\n")


def plot(regions,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8));colors={"howe":"#176aab","sauvola":"#d37710","csc":"#8d428c"}
    for ax,region in zip(axes.ravel(),regions):
        for family,color in colors.items():
            rr=[r for r in rows if r["region"]==region["id"] and r["mode"]=="high" and r["key"].startswith(family) and r["key"]!="howe_auto"]
            ax.scatter([100*r["nonink_false"]/r["nonink_n"] for r in rr],
                       [100*r["ink_r2"]/r["ink_n"] for r in rr],s=22,alpha=.6,label=family,color=color)
        ax.set(title=region["id"],xlabel="Selected non-ink foreground (%)",ylabel="Ink centerline coverage, radius 2 (%)",ylim=(-3,103))
        ax.grid(alpha=.2);ax.legend()
    axes.ravel()[-1].axis("off")
    fig.suptitle("Provisional partial labels on known development images; not dense expert ground truth")
    fig.tight_layout();fig.savefig(OUT/"保持と非文字残存.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["prepare","score"]);args=ap.parse_args()
    if args.action=="prepare":prepare()
    else:score()
