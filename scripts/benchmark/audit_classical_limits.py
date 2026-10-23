"""Recompute classical black-text baselines and exploratory parameter frontiers.

Run with the diffusion environment. All algorithms see native full pages; crops
are made afterwards. Reused sparse annotations are development probes, never
dense ground truth. No supervised training or DDRM is run here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tmp_work/classical_limit_audit_20261021"
OLD = ROOT / "tmp_work/classical_black_comparison_20261008"
DIAG = ROOT / "tmp_work/howe_diagnosis_20261008"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
JOINT = {"howe_t0375_c10": (.375, 10), "howe_t025_c20": (.25, 20),
         "howe_t025_c21": (.25, 21), "howe_t025_c22": (.25, 22)}
TITLES = {"original": "原画像", "csc": "ACE + CSC v2（現行コード）",
          "sauvola_w61_k0.20": "Sauvola  w61 / k=.20",
          "howe_fixed": "Howe固定  c160 / thi=.4",
          "howe_auto": "Howe Alg3（近似Canny）",
          "howe_t0375_c10": "Howe候補A  c113.1 / thi=.375",
          "howe_t025_c20": "Howe候補B  c640 / thi=.25"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L"))
    assert np.isin(a, [0, 255]).all(), path
    return a == 0


def save(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)


def binary(mask):
    return np.repeat(np.where(mask, 0, 255).astype(np.uint8)[..., None], 3, 2)


def crop(a, xywh):
    x, y, w, h = xywh
    result = a[y:y+h, x:x+w]
    assert result.shape[:2] == (h, w)
    return result


def ace_chunked(image, alpha=5., n_samples=200, seed=0, chunk=16000):
    """Same float32 operations and common stretch as v2 ACE, bounded memory."""
    h, w = image.shape[:2]
    n = h*w
    flat = image.reshape(-1, 3).astype(np.float32)/255.
    sample = np.random.default_rng(seed).integers(0, n, size=n_samples)
    sy = (sample//w).astype(np.float32)
    sx = (sample % w).astype(np.float32)
    result = np.zeros((n, 3), np.float32)
    for start in range(0, n, chunk):
        idx = np.arange(start, min(start+chunk, n))
        dy = (idx//w).astype(np.float32)[:, None]-sy[None, :]
        dx = (idx % w).astype(np.float32)[:, None]-sx[None, :]
        inv = 1./np.maximum(np.sqrt(dy**2+dx**2), 1.)
        den = inv.sum(axis=1)
        for ch in range(3):
            diff = flat[idx, ch, None]-flat[sample, ch][None, :]
            result[idx, ch] = (np.clip(alpha*diff, -1., 1.)*inv).sum(axis=1)/den
    lo, hi = result.min(), result.max()
    result = (result-lo)/(hi-lo) if hi-lo > 1e-6 else flat.copy()
    return (np.clip(result, 0., 1.)*255).astype(np.uint8).reshape(h, w, 3)


def component_filter(image, candidate, support_min):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    seeds = ((hsv[..., 0] <= 12) | (hsv[..., 0] >= 168)) & (hsv[..., 1] > 180)
    n, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    total = np.bincount(labels.ravel(), minlength=n)
    supported = np.bincount(labels[seeds], minlength=n)
    keep = supported/np.maximum(total, 1) >= support_min
    keep[0] = False
    return keep[labels]


def prepare():
    assert not (OUT/"protocol.json").exists(), "Existing protocol must be reused."
    selection = json.loads((ROOT/"tmp_work/classical_limits_by_category_20261008/selection.json").read_text())
    cases = []
    for c in selection["cases"][:4]:
        cases.append(dict(id="D"+c["id"][:2], title=c["category"], page=c["page"],
                          source=c["original"], roi=c["roi_xywh"], detail=c["detail_xywh"],
                          probes=[], role="known development example"))
    old_proto = json.loads((OLD/"protocol.json").read_text())
    ann = {a["id"]: a for a in json.loads((OLD/"annotations_frozen.json").read_text())["cases"]}
    valid = json.loads((OLD/"annotation_audit.json").read_text())["valid_original_point_indices"]
    rois = {"N01": [390, 300, 440, 420], "T01": [760, 360, 600, 600],
            "N05": [1019, 1617, 640, 580]}
    for c in old_proto["cases"]:
        cid = c["id"]
        ox, oy, _, _ = c["roi_xywh"]
        points = [[ox+ann[cid]["ink_points_xy"][i][0], oy+ann[cid]["ink_points_xy"][i][1]] for i in valid[cid]]
        negatives = [[ox+x, oy+y, w, h] for x, y, w, h in ann[cid]["nonink_regions_xywh"]]
        cases.append(dict(id=cid, title={"N01": "薄い曲線", "T01": "広い染みの対照", "N05": "細い文字・通常筆画"}[cid],
                          page=Path(c["original"]).stem, source=c["original"], roi=rois[cid],
                          detail=c["roi_xywh"], probes=[dict(id=cid+"_sparse", points=points, negatives=negatives)],
                          role="known development example; audited pre-existing sparse labels"))
    old_probes = json.loads((DIAG/"evaluation/frozen_probes.json").read_text())["probes"]
    for c, probe in zip([cases[0], cases[2], cases[3]], old_probes):
        c["probes"] = [dict(id=probe["id"], points=probe.get("points_xy", []),
                            negatives=[probe["box_xywh"]] if "box_xywh" in probe else [],
                            corridor_radius=probe.get("corridor_radius"))]
    later = json.loads((ROOT/"tmp_work/test_only_challenge_search_20261009/test_candidates.json").read_text())
    for c in later["cases"]:
        if c["id"] in ("03", "07"):
            cases.append(dict(id="L"+c["id"], title=c["label"], page=c["name"], source=c["source"],
                              roi=[750,700,850,850] if c["id"] == "03" else [800,750,850,900],
                              detail=None, probes=[], role="known development; unknown labels; visual only"))
    for c in cases:
        p = ROOT/c["source"]
        c["source_sha256"] = digest(p)
        with Image.open(p) as im:
            c["size"] = list(im.size)
        c["book"] = p.parent.name
        for x, y in [p for probe in c["probes"] for p in probe["points"]]:
            assert 0 <= x < c["size"][0] and 0 <= y < c["size"][1]
    proto = dict(date="2026-10-21", cases=cases,
                 interpretation="Exploratory known development probes. Not dense GT, test accuracy, or proof of a whole method family's limit.",
                 labels="Reused without editing positions; 52 audited centers + 16 old centers; 9 nonink patches + 1 spot patch. Ambiguous areas unknown.",
                 input="Native full raw page. CSC alone includes ACE and JPEG95 round trip, following existing pipeline. All crops after page processing.",
                 howe=dict(c_values=np.exp(np.linspace(np.log(20), np.log(5120), 33)).tolist(), high=[.25,.375,.5], fixed=[160,.4],
                           canny="skimage approximation; MATLAB canny_old parity unresolved", auto="author Alg3 selection port"),
                 sauvola=dict(windows=[31,61,101], k=[.1,.2,.3], R=.5),
                 csc=dict(variant="current local v2, default arguments", ace=dict(alpha=5.,samples=200,seed=0,jpeg_quality=95), parameter_search=False),
                 prior_joint_candidates=JOINT,
                 freeze="Protocol written before outputs in this directory; all source cases previously inspected, never a new holdout.",
                 limits=["Sparse ink points do not establish pixel precision or character identity.",
                         "Negative patches include paper; counts are patch residuals, not stain-mask recall.",
                         "A corridor connection can follow dirt; inspect aligned image.",
                         "Per-case best is an oracle diagnostic using known probes; never a deployable setting."])
    dump(OUT/"protocol.json", proto)
    deps = [Path(__file__), DIAG/"experiment.py", DIAG/"sweep_cli", DIAG/"contact_stroke_probe.json",
            DIAG/"evaluation/frozen_probes.json", OLD/"annotations_frozen.json", OLD/"annotation_audit.json",
            ROOT/"tmp_work/classical_deep_review_20261008/run.py", ROOT/"tmp_work/classical_deep_review_20261008/graphcut_cli",
            OLD/"compare.py", ROOT/"modules/ddrm_npj_v2/scripts/csc.py", ROOT/"modules/ddrm_npj_v2/scripts/ace.py"]
    dump(OUT/"dependencies.json", {str(p.relative_to(ROOT)): digest(p) for p in deps})
    for p in deps:
        snapshot=OUT/"source_snapshot"/p.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,snapshot)
    # Numerical checks validate the memory-saving implementation before real data.
    ace = module(ROOT/"modules/ddrm_npj_v2/scripts/ace.py", "audit_ace_reference")
    csc = module(ROOT/"modules/ddrm_npj_v2/scripts/csc.py", "audit_csc_reference")
    rng = np.random.default_rng(20261021)
    for shape in [(23,31,3), (41,37,3)]:
        a = rng.integers(0,256,shape,dtype=np.uint8)
        assert np.array_equal(ace_chunked(a, chunk=113), ace.automatic_color_equalization(a))
    for _ in range(10):
        a = rng.integers(0,256,(43,57,3),dtype=np.uint8)
        candidate = rng.random((43,57)) > .7
        assert np.array_equal(component_filter(a,candidate,.05), csc._filter_red_candidate_by_component(a,candidate,.05))
    dump(OUT/"implementation_checks.json", dict(ace_chunked_exact_small_reference=True, component_support_exact_10_samples=True))
    originals_only(proto)
    print(f"Frozen: {len(cases)} ROIs, {len(set(c['page'] for c in cases))} pages; sparse labels unchanged", flush=True)


def probe_measure(mask, c):
    rows = []
    for p in c["probes"]:
        pts = np.asarray(p["points"], dtype=int).reshape(-1,2)
        hit = [bool(mask[y,x]) for x,y in pts]
        near = [bool(mask[max(0,y-2):y+3,max(0,x-2):x+3].any()) for x,y in pts]
        areas = [crop(mask,b) for b in p["negatives"]]
        row = dict(probe=p["id"], ink_total=len(pts), ink_hit=int(sum(hit)), ink_hit_r2=int(sum(near)),
                   hit_vector=hit, nonink_black=sum(int(a.sum()) for a in areas), nonink_area=sum(a.size for a in areas))
        if p.get("corridor_radius"):
            path = np.zeros_like(mask, dtype=np.uint8)
            for a,b in zip(pts[:-1],pts[1:]):
                cv2.line(path, tuple(a), tuple(b), 1, 1)
            corridor = ndi.binary_dilation(path, iterations=p["corridor_radius"])
            labels,_ = ndi.label(mask & corridor, np.ones((3,3)))
            def ends(q):
                x,y=q
                return set(np.unique(labels[max(0,y-2):y+3,max(0,x-2):x+3]))-{0}
            row.update(path_coverage=float(mask[path>0].mean()), connected=bool(ends(pts[0]) & ends(pts[-1])))
        rows.append(row)
    return rows


def worker(page):
    cv2.setNumThreads(1)
    proto = json.loads((OUT/"protocol.json").read_text())
    cases = [c for c in proto["cases"] if c["page"] == page]
    assert cases, page
    for name, sha in json.loads((OUT/"dependencies.json").read_text()).items():
        assert digest(ROOT/name) == sha, name
    status_path = OUT/"status"/(page+".json")
    assert not (OUT/"results"/(page+".json")).exists(), "Already completed"
    t0 = time.monotonic()
    status = dict(page=page, stage="start", branches=0)
    dump(status_path,status)
    e = module(DIAG/"experiment.py", "audit_howe")
    c = cases[0]
    assert digest(ROOT/c["source"]) == c["source_sha256"]
    raw = np.asarray(Image.open(ROOT/c["source"]).convert("RGB"))
    gray = e.deep.base.gray(raw)
    scratch = OUT/"scratch"/page
    folder = OUT/"masks"/page
    rows = []
    hashes = {}

    def record(key, m, meta=None, full=False):
        assert m.shape == gray.shape and m.dtype == bool
        if full:
            p = folder/(key+".png")
            save(p,m)
            assert np.array_equal(load(p),m)
            hashes[str(p.relative_to(ROOT))] = digest(p)
        for cc in cases:
            p = OUT/"crops"/cc["id"]/(key+".png")
            save(p,crop(m,cc["roi"]))
            rows.append(dict(case=cc["id"], page=page, key=key, params=meta or {},
                             probes=probe_measure(m,cc), path=str(p.relative_to(ROOT)), sha256=digest(p)))

    for w in proto["sauvola"]["windows"]:
        mean,std,_ = e.deep.base.stats(gray,w)
        for k in proto["sauvola"]["k"]:
            key=f"sauvola_w{w}_k{k:.2f}"
            record(key,gray < mean*(1+k*(std/.5-1)),dict(window=w,k=k),full=(w==61 and k==.2))
    status["stage"]="howe"; dump(status_path,status)
    terms,_ = e.terms(gray)
    # Exhaust even a single-element sweep so its temporary binary files close
    # and are removed by the generator's normal completion path.
    [(pre,meta)] = e.sweep(terms,[160],scratch,"fixed")
    fixed,_ = e.post(pre)
    record("howe_fixed",fixed,dict(c=160,thi=.4),full=True)
    old_scores = json.loads((DIAG/"evaluation/scores.json").read_text())["rows"]
    parity=[]
    for cc in cases:
        for r in probe_measure(fixed,cc):
            old = [v for v in old_scores if v["page"]==page and v["method"]=="fixed" and v["probe"]==r["probe"]]
            if old:
                expected = old[0]
                assert r["ink_hit"]==expected.get("exact_points_retained",0)
                assert r["nonink_black"]==expected.get("black_pixels",0)
                parity.append(r["probe"])
    del terms
    branches=[]
    for thi in proto["howe"]["high"]:
        terms,_ = e.terms(gray,thi=thi)
        masks=[]; bsd=[]
        for idx,(pre,meta) in enumerate(e.sweep(terms,e.CS,scratch,f"thi{thi}")):
            m,_ = e.post(pre)
            if masks:
                bsd.append(float(np.mean(masks[-1] != m)))
            masks.append(m)
            key=f"howe_t{str(thi).replace('.','')}_c{idx:02d}"
            record(key,m,dict(c=float(e.CS[idx]),thi=thi,index=idx),full=key in JOINT)
        chosen,stability = e.stability_select(bsd)
        branches.append(dict(thi=thi,c=float(e.CS[chosen]),index=chosen,mask=masks[chosen],stability=stability))
        del masks,terms
        status["branches"]+=1; dump(status_path,status)
        print(page,"Howe branch",thi,"complete",flush=True)
    lo,mid,hi=[b["mask"] for b in branches]
    dlo=float(np.mean(lo!=mid));dhi=float(np.mean(hi!=mid))
    selected=branches[0 if dlo < dhi else 2]
    record("howe_auto",selected["mask"],dict(c=selected["c"],thi=selected["thi"]),full=True)
    branch_summary=[{k:v for k,v in b.items() if k!="mask"} for b in branches]
    del branches,lo,mid,hi,selected,raw,gray,fixed
    status["stage"]="ace_csc"; dump(status_path,status)
    print(page,"ACE + CSC start",flush=True)
    bgr=cv2.imread(str(ROOT/c["source"]))
    ace=ace_chunked(bgr)
    # Existing pipeline saves ACE with cv2.imwrite('.jpg') before CSC.
    ok,jpeg=cv2.imencode(".jpg",ace,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
    ace=cv2.imdecode(jpeg,cv2.IMREAD_COLOR)
    csc=module(ROOT/"modules/ddrm_npj_v2/scripts/csc.py","audit_csc")
    csc._filter_red_candidate_by_component=component_filter
    with threadpool_limits(limits=1):
        result=csc.color_space_clustering(ace)
    # Restore gray only for record's shape assertion.
    gray=np.zeros(bgr.shape[:2],dtype=np.uint8)
    record("csc",result.black_text_mask>0,dict(variant="current v2 default",ace_jpeg95=True),full=True)
    dump(OUT/"results"/(page+".json"),dict(page=page,rows=rows,auto_branches=branch_summary,
         auto_mismatch=dict(low_middle=dlo,high_middle=dhi),old_fixed_probe_parity=parity,
         full_mask_sha256=hashes,seconds=time.monotonic()-t0))
    status.update(stage="complete",seconds=time.monotonic()-t0);dump(status_path,status)
    print(page,"DONE",round(status["seconds"],1),flush=True)


def originals_only(proto):
    for c in proto["cases"]:
        raw=Image.open(ROOT/c["source"]).convert("RGB")
        x,y,w,h=c["roi"]
        tile=raw.crop((x,y,x+w,y+h))
        dest=OUT/"originals"/(c["id"]+".png");dest.parent.mkdir(parents=True,exist_ok=True)
        tile.save(dest)
        annotated=tile.resize((w*2,h*2),Image.Resampling.NEAREST)
        d=ImageDraw.Draw(annotated)
        for p in c["probes"]:
            for px,py in p["points"]:
                xx,yy=(px-x)*2,(py-y)*2
                d.ellipse((xx-5,yy-5,xx+5,yy+5),outline="#0076ff",width=2)
            for px,py,pw,ph in p["negatives"]:
                xx,yy=(px-x)*2,(py-y)*2
                d.rectangle((xx,yy,xx+pw*2,yy+ph*2),outline="#00a060",width=3)
        annotated.save(OUT/"originals"/(c["id"]+"_probes.png"))
    dump(OUT/"annotation_review.json",dict(
        label_status="Reused exploratory sparse labels; no new expert verification",
        cases=[dict(case=c["id"],source=c["source"],roi=c["roi"],
                    status="sparse_only" if c["probes"] else "unknown",
                    expert_review="pending",dense_annotation=None) for c in proto["cases"]]))


def pareto(rows):
    """Two objectives on this probe only. Never compare areas across cases."""
    return [r for r in rows if not any(
        (s["ink_hit"]>=r["ink_hit"] and s["nonink_black"]<=r["nonink_black"] and
         (s["ink_hit"]>r["ink_hit"] or s["nonink_black"]<r["nonink_black"])) for s in rows)]


def render():
    proto=json.loads((OUT/"protocol.json").read_text())
    results=[json.loads((OUT/"results"/(page+".json")).read_text()) for page in sorted({c["page"] for c in proto["cases"]})]
    rows=[r for result in results for r in result["rows"]]
    scores=[]
    for r in rows:
        assert digest(ROOT/r["path"])==r["sha256"]
        for p in r["probes"]:
            scores.append(dict(case=r["case"],page=r["page"],key=r["key"],**p))
    dump(OUT/"scores.json",scores)
    fields=["case","page","key","probe","ink_total","ink_hit","ink_hit_r2","nonink_black","nonink_area","connected","path_coverage"]
    with (OUT/"scores.csv").open("w") as f:
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");wr.writeheader();wr.writerows(scores)
    frontiers=[]
    for c in proto["cases"]:
        for family in ("howe","sauvola"):
            rr=[r for r in scores if r["case"]==c["id"] and r["key"].startswith(family) and r["key"]!="howe_auto"]
            if rr:
                frontiers.append(dict(case=c["id"],family=family,candidates=len(rr),pareto=pareto(rr),
                                      per_case_oracle=min(rr,key=lambda r:(-r["ink_hit"],r["nonink_black"]))))
    dump(OUT/"frontiers.json",frontiers)
    figures=[]
    keys=["original","csc","sauvola_w61_k0.20","howe_fixed","howe_t0375_c10","howe_t025_c20"]
    for c in proto["cases"]:
        raw=np.asarray(Image.open(ROOT/c["source"]).convert("RGB"))
        arrays={"original":crop(raw,c["roi"])}
        for key in keys[1:]:arrays[key]=binary(load(OUT/"crops"/c["id"]/(key+".png")))
        figures.append(sheet(c,keys,arrays,c["id"]+"_固定と両立候補.png"))
        # Audit each whole-page candidate against the fixed mask. These are
        # changes, not measured damage to text in unlabeled pixels.
    changes=[]
    for result in results:
        p=result["page"]
        fixed=load(OUT/"masks"/p/"howe_fixed.png")
        for key in JOINT:
            m=load(OUT/"masks"/p/(key+".png"))
            changes.append(dict(page=p,key=key,added=int((m&~fixed).sum()),removed=int((fixed&~m).sum()),
                                interpretation="Full-page foreground change, not ink/noise accuracy"))
    dump(OUT/"whole_page_changes.json",changes)
    overview(proto,keys,figures)
    frontier_plot(frontiers)
    dump(OUT/"figure_verification.json",dict(figures=figures,exact_panels=sum(len(f["panels"]) for f in figures),
         all_crop_hashes_checked=True,source_unchanged=all(digest(ROOT/c["source"])==c["source_sha256"] for c in proto["cases"])))
    print("Complete",len(rows),"candidates",len(scores),"probe rows",len(figures),"figures",flush=True)


def sheet(c,keys,arrays,name):
    w,h=c["roi"][2:];gap=20;head=102;label=43
    canvas=Image.new("RGB",(3*w+4*gap,head+2*(h+label+gap)+80),"#faf9f5")
    d=ImageDraw.Draw(canvas)
    def text(xy,s,size=19):d.text(xy,s,font=ImageFont.truetype(FONT,size),fill="#203442")
    text((16,12),c["id"]+"  "+c["title"],27)
    text((16,51),c["page"]+" / 原寸全ページ処理後・同じ座標 / 黒=文字候補",18)
    panels=[]
    for i,key in enumerate(keys):
        x=gap+(i%3)*(w+gap);y=head+(i//3)*(h+label+gap)
        text((x,y),TITLES.get(key,key),18)
        a=arrays[key];canvas.paste(Image.fromarray(a),(x,y+label))
        panels.append(dict(key=key,canvas_xywh=[x,y+label,w,h]))
    text((16,canvas.height-65),"候補A/Bは既知の接触筆画と別画像の斑点で選んだ設定。全体精度・独立評価ではない。",16)
    text((16,canvas.height-37),"CSCはACE経路、他は原画像グレー。Howeは近似Cannyを用いるローカル移植。",16)
    dest=OUT/"figures"/name;dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest)
    saved=np.asarray(Image.open(dest))
    for panel in panels:
        assert np.array_equal(crop(saved,panel["canvas_xywh"]),arrays[panel["key"]])
    return dict(path=str(dest.relative_to(ROOT)),sha256=digest(dest),case=c["id"],panels=panels)


def overview(proto,keys,figures):
    # Thumbnail navigation only; original-scale images above are primary evidence.
    tw,th,gap=300,320,15;head=110
    canvas=Image.new("RGB",(6*(tw+gap)+gap,head+len(proto["cases"])*(th+65)),"#faf9f5")
    d=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT,19)
    d.text((15,12),"黒文字の限界調査：固定条件と既知2領域の両立候補（開発例）",font=ImageFont.truetype(FONT,28),fill="#203442")
    for i,key in enumerate(keys):d.text((gap+i*(tw+gap),60),TITLES[key],font=font,fill="#203442")
    for j,c in enumerate(proto["cases"]):
        y=head+j*(th+65);d.text((15,y),c["id"]+" "+c["title"]+" / "+c["page"],font=font,fill="#203442")
        for i,key in enumerate(keys):
            p=OUT/"originals"/(c["id"]+".png") if key=="original" else OUT/"crops"/c["id"]/(key+".png")
            im=Image.open(p).convert("RGB");im.thumbnail((tw,th),Image.Resampling.NEAREST)
            canvas.paste(im,(gap+i*(tw+gap),y+38))
    canvas.save(OUT/"比較一覧.png")


def frontier_plot(frontiers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs=[cid for cid in ("N01","T01","N05") if any(f["case"]==cid for f in frontiers)]
    fig,axes=plt.subplots(1,len(cs),figsize=(13,4),squeeze=False)
    scores=json.loads((OUT/"scores.json").read_text())
    for ax,cid in zip(axes[0],cs):
        for family,color in [("howe","#176aab"),("sauvola","#d37710"),("csc","#8d428c")]:
            rr=[r for r in scores if r["case"]==cid and r["key"].startswith(family) and r["key"]!="howe_auto"]
            ax.scatter([100*r["nonink_black"]/r["nonink_area"] for r in rr],
                       [100*r["ink_hit"]/r["ink_total"] for r in rr],s=24,alpha=.55,label=family,color=color)
        ax.set(title=cid,xlabel="Black pixels in selected non-ink patches (%)",ylabel="Selected ink-center retention (%)")
        ax.set_ylim(-3,103);ax.grid(alpha=.2);ax.legend()
    fig.suptitle("Exploratory development probes: upper-left preferred; no dense ground truth",fontsize=12)
    fig.tight_layout();fig.savefig(OUT/"文字保持と非文字残存.png",dpi=180);plt.close(fig)


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["prepare","worker","render"]);ap.add_argument("page",nargs="?")
    args=ap.parse_args()
    if args.action=="prepare":prepare()
    elif args.action=="worker":worker(args.page)
    else:render()
