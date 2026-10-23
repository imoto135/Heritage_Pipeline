"""Prepare grouped Makura source pages, provisional labels and paired damage.

Candidate labels and algorithm agreement are explicitly NOT reviewed ground
truth. This script prepares a reviewable pilot, never a formal accuracy claim.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"tmp_work/makura_supervised_pilot_20261021"
SRC=ROOT/"data/makura_kiridashi"
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SPLITS=("train","val","pilot_eval")
COUNTS=dict(train=24,val=8,pilot_eval=8)
PATCH_SIZE=512
VARIANTS=[("identity",0.),("stain_overlap",.25),("stain_overlap",.5),("stain_overlap",.75),
          ("fade",.3),("fade",.55),("fade",.8),("mixed",.35),("mixed",.65)]


def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:8],"little")


def rel(p):return str(Path(p).relative_to(ROOT))


def save_rgb(p,a):
    p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(a.astype(np.uint8)).save(p)


def save_mask(p,m):save_rgb(p,np.where(m,0,255).astype(np.uint8))


def read(p):return np.asarray(Image.open(p).convert("RGB"))


def crop(a,xywh):
    x,y,w,h=xywh;r=a[y:y+h,x:x+w]
    assert r.shape[:2]==(h,w);return r


def select_patches(raw,n=4):
    """Source-only heuristic. Scores screen candidates, not confirmed clean GT."""
    h,w=raw.shape[:2];size=PATCH_SIZE;gray=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY)
    xs=np.linspace(round(w*.1),w-round(w*.08)-size,4).round().astype(int)
    ys=np.linspace(round(h*.18),h-round(h*.1)-size,4).round().astype(int)
    candidates=[]
    for y in ys:
        for x in xs:
            q=gray[y:y+size,x:x+size];ink=q<150;paper=q[q>170]
            fraction=float(ink.mean())
            if not (.025<=fraction<=.25) or paper.size<q.size*.5:continue
            spread=float(np.percentile(paper,95)-np.percentile(paper,5))
            if spread>55:continue
            # Prefer high contrast and representative text occupancy. No CSC or
            # candidate binarization output enters page/patch selection.
            score=spread+100*abs(fraction-.09)
            candidates.append(dict(xywh=[int(x),int(y),size,size],rough_ink_fraction=fraction,
                bright_paper_spread=spread,source_only_score=score))
    chosen=[]
    for c in sorted(candidates,key=lambda c:(c["source_only_score"],c["xywh"])):
        x,y,_,_=c["xywh"]
        if any(max(0,min(x+size,xx+size)-max(x,xx))*max(0,min(y+size,yy+size)-max(y,yy))>0
               for xx,yy,_,_ in [v["xywh"] for v in chosen]):continue
        chosen.append(c)
        if len(chosen)==n:break
    return chosen


def prepare():
    assert not (OUT/"protocol.json").exists(),"Use the frozen protocol already present."
    inventory=[];groups={}
    for p in sorted(SRC.glob("*.png")):
        match=re.fullmatch(r"(\d+)_(\d+)_([01])",p.stem);assert match,p
        book,scan,side=match.groups();gid=f"{book}_{scan}"
        with Image.open(p) as im:size=list(im.size)
        entry=dict(page=p.stem,source=rel(p),group=gid,scan=int(scan),side=int(side),size=size)
        inventory.append(entry);groups.setdefault(gid,[]).append(entry)
    ordered=sorted(groups,key=lambda k:(groups[k][0]["scan"],k))
    block_size=20;nblocks=(len(ordered)+block_size-1)//block_size
    order=np.random.default_rng(20261021).permutation(nblocks).tolist()
    block_split={b:("pilot_eval" if i<3 else "val" if i<6 else "train") for i,b in enumerate(order)}
    initial=[block_split[i//block_size] for i in range(len(ordered))]
    assignments={}
    for i,gid in enumerate(ordered):
        # Embargo two original scan groups on either side of a split boundary.
        neighbors=initial[max(0,i-2):min(len(initial),i+3)]
        split=initial[i] if all(s==initial[i] for s in neighbors) else "embargo"
        assignments[gid]=dict(split=split,block=i//block_size,rank=i)
    for r in inventory:r.update(assignments[r["group"]])
    dump(OUT/"inventory.json",inventory)
    dump(OUT/"split_assignments.json",dict(seed=20261021,block_size=block_size,embargo_radius=2,
        block_split=block_split,groups=assignments,scope="Within-book pilot; not held-out-book generalization"))
    selected=[];rejected=[]
    for split in SPLITS:
        gids=[g for g in ordered if assignments[g]["split"]==split]
        rng=np.random.default_rng(seed("page-selection",split));rng.shuffle(gids)
        found=0
        for gid in gids:
            options=sorted(groups[gid],key=lambda v:seed("side",v["page"]))
            c=options[0];raw=read(ROOT/c["source"]);patches=select_patches(raw)
            if len(patches)<4:
                rejected.append(dict(page=c["page"],reason="Fewer than 4 nonoverlapping source-screened text patches",patches_found=len(patches)))
                continue
            selected.append(dict(**c,source_sha256=sha(ROOT/c["source"]),patches=patches))
            found+=1
            if found==COUNTS[split]:break
        assert found==COUNTS[split],(split,found)
    protocol=dict(date="2026-10-21",source="data/makura_kiridashi",book="200017197",pages=selected,
        requested_counts=COUNTS,patch_size=PATCH_SIZE,patches_per_page=4,variants=VARIANTS,
        role="Paired synthetic pilot and teacher-label candidate review; no model training or formal evaluation yet",
        split="Original scan groups and contiguous 20-group blocks split first; radius-2-group embargo at split boundaries",
        selection="Deterministic source-only text/brightness heuristic within each fixed split; not confirmed clean ground truth",
        teacher=dict(candidate="Sauvola w61 k.2 R.5",comparators=["raw Otsu","local Howe thi.25 c640"],
            agreement="All three agree; 1px neighborhood of disagreements excluded",review_status="pending",
            limitation="Agreement is confidence only, not independent correctness; never score CSC/Howe against these as formal GT"),
        damage=dict(families="Identity, brown alpha-overlay crossing ink, localized contrast fading, mixed",
            strengths="Matched spatial mask/color across severity within a patch; split-specific procedural seeds",
            exclude="No full-opacity cover; no red seal synthesis in this black-text pilot",
            realism="Synthetic diagnostic model; physical realism and transfer to real damage not established"),
        train_ready=False,final_test=False,
        prior_exposure="Makura has prior CSC/benchmark usage. This freezes splits before this new model experiment, not a pristine historical holdout.")
    dump(OUT/"protocol.json",protocol);dump(OUT/"selection_rejections.json",rejected)
    sources=[Path(__file__),ROOT/"tmp_work/howe_diagnosis_20261008/experiment.py",
        ROOT/"tmp_work/howe_diagnosis_20261008/sweep_cli",ROOT/"tmp_work/classical_deep_review_20261008/run.py",
        ROOT/"tmp_work/classical_black_comparison_20261008/compare.py"]
    dump(OUT/"dependencies.json",{rel(p):sha(p) for p in sources})
    for p in sources:
        q=OUT/"source_snapshot"/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    print("Frozen",len(inventory),"source images",len(groups),"groups; pilot",len(selected),"pages",flush=True)


def spatial_field(h,w,anchor,rng):
    yy,xx=np.mgrid[:h,:w];cx,cy=anchor
    rx,ry=rng.uniform(55,125,2)
    field=np.exp(-.5*(((xx-cx)/rx)**2+((yy-cy)/ry)**2)).astype(np.float32)
    rough=cv2.GaussianBlur(rng.random((h,w),dtype=np.float32),(0,0),sigmaX=12)
    rough=(rough-rough.min())/max(float(rough.max()-rough.min()),1e-8)
    field*=.65+.35*rough
    # Compact support, with max=1 so the requested severity is the peak alpha.
    field[field<.08]=0;field/=max(float(field.max()),1e-8)
    return field


def damage(raw,foreground,split,patch_id,family,strength):
    if family=="identity":return raw.copy(),np.zeros(raw.shape[:2],np.float32),dict(seed=None)
    rng=np.random.default_rng(seed("damage-v1",split,patch_id,family))
    points=np.argwhere(foreground);assert len(points)>0
    y,x=points[int(rng.integers(len(points)))];h,w=raw.shape[:2]
    field=spatial_field(h,w,(int(x),int(y)),rng)
    color=np.array([rng.uniform(100,175),rng.uniform(65,120),rng.uniform(35,85)],np.float32)
    result=raw.astype(np.float32)
    if family in ("stain_overlap","mixed"):
        alpha=(field*strength)[...,None]
        result=result*(1-alpha)+color*alpha
    if family in ("fade","mixed"):
        paper=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((61,61),np.uint8)).astype(np.float32)
        fade_alpha=field*strength*(.7 if family=="mixed" else 1.)
        result=result*(1-fade_alpha[...,None])+paper*fade_alpha[...,None]
    return np.rint(np.clip(result,0,255)).astype(np.uint8),field*strength,dict(
        seed=seed("damage-v1",split,patch_id,family),anchor_xy=[int(x),int(y)],brown_rgb=color.tolist())


def worker(page):
    cv2.setNumThreads(1);start=time.monotonic()
    for p,digest in json.loads((OUT/"dependencies.json").read_text()).items():assert sha(ROOT/p)==digest,p
    done=OUT/"page_results"/(page["page"]+".json")
    assert not done.exists(),"Completed page should be reused by the driver."
    assert sha(ROOT/page["source"])==page["source_sha256"]
    raw=read(ROOT/page["source"])
    p=ROOT/"tmp_work/howe_diagnosis_20261008/experiment.py"
    spec=importlib.util.spec_from_file_location("pilot_howe",p);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    gray=e.deep.base.gray(raw);sauvola=e.deep.base.sauvola(gray,61,.2)
    otsu_threshold,otsu=cv2.threshold(cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    terms,_=e.terms(gray,thi=.25)
    [(pre,meta)]=e.sweep(terms,[640],OUT/"scratch"/page["page"],"teacher")
    howe,_=e.post(pre);del terms,pre
    otsu=otsu>0
    disagreement=(sauvola!=howe)|(sauvola!=otsu)
    valid=~cv2.dilate(disagreement.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    records=[];patch_info=[]
    for i,pch in enumerate(page["patches"]):
        patch_id=page["page"]+f"_p{i:02d}"
        folder=OUT/"patches"/page["split"]/patch_id;folder.mkdir(parents=True,exist_ok=True)
        box=pch["xywh"];clean=crop(raw,box);candidate=crop(sauvola,box);v=crop(valid,box)
        save_rgb(folder/"clean.png",clean);save_mask(folder/"target_candidate.png",candidate)
        save_rgb(folder/"agreement_valid.png",v.astype(np.uint8)*255)
        save_mask(folder/"otsu_candidate.png",crop(otsu,box));save_mask(folder/"howe_candidate.png",crop(howe,box))
        known_ink=candidate&v
        assert known_ink.sum()>200,(patch_id,"No confident ink")
        target_hash=sha(folder/"target_candidate.png")
        patch_info.append(dict(patch_id=patch_id,folder=rel(folder),candidate_ink=int(candidate.sum()),
            agreement_ink=int(known_ink.sum()),agreement_valid_fraction=float(v.mean()),
            disagreements=int((~v).sum()),source_xywh=box))
        for family,strength in VARIANTS:
            tag=family+f"_{strength:.2f}";image,alpha,info=damage(clean,known_ink,page["split"],patch_id,family,strength)
            path=folder/(tag+".png");save_rgb(path,image)
            alpha_path=folder/(tag+"_alpha.png");save_rgb(alpha_path,np.rint(alpha*255).astype(np.uint8))
            changed=np.max(np.abs(image.astype(np.int16)-clean.astype(np.int16)),axis=2)>=2
            if family!="identity":assert (changed&known_ink).sum()>0,(patch_id,tag,"No actual ink overlap")
            else:assert np.array_equal(image,clean)
            records.append(dict(id=patch_id+"_"+tag,patch_id=patch_id,page=page["page"],group=page["group"],block=page["block"],
                split=page["split"],source=page["source"],source_sha256=page["source_sha256"],source_xywh=box,
                input=rel(path),clean=rel(folder/"clean.png"),target_candidate=rel(folder/"target_candidate.png"),
                agreement_valid=rel(folder/"agreement_valid.png"),alpha=rel(alpha_path),
                input_sha256=sha(path),target_sha256=target_hash,review_status="pending",training_eligible=False,
                damage_family=family,damage_strength=strength,damage_metadata=info,
                changed_pixels=int(changed.sum()),changed_known_ink=int((changed&known_ink).sum()),known_ink=int(known_ink.sum())))
        assert sha(folder/"target_candidate.png")==target_hash
    result=dict(page=page["page"],split=page["split"],seconds=time.monotonic()-start,otsu_threshold=float(otsu_threshold),
        teacher_howe=dict(c=640,thi=.25,energy=meta["actual_energy"],canny="local skimage approximation"),
        patches=patch_info,records=records)
    dump(done,result)
    print(page["page"],page["split"],"DONE",round(result["seconds"],1),flush=True)
    return page["page"]


def build(workers):
    protocol=json.loads((OUT/"protocol.json").read_text())
    todo=[p for p in protocol["pages"] if not (OUT/"page_results"/(p["page"]+".json")).exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,p) for p in todo]
        for f in as_completed(futures):f.result()
    results=[json.loads((OUT/"page_results"/(p["page"]+".json")).read_text()) for p in protocol["pages"]]
    for split in SPLITS:
        rows=[r for result in results for r in result["records"] if r["split"]==split]
        (OUT/(split+"_manifest.jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    dump(OUT/"patch_index.json",[p for result in results for p in result["patches"]])
    print("Paired candidates prepared",sum(len(r["records"]) for r in results),flush=True)


def checks():
    # Source groups, scan-blocks and patch descendants must never cross splits.
    protocol=json.loads((OUT/"protocol.json").read_text());inventory=json.loads((OUT/"inventory.json").read_text())
    group_roles={};block_roles={}
    for p in inventory:
        group_roles.setdefault(p["group"],set()).add(p["split"])
        if p["split"]!="embargo":block_roles.setdefault(p["block"],set()).add(p["split"])
    assert all(len(v)==1 for v in group_roles.values()) and all(len(v)==1 for v in block_roles.values())
    rows=[json.loads(line) for split in SPLITS for line in (OUT/(split+"_manifest.jsonl")).read_text().splitlines()]
    patches={};seed_roles={}
    for r in rows:
        patches.setdefault(r["patch_id"],[]).append(r)
        assert r["split"] in group_roles[r["group"]]
        assert sha(ROOT/r["input"])==r["input_sha256"]
        assert sha(ROOT/r["target_candidate"])==r["target_sha256"]
        assert r["review_status"]=="pending" and not r["training_eligible"]
        d=r["damage_metadata"]["seed"]
        if d is not None:seed_roles.setdefault(d,set()).add(r["split"])
    assert all(len(v)==1 for v in seed_roles.values())
    assert len(rows)==len(patches)*len(VARIANTS) and len(patches)==len(protocol["pages"])*4
    for patch,rr in patches.items():
        assert len({r["split"] for r in rr})==1 and len({r["target_sha256"] for r in rr})==1
        for family in ["stain_overlap","fade","mixed"]:
            ff=sorted([r for r in rr if r["damage_family"]==family],key=lambda r:r["damage_strength"])
            assert len({r["damage_metadata"]["seed"] for r in ff})==1
            assert all(r["changed_known_ink"]>0 for r in ff)
        clean=read(ROOT/rr[0]["clean"])
        identity=next(r for r in rr if r["damage_family"]=="identity")
        assert np.array_equal(read(ROOT/identity["input"]),clean)
    dump(OUT/"verification.json",dict(source_groups_disjoint=True,blocks_disjoint=True,patch_descendants_disjoint=True,
        damage_seeds_disjoint_between_splits=True,matched_severity_seeds=True,identity_exact=True,actual_ink_overlap_all_damage=True,
        targets_unchanged_all_variants=True,all_hashes_checked=True,candidate_labels_not_training_eligible=True,
        pages=len(protocol["pages"]),patches=len(patches),pairs=len(rows),counts={s:sum(r["split"]==s for r in rows) for s in SPLITS}))
    print("All data checks passed",len(rows),"pairs",flush=True)


def panels(path,title,tiles,labels,ncol=3,footer=""):
    w,h=tiles[0].shape[1],tiles[0].shape[0];gap=20;head=100;lh=42;nrow=(len(tiles)+ncol-1)//ncol
    canvas=Image.new("RGB",(ncol*w+(ncol+1)*gap,head+nrow*(h+lh+gap)+80),"#faf9f5");d=ImageDraw.Draw(canvas)
    def txt(pos,s,size=18):d.text(pos,s,font=ImageFont.truetype(FONT,size),fill="#203442")
    txt((15,12),title,26);txt((15,53),"枕草子・原画素のまま512×512。教師は未確認の候補。",18)
    positions=[]
    for i,(tile,label) in enumerate(zip(tiles,labels)):
        x=gap+(i%ncol)*(w+gap);y=head+(i//ncol)*(h+lh+gap)
        txt((x,y),label,18);canvas.paste(Image.fromarray(tile),(x,y+lh));positions.append([x,y+lh,w,h])
    txt((15,canvas.height-60),footer or "合成の診断用。実損傷への汎化や正式な精度を示す画像ではない。",16)
    txt((15,canvas.height-32),"同じ元パッチと教師候補。輪郭強調・拡大縮小なし。",16)
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path);saved=read(path)
    for xywh,tile in zip(positions,tiles):assert np.array_equal(crop(saved,xywh),tile)
    return dict(path=rel(path),sha256=sha(path),panels=positions)


def render():
    protocol=json.loads((OUT/"protocol.json").read_text());index=json.loads((OUT/"patch_index.json").read_text())
    figures=[];review=[]
    for split in SPLITS:
        pages=[p for p in protocol["pages"] if p["split"]==split]
        # Fixed first and middle pages, before inspecting the teacher candidates.
        for page in [pages[0],pages[len(pages)//2]]:
            patch=next(p for p in index if p["patch_id"]==page["page"]+"_p00")
            folder=ROOT/patch["folder"];clean=read(folder/"clean.png");cand=read(folder/"target_candidate.png")
            valid=np.asarray(Image.open(folder/"agreement_valid.png"))>0
            overlay=clean.copy();overlay[~valid]=[235,40,150]
            tiles=[clean,cand,read(folder/"otsu_candidate.png"),read(folder/"howe_candidate.png"),overlay,
                   read(folder/"stain_overlap_0.50.png")]
            figures.append(panels(OUT/"review"/(patch["patch_id"]+"_教師候補.png"),split+" / "+patch["patch_id"],tiles,
                ["原画像","教師候補：Sauvola","参考：大津","参考：Howe c640 / thi.25","桃=不一致近傍・未確認","入力例：文字と重なる染み"],
                footer="3手法の一致は正しさを保証しない。教師候補の薄線・点画・汚れは確認が必要。"))
            variants=["clean.png","stain_overlap_0.25.png","stain_overlap_0.75.png","fade_0.30.png","fade_0.80.png","mixed_0.65.png"]
            figures.append(panels(OUT/"review"/(patch["patch_id"]+"_損傷比較.png"),split+" / 損傷比較 / "+patch["patch_id"],
                [read(folder/f) for f in variants],["原画像","染み .25","染み .75","退色 .30","退色 .80","染み＋退色 .65"]))
            review.append(dict(patch_id=patch["patch_id"],split=split,status="pending",source=page["source"],candidate_gt=False))
    dump(OUT/"figure_verification.json",figures);dump(OUT/"review_queue.json",review)
    verification=json.loads((OUT/"verification.json").read_text())
    report=["# 枕草子の教師あり学習用・合成データ試作","", "2026-10-21。DKDSの対応ペア作成の考え方を参考にした独自の染み・退色合成。DKDS配布データや公式合成実装の再現とは扱わない。", "",
        f"原本{verification['pages']}ページ、{verification['patches']}パッチ、{verification['pairs']}入力・教師候補ペアを作成。学習24ページ、検証8ページ、同資料内の試行評価8ページ。", "",
        "## 分割", "", "665画像を原画像番号338群にまとめ、20群ずつの連続ブロックで先に分割した。分割境界の前後2群は除外。`_0/_1`・同じ原ページのパッチ・全損傷版は同じ分割に置いた。枕草子1資料内の試行であり、別資料への汎化を検証する最終評価ではない。", "",
        "## 教師候補", "", "原画像全ページからSauvola、大津、Howe移植を計算。Sauvolaを二値教師候補として保存し、3法不一致の1画素近傍を除いた一致範囲も保存した。一致は正しさの保証ではなく、専門家による確認・修正も未実施。全マニフェストは`review_status=pending`、`training_eligible=false`。正式な精度比較の正解にも使わない。", "",
        "## 合成", "", "原画像・染み3強度・退色3強度・混合2強度の9条件。黒文字の一致領域に中心を置き、実際に文字画素が変化したことを検証した。既存の文字を避ける合成とは別系列。強度間では位置・形状を揃え、分割間では生成シードを共有しない。", "",
        "## 確認用PNG", ""]
    for p in review:
        report.append(f"- {p['split']} / {p['patch_id']}：[教師候補](review/{p['patch_id']}_教師候補.png)・[損傷比較](review/{p['patch_id']}_損傷比較.png)。")
    report += ["", "## 検証と残る作業", "", "グループ・連続ブロック・損傷シードの分割間非重複、教師候補が全損傷で不変、正常入力の一致、全損傷で実際に文字に重なること、保存画像ハッシュ、比較PNGの画素一致を検証した。", "",
        "次は、少量の教師候補について薄線・点画・汚れの確認と修正を行う。合成画像の自然さと実際の難例との対応も確認する。今回の候補を無確認で正式な教師正解として学習・採点しない。RA-GAN・U-Netの学習はこの生成処理には含まない。", "",
        "[設定](protocol.json) · [分割](split_assignments.json) · [生成検証](verification.json) · [パッチ一覧](patch_index.json)"]
    (OUT/"データ試作報告.md").write_text("\n".join(report)+"\n")
    print("Review PNGs",len(figures),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["prepare","build","check","render"]);ap.add_argument("--workers",type=int,default=2)
    args=ap.parse_args()
    if args.action=="prepare":prepare()
    elif args.action=="build":build(args.workers)
    elif args.action=="check":checks()
    else:render()
