from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

CASES = [
    dict(id="01", label="薄い筆画", name="200010454_00016_2", assessment="有力", note="褪色した表文字がページ全体にあり、目立つ裏移りはない。"),
    dict(id="02", label="広い染みと低コントラスト文字", name="200015843_00107_1", assessment="弱い代替", note="色のある紙面の褐色の濃淡と本文。ただし旧02ほど文字が淡くない。"),
    dict(id="03", label="染みと筆画の接触", name="200015843_00105_1", assessment="候補", note="褐色の染み状部分が本文の文字列に重なる。"),
    dict(id="04", label="本文外の孤立斑点", name="200021071_00032_1", assessment="弱い代替", note="本文の上の余白に孤立した黒点がある。旧04より点は小さい。"),
    dict(id="06", label="虫食い状の紙面損傷", name="200003803_00003_1", assessment="暫定・既視", note="左上の枝分かれ状の欠損。原因未確定。過去の方法出力確認歴あり。"),
    dict(id="07", label="裏移りと表文字", name="200021071_00033_2", assessment="有力", note="左側も含め表文字があり、青灰色像は直前ページの左右反転像と対応。"),
]


def source(name: str) -> Path:
    book = name.split("_")[0]
    path = ROOT / "data" / "test" / book / f"{name}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    small = ImageFont.truetype(FONT, 25)
    med = ImageFont.truetype(FONT, 25)
    canvas = Image.new("RGB", (1805, 1800), "#f4f4f4")
    draw = ImageDraw.Draw(canvas)
    for idx, case in enumerate(CASES):
        x = 20 + idx % 3 * 595
        y = 25 + idx // 3 * 880
        draw.rounded_rectangle((x, y, x + 575, y + 850), radius=12, fill="white", outline="#bbb", width=2)
        draw.text((x + 15, y + 10), case["id"] + "  " + case["label"], font=med, fill="black")
        if case["name"]:
            path = source(case["name"])
            with Image.open(path) as original:
                case["dimensions"] = list(original.size)
                im = ImageOps.exif_transpose(original).convert("RGB")
                im.thumbnail((545, 755))
                canvas.paste(im, (x + (575 - im.width) // 2, y + 55 + (755 - im.height) // 2))
            case["source"] = str(path.relative_to(ROOT))
            case["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            draw.text((x + 15, y + 816), case["name"], font=small, fill="#333")
        else:
            draw.text((x + 150, y + 400), "NO VERIFIED MATCH", font=med, fill="#8b3a3a")
    canvas.save(OUT / "test由来_6課題候補.png")
    (OUT / "test_candidates.json").write_text(json.dumps({"scope": "data/test raw JPEG only", "screened_pages": 1404, "books": 11, "selection_stage": "page-level shortlist; ROIs and final evaluation use not fixed", "omitted_case": "05: color-page dense granular marks; no convincing test example found", "cases": CASES}, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# test原画像からの6課題再検索",
        "",
        "2026-10-09。`data/test` の11資料・1,404 JPEGを原画像の縮小索引で通覧し、候補ページを原画像で再確認した。CSC・Sauvola・Howe・DDRMの出力は、この検索では判定に使っていない。",
        "",
        "[6課題の候補一覧PNG](test由来_6課題候補.png)／[機械可読一覧](test_candidates.json)",
        "",
        "[原画像・CSC・Sauvola・Howeの6課題比較PNG](three_method_comparison/6課題_CSC_Sauvola_Howe_比較.png)／[比較条件と画像ハッシュ](three_method_comparison/verification.json)。原寸ページで各手法を処理し、同一座標を切り出した。",
        "",
        "[Sauvola・Howeのパラメータ探索結果](parameter_tuning/調整結果.md)に、固定候補の比較画像と共通設定を維持する判断を記録した。",
        "",
        "| ID | 判定 | testの候補ページ | 原画像で見た理由 |",
        "| --- | --- | --- | --- |",
    ]
    for case in CASES:
        page = f"[{case['name']}](../../data/test/{case['name'].split('_')[0]}/{case['name']}.jpg)" if case["name"] else "—"
        lines.append(f"| {case['id']} {case['label']} | {case['assessment']} | {page} | {case['note']} |")
    lines.extend([
        "",
        "この一覧は**ページ単位の候補**で、ROI・画素正解・最終評価セットは固定していない。01は目立つ裏移りのない褪色した表文字に変更した。02と04は旧例より弱い代替であり、同等の課題を再現できたとは言わない。02と03は同じ資料に由来する。04は本文付きだが黒点が小さく、旧04の濃い斑点とは難しさが異なる。06の『虫食い』は画像上の形状説明で、虫害と確定していない。07は左側にも表文字があり、青灰色像は[直前ページの左右反転像](07_修正候補隣接照合.png)と字形・位置が対応するため裏移りと判断できる。ただし画素単位の裏移り正解マスクはない。",
        "",
        "01の追加探索では、[薄い表文字の候補比較](01_薄い文字候補比較.png)を作成した。同じ資料の全ページと他のtest資料の索引を再確認したが、`00016_2` より明確に淡く、文字量が十分で、かつ裏移りが目立たないページは確認できていない。明るいが本文が少ないページは01の代表例には採らない。",
        "",
        "06は過去の二値化出力を確認したページを含む。testフォルダ由来でも、これを今後のパラメータ調整の**未使用・独立テスト**とは呼べない。候補の特徴を見ながらパラメータを調整すれば、他の候補も評価用の独立性を失う。旧[開発用7例](../challenge_image_catalog_20261009/対象画像.md)と混ぜず、調整用と最終評価用の役割を先に分ける必要がある。",
        "",
        "05はtest内で旧例に相当する濃い粒状像が確認できず、ユーザーの判断により今回の候補一覧から外した。既に方法出力を見た資料は、分割し直しても未使用評価には戻らない。",
        "",
        "索引の再生成: `python tmp_work/test_only_challenge_search_20261009/screen_originals.py`。候補一覧の再生成: `python tmp_work/test_only_challenge_search_20261009/make_shortlist.py`。",
    ])
    (OUT / "test由来候補.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
