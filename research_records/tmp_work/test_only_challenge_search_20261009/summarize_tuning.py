"""Render comparable parameter columns and record the bounded tuning result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageChops


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "parameter_tuning"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
ROIS = {
    "01": [570, 600, 750, 850], "02": [700, 1200, 850, 850],
    "03": [750, 700, 850, 850], "04": [550, 160, 800, 820],
    "06": [80, 95, 620, 530], "07": [800, 750, 850, 900],
}
METHODS = [
    ("sauvola", ["w61_k0.10", "w61_k0.20", "w61_k0.30"], "Sauvola：窓61でkを比較", "Sauvola_k比較.png"),
    ("howe", ["edge0.25_c160", "edge0.40_c160", "edge0.50_c160"], "Howe移植：c=160でエッジ閾値を比較", "Howe_エッジ閾値比較.png"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cases = json.loads((HERE / "test_candidates.json").read_text())["cases"]
    result = json.loads((OUT / "grid_results.json").read_text())
    assert result["status"] == "complete" and len(result["cases"]) == 6
    audit = {"fixed_condition_baseline_exact": [], "figures": {}}
    for method, keys, title, filename in METHODS:
        max_w = max(ROIS[c["id"]][2] for c in cases)
        gap, left, head, row_head, footer = 18, 25, 134, 62, 88
        width = 2*left + 4*max_w + 3*gap
        height = head + sum(row_head+ROIS[c["id"]][3]+gap for c in cases)+footer
        sheet = Image.new("RGB", (width, height), "#f5f3ed")
        d = ImageDraw.Draw(sheet)
        f1 = ImageFont.truetype(FONT, 29)
        f2 = ImageFont.truetype(FONT, 21)
        f3 = ImageFont.truetype(FONT, 17)
        d.text((left, 10), title, font=f1, fill="#20323c")
        d.text((left, 53), "全ページで処理後、同一座標を原寸で切り出し。黒＝前景。", font=f3, fill="#405462")
        d.text((left, 89), "原画像", font=f2, fill="#1d4357")
        for j, key in enumerate(keys, 1):
            d.text((left+j*(max_w+gap), 89), key, font=f2, fill="#1d4357")
        y = head
        for case in cases:
            cid = case["id"]
            x, yy, w, h = ROIS[cid]
            d.rectangle((left-8, y, width-left+8, y+row_head+h+4), fill="white", outline="#cad2d5", width=2)
            d.text((left, y+7), f"{cid} {case['label']}", font=f2, fill="#20323c")
            d.text((left, y+36), f"{case['name']} / {x},{yy},{w},{h}", font=f3, fill="#536a77")
            with Image.open(ROOT / case["source"]) as original:
                image = original.convert("RGB").crop((x, yy, x+w, yy+h))
            sheet.paste(image, (left, y+row_head))
            for j, key in enumerate(keys, 1):
                p = OUT / "crops" / method / cid / f"{key}.png"
                im = Image.open(p).convert("RGB")
                assert im.size == (w, h)
                sheet.paste(im, (left+j*(max_w+gap), y+row_head))
                if key in ("w61_k0.20", "edge0.40_c160"):
                    baseline = Image.open(HERE / "three_method_comparison" / "masks" /
                                          case["name"] / f"{method}.png").convert("L")
                    assert ImageChops.difference(baseline.crop((x, yy, x+w, yy+h)), im.convert("L")).getbbox() is None
                    audit["fixed_condition_baseline_exact"].append(f"{cid}/{method}")
            y += row_head+h+gap
        d.text((left, height-footer+12), "探索用のパラメータ比較。黒画素の増減は正解率を意味しない。05は除外。", font=f3, fill="#405462")
        d.text((left, height-footer+42), "Howeは局所移植・Canny近似。調整に使った6例は独立テスト評価に使わない。", font=f3, fill="#405462")
        path = OUT / filename
        sheet.save(path)
        audit["figures"][method] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "size": list(sheet.size)}
    (OUT / "summary_verification.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")

    lines = [
        "# Sauvola・Howeのパラメータ探索",
        "",
        "2026-10-09。test由来の6事例を**開発用事例**として使用した。原寸の原画像全ページを処理し、課題ROIを同じ座標で切り出した。これは未使用テストの精度評価ではない。",
        "",
        "[Sauvolaのk比較](Sauvola_k比較.png) · [Howeのエッジ閾値比較](Howe_エッジ閾値比較.png) · [全9条件の画像一覧](grids/) · [探索表](grid_results.json)",
        "",
        "Sauvolaは窓幅31/61/101、k=0.10/0.20/0.30を探索した。Howe移植は平滑化係数c=80/160/320、Canny高閾値の最大勾配比0.25/0.40/0.50を探索した。低閾値比0.10とsigma=0.6は固定した。既定値はSauvola窓61・k=0.20、Howe c=160・高閾値比0.40。比較画像に使った既定値の切り出しと今回の再計算結果は12件すべて画素一致した。",
        "",
        "**結論：6例に共通する設定は変更しない。** 01の薄い筆画を多く残すSauvola k=0.10は07の裏移りも増やし、k=0.30は裏移りを減らす代わりに01の細線も削る。窓31では細かな黒点が増え、窓101では文字が太りやすい。Howeはcの差が小さく、高閾値比0.25では03の染み周辺の黒像が増える。高閾値比0.50は03の黒像を減らすが、文字との境界を正解マスクで区別できない。04の孤立黒点は探索した全条件で残る。したがって、目視だけでいずれかを全例の改善条件と主張しない。",
        "",
        "| 対象 | Sauvola k=0.10 | 既定 k=0.20 | k=0.30 | Howe 高閾値0.25 | 既定0.40 | 0.50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in result["cases"]:
        s = entry["sauvola"]
        h = entry["howe"]
        vals = [s[f"w61_k{k:.2f}"]["black_pixels"] for k in (.1, .2, .3)]
        vals += [h[f"edge{k:.2f}_c160"]["black_pixels"] for k in (.25, .4, .5)]
        lines.append("| " + entry["id"] + " | " + " | ".join(f"{n:,}" for n in vals) + " |")
    lines.extend(["", "表の数値は各ROIの黒画素数であり、文字・汚れ・裏移りの内訳を表さない。06は紙面損傷なので、本文二値化の調整目的から除く。02と04は旧例より弱い代替。", "",
                  "次に改善を狙うなら、濃淡だけの単一設定より、表文字・裏移り・染みの領域注釈または別資料による検証が必要。今回の6例でさらに手動調整した値を独立テスト結果として報告しない。"])
    (OUT / "調整結果.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
