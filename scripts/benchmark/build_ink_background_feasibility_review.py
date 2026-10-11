"""Build a review of three completed, frozen decomposition feasibility pilots."""
from pathlib import Path
import html
import json
from html.parser import HTMLParser
import cv2
import numpy as np
import probe_ink_background_decomposition as base

ROOT = base.ROOT
OUT = ROOT/"tmp_work/ink_background_feasibility_review_20260930"
PILOTS = [
    ROOT/"tmp_work/ink_background_profile_pilot_v0_20260930",
    ROOT/"tmp_work/ink_background_chromatic_pilot_v1_20260930",
    ROOT/"tmp_work/ink_background_flexible_pilot_v2_20260930",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checked = 0
    dependencies = {}
    for pilot in PILOTS:
        complete = json.loads((pilot/"complete.json").read_text())
        for rel, expected in complete["outputs"].items():
            assert base.sha(pilot/rel) == expected, str(pilot/rel)
            checked += 1
        for name in ["complete.json", "manifest.json", "verification.json", "diagnostics.json"]:
            p = pilot/name; dependencies[str(p.relative_to(ROOT))] = base.sha(p)
    rows = json.loads((PILOTS[2]/"diagnostics.json").read_text())
    cases = json.loads((base.V6/"manifest.json").read_text())["cases"]
    cases = {r["id"]: r for r in cases}
    table, cards = [], []
    for row in rows:
        page, name = row["page"], row["roi"]
        crop = base.read(PILOTS[2]/page/name/"original.png")
        x, y, w, h = row["xywh"]
        edit = base.read(base.V6/page/"split_direction_edit_allowed.png", True)[y:y+h, x:x+w]
        baseline = crop.copy()
        baseline[edit] = np.rint(.55*crop[edit]+.45*np.array([23, 178, 235])).astype(np.uint8)
        path = OUT/page/name/"v6_edit_allowed.png"; base.write(path, baseline)
        source = base.V6/page/"split_direction_edit_allowed.png"
        dependencies[str(source.relative_to(ROOT))] = base.sha(source)
        limited = row["scenarios"]["restricted_background"]["all"]
        flexible = row["scenarios"]["flexible_colored_background"]["all"]
        keys = ["ink_required", "ambiguous", "low_ink", "unexplained"]
        table.append("| "+name+" | "+str(row["profiles"])+" | "+
                     " / ".join(str(limited[k]) for k in keys)+" | "+
                     " / ".join(str(flexible[k]) for k in keys)+" |")
        figures = [
            ("原画像", "../"+PILOTS[2].name+"/"+page+"/"+name+"/original.png"),
            ("v6の編集許可（面）", page+"/"+name+"/v6_edit_allowed.png"),
            ("通常背景での断面判定（点）", "../"+PILOTS[2].name+"/"+page+"/"+name+"/restricted_background.png"),
            ("柔軟な色付き背景も許容（点）", "../"+PILOTS[2].name+"/"+page+"/"+name+"/flexible_colored_background.png")]
        cards.append('<article><h2>'+html.escape(page+" / "+name)+'</h2><p>同じ自動断面 '+str(row["profiles"])+' 件。点の分類は編集マスクではない。</p><div class="grid">'+
                     "".join('<figure><figcaption>'+label+'</figcaption><a href="'+rel+'"><img src="'+rel+'"></a></figure>'
                             for label, rel in figures)+'</div></article>')
    note = """# インク・背景分解を試した結果と勝算

2026-09-30。ユーザーの「やってみよう、勝算はどれくらいあるだろう」に対し、tmux codexで断面の成立確認を3方式実装・実行した。対象は既知の3ページ・5矩形・692自動断面。全て開発資料で、画素GTや独立の精度評価ではない。

## 判断

**色材の共通性は利用候補になるが、今回の断面方式はv6を超える前処理として成立していない。** 編集許可へは採用しない。単純な色材モデルだけで大きな改善を出す勝算は低め。色材と2次元の筆画・背景の空間制約を組み合わせる研究は、引き続き低〜中程度の見込みと考える。これは研究上の主観的な判断であり、統計的な成功確率・採択確率ではない。

## 実際に試したもの

1. v0：RGB差・encoded OD・線形sRGBのODで、背景とGaussian型のインクを同時に推定。背景は直線・曲線・境界・斑点を許容し、色付き折り目も比較。
2. v1：v0の位置・色方向を保ち、共通の明暗成分を除いた色差2成分を再構成。紙の明暗変化を分ける効果を確認。
3. v2：v1に非負の区分線形インク量を追加し、実際の凹凸や複数ピークを説明できるか確認。同じ細かさの任意の色付き背景による説明も比較。

各断面は4分割し、当てはめに使わない画素値で再構成誤差を確認した。ただし断面の位置・方向は元画像全体で決め、画素にも相関がある。これは独立データでの精度ではない。教師ラベル・ニューラル学習・手動画素GT・画像修復・密な新規編集マスクは使用していない。

## 得られた手掛かり

明瞭な朱候補を、対象矩形と40px周囲を除いて自動抽出した。encoded ODで、推定に使わない断面の色方向と推定方向の角度中央値は、朱印ページで約3.0度、濃い朱印のある対照ページで約3.5度だった。ページ内で共通の色変化を使う仮説には手掛かりがある。ただし元の色候補で選択した範囲の性質であり、薄い朱・赤褐色筆記・別冊にも成立する証明ではない。

一本のGaussian型では実際の印影や紙の模様を表しきれない。v2の柔軟なインク量では、薄い朱印の160断面中81断面で、検討した通常背景の下ではインク寄与が必要になった。同じ場所のv1は3断面だった。これは条件付きの説明件数で、朱の検出率や改善率ではない。

## 問題になった点

**背景の自由度を増やすと、同じ印影を背景だけでも説明できる。** 柔軟な色付き背景を許すと、薄い朱印は1件がインク必要、19件が曖昧、114件が低寄与、26件が説明不足になった。原寸の目視では、朱の痕跡が見える場所にも低寄与の点がある。したがって青い点の「低寄与」を、文字なし・編集安全の根拠に使わない。検討した色方向・モデル群が実際のインクの変化を十分に含まない可能性もある。再構成誤差が小さいことだけでは成分の意味は決まらない。

朱近傍の茶色・折り目でも、通常背景では52断面中37断面がインク必要になった。柔軟な背景を許すと必要0、曖昧12、低寄与34、説明不足6となった。この矩形の目視では色付き折り目にもインクの説明を当てはめる点がある。

赤褐色筆記ページは、既存の確定朱のrobust seedが26画素、対象矩形外の自動断面は6件しかなかった。色方向の推定用・確認用に分ける条件を満たさず、2矩形とも推定不能として残した。判定のしきい値を緩めたり、別ページの色方向を借りたりして埋めなかった。既存CSCに依存した初期化では、改善したい難例で初期化できない、という課題がある。

## 同じ断面のモデル依存

件数の順は「インク必要／曖昧／低寄与／説明不足」。色差・初期化・空間モデルの仮定に依存する。矩形は墨や背景も含み、各断面をインク／汚れのGTとはしていない。

| 既知矩形 | 自動断面数 | v2・通常背景 | v2・柔軟な色付き背景も許容 |
|---|---:|---|---|
"""+"\n".join(table)+"""

## 次に試す条件

現在のv6と、別冊6冊の入力は固定したまま残す。断面だけの探索をいったん区切り、次の試作は2次元の局所パッチで行う。

- インクの色方向と、観測された線の向き・幅・近傍との色の整合性を一緒に推定する。紙・汚れのモデルにも境界・折り目を持たせる。線状であるという条件だけでは確定しない。
- 赤褐色筆記で、既存の確定朱だけに依存しない自動初期化を検討する。初期化用と確認用の空間分割を保ち、対象矩形を初期化の答えとして使わない。
- 分解の再構成が良くても「インクなし」と確定しない。有限候補の寄与範囲と、モデル不一致の保留を分ける。単に保留を増やす方式との比較が必要。

この2次元の方法は未実装。初期化と背景自由度を解消できるかが次の継続判断になる。手動画素GTは、その成立確認の後でよい。論文の優越性を示す段階では少数の確認済みGTと冊単位の固定比較を残す。

## 保存した出力

- [原寸の比較ビューア](index.html)
- [v0：断面の画像形成モデル](../ink_background_profile_pilot_v0_20260930/report.md)
- [v1：共通の明暗成分を除いた色差](../ink_background_chromatic_pilot_v1_20260930/report.md)
- [v2：非負のインク量曲線](../ink_background_flexible_pilot_v2_20260930/report.md)
- [元の提案と先行研究](../preprocessing_fundamental_proposal_v2_20260930.md)
"""
    (OUT/"report.md").write_text(note)
    (OUT/"index.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>インクと背景の分解：成立確認</title><style>body{font-family:sans-serif;margin:24px;background:#faf8f3;color:#222}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{max-width:100%}article{margin:36px 0}a{color:#245889}p{line-height:1.6}</style><h1>インクと背景の分解：成立確認</h1><p>3ページ・5矩形・692自動断面。原寸画像をクリックして確認。既知開発資料であり、GT精度ではありません。</p><p>点の色：緑＝検討した説明ではインク必要、黄＝曖昧、青＝条件付き低寄与、紫＝説明不足・初期化不足。<strong>青い点を編集安全とは判定しません。</strong></p><p>v6は面の編集許可、試作は点の診断で、同じ種類の出力ではありません。</p><a href="report.md">分析と勝算</a> · <a href="verification.json">検証</a>'+''.join(cards)+'</html>')
    base.dump(OUT/"summary.json", dict(pages=3, rois=5, profiles=sum(r["profiles"] for r in rows),
        subjectively_estimated_outlook=dict(simple_pigment_only="low", pigment_plus_2d_structure="low_to_medium"),
        improvement_over_v6_demonstrated=False, independent_accuracy_test=False, next_2d_method_implemented=False))
    base.dump(OUT/"verification.json", dict(status="pending"))
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, val in attrs:
                if key in ["href", "src"] and val:
                    assert (OUT/val).is_file(), val
    Links().feed((OUT/"index.html").read_text())
    for p, h in dependencies.items():
        assert base.sha(ROOT/p) == h, p
    base.dump(OUT/"verification.json", dict(status="passed", frozen_pilot_output_hashes_checked=checked,
        dependencies_unchanged=True, local_viewer_links_checked=True, browser_rendering_tested=False,
        original_native_crops_reused=True, pixel_GT=False, neural_training=False))
    base.dump(OUT/"manifest.json", dict(source_sha256=base.sha(Path(__file__)), dependencies=dependencies))
    print(json.dumps(json.loads((OUT/"verification.json").read_text()), ensure_ascii=False))


if __name__ == "__main__":
    main()
