"""Package input-only red examples chosen by visual inspection before method use.

Rectangles locate review crops; they are not pixel GT or training labels.
Run with --verify-only to recheck source identity, native crops and provenance.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import shutil

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/red_positive_collection_20260930'
PRIOR = ROOT / 'tmp_work/training_free_protection_v2_20260930/manifest.json'

# Selected after inspecting all 100 candidate thumbnails and these 14 native crops.
# The color screening scores and later segmentation outputs are not ink GT.
CASES = [
    ('100249376_00056_1', ['薄い朱印', '黄変紙', '薄い灰青色の文字と重なり'],
     '黄変した紙の矩形印影。灰青色の文字と重なる。退色原因や裏写りかどうかは未確定。',
     [('薄い矩形印影', [340, 2170, 550, 520])]),
    ('100249537_00003_2', ['朱線の囲み', '墨と重なり', '黄変紙'],
     '墨書の上に赤い囲み線がある。線幅・濃度にむらがある。',
     [('上部の囲み線と墨', [1190, 480, 430, 970]),
      ('下部の囲み線と墨', [960, 1550, 380, 600])]),
    ('200006663_00002_2', ['朱印', '細い印影', '黒い挿絵と重なり'],
     '細く欠けた矩形印影が黒い挿絵と交差する。',
     [('挿絵に重なる印影', [730, 2180, 730, 780])]),
    ('200020019_00002_2', ['朱印', '墨と重なり', '印影内の濃淡'],
     '上部・下部の矩形印影が墨書と重なり、朱にも濃淡がある。',
     [('上部の印影と墨', [1150, 50, 600, 850]),
      ('下部の印影と墨', [1550, 2520, 300, 420])]),
    ('200020019_00082_2', ['薄い朱印', '濃い朱印との対比', '同冊別ページ'],
     '上側の薄い赤色印影と下側の明瞭な印影。同冊の別ページであり、独立した冊として数えない。',
     [('薄い印影', [40, 1300, 480, 500]),
      ('明瞭な印影', [580, 2380, 500, 480])]),
    ('200021086_00032_2', ['薄い朱印', '低コントラスト', '紙色に近い赤'],
     '淡い大きな印影。紙面の斑点もある。近代の蔵書管理印は主な確認範囲に含めていない。',
     [('低コントラストの印影', [570, 350, 650, 1100])]),
    ('200021763_00002_2', ['赤褐色の筆記', '細い筆画', '筆画内の濃淡', '黄変紙'],
     '赤褐色の縦書き筆記。濃い箇所と薄い細線が共存する。暗い朱の代表例として十分かは未確定。',
     [('上側の赤褐色筆記', [190, 580, 520, 800]),
      ('下側の赤褐色筆記と細線', [190, 1280, 560, 1260])]),
    ('200025191_00004_2', ['朱印', '細い印影', '墨と接触'],
     '明るい朱の細い矩形印影。左枠の一部が墨と接触する。',
     [('細い印影と墨の接触', [550, 1850, 440, 850])]),
    ('100249371_00002_1', ['赤色の図形・文字', '墨と重なり', '青い罫線と重なり'],
     '赤色の図形・文字が墨と青い罫線に重なる。朱印か色刷りかは未確定。',
     [('上部の赤と青・墨', [880, 60, 740, 680]),
      ('下部の赤と墨', [180, 2050, 530, 650])]),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def verify():
    selection = json.loads((OUT / 'selection.json').read_text())
    outputs = json.loads((OUT / 'complete.json').read_text())['outputs']
    assert sha(Path(__file__)) == selection['packaging_source_sha256']
    for relative, digest in selection['dependencies'].items():
        assert sha(ROOT / relative) == digest, relative
    for relative, digest in outputs.items():
        assert sha(OUT / relative) == digest, relative
    excluded = set(selection['excluded_development_book_ids'])
    all_rois = 0
    for row in selection['cases']:
        assert row['book'] not in excluded
        assert sha(ROOT / row['original']) == row['original_sha256']
        im = cv2.imread(str(ROOT / row['original']))
        h, w = im.shape[:2]
        assert (w, h) == (row['width'], row['height'])
        saved = cv2.imread(str(OUT / row['original_png']))
        assert np.array_equal(im, saved), row['id']
        for roi in row['rois']:
            x, y, rw, rh = roi['xywh']
            assert min(x, y) >= 0 and min(rw, rh) > 0
            assert x + rw <= w and y + rh <= h
            saved = cv2.imread(str(OUT / roi['png']))
            assert np.array_equal(saved, im[y:y+rh, x:x+rw]), roi['png']
            all_rois += 1
    # Check all generated HTML links to local artifacts; the viewer uses no JS/network.
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ('href', 'src') and value and not value.startswith('#'):
                    assert (OUT / value).is_file(), value
    Links().feed((OUT / 'index.html').read_text())
    result = dict(status='passed', pages=len(selection['cases']),
                  books=len({r['book'] for r in selection['cases']}), native_rois=all_rois,
                  output_hashes_checked=len(outputs), original_decoded_pixels_unchanged=True,
                  native_crops_equal_source=True, development_books_disjoint=True,
                  frozen_method_provenance_unchanged=True, html_links_checked=True,
                  browser_rendering_tested=False, method_applied=False,
                  training_performed=False, pixel_ground_truth_created=False)
    dump(OUT / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build():
    cv2.setNumThreads(1)
    screened = json.loads((OUT / 'screening_manifest.json').read_text())
    completion = json.loads((OUT / 'screening_complete.json').read_text())
    queue = {r['id']: r for r in json.loads((OUT / 'inspection_queue.json').read_text())}
    prior = json.loads(PRIOR.read_text())
    script = ROOT / 'scripts/benchmark/refine_training_free_protection.py'
    template = script.with_name('training_free_protection_viewer.html')
    assert sha(script) == prior['source_sha256']
    assert sha(template) == prior['viewer_template_sha256']
    dependencies = {str(p.relative_to(ROOT)): sha(p) for p in [PRIOR, script, template,
        OUT / 'screening_manifest.json', OUT / 'candidate_pages.json',
        OUT / 'inspection_queue.json', OUT / 'screening_complete.json',
        OUT / 'screening_scores.json', OUT / 'book_coverage.json',
        ROOT / 'scripts/benchmark/collect_red_source_candidates.py']}
    selected = []
    for stem, tags, observation, rois in CASES:
        source = queue[stem]
        original = ROOT / source['original']
        im = cv2.imread(str(original))
        if im is None:
            raise OSError(original)
        h, w = im.shape[:2]
        assert source['book'] not in screened['excluded_development_book_ids']
        dest = OUT / 'pages' / stem
        dest.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(dest / 'original.png'), im)
        scale = min(1., 1100 / h, 800 / w)
        preview = cv2.resize(im, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(dest / 'preview.jpg'), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
        row = dict(id=stem, original=source['original'], original_sha256=sha(original),
                   book=source['book'], split=source['split'], width=w, height=h,
                   screening_candidate_index=source['candidate_index'], tags=tags,
                   observation=observation, original_png=f'pages/{stem}/original.png',
                   preview=f'pages/{stem}/preview.jpg', rois=[])
        for i, (name, box) in enumerate(rois, 1):
            x, y, rw, rh = box
            assert min(x, y) >= 0 and x + rw <= w and y + rh <= h
            relative = f'pages/{stem}/roi_{i:02d}.png'
            assert cv2.imwrite(str(OUT / relative), im[y:y+rh, x:x+rw])
            row['rois'].append(dict(id=f'{stem}_roi_{i:02d}', name=name, xywh=box, png=relative,
                                    visually_inspected=True, pixel_ground_truth=False))
        selected.append(row)
    selection = dict(
        purpose='Input-only red-positive challenge collection before fixed-method check',
        scope='Books not used in the latest 12-page development/regression set; not claimed unused throughout the project',
        observer='Codex image viewing; user review pending',
        selection_protocol='All 100 input-only candidate thumbnails inspected, then all 14 native crops inspected; no segmentation results used for selection',
        caveat='Color-based purposive positive collection; cannot estimate population prevalence or precision/recall. Crops are review locations, not pixel GT. Ink chemistry and print/stamp origin unverified.',
        dark_red_coverage='Red-brown handwriting with dark portions included; truly dark red coverage remains insufficient',
        screening=completion, excluded_development_book_ids=screened['excluded_development_book_ids'],
        frozen_method=dict(manifest=str(PRIOR.relative_to(ROOT)), settings=prior['settings'],
                           source_sha256=prior['source_sha256'], viewer_template_sha256=prior['viewer_template_sha256']),
        method_applied=False, training_performed=False, pixel_ground_truth_created=False,
        packaging_source_sha256=sha(Path(__file__)), dependencies=dependencies, cases=selected)
    existing = OUT / 'selection.json'
    if existing.exists() and json.loads(existing.read_text()) != selection:
        raise RuntimeError('Collection/provenance changed; create a new output version')
    dump(existing, selection)
    shutil.copyfile(__file__, OUT / 'packaging_source_snapshot.py')
    cards = []
    report_rows = []
    for row in selected:
        esc = html.escape
        regions = ''.join(
            f'<figure><figcaption>{esc(roi["name"])} · {roi["xywh"]}</figcaption>'
            f'<a href="{roi["png"]}"><img loading="lazy" src="{roi["png"]}" '
            f'alt="{esc(roi["name"])}の原寸切り出し"></a></figure>' for roi in row['rois'])
        cards.append(f'<article id="{row["id"]}"><h2>{row["id"]}</h2>'
                     f'<p>{esc(" / ".join(row["tags"]))}</p><p>{esc(row["observation"])}</p>'
                     f'<p>{row["split"]} · 冊 {row["book"]} · 原画像 {row["width"]}×{row["height"]} px</p>'
                     '<div class="images"><figure><figcaption>ページ全体（表示用縮小）</figcaption>'
                     f'<a href="{row["original_png"]}"><img loading="lazy" src="{row["preview"]}" '
                     f'alt="{row["id"]}のページ全体"></a></figure>{regions}</div></article>')
        report_rows.append(f'| {row["id"]} | {row["book"]} | {"・".join(row["tags"])} | {len(row["rois"])} |')
    toc = ' · '.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in selected)
    viewer = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>別冊の赤色筆記・朱印：入力画像の確認</title>
<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;line-height:1.7;color:#202326;background:#f3f1ed}a{color:#075ea0}header,article{background:white;border-radius:10px;padding:20px;margin-bottom:24px}h1{font-size:1.6rem}h2{font-size:1.2rem}.images{display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start}figure{margin:0;flex:1 1 300px;max-width:620px}img{width:100%;height:auto;display:block}figcaption{margin-bottom:10px}nav{overflow-wrap:anywhere}article{scroll-margin-top:16px}</style>
<header><h1>別冊の赤色筆記・朱印：入力画像の確認</h1>
<p>8冊・9ページ・14か所。直近12ページの調整に使った冊を除外し、33冊・4,628ページから色で候補を探し、100候補と選定箇所を目視した。</p>
<p>方法を適用する前の入力確認。学習・画素注釈・画像修復は実施していない。切り出しは観察箇所で、画素の正解ラベルではない。確認者はCodexで、利用者の確認は未実施。</p>
<p>画像をクリックすると元解像度のPNGを開く。PNGは元JPEGを復号した画素と同一。全体のプレビューのみ縮小している。</p>
<p>朱線、薄い印影、細い印影、墨との重なり、赤褐色筆記を含む。暗い朱の例はまだ不足。色から集めた例のため、資料全体の精度や出現頻度は測れない。</p>
<p><a href="report.md">所見と次の確認</a> · <a href="selection.json">選定・原画像・固定条件の記録</a> · <a href="verification.json">保存内容の検証</a></p><nav>'''
    (OUT / 'index.html').write_text(viewer + toc + '</nav></header>' + ''.join(cards) + '</html>\n')
    report = '''# 赤色筆記・朱印の入力コレクション（2026-09-30）

直近の開発・回帰確認12ページで使った冊を除き、**8冊・9ページ・14か所**を確保した。今回は入力収集までで、既存方式・改良方式を新しい9ページへ適用していない。

## 選定と記録

- 原JPEG（train/val/test）のうち33冊・4,628ページを、4分の1復号画像のHSV・RGB条件で候補探索した。色の条件は `screening_manifest.json`、全ページのスコアは `screening_scores.json` に保存した。
- 100候補のサムネイルを見た後、選定した14か所を原寸で確認した。方法の出力を見て選んでいない。確認者はCodex画像閲覧であり、利用者による確認は未実施。
- 検出スコアの高い茶色い表紙・撮影背景・管理ラベルのみの例は今回の陽性例に採用していない。これらを赤い画素の正解として扱うこともしていない。
- 冊はtrain由来7冊、val由来1冊。既存ディレクトリの名前を引き継いだもので、学習はしていない。test由来の陽性ページは今回含まれず、分割間の釣り合いはない。
- 原画像のSHA256、元寸法、確認用矩形、復号した元画素のPNGを保存した。矩形は観察位置の指定であり、画素注釈・インクマスクのGTではない。
- 赤色の観察は色・図形・筆画に基づく。朱の化学組成、印影の歴史、退色の原因は判定していない。

## 確保した例

| ページ | 冊 | 観察した条件 | 確認箇所数 |
|---|---|---|---:|
'''
    report += '\n'.join(report_rows) + '''

`200020019` は同冊の2ページで、明瞭な印影と薄い印影を対比する。9ページを9冊として数えず、将来の集計も冊を単位にまとめる。`100249371` の赤色の図形・文字は朱印か色刷りか未確定だが、保護対象の赤色内容として残した。

## 限界

色を使った目的抽出なので代表的な無作為標本ではなく、薄い赤・暗い赤が候補探索から漏れる可能性がある。赤褐色筆記に暗い部分はあるが、**本当に暗い朱の条件が十分に揃ったとは言えない**。この9ページだけで朱の再現率・全体精度・原論文への優越を主張できない。筆記の例は2冊、残りは主に印影や赤色図形であり、筆記の種類の広さにも限界がある。

このコレクションは直近12ページの調整用冊と重複しないが、プロジェクト全体で過去に一度も閲覧・利用されていない資料だとは主張しない。今後この9ページに合わせて条件を変更する場合は開発資料に移し、別冊の最終確認用資料を残す。

## 次にする比較

1. `training_free_protection_v2_20260930/manifest.json` の条件を変更せず適用する。前段の局所・色・形処理も対応する既存条件を固定する。コード・設定・入力のハッシュは `selection.json` に記録済み。
2. 凍結したローカルnpj v1（CPU ACE条件も固定）、局所しきい値、色・形の前版、保護・保留を分けた改良版を同じ原画像・同じ14矩形で比較する。これはローカル実装との比較であり、原論文の完全再現ではない。
3. 赤が確定朱・色未確定の保護・保留・編集許可のどこへ入るかを目視する。赤ラベルから外れても保護される場合と、赤の画素が編集許可に入る場合を分ける。薄い部分・墨との交差・青線との交差も確認する。
4. 未注釈なのでマスク面積・候補の差分・具体的な見落とし／誤保護として報告し、Precision/Recallとは呼ばない。まず固定条件の成功と失敗を確認してから、追加例や方式修正を決める。

現在は方法適用・学習・拡散推論・画素削除を実施していない。手動の画素GT作成も見送っている。

## 保存確認

`verification.json` に元JPEGのハッシュ、復号画素とPNGの一致、14切り出しの座標・画素一致、調整冊との非重複、既存の固定コード・設定の同一性、HTML参照ファイルの存在を記録する。ブラウザでの実表示は未検証。ビューアはネット接続・JavaScript不要の静的HTML。
'''
    (OUT / 'report.md').write_text(report)
    products = [OUT / 'selection.json', OUT / 'index.html', OUT / 'report.md',
                OUT / 'packaging_source_snapshot.py'] + sorted((OUT / 'pages').rglob('*.*'))
    dump(OUT / 'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in products}))
    dump(OUT / 'verification.json', dict(status='pending'))
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        verify()
    else:
        build()
