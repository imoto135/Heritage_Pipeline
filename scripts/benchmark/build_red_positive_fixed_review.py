"""Package and verify native crops from the fixed red-positive comparison.

Counts describe masks, not accuracy. Visual observations may be added in
visual_findings.json without changing or rerunning classification conditions.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/red_positive_fixed_check_20260930'
COLLECTION = ROOT / 'tmp_work/red_positive_collection_20260930'
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')

import cv2
import numpy as np
cv2.setNumThreads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def read(path, mask=False):
    v = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
    if v is None:
        raise OSError(path)
    return v > 0 if mask else v


def write(path, v):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), v)


def overlay(original, mask, color):
    value = original.copy()
    value[mask] = np.round(.45 * original[mask] + .55 * np.array(color)).astype(np.uint8)
    return value


def get_masks(row):
    stem = row['id']
    refined = OUT / 'refined' / stem
    masks = {p.stem: read(p, True) for p in refined.glob('*.png')}
    for mode in ['v1_no_ace', 'v1_ace']:
        for key in ['black', 'red', 'stain', 'background']:
            masks[mode + '_' + key] = read(OUT / 'baseline' / stem / mode / (key + '.png'), True)
        masks[mode + '_ink'] = masks[mode + '_black'] | masks[mode + '_red']
    masks['local'] = read(OUT / 'pilot' / stem / 'sauvola_local/foreground.png', True)
    return masks


def mask_counts(masks, rect):
    x, y, w, h = rect
    m = {key: value[y:y+h, x:x+w] for key, value in masks.items()}
    return dict(xywh=rect, **{key + '_pixels': int(v.sum()) for key, v in m.items()},
        old_red_protected=int((m['legacy_red'] & m['protect']).sum()),
        old_red_unprotected_held=int((m['legacy_red'] & ~m['protect'] & m['hold']).sum()),
        old_red_edit_allowed=int((m['legacy_red'] & m['edit_allowed']).sum()),
        old_ink_no_longer_protected=int((m['legacy_ink'] & ~m['protect']).sum()),
        added_protection=int((m['protect'] & ~m['legacy_ink']).sum()),
        protection_hold_overlap=int((m['protect'] & m['hold']).sum()),
        stain_protection_overlap=int((m['stain_candidate'] & m['protect']).sum()))


def verify():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert sha(ROOT / 'scripts/benchmark/check_red_positive_fixed.py') == manifest['source_sha256']
    for relative, expected in manifest['dependencies'].items():
        assert sha(ROOT / relative) == expected, relative
    complete = json.loads((OUT / 'complete.json').read_text())
    for rel, expected in complete['outputs'].items():
        assert sha(OUT / rel) == expected, rel
    assert sha(Path(__file__)) == complete['review_source_sha256']
    checks = []
    for row in manifest['cases']:
        assert sha(ROOT / row['original']) == row['original_sha256']
        original = read(ROOT / row['original'])
        assert original.shape[:2] == (row['height'], row['width'])
        assert np.array_equal(original, read(OUT / 'pilot' / row['id'] / 'original.png'))
        assert np.array_equal(original, read(OUT / 'color' / row['id'] / 'original.png'))
        masks = get_masks(row)
        for key, mask in masks.items():
            assert mask.shape == original.shape[:2], key
        for path in (OUT / 'refined' / row['id']).glob('*.png'):
            assert set(np.unique(cv2.imread(str(path), 0))).issubset({0, 255}), path
        for excluded in ['protect', 'hold', 'protection_guard', 'legacy_red']:
            assert not (masks['edit_allowed'] & masks[excluded]).any(), excluded
        assert np.array_equal(masks['protect'], masks['black'] | masks['red'] | masks['ink_color_uncertain'])
        assert not (masks['red'] & masks['black']).any()
        assert not (masks['red'] & masks['ink_color_uncertain']).any()
        for mode in ['v1_no_ace', 'v1_ace']:
            count = sum(masks[mode + '_' + key].astype(np.uint8) for key in ['black', 'red', 'stain', 'background'])
            assert np.all(count == 1)
        for i, roi in enumerate(row['rois'], 1):
            x, y, w, h = roi['xywh']
            assert np.array_equal(original[y:y+h, x:x+w], read(OUT / 'review' / row['id'] / f'roi_{i:02d}_original.png'))
        checks.append(row['id'])
    parity = json.loads((OUT / 'refinement_adapter_parity.json').read_text())
    assert parity['status'] == 'passed' and len(parity['cases']) == 12
    assert sha(OUT / 'refinement_adapter_snapshot.py') == parity['adapted_source_sha256']
    selected = json.loads((COLLECTION / 'selection.json').read_text())
    assert not {r['book'] for r in manifest['cases']} & set(selected['excluded_development_book_ids'])
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ('src', 'href') and value and not value.startswith('#'):
                    assert (OUT / value).is_file(), value
    Links().feed((OUT / 'index.html').read_text())
    subprocess.run(['node', '--check', str(OUT / 'viewer.js')], check=True, cwd=ROOT)
    result = dict(status='passed', pages=len(checks), books=len({r['book'] for r in manifest['cases']}),
        fixed_rois=sum(len(r['rois']) for r in manifest['cases']), output_hashes_checked=len(complete['outputs']),
        original_decoded_pixels_unchanged=True, crops_equal_native_source=True,
        edit_excludes_protect_hold_guard_and_old_red=True, independent_mask_semantics_checked=True,
        adapter_parity_pages=12, adapter_parity_masks=17 * 12, development_books_disjoint=True,
        color_parameters_equal_frozen=json.loads((OUT / 'color/manifest.json').read_text())['parameters'] == manifest['color_settings'],
        threshold_tuning=False, neural_training=False, manual_pixel_GT=False, restoration=False,
        html_local_links_checked=True, viewer_javascript_syntax_checked=True, browser_rendering_tested=False)
    assert result['color_parameters_equal_frozen']
    dump(OUT / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    manifest = json.loads((OUT / 'manifest.json').read_text())
    modes = dict(original='原画像', ace='CPU ACE後', v1_no_ace='v1 CSC・ACEなし',
                 v1_ace='v1 CSC・ACEあり', local='局所Sauvola：前景', legacy_red='前版：朱候補',
                 red='改良：確定朱候補', protect='改良：文字保護', hold='改良：保留（保護と重なる）',
                 edit_allowed='改良：編集許可', stain_candidate='改良：汚れ候補（保護と重なる）',
                 ink_color_uncertain='改良：保護・色未確定')
    colors = dict(protect=(225,100,38), local=(225,100,38), red=(69,52,228), legacy_red=(69,52,228),
                  hold=(176,65,155), edit_allowed=(23,178,235), stain_candidate=(23,178,235),
                  ink_color_uncertain=(180,165,0))
    plots = ['original', 'v1_ace', 'local', 'legacy_red', 'red', 'protect', 'hold', 'edit_allowed']
    findings_path = OUT / 'visual_findings.json'
    findings = json.loads(findings_path.read_text()) if findings_path.exists() else {'cases': {}}
    conclusion = findings.get('conclusion', '')
    next_steps = findings.get('next_steps', '')
    data, metrics, cards = [], [], []
    esc = html.escape
    for row in manifest['cases']:
        stem = row['id']; original = read(ROOT / row['original']); masks = get_masks(row)
        images = dict(original=original, ace=read(OUT / 'baseline' / stem / 'ace.png'))
        for mode in ['v1_no_ace', 'v1_ace']:
            value = original.copy()
            for key, color in [('stain', (23,178,235)), ('black', (225,100,38)), ('red', (69,52,228))]:
                mask = masks[mode + '_' + key]
                value[mask] = np.round(.45 * original[mask] + .55 * np.array(color)).astype(np.uint8)
            images[mode] = value
        images.update({key: overlay(original, masks[key], color) for key, color in colors.items()})
        page_metrics = dict(page=stem, book=row['book'], roi=None, **mask_counts(masks, [0,0,row['width'],row['height']]))
        metrics.append(page_metrics)
        page_findings = findings.get('cases', {}).get(stem, '')
        observations = f'<p class="finding">{esc(page_findings)}</p>' if page_findings else ''
        parts = []
        case_data = dict(id=stem, rois=[])
        for i, roi in enumerate(row['rois'], 1):
            x, y, w, h = roi['xywh']; sl = (slice(y,y+h), slice(x,x+w))
            paths = {}
            for mode, image in images.items():
                rel = f'review/{stem}/roi_{i:02d}_{mode}.png'
                write(OUT / rel, image[sl]); paths[mode] = rel
            roi_stats = dict(page=stem, book=row['book'], roi=i, title=roi['name'], **mask_counts(masks, roi['xywh']))
            metrics.append(roi_stats)
            fig, axes = plt.subplots(2,4,figsize=(16,2*max(2.5,min(6.,4.*h/w+.65))))
            for ax, mode in zip(axes.flat, plots):
                ax.imshow(cv2.cvtColor(images[mode][sl],cv2.COLOR_BGR2RGB),interpolation='nearest')
                ax.set_title(modes[mode],fontproperties=font,fontsize=11); ax.axis('off')
            fig.suptitle(f'{stem} · {roi["name"]}（未注釈：正解マスクではない）',fontproperties=font,fontsize=13)
            fig.tight_layout(rect=(0,0,1,.95))
            plot_rel = f'review/{stem}/roi_{i:02d}_comparison.png'
            fig.savefig(OUT / plot_rel,dpi=145,facecolor='white'); plt.close(fig)
            panels = ''.join(f'<figure data-mode="{mode}"><figcaption>{esc(modes[mode])}</figcaption>'
                f'<a href="{path}"><img loading="lazy" src="{path}" alt="{esc(modes[mode])}"></a></figure>'
                for mode,path in paths.items())
            parts.append(f'<section><h3>{esc(roi["name"])} · {roi["xywh"]}</h3>'
                f'<p>前版朱候補 {roi_stats["legacy_red_pixels"]:,} → 確定朱候補 {roi_stats["red_pixels"]:,}画素。'
                f'前版朱候補のうち保護 {roi_stats["old_red_protected"]:,}、保護外の保留 {roi_stats["old_red_unprotected_held"]:,}。'
                'この数は候補の移動で、精度・再現率ではない。</p>'
                f'<p><a href="{plot_rel}">8表示の比較図を開く</a></p><div class="grid">{panels}</div></section>')
            case_data['rois'].append(dict(name=roi['name'],xywh=roi['xywh'],images=paths,comparison=plot_rel))
        # Full-resolution masks stay separate from overlays and plot display.
        mask_links = ' · '.join(f'<a href="refined/{stem}/{key}.png">{esc(modes[key])}の白黒マスク</a>'
                              for key in ['red','protect','hold','edit_allowed','stain_candidate'])
        cards.append(f'<article id="{stem}"><h2>{stem}</h2><p>{esc(" / ".join(row["tags"]))}</p>'
                     f'<p>{esc(row["observation"])}</p>{observations}<details><summary>ページ全体のマスク</summary>'
                     f'<p>{mask_links}</p></details>{"".join(parts)}</article>')
        data.append(case_data)
    dump(OUT / 'diagnostics.json', metrics)
    dump(OUT / 'gallery.json', dict(cases=data,modes=modes))
    # Book grouping keeps two pages of the same book together; still not independent GT scores.
    page_rows = [m for m in metrics if m['roi'] is None]
    books = []
    for book in sorted({m['book'] for m in page_rows}):
        rows = [m for m in page_rows if m['book'] == book]
        keys = [k for k,v in rows[0].items() if isinstance(v,int)]
        books.append(dict(book=book,pages=len(rows),**{k:sum(r[k] for r in rows) for k in keys}))
    dump(OUT / 'book_diagnostics.json', books)
    controls = ''.join(f'<label><input type="checkbox" data-toggle="{mode}" '
                      f'{"checked" if mode in plots else ""}>{esc(label)}</label>' for mode,label in modes.items())
    toc = ' · '.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in manifest['cases'])
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>別冊の赤色筆記・朱印：固定条件の比較</title>
<style>body{font-family:system-ui,sans-serif;margin:20px auto;max-width:1800px;padding:0 16px;background:#f3f1ed;color:#202326;line-height:1.65}header,article,.controls{background:white;padding:20px;border-radius:10px;margin-bottom:22px}h1{font-size:1.6rem}h2{font-size:1.3rem}h3{font-size:1.1rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}figure{margin:0}figure img{width:100%;height:auto}figcaption{margin:8px 0}section{padding:12px 0 24px;border-top:1px solid #ccc}a{color:#075ea0}.controls label{display:inline-block;margin:6px 18px 6px 0}.finding{background:#f4f4ff;padding:12px}nav{overflow-wrap:anywhere}article{scroll-margin-top:20px}figure[hidden]{display:none}body.native .grid{display:flex;overflow:auto;flex-wrap:nowrap;align-items:flex-start}body.native figure{flex:none}body.native figure img{width:auto;max-width:none}.native-note{font-size:.9rem}</style>
<header><h1>別冊の赤色筆記・朱印：固定条件の比較</h1><p>8冊・9ページ・入力だけで選んだ14か所。同じ原寸入力で、ローカルnpj v1、局所Sauvola、前版の色・形、保護・保留を分けた改良版を比較した。条件の変更・学習・画素GTの作成・修復は実施していない。</p>
<p>青＝文字保護・前景、赤＝朱候補、紫＝保留、黄＝汚れ候補／編集許可、青緑＝保護・色未確定。汚れ候補・保護・保留は重なり得る独立マスク。編集許可は保護・保留・3px保護ガードを除外する。ACE後以外は同じ原RGBに候補を重ねている。</p>
<p>色で集めた陽性例であり、面積の変化を精度と呼ばない。v1はローカル凍結実装との比較で、原論文の完全再現ではない。画像をクリックすると元解像度の切り出しを開く。</p>
<p><a href="report.md">所見と次の方針</a> · <a href="diagnostics.json">ページ・矩形の候補数</a> · <a href="verification.json">検証記録</a> · <a href="../red_positive_collection_20260930/index.html">適用前の入力一覧</a></p><nav>'''
    js = '''const toggles=[...document.querySelectorAll('[data-toggle]')];function update(){for(const cb of toggles){for(const fig of document.querySelectorAll('figure[data-mode="'+cb.dataset.toggle+'"]'))fig.hidden=!cb.checked;}}toggles.forEach(cb=>cb.addEventListener('change',update));document.querySelector('#native').addEventListener('change',e=>document.body.classList.toggle('native',e.target.checked));update();'''
    (OUT / 'viewer.js').write_text(js + '\n')
    headline = '<p class="finding">' + esc(conclusion) + '</p>' if conclusion else ''
    (OUT / 'index.html').write_text(page + toc + '</nav>' + headline + '</header><div class="controls">' + controls +
        '<p><label><input id="native" type="checkbox">原寸表示（横スクロール）</label></p></div>' + ''.join(cards) + '<script src="viewer.js"></script></html>\n')
    summary = {k:sum(r[k] for r in page_rows) for k in ['legacy_red_pixels','red_pixels','old_red_protected','old_red_unprotected_held','old_red_edit_allowed']}
    dump(OUT / 'summary.json', summary)
    rows = '\n'.join(f'| {m["page"]} | {m["legacy_red_pixels"]:,} | {m["red_pixels"]:,} | {m["old_red_protected"]:,} | {m["old_red_unprotected_held"]:,} | {m["edit_allowed_pixels"]:,} |' for m in page_rows)
    observed = '\n\n'.join(f'- **{stem}**：{text}' for stem,text in findings.get('cases',{}).items()) or '原寸比較図を作成済み。所見は画像閲覧後に追記する。'
    (OUT / 'report.md').write_text(f'''# 別冊の赤色筆記・朱印：固定条件の比較

{conclusion}

8冊・9ページ・14か所へ条件を変更せず適用した。学習・手動画素GT・拡散推論・削除／復元は行っていない。比較はローカル凍結npj v1（ACEなし／CPU ACEあり）、局所背景補正Sauvola、前版の色・形、改良版の独立した保護・汚れ候補・保留・編集許可で行った。

## 固定条件と計算の確認

原画像と14矩形は方法適用前の収集記録から引き継いだ。直近の調整用12ページの冊とは非重複だが、色で目的抽出した陽性資料であり、無作為な代表標本ではない。9ページのうち2ページは同じ冊 `200020019` で、冊数は8。暗い朱の十分な網羅もまだない。

局所Sauvolaは既存の入力準備関数をそのまま呼び、保存済みの既知1ページと局所前景・3尺度投票・正規化画像が一致した。色・形処理はハッシュを確認した既存スクリプトを変更せず使用した。改良版は凍結スクリプトのASTから計算部分を抽出し、ファイル読込だけを引数へ置き換えた。既知12ページ・各17マスク（計204）の画素一致を確認した。抽出コードと一致記録を保存している。

CPU ACEはalpha=5、200参照点、seed=0。CSCはseed=42、最大50,000標本、外周4%を強制背景とする以前の条件。CSCは各入力のPCA/GMMを推定する既存比較方式であり、教師データやニューラルモデルの学習は行っていない。原論文の完全再現が成立したとは主張しない。

## ページ全体の候補数

| ページ | 前版の朱候補 | 改良の確定朱候補 | 前版朱のうち保護 | 前版朱のうち保護外の保留 | 改良の編集許可 |
|---|---:|---:|---:|---:|---:|
{rows}

前版朱候補は合計 {summary['legacy_red_pixels']:,}、改良の確定朱候補は {summary['red_pixels']:,}。前版朱候補のうち {summary['old_red_protected']:,} は保護され、保護外の {summary['old_red_unprotected_held']:,} は保留となる。前版朱候補が編集許可に入る画素は {summary['old_red_edit_allowed']:,}。

これは**候補マスクの移動**で、朱のRecallや正しいインクの保存率ではない。前版の候補にも汚れや紙色が含まれ得る。前版朱を保留で編集対象から除外するのは設計上の性質であり、その数だけで元画像の全ての朱が守られたとは言えない。元画像の朱を前版も取りこぼした場合、編集許可との交差があり得る。ページ全体の値は撮影背景・紙の縁も含むため、14か所の目視結果を別に確認する。

## 原寸画像の所見

{observed}

## 失敗箇所の拡大と次の方針

[赤褐色筆記の下部](early_inspection/200021763_00002_2_lower_handwriting_detail.png)・[上部](early_inspection/200021763_00002_2_upper_handwriting_detail.png)・[薄い印影](early_inspection/200021086_00032_2_faint_stamp_detail.png)の拡大図で、同じ原RGBに保護・保留・編集許可を重ねた。これは固定14矩形内の拡大位置の指定で、追加の画素正解ラベルではない。

`failure_zoom_diagnostics.json` では拡大領域の編集許可が全て既存の `brown_field` に由来し、前版朱候補との交差は0だと確認した。弱い赤の色条件に入っても連結成分の種支持がなく前版朱にならない画素と、弱い色条件にも入らない画素の両方がある。したがって確定朱のしきい値だけを変えても、前版の候補域そのものの見落としは解消しない。これらの数は領域内のマスクの交差であり、真の朱画素数ではない。

{next_steps}

## 保存と検証

原寸マスク、各確認矩形の12表示、8表示の比較図、比較ビューアを保存した。`diagnostics.json` は全体と14矩形、`book_diagnostics.json` は同冊をまとめた候補数。保護と保留・汚れ候補の重なりは消していない。`edit_allowed` が保護・保留・ガード・前版朱候補と交差しないこと、原画素と切り出しの一致、コード・設定・入力・出力ハッシュの検証結果は `verification.json` に保存した。ブラウザ実表示は未検証で、JavaScriptの構文と参照ファイルを確認した。
''')
    shutil.copyfile(__file__, OUT / 'review_source_snapshot.py')
    dump(OUT / 'verification.json', dict(status='pending'))
    # Hash classification and review artifacts, excluding caches and self-referential completion records.
    products = [p for p in OUT.rglob('*') if p.is_file() and 'plot_cache' not in p.parts
                and p.name not in ['complete.json','verification.json']]
    dump(OUT / 'complete.json', dict(status='complete',pages=9,books=8,fixed_rois=14,
        review_source_sha256=sha(Path(__file__)), outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(products)}))
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify() if args.verify_only else build()
