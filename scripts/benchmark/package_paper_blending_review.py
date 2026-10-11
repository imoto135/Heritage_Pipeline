"""Verify fixed-input blending experiments and package PNG comparisons."""
import gc
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
import probe_adjacent_paper_boundary as adjacent

ROOT = adjacent.ROOT
FIRST = adjacent.previous.OUT
SECOND = adjacent.OUT
DDRM = adjacent.DDRM
MASKS = adjacent.MASKS
OUT = ROOT / 'tmp_work/paper_blending_review_20261001'
MODES = adjacent.MODES
HOLES = ['hole_edge_new', 'new_margin_hole']


def boundary_metric(image, edited, known):
    """Independently enumerate oriented inside/outside neighbor pairs."""
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    yy, xx = np.nonzero(edited)
    values = []
    h, w = edited.shape
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = yy + dy, xx + dx
        valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        iy, ix, oy, ox = yy[valid], xx[valid], ny[valid], nx[valid]
        valid = known[oy, ox]
        delta = lab[iy[valid], ix[valid]] - lab[oy[valid], ox[valid]]
        values.extend(np.linalg.norm(delta, axis=1).tolist())
    return len(values), float(np.median(values)) if values else None


def check_bundle(folder, manifest='manifest.json', complete='complete.json'):
    metadata = json.loads((folder / manifest).read_text())
    for rel, digest in metadata['dependencies'].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in json.loads((folder / complete).read_text())['outputs'].items():
        assert sha(folder / rel) == digest, rel
    if 'source_sha256' in metadata:
        snapshot = 'candidate_source_snapshot.py' if manifest == 'candidate_manifest.json' else 'source_snapshot.py'
        assert sha(folder / snapshot) == metadata['source_sha256']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in [FIRST, SECOND, DDRM]:
        check_bundle(folder)
    check_bundle(MASKS, 'candidate_manifest.json', 'candidate_complete.json')
    first_rows = {r['id']: r for r in json.loads((FIRST / 'diagnostics.json').read_text())}
    rows = json.loads((SECOND / 'diagnostics.json').read_text())
    lookup = {r['id']: r for r in rows}
    checked = []
    books, pages = set(), set()
    for source in adjacent.previous.margin_probe.SOURCES:
        for page in json.loads((source / 'baseline_review.json').read_text())['cases']:
            pages.add(page['page']); books.add(page['book'])
            original = read(ROOT / page['original'])
            h, w = original.shape[:2]
            edit = read(MASKS / 'candidate_pages' / page['page'] / 'dark_support/edit.png', True)
            margin = read(MASKS / 'candidate_pages' / page['page'] / 'features/top_margin.png', True)
            for roi in page['rois']:
                key = roi['id']; x, y, rw, rh = roi['xywh']
                x0, y0 = max(0, x - 64), max(0, y - 64)
                x1, y1 = min(w, x + rw + 64), min(h, y + rh + 64)
                local = (slice(y - y0, y - y0 + rh), slice(x - x0, x - x0 + rw))
                crop = original[y:y+rh, x:x+rw]
                base = read(DDRM / 'native' / key / 'candidate.png')
                evaluation = read(FIRST / 'native' / key / 'adjusted.png', True)
                ref = read(SECOND / 'native' / key / 'reference_context.png', True)
                assert ref.shape == (y1-y0, x1-x0)
                assert not np.any(ref & edit[y0:y1, x0:x1])
                for folder in [FIRST, SECOND]:
                    native = folder / 'native' / key
                    assert np.array_equal(read(native / 'original.png'), crop)
                    assert np.array_equal(read(native / 'base.png'), base)
                    assert np.array_equal(read(native / 'v6.png'), read(DDRM / 'native' / key / 'v6.png'))
                    local_edit = read(native / 'edit.png', True)
                    adjusted = read(native / 'adjusted.png', True)
                    assert np.array_equal(local_edit, edit[y:y+rh, x:x+rw])
                    assert np.array_equal(local_edit, read(DDRM / 'native' / key / 'candidate_edit.png', True))
                    assert not np.any(adjusted & ~(local_edit & margin[y:y+rh, x:x+rw]))
                    assert int(adjusted.sum()) == (first_rows if folder == FIRST else lookup)[key]['adjusted_pixels']
                    for name in ['edit', 'adjusted']:
                        raw = cv2.imread(str(native / (name + '.png')), cv2.IMREAD_GRAYSCALE)
                        assert set(np.unique(raw)) <= {0, 255}
                    for mode in MODES:
                        output = read(native / (mode + '.png'))
                        assert output.shape == crop.shape
                        assert not np.any(np.any(output != base, axis=2) & ~adjusted)
                        assert not np.any(np.any(output != crop, axis=2) & ~local_edit)
                seeds = read(SECOND / 'native' / key / 'boundary_seeds.png', True)
                assert not np.any(seeds & ~read(SECOND / 'native' / key / 'adjusted.png', True))
                variants = dict(base=base, plane=read(FIRST / 'native' / key / 'tone_boundary.png'))
                variants.update({m: read(SECOND / 'native' / key / (m + '.png')) for m in MODES})
                for mode, value in variants.items():
                    count, median = boundary_metric(value, evaluation, ref[local])
                    recorded = lookup[key]['diagnostics'][mode]
                    assert count == recorded['boundary_pairs']
                    if median is None:
                        assert recorded['median_known_paper_boundary_deltaLab'] is None
                    else:
                        assert abs(median - recorded['median_known_paper_boundary_deltaLab']) < 1e-6
                    assert int(np.any(value != base, axis=2).sum()) == recorded['changed_vs_base']
                checked.append(key)
            del original, edit, margin
            gc.collect()
    assert len(checked) == len(set(checked)) == 22
    assert len(pages) == 14 and len(books) == 11
    unchanged = [r['id'] for r in rows if r['adjusted_pixels'] == 0]
    assert len(unchanged) == 18
    stacks = [
        ('01_hole_blending.png', SECOND, 'comparisons', HOLES, '虫食い周辺の紙面への馴染み',
         ['同じマスク・同じ保存済みDDRM出力。隣接紙面による色と境界の補正。',
          '色差は小さくなるが、平坦な塗り跡・薄い輪郭は残る。右上のマスク外の損傷は未処理。']),
        ('02_tone_boundary_ablation.png', SECOND, 'ablation', HOLES, '紙色と境界を分けた比較',
         ['紙色のみ／境界のみ／両方を同じ出力に適用。両方が常に最良という結果ではない。']),
        ('03_reference_ablation.png', SECOND, 'reference_ablation', HOLES, '補正に使う紙面の選び方',
         ['初回の明るい紙の平面による補正も保存。直近の未編集紙面を参照する補正と比較。']),
        ('04_controls.png', SECOND, 'comparisons', ['new_frame_text', 'new_header_ink', 'new_mono_text'], '枠・見出し・本文の対照',
         ['この3箇所の補正画素は0。現在の余白試作の保存出力から変わらない。']),
        ('05_paper_and_stamp.png', SECOND, 'comparisons', ['brown_false_red_new', 'new_brown_stamp'], '広い紙面と印影の対照',
         ['上端の茶色い紙面は364画素のみ補正。広い色ムラ全体が直る結果ではない。',
          '印影の小さな点の消失は前回マスクによる変化。利用者レビューを受け、今回の阻害要因にはしない。']),
        ('06_boundary_only.png', SECOND, 'ablation', ['brown_false_red_new', 'new_brown_stamp'], '補正要素の別例',
         ['この2箇所は境界だけ・両方より、紙色だけの方が境界色差の診断値は小さい。']),
    ]
    for filename, folder, group, keys, title, notes in stacks:
        stack([folder / group / (key + '.png') for key in keys], OUT / filename, title, notes)
    items = [(OUT / entry[0], 'overview/' + entry[0]) for entry in stacks]
    for folder, group, prefix in [(SECOND, 'comparisons', 'adjacent/comparisons'),
                                  (SECOND, 'ablation', 'adjacent/ablation'),
                                  (SECOND, 'reference_ablation', 'reference_ablation'),
                                  (FIRST, 'ablation', 'first_plane/ablation')]:
        items.extend((folder / group / (r['id'] + '.png'), prefix + '/' + r['id'] + '.png') for r in rows)
    archive_path = OUT / 'paper_blending_comparison_images.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items:
            archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(items) == 94 and archive.testzip() is None
        for path, name in items:
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    value = lambda r, m: r['diagnostics'][m]['median_known_paper_boundary_deltaLab']
    metric_table = '\n'.join(
        f'| `{r["id"]}` | {r["diagnostics"]["base"]["boundary_pairs"]} | {value(r,"base"):.3f} | {value(r,"plane"):.3f} | {value(r,"tone"):.3f} | {value(r,"boundary"):.3f} | {value(r,"tone_boundary"):.3f} |'
        for r in rows if value(r, 'base') is not None)
    all_table = '\n'.join(
        f'| [{r["title"]}](../{SECOND.name}/comparisons/{r["id"]}.png) · `{r["id"]}` | {first_rows[r["id"]]["adjusted_pixels"]:,} | {r["adjusted_pixels"]:,} | {"紙色・境界の診断対象" if r["adjusted_pixels"] else "補正を加えず保存出力と完全一致"} |'
        for r in rows)
    report = f'''# 保存済みDDRMの紙色・境界補正

2026-10-01。**隣接する未編集紙面を参照すると、虫食い周辺の境界の色差を小さくできた。目視の改善は控えめで、平坦な塗り跡・薄い輪郭はまだ残る。** 利用者の「失敗例は特に問題なさそう」「まだ違和感が残るが前進している」というレビューを受け、現在の余白試作を基準に、背景への馴染みを改善した。

今回の比較は**DDRM使用後の合成後処理**である。前処理のマスク、DDRMの重み・保存済み出力を固定し、追加推論も再学習もしていない。前回と同じ14ページ・11冊・22箇所を全て扱った。npjやv6のマスク精度を新しく比較した実験ではない。前回のnpj→DDRM比較は[前回報告](../paper_margin_defect_review_20261001/report.md)に保存している。

## 比較PNG

- [虫食い2例の補正前後](01_hole_blending.png)：原画像、固定v6→DDRM、現在の余白試作、隣接紙面による紙色＋境界補正。
- [紙色のみ・境界のみ・両方](02_tone_boundary_ablation.png)：両方を加えることが常に最良ではない。
- [初回の紙面平面と隣接紙面の比較](03_reference_ablation.png)：改善しなかった初回条件も残した。
- [枠・見出し・本文の対照](04_controls.png)、[茶色い紙面と印影](05_paper_and_stamp.png)、[この2例の要素比較](06_boundary_only.png)。
- [全比較94枚ZIP](paper_blending_comparison_images.zip)：概要6、隣接紙面比較22、隣接補正の要素比較22、参照方法の比較22、初回の要素比較22。

## 補正の条件

対象は保存済み編集マスクと、前回の上余白候補の共通部分。面積64画素以上の連結成分だけを扱う。本文やマスク外へ補正範囲を広げない。原画像から参照するのは、編集マスク・残存インク保護を除き、局所紙色とのLab差で絞った未編集紙面。参照領域の判定は全ページから作り、補正は保存済みROIと周囲64pxを使用する。全ページの新しい復元画像を生成した実験ではない。

初回は、成分から3〜48px離れた明るい紙面を、暗部の4pxガードと勾配条件でさらに絞り、ロバストなLab平面を当てた。この参照では4箇所全てで境界の色差が増えた。明るく平滑な紙の参照が直近の黄み・陰影からずれることが一因と考えられるが、この説明は参照条件と出力からの推測である。

次に、同じLabの紙面条件を使いつつ、編集領域に直接隣接する未編集画素を参照した。1成分あたり16組以上の4近傍の境界対が得られる場合に補正する。紙色のみは境界対のLab差の中央値を成分全体へ加える。境界のみは、生成画素だけから得た低周波成分（σ12px）と隣接紙面との差を境界上で取り、σ8pxで平滑化して内側へ6pxの尺度で弱める。両方は紙色補正後に境界の残差を加える。補正量はL±6、a/b±4に制限し、新しい粒状模様は加えない。

原画像の編集対象画素を出力に混ぜず、既存の生成画素に色の差分を加える。参照領域を固定して原画像の編集対象を任意の色に置き換えても、穴2例と印影1例の全3方式の出力が完全一致した。これは補正量を計算する段階での不変確認であり、元の参照領域の判定まで損傷画素と独立だと証明するものではない。

## 境界の診断値

編集側の境界画素と、紙面条件を満たす隣接未編集画素とのLab距離の中央値。値が小さいほど、その境界での色の段差が小さい。ΔE2000・画素正解との精度・目視品質の評点ではない。全条件を**初回の対象マスクから作った同じ境界対**で測った。隣接方式の対象が少ない上端紙面も、評価対を選び直していない。

| ROI | 境界対 | 補正前 | 初回平面・両方 | 隣接紙色のみ | 隣接境界のみ | 隣接・両方 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{metric_table}

穴2例は境界のみで3.550→1.870、3.526→2.039。紙色＋境界は後者が2.086で、境界のみよりわずかに大きい。紙面上端と印影は紙色のみの値が最も小さい。4例から全資料の改善率を推定せず、両方を自動的に最良条件として採用しない。この診断値は補正が参照する境界と共通なので、独立した品質評価でもない。

## 全22箇所と残る問題

| ROI | 初回補正の対象画素 | 隣接方式の対象画素 | 今回の状態 |
| --- | ---: | ---: | --- |
{all_table}

穴の既存例は色の境目が少し弱くなるが、薄い縁と平坦な領域が残る。新しい冊の穴は左・下側の色が隣接紙面へ近づく一方、右上の濃い損傷は前回からマスク外にあり、そのまま残る。ここは色の補正だけでは消せない。茶色い紙面上端は大半の成分に十分な境界対がなく、初回15,079画素に対して今回は364画素だけを補正した。広い紙色ムラを解決した結果ではない。

印影の小さな点は前回マスクで既に消えている。この変化は利用者レビューで現在の研究継続の阻害要因とは扱わない判断になった。今回の補正でその点を回復したとも、印影保護を解決したとも数えない。ほか18箇所は補正対象0で、現在の試作出力と完全一致した。条件で対象外だった結果であり、未知資料の文字保存精度を示す数字ではない。

紙の質感については、Lの高周波成分（σ1.2px）の標準偏差も保存した。穴の既存例では初回の参照0.491に対して生成側0.628、新しい穴では参照0.205に対して生成側0.401だった。参照側は平滑な紙面に偏る条件なので、この差を真の紙の分散差と見なさない。少なくとも「平坦に見えるので一律にノイズを加える」という根拠は得られていない。粒の量だけでなく模様の空間的なつながりが残る診断課題である。

## 確認と次の段階

入力・マスク・重み・保存DDRM・ソース・PNG/診断値のハッシュ、22箇所の原寸画素、補正対象外の保存DDRMとの一致、編集対象外の原画素との一致、参照と編集マスクの非重複を確認した。境界対を別実装で列挙して全条件の診断値を再計算し、ZIPの94PNGも内容ハッシュで確認した。[確認JSON](verification.json)。

次の開発比較では、隣接紙面による境界補正を基準として残す。紙色のみ・境界のみ・両方の画像も保持する。残る違和感は、マスク外に残った細い損傷と、編集内部の紙模様の不連続を分けて検証する。前者のマスク範囲を変える比較と、後者の合成を変える比較は別条件にする。今回の22箇所は開発例であり、正式な精度評価・新手法の確立・全資料への適用判断はまだ行わない。

初回[条件と全診断](../{FIRST.name}/diagnostics.json)、隣接方式[条件](../{SECOND.name}/manifest.json)・[全診断](../{SECOND.name}/diagnostics.json)。実装：[初回紙面平面](../../scripts/benchmark/probe_paper_tone_boundary.py)、[隣接境界](../../scripts/benchmark/probe_adjacent_paper_boundary.py)、[検証・PNG梱包](../../scripts/benchmark/package_paper_blending_review.py)。[利用者レビューによる方針更新](../paper_margin_defect_user_review_20261001.md)。
'''
    (OUT / 'report.md').write_text(report)
    dump(OUT / 'verification.json', dict(status='passed', pages=len(pages), books=len(books),
        regions=checked, unchanged_regions=unchanged, same_mask_and_saved_DDRM=True,
        outside_adjusted_changed_pixels=0, outside_edit_changed_pixels=0,
        independent_boundary_pair_recalculation=True, reference_excludes_edit=True,
        original_damage_poison_check_source=[str(FIRST / 'verification.json'), str(SECOND / 'verification.json')],
        hash_dependencies_and_outputs=True, zip_PNG_count=len(items), zip_content_hashes=True))
    dump(OUT / 'manifest.json', dict(source_sha256=sha(Path(__file__)),
        dependencies={str((f / n).relative_to(ROOT)): sha(f / n)
                      for f in [FIRST, SECOND, DDRM] for n in ['manifest.json', 'complete.json']},
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.iterdir()
                 if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: 14 pages, 11 books, 22 ROIs; 94 PNGs packaged', flush=True)


if __name__ == '__main__':
    main()
