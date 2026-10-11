"""Render and audit role-separated local postprocessing of real saved npj."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_npj_v6_comparison import ROOT, FONT_PATH, read, write, sha, dump
from package_saved_npj_inference_v6_review import check

PILOT = ROOT/'tmp_work/npj_role_separated_postprocess_pilot_20261002'
OLD = ROOT/'tmp_work/saved_npj_inference_v6_pilot_20261002'
OUT = ROOT/'tmp_work/npj_role_separated_postprocess_review_20261002'
NOTES = ROOT/'tmp_work/npj_role_separated_postprocess_notes_20261002.md'


def figure(images, labels, path, title, footers, overview=False):
    h, w = images[0].shape[:2]
    assert all(a.shape == (h, w, 3) for a in images)
    if overview:
        ratio = min(540/w, 900/h, 1)
        dw, dh = round(w*ratio), round(h*ratio)
        panels = [cv2.resize(a, (dw, dh), interpolation=cv2.INTER_AREA) for a in images]
        scale_note = '全体像は閲覧用縮小。拡大は元画素の整数反復。'
    else:
        scale = min(12, max(1, min(680//w, 1000//h)))
        dw, dh = w*scale, h*scale
        panels = [np.repeat(np.repeat(a, scale, axis=0), scale, axis=1) for a in images]
        scale_note = f'同位置を{scale}倍・最近傍表示。色補正なし。'
    margin, gap, top = 24, 24, 120
    canvas = Image.new('RGB', (len(images)*dw+(len(images)-1)*gap+2*margin, dh+270), '#faf8f2')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 25); small = ImageFont.truetype(FONT_PATH, 20)
    draw.text((margin, 12), title, font=font, fill='black')
    draw.text((margin, 49), scale_note, font=small, fill='black')
    for i, (a, label) in enumerate(zip(panels, labels)):
        x = margin+i*(dw+gap)
        draw.text((x, 85), label, font=small, fill='black')
        canvas.paste(Image.fromarray(a[:, :, ::-1]), (x, top))
    for i, note in enumerate(footers):
        draw.text((margin, dh+136+i*32), note, font=small, fill='black')
    canvas.save(path)
    saved = np.asarray(Image.open(path).convert('RGB'))
    for i, a in enumerate(panels):
        x = margin+i*(dw+gap)
        assert np.array_equal(saved[top:top+dh, x:x+dw], a[:, :, ::-1])
    return dict(panels_exact=True, integer_nearest=not overview, display_size=[dw, dh])


def main(render_only):
    assert not (OUT/'complete.json').exists(), 'Review already frozen'
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = check(PILOT); source = check(OLD)
    frames = {}; records = []; gap_records = []
    labels = ['原画像', '保存npj', '紙面のみ補修', '紙面＋短い接続補完']
    keys = ['original', 'saved_npj', 'paper_only', 'combined']
    for spec in manifest['frames']:
        fid = spec['id']; native = PILOT/'native'/fid
        a = {k:read(native/(k+'.png')) for k in ['original', 'saved_npj', 'paper_only', 'stroke_only', 'combined']}
        m = {k:read(native/(k+'.png'), True) for k in ['paper_update', 'stroke_update', 'original_support', 'npj_anchors']}
        for k in ['original', 'saved_npj']:
            assert np.array_equal(a[k], read(OLD/'native'/fid/(k+'.png')))
        assert np.array_equal(a['saved_npj'], read(ROOT/spec['saved_npj']))
        assert not (m['paper_update'] & m['stroke_update']).any()
        combined = a['saved_npj'].copy()
        combined[m['paper_update']] = a['paper_only'][m['paper_update']]
        combined[m['stroke_update']] = a['stroke_only'][m['stroke_update']]
        assert np.array_equal(combined, a['combined'])
        assert np.array_equal(a['paper_only'][~m['paper_update']], a['saved_npj'][~m['paper_update']])
        assert np.array_equal(a['stroke_only'][~m['stroke_update']], a['saved_npj'][~m['stroke_update']])
        for k in ['red', 'extra', 'hold']:
            keep = read(OLD/'native'/fid/(k+'.png'), True)
            assert np.array_equal(combined[keep], a['saved_npj'][keep])
        frames[fid] = (a, m)
        figure([a[k] for k in keys], labels, OUT/('overview_'+fid+'.png'),
               spec['page']+'：保存npjを維持する局所補修',
               ['保存npjを再推論せず使用。新候補は互いに重ならない紙面／短い接続部だけを更新。',
                '紙面参照は保存npj、接続補完は原画像の観測濃淡。原画像全体へ戻さない。',
                '同じ開発2ページ。人の筆画GT・精度・全文書の優越は未確認。'], overview=True)
        diagnostics = json.loads((native/'diagnostics.json').read_text())
        num, weak_labels = cv2.connectedComponents(
            read(native/'weak_candidate.png', True).astype(np.uint8), connectivity=8)
        _, anchor_labels = cv2.connectedComponents(m['npj_anchors'].astype(np.uint8), connectivity=8)
        N = cv2.cvtColor(a['saved_npj'].astype(np.float32)/255., cv2.COLOR_BGR2LAB)[:, :, 0]
        afterL = cv2.cvtColor(a['stroke_only'].astype(np.float32)/255., cv2.COLOR_BGR2LAB)[:, :, 0]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        fixed_paper = cv2.morphologyEx(N, cv2.MORPH_CLOSE, kernel)
        after_anchors = m['original_support'] & ((fixed_paper-afterL) >= 12)
        _, after_labels = cv2.connectedComponents(after_anchors.astype(np.uint8), connectivity=8)
        H, W = N.shape
        for c in diagnostics['components']:
            if not c['accepted']: continue
            gx, gy, gw, gh = c['xywh']
            component = weak_labels == c['id']
            assert int(component.sum()) == c['pixels']
            assert m['stroke_update'][component].all()
            near = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            contacts = near & np.isin(anchor_labels, c['anchor_components'])
            after_ids = np.unique(after_labels[contacts])
            connected = len(after_ids) == 1 and after_ids[0] > 0
            x, y = max(0, gx-22), max(0, gy-22)
            x2, y2 = min(W, gx+gw+22), min(H, gy+gh+22)
            s = np.s_[y:y2, x:x2]
            ordinal = len(gap_records)+1
            filename = f'gap_{ordinal:02d}_{fid}_{gx}_{gy}.png'
            rendering = figure([a[k][s] for k in ['original', 'saved_npj', 'stroke_only', 'combined']],
                               ['原画像', '保存npj', '短い接続補完のみ', '紙面＋短い接続補完'],
                               OUT/filename, f'自動採択 {ordinal:02d} / {spec["page"]} / 候補は筆画GTではない',
                               [f'この成分の補完 {c["pixels"]}画素。観測支持と別々の既存筆画端を条件に選択。',
                                f'固定局所コントラスト12の接続判定：{"つながる" if connected else "未接続"}。読解や材質の正解ではない。',
                                '全採択候補を提示。失敗・曖昧な候補も除外しない。'])
            ox, oy, _, _ = spec['frame_xywh']
            gap_records.append(dict(frame=fid, component=c, filename=filename,
                                    context_frame_xywh=[x, y, x2-x, y2-y],
                                    page_gap_xywh=[ox+gx, oy+gy, gw, gh],
                                    fixed_contrast_contacts_connected=bool(connected), **rendering))
    for spec in manifest['regions']:
        a, m = frames[spec['frame']]
        x, y, w, h = spec['xywh']; s = np.s_[y:y+h, x:x+w]
        base = a['saved_npj'][s]
        changed = {k:int(np.any(a[k][s] != base, axis=2).sum())
                   for k in ['paper_only', 'stroke_only', 'combined']}
        counts = {k:int(v[s].sum()) for k, v in m.items()}
        rendering = figure([a[k][s] for k in keys], labels, OUT/(spec['id']+'.png'), spec['title'],
                           ['基準は実際の保存npj。文字の黒さを保ち、紙面側と短い接続部の更新を分ける。',
                            f'このROIの紙面更新 {counts["paper_update"]:,}／接続補完 {counts["stroke_update"]:,}画素。',
                            '更新数は筆画GTや精度ではない。紙面のみ列と併用列で文字側の寄与を確認する。'])
        for k, v in a.items(): write(OUT/'native'/spec['id']/(k+'.png'), v[s])
        for k, v in m.items(): write(OUT/'native'/spec['id']/(k+'.png'), v[s])
        records.append(dict(**spec, changed_pixels=changed, mask_pixels=counts, **rendering))
    expected_figures = 2+len(records)+len(gap_records)
    assert len(list(OUT.glob('*.png'))) == expected_figures
    dump(OUT/'diagnostics.json', dict(regions=records, all_accepted_gaps=gap_records,
         fixed_contrast_connected=sum(r['fixed_contrast_contacts_connected'] for r in gap_records),
         gap_count=len(gap_records), figure_count=expected_figures))
    if render_only:
        print(f'Rendered {expected_figures} draft PNGs, including every accepted gap.', flush=True)
        return
    assert NOTES.exists(), 'Write visual findings before freezing'
    report = '''# 保存npjを維持し、紙面補修と短い接続補完を分ける

2026-10-02。前回は原画像を保つv6編集候補と実保存npjを比較し、npjの主要筆画と背景整理が良好な例を確認した。今回は実保存npjを土台とし、更新範囲の外をそのまま残す局所後処理を試した。元npjの再設計・再推論も、拡散モデルの再実行もない。

## 方式と評価範囲

紙面側：固定v6の編集候補から、protect・hold・朱・追加暖色保留の周囲2pxと入力外縁1pxを除外。保存npjの直近の周囲画素から4近傍Dirichlet調和補間を行う。原画像の紙色・模様へ戻す処理ではない。正常紙面を文字以外という理由で更新しない。

文字側：元のprotectのうち朱・追加暖色保留・hold・editを除外し、原画像に局所Lコントラスト8以上の観測支持がある範囲を使う。保存npjで局所コントラスト12以上の筆画端を求め、保存npjが原画像よりLで8を超えて薄い、小さな成分を候補とする。2〜64画素、最長辺12px以下、異なる8画素以上の端成分2つ以上へ8近傍で直接接触、入力外縁8pxは除外。採択した部分だけ、原画像／17px形態学的closingで推定した原紙面のBGR比を保存npjへ掛ける。黒い線を新しく描く処理ではなく、観測濃淡の保守的な補完。BGR比は光学的なインク濃度の推定や学習済み文字生成ではない。

文字用支持と紙面更新は排他的。朱・保留・追加暖色保留、および更新外は原画像ではなく**保存npjと完全一致**する。保護を全部原画像へ戻さず、保存npjで得られた明瞭さを保つ。前回の新候補とのノイズ一致比較や、v6固有要素の効果の証明ではない。

2ページは既知の開発例。設計前の原画像・保存npj・マスク観察を踏まえた閾値で、独立評価ではない。推定は保存入力全体で行い、8ROIは表示だけに使う。最初のページは550×500保存切り出し、2ページ目は元のページ全体。旧黒マスクは保持すべき筆画のGTではない。自動採択した接続候補を全件提示し、改善した候補だけを選ばない。

## 目視所見

'''+NOTES.read_text()+'''

## 固定範囲の比較PNG

'''
    for r in records:
        report += f'- [{r["title"]}]({r["id"]}.png)：紙面更新 {r["mask_pixels"]["paper_update"]:,}／接続補完 {r["mask_pixels"]["stroke_update"]:,}画素。\n'
    report += '\n## 自動採択した接続候補の全件\n\n'
    for r in gap_records:
        c = r['component']
        report += f'- [{r["frame"]}・{c["xywh"]}]({r["filename"]})：{c["pixels"]}画素。固定コントラスト上の接続={r["fixed_contrast_contacts_connected"]}。\n'
    report += f'''
固定した局所コントラストによる接続は {sum(r['fixed_contrast_contacts_connected'] for r in gap_records)}/{len(gap_records)} 候補。これは読解・筆画GT・正しい文字接続の率ではない。

## 検証

保存出力・マスク・入力と依存／出力ハッシュを照合。線形の紙面を欠損から正確に回復し、未知画素の異色置換でも紙面解が不変であることを確認。実入力でも更新内を異色へ置換して紙面結果一致。独立の合成再構成、各処理の更新外と朱／保留／追加暖色保留の保存npj一致、パネルの全画素一致、ZIP全PNGのCRC／バイトを確認。候補全件、全8ROI、全体2画像を含む。人の筆画注釈と別冊陽性評価は未実施、採用は保留。

[全{expected_figures}PNG ZIP](comparison_pngs.zip)、[検証](verification.json)、[上部全体](overview_upper_crop.png)、[本文全体](overview_full_page.png)。
'''
    (OUT/'report.md').write_text(report)
    dependencies = {str((PILOT/name).relative_to(ROOT)):sha(PILOT/name)
                    for name in ['manifest.json', 'complete.json', 'verification.json', 'source_snapshot.py']}
    for p in [Path(__file__), NOTES, Path(__file__).parent/'build_npj_v6_comparison.py',
              Path(__file__).parent/'package_saved_npj_inference_v6_review.py']:
        dependencies[str(p.relative_to(ROOT))] = sha(p)
    dump(OUT/'manifest.json', dict(dependencies=dependencies, source_sha256=sha(Path(__file__)),
         primary_artifact='PNG', all_accepted_gap_candidates_shown=True, pixel_GT=False))
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    dump(OUT/'verification.json', dict(status='passed', regions=records, gaps=gap_records,
         real_saved_npj_exact=True, independent_composite_exact=True,
         each_role_outside_update_exact=True, red_hold_extra_exact_saved_npj=True,
         panels_exact=True, all_accepted_gap_candidates_shown=True,
         pixel_GT=False, adopted=False))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')): z.write(p, p.name)
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert z.testzip() is None and len(z.namelist()) == expected_figures
        for name in z.namelist(): assert z.read(name) == (OUT/name).read_bytes()
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)):sha(p)
         for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print(f'Packaged {expected_figures} PNGs, all candidates and controls.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--render-only', action='store_true')
    main(parser.parse_args().render_only)
