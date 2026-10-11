"""Fixed-ROI, native-pixel review of literature candidates against saved npj."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, write, sha, dump
from package_saved_npj_inference_v6_review import check
from package_saved_npj_role_separated_review import figure

TRIAL = ROOT/'tmp_work/npj_literature_trial_20261002'
OUT = ROOT/'tmp_work/npj_literature_trial_review_20261002'
NOTES = ROOT/'tmp_work/npj_literature_trial_notes_20261002.md'


def tint(image, mask):
    result = image.copy()
    result[mask] = np.rint(.55*image[mask]+.45*np.array([245,110,35])).astype(np.uint8)
    return result


def main(render_only):
    assert not (OUT/'complete.json').exists(), 'Frozen review'
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads((TRIAL/'selection.json').read_text())
    selected = json.loads((TRIAL/'selection_complete.json').read_text())
    assert sha(TRIAL/'selection.json') == selected['selection_sha']
    for rel, digest in selected['outputs'].items():
        assert sha(TRIAL/rel) == digest, rel
    for rel, digest in spec['dependencies'].items():
        assert sha(ROOT/rel) == digest, rel
    for method in ['stain', 'docres', 'structural', 'matched_stain', 'matched_docres']:
        check(TRIAL/method)
    records = []; inputs = []; rendering = []
    labels = ['参照原画像', '実保存npj', 'StainRestorer 生出力', 'DocRes 文字候補']
    for frame in spec['frames']:
        fid = frame['id']; native = TRIAL/'inputs'/fid
        a = {k:read(native/(k+'.png')) for k in ['original', 'saved_input', 'saved_npj']}
        a['stain'] = read(TRIAL/'stain'/fid/'output.png')
        a['docres'] = read(TRIAL/'docres'/fid/'output.png')
        assert np.array_equal(a['saved_npj'], read(ROOT/frame['saved_npj']))
        shape = a['original'].shape
        assert all(v.shape == shape for v in a.values())
        ink = read(TRIAL/'docres'/fid/'ink_candidate.png', True)
        probability = np.load(TRIAL/'docres'/fid/'background_probability.npy', allow_pickle=False)
        assert probability.shape == shape[:2] and np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()
        assert not (probability == .5).any(), 'No argmax tie versus threshold ambiguity'
        assert set(np.unique(a['docres'])).issubset({0, 255})
        assert np.array_equal(a['docres'][:,:,0] == 0, ink)
        assert np.array_equal(a['docres'][:,:,0], a['docres'][:,:,1])
        assert np.array_equal(ink, probability < .5), 'Independent softmax decode'
        m = {k:read(native/(k+'.png'), True) for k in ['protect', 'hold', 'red', 'extra']}
        m['structural'] = read(TRIAL/'structural'/fid/'ink_support.png', True)
        m['docres'] = ink
        assert all(v.shape == shape[:2] for v in m.values())
        mae = float(np.abs(a['original'].astype(float)-a['saved_input']).mean())
        exact = bool(np.array_equal(a['original'], a['saved_input']))
        assert exact == frame['input_reference_exact']
        input_note = '参照原画像と保存npj入力は全画素一致。' if exact else '注意：参照原画像と保存npj入力は異なる。下の出自比較を参照。'
        inputs.append(dict(frame=fid, exact=exact, reference_saved_input_MAE_255=mae))
        footers = [input_note, 'DocRes列は白黒の文字候補。色復元やマスク改善後のDDRM出力ではない。',
                   '新モデルは原画像だけへ適用。保存npj・旧マスクは入力せず、出力の合成はしない。']
        keys = ['original', 'saved_npj', 'stain', 'docres']
        name = 'overview_'+fid+'.png'
        rendering.append(dict(file=name, **figure([a[k] for k in keys], labels, OUT/name,
                         frame['page']+' / '+frame['baseline_label'], footers, overview=True)))
        if not exact:
            name = 'input_provenance_'+fid+'.png'
            rendering.append(dict(file=name, **figure([a[k] for k in ['original','saved_input','saved_npj']],
                ['新候補の参照原画像', '実保存npjの入力', '実保存npj出力'], OUT/name,
                frame['page']+'：入力の斑点・濃淡差を別表示',
                [f'入力差のMAE {mae:.4f}/255。モデルだけの同一条件比較と解釈しない。',
                 '保存npjを再推論しない。参照原画像へ新モデルを適用した比較。'], overview=True)))
            matched = [a['saved_input'], a['saved_npj'],
                       read(TRIAL/'matched_stain'/fid/'output.png'),
                       read(TRIAL/'matched_docres'/fid/'output.png')]
            pm = np.load(TRIAL/'matched_docres'/fid/'background_probability.npy', allow_pickle=False)
            assert np.isfinite(pm).all() and not (pm == .5).any()
            assert np.array_equal(matched[3][:,:,0] == 0, pm < .5)
            matched_labels = ['実保存入力（原画像）', '実保存npj', 'Stain 同一入力', 'DocRes 同一入力']
            matched_notes = ['全列は過去の実保存入力に由来。新候補へも同じ入力・固定設定を使用。',
                             'DocResは文字候補。新しいDDRM出力や、色を保つ汚れ除去結果ではない。',
                             '参照原画像による主比較とは別対照。ここでもモデル出力の合成はしない。']
            name = 'matched_overview_'+fid+'.png'
            rendering.append(dict(file=name, **figure(matched, matched_labels, OUT/name,
                frame['page']+'：過去のnpjと同一入力の対照', matched_notes, overview=True)))
            for region in frame['regions']:
                x,y,w,h = region['xywh']; s = np.s_[y:y+h,x:x+w]
                name = 'matched_'+region['id']+'.png'
                rendering.append(dict(file=name, **figure([v[s] for v in matched], matched_labels,
                    OUT/name, region['title']+'：過去のnpjと同一入力', matched_notes)))
        overlays = [a['original'], tint(a['original'], m['protect']|m['hold']),
                    tint(a['original'], m['structural']), tint(a['original'], m['docres'])]
        for region in frame['regions']:
            rid = region['id']; x,y,w,h = region['xywh']; s = np.s_[y:y+h,x:x+w]
            assert a['original'][s].shape == (h,w,3)
            name = rid+'.png'
            rendering.append(dict(file=name, **figure([a[k][s] for k in keys], labels, OUT/name,
                 region['title']+' / '+frame['page'], footers)))
            name = 'mask_'+rid+'.png'
            rendering.append(dict(file=name, **figure([v[s] for v in overlays],
                 ['参照原画像', '従来 保護＋保留', '構造ペア候補', 'DocRes インク候補'], OUT/name,
                 region['title']+'：青は候補、文字の正解ではない',
                 ['従来マスクは実保存入力由来、新候補は参照原画像由来。朱・不確実域も保護に含む。',
                  '構造候補は論文着想の試作。DocResは原画像の二値化。補集合を汚れとしない。', input_note])))
            cuts = OUT/'native'/rid
            for k in keys: write(cuts/(k+'_image.png'), a[k][s])
            for k in m: write(cuts/(k+'_mask.png'), m[k][s])
            counts = {k:int(v[s].sum()) for k,v in m.items()}
            records.append(dict(frame=fid, **region, mask_pixels=counts,
                docres_high_confidence_background_pixels=int((probability[s] >= .9).sum()),
                structural_but_docres_background_pixels=int((m['structural'][s] & ~m['docres'][s]).sum()),
                stain_mean_abs_reference_delta=float(np.abs(a['stain'][s].astype(float)-a['original'][s]).mean()),
                precision=False, pixel_GT=False))
        if fid == 'crop_hnsd':
            for method in ['stain','docres']:
                whole = read(TRIAL/method/fid/'whole_frame_output.png')
                name = 'tile_sensitivity_'+method+'.png'
                rendering.append(dict(file=name, **figure([a['original'], a[method], whole],
                     ['参照原画像', '512pxタイル・64px余裕', '画像全体・32倍数padding'], OUT/name,
                     'hnsd：推論範囲の違い / '+method,
                     ['両方とも同じ重み・原画像。タイル切り方の感度を見る補助対照。',
                      '学習時入力範囲や全体attentionが変わり、論文の評価値を再現する条件ではない。'])))
    assert len(records) == 20 and len(inputs) == 6
    assert len(rendering) == 58 and len(list(OUT.glob('*.png'))) == 58
    dump(OUT/'diagnostics.json', dict(regions=records, inputs=inputs, rendering=rendering,
        count_note='6 frames from 5 pages, all previously known development cases; 20 ROIs not documents',
        counts_are_precision=False, adopted=False))
    if render_only:
        print('Rendered 58 comparison PNGs and verified all native panels.'); return
    assert NOTES.exists(), 'Inspect figures and write findings before freeze'
    report = '''# 文献の手掛かりを保存npjの実例で検証

2026-10-02。「その手順で進めてみよう」を受け、保存済みnpjを固定したまま、背景補償＋対向エッジの構造試作、StainRestorer、DocResを実画像へ適用した。6保存フレーム・5ページ・20固定ROI。すべて既知の開発例で、独立した新規資料の精度評価ではない。イラストページは対象外。ROIは外部モデル出力を得る前に固定し、結果に応じた選別はしない。

## 今回の結論と目視所見

'''+NOTES.read_text()+'''

## 比較の意味と実装条件

保存npjは実際の既存推論出力そのもの。npj再推論、DDRMの再実行、文字の描き足し、出力の選択合成はしていない。新候補は参照原画像だけから推定。npjの版は混在しており、同じ版の統一ベンチマークではない。各全体PNGの見出しに保存版を示す。200010454の旧切り出しとCSC v2全体は同じページの別条件で、独立2資料として数えない。

StainRestorer列は通常のStainDoc重みによる色出力。DocRes列はbinarizationの文字候補で、白黒の結果を色復元の成功・失敗と混同しない。マスク列は推定の比較で、マスク精度によるDDRM後の差ではない。従来protect／holdは文字だけの正解マスクではなく、朱・未確定の材質も含む。

全画像を原寸で推定。外部モデルは512pxタイル、64px周辺余裕、384pxコアを一度ずつ採用し、端はreflect101。入力縮小なし、float32、TF32無効、全重みstrict一致。StainRestorerはRGBの0〜1入力、返却リスト先頭を採用。DocResは著者コードのBGR＋原画像全体で求めたSauvola/threshold/gradient prompt、最初の2チャネルをsoftmaxして背景クラス1を白へ復号した。著者のモデル構造・重みは改変していないが、原寸タイル推論は本プロジェクトの適用方法であり、論文のベンチマーク条件の忠実再現ではない。hnsdでは全体推論の対照も提示する。

構造試作はSobel対向勾配・3〜32pxの暗い内部・幅推定・形態学的背景補償・再判定を使用。Xiong/Jiaの着想を試した独自診断で、論文方式の完全再現ではない。ページ／ROI固有の座標や正解ラベルを推定へ入れていない。

## 重み・ソースの出自

- [StainRestorer論文](https://arxiv.org/abs/2410.22922)／[著者コード](https://github.com/CXH-Research/StainRestorer)：コミット f25239eb0ea817ce04bdfc2a3209157aa2e7998a、著者公開releaseの[通常StainDoc.pth](https://github.com/CXH-Research/StainRestorer/releases/download/Weights/StainDoc.pth)。mark／seal用重みは使わない。現代の染み付き紙の学習で、古文書での筆画・印章保全を保証しない。
- [DocRes論文](https://arxiv.org/abs/2405.04408)／[著者コード](https://github.com/ZZZHANG-jx/DocRes)：コミット d3e3a18c7c7ad10e4615ca2eb6a10f83128d8560。著者OneDriveからの取得は認証で通らず、[公開再配布](https://huggingface.co/DaVinciCode/doctra-docres-main/blob/main/docres.pkl)を利用。SHA256 `1d6a89d754fe1e58ffd1865eab0ef3f03344798d39197b2d9a77ce4fbc8c02fd` が別公開mirrorとも一致し、著者モデルへ全パラメータ一致。ただし著者配布原本との同一性を直接確認できていない。
- [背景補償の論文](https://link.springer.com/article/10.1186/s13640-021-00556-4)／[筆画の構造対称性の論文](https://doi.org/10.1016/j.patcog.2017.09.032)：着想として参照。[前回の文献調査](../document_restoration_literature_20261002/report.md)に適用範囲と次候補を記録。

## 入力の違い

4フレームは参照原画像と実保存npj入力が全画素一致。200017197は実保存入力に暖色の斑点があり、参照原画像には同じ斑点が見られない。圧縮差だけとは扱わず、付与目的・材質の正解は未確定とする。hnsdは濃淡が大きく異なる。主比較では新モデルは参照原画像、保存npjは過去の保存入力に由来する。特にhnsdでの汚れ残留差を主比較だけでモデルの効果と解釈しない。保存入力自体も並べた[200017197出自比較](input_provenance_page_200017197.png)、[hnsd出自比較](input_provenance_crop_hnsd.png)を参照。

この入力差を切り分けるため、両新モデルにも過去の実保存入力をそのまま渡す追加対照を実行した。設定・重みは主比較と同じ。新たな独立2資料ではない。[200017197の同一入力対照](matched_overview_page_200017197.png)、[hnsdの同一入力対照](matched_overview_crop_hnsd.png)。対応する6固定ROIも全件保存し、特に[濃い塊の同一入力対照](matched_10_rule_crossing_dark_mass.png)で確認する。

## 全固定ROIの比較PNG

'''
    for r in records:
        report += f'- [{r["title"]}]({r["id"]}.png)／[対応する分類候補](mask_{r["id"]}.png)\n'
        if r['frame'] in ['crop_hnsd','page_200017197']:
            report += f'  [過去のnpjと同一入力の追加対照](matched_{r["id"]}.png)\n'
    report += '\n## 全体・タイル感度・検証\n\n'
    for frame in spec['frames']:
        report += f'- [{frame["page"]}・{frame["baseline_label"]}](overview_{frame["id"]}.png)\n'
    report += '''
[Stainタイル感度](tile_sensitivity_stain.png)、[DocResタイル感度](tile_sensitivity_docres.png)、[全58比較PNG ZIP](comparison_pngs.zip)。拡大パネルは同じ座標の整数最近傍表示で色補正なし。全体像のみ閲覧用に縮小。

[数値診断](diagnostics.json)の候補画素数・信頼度・出力差は精度、筆画保持率、汚れ除去率ではない。手描きGT、材質確定、未知の別冊評価は未実施。画像を見るための診断値に留める。

入力・保存npj・全依存ソース・モデル出力ハッシュ、全パネル画素一致、原寸・有限値・DocRes背景確率からの独立復号、全ZIP項目CRCとバイト一致を確認。[検証](verification.json)に保存。今回モデルを実行しただけで、現行処理への採用・NPJへの上書きはしていない。
'''
    (OUT/'report.md').write_text(report)
    dependencies = {}
    for method in ['stain','docres','structural','matched_stain','matched_docres']:
        for name in ['manifest.json','complete.json']:
            p = TRIAL/method/name; dependencies[str(p.relative_to(ROOT))] = sha(p)
    for p in [TRIAL/'selection.json', TRIAL/'selection_complete.json', NOTES, Path(__file__),
              Path(__file__).parent/'package_saved_npj_role_separated_review.py',
              Path(__file__).parent/'package_saved_npj_inference_v6_review.py',
              Path(__file__).parent/'build_npj_v6_comparison.py']:
        dependencies[str(p.relative_to(ROOT))] = sha(p)
    dump(OUT/'manifest.json', dict(dependencies=dependencies, primary_artifact='PNG',
        frames=6, pages=5, fixed_ROIs=20, figures=58, pixel_GT=False, adopted=False))
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')): z.write(p,p.name)
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert z.testzip() is None and len(z.namelist()) == 58
        for name in z.namelist(): assert z.read(name) == (OUT/name).read_bytes()
    dump(OUT/'verification.json', dict(status='passed', saved_inputs_and_npj_unchanged=True,
        all_source_model_selection_output_hashes=True, native_shapes=True,
        finite_probabilities=True, independent_docres_decode=True,
        all_panels_pixel_exact=True, zip_CRC_bytes_exact=True,
        fixed_ROIs_complete=True, inputs=inputs, pixel_GT=False, adopted=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)):sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('Packaged and froze 58 PNGs; no candidate adopted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--render-only',action='store_true')
    main(parser.parse_args().render_only)
