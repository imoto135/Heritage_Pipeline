"""Finalize provenance after visual inspection of the local DDRM experiment."""
from pathlib import Path
import json
import ast
import cv2
import numpy as np
from PIL import Image
import run_ddrm_pilot as driver

BASE=Path(__file__).resolve().parent
ROOT=driver.ROOT
OUT=BASE/'bleed_local_review'
TRIAL=OUT/'local_exclusion_trial'
PAGE='200021071_00033_2'

def main():
    annotation=json.loads((OUT/'局所分類.json').read_text())
    complete=json.loads((TRIAL/'complete.json').read_text())
    for rel, expected in annotation['dependencies'].items():
        assert driver.digest(ROOT/rel)==expected,rel
    assert driver.digest(OUT/'局所分類.json')==complete['annotation_sha256']
    olddir=BASE/'six_challenge_fixed_review/masks/howe'/PAGE
    newdir=TRIAL/'masks/local_review'/PAGE
    old=np.asarray(Image.open(olddir/'02_text_black.png').convert('L'))<127
    new=np.asarray(Image.open(newdir/'02_text_black.png').convert('L'))<127
    removed=old&~new
    assert int(removed.sum())==515 and not np.any(new&~old)
    assert driver.digest(olddir/'02_text_black.png')==complete['black_before_sha256']
    assert driver.digest(newdir/'02_text_black.png')==complete['black_after_sha256']
    assert driver.digest(olddir/'03_text_red.png')==driver.digest(newdir/'03_text_red.png')==complete['red_before_sha256']
    beforepath=BASE/'six_challenge_fixed_review/outputs/howe'/PAGE/'result'/f'{PAGE}.png'
    afterpath=ROOT/complete['result']['output']
    assert driver.digest(beforepath)==complete['baseline_output_sha256']
    assert driver.digest(afterpath)==complete['result']['sha256']
    before=np.asarray(Image.open(beforepath).convert('RGB'));after=np.asarray(Image.open(afterpath).convert('RGB'))
    assert before.shape==after.shape==(3264,1968,3)
    protected=old&~removed
    bg=cv2.cvtColor(before,cv2.COLOR_RGB2GRAY);ag=cv2.cvtColor(after,cv2.COLOR_RGB2GRAY)
    diagnostic=dict(retained_black_mask_pixels=int(protected.sum()),
                    retained_black_pixels_rgb_changed=int(np.any(before!=after,axis=2)[protected].sum()),
                    retained_dark_to_light_pixels=int((protected&(bg<100)&(ag>=100)).sum()),
                    retained_light_to_dark_pixels=int((protected&(bg>=100)&(ag<100)).sum()),
                    full_output_changed_pixels=int(np.any(before!=after,axis=2).sum()))
    for item in complete['component_diagnostics']:
        if item['excluded']:assert item['after_dark_pixels']==0
        else:assert item['before_dark_pixels']==item['after_dark_pixels']
    log=TRIAL/'logs/07_local_review.log'
    assert '[ERROR]' not in log.read_text(errors='replace')
    for name in ['review_bleed_local.py','trial_bleed_local.py','finalize_bleed_local.py']:
        ast.parse((BASE/name).read_text())
    fits=annotation['next_page_local_translation']['fits'][1]['probes']
    score=' / '.join(f"{r['ncc']:.3f}" for r in fits)
    lines=['# 07：残存像の局所分類と除外実験', '',
           '**上部に残った文字状の像は、次ページ00034_1の鏡像の濃い文字と形状が対応した。対応する4成分（合計515画素）を局所的に除外したDDRM比較を作成した。** 直前ページだけでは説明できなかった像について、近隣ページとの照合で新しい根拠が得られた。', '',
           'この4成分は一つの文字状のまとまりに属する。4つの独立した裏移り事例ではない。鏡像の形状対応から「近隣ページ由来の裏移り・転写候補」と分類するが、紙の透過、接触転写、その他の物理的な発生機序を画像だけで確定しない。', '',
           '## 比較画像', '',
           '- [局所除外前後のDDRM比較：原画像を含む](local_exclusion_trial/07_局所除外_DDRM比較.png)',
           '- [全ページ比較](local_exclusion_trial/07_局所除外_全ページ比較.png)',
           '- [次ページとの形状対応と3手法](07_次ページ対応_拡大比較.png)',
           '- [残存8成分の暫定分類](07_残存8か所_分類と比較.png)',
           '- [近隣ページの文脈確認](07_残存8か所_近隣ページ確認.png)', '',
           '## 局所分類', '',
           '| 成分 | 暫定分類 | 今回の扱い |', '|---|---|---|']
    for r in annotation['components']:
        decision='局所除外の確認実験' if r.get('local_shape_match') else '保持'
        lines.append(f"| {r['label']} | {r['category']} | {decision} |")
    lines += ['', '点状像2成分は見た目による分類で、虫食い・汚れなどの原因の確定ではない。残りの不明2成分は短線と本文付近の像であり、薄い表文字の可能性を排除できないため保持する。この8成分は前回抽出した淡い残存候補で、ページ全体の全劣化を網羅した注釈ではない。', '',
        '## 次ページとの照合', '',
        f'左右反転して対象寸法に合わせた00034_1を使用した。紙面の局所的な濃淡差を強調し、手動で選んだ診断領域[1430,610,155,205]の周囲65pxを正規化相互相関で照合した。全体・上部・下部の相関は順に{score}で、3領域とも参照位置−対象位置が[14,-3]pxとなった。参照を[-14,+3]px動かした拡大比較で、輪と下部曲線の対応も目視確認した。', '',
        '上下領域は同一の文字状のまとまりを分割した確認であり、独立した資料での検証ではない。比較した00032_2では最良の移動量が上下で一致せず、相関も約0.21〜0.31だった。00034_1への移動は上部領域にだけ適用する局所的な診断で、ページ全体の位置合わせを保証しない。近隣ページ一覧の00032_2と00034_1は寸法合わせのみで、同一座標をそのまま対応点とは見なしていない。', '',
        '## DDRMの確認実験', '',
        '既存Howeマスクの成分44・50・51・53だけを手動レビューに基づいて除外した。これらは保存CSC黒・赤マスクと重ならない。除外以外の黒マスク画素と赤マスクは完全に保持した。Howe c160、Sauvolaの設定、朱印判定は再調整していない。元の基本マスク・元画像・過去の出力は変更していない。', '',
        '推論は既存の凍結コード・モデル・背景、seed999、20steps、patch128、overlap32、shared initial noiseで1ページを新規実行した。黒マスク変更により各段の画素別乱数対応は一致を保証できない。全出力の差を除外成分の効果だけと解釈しない。', '',
        '| 成分 | 除外 | 出力の暗画素：前 | 後 |', '|---|---|---:|---:|']
    for r in complete['component_diagnostics']:
        lines.append(f"| {r['label']} | {'有' if r['excluded'] else '無'} | {r['before_dark_pixels']} | {r['after_dark_pixels']} |")
    lines += ['', '暗画素は成分領域内の出力グレー100未満の画素数で、認識精度や劣化除去率ではない。対象4成分の黒い残存像は除去できた。確認画像の主要な表文字は保持されており、特に右欄でCSCが欠落させる字形をHowe条件は保っている。同じ上部のまとまりには未選択の短線がなお残る。不明成分も引き続き残り、淡い曲線全体を修復・復元したという結果ではない。', '',
        f"元Howeの保持対象{diagnostic['retained_black_mask_pixels']}画素のうち、前後でRGBが変わった画素は{diagnostic['retained_black_pixels_rgb_changed']}。ただし、同領域でグレー100未満の暗画素から100以上に変わった画素は{diagnostic['retained_dark_to_light_pixels']}、逆方向も{diagnostic['retained_light_to_dark_pixels']}。全ページの出力変更は{diagnostic['full_output_changed_pixels']}画素で、周辺・背景の推論変化も含む。保持対象は元マスク由来であり、表文字の正解注釈ではない。", '',
        '## 次の検討', '',
        'Howeを基本の文字マスクとして維持する。今回の局所除外は07の説明と確認用の候補で、画像IDや手指定座標を使う処理を全資料へ適用する仕様にはしない。一般化には、近隣ページの候補選択・局所形状対応・表文字との重なりの保護を別資料で検証する必要がある。明るさやCSC未検出だけを理由に本文中の短線を削除しない。', '',
        '6課題は選定・調整済みの開発例であり、この結果を独立test精度として報告しない。現在のDDRM入力は黒・赤マスクで、CSC背景・劣化分類の直接統合は未実装。次の開発作業では撮影境界と紙面内の文字・枠線の区別、およびCSCの他クラスとの接続仕様を整理する。', '',
        '再現用： [局所比較](../review_bleed_local.py)、[除外推論](../trial_bleed_local.py)、[検証](../finalize_bleed_local.py)。条件とハッシュは[局所分類](局所分類.json)・[実験条件](local_exclusion_trial/protocol.json)・[推論結果](local_exclusion_trial/complete.json)に保存した。', '']
    (OUT/'検討結果.md').write_text('\n'.join(lines))
    annotation.update(status='visual_review_complete',
        visual_review='candidate gallery, mirrored neighboring scans, source/three-method controls and local trial/full-page inspected',
        default_mask_modified=False,experimental_trial='local_exclusion_trial',
        classification={'点・傷状の像':2,'不明':2,'近隣ページ対応の裏移り・転写候補':4})
    # Preserve the exact pre-inference annotation hashed by the trial.
    driver.save_json(OUT/'局所分類_確認済.json',annotation)
    complete.update(status='visual_review_complete',retained_mask_output_diagnostic=diagnostic)
    driver.save_json(TRIAL/'complete_visual_review.json',complete)
    artifacts=[OUT/'検討結果.md',OUT/'局所分類_確認済.json',TRIAL/'complete_visual_review.json']+list(OUT.glob('*.png'))+list(TRIAL.glob('*.png'))
    for p in artifacts:
        if p.suffix=='.png':Image.open(p).verify()
    driver.save_json(OUT/'final_audit.json',dict(status='complete',sources_unchanged=True,
        frozen_parameters_unchanged=True,default_masks_unchanged=True,new_inferences=1,
        excluded_components=[44,50,51,53],excluded_pixels=515,development_only=True,
        artifacts_sha256={str(p.relative_to(ROOT)):driver.digest(p) for p in artifacts},
        scripts_sha256={name:driver.digest(BASE/name) for name in ['review_bleed_local.py','trial_bleed_local.py','finalize_bleed_local.py']}))
    for name,target in [('07_局所除外_DDRM比較.png','bleed_local_review/local_exclusion_trial/07_局所除外_DDRM比較.png'),
                        ('07_残存像_検討結果.md','bleed_local_review/検討結果.md')]:
        link=BASE/name
        if not link.exists():link.symlink_to(target)
    print(json.dumps(diagnostic,ensure_ascii=False))

if __name__=='__main__':main()
