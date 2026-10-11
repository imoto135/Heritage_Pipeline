"""Read-only diagnosis of the completed pilot; writes new reports only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'tmp_work/fidelity_pilot_20260923_run1'
OUT = ROOT / 'tmp_work/fidelity_diagnosis_20260930'


def read(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Unreadable image: {path}')
    return image


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def average(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    status = json.loads((RUN / 'status.json').read_text())
    assert status['status'] == 'complete', status
    manifest = json.loads((RUN / 'manifest.json').read_text())
    failures = []
    checked = 0
    for marker in sorted(RUN.glob('*/*/*.complete.json')):
        record = json.loads(marker.read_text())
        assert record['run_signature'] == status['run_signature'], marker
        for path, expected in record['outputs'].items():
            checked += 1
            if sha(path) != expected:
                failures.append(path)
    assert checked > 0 and not failures, failures
    source_changes = [path for path, expected in manifest['source_hashes'].items()
                      if not (ROOT / path).exists() or sha(ROOT / path) != expected]
    with open(RUN / 'metrics.csv', newline='') as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == status['rows'] == 42
    rows = []
    for key, item in sorted(manifest['cases'].items()):
        stem, intensity = key.split('/')
        case = RUN / stem / intensity
        clean = read(ROOT / 'tmp_work/makura_benchmark/clean' / f'{stem}.png')
        observed = read(item['input'])
        g = gray(clean)
        masks = {}
        for name, filename, foreground in (
            ('black', '02_text_black.png', 0),
            ('red', '03_text_red.png', 0),
            ('degradation', '04_degradation.png', 255)):
            mask = cv2.imread(str(case / 'masks' / stem / filename), 0)
            assert mask is not None and mask.shape == g.shape
            masks[name] = mask == foreground
        text = cv2.imread(str(case / 'evaluation/text_proxy.png'), 0) == 255
        ink = masks['black'] | masks['red']
        missed = text & ~ink
        p1 = read(case / 'prepared/P1.png')
        assert np.array_equal(p1[ink], observed[ink]), key
        base = dict(stem=stem, intensity=intensity,
                    text_proxy_pixels=int(text.sum()),
                    csc_ink_coverage_of_text_proxy=average(ink, text),
                    text_proxy_classified_degradation=average(masks['degradation'], text),
                    p1_missed_text_mae=average(np.abs(gray(p1)-g), missed),
                    p1_missed_text_lightened_20_fraction=average(gray(p1)-g >= 20, missed))
        for method, path in (
            ('X', Path(item['input'])), ('P1', case / 'prepared/P1.png'),
            ('R1', case / 'R1/output.png'), ('F0', case / 'F0/output.png'),
            ('Fp', case / 'Fp/output.png'), ('F1', case / 'F1/output.png')):
            image = read(path)
            assert image.shape == clean.shape, path
            r = gray(image)
            signed = r-g
            rows.append(dict(base, method=method,
                             text_proxy_mae=average(np.abs(signed), text),
                             text_proxy_darkening=average(np.maximum(-signed, 0), text),
                             text_proxy_near_black_fraction=average(r <= 5, text),
                             csc_black_near_black_fraction=average(r <= 5, masks['black'])))
    OUT.mkdir(exist_ok=True)
    with open(OUT / 'diagnostics.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ['| ページ | 強度 | CSC文字被覆率 | P1文字MAE | F1文字MAE | F1文字の輝度≤5率 |',
             '|---|---:|---:|---:|---:|---:|']
    for key in sorted(manifest['cases']):
        stem, intensity = key.split('/')
        selected = {r['method']: r for r in rows if r['stem'] == stem and r['intensity'] == intensity}
        f1, p1 = selected['F1'], selected['P1']
        table.append(f"| {stem} | {intensity} | {f1['csc_ink_coverage_of_text_proxy']:.1%} | "
                     f"{p1['text_proxy_mae']:.2f} | {f1['text_proxy_mae']:.2f} | "
                     f"{f1['text_proxy_near_black_fraction']:.1%} |")
    report = '''# 忠実度試走の診断（2026-09-30）

9月23日の2ページ×3強度の実行は完了。42行の評価結果を確認した。
推論出力は自動文字proxy上で、前処理の文字欠損とDDRM段階の濃色化を示す。
提案条件F1の優位性はこの試走から確認できない。

## 原因の切り分け

1. `fidelity_core.build_references`は文字・朱書を含めてinpaintし、CSCが文字と判定した画素だけを入力から戻す。
   CSCの取りこぼしが補間領域に入ると、その文字は前処理P1ですでに消え得る。
   P1のCSC文字領域が入力と完全一致することを全6入力で検証した。
2. `fidelity_stage`のDDRM分岐は元入力をパッチの`00_input.png`に保存するが、
   `Inpainting.sample_sequence`がモデルの観測値として読むのは背景参照の`SingleDataset`。
   CSC黒文字はmissingであり、元入力の墨色を拘束する観測には使われない。
   元画像から戻す処理は朱書のみ。黒文字は生成結果になる。
   したがって元の墨色を保つ目的は、この実行経路では保証されない。
3. サンプラーは早期ステップで`xt *= mask`を適用する。黒文字をゼロにする設定があるが、
   濃色化の寄与を特定するには、この設定を変えた対応実験が必要。
   墨の一律gainや紙色203への固定は今回の参照生成では使っていない。

## 自動proxyによる診断値

文字MAEは8bit grayで、元画像との差。CSC文字被覆率は手動注釈でない。
輝度≤5は生成結果の極端な黒の診断であり、文字形状の正確さを表さない。

''' + '\n'.join(table) + '''

## 次に検証する条件

- 凍結済み試走と既存モジュールは保存し、入力の黒・赤文字を最終合成する条件を別名で比較する。
  この条件は観測墨色を保持するが、シミが文字に重なる領域やCSC取りこぼしを解決しない。
- 文字マスクの欠損を手動確認ROIで検証する。評価GTを推論に渡さない。
- サンプリングの文字ゼロ化の有無を同一seed・参照・マスクで比較し、墨色への寄与を切り分ける。
- 本番拡大や仮説検定の前に、文字形状と色保存の評価を固める。

本書は既存出力の解析のみ。モデル変更・追加推論は未実施。
LPIPS、手動注釈、独立データ、rescue有無比較は未完了。
'''
    (OUT / 'report.md').write_text(report)
    summary = dict(status='complete', verified_output_hashes=checked,
                   source_files_changed_since_run=source_changes,
                   diagnostic_rows=len(rows), cases=len(manifest['cases']),
                   report=str(OUT / 'report.md'))
    (OUT / 'status.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))
    print('\n'.join(table))


if __name__ == '__main__':
    main()
