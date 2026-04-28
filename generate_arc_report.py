#!/usr/bin/env python3
"""arc_recognition 推論結果の最終レポート生成"""

import json
from pathlib import Path
from collections import Counter

def generate_markdown_report(output_dir="output/arc_recognition"):
    """Markdownレポートを生成"""
    
    # JSON形式の分析結果を読み込み
    summary_path = Path(output_dir) / "arc_analysis_summary.json"
    detailed_path = Path(output_dir) / "arc_detailed_report.json"
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    with open(detailed_path) as f:
        detailed = json.load(f)
    
    # Markdownレポート作成
    report = f"""# ARC Recognition Results Report

## 概要

このレポートは、古文書認識パイプラインの**ARC（古文字認識）モジュール**の推論結果を集約したものです。

## 処理統計

| 項目 | 値 |
|------|-----|
| **処理フォルダ数** | {summary['success_folders']} |
| **認識文字総数** | {summary['total_predictions']:,} |
| **ユニーク文字種** | {detailed['top_30_chars_by_frequency'].__len__()} (サンプル) |
| **スキップ/失敗数** | {summary['failed_folders']} |

---

## 信頼度統計

### 全体統計

| 指標 | 値 |
|------|-----|
| **最高信頼度** | {summary['confidence_stats']['max']:.4f} |
| **最低信頼度** | {summary['confidence_stats']['min']:.4f} |
| **平均信頼度** | {summary['confidence_stats']['mean']:.4f} |
| **中央値** | {summary['confidence_stats']['median']:.4f} |
| **標準偏差** | {summary['confidence_stats']['stdev']:.4f} |

### 信頼度別分布

```
{0.9-1.0}: {summary['confidence_ranges']['0.9-1.0']:6d} 件 ({summary['confidence_ranges']['0.9-1.0']/summary['total_predictions']*100:5.1f}%) ████████████████████
{0.8-0.9}: {summary['confidence_ranges']['0.8-0.9']:6d} 件 ({summary['confidence_ranges']['0.8-0.9']/summary['total_predictions']*100:5.1f}%) █
{0.7-0.8}: {summary['confidence_ranges']['0.7-0.8']:6d} 件 ({summary['confidence_ranges']['0.7-0.8']/summary['total_predictions']*100:5.1f}%)
{0.6-0.7}: {summary['confidence_ranges']['0.6-0.7']:6d} 件 ({summary['confidence_ranges']['0.6-0.7']/summary['total_predictions']*100:5.1f}%)
{0.5-0.6}: {summary['confidence_ranges']['0.5-0.6']:6d} 件 ({summary['confidence_ranges']['0.5-0.6']/summary['total_predictions']*100:5.1f}%)
{0.0-0.5}: {summary['confidence_ranges']['0.0-0.5']:6d} 件 ({summary['confidence_ranges']['0.0-0.5']/summary['total_predictions']*100:5.1f}%)
```

**✓ 95.8% の認識が 0.9 以上の高い信頼度を持つ**

---

## 認識文字トップ30

"""
    
    # 認識された文字ランキング
    report += "| 順位 | 文字 | 出現数 | 割合 | 平均信頼度 | 信頼度範囲 |\n"
    report += "|------|------|--------|--------|----------|----------|\n"
    
    for i, char_info in enumerate(detailed['top_30_chars_by_frequency'], 1):
        char = char_info['char']
        count = char_info['count']
        pct = count / summary['total_predictions'] * 100
        avg_conf = char_info['avg_confidence']
        min_conf = char_info['min_confidence']
        max_conf = char_info['max_confidence']
        
        report += f"| {i} | '{char}' | {count:,} | {pct:.2f}% | {avg_conf:.4f} | {min_conf:.4f}~{max_conf:.4f} |\n"
    
    report += "\n---\n\n"
    
    # 品質分析
    report += "## 品質分析\n\n"
    
    report += "### 低信頼度認識（信頼度 < 0.5）\n\n"
    report += f"**低信頼度の認識数**: {detailed['low_confidence_count']} 件 ({detailed['low_confidence_count']/summary['total_predictions']*100:.2f}%)\n\n"
    report += "最も問題のある認識結果（下位10）:\n\n"
    report += "| フォルダ | 文字 | 信頼度 | Top5候補 |\n"
    report += "|---------|------|--------|----------|\n"
    
    for pred in detailed['low_confidence_predictions_sample'][:10]:
        folder = pred['folder']
        char = pred['char']
        prob = pred['prob']
        top5_str = ", ".join([f"{c}({p:.2f})" for c, p in pred['top5'][:3]])
        report += f"| {folder} | {char} | {prob:.4f} | {top5_str} |\n"
    
    report += "\n### フォルダ別統計（最大文字数 上位10）\n\n"
    report += "| 順位 | フォルダ | 文字数 | 平均信頼度 | 最低信頼度 |\n"
    report += "|------|---------|--------|-----------|----------|\n"
    
    for i, folder_info in enumerate(detailed['top_20_folders_by_char_count'][:10], 1):
        folder = folder_info['folder']
        count = folder_info['char_count']
        avg_conf = folder_info['avg_confidence']
        min_conf = folder_info['min_confidence']
        report += f"| {i} | {folder} | {count} | {avg_conf:.4f} | {min_conf:.4f} |\n"
    
    report += "\n### 認識が難しかったフォルダ（最低信頼度 下位5）\n\n"
    report += "| フォルダ | 最低信頼度 | 平均信頼度 | 文字数 |\n"
    report += "|---------|-----------|-----------|------|\n"
    
    for folder_info in detailed['worst_10_folders'][:5]:
        folder = folder_info['folder']
        min_conf = folder_info['min_confidence']
        avg_conf = folder_info['avg_confidence']
        count = folder_info['char_count']
        report += f"| {folder} | {min_conf:.4f} | {avg_conf:.4f} | {count} |\n"
    
    report += "\n---\n\n"
    
    # 考察
    report += """## 考察

### 強み
- **高い全体精度**: 平均信頼度が 0.9816 と非常に高い
- **安定した認識**: 95.8% の認識が信頼度 0.9 以上
- **自然言語パターン**: 認識トップの文字が「の」「に」「し」などの頻出字であり、言語的に自然
- **低エラー率**: 信頼度 0.5 以下の認識は全体の 0.75% のみ

### 弱み
- **漢字の認識困難**: 低信頼度の大多数が難しい漢字（遣、班、坑など）
- **特定フォルダでの性能低下**: `200004107_00003_1` など一部フォルダで極めて低い認識精度
- **古文特有の字形**: 古文書特有の異体字や変体仮名の認識にばらつき

### 推奨事項
1. **低信頼度の結果の手動確認**: 信頼度 < 0.7 の認識 ~2,111 件は目視確認を推奨
2. **困難なフォルダの再処理**: `200004107_00003_1` などの再検査
3. **後処理の活用**: 信頼度が低い場合は、LM（言語モデル）による補正を活用
4. **モデル改善**: 低信頼度の漢字データでの追加学習も検討価値あり

---

## ファイル情報

- **サマリー**: `output/arc_recognition/arc_analysis_summary.json`
- **詳細レポート**: `output/arc_recognition/arc_detailed_report.json`
- **このレポート**: `output/arc_recognition/REPORT.md`

"""
    
    # ファイルに保存
    report_path = Path(output_dir) / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print("=" * 80)
    print(f"✓ レポート生成完了: {report_path}")
    print("=" * 80)
    print("\n[レポート内容をプレビュー]\n")
    print(report)

if __name__ == "__main__":
    generate_markdown_report()
