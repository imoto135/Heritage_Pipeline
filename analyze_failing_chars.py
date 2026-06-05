#!/usr/bin/env python3
"""失敗率（低信頼度）が高い文字を分析するスクリプト"""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_failing_chars(output_dir="output/arc_recognition"):
    """信頼度が低い文字を分析"""
    
    output_path = Path(output_dir)
    
    # 文字ごとの信頼度を集計
    char_confidences = defaultdict(list)
    char_occurrences = defaultdict(int)
    char_low_count = defaultdict(int)  # 信頼度<0.7の件数
    
    # すべてのサブフォルダを走査
    for folder in sorted(output_path.iterdir()):
        if not folder.is_dir():
            continue
        
        json_path = folder / "arc_result.json"
        if not json_path.exists():
            continue
        
        try:
            with open(json_path, encoding="utf-8") as f:
                predictions = json.load(f)
            
            for pred in predictions:
                char = pred.get("char", "?")
                prob = pred.get("prob", 0)
                
                char_confidences[char].append(prob)
                char_occurrences[char] += 1
                
                if prob < 0.7:
                    char_low_count[char] += 1
        except:
            pass
    
    # 分析結果を計算
    char_stats = {}
    for char, probs in char_confidences.items():
        low_ratio = char_low_count[char] / len(probs) * 100 if probs else 0
        char_stats[char] = {
            "count": char_occurrences[char],
            "avg_confidence": statistics.mean(probs),
            "min_confidence": min(probs),
            "max_confidence": max(probs),
            "stdev": statistics.stdev(probs) if len(probs) > 1 else 0,
            "low_confidence_count": char_low_count[char],
            "low_confidence_ratio": low_ratio,
        }
    
    print("=" * 90)
    print("失敗率が高い文字の分析")
    print("=" * 90)
    print()
    
    # 1. 信頼度が最も低い文字トップ20
    print("【1. 平均信頼度が最も低い文字 Top 20】")
    print(f"{'順位':<6} {'文字':<5} {'出現数':<10} {'平均信頼度':<15} {'最低':<10} {'失敗率':<12}")
    print("-" * 90)
    
    sorted_by_avg = sorted(char_stats.items(), 
                           key=lambda x: x[1]["avg_confidence"])
    
    for i, (char, stats) in enumerate(sorted_by_avg[:20], 1):
        if stats["count"] >= 3:  # 出現数が3以上のみ
            print(f"{i:<6} '{char}'   {stats['count']:<10} {stats['avg_confidence']:<15.4f} "
                  f"{stats['min_confidence']:<10.4f} {stats['low_confidence_ratio']:<12.2f}%")
    
    print()
    
    # 2. 失敗率（信頼度<0.7の割合）が最も高い文字
    print("【2. 失敗率が最も高い文字 Top 20（出現3回以上）】")
    print(f"{'順位':<6} {'文字':<5} {'出現数':<10} {'失敗数':<10} {'失敗率':<12} {'平均信頼度':<15}")
    print("-" * 90)
    
    sorted_by_failure = sorted(
        [(c, s) for c, s in char_stats.items() if s["count"] >= 3],
        key=lambda x: x[1]["low_confidence_ratio"],
        reverse=True
    )
    
    for i, (char, stats) in enumerate(sorted_by_failure[:20], 1):
        print(f"{i:<6} '{char}'   {stats['count']:<10} {stats['low_confidence_count']:<10} "
              f"{stats['low_confidence_ratio']:<12.2f}% {stats['avg_confidence']:<15.4f}")
    
    print()
    
    # 3. 最も不安定な文字（標準偏差が大きい）
    print("【3. 認識が不安定な文字 Top 20（標準偏差が大きい）】")
    print(f"{'順位':<6} {'文字':<5} {'出現数':<10} {'平均信頼度':<15} {'標準偏差':<15}")
    print("-" * 90)
    
    sorted_by_stdev = sorted(
        [(c, s) for c, s in char_stats.items() if s["count"] >= 3],
        key=lambda x: x[1]["stdev"],
        reverse=True
    )
    
    for i, (char, stats) in enumerate(sorted_by_stdev[:20], 1):
        print(f"{i:<6} '{char}'   {stats['count']:<10} {stats['avg_confidence']:<15.4f} "
              f"{stats['stdev']:<15.4f}")
    
    print()
    
    # 4. 最も失敗した個別ケース（信頼度が最も低い）
    print("【4. 最悪の認識結果（信頼度が最も低い個別ケース）】")
    
    worst_predictions = []
    for folder in sorted(output_path.iterdir()):
        if not folder.is_dir():
            continue
        
        json_path = folder / "arc_result.json"
        if not json_path.exists():
            continue
        
        try:
            with open(json_path, encoding="utf-8") as f:
                predictions = json.load(f)
            
            for pred in predictions:
                worst_predictions.append({
                    "folder": folder.name,
                    "char": pred.get("char", "?"),
                    "prob": pred.get("prob", 0),
                    "top5": pred.get("top5", [])
                })
        except:
            pass
    
    sorted_worst = sorted(worst_predictions, key=lambda x: x["prob"])
    
    print(f"{'フォルダ':<25} {'文字':<5} {'確信度':<12} {'Top3候補':<45}")
    print("-" * 90)
    
    for pred in sorted_worst[:20]:
        top3 = ", ".join([f"{c}({p:.3f})" for c, p in pred["top5"][:3]])
        print(f"{pred['folder']:<25} '{pred['char']}'  {pred['prob']:<12.4f} {top3:<45}")
    
    print()
    print("=" * 90)
    print("まとめ")
    print("=" * 90)
    
    # 出現3回以上で平均信頼度が0.7以下の文字
    very_difficult = [c for c, s in char_stats.items() 
                     if s["count"] >= 3 and s["avg_confidence"] < 0.7]
    print(f"✗ 平均信頼度が 0.7 以下の文字: {len(very_difficult)} 種")
    if very_difficult:
        print(f"  対象文字: {', '.join(sorted(very_difficult))}")
    
    # 失敗率が50%以上の文字
    high_failure = [c for c, s in char_stats.items()
                   if s["count"] >= 3 and s["low_confidence_ratio"] >= 50]
    print(f"✗ 失敗率が 50% 以上の文字: {len(high_failure)} 種")
    if high_failure:
        for char in sorted(high_failure):
            s = char_stats[char]
            print(f"  '{char}': {s['low_confidence_ratio']:.1f}% ({s['low_confidence_count']}/{s['count']})")
    
    # JSON形式で詳細を保存
    summary = {
        "analysis_type": "character_failure_rate",
        "worst_by_avg_confidence": [
            {
                "char": c,
                "avg_confidence": s["avg_confidence"],
                "count": s["count"],
                "low_confidence_ratio": s["low_confidence_ratio"],
                "min_confidence": s["min_confidence"]
            }
            for c, s in sorted_by_avg[:30] if s["count"] >= 3
        ],
        "worst_by_failure_rate": [
            {
                "char": c,
                "low_confidence_ratio": s["low_confidence_ratio"],
                "count": s["count"],
                "low_confidence_count": s["low_confidence_count"],
                "avg_confidence": s["avg_confidence"]
            }
            for c, s in sorted_by_failure[:30]
        ],
        "most_unstable": [
            {
                "char": c,
                "stdev": s["stdev"],
                "avg_confidence": s["avg_confidence"],
                "count": s["count"],
                "min_confidence": s["min_confidence"],
                "max_confidence": s["max_confidence"]
            }
            for c, s in sorted_by_stdev[:30]
        ]
    }
    
    output_json = Path(output_dir) / "failing_characters_analysis.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 詳細分析を保存: {output_json}")

if __name__ == "__main__":
    analyze_failing_chars()
