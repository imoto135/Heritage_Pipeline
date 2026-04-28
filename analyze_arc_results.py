#!/usr/bin/env python3
"""arc_recognition のJSON結果を分析・集約するスクリプト"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import statistics

def analyze_arc_results(output_dir="output/arc_recognition"):
    """arc_recognition 配下のすべての arc_result.json を読み込んで統計を出力"""
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Error: {output_dir} が見つかりません")
        return
    
    # データ集計用
    all_predictions = []
    char_counts = Counter()
    prob_scores = []
    failed_folders = []
    success_count = 0
    
    # すべてのサブフォルダを走査
    for folder in sorted(output_path.iterdir()):
        if not folder.is_dir():
            continue
        
        json_path = folder / "arc_result.json"
        if not json_path.exists():
            failed_folders.append(folder.name)
            continue
        
        try:
            with open(json_path, encoding="utf-8") as f:
                predictions = json.load(f)
            
            success_count += 1
            for pred in predictions:
                char = pred.get("char", "?")
                prob = pred.get("prob", 0)
                char_counts[char] += 1
                prob_scores.append(prob)
                all_predictions.append({
                    "folder": folder.name,
                    "char": char,
                    "prob": prob,
                    "top5": pred.get("top5", [])
                })
        except Exception as e:
            failed_folders.append(f"{folder.name} (Error: {e})")
    
    # 統計情報を表示
    print("=" * 80)
    print("ARC Recognition Results Analysis")
    print("=" * 80)
    print()
    
    print(f"処理済みフォルダ数: {success_count}")
    print(f"失敗/スキップ: {len(failed_folders)}")
    print(f"認識された文字総数: {len(all_predictions)}")
    print()
    
    # 信頼度の統計
    if prob_scores:
        print("信頼度統計:")
        print(f"  最高: {max(prob_scores):.4f}")
        print(f"  最低: {min(prob_scores):.4f}")
        print(f"  平均: {statistics.mean(prob_scores):.4f}")
        print(f"  中央値: {statistics.median(prob_scores):.4f}")
        if len(prob_scores) > 1:
            print(f"  標準偏差: {statistics.stdev(prob_scores):.4f}")
        print()
    
    # 信頼度のレンジ別集計
    confidence_ranges = {
        "0.9-1.0": 0,
        "0.8-0.9": 0,
        "0.7-0.8": 0,
        "0.6-0.7": 0,
        "0.5-0.6": 0,
        "0.0-0.5": 0,
    }
    for prob in prob_scores:
        if prob >= 0.9:
            confidence_ranges["0.9-1.0"] += 1
        elif prob >= 0.8:
            confidence_ranges["0.8-0.9"] += 1
        elif prob >= 0.7:
            confidence_ranges["0.7-0.8"] += 1
        elif prob >= 0.6:
            confidence_ranges["0.6-0.7"] += 1
        elif prob >= 0.5:
            confidence_ranges["0.5-0.6"] += 1
        else:
            confidence_ranges["0.0-0.5"] += 1
    
    print("信頼度別分布:")
    for range_name in ["0.9-1.0", "0.8-0.9", "0.7-0.8", "0.6-0.7", "0.5-0.6", "0.0-0.5"]:
        count = confidence_ranges[range_name]
        pct = (count / len(prob_scores) * 100) if prob_scores else 0
        print(f"  {range_name}: {count:6d} ({pct:5.1f}%)")
    print()
    
    # 最頻出の文字トップ30
    print("認識された文字トップ30:")
    for i, (char, count) in enumerate(char_counts.most_common(30), 1):
        pct = (count / len(all_predictions) * 100) if all_predictions else 0
        print(f"  {i:2d}. '{char}': {count:6d} ({pct:5.1f}%)")
    print()
    
    # 信頼度別の結果を別ファイルに保存
    output_json = Path(output_dir) / "arc_analysis_summary.json"
    summary = {
        "total_predictions": len(all_predictions),
        "success_folders": success_count,
        "failed_folders": len(failed_folders),
        "confidence_stats": {
            "max": max(prob_scores) if prob_scores else 0,
            "min": min(prob_scores) if prob_scores else 0,
            "mean": statistics.mean(prob_scores) if prob_scores else 0,
            "median": statistics.median(prob_scores) if prob_scores else 0,
            "stdev": statistics.stdev(prob_scores) if len(prob_scores) > 1 else 0,
        },
        "confidence_ranges": confidence_ranges,
        "top_30_chars": dict(char_counts.most_common(30)),
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 分析結果を保存: {output_json}")
    print()
    
    # 失敗したフォルダを表示
    if failed_folders:
        print("スキップされたフォルダ:")
        for folder in failed_folders[:20]:  # 最初の20個まで表示
            print(f"  - {folder}")
        if len(failed_folders) > 20:
            print(f"  ... 他 {len(failed_folders) - 20} 件")
    
    return summary, all_predictions


if __name__ == "__main__":
    analyze_arc_results()
