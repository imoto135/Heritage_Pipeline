#!/usr/bin/env python3
"""arc_recognition のJSON結果を詳細分析・レポート出力するスクリプト"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import statistics

def generate_detailed_report(output_dir="output/arc_recognition"):
    """詳細レポートを生成"""
    
    output_path = Path(output_dir)
    
    # データ集計用
    folder_stats = {}
    all_chars = []
    low_confidence_predictions = []  # 信頼度が低い認識
    
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
            
            # フォルダごとの統計
            if predictions:
                probs = [p.get("prob", 0) for p in predictions]
                folder_stats[folder.name] = {
                    "char_count": len(predictions),
                    "avg_confidence": statistics.mean(probs),
                    "min_confidence": min(probs),
                    "max_confidence": max(probs),
                }
                
                # 全文字を集約
                for pred in predictions:
                    char = pred.get("char", "?")
                    prob = pred.get("prob", 0)
                    all_chars.append((char, prob, folder.name))
                    
                    # 信頼度が低い認識を記録
                    if prob < 0.5:
                        low_confidence_predictions.append({
                            "folder": folder.name,
                            "char": char,
                            "prob": prob,
                            "position": pred.get("x1", -1),
                            "top5": pred.get("top5", [])
                        })
        except Exception as e:
            print(f"Error processing {folder.name}: {e}")
    
    # レポート出力
    print("\n" + "=" * 80)
    print("Detailed ARC Recognition Report")
    print("=" * 80)
    
    # フォルダ別の統計
    print("\n【フォルダ別統計（上位20）】")
    print(f"{'フォルダ':<30} {'文字数':<10} {'平均信頼度':<15} {'最低信頼度':<15}")
    print("-" * 70)
    
    sorted_folders = sorted(folder_stats.items(), 
                           key=lambda x: x[1]["char_count"], 
                           reverse=True)
    for folder_name, stats in sorted_folders[:20]:
        print(f"{folder_name:<30} {stats['char_count']:<10} "
              f"{stats['avg_confidence']:<15.4f} {stats['min_confidence']:<15.4f}")
    
    # 信頼度が最も低かったフォルダ
    print("\n【信頼度が最も低かったフォルダ（下位10）】")
    worst_folders = sorted(folder_stats.items(), 
                          key=lambda x: x[1]["min_confidence"])
    for folder_name, stats in worst_folders[:10]:
        print(f"{folder_name}: 最低={stats['min_confidence']:.4f}, "
              f"平均={stats['avg_confidence']:.4f}, 文字数={stats['char_count']}")
    
    # 信頼度が最も低い認識結果
    print("\n【信頼度が最も低い認識（下位30）】")
    print(f"{'フォルダ':<25} {'文字':<5} {'確信度':<12} {'Top5候補':<40}")
    print("-" * 82)
    
    sorted_low = sorted(low_confidence_predictions, 
                       key=lambda x: x["prob"])
    for pred in sorted_low[:30]:
        top5_str = ", ".join([f"{c}({p:.2f})" for c, p in pred["top5"][:3]]) if pred["top5"] else "N/A"
        print(f"{pred['folder']:<25} {pred['char']:<5} {pred['prob']:<12.4f} {top5_str:<40}")
    
    # 文字出現頻度
    char_counter = Counter([char for char, _, _ in all_chars])
    char_conf_avg = defaultdict(list)
    for char, prob, _ in all_chars:
        char_conf_avg[char].append(prob)
    
    # 信頼度別に平均信頼度を計算
    print("\n【文字別の認識信頼度（出現頻度上位30）】")
    print(f"{'順位':<6} {'文字':<5} {'出現数':<10} {'平均信頼度':<15} {'信頼度範囲':<25}")
    print("-" * 65)
    
    for i, (char, count) in enumerate(char_counter.most_common(30), 1):
        probs = char_conf_avg[char]
        avg_conf = statistics.mean(probs)
        min_conf = min(probs)
        max_conf = max(probs)
        conf_range = f"{min_conf:.4f}~{max_conf:.4f}"
        print(f"{i:<6} '{char}'    {count:<10} {avg_conf:<15.4f} {conf_range:<25}")
    
    # JSON形式で詳細情報を保存
    detailed_report = {
        "folder_count": len(folder_stats),
        "total_characters": len(all_chars),
        "low_confidence_count": len(low_confidence_predictions),
        "top_20_folders_by_char_count": [
            {
                "folder": name,
                "char_count": stats["char_count"],
                "avg_confidence": stats["avg_confidence"],
                "min_confidence": stats["min_confidence"],
                "max_confidence": stats["max_confidence"]
            }
            for name, stats in sorted_folders[:20]
        ],
        "worst_10_folders": [
            {
                "folder": name,
                "min_confidence": stats["min_confidence"],
                "avg_confidence": stats["avg_confidence"],
                "char_count": stats["char_count"]
            }
            for name, stats in worst_folders[:10]
        ],
        "low_confidence_predictions_sample": low_confidence_predictions[:30],
        "top_30_chars_by_frequency": [
            {
                "char": char,
                "count": count,
                "avg_confidence": statistics.mean(char_conf_avg[char]),
                "min_confidence": min(char_conf_avg[char]),
                "max_confidence": max(char_conf_avg[char])
            }
            for char, count in char_counter.most_common(30)
        ]
    }
    
    report_path = Path(output_dir) / "arc_detailed_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(detailed_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 詳細レポートを保存: {report_path}")
    
    # まとめ
    print("\n" + "=" * 80)
    print("まとめ")
    print("=" * 80)
    print(f"処理フォルダ数: {len(folder_stats)}")
    print(f"認識文字総数: {len(all_chars)}")
    print(f"低信頼度（<0.5）の認識: {len(low_confidence_predictions)} ({len(low_confidence_predictions)/len(all_chars)*100:.2f}%)")
    print(f"ユニークな文字種: {len(char_counter)}")

if __name__ == "__main__":
    generate_detailed_report()
