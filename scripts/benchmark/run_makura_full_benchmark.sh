#!/bin/bash
# 枕草子20ページ×4強度 本番一括実行スクリプト
#
# DDRM(v2, CSC修正版・外部参照isemonogatari.png)とDDNM(v15完成版レシピ)を
# 強度ごとにGPU0/GPU1で並列実行し、全強度が終わったらevaluate_restoration.pyで
# PSNR/SSIM/LPIPS/CIEDE2000/stain-F1を計算する。
#
# 前提: tmp_work/makura_benchmark/{degraded,masks,clean,stain_gt}が
# scripts/benchmark/generate_makura_benchmark.pyで生成済みであること。
#
# 実行時間の目安: 1ページあたりDDRM約5分・DDNM約4.5分（page 2387x3212規模の実測値）。
# 20ページ×4強度をGPU0(DDRM)/GPU1(DDNM)並列で回すと、強度1つあたり約1.8時間、
# 全体で約7時間程度（試走2回の実測に基づく見積もり。実測次第で前後する）。
#
# 使い方:
#   nohup bash scripts/benchmark/run_makura_full_benchmark.sh > tmp_work/makura_full_run.log 2>&1 &
#   （手動実行。cron等では起動しない）

REPO=/home/imoto/Heritage_Pipeline
BENCH="$REPO/tmp_work/makura_benchmark"
OUT="$REPO/tmp_work/makura_full_run"
CKPT_REL='../stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'
INTENSITIES="0.30 0.50 0.70 1.00"

source /home/imoto/miniconda3/etc/profile.d/conda.sh
conda activate diffusion

mkdir -p "$OUT"
FAILED=0

for I in $INTENSITIES; do
    DEG_DIR="$BENCH/degraded/intensity_$I"
    MASK_DIR="$BENCH/masks/intensity_$I"
    DDRM_RAW="$OUT/ddrm_raw/intensity_$I"
    DDRM_FLAT="$OUT/ddrm/intensity_$I"
    DDNM_OUT="$OUT/ddnm/intensity_$I"

    if [ ! -d "$DEG_DIR" ] || [ ! -d "$MASK_DIR" ]; then
        echo "[SKIP] intensity=$I: 入力またはマスクが見つかりません ($DEG_DIR / $MASK_DIR)"
        FAILED=1
        continue
    fi

    mkdir -p "$DDRM_RAW" "$DDRM_FLAT" "$DDNM_OUT"

    echo "=== intensity=$I: DDRM(GPU0)とDDNM(GPU1)を並列実行開始 $(date '+%Y-%m-%d %H:%M:%S') ==="

    (
        cd "$REPO/modules/ddrm_npj_v2" && \
        python scripts/pipeline_ddrm.py \
            --input_dir "$DEG_DIR" \
            --output_dir "$DDRM_RAW" \
            --mask_dir "$MASK_DIR" \
            --ckpt "$CKPT_REL" \
            --back_ground ./background/isemonogatari.png \
            --device cuda:0 --patch_size 128 --overlap 32 --no_wandb
    ) > "$OUT/ddrm_intensity_${I}.log" 2>&1 &
    DDRM_PID=$!

    (
        cd "$REPO/modules/ddnm_npj" && \
        python scripts/pipeline_ddnm.py --ddrm_style --degradation_only --full_kept \
            --enhance --residual_thresh 10 --no_final_correction \
            --ckpt "$CKPT_REL" --num_res_blocks 2 --eta 0.5 \
            --input_dir "$DEG_DIR" \
            --output_dir "$DDNM_OUT" \
            --mask_dir "$MASK_DIR" \
            --device cuda:1 --patch_size 128 --overlap 32 --ext .png
    ) > "$OUT/ddnm_intensity_${I}.log" 2>&1 &
    DDNM_PID=$!

    wait "$DDRM_PID"; DDRM_STATUS=$?
    wait "$DDNM_PID"; DDNM_STATUS=$?

    echo "    DDRM intensity=$I 終了 (exit=$DDRM_STATUS)"
    echo "    DDNM intensity=$I 終了 (exit=$DDNM_STATUS)"
    if [ "$DDRM_STATUS" -ne 0 ] || [ "$DDNM_STATUS" -ne 0 ]; then
        FAILED=1
    fi

    # DDRMの出力は <out>/<stem>/result/<stem>.png にネストしているので平坦化する
    n_flat=0
    for d in "$DDRM_RAW"/*/; do
        stem=$(basename "$d")
        if [ -f "$d/result/$stem.png" ]; then
            cp "$d/result/$stem.png" "$DDRM_FLAT/$stem.png"
            n_flat=$((n_flat + 1))
        fi
    done
    echo "    DDRM結果を平坦化: ${n_flat}枚 -> $DDRM_FLAT"
done

echo "=== 全強度の復元が完了 $(date '+%Y-%m-%d %H:%M:%S')。指標を計算します ==="

for I in $INTENSITIES; do
    DDRM_FLAT="$OUT/ddrm/intensity_$I"
    DDNM_OUT="$OUT/ddnm/intensity_$I"
    if [ ! -d "$DDRM_FLAT" ] || [ ! -d "$DDNM_OUT" ]; then
        echo "[SKIP] intensity=$I: 復元結果が見つからないため指標計算をスキップ"
        continue
    fi
    python "$REPO/scripts/benchmark/evaluate_restoration.py" \
        --gt_dir "$BENCH/clean" \
        --stain_gt_dir "$BENCH/stain_gt/intensity_$I" \
        --methods degraded="$BENCH/degraded/intensity_$I" \
                  ddrm="$DDRM_FLAT" \
                  ddnm="$DDNM_OUT" \
        --out "$OUT/metrics_intensity_$I.csv" \
        2>&1 | tee "$OUT/eval_intensity_${I}.log"
done

echo "=== 完了 $(date '+%Y-%m-%d %H:%M:%S')。結果: $OUT/metrics_intensity_*.csv ==="
if [ "$FAILED" -ne 0 ]; then
    echo "[WARN] 途中でエラーがありました。各intensityのログ($OUT/ddrm_intensity_*.log, ddnm_intensity_*.log)を確認してください。"
    exit 1
fi
