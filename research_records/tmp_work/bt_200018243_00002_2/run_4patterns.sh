#!/bin/bash
set -x
REPO=/home/imoto/Heritage_Pipeline
BASE="$REPO/tmp_work/bt_200018243_00002_2"
CKPT_REL='../stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'

source /home/imoto/miniconda3/etc/profile.d/conda.sh
conda activate diffusion

# --- Wave1: v15背景生成(GPU1) + DDRM外部参照(GPU0) を並列 ---
(
  cd "$REPO/modules/ddnm_npj"
  python scripts/pipeline_ddnm.py --ddrm_style --degradation_only --full_kept \
    --enhance --residual_thresh 10 --no_final_correction \
    --ckpt "$CKPT_REL" --num_res_blocks 2 --eta 0.5 \
    --input_dir "$BASE/input" --output_dir "$BASE/v15_bg_out" \
    --mask_dir "$BASE/mask_dir" --device cuda:1 --patch_size 128 --overlap 32
) > "$BASE/log_v15bg.txt" 2>&1 &
PID_V15BG=$!

(
  cd "$REPO/modules/ddrm_npj_v2"
  python scripts/pipeline_ddrm.py \
    --input_dir "$BASE/input" --output_dir "$BASE/ddrm_extref_out" \
    --mask_dir "$BASE/mask_dir" --ckpt "$CKPT_REL" \
    --back_ground ./background/isemonogatari.png \
    --device cuda:0 --patch_size 128 --overlap 32 --no_wandb
) > "$BASE/log_ddrm_extref.txt" 2>&1 &
PID_DDRM_EXT=$!

wait $PID_V15BG
echo "v15bg done"
wait $PID_DDRM_EXT
echo "ddrm_extref done"

V15BG_IMG="$BASE/v15_bg_out/200018243_00002_2.png"
echo "V15BG_IMG=$V15BG_IMG"

# --- Wave2: DDRM+v15背景(GPU0) + DDNM本番+外部参照(GPU1) を並列 ---
(
  cd "$REPO/modules/ddrm_npj_v2"
  python scripts/pipeline_ddrm.py \
    --input_dir "$BASE/input" --output_dir "$BASE/ddrm_v15bg_out" \
    --mask_dir "$BASE/mask_dir" --ckpt "$CKPT_REL" \
    --back_ground "$V15BG_IMG" \
    --device cuda:0 --patch_size 128 --overlap 32 --no_wandb
) > "$BASE/log_ddrm_v15bg.txt" 2>&1 &
PID_DDRM_V15=$!

(
  cd "$REPO/modules/ddnm_npj"
  python scripts/pipeline_ddnm.py --ddrm_style \
    --input_dir "$BASE/input" --output_dir "$BASE/ddnm_extref_out" \
    --mask_dir "$BASE/mask_dir" --back_ground "$REPO/modules/ddrm_npj_v2/background/isemonogatari.png" \
    --missing_pull 0.6 --text_dark_value 30 --num_res_blocks 2 \
    --ckpt "$CKPT_REL" --device cuda:1 --patch_size 128 --overlap 32
) > "$BASE/log_ddnm_extref.txt" 2>&1 &
PID_DDNM_EXT=$!

wait $PID_DDRM_V15
echo "ddrm_v15bg done"
wait $PID_DDNM_EXT
echo "ddnm_extref done"

# --- Wave3: DDNM本番+v15背景(GPU0) ---
(
  cd "$REPO/modules/ddnm_npj"
  python scripts/pipeline_ddnm.py --ddrm_style \
    --input_dir "$BASE/input" --output_dir "$BASE/ddnm_v15bg_out" \
    --mask_dir "$BASE/mask_dir" --back_ground "$V15BG_IMG" \
    --missing_pull 0.6 --text_dark_value 30 --num_res_blocks 2 \
    --ckpt "$CKPT_REL" --device cuda:0 --patch_size 128 --overlap 32
) > "$BASE/log_ddnm_v15bg.txt" 2>&1 &
PID_DDNM_V15=$!
wait $PID_DDNM_V15
echo "ddnm_v15bg done"

echo "ALL_4_PATTERNS_DONE"
