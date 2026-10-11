#!/bin/bash
set -x
REPO=/home/imoto/Heritage_Pipeline
BASE="$REPO/tmp_work/bt_00003_2_crop"
CKPT_REL='../stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'

source /home/imoto/miniconda3/etc/profile.d/conda.sh
conda activate diffusion

# --- Wave1: v15背景1回目(GPU1) + DDRM外部参照(GPU0) 並列 ---
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
wait $PID_DDRM_EXT
echo "wave1 done"

cp "$BASE/v15_bg_out/200018243_00003_2.png" "$BASE/v15_pass2_input/"

# --- Wave2: v15背景2回目(GPU1) + DDRM+v15背景1回目(GPU0) 並列 ---
(
  cd "$REPO/modules/ddnm_npj"
  python scripts/pipeline_ddnm.py --ddrm_style --degradation_only --full_kept \
    --enhance --residual_thresh 10 --no_final_correction \
    --ckpt "$CKPT_REL" --num_res_blocks 2 --eta 0.5 \
    --input_dir "$BASE/v15_pass2_input" --output_dir "$BASE/v15_bg_out2" \
    --mask_dir "$BASE/mask_dir" --device cuda:1 --patch_size 128 --overlap 32 --ext .png
) > "$BASE/log_v15bg_pass2.txt" 2>&1 &
PID_V15BG2=$!

(
  cd "$REPO/modules/ddrm_npj_v2"
  python scripts/pipeline_ddrm.py \
    --input_dir "$BASE/input" --output_dir "$BASE/ddrm_v15bg_out" \
    --mask_dir "$BASE/mask_dir" --ckpt "$CKPT_REL" \
    --back_ground "$BASE/v15_bg_out/200018243_00003_2.png" \
    --device cuda:0 --patch_size 128 --overlap 32 --no_wandb
) > "$BASE/log_ddrm_v15bg.txt" 2>&1 &
PID_DDRM_V15=$!

wait $PID_V15BG2
wait $PID_DDRM_V15
echo "wave2 done"

# --- Wave3: DDNM本番+v15背景x2(GPU0) ---
(
  cd "$REPO/modules/ddnm_npj"
  python scripts/pipeline_ddnm.py --ddrm_style \
    --input_dir "$BASE/input" --output_dir "$BASE/ddnm_v15bgx2_out" \
    --mask_dir "$BASE/mask_dir" --back_ground "$BASE/v15_bg_out2/200018243_00003_2.png" \
    --missing_pull 0.6 --text_dark_value 30 --num_res_blocks 2 \
    --ckpt "$CKPT_REL" --device cuda:0 --patch_size 128 --overlap 32
) > "$BASE/log_ddnm_v15bgx2.txt" 2>&1 &
PID_DDNM_V15X2=$!
wait $PID_DDNM_V15X2
echo "wave3 done"

echo "ALL_DONE"
