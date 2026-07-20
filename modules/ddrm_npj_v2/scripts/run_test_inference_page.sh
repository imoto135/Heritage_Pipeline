#!/bin/bash
# Stage2 ページ推論: patch推論結果 → pageモデル → ガンマ補正
# pipeline_ddrm.py の出力を入力として使う（stain_removal論文の2段階手法）。
# GPU0とGPU1で並列実行する想定。
# 使い方:
#   bash run_test_inference_page.sh 0   # cuda:0 担当文献を処理
#   bash run_test_inference_page.sh 1   # cuda:1 担当文献を処理

GPU=${1:-0}
PYTHON=/home/imoto/miniconda3/envs/diffusion/bin/python
SCRIPT="$(dirname "$0")/pipeline_ddrm_page.py"
CKPT=/home/imoto/Heritage_Pipeline/modules/stain_removal/model/model_page/ckpt_epoch0400.pth
BG="$(dirname "$0")/../background/isemonogatari.png"
PATCH_OUT=/home/imoto/Heritage_Pipeline/tmp_work/ddrm_result   # patch推論の出力先
OUT=/home/imoto/Heritage_Pipeline/tmp_work/ddrm_page_result
TMPBASE=/tmp/ddrm_page_tmp_gpu${GPU}

# GPU0担当: 200003803 200004107 200005798 200008003 200010454 200015843
# GPU1担当: 200017458 200018243 200019865 200021063 200021071
if [ "$GPU" = "0" ]; then
    DOCS="200003803 200004107 200005798 200008003 200010454 200015843"
else
    DOCS="200017458 200018243 200019865 200021063 200021071"
fi

for DOC in $DOCS; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU${GPU} Stage2開始: ${DOC}"
    $PYTHON "$SCRIPT" \
        --patch_result_dir "$PATCH_OUT/$DOC" \
        --output_dir       "$OUT/$DOC" \
        --ckpt             "$CKPT" \
        --device           "cuda:${GPU}" \
        --back_ground      "$BG" \
        --tmp_dir          "${TMPBASE}_${DOC}" \
        --wandb_run_name   "ddrm-page-infer-otsu-ep400-gpu${GPU}-${DOC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU${GPU} Stage2完了: ${DOC}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU${GPU} 全文献Stage2処理完了"
