#!/bin/bash
# テストデータ全文献にDDRM推論を適用する。
# GPU0とGPU1で並列実行する想定。
# 使い方:
#   bash run_test_inference.sh 0   # cuda:0 担当文献を処理
#   bash run_test_inference.sh 1   # cuda:1 担当文献を処理

GPU=${1:-0}
PYTHON=/home/imoto/miniconda3/envs/diffusion/bin/python
SCRIPT="$(dirname "$0")/pipeline_ddrm.py"
CKPT=/home/imoto/Heritage_Pipeline/modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth
BG="$(dirname "$0")/../background/isemonogatari.png"
DATA=/home/imoto/Heritage_Pipeline/data/test
OUT=/home/imoto/Heritage_Pipeline/tmp_work/ddrm_result
TMPBASE=/tmp/ddrm_tmp_gpu${GPU}

# GPU0担当: 200003803 200004107 200005798 200008003 200010454 200015843
# GPU1担当: 200017458 200018243 200019865 200021063 200021071
if [ "$GPU" = "0" ]; then
    DOCS="200003803 200004107 200005798 200008003 200010454 200015843"
else
    DOCS="200017458 200018243 200019865 200021063 200021071"
fi

for DOC in $DOCS; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU${GPU} 開始: ${DOC}"
    $PYTHON "$SCRIPT" \
        --input_dir      "$DATA/$DOC" \
        --output_dir     "$OUT/$DOC" \
        --ckpt           "$CKPT" \
        --device         "cuda:${GPU}" \
        --back_ground    "$BG" \
        --patch_size     128 \
        --overlap        32 \
        --tmp_dir        "${TMPBASE}_${DOC}" \
        --wandb_run_name "ddrm-infer-otsu-ep350-gpu${GPU}-${DOC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU${GPU} 完了: ${DOC}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU${GPU} 全文献処理完了"
