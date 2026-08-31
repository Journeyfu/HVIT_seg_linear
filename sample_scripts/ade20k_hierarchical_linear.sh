#!/bin/bash

set -euo pipefail

ngpus="${NGPUS:-8}"
data_root="${ADE20K_ROOT:-data/ADEChallengeData2016}"

torchrun --nproc_per_node="${ngpus}" \
    evaluate_dense_tasks.py \
    evaluation/configs/pvtv2_b2_ade20k_linear_config.py \
    --backbone-type pvt_v2_b2.in1k \
    --task segmentation \
    --work-dir work_dirs/segmentation_eval/pvtv2_b2_ade20k_linear \
    --cfg-options \
        data.train.data_root="${data_root}" \
        data.val.data_root="${data_root}" \
        data.test.data_root="${data_root}"

torchrun --nproc_per_node="${ngpus}" \
    evaluate_dense_tasks.py \
    evaluation/configs/swin_base_ade20k_linear_config.py \
    --backbone-type swin_base_patch4_window7_224.ms_in1k \
    --task segmentation \
    --work-dir work_dirs/segmentation_eval/swin_base_ade20k_linear \
    --cfg-options \
        data.train.data_root="${data_root}" \
        data.val.data_root="${data_root}" \
        data.test.data_root="${data_root}"
