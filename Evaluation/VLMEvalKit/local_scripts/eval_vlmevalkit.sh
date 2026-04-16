#!/bin/bash

# 切换到 VLMEvalKit 目录
cd /hnvme/workspace/v100dd13-reasoning_model/Processthinker/Evaluation/VLMEvalKit

# VLMEvalKit 数据目录 - 使用 tmp 临时存储 (首次运行会自动下载)
# 注意：/tmp 和 /scratch 是同一个目录，任务结束后数据会被清空
export LMUData=/tmp/v100dd13_vlmeval_data

# 创建数据目录
mkdir -p $LMUData

# 模型名称 (已在 vlmeval/config.py 中注册)
MODEL=Processthinker-650steps

# 结果保存路径
WORK_BASE=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/Evaluation/VLMEvalKit/outputs/Rformat=R1_0.5+BETA_0.5-R2=acc_0.0+cot_1.0-penalty_true-stepbonus_true-ALPHA_0.2-l_min_220-l_max_420_650steps

# 创建输出目录
mkdir -p $WORK_BASE

DATASETS=(

  "MathVista_MINI"
  "MathVerse_MINI"
  "MMBench_DEV_EN"
  "MMMU_DEV_VAL"
  "MMStar"
  "AI2D_TEST"
  "ScienceQA_TEST"
  "MMT-Bench_VAL"
  "MMSci_DEV_Captioning_image_only"
  "MMVet"

  "Video_Holmes_128frame"
  'LongVideoBench_128frame'
  'Video_MMLU_CAP_128frame'


)


SUFFIX="results"

export CUDA_VISIBLE_DEVICES=0

for DATA in "${DATASETS[@]}"; do

  WORK_DIR="${WORK_BASE}/${SUFFIX}"

  python run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}" \
    --verbose \
    --reuse
done
