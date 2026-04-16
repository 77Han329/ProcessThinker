#!/usr/bin/env bash
set -x

# ==================== Environment setup ====================
# Note: inside the container we don't need module load; those go in the outer slurm script

export PATH=/usr/bin:$PATH

# ==================== Ray debug mode (uncomment to enable) ====================
# Enable Ray debug (comment out for production training)
# export RAY_DEBUG=1
# Do not deduplicate logs, show output from all workers (comment out for production)
# export RAY_DEDUP_LOGS=0

# ==================== Cache directories (avoid disk quota issues) ====================
export DECORD_EOF_RETRY_MAX=2048001
export WANDB_API_KEY='<YOUR_WANDB_API_KEY>'
export WANDB_ENTITY="<YOUR_WANDB_ENTITY>"  # <- set to your entity name
export HF_HOME=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.cache/hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export XDG_CACHE_HOME=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.cache
export XDG_CONFIG_HOME=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.config

# PyTorch cache directories (critical! avoids disk quota exceeded)
export TORCH_HOME=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.cache/torch_inductor
mkdir -p $TORCH_HOME $TORCHINDUCTOR_CACHE_DIR

# Python user base directory (for loading correct OpenCV version, etc.)
export PYTHONUSERBASE=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.local
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH

# Force using local model, do not check online
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ==================== Project configuration ====================
project_name='EasyR1-processthinker-rl'
exp_name='qwen3_vl_processthinker_rl'

MODEL_PATH="/hnvme/workspace/v100dd13-reasoning_model/Processthinker/LLaMA-Factory/models/Processthinker_SFT_8B_500steps"
TRAIN_FILE="/hnvme/workspace/v100dd13-reasoning_model/Processthinker/LLaMA-Factory/data/train_data/processthinker_rl_10k.json"
TEST_FILE="/hnvme/workspace/v100dd13-reasoning_model/Processthinker/LLaMA-Factory/data/train_data/processthinker_rl_10k.json"

# ==================== GPU configuration ====================
# 4-GPU layout: GPU 0 for vLLM (tensor_parallel=1), GPU 1,2,3 for training
# Launch vLLM with: CUDA_VISIBLE_DEVICES=0 --tensor-parallel-size 1
# Launch training with: CUDA_VISIBLE_DEVICES=1,2,3

ROLLOUT_BS=3  # batch size must be a multiple of the number of GPUs (3)
GLOBAL_BS=3
MB_PER_UPDATE=1
MB_PER_EXP=1
TP_SIZE=1  # tensor parallel size; raise to 2 or 3 if the model is too large
N_GPUS_PER_NODE=3  # train on 3 GPUs
NNODES=1

python3 -m verl.trainer.main \
    config=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/EasyR1/examples/config_processthinker_grpo.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=2e-6 \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    algorithm.filter_low=0.0 \
    algorithm.filter_high=1.0 \
    algorithm.online_filtering=false \
    algorithm.filter_key=accuracy \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=50 \
    trainer.save_checkpoint_path=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/EasyR1/checkpoints/
