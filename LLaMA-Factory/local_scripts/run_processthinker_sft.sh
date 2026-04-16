export OMP_NUM_THREADS=8
export DECORD_EOF_RETRY_MAX=2048001
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4  # must match the GPU count in the SLURM config
export PYTHONNOUSERSITE=1
export PYTHONPATH=

# Redirect all cache directories to workspace (avoids home directory quota overflow)
export HF_HOME=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.cache/huggingface
export HF_DATASETS_CACHE=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.cache/huggingface/transformers
export TORCH_HOME=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.cache/torch
export XDG_CACHE_HOME=/hnvme/workspace/v100dd13-reasoning_model/Processthinker/.cache

# Create cache directories
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE $TORCH_HOME

llamafactory-cli train LLaMA-Factory/examples/train_full/processthinker_sft.yaml
