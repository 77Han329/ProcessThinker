# ProcessThinker: Enhancing Multimodal LLM Reasoning via Rollout-based Process Reward

> Under review at ICLR 2026.

ProcessThinker provides **step-level process rewards without training a separate PRM**: for each intermediate step we sample several continuations from the current policy and use the empirical answer success rate as the step reward. Plugged into GRPO, it consistently improves Qwen3-VL-8B-Instruct on video reasoning.

<p align="center">
  <img src="assets/pipeline.png" alt="ProcessThinker pipeline" width="100%">
</p>

---

## 🧠 Method

Generations are constrained to `<think><step>...</step>...</think><answer>...</answer>`. Each step prefix is scored by the success rate of M=4 continuation rollouts; the average over the first K_max=6 steps is the process reward `R_proc`. Combined with a format/length reward, an outcome reward `R_acc`, and a bounded step-count bonus `B(K)`, the final reward is

```
r = (r_fmt + β) + λ_acc · R̄_acc + λ_proc · R̄_proc + B(K)
```

with penalty gating (`R̄_acc = 1` if correct else `-B(K)`; `R̄_proc = R_proc` if `≥ τ=0.5` else `-B(K)`) and `λ_acc + λ_proc = 1`. Invalid format ⇒ `r = 0` (and we skip the expensive rollouts). Full implementation: [`EasyR1/verl/reward_function/processthinker_reward.py`](EasyR1/verl/reward_function/processthinker_reward.py).

---

## 📊 Results

Accuracy (%) on four video reasoning benchmarks, starting from **Qwen3-VL-8B-Instruct**:

| Model | Video-MMMU | MMVU (mc) | VideoMathQA | LongVideoBench | Avg. |
|---|---:|---:|---:|---:|---:|
| Video-R1-7B                 | 53.89 | 65.92 | 26.67 | 58.30 | 51.20 |
| Qwen3-VL-8B-Instruct (base) | 62.89 | 65.60 | 25.20 | 71.50 | 56.30 |
| ProcessThinker-SFT          | 58.78 | 64.48 | 23.57 | 68.50 | 53.83 |
| ProcessThinker (outcome-only)       | 60.78 | 67.36 | 27.86 | 74.20 | 57.55 |
| ProcessThinker (outcome + process)  | 61.67 | 67.52 | 27.86 | 74.60 | 57.91 |
| **ProcessThinker (process-only)**   | **63.33** | **68.48** | **31.67** | **75.40** | **59.72** |

All variants share the same SFT warm-up; **process-only** (`λ_acc=0, λ_proc=1`) is the recommended default.

---

## 📁 Repo layout

```
.
├── LLaMA-Factory/                    # Stage 1 — SFT cold start
├── EasyR1/                           # Stage 2 — GRPO with process reward
│   └── verl/reward_function/processthinker_reward.py   # ← rollout-based process reward
├── Evaluation/                       # VLMEvalKit launcher
└── assets/
```

---

## 📐 Setup

```bash
# SFT env
conda create -n llamafactory python=3.11 -y && conda activate llamafactory
cd LLaMA-Factory && pip install -e ".[torch,metrics]" --no-build-isolation && cd ..

# RL env
conda create -n easyr1 python=3.11 -y && conda activate easyr1
cd EasyR1 && pip install -e . && cd ..
```

See upstream [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1](https://github.com/hiyouga/EasyR1) for env details.

---

## 🔍 Data

Two step-tagged splits, both video-only:

| Split | Size | Purpose |
|---|---:|---|
| `processthinker_sft_video` | **19k** | SFT cold start (registered in `LLaMA-Factory/data/dataset_info.json`) |
| `processthinker_rl_1250.json` | **1.25k** | GRPO prompts |

Derived from [VIDEO-R1-COT-165K](https://github.com/tulerfeng/Video-R1), rewritten into the `<step>` format by Qwen3-VL-30B-A3B-Instruct and filtered for answer fidelity, step–answer consistency, and step quality. Rewriting/filtering scripts are not part of this release.

Update local paths in: `LLaMA-Factory/examples/train_full/processthinker_sft.yaml`, `EasyR1/examples/config_processthinker_grpo.yaml`, `EasyR1/local_scripts/run_processthinker_rl.sh`.

---

## 🚀 Training

Recommended hardware: **8 × 80 GB GPUs** (e.g. H100).

**Stage 1 — SFT**
```bash
bash LLaMA-Factory/local_scripts/run_processthinker_sft.sh
```

**Stage 2 — GRPO with process reward**

> ⚠️ RL needs a **separate vLLM server** for the current policy (the reward function hits it for the M=4 continuations per step). Launch vLLM on its own GPU(s) **before** training with `--served-model-name qwen`, and point `process_model_endpoint` at it (default: `http://127.0.0.1:8000/v1/chat/completions`).

```bash
bash EasyR1/local_scripts/run_processthinker_rl.sh
```

Key knobs in `EasyR1/examples/config_processthinker_grpo.yaml`: `acc_weight` (λ_acc), `cot_weight` (λ_proc), `process_n` (M=4), `process_max_steps` (K_max=6), `step_min/step_max`, `l_min/l_max`, `alpha`, `penalty`, `process_model_endpoint`. ~200 GRPO steps already give strong performance.

---

## 🔮 Inference & Evaluation

```bash
# Four paper benchmarks via VLMEvalKit
bash Evaluation/VLMEvalKit/local_scripts/eval_vlmevalkit.sh
```

The shipped script also runs other benchmarks for completeness; only Video-MMMU, MMVU, VideoMathQA, and LongVideoBench are reported in the paper.

---

## 🙏 Acknowledgements

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [EasyR1](https://github.com/hiyouga/EasyR1) / [verl](https://github.com/volcengine/verl), [Video-R1](https://github.com/tulerfeng/Video-R1), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## 📜 Citation

```bibtex
@inproceedings{processthinker2026,
  title  = {ProcessThinker: Enhancing Multi-modal Large Language Models Reasoning via Rollout-based Process Reward},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review at ICLR}
}
```
