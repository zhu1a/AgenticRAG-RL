#!/bin/bash
# Agentic GRPO 训练入口。模型、GPU、输出目录均可通过环境变量覆盖。
# 用法: VERL_DIR=/path/to/verl MODEL_PATH=/path/to/model bash training/start_grpo.sh
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
VERL_DIR=${VERL_DIR:?"请设置 VERL_DIR 指向 verl 安装目录"}
RETRIEVAL_SERVER_URL=${RETRIEVAL_SERVER_URL:-http://localhost:8790}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
N_GPUS=${N_GPUS:-$(printf '%s' "${TRAIN_GPUS}" | awk -F, '{print NF}')}

# ── 检查 retrieval server ──
if ! curl -fsS "${RETRIEVAL_SERVER_URL}/health" | grep -q "ok"; then
    echo "ERROR: Retrieval server not running at ${RETRIEVAL_SERVER_URL}."
    exit 1
fi
echo "Retrieval server OK"

# 默认从 SFT 合并模型继续训练；可用 MODEL_PATH 指定其他 checkpoint。
MODEL_PATH=${MODEL_PATH:-${PROJECT_DIR}/training/outputs/qwen3-4b-sft-zh-react-merged}
DATA_DIR=${PROJECT_DIR}/data/financial_eval
REWARD_FN=${PROJECT_DIR}/training/reward_agentic_rag.py
TOOL_CONFIG=${PROJECT_DIR}/training/config/financial_tool_config.yaml
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/training/outputs/grpo_agentic}
LOG=${LOG:-${PROJECT_DIR}/logs/grpo_agentic.log}

mkdir -p "${PROJECT_DIR}/logs"
for required_path in "${MODEL_PATH}" "${DATA_DIR}/grpo_agentic_train.parquet" "${DATA_DIR}/grpo_agentic_val.parquet"; do
    if [ ! -e "${required_path}" ]; then
        echo "ERROR: Required path does not exist: ${required_path}"
        exit 1
    fi
done
if [ -e "${OUTPUT_DIR}" ]; then
    echo "ERROR: Output directory already exists: ${OUTPUT_DIR}"
    echo "Set OUTPUT_DIR to a new path to avoid overwriting a previous run."
    exit 1
fi

export PATH="${VERL_PYTHON_DIR:-$(dirname "$(command -v python)")}:$PATH"
export ATTN_BACKEND=flash_attn
export PROJECT_DIR RETRIEVAL_SERVER_URL
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

cd "${VERL_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/grpo_agentic_train.parquet \
    data.val_files=${DATA_DIR}/grpo_agentic_val.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=7 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path=${REWARD_FN} \
    reward.custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=cv_agenticrag \
    trainer.experiment_name=qwen3_4b_grpo_agentic \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=3 \
    trainer.total_epochs=2 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    2>&1 | tee "${LOG}"
