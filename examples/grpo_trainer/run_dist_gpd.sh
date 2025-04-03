#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORK_DIR=`dirname $(dirname $SCRIPT_DIR)`
echo "WORK_DIR=$WORK_DIR"

# 配置环境变量 A800使用
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export TORCH_DISTRIBUTED_DEBUG=INFO
export GLOO_SOCKET_IFNAME=bond0
export TP_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8

# vllm缓存目录
export VLLM_CONFIG_ROOT=${CHECKPOINT_SAVE}
export VLLM_CACHE_ROOT=${CHECKPOINT_SAVE}

# Python环境
export PYTHONPATH="$(pwd):$PYTHONPATH"
# ray环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_MASTER_PORT=6379
# ray debug环境变量
export RAY_record_ref_creation_sites=1
export HYDRA_FULL_ERROR=1
export VERL_PPO_LOGGING_LEVEL=DEBUG

# 默认参数声明（可根据需要修改）
declare -A defaults=(
  ["algorithm.adv_estimator"]="grpo"
  ["data.train_files"]="${train_files}"
  ["data.val_files"]="${val_files}"
  ["data.train_batch_size"]=16
  ["data.val_batch_size"]=16
  ["data.max_prompt_length"]=400
  ["data.max_response_length"]=2048
  ["actor_rollout_ref.model.path"]="model_name_or_path"
  ["actor_rollout_ref.actor.optim.lr"]=3e-7
  ["actor_rollout_ref.model.use_remove_padding"]=True
  ["actor_rollout_ref.actor.ppo_mini_batch_size"]=256
  ["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"]=16
  ["actor_rollout_ref.actor.use_kl_loss"]=True
  ["actor_rollout_ref.actor.kl_loss_coef"]=0.001
  ["actor_rollout_ref.actor.kl_loss_type"]="low_var_kl"
  ["actor_rollout_ref.model.enable_gradient_checkpointing"]=True
  ["actor_rollout_ref.actor.fsdp_config.param_offload"]=True
  ["actor_rollout_ref.actor.fsdp_config.grad_offload"]=True
  ["actor_rollout_ref.actor.fsdp_config.optimizer_offload"]=True
  ["actor_rollout_ref.actor.use_dynamic_bsz"]=True
  ["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"]=24000
#  ["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"]=16
  ["actor_rollout_ref.rollout.tensor_model_parallel_size"]=2
  ["actor_rollout_ref.rollout.name"]="vllm"
  ["actor_rollout_ref.rollout.gpu_memory_utilization"]=0.6
  ["actor_rollout_ref.rollout.n"]=8
#  ["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"]=16
  ["actor_rollout_ref.ref.fsdp_config.param_offload"]=True
  ["algorithm.kl_ctrl.kl_coef"]=0.001
  ["trainer.critic_warmup"]=0
  ["tensorboard_log_dir"]="${CHECKPOINT_SAVE}/run"
  ["trainer.project_name"]="GRPO_logic_KK"
  ["trainer.experiment_name"]="Qwen-7B"
  ["trainer.default_local_dir"]="${CHECKPOINT_SAVE}"
  ["trainer.default_hdfs_dir"]="null"
  ["trainer.save_freq"]=20
  ["trainer.test_freq"]=-1
  ["trainer.total_epochs"]=5
)

declare -A user_params=()
other_args=()

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  if [[ $key == --* ]]; then
    param_name="${key#--}" # 移除--前缀

    if [[ -n "${defaults[$param_name]+x}" ]]; then
      # 处理已知参数
      shift
      value=""
      if [[ $# -gt 0 && $1 != --* ]]; then
        value="$1"
        shift
      fi
      user_params["$param_name"]="$value"
    else
      # 处理未知的--参数
      shift
      # 检查并添加可能的值
      if [[ $# -gt 0 && $1 != --* ]]; then
        user_params["$param_name"]="$1"
        shift
      else
        other_args+=("$key")
      fi
    fi
  else
    # 处理普通参数
    other_args+=("$key")
    shift
  fi
done

# 合并默认参数和用户参数
declare -A final_params
for key in "${!defaults[@]}"; do
  final_params["$key"]="${defaults[$key]}"
done
for key in "${!user_params[@]}"; do
  final_params["$key"]="${user_params[$key]}"
done

# 构建命令参数数组
cmd_args=()
for key in "${!final_params[@]}"; do
  cmd_args+=("$key=${final_params[$key]}")
done
cmd_args+=("${other_args[@]}")

# 准备huggingface缓存
export HF_MODULES_CACHE=/data_train/code/search/guopeidong/cache/modules
if [ "${CHECKPOINT_SAVE}" != "" ]; then
  export HF_MODULES_CACHE=${CHECKPOINT_SAVE}
fi

# 拷贝模型代码文件到缓存目录
if [ ${RANK} == 0 ]; then
  dirname=$(basename ${final_params["actor_rollout_ref.model.path"]})
  module_cache_dir=${HF_MODULES_CACHE}/transformers_modules/${dirname}
  if [ ! -d "${module_cache_dir}" ]; then
    mkdir -p ${module_cache_dir}
    touch ${HF_MODULES_CACHE}/transformers_modules/__init__.py
    touch ${module_cache_dir}/__init__.py
    cp ${final_params["actor_rollout_ref.model.path"]}/*py ${module_cache_dir}
  fi
fi

touch /checkpoint_save/_worker_${RANK}_ready
# 等待所有worker就绪
while true; do
  count=0 # 初始化就绪计数器
  # 遍历所有worker编号
  for ((x = 0; x < WORLD_SIZE; x++)); do
    # 检测对应worker的就绪文件
    if [[ -f "/checkpoint_save/_worker_${x}_ready" ]]; then
      ((count++)) # 存在则计数器+1
    fi
  done
  # 显示实时进度
  echo "Progress: ${count}/${WORLD_SIZE} workers ready"

  # 判断是否全部就绪
  if [[ $count -eq ${WORLD_SIZE} ]]; then
    break # 满足条件退出循环
  else
    sleep 5 # 等待5秒后再次检查
  fi
done

which python
set -x

if [ ${RANK} == 0 ]; then
  # 启动ray集群head节点
  ray start --head --port=${RAY_MASTER_PORT} --dashboard-host 0.0.0.0
  # 提交ray任务
  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "'$WORK_DIR'"}' \
    -- python ./run.py ${cmd_args[@]} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.logger=['console','tensorboard'] \
    +trainer.val_before_train=True 2>&1 | tee ${CHECKPOINT_SAVE}/run.log
  # 停止ray集群
  ray stop --force
else
  sleep 20s
  # 启动ray集群
  ray start --address ${MASTER_ADDR}:${RAY_MASTER_PORT} --num-gpus 8 --block
fi

