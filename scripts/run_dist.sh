#!/usr/bin/env bash
# set -euxo pipefail
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
export RUNTIME_ENV="./verl/trainer/runtime_env.yaml"

# Ray
export RAY_MASTER_PORT=6379
RAY_ADDRESS=${RAY_ADDRESS:-"http://${MASTER_ADDR}:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
echo "WORKING_DIR=$WORKING_DIR"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}


# 默认参数声明（可根据需要修改）
declare -A defaults=(
    ["algorithm.adv_estimator"]="grpo"
)

declare -A ray_start_params=(
  ["metrics-export-port"]=20100
  ["runtime-env-agent-port"]=20101
  ["dashboard-agent-grpc-port"]=20102
  ["dashboard-agent-listen-port"]=20103
)

ray_start_args=()
for key in "${!ray_start_params[@]}"; do
  ray_start_args+=("--$key" "${ray_start_params[$key]}")
done

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

# # 准备huggingface缓存
# export HF_MODULES_CACHE=/data_train/code/search/guopeidong/cache/modules
# if [ "${CHECKPOINT_SAVE}" != "" ]; then
#   export HF_MODULES_CACHE=${CHECKPOINT_SAVE}
# fi

# # 拷贝模型代码文件到缓存目录
# if [ ${RANK} == 0 ]; then
#   dirname=$(basename ${final_params["actor_rollout_ref.model.path"]})
#   module_cache_dir=${HF_MODULES_CACHE}/transformers_modules/${dirname}
#   if [ ! -d "${module_cache_dir}" ]; then
#     mkdir -p ${module_cache_dir}
#     touch ${HF_MODULES_CACHE}/transformers_modules/__init__.py
#     touch ${module_cache_dir}/__init__.py
#     cp ${final_params["actor_rollout_ref.model.path"]}/*py ${module_cache_dir}
#   fi
# fi

which python
set -x

if [ ${RANK} == 0 ]; then
  # 启动ray集群head节点
  ray start --head --port=${RAY_MASTER_PORT} --dashboard-host 0.0.0.0 ${ray_start_args[@]}
  while true; do
    count=`python scripts/ray_available_node_count.py`
    # 判断ray是否和worker connected
    if [[ $count -eq ${WORLD_SIZE} ]]; then
      break # 全部就绪继续执行
    else
      echo "重试检查所有 GPU 结点就绪"
      sleep 5s
    fi
  done
  
  # 提交ray任务
  ray job submit --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    --address="${RAY_ADDRESS}" \
    -- python -u -m verl.trainer.main_ppo ${cmd_args[@]} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    2>&1 | tee -a "${CHECKPOINT_SAVE}/run_`hostname`.log"
  exit_code=${PIPESTATUS[0]}
  # 停止ray集群
  ray stop --force
  exit ${exit_code}
else
  # 启动ray集群
  ray start --address ${MASTER_ADDR}:${RAY_MASTER_PORT} --num-gpus 8 --block ${ray_start_args[@]}
fi