# set -x
# (bash /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/train_scripts/train_exp3_uniform.sh &)

ray stop --force
# ps -ef | grep bash | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9

cd /mnt/ali-sh-1/usr/huaan1/ocean/code/verl # Repo root

export MASTER_ADDR="10.148.19.72"
export WORLD_SIZE=5
export VERIFIER_API_IP_ADDR="10.205.180.108"
# python ./verl/utils/reward_score/rel_label.py # 测试 verifier api

export http_proxy=10.140.24.177:3128
export https_proxy=10.140.24.177:3128
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY="0e112ac0ae8b584f4da8d11a5443a11f58c002a0"

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
export VLLM_ATTENTION_BACKEND="XFORMERS" 
export GLOO_SOCKET_IFNAME="eth0"

pip config set global.trusted-host "pypi.devops.xiaohongshu.com" && pip config set global.index-url "http://pypi.devops.xiaohongshu.com/simple"
# USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh # 使用最新的 verl 依赖 for qwen3 ，vllm 0.8.5.post1
pip install "ray[default,train,tune,serve]"
pip install torchdata
pip install wandb
pip install openai
pip install peft


export project_name='RankRL'
export exp_name='GRM_exp3_grpo_redone_sft_v4_pev5_5w'

# Paths
export MODEL_PATH="/mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/models/redone_32B-rel_sft_v4_process_pev5_5w"
export CHECKPOINT_SAVE="/mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/$exp_name"
mkdir -p $CHECKPOINT_SAVE

uniform_pev0="/mnt/ali-sh-1/usr/huaan1/ocean/data/rl/rl_v3p1_2w8_uniform_biased.pev0.train.parquet"
uniform_pev5="/mnt/ali-sh-1/usr/huaan1/ocean/data/rl/rl_v3p1_2w8_uniform_biased.process.pev5.train.parquet"
random_pev0="/mnt/ali-sh-1/usr/huaan1/ocean/data/rl/rl_v3_4w9_random_biased.pev0.train.parquet"
random_pev5="/mnt/ali-sh-1/usr/huaan1/ocean/data/rl/rl_v3_4w9_random_biased.process.pev5.train.parquet"
vanilla_multi_epochs_pev5="/mnt/ali-sh-1/usr/huaan1/ocean/data/rl/relone_rl_v0_uniform3_random1.process.pev5.train.parquet"
extreme_multi_epochs_pev5="/mnt/ali-sh-1/usr/huaan1/ocean/data/rl/relone_rl_v1_3epochs_uniform_random.process.pev5.train.parquet"
train_files="['$uniform_pev5']"

adv_estimator=grpo

kl_coef=0.0 # in-reward kl_penalty controller 

use_kl_loss=True # to use kl loss in actor. When used, we are not applying KL in the reward function.
kl_loss_coef=0.001 # The coefficient of kl loss. Default is 0.001.

max_prompt_length=$((1024 * 7))
max_response_length=$((1024 * 2))

loss_agg_mode="seq-mean-token-mean" # GRPO: 每个序列内先归一化，每个序列等权重，def agg_loss
# loss_agg_mode="seq-mean-token-sum-norm" # Dr.GRPO: 用最大输出token数进行归一化

n_resp_per_prompt=8
train_prompt_bsz=40
train_prompt_mini_bsz=40

# off_policy paras
clip_ratio_low=0.2
clip_ratio_high=0.28

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
repetition_penalty=1.0

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( 2 * (max_prompt_length + max_response_length) ))
infer_ppo_max_token_len=$(( 3 * (max_prompt_length + max_response_length) ))

offload=True

bash scripts/run_dist.sh \
  --data.train_files "$train_files" \
  --data.prompt_key prompt \
  --data.filter_overlong_prompts True \
  --data.truncation 'error' \
  --data.shuffle False \
  --data.max_prompt_length ${max_prompt_length} \
  --data.max_response_length ${max_response_length} \
  --data.train_batch_size ${train_prompt_bsz} \
  --actor_rollout_ref.rollout.n ${n_resp_per_prompt} \
  --algorithm.adv_estimator ${adv_estimator} \
  --algorithm.kl_ctrl.kl_coef ${kl_coef} \
  --algorithm.dynamic_weighted_adv False \
  --algorithm.dynamic_weighted_adv_steps -1 \
  --actor_rollout_ref.actor.use_kl_loss ${use_kl_loss} \
  --actor_rollout_ref.actor.use_step_loss False \
  --actor_rollout_ref.actor.kl_loss_coef ${kl_loss_coef} \
  --actor_rollout_ref.actor.clip_ratio_low ${clip_ratio_low} \
  --actor_rollout_ref.actor.clip_ratio_high ${clip_ratio_high} \
  --actor_rollout_ref.model.use_remove_padding True \
  --actor_rollout_ref.actor.use_dynamic_bsz ${use_dynamic_bsz} \
  --actor_rollout_ref.ref.log_prob_use_dynamic_bsz ${use_dynamic_bsz} \
  --actor_rollout_ref.rollout.log_prob_use_dynamic_bsz ${use_dynamic_bsz} \
  --actor_rollout_ref.actor.ppo_max_token_len_per_gpu ${actor_ppo_max_token_len} \
  --actor_rollout_ref.ref.log_prob_max_token_len_per_gpu ${infer_ppo_max_token_len} \
  --actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu ${infer_ppo_max_token_len} \
  --actor_rollout_ref.model.path "${MODEL_PATH}" \
  --+actor_rollout_ref.model.override_config.attention_dropout 0. \
  --+actor_rollout_ref.model.override_config.embd_pdrop 0. \
  --+actor_rollout_ref.model.override_config.resid_pdrop 0. \
  --trainer.total_epochs 10 \
  --actor_rollout_ref.actor.optim.lr 1e-6 \
  --actor_rollout_ref.actor.optim.warmup_style "cosine" \
  --actor_rollout_ref.actor.optim.lr_warmup_steps 20 \
  --actor_rollout_ref.actor.optim.lr_warmup_steps_ratio -1 \
  --actor_rollout_ref.actor.optim.min_lr_ratio 0.0 \
  --actor_rollout_ref.actor.optim.num_cycles 0.5 \
  --actor_rollout_ref.actor.ppo_mini_batch_size ${train_prompt_mini_bsz} \
  --actor_rollout_ref.actor.fsdp_config.param_offload ${offload} \
  --actor_rollout_ref.actor.fsdp_config.optimizer_offload ${offload} \
  --actor_rollout_ref.actor.entropy_coeff 0 \
  --actor_rollout_ref.actor.grad_clip 1.0 \
  --actor_rollout_ref.actor.loss_agg_mode ${loss_agg_mode} \
  --actor_rollout_ref.actor.ulysses_sequence_parallel_size 1 \
  --actor_rollout_ref.rollout.gpu_memory_utilization 0.5 \
  --actor_rollout_ref.rollout.tensor_model_parallel_size 4 \
  --actor_rollout_ref.rollout.enable_chunked_prefill True \
  --actor_rollout_ref.rollout.max_num_batched_tokens $((max_prompt_length + max_response_length)) \
  --actor_rollout_ref.rollout.temperature ${temperature} \
  --actor_rollout_ref.rollout.top_p ${top_p} \
  --actor_rollout_ref.rollout.top_k ${top_k} \
  --actor_rollout_ref.rollout.repetition_penalty ${repetition_penalty} \
  --actor_rollout_ref.rollout.val_kwargs.temperature 0.6 \
  --actor_rollout_ref.rollout.val_kwargs.top_p 0.95 \
  --actor_rollout_ref.rollout.val_kwargs.top_k ${top_k} \
  --actor_rollout_ref.rollout.val_kwargs.repetition_penalty ${repetition_penalty}  \
  --actor_rollout_ref.rollout.val_kwargs.do_sample True \
  --actor_rollout_ref.rollout.val_kwargs.n 1 \
  --actor_rollout_ref.ref.fsdp_config.param_offload ${offload} \
  --actor_rollout_ref.ref.ulysses_sequence_parallel_size 1 \
  --actor_rollout_ref.actor.fsdp_config.fsdp_size -1 \
  --reward_model.reward_manager "naive" \
  --trainer.project_name ${project_name} \
  --trainer.experiment_name ${exp_name} \
  --trainer.val_before_train True \
  --trainer.test_freq 25 \
  --trainer.save_freq -1 \
  --trainer.default_local_dir ${CHECKPOINT_SAVE} \
  --trainer.resume_mode "disable" \
  --trainer.resume_from_path "" \
  --track_data_path "${CHECKPOINT_SAVE}/train_sample"


# --trainer.resume_mode disable, auto and resume_path
# https://verl.readthedocs.io/en/latest/advance/checkpoint.html#

# python scripts/model_merger.py merge \
#     --backend fsdp \
#     --local_dir /mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/exp1-7-redone-sft-v3-rl-v1_no_hard_biased_pev5_process_no_kl_wa_step1925_no_kl_no_wa/global_step_375/actor \
#     --target_dir /mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/exp1-7-redone-sft-v3-rl-v1_no_hard_biased_pev5_process_no_kl_wa_step1925_no_kl_no_wa/global_step_375/hf_model