# set -x
# (bash /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/train.sh &)
ray stop --force
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9


cd /mnt/ali-sh-1/usr/huaan1/ocean/code/verl # Repo root

export MASTER_ADDR="10.148.2.255"
export VERIFIER_API_IP_ADDR="10.205.180.108"
# python ./verl/utils/reward_score/rel_label.py # 测试 verifier api

export http_proxy=10.140.24.177:3128
export https_proxy=10.140.24.177:3128
pip config set global.trusted-host "pypi.devops.xiaohongshu.com" && pip config set global.index-url "http://pypi.devops.xiaohongshu.com/simple"
pip install "ray[default,train,tune,serve]"
pip install torchdata
pip install wandb
pip install openai
pip install peft
export WANDB_API_KEY="0e112ac0ae8b584f4da8d11a5443a11f58c002a0"


export project_name='RankRL'
export exp_name='Qwen2p5_step2400-rl-v2_random_kl'

# Paths
export MODEL_PATH="/mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/Qwen2p5-32B-cpt-sft-v2-rl-v1_no_hard/global_step_2400/hf_model"
export CHECKPOINT_SAVE="/mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/$exp_name"
mkdir -p $CHECKPOINT_SAVE


all_course_filter_easy_hard="/mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/data/rl_v1_2w6.train.parquet"
all_course_filter_hard="/mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/data/rl_v2_2w9.train.parquet"
rl_v2="/mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/data/rl_v2_5w_random.train.parquet"
train_files="['$rl_v2']"

adv_estimator=grpo

total_epochs=3

kl_coef=0.0 # in-reward kl_penalty controller 

use_kl_loss=False # to use kl loss in actor. When used, we are not applying KL in the reward function.
kl_loss_coef=0.0 # The coefficient of kl loss. Default is 0.001.

max_prompt_length=$((1024 * 7))
max_response_length=$((1024 * 2))

loss_agg_mode="seq-mean-token-mean" # 每个序列内先归一化，每个序列等权重，def agg_loss

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
  --data.max_prompt_length ${max_prompt_length} \
  --data.max_response_length ${max_response_length} \
  --data.train_batch_size ${train_prompt_bsz} \
  --actor_rollout_ref.rollout.n ${n_resp_per_prompt} \
  --algorithm.adv_estimator ${adv_estimator} \
  --algorithm.kl_ctrl.kl_coef ${kl_coef} \
  --actor_rollout_ref.actor.use_kl_loss ${use_kl_loss} \
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
  --actor_rollout_ref.actor.optim.lr 8e-7 \
  --actor_rollout_ref.actor.optim.warmup_style "cosine" \
  --actor_rollout_ref.actor.optim.lr_warmup_steps_ratio 0.03 \
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
  --actor_rollout_ref.rollout.val_kwargs.temperature 0.6 \
  --actor_rollout_ref.rollout.val_kwargs.top_p 0.95 \
  --actor_rollout_ref.rollout.val_kwargs.top_k ${top_k} \
  --actor_rollout_ref.rollout.val_kwargs.do_sample True \
  --actor_rollout_ref.rollout.val_kwargs.n 1 \
  --actor_rollout_ref.ref.fsdp_config.param_offload ${offload} \
  --actor_rollout_ref.ref.ulysses_sequence_parallel_size 1 \
  --actor_rollout_ref.actor.fsdp_config.fsdp_size -1 \
  --reward_model.reward_manager "naive" \
  --trainer.project_name ${project_name} \
  --trainer.experiment_name ${exp_name} \
  --trainer.val_before_train True \
  --trainer.test_freq 100 \
  --trainer.save_freq 300 \
  --trainer.total_epochs ${total_epochs} \
  --trainer.default_local_dir ${CHECKPOINT_SAVE} \
  --trainer.resume_mode "disable" \
  --trainer.resume_from_path "" \
  --track_data_path "${CHECKPOINT_SAVE}/train_sample"


# --trainer.resume_mode disable, auto and resume_path
# https://verl.readthedocs.io/en/latest/advance/checkpoint.html#

# python scripts/model_merger.py merge \
#     --backend fsdp \
#     --local_dir /mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/Qwen2p5-rl-step1800-rl_no_hard_no_kl/global_step_1400/actor \
#     --target_dir /mnt/ali-sh-1/usr/huaan1/ocean/code/verl_checkpoint_save/Qwen2p5-rl-step1800-rl_no_hard_no_kl/global_step_1400/hf_model