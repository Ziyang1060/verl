export http_proxy=http://172.23.3.11:3128
export https_proxy=http://172.23.3.11:3128
pip install torchdata
pip install pylatexenc
pip install vertexai
pip install -U sentence-transformers
unset http_proxy https_proxy

# if [ ${RANK} == 0 ]; then
#   tensorboard --logdir ${CHECKPOINT_SAVE} --host 0.0.0.0 --port 6008 > tensorboard.log 2>&1 &
#   streamlit run ${CODE_PATH}/rl-board/rl_logging_board.py  --server.port 8901 > rlboard.log 2>&1 &
# fi

cd /global_data/med/zengziyang/verl # Repo root

set -euxo pipefail

project_name='DAPO'
exp_name='DAPO-Qwen2.5-32B'

adv_estimator=grpo

kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=256
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=16
train_prompt_mini_bsz=16

# Paths
TRAIN_FILE=${TRAIN_FILE:-"/global_data/med/zengziyang/verl_data/verl/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/global_data/med/zengziyang/verl_data/verl/data/deepscaler/aime.parquet"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_LOAD}/Qwen2p5-32B"}
CKPTS_DIR=${CHECKPOINT_SAVE:-"/global_data/med/zengziyang/verl_data/verl/checkpoints/"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=4
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4

bash recipe/dapo/run_dist_dapo.sh \
  --data.train_files "${TRAIN_FILE}" \
  --data.val_files "${TEST_FILE}" \
  --data.prompt_key prompt \
  --data.truncation 'left' \
  --data.max_prompt_length ${max_prompt_length} \
  --data.max_response_length ${max_response_length} \
  --data.gen_batch_size ${gen_prompt_bsz} \
  --data.train_batch_size ${train_prompt_bsz} \
  --actor_rollout_ref.rollout.n ${n_resp_per_prompt} \
  --algorithm.adv_estimator ${adv_estimator} \
  --algorithm.kl_ctrl.kl_coef ${kl_coef} \
  --actor_rollout_ref.actor.use_kl_loss ${use_kl_loss} \
  --actor_rollout_ref.actor.kl_loss_coef ${kl_loss_coef} \
  --actor_rollout_ref.actor.clip_ratio_low ${clip_ratio_low} \
  --actor_rollout_ref.actor.clip_ratio_high ${clip_ratio_high} \
  --algorithm.filter_groups.enable ${enable_filter_groups} \
  --algorithm.filter_groups.max_num_gen_batches ${max_num_gen_batches} \
  --algorithm.filter_groups.metric ${filter_groups_metric} \
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
  --actor_rollout_ref.model.enable_gradient_checkpointing True \
  --actor_rollout_ref.actor.optim.lr 1e-6 \
  --actor_rollout_ref.actor.optim.lr_warmup_steps 10 \
  --actor_rollout_ref.actor.optim.weight_decay 0.1 \
  --actor_rollout_ref.actor.ppo_mini_batch_size ${train_prompt_mini_bsz} \
  --actor_rollout_ref.actor.fsdp_config.param_offload ${offload} \
  --actor_rollout_ref.actor.fsdp_config.optimizer_offload ${offload} \
  --actor_rollout_ref.actor.entropy_coeff 0 \
  --actor_rollout_ref.actor.grad_clip 1.0 \
  --actor_rollout_ref.actor.loss_agg_mode ${loss_agg_mode} \
  --actor_rollout_ref.actor.ulysses_sequence_parallel_size ${sp_size} \
  --actor_rollout_ref.rollout.gpu_memory_utilization 0.80 \
  --actor_rollout_ref.rollout.tensor_model_parallel_size ${gen_tp} \
  --actor_rollout_ref.rollout.enable_chunked_prefill True \
  --actor_rollout_ref.rollout.max_num_batched_tokens $((max_prompt_length + max_response_length)) \
  --actor_rollout_ref.rollout.temperature ${temperature} \
  --actor_rollout_ref.rollout.top_p ${top_p} \
  --actor_rollout_ref.rollout.top_k "${top_k}" \
  --actor_rollout_ref.rollout.val_kwargs.temperature ${temperature} \
  --actor_rollout_ref.rollout.val_kwargs.top_p ${top_p} \
  --actor_rollout_ref.rollout.val_kwargs.top_k ${top_k} \
  --actor_rollout_ref.rollout.val_kwargs.do_sample True \
  --actor_rollout_ref.rollout.val_kwargs.n 32\
  --actor_rollout_ref.ref.fsdp_config.param_offload ${offload} \
  --actor_rollout_ref.ref.ulysses_sequence_parallel_size ${sp_size} \
  --actor_rollout_ref.actor.fsdp_config.fsdp_size -1 \
  --reward_model.reward_manager dapo \
  --custom_reward_function.overlong_buffer.enable ${enable_overlong_buffer} \
  --custom_reward_function.overlong_buffer.len ${overlong_buffer_len} \
  --custom_reward_function.overlong_buffer.penalty_factor ${overlong_penalty_factor} \
  --trainer.project_name "${project_name}" \
  --trainer.experiment_name "${exp_name}" \
  --+trainer.val_before_train True \
  --trainer.test_freq 5 \
  --trainer.save_freq 5 \
  --trainer.total_epochs 1 \
  --trainer.default_local_dir "${CKPTS_DIR}" \
  --trainer.resume_mode auto \
  --track_data_path ${CHECKPOINT_SAVE}/train_sample
