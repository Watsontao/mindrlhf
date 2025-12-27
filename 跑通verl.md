跑通verl-Ascend

## 环境准备

```shell
# source ~/.bashrc

# conda init 

# conda create -n test01 python=3.10 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

# conda activate test01

# vllm
# git clone -b v0.11.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm

VLLM_TARGET_DEVICE=empty pip install .

cd ../

# vllm-ascend
# git clone -b v0.11.0rc1 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

pip install  -e .

cd ../

# 安装verl
# git clone https://github.com/volcengine/verl.git
cd verl
pip install -r requirements-npu.txt
pip install -e .


# 配置mindspeed
cd ../
# git clone https://gitcode.com/Ascend/MindSpeed.git
pip install -e MindSpeed

# git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1


# python -c "from vllm import LLM; print('恭喜！vLLM 导入成功！')"


# export VLLM_ASCEND_ENABLE_NZ=0

# pip install torch==2.7.1 torch-npu==2.7.1 torchvision

# 跑one step off policy 和 full async rollout需要打ray的patch补丁，这里需要ray==2.51.1
# ray_dir=$(python -c 'import ray; import os; print(os.path.dirname(ray.__file__))')
# patch -p1 -d ${ray_dir} < ray_2_51_1_hccl.patch
```



## quick start

### 1. 下载数据和模型

```shell
# 如果遇到 ImportError: /lib/aarch64-linux-gnu/libstdc++.so.6:
# version `CXXABI_1.3.15' not found

# 试着export LD_LIBRARY_PATH=/home/ma-user/.conda/envs/test01/lib:$LD_LIBRARY_PATH
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k


pip install modelscope
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct  --local_dir   /home/ma-user/work/models/Qwen2.5-0.5B-Instruct
```

### 2. 执行训练

```shell
set -x

export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/ma-user/work/data/gsm8k/train.parquet \
    data.val_files=/home/ma-user/work/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/ma-user/work/models/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.device=npu $@
```

### 3. 异步

```shell
set -x

export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/ma-user/work/data/gsm8k/train.parquet \
    data.val_files=/home/ma-user/work/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/ma-user/work/models/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.device=npu $@ 
```



4. **加上了segment rollout**

```shell
set -x

export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/ma-user/work/data/gsm8k/train.parquet \
    data.val_files=/home/ma-user/work/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/ma-user/work/models/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.device=npu \
    +actor_rollout_ref.rollout.segment.enable=True \
    +actor_rollout_ref.rollout.segment.segment_length=1024
```

### 4. one step off policy

```shell
# grpo_qwen3_8b_gsm8k_fsdp2_8_8_npu.sh
# The script has been validated on the Ascend Atlas 800T A3.
set -x

export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_EXEC_TIMEOUT=60000
export HCCL_CONNECT_TIMEOUT=7200

project_name='GRPO'
exp_name='GRPO-Qwen2.5-0.5b-gsm8k-2cards'


RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/work/verl"}

MODEL_PATH="/home/ma-user/work/models/Qwen2.5-0.5B-Instruct"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE="/home/ma-user/work/data/gsm8k/train.parquet"
TEST_FILE="/home/ma-user/work/data/gsm8k/test.parquet"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

n_gpus_rollout=1
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 16))

use_dynamic_bsz=True
sp_size=1
fsdp_size=1
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))

python3 -m recipe.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.truncation='error' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.nccl_timeout=14400 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.device=npu \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" $@
```

### 5. fully_async_policy

#### 5.1 fsdp

```shell
#!/usr/bin/env bash

project_name='GRPO-Qwen2.5-0.5b-Base-MATH'
exp_name='GRPO-Qwen2.5-0.5b-Base-MATH-2gpu-async'

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH="/home/ma-user/work/models/Qwen2.5-0.5B-Instruct"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE="/home/ma-user/work/data/gsm8k/train.parquet"
TEST_FILE="/home/ma-user/work/data/gsm8k/test.parquet"



rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*400)))
test_freq=10
staleness_threshold=0
trigger_parameter_sync_step=16
partial_rollout=False


python -m recipe.fully_async_policy.fully_async_main \
	train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.partial_rollout="${partial_rollout}"
```

test

```shell

```



#### 5.2 megatron

```shell
#!/usr/bin/env bash
set -xuo pipefail  # 去掉e，不关闭shell

project_name='GRPO-Qwen2.5-0.5b-Base-MATH'
exp_name='GRPO-Qwen2.5-0.5b-Base-MATH-2gpu-async'

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH="/home/ma-user/work/models/Qwen2.5-0.5B-Instruct"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE="/home/ma-user/work/data/gsm8k/train.parquet"
TEST_FILE="/home/ma-user/work/data/gsm8k/test.parquet"

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi
# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
offload=false # 小模型 关闭
train_ppo_micro_batch_size_per_gpu=2
infer_ppo_micro_batch_size_per_gpu=2

optimizer_offload_fraction=0

COMMON_PP=1
COMMON_VPP=null  #设置为了null
COMMON_CP=1
COMMON_TP=1
COMMON_EP=1
COMMON_ETP=1

TRAIN_TP=1
INFER_TP=1

# === 强制使用脚本内定义的配置，忽略外部环境变量干扰 ===
ACTOR_PP=$COMMON_PP
ACTOR_VPP=$COMMON_VPP
ACTOR_CP=$COMMON_CP
ACTOR_TP=$TRAIN_TP
ACTOR_EP=$COMMON_EP
ACTOR_ETP=$COMMON_ETP

ROLLOUT_TP=$INFER_TP

REF_PP=$COMMON_PP
REF_VPP=$COMMON_VPP
REF_CP=$COMMON_CP
REF_TP=$TRAIN_TP
REF_EP=$COMMON_EP
REF_ETP=$COMMON_ETP

CRITIC_PP=$COMMON_PP
CRITIC_VPP=$COMMON_VPP
CRITIC_CP=$COMMON_CP
CRITIC_TP=$TRAIN_TP
CRITIC_EP=$COMMON_EP
CRITIC_ETP=$COMMON_ETP

RM_PP=$COMMON_PP
RM_VPP=$COMMON_VPP
RM_CP=$COMMON_CP
RM_TP=$TRAIN_TP
RM_EP=$COMMON_EP
RM_ETP=$COMMON_ETP
# ====================================================

# install mbridge
# pip3 install git+https://github.com/ISEEKYAN/mbridge
USE_MBRIDGE=False
USE_DIST_CKPT=False

# Fully async specific parameters
NNODES_ROLLOUT=1
NNODES_TRAIN=1
NGPUS_PER_NODE=1

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=4
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*100)))
test_freq=20
staleness_threshold=0.5
trigger_parameter_sync_step=4
require_batches=1
partial_rollout=True

python -m recipe.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml'\
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    
    
    # 因为Qwen2.5Dense（稠密）模型，不是 MoE 模型，所以删掉下面的参数
    # +acto_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="flex" \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \  
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.device=npu \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs=10 \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=True \


```





## 附录

### 附录1 环境准备（旧）

```shell
# vllm
git clone -b v0.11.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/build.txt

# for Atlas 900 A2 PODc or Atlas 800T A3
VLLM_TARGET_DEVICE=empty pip install -e .

cd ../

# vllm-ascend
git clone -b v0.11.0rc1 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
#===================
# 安装下面这个pip install -e .的时候会报错，原因是因为当前的操作系统默认 C++ 编译器版本是 GCC 7.3.0，但是你安装的 PyTorch (2.6.0) 必须要求 GCC 9 或更高版本 才能编译扩展插件（因为使用了 C++17 标准），详细见附录1

# 因此执行下面这个命令
# conda install -c conda-forge gxx_linux-aarch64=11 gcc_linux-aarch64=11 sysroot_linux-aarch64=2.17
conda install -c conda-forge gcc_linux-aarch64=11 gxx_linux-aarch64=11
# 注意py310这个名字是python虚拟环境的名字
export CC=/home/ma-user/anaconda3/envs/Pytorch-2.6.0/bin/aarch64-conda-linux-gnu-gcc
export CXX=/home/ma-user/anaconda3/envs/Pytorch-2.6.0/bin/aarch64-conda-linux-gnu-g++
#===================

pip install -e .

cd ../

# 安装verl
git clone https://github.com/volcengine/verl.git
cd verl
pip install -r requirements-npu.txt
pip install -e .
```



### 附录2

**grpo_qwen3_8b_gsm8k_fsdp2_8_8_npu.sh 但是会报错**，报错如下：

```shell
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=/home/ma-user/work/data/gsm8k/train.parquet', 'data.val_files=/home/ma-user/work/data/gsm8k/test.parquet', 'data.train_batch_size=32', 'data.max_prompt_length=2048', 'data.max_response_length=32768', 'data.filter_overlong_prompts=True', 'data.filter_overlong_prompts_workers=64', 'data.truncation=error', 'actor_rollout_ref.actor.strategy=fsdp2', 'critic.strategy=fsdp2', 'actor_rollout_ref.model.path=/home/ma-user/work/models/Qwen2.5-0.5B-Instruct', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.hybrid_engine=False', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=32', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1', 'actor_rollout_ref.actor.use_kl_loss=False', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.actor.entropy_coeff=0', 'actor_rollout_ref.actor.use_torch_compile=False', 'actor_rollout_ref.ref.use_torch_compile=False', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.fsdp_config.param_offload=True', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=True', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1', 'actor_rollout_ref.rollout.tensor_model_parallel_size=1', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.mode=async', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.8', 'actor_rollout_ref.rollout.max_num_batched_tokens=34816', 'actor_rollout_ref.rollout.n=8', 'actor_rollout_ref.rollout.enforce_eager=True', 'actor_rollout_ref.rollout.load_format=safetensors', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1', 'actor_rollout_ref.ref.fsdp_config.param_offload=True', 'actor_rollout_ref.actor.fsdp_config.fsdp_size=1', 'actor_rollout_ref.actor.use_dynamic_bsz=True', 'actor_rollout_ref.actor.ppo_max_token_len_per_gpu=34816', 'actor_rollout_ref.actor.ulysses_sequence_parallel_size=1', 'actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True', 'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=34816', 'actor_rollout_ref.ref.ulysses_sequence_parallel_size=1', 'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True', 'algorithm.use_kl_in_reward=False', 'actor_rollout_ref.nccl_timeout=14400', 'trainer.critic_warmup=0', 'trainer.val_before_train=False', 'trainer.logger=[console,tensorboard]', 'trainer.project_name=GRPO', 'trainer.experiment_name=GRPO-Qwen2.5-0.5b-gsm8k-2cards', 'trainer.default_local_dir=/home/ma-user/work/verl/ckpts/GRPO/GRPO-Qwen2.5-0.5b-gsm8k-2cards', 'trainer.save_freq=10', 'trainer.test_freq=-1', 'trainer.total_epochs=15', 'trainer.resume_mode=auto', 'trainer.nnodes=1', 'trainer.device=npu', 'trainer.n_gpus_per_node=1', 'rollout.nnodes=1', 'rollout.n_gpus_per_node=1']
Traceback (most recent call last):
  File "/home/ma-user/work/verl/recipe/one_step_off_policy/main_ppo.py", line 230, in main
    run_ppo(config, task_runner_class=OneStepTaskRunner)
  File "/home/ma-user/work/verl/verl/trainer/main_ppo.py", line 99, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/worker.py", line 2967, in get
    values, debugger_breakpoint = worker.get_objects(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/worker.py", line 1015, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(RuntimeError): ray::OneStepTaskRunner.run() (pid=288089, ip=172.16.0.134, actor_id=6e88e9b4e4bf0f6489bfdeeb01000000, repr=<main_ppo.OneStepTaskRunner object at 0xffe17414cf40>)
  File "/home/ma-user/work/verl/recipe/one_step_off_policy/main_ppo.py", line 214, in run
    trainer.init_workers()
  File "/home/ma-user/work/verl/recipe/one_step_off_policy/ray_trainer.py", line 148, in init_workers
    self._init_models()
  File "/home/ma-user/work/verl/recipe/one_step_off_policy/ray_trainer.py", line 255, in _init_models
    self._create_weight_sync_group()
  File "/home/ma-user/work/verl/recipe/one_step_off_policy/ray_trainer.py", line 265, in _create_weight_sync_group
    collective.create_collective_group(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/util/collective/collective.py", line 234, in create_collective_group
    _check_backend_availability(backend)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/util/collective/collective.py", line 816, in _check_backend_availability
    raise RuntimeError("NCCL is not available.")
RuntimeError: NCCL is not available
```

分析原因是，在recipe/one_step_off_policy/ray_trainer.py脚本下，` def _create_weight_sync_group(self):`方法并没有实现HCCL。

```py
    def _create_weight_sync_group(self):
        # TODO: NPU support
        from verl.utils.device import get_nccl_backend

        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        n_workers = len(actor_rollout_workers)

        # Create Ray collective group for fallback communication
        collective.create_collective_group(
            actor_rollout_workers,
            n_workers,
            list(range(0, n_workers)),
            backend=get_nccl_backend(),
            group_name="actor_rollout",
        )
```

### test-megatron-v1(ctrl+F找到v2版本)

```shell
# 加上这个export 不然会提示找不到megatron
export PYTHONPATH=$PYTHONPATH:/home/ma-user/work/Megatron-LM

#!/usr/bin/env bash
set -xuo pipefail

project_name='GRPO-Qwen2.5-0.5b-Base-MATH'
exp_name='GRPO-Qwen2.5-0.5b-Base-MATH-2gpu-async'

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH="/home/ma-user/work/models/Qwen2.5-0.5B-Instruct"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE="/home/ma-user/work/data/gsm8k/train.parquet"
TEST_FILE="/home/ma-user/work/data/gsm8k/test.parquet"

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi
# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
offload=false
train_ppo_micro_batch_size_per_gpu=2
infer_ppo_micro_batch_size_per_gpu=2

optimizer_offload_fraction=0

COMMON_PP=1
COMMON_VPP=null
COMMON_CP=1
COMMON_TP=1
COMMON_EP=1
COMMON_ETP=1

TRAIN_TP=1
INFER_TP=1

# === 强制使用脚本内定义的配置，忽略外部环境变量干扰 ===
ACTOR_PP=$COMMON_PP
ACTOR_VPP=$COMMON_VPP
ACTOR_CP=$COMMON_CP
ACTOR_TP=$TRAIN_TP
ACTOR_EP=$COMMON_EP
ACTOR_ETP=$COMMON_ETP

ROLLOUT_TP=$INFER_TP

REF_PP=$COMMON_PP
REF_VPP=$COMMON_VPP
REF_CP=$COMMON_CP
REF_TP=$TRAIN_TP
REF_EP=$COMMON_EP
REF_ETP=$COMMON_ETP

CRITIC_PP=$COMMON_PP
CRITIC_VPP=$COMMON_VPP
CRITIC_CP=$COMMON_CP
CRITIC_TP=$TRAIN_TP
CRITIC_EP=$COMMON_EP
CRITIC_ETP=$COMMON_ETP

RM_PP=$COMMON_PP
RM_VPP=$COMMON_VPP
RM_CP=$COMMON_CP
RM_TP=$TRAIN_TP
RM_EP=$COMMON_EP
RM_ETP=$COMMON_ETP
# ====================================================

# install mbridge
# pip3 install git+https://github.com/ISEEKYAN/mbridge
USE_MBRIDGE=False
USE_DIST_CKPT=False

# Fully async specific parameters
NNODES_ROLLOUT=1
NNODES_TRAIN=1
NGPUS_PER_NODE=1

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=4
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*100)))
test_freq=20
staleness_threshold=0.5
trigger_parameter_sync_step=4
require_batches=1
partial_rollout=True

python -m recipe.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml'\
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.device=npu \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs=10 \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=True \


```

会报错：
```shell
Traceback (most recent call last):
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 296, in main
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
  File "/home/ma-user/work/verl/verl/trainer/main_ppo.py", line 99, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/worker.py", line 2961, in get
    values, debugger_breakpoint = worker.get_objects(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/worker.py", line 1026, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ActorDiedError): ray::FullyAsyncTaskRunner.run() (pid=787341, ip=172.16.0.86, actor_id=2e7f87c25b36b705deb75f7801000000, repr=<fully_async_main.FullyAsyncTaskRunner object at 0xffe1ba1663e0>)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 139, in run
    self._initialize_components(config)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 206, in _initialize_components
    ray.get(param_synchronizer.sync_weights.remote(version=param_version, validate=val_before_train))
ray.exceptions.ActorDiedError: The actor died because of an error raised in its creation task, ray::ParameterSynchronizer.__init__() (pid=798168, ip=172.16.0.86, actor_id=50edde67c3ec4cde90fd6a2401000000, repr=<recipe.fully_async_policy.param_sync.ParameterSynchronizer object at 0xffe200f08a60>)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/param_sync.py", line 53, in __init__
    self._init_sync_group()
  File "/home/ma-user/work/verl/recipe/fully_async_policy/param_sync.py", line 73, in _init_sync_group
    collective.create_collective_group(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/util/collective/collective.py", line 277, in create_collective_group
    _check_backend_availability(backend)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/util/collective/collective.py", line 854, in _check_backend_availability
    raise RuntimeError("NCCL is not available.")
RuntimeError: NCCL is not available.
```

修改`verl/recipe/fully_async_policy/param_sync.py`的`_init_sync_group`，改为

```python
def _init_sync_group(self):
        print("[ParameterSynchronizer] Initializing parameter synchronization group...")
        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        collective.create_collective_group(
            actor_rollout_workers,
            len(actor_rollout_workers),
            list(range(0, len(actor_rollout_workers))),
            # backend=get_nccl_backend(), 
            backend="hccl", # 改这里
            group_name=self.sync_group_name,
        )z
```

重新运行之后报错：

**（我在云上跑的，npu的ID并非从0开始，所以报下面错误，在本地跑应该不会出现下面报错。按说应该跑通不会在报错了）**

```shell
Traceback (most recent call last):
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 296, in main
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
  File "/home/ma-user/work/verl/verl/trainer/main_ppo.py", line 99, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/worker.py", line 2961, in get
    values, debugger_breakpoint = worker.get_objects(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/worker.py", line 1026, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ActorDiedError): ray::FullyAsyncTaskRunner.run() (pid=814498, ip=172.16.0.86, actor_id=5dd3fa5629474a38234fddbd01000000, repr=<fully_async_main.FullyAsyncTaskRunner object at 0xffe1d49e22f0>)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 139, in run
    self._initialize_components(config)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 206, in _initialize_components
    ray.get(param_synchronizer.sync_weights.remote(version=param_version, validate=val_before_train))
ray.exceptions.ActorDiedError: The actor died because of an error raised in its creation task, ray::ParameterSynchronizer.__init__() (pid=825287, ip=172.16.0.86, actor_id=939f565db3cf709e492766a301000000, repr=<recipe.fully_async_policy.param_sync.ParameterSynchronizer object at 0xffe1f2bfc1c0>)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/param_sync.py", line 56, in __init__
    self._init_actor_rollout_checkpoint_engine()
  File "/home/ma-user/work/verl/recipe/fully_async_policy/param_sync.py", line 83, in _init_actor_rollout_checkpoint_engine
    ray.get(
ray.exceptions.RayTaskError(ValueError): ray::WorkerDict.actor_init_checkpoint_engine() (pid=823053, ip=172.16.0.86, actor_id=c3e5e569a8be0054904a8b4b01000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0xffe169fea140>)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['npu-smi', 'info', '-t', 'proc-mem', '-i', '0']' returned non-zero exit status 215.

The above exception was the direct cause of the following exception:

ray::WorkerDict.actor_init_checkpoint_engine() (pid=823053, ip=172.16.0.86, actor_id=c3e5e569a8be0054904a8b4b01000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0xffe169fea140>)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/concurrent/futures/_base.py", line 458, in result
    return self.__get_result()
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
  File "/home/ma-user/work/verl/verl/single_controller/ray/base.py", line 841, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
  File "/home/ma-user/work/verl/verl/single_controller/base/decorator.py", line 456, in inner
    return func(*args, **kwargs)
  File "/home/ma-user/work/verl/verl/utils/transferqueue_utils.py", line 213, in dummy_inner
    return func(*args, **kwargs)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/megatron_worker.py", line 73, in init_checkpoint_engine
    self.checkpoint_engine = CheckpointEngine(
  File "/home/ma-user/work/verl/recipe/fully_async_policy/checkpoint_engine.py", line 281, in __init__
    self._device_uuid = _get_physical_device_id(device_index)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/checkpoint_engine.py", line 197, in _get_physical_device_id
    return f"NPU-{npu_generate_uuid()}"
  File "/home/ma-user/work/verl/recipe/fully_async_policy/checkpoint_engine.py", line 188, in npu_generate_uuid
    raise ValueError("The current process is not running on the npu device") from e
ValueError: The current process is not running on the npu device
```

修改`/verl/recipe/fully_async_policy/checkpoint_engine.py`的~~`_get_physical_device_id`，~~`npu_generate_uuid`改为

```python
def npu_generate_uuid() -> str:
    """Generate uuid for each npu device"""
    str_pid = str(os.getpid())
    npu_num = 2 # 改为2 试试  这里的数字对应的是npu的数量
    try:
        for npu_id in range(npu_num):
            cmd = ["npu-smi", "info", "-t", "proc-mem", "-i", str(npu_id)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
            str_result = str(result.stdout)
            if str_pid in str_result:
                # In A3 server, one NPU has two chips.
                match_chip_count = re.search(r"Chip Count[^\d]*(\d+)", str_result)
                chip_count = int(match_chip_count.group(1))
                search_after_pid = str_result[str_result.find(str_pid) + len(str_pid) :]
                match_chip_id = re.search(r"Chip ID[^\d]*(\d+)", search_after_pid)
                chip_id = int(match_chip_id.group(1))
                return f"{get_ip()}-{npu_id * chip_count + chip_id}"
        raise ValueError("The current process is not running on the npu device")
    except subprocess.CalledProcessError as e:
        raise ValueError("The current process is not running on the npu device") from e
```

接着往下运行，此时已经跑起来了，开始train的时候，报错：
```shell
Traceback (most recent call last):
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 296, in main
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
  File "/home/ma-user/work/verl/verl/trainer/main_ppo.py", line 99, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/worker.py", line 2961, in get
    values, debugger_breakpoint = worker.get_objects(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/ray/_private/worker.py", line 1026, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(RuntimeError): ray::FullyAsyncTaskRunner.run() (pid=170944, ip=172.16.0.130, actor_id=a70a6486deacca14e3fb7a5701000000, repr=<fully_async_main.FullyAsyncTaskRunner object at 0xffe1dc3222f0>)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 140, in run
    self._run_training_loop()
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 272, in _run_training_loop
    raise e
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 266, in _run_training_loop
    ray.get(future)
ray.exceptions.RayTaskError(RuntimeError): ray::FullyAsyncTrainer.fit() (pid=178702, ip=172.16.0.130, actor_id=e83b1ea3a0b0380c9a6671b901000000, repr=<recipe.fully_async_policy.fully_async_trainer.FullyAsyncTrainer object at 0xffe13d5962f0>)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_trainer.py", line 263, in fit
    batch, reward_extra_infos_dict = self._process_batch_common(
  File "/home/ma-user/work/verl/recipe/fully_async_policy/ray_trainer.py", line 400, in _process_batch_common
    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
  File "/home/ma-user/work/verl/verl/single_controller/ray/base.py", line 53, in __call__
    output = ray.get(output)
ray.exceptions.RayTaskError(RuntimeError): ray::WorkerDict.ref_compute_ref_log_prob() (pid=180886, ip=172.16.0.130, actor_id=c7219c7d4cebe6ab868fa64201000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0xffe1680ee140>)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/concurrent/futures/_base.py", line 458, in result
    return self.__get_result()
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
  File "/home/ma-user/work/verl/verl/single_controller/ray/base.py", line 841, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
  File "/home/ma-user/work/verl/verl/single_controller/base/decorator.py", line 456, in inner
    return func(*args, **kwargs)
  File "/home/ma-user/work/verl/verl/utils/transferqueue_utils.py", line 213, in dummy_inner
    return func(*args, **kwargs)
  File "/home/ma-user/work/verl/verl/utils/profiler/performance.py", line 105, in f
    return self.log(decorated_function, *args, **kwargs)
  File "/home/ma-user/work/verl/verl/utils/profiler/performance.py", line 118, in log
    output = func(*args, **kwargs)
  File "/home/ma-user/work/verl/verl/utils/profiler/profile.py", line 256, in wrapper
    return func(self_instance, *args, **kwargs_inner)
  File "/home/ma-user/work/verl/verl/workers/megatron_workers.py", line 831, in compute_ref_log_prob
    output, _, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
  File "/home/ma-user/work/verl/verl/utils/profiler/performance.py", line 105, in f
    return self.log(decorated_function, *args, **kwargs)
  File "/home/ma-user/work/verl/verl/utils/profiler/performance.py", line 118, in log
    output = func(*args, **kwargs)
  File "/home/ma-user/work/verl/verl/workers/actor/megatron_actor.py", line 235, in compute_log_prob
    output = self.forward_backward_batch(
  File "/home/ma-user/work/verl/verl/workers/actor/megatron_actor.py", line 693, in forward_backward_batch
    losses_reduced = forward_backward_func(
  File "/home/ma-user/work/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 480, in forward_backward_no_pipelining
    output_tensor, num_tokens = forward_step(
  File "/home/ma-user/work/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 277, in forward_step
    output_tensor, loss_func = forward_step_func(data_iterator, model)
  File "/home/ma-user/work/verl/verl/workers/actor/megatron_actor.py", line 644, in forward_step
    output = forward_fn(
  File "/home/ma-user/work/verl/verl/models/mcore/model_forward.py", line 88, in model_forward
    output_orig = model(**input_args)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ma-user/work/Megatron-LM/megatron/core/transformer/module.py", line 178, in forward
    outputs = self.module(*inputs, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ma-user/work/Megatron-LM/megatron/core/models/gpt/gpt_model.py", line 334, in forward
    hidden_states = self.decoder(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ma-user/work/MindSpeed/mindspeed/te/pytorch/module/checkpoint.py", line 223, in transformer_block_forward
    hidden_states, context = layer(
  File "/home/ma-user/work/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 786, in __call__
    return super(MegatronModule, self).__call__(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ma-user/work/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 389, in forward
    pre_mlp_layernorm_output, residual, context = self._forward_attention(*args, **kwargs)
  File "/home/ma-user/work/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 449, in _forward_attention
    attention_output_with_bias = self.self_attention(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ma-user/work/Megatron-LM/megatron/core/transformer/attention.py", line 636, in forward
    core_attn_out = self.core_attention(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ma-user/work/MindSpeed/mindspeed/te/pytorch/attention/dot_product_attention/dot_product_attention.py", line 623, in forward
    attention_mask = get_attention_mask(self.config)
  File "/home/ma-user/work/MindSpeed/mindspeed/core/transformer/flash_attention/generate_mask/generate_mask.py", line 68, in get_attention_mask
    generate_attention_mask(config, compress, device)
  File "/home/ma-user/work/MindSpeed/mindspeed/core/transformer/flash_attention/generate_mask/generate_mask.py", line 21, in generate_attention_mask
    raise RuntimeError('Please set micro_batch_size or set use_flash_attn=True in config.')
RuntimeError: Please set micro_batch_size or set use_flash_attn=True in config.
```

原因是：

```shell
MindSpeed (华为 Ascend 适配库) 的限制：在 NPU 上运行 Megatron 时，MindSpeed 的注意力掩码生成函数 (generate_mask) 有一个强制检查。它要求：要么明确设置了 micro_batch_size（但在 VeRL 的 PPO 动态 batch 模式下，这个值有时在底层 config 中是 None），要么必须开启 Flash Attention。

默认值问题：尽管你在脚本里看到 attention_backend 似乎是 flash，但 MindSpeed 代码显式检查的是 config.use_flash_attn 这个布尔值。如果脚本中没有显式传递这个参数，它可能默认为 False，导致进入了传统的 Attention Mask 生成逻辑，从而触发了报错。
解决方法
你需要显式地在 Megatron 的 Transformer 配置中开启 Flash Attention。
请修改你的启动脚本，在 python -m recipe.fully_async_policy.fully_async_main \ 下面的参数列表中，添加以下这行配置：
Bash
+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \

# to-do: ref和和critic是不是也要加上Flash Attention的设置？
```

修改后的命令为：**test-megatron-v2**

```shell
# 加上这个export 不然会提示找不到megatron
export PYTHONPATH=$PYTHONPATH:/home/ma-user/work/Megatron-LM

#!/usr/bin/env bash
set -xuo pipefail

project_name='GRPO-Qwen2.5-0.5b-Base-MATH'
exp_name='GRPO-Qwen2.5-0.5b-Base-MATH-2gpu-async'

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH="/home/ma-user/work/models/Qwen2.5-0.5B-Instruct"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE="/home/ma-user/work/data/gsm8k/train.parquet"
TEST_FILE="/home/ma-user/work/data/gsm8k/test.parquet"

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi
# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
offload=false
train_ppo_micro_batch_size_per_gpu=2
infer_ppo_micro_batch_size_per_gpu=2

optimizer_offload_fraction=0

COMMON_PP=1
COMMON_VPP=null
COMMON_CP=1
COMMON_TP=1
COMMON_EP=1
COMMON_ETP=1

TRAIN_TP=1
INFER_TP=1

# === 强制使用脚本内定义的配置，忽略外部环境变量干扰 ===
ACTOR_PP=$COMMON_PP
ACTOR_VPP=$COMMON_VPP
ACTOR_CP=$COMMON_CP
ACTOR_TP=$TRAIN_TP
ACTOR_EP=$COMMON_EP
ACTOR_ETP=$COMMON_ETP

ROLLOUT_TP=$INFER_TP

REF_PP=$COMMON_PP
REF_VPP=$COMMON_VPP
REF_CP=$COMMON_CP
REF_TP=$TRAIN_TP
REF_EP=$COMMON_EP
REF_ETP=$COMMON_ETP

CRITIC_PP=$COMMON_PP
CRITIC_VPP=$COMMON_VPP
CRITIC_CP=$COMMON_CP
CRITIC_TP=$TRAIN_TP
CRITIC_EP=$COMMON_EP
CRITIC_ETP=$COMMON_ETP

RM_PP=$COMMON_PP
RM_VPP=$COMMON_VPP
RM_CP=$COMMON_CP
RM_TP=$TRAIN_TP
RM_EP=$COMMON_EP
RM_ETP=$COMMON_ETP
# ====================================================

# install mbridge
# pip3 install git+https://github.com/ISEEKYAN/mbridge
USE_MBRIDGE=False
USE_DIST_CKPT=False

# Fully async specific parameters
NNODES_ROLLOUT=1
NNODES_TRAIN=1
NGPUS_PER_NODE=1

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=4
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*100)))
test_freq=20
staleness_threshold=0.5
trigger_parameter_sync_step=4
require_batches=1
partial_rollout=True

python -m recipe.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml'\
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.device=npu \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs=10 \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=True \


```

