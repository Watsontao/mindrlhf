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


# python -c "from vllm import LLM; print('恭喜！vLLM 导入成功！')"


# export VLLM_ASCEND_ENABLE_NZ=0

# pip install torch==2.7.1 torch-npu==2.7.1 torchvision
```



## quick start

1. **下载数据和模型**

```shell
# 如果遇到 ImportError: /lib/aarch64-linux-gnu/libstdc++.so.6:
# version `CXXABI_1.3.15' not found

# 试着export LD_LIBRARY_PATH=/home/ma-user/.conda/envs/test01/lib:$LD_LIBRARY_PATH
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k


pip install modelscope
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct  --local_dir   /home/ma-user/work/models/Qwen2.5-0.5B-Instruct
```

2. **grpo_qwen3_8b_gsm8k_fsdp2_8_8_npu.sh**

```shell
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
max_response_length=$((1024 * 32))

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

分析原因是，在recipe/one_step_off_policy/ray_trainer.py脚本下，` def _create_weight_sync_group(self):`方法（271行左右）并没有实现HCCL。

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
            backend=get_nccl_backend(),   # 这里写死了nccl
            group_name="actor_rollout",
        )
```

**verl版本信息如下：**

```shell
+ git describe --tags
v0.5.0-716-gccd7d934
```

