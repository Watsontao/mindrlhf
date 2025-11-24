跑通verl-Ascend

## 环境准备

目前的镜像为`pytorch_2.6.0-cann_8.2.rc1-py_3.11-euler_2.10.11-aarch64-snt9b`

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
export CC=/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/aarch64-conda-linux-gnu-gcc
export CXX=/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/aarch64-conda-linux-gnu-g++
#===================

pip install -e .

cd ../

# 安装verl
git clone https://github.com/volcengine/verl.git
cd verl
pip install -r requirements-npu.txt
pip install -e .
```



## quick start

1. **下载数据和模型**

```shell
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k


pip install modelscope
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct  --local_dir   /home/ma-user/work/models/Qwen2.5-0.5B-Instruct
```

2. **执行训练**

```shell
set -x

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
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.device=npu $@
```









## 附录

### 附录1 gcc版本

```shell
(PyTorch-2.6.0) [ma-user vllm]$gcc -v
Using built-in specs.
COLLECT_GCC=gcc
COLLECT_LTO_WRAPPER=/usr/libexec/gcc/aarch64-linux-gnu/7.3.0/lto-wrapper
Target: aarch64-linux-gnu
Configured with: ../configure --prefix=/usr --mandir=/usr/share/man --infodir=/usr/share/info --enable-shared --enable-threads=posix --enable-checking=release --with-system-zlib --enable-__cxa_atexit --disable-libunwind-exceptions --enable-gnu-unique-object --enable-linker-build-id --with-linker-hash-style=gnu --enable-languages=c,c++,objc,obj-c++,fortran,lto --enable-plugin --enable-initfini-array --disable-libgcj --without-isl --without-cloog --enable-gnu-indirect-function --build=aarch64-linux-gnu --with-stage1-ldflags=' -Wl,-z,relro,-z,now' --with-boot-ldflags=' -Wl,-z,relro,-z,now' --with-multilib-list=lp64
Thread model: posix
gcc version 7.3.0 (GCC) 
(PyTorch-2.6.0) [ma-user vllm]$
```

