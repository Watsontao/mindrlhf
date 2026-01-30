UloRL:An Ultra-Long Output Reinforcement Learning Approach for Advancing Large Language Models’ Reasoning Abilities



StreamingRL: Scalable. Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation



Laminar: A Scalable Asynchronouts RL Post-Training Framework



Taming the Long-Tail: efficient Reasoning RL Training with Adaptive Drafter



Fast LLM Posting-training via Decoupled and Best-of-N Speculation



RLBoost: Harvesting Preemtible Resources for Cost-Efficient Reinforcement Learning on LLMs



History Rhymes: Accelerating LLM Reinforcement Learning with RhymeRL





























你是一个专业的研究大模型的大厂开发人员，我的需求是不需要改太多代码 就在verl上进行修改 做出一些创新来缓解长尾问题并发表一篇会议论文 实验用GRPO或者DAPO 结合前面我们对那么多论文的讨论 在这些基础上 你给出我一些建议 要有逻辑 一步接着一步 把故事讲好 一两个创新点肯定是不够发顶会的，同时还不能水 工作量要足 最好还是要能对这个领域做出贡献，我们现在已经确定了要采用segment rollout方法 同时采用POIS。



我的初步动机是：
**1. 现有问题（The Villain）：**

- **背景**：DeepSeek-R1 等模型需要超长 CoT。
- **痛点 1（显存气泡）**：同步 GRPO 训练受限于最长样本，UloRL 提出的 Segment Rollout 虽然通过“切片”解决了这个问题，但它是**静态切片**（固定长度，如每 1024 token 切一次）。
- **痛点 2（无效计算）**：在分段生成中，很多组（Group）内的样本在第一个 Segment 其实就已经跑偏了（出现幻觉或逻辑错误），或者已经完全收敛（复读机）。继续对这些 Segment 进行 Rollout 是纯粹的算力浪费（VADE 的观点）。
- **痛点 3（训练干扰）**：对这些错误的中间 Segment 进行训练，会引入噪声（DMMPTs 的观点）。

**2. 你的洞察（The Insight）：**

- **核心假设**：在 GRPO 的 Group 生成中，**“好”的轨迹往往是相似的，而“坏”的轨迹各有各的坏法**，且坏的轨迹往往在早期就能通过**组内统计特征（方差/熵）**被识别出来。

  

你在思考的时候 要确保 我们讨论出来的方案要有鲁棒性，不能只是在单一模型/数据集/算法上work 要提高时间效率  同时思考的时候尽可能的要参考前面的论文 并标记出参考了哪篇论文

我在想要不要结合RhymeRL













```shell
Role: 你现在是一位顶级的大模型训练系统架构师（Principal Research Engineer），专注于 Post-training 和 RLHF 效率优化。你熟悉 verl 代码库，且对 2025 年最新的 RL 论文（UloRL, RhymeRL, VADE, Laminar, TLT, etc.）有深刻理解。

Goal: 我们需要在 verl 的recipe的full_async_poliicy全异步框架上进行二次开发，发表一篇 NeurIPS/ICLR 级别的会议论文。核心目标是解决 DeepSeek-R1 这类推理模型在强化学习（GRPO/DAPO）中的长尾效率问题。 我们已确定采用 Segment Rollout (from UloRL) 作为基座，并结合 POIS (from Pseudo On-Policy) 来保证训练稳定性。

Context & Motivation (The Villain): 目前 GRPO 训练面临“不可能三角”：

显存气泡：同步训练受限于最长样本。UloRL 虽然提出了 Segment Rollout，但它是静态切片（如固定 1024 token）。对于大部分在 Segment 1 就结束的短样本（Easy Set），强制分段反而增加了 KV Cache 搬运和调度开销（Overhead）。

无效计算（The VADE Perspective）：在分段生成中，很多 Group 内的样本在早期 Segment 就已经崩坏（幻觉）或过度收敛（复读机）。继续 Rollout 这些样本是纯粹的算力浪费。

训练噪声（The DMMPTs Perspective）：对这些错误的中间 Segment 进行训练会污染 Policy。

Our Insight (The Hero): 我们认为：“好”的轨迹在组内统计上是相似的，而“坏”的轨迹在早期就能通过组内特征（方差/熵/长度预测）被识别出来。 因此，我们不应该用“一把尺子（固定分段）量到底”，而应该实现 "Elastic Segmentation & Pruning"（弹性分段与剪枝）。

Constraints:

基于 verl 修改，工作量要充实，不能只是简单的 A+B。

必须有显著的时间效率提升（Time-to-convergence）。

方案必须具有鲁棒性，不能只在特定数据集生效。

Task: 请基于上述背景，结合 RhymeRL（利用历史预测长度）和 VADE（利用方差筛选）的思想，为我设计一个完整的算法框架。请详细回答以下 4 个部分：

1. 核心算法设计（Storytelling）
请构建一个名为 "Elastic-Segment GRPO" (ES-GRPO) 的方法论。

RhymeRL 的融合：如何利用 RhymeRL 的“历史押韵”思想来解决你提到的“大部分样本只在 Segment 1 就结束”的问题？（提示：是否可以在 Rollout 前加一个 Router？）

动态分段逻辑：如何把 UloRL 的静态分段改为动态？如果 Group 内 80% 的样本都结束了，剩下的 20% 长尾样本该怎么处理？

组内剪枝（Group Pruning）：结合 VADE/DMMPTs，在 Segment 结束时，如何利用 Group 的统计信息（如 Variance 或 Reward Model 预测）提前砍掉那些“没救了”的 Segment，不再进行后续 Rollout？

2. 效率贡献分析（Efficiency Ranking）
请从系统工程角度，预估以下模块对“训练总时长”的加速贡献，并排序（High/Medium/Low），给出理由：

Module A: 基于 RhymeRL 的长短任务路由（Short-circuiting short tasks）。

Module B: 基于 UloRL 的 Segment Rollout（解决显存 OOM 和并行度）。

Module C: 基于 VADE 思想的 Early Pruning（减少无效 Token 生成）。

Module D: 动态调整 Segment Size（根据当前 Group 的未完成率动态决定下一段跑多少）。

3. 鲁棒性与潜在坑点（Devil's Advocate）
RhymeRL 的冷启动问题：RhymeRL 需要历史数据，刚开始训练时（Step 0）怎么办？预测不准会导致 OOM 吗？

剪枝的风险：如果把“大器晚成”（前面很乱，最后一步推理对的）的样本剪掉了怎么办？如何设计一个保守但有效的剪枝阈值？

4. 实验设计（The Evidence）
除了常规的 Math/Code 榜单，我需要设计什么特殊的实验指标来证明“解决了长尾”？（例如：Bubble Rate, Effective Token Throughput, etc.）

对比 Baseline 应该是谁？（Standard GRPO vs. Static UloRL vs. Our Method）

请像写论文的 Methodology 和 Experiments 章节一样逻辑严密地回答。
```







```

1. Rollout Scheduler
根据 Prompt 队列输入，调用Ranker 模型预测每个队列的“执行优先级分数”，再按分数从短到长排序分发给 Actor。
实时接收 DOC 的系统信号（entropy、GPU 利用率、Buffer 占用），并汇报最新的 rollout 完成时间、loss、entropy 变化。
2. Actor Worker
从 Scheduler 接收 Prompt 和参数版本 vt，执行完整 rollout，在生成过程中监控 entropy、token 速率等指标并传给Scheduler。
3. Experience Buffer
缓存 <rollout_id, rollout, vt, status> 数据结构用于Learner 训练。负责计算当前缓冲区填充率并上报给 DOC。
4. Learner
持续监听 DOC 指定的 train_trigger_size，当达到阈值即启动训练。训练结束后汇报 loss、entropy 变化、训练耗时给 Scheduler。
5. Parameter Buffer
存储所有模型版本参数 (v₀, v₁, …)，提供权重拉取接口。
6. DOC（Dynamic Overlap Controller）
接收来自 Scheduler 的系统指标并计算动态掩盖率：
overlap_ratio = f(entropy, buffer_fill, gpu_util, loss)
并生成训练触发阈值：train_trigger_size=max_batch_size×overlap_ratio, 将train_trigger_size下发给 Learner 和 Experience Buffer。
```





```shell

# 加上这个export 不然会提示找不到megatron
# export PYTHONPATH=$PYTHONPATH:/home/ma-user/work/Megatron-LM
# 配置为4张卡
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

#!/usr/bin/env bash
set -xuo pipefail

project_name='GRPO-Qwen2.5-0.5b-Base-MATH'
exp_name='GRPO-Qwen2.5-0.5b-Base-MATH-2gpu-async'

RAY_DATA_HOME=${RAY_DATA_HOME:-"/mnt/disk0/q00596439/test_wst/log_verl"}
MODEL_PATH="/mnt/disk0/q00596439/Qwen2.5_0.5B_Instruct"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE="/mnt/disk0/q00596439/verl_wst/data/gsm8k/train.parquet"
TEST_FILE="/mnt/disk0/q00596439/verl_wst/data/gsm8k/test.parquet"

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
max_response_length=$((1024 / 1))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 / 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
# actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
actor_ppo_max_token_len=16384
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))

offload=false
train_ppo_micro_batch_size_per_gpu=32
infer_ppo_micro_batch_size_per_gpu=20

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
# NGPUS_PER_NODE=4

TRAIN_NGPUS=2    
ROLLOUT_NGPUS=2  


train_prompt_bsz=0
gen_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=64
total_rollout_steps=$(((512*100)))
test_freq=20
staleness_threshold=0.2
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
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${TRAIN_NGPUS}" \
    trainer.device=npu \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${ROLLOUT_NGPUS}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs=10 \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=True \

同时回答下这几个指标是啥意思
(FullyAsyncRollouter pid=2089415) [FullyAsyncRollouter][MonitorLoop][Statistics] {'count/current_param_version': 0,
(FullyAsyncRollouter pid=2089415)  'count/dropped_stale_samples': 0,
(FullyAsyncRollouter pid=2089415)  'count/staleness_samples': 99,
(FullyAsyncRollouter pid=2089415)  'count/total_generated_samples': 66,
(FullyAsyncRollouter pid=2089415)  'monitor/active_tasks_size': 32,
(FullyAsyncRollouter pid=2089415)  'monitor/queue/cancel_queue_size': 0,
(FullyAsyncRollouter pid=2089415)  'monitor/queue/mq_queue_size': 2,
(FullyAsyncRollouter pid=2089415)  'monitor/queue/pending_queue_size': 128,
(FullyAsyncRollouter pid=2089415)  'monitor/queue/result_queue_size': 0,
(FullyAsyncRollouter pid=2089415)  'static/max_concurrent_samples': 32,
(FullyAsyncRollouter pid=2089415)  'static/max_queue_size': 307,
(FullyAsyncRollouter pid=2089415)  'static/max_required_samples': 307,
(FullyAsyncRollouter pid=2089415)  'static/required_samples': 64,
(FullyAsyncRollouter pid=2089415)  'static/staleness_threshold': 0.2}


```

```shell
Traceback (most recent call last):
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 301, in <module>
    main()
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
    raise ex
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/hydra/_internal/hydra.py", line 132, in run
    _ = ret.return_value
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/hydra/core/utils.py", line 260, in return_value
    raise self._return_value
  File "/home/ma-user/.conda/envs/test01/lib/python3.10/site-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
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
ray.exceptions.RayTaskError(SyntaxError): ray::FullyAsyncTaskRunner.run() (pid=159799, ip=172.16.0.116, actor_id=241ebfb71906fe33edd3bae901000000, repr=<fully_async_main.FullyAsyncTaskRunner object at 0xffe1cf9449d0>)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 139, in run
    self._initialize_components(config)
  File "/home/ma-user/work/verl/recipe/fully_async_policy/fully_async_main.py", line 191, in _initialize_components
    from recipe.fully_async_policy.param_sync import ParameterSynchronizer
  File "/home/ma-user/work/verl/recipe/fully_async_policy/param_sync.py", line 78
    backend='hccl',
    ^^^^^^^^^^^^^^
SyntaxError: keyword argument repeated: backend
[ERROR] 2026-01-01-23:26:58 (PID:154980, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
(FullyAsyncTaskRunner pid=159799) [ASYNC MAIN] Setting up parameter synchronization...
(MessageQueue pid=170926) [MessageQueue] initialized with max_queue_size=192,staleness_threshold=0.5
/home/ma-user/.conda/envs/test01/lib/python3.10/subprocess.py:1072: ResourceWarning: subprocess 156107 is still running
  _warn("subprocess %s is still running" % self.pid,  是因为这个吗？
```

