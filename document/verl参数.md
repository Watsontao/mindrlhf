```
project_name
exp_name
RAY_DATA_HOME
MODEL_PATH
CKPTS_DIR
TRAIN_FILE
TEST_FILE
rollout_mode
rollout_name
```



## mbridge

```
enable_overlong_buffer
overlong_buffer_len
overlong_penalty_facto
USE_MBRIDGE
USE_DIST_CKPT
```



## Fully async specific parameters

| super params                                                 | implication                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `trainer.nnodes`                                             | Trainer的node数量                                            |
| `trainer.n_gpus_per_node`                                    | Trainer每个node上gpu的数量                                   |
| `rollout.nnodes`                                             | Rollouter的node数量                                          |
| `rollout.n_gpus_per_node`                                    | Rollouter每个node上gpu的数量                                 |
| `data.train_batch_size`                                      | 在fully async策略中，该值不生效（默认设置为0）               |
| `data.gen_batch_size`                                        | 在fully async策略中，使用流式的样本生产逻辑（默认设置为1)    |
| `rollout.total_rollout_steps`                                | 总的rollout的sample数量                                      |
| `rollout.test_freq`                                          | Rollouter每更新多少次参数，进行一次validation                |
| `actor_rollout_ref.actor.ppo_mini_batch_size`                | The ppo_mini_batch_size is a global num across all workers/gpus |
| `async_training.require_batches`                             | FullyAsyncTrainer一次性获取的ppo_mini_batch_size的数量       |
| `async_training.trigger_parameter_sync_step`                 | 表示FullyAsyncTrainer进行多少次本地更新后,进行一次参数同步   |
| `async_training.staleness_threshold`                         | 新鲜度控制                                                   |
| `async_training.partial_rollout`                             | 是否进行partial_rollout                                      |
| `async_training.use_rollout_log_probs`                       | 使用rollout产生的log_probs                                   |
| `async_training.compute_prox_log_prob`（experimental）       | 是否在train阶段，使用train模型的参数计算token的 log_prob     |
| `async_training.checkpoint_engine.enable`                    | 是否开启checkpoint_engine模式的加速，默认值True              |
| `async_training.checkpoint_engine.overlap_broadcast_and_consume` | 启动checkpoint_engine时，是否在参数同步时在broadcast和加载之间使用流水，默认值False |
| `async_training.checkpoint_engine.device_buffer_size_M`      | 启动checkpoint_engine时，组装的bucket的大小(MB)，默认为4096  |



### 关键指标

| metrics                                        | implication                                                  |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `trainer/idle_ratio`                           | Trainer闲置率                                                |
| `rollouter/idle_ratio`                         | Rollouter闲置率                                              |
| `fully_async/count/stale_samples_processed`    | 训练使用的旧sample总数                                       |
| `fully_async/count/stale_trajectory_processed` | 训练使用的旧trajectory总数(一个sample会生产rollout.n条trajectory) |
| `fully_async/partial/total_partial_num`        | 两次trigger_parameter_sync_step之间Trainer处理的partial样本数 |
| `fully_async/partial/partial_ratio`            | 两次trigger_parameter_sync_step之间Trainer处理的partial样本的比例 |
| `fully_async/partial/max_partial_span`         | 两次trigger_parameter_sync_step之间Trainer处理的partial样本的最大参数跨度 |



## Performance Related Parameter

```
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
offload=True
train_ppo_micro_batch_size_per_gpu=2
infer_ppo_micro_batch_size_per_gpu=2
```

