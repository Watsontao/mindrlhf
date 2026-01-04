import matplotlib

# ★★★ 强制使用 Agg 后端，防止无 GUI 环境报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import os
import re


def clean_key_name(key):
    """清理键名，移除 ANSI 颜色代码和杂乱前缀"""
    # 移除颜色代码 (如 [36m, [0m)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    key = ansi_escape.sub('', key).strip()

    # 处理 step 键 (如 "(FullyAsyncTrainer pid=...) step")
    if key.endswith('step') and '/' not in key:
        return 'step'
    return key


def plot_async_metrics(log_file_path):
    data = []

    if not os.path.exists(log_file_path):
        print(f"Error: File {log_file_path} not found.")
        return

    print(f"Reading file: {log_file_path}...")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # 1. 必须包含 'step:'
        if "step:" not in line:
            continue

        # 2. 必须包含 ' - ' 分隔符 (这是指标行的特征)
        if " - " not in line:
            continue

        row = {}
        segments = line.strip().split(' - ')

        for seg in segments:
            if ':' not in seg:
                continue

            parts = seg.rsplit(':', 1)
            if len(parts) != 2:
                continue

            k, v = parts
            k = clean_key_name(k)

            try:
                val = float(v)
                row[k] = val
            except ValueError:
                continue

        if 'step' in row:
            data.append(row)

    if not data:
        print("No valid metrics data found.")
        return

    df = pd.DataFrame(data)
    df = df.sort_values(by='step')

    print(f"Found {len(df)} data points.")

    # --- 定义异步训练的关键指标 ---
    metrics_to_plot = [
        # 1. 基础 RL 指标
        ('critic/rewards/mean', 'Critic Mean Reward'),
        ('actor/ppo_kl', 'Actor PPO KL'),
        ('actor/pg_loss', 'Actor PG Loss'),

        # 2. 异步特有指标 (关注数据陈旧度和处理延迟)
        ('fully_async/count/staleness_samples', 'Staleness Samples (Async Lag)'),
        ('fully_async/processing_time/avg', 'Avg Processing Time (s)'),

        # 3. 性能指标
        ('perf/throughput', 'Throughput (tokens/s)'),
        ('rollout_corr/rollout_ppl', 'Rollout PPL')
    ]

    # 筛选存在的指标
    valid_metrics = [m for m in metrics_to_plot if m[0] in df.columns]

    if not valid_metrics:
        print("None of the target metrics found.")
        return

    # 绘图布局
    num_plots = len(valid_metrics)
    cols = 2
    rows = (num_plots + 1) // cols

    plt.figure(figsize=(16, 5 * rows))

    for i, (key, title) in enumerate(valid_metrics):
        plt.subplot(rows, cols, i + 1)

        sub_df = df[['step', key]].dropna()
        if sub_df.empty:
            continue

        # 绘制曲线
        plt.plot(sub_df['step'], sub_df[key], marker='.', linestyle='-', alpha=0.8, label=key)

        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

    plt.suptitle(f'Async Training Metrics: {os.path.basename(log_file_path)}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图片
    base_name = os.path.splitext(os.path.basename(log_file_path))[0]
    output_png = f"{base_name}_metrics.png"

    plt.savefig(output_png, dpi=150)
    print(f"Successfully saved chart to: {output_png}")
    plt.close()


# --- 执行 ---
# 请修改为您的异步日志文件名
async_log_file = "async_train_log_20260102_170319.txt"
plot_async_metrics(async_log_file)