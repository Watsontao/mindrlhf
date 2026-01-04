import matplotlib

# ★★★ 强制使用 Agg 后端，防止无 GUI 环境报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import os
import re


def clean_key_name(key):
    """清理键名，移除 ANSI 颜色代码和杂乱前缀"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    key = ansi_escape.sub('', key).strip()

    # 统一 step 键名
    if key.endswith('step') and '/' not in key and '_' not in key:
        return 'step'
    # 处理 (TaskRunner pid=...) step 这种情况
    if key.endswith(' step'):
        return 'step'
    return key


def parse_log(filename):
    data = []
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found.")
        return pd.DataFrame()

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if "step:" not in line or " - " not in line:
            continue

        row = {}
        segments = line.strip().split(' - ')

        for seg in segments:
            if ':' not in seg:
                continue
            parts = seg.rsplit(':', 1)
            if len(parts) != 2: continue

            k, v = parts
            k = clean_key_name(k)

            try:
                val = float(v)
                row[k] = val
            except ValueError:
                continue

        if 'step' in row:
            data.append(row)

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by='step')
    return df


def plot_comparison(sync_file, async_file):
    print(f"Parsing Sync Log: {sync_file}...")
    df_sync = parse_log(sync_file)

    print(f"Parsing Async Log: {async_file}...")
    df_async = parse_log(async_file)

    if df_sync.empty or df_async.empty:
        print("Error: One of the dataframes is empty. Check file paths.")
        return

    # 定义要对比的公共指标
    # (指标Key, 图表标题)
    metrics_to_compare = [
        ('critic/rewards/mean', 'Mean Reward (Sync vs Async)'),
        ('critic/score/mean', 'Mean Score (Sync vs Async)'),
        ('actor/ppo_kl', 'PPO KL Divergence'),
        ('actor/pg_loss', 'Actor PG Loss'),
        ('perf/throughput', 'Throughput (tokens/s)'),
        ('perf/time_per_step', 'Time per Step (s)')
    ]

    # 筛选出两个日志中都存在的指标
    common_metrics = [
        m for m in metrics_to_compare
        if m[0] in df_sync.columns and m[0] in df_async.columns
    ]

    if not common_metrics:
        print("No common metrics found to compare.")
        return

    # 绘图布局
    num_plots = len(common_metrics)
    cols = 2
    rows = (num_plots + 1) // cols

    plt.figure(figsize=(16, 5 * rows))

    for i, (key, title) in enumerate(common_metrics):
        plt.subplot(rows, cols, i + 1)

        # 绘制 Sync 曲线 (蓝色)
        sync_data = df_sync[['step', key]].dropna()
        plt.plot(sync_data['step'], sync_data[key],
                 label='Synchronous', color='tab:blue', linestyle='-', marker='o', markersize=3, alpha=0.7)

        # 绘制 Async 曲线 (红色)
        async_data = df_async[['step', key]].dropna()
        plt.plot(async_data['step'], async_data[key],
                 label='Asynchronous', color='tab:red', linestyle='-', marker='x', markersize=3, alpha=0.7)

        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel(key.split('/')[-1])  # 简化 Y 轴标签
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

    plt.suptitle('Synchronous vs Asynchronous Training Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_file = "comparison_sync_async.png"
    plt.savefig(output_file, dpi=150)
    print(f"Comparison chart saved to: {output_file}")
    plt.close()


# --- 运行配置 ---
sync_log = "training_log_20260102_235402.txt"
async_log = "async_train_log_20260102_170319.txt"

plot_comparison(sync_log, async_log)