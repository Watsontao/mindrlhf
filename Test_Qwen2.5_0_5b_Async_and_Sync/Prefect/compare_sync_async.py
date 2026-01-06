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

    # --- 修改配置：定义您要求的5个关键指标 ---
    # 格式：(日志中的Key, 图表标题)
    metrics_to_compare = [
        ('critic/rewards/mean', 'Mean Reward'),  # 1. Reward
        ('val-core/openai/gsm8k/acc/mean@1', 'Validation Accuracy'),  # 2. Acc (准确率)
        ('perf/time_per_step', 'Time per Step (s)'),  # 3. 单步消耗时间
        ('actor/ppo_kl', 'KL Divergence'),  # 4. KL散度变化
        ('actor/pg_loss', 'Policy Gradient Loss')  # 5. Loss曲线
    ]

    # 筛选出两个日志中都存在的指标
    # 注意：如果某个指标（如Acc）只在一个日志里有，这里会跳过。
    # 为了防止因为列名不完全匹配导致画不出来，我们分别检查
    common_metrics = []
    for key, title in metrics_to_compare:
        if key in df_sync.columns and key in df_async.columns:
            common_metrics.append((key, title))
        else:
            print(f"Warning: Metric '{key}' missing in one of the logs, skipping.")

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
        # 验证集数据点少，增加点的大小
        marker_size = 5 if 'acc' in key else 2

        if not sync_data.empty:
            plt.plot(sync_data['step'], sync_data[key],
                     label='Synchronous', color='tab:blue', linestyle='-', marker='o', markersize=marker_size,
                     alpha=0.7)

        # 绘制 Async 曲线 (红色)
        async_data = df_async[['step', key]].dropna()
        if not async_data.empty:
            plt.plot(async_data['step'], async_data[key],
                     label='Asynchronous', color='tab:red', linestyle='-', marker='x', markersize=marker_size,
                     alpha=0.7)

        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

    plt.suptitle('Sync vs Async: Reward, Acc, Time, KL, Loss', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_file = "comparison_5_metrics.png"
    plt.savefig(output_file, dpi=150)
    print(f"Comparison chart saved to: {output_file}")
    plt.close()


# --- 运行配置 ---
# 请确保这两个文件在当前目录下
sync_log = "sync_training_log_20260105_171127.txt"
async_log = "async_train_log_20260106_001332.txt"

plot_comparison(sync_log, async_log)