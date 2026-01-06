import matplotlib

# 强制使用 Agg 后端
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import os
import re


def clean_key_name(key):
    """清理键名，移除 ANSI 颜色代码"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    key = ansi_escape.sub('', key).strip()
    # 统一 step 键名
    if key.endswith('step') and '/' not in key and '_' not in key:
        return 'step'
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
        if "step:" not in line:
            continue

        # 简单的预过滤，只保留包含关键指标的行，防止读取无关的 step 日志
        # 如果你的日志里只有关键行才包含 'timing_s'，这能过滤掉很多杂音
        # if "timing_s" not in line and "perf" not in line:
        #    continue

        row = {}
        # 某些日志格式可能是 "key:val  key:val"，这里做个简单的通用分割处理
        # 先尝试按 " - " 分割（verl 默认格式）
        if " - " in line:
            segments = line.strip().split(' - ')
        else:
            # 如果不是标准格式，尝试按空格或tab分割
            segments = line.strip().split()

        for seg in segments:
            if ':' not in seg:
                continue
            try:
                k_raw, v_raw = seg.rsplit(':', 1)
                k = clean_key_name(k_raw)
                v = float(v_raw)
                row[k] = v
            except ValueError:
                continue

        if 'step' in row:
            data.append(row)

    df = pd.DataFrame(data)

    if df.empty:
        return df

    # ★★★ 核心修复：按 step 分组并去重 ★★★
    # 这会合并同一个 step 的多行日志。
    # .last() 表示如果同一个 key 出现多次，取最后出现的值（通常最后的日志最全）
    # .mean() 也可以，但在日志分析中 .last() 更符合逻辑
    df = df.groupby('step').last().reset_index()

    # 再次排序确保连线正确
    df = df.sort_values(by='step')

    return df


def plot_comparison(sync_file, async_file):
    print(f"Parsing Sync Log: {sync_file}...")
    df_sync = parse_log(sync_file)

    print(f"Parsing Async Log: {async_file}...")
    df_async = parse_log(async_file)

    if df_sync.empty or df_async.empty:
        print("Error: One of the dataframes is empty.")
        return

    # 定义对比指标：优先使用 timing_s/step，它是最准确的算法步耗时
    metrics_to_compare = [
        ('critic/score/mean', 'Mean Score (GSM8k Accuracy)'),
        ('actor/ppo_kl', 'PPO KL Divergence'),
        ('timing_s/step', 'Time per Step (s)'),  # 改用这个 key，通常比 perf/time_per_step 更稳定
        ('perf/throughput', 'Throughput (tokens/s)'),
    ]

    # 绘图布局
    plt.figure(figsize=(16, 10))  # 调整大小

    # 找出共有的列
    valid_metrics = []
    for m_key, m_title in metrics_to_compare:
        # 尝试找 key，如果找不到，尝试找备选 key (比如 perf/time_per_step)
        if m_key not in df_sync.columns and m_key == 'timing_s/step':
            if 'perf/time_per_step' in df_sync.columns:
                m_key = 'perf/time_per_step'

        if m_key in df_sync.columns and m_key in df_async.columns:
            valid_metrics.append((m_key, m_title))
        else:
            print(f"Skipping {m_key}: not found in both logs.")

    for i, (key, title) in enumerate(valid_metrics):
        plt.subplot(2, 2, i + 1)  # 固定 2x2 布局

        # 绘制 Sync
        plt.plot(df_sync['step'], df_sync[key],
                 label='Synchronous', color='#1f77b4', linewidth=1.5)

        # 绘制 Async
        plt.plot(df_async['step'], df_async[key],
                 label='Asynchronous', color='#d62728', linewidth=1.5)

        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel(key)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()

    plt.suptitle('Sync vs Async PPO Training (Deduplicated Steps)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_file = "comparison_fixed.png"
    plt.savefig(output_file, dpi=150)
    print(f"Chart saved to: {output_file}")


# --- 请替换为你的实际文件名 ---
sync_log = "training_log_20260102_235402.txt"
async_log = "async_train_log_20260102_170319.txt"
# 如果你在 notebook 环境可以直接运行，如果是在终端请取消注释下面这一行
plot_comparison(sync_log, async_log)