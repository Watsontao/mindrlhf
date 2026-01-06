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
    metrics_data = []
    progress_data = []

    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found.")
        return pd.DataFrame()

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 正则表达式匹配进度条行
    # 匹配模式：Training Progress: ... | 9/58 [49:43<4:25:49, 325.50s/it]
    # group(1): step (9)
    # group(2): time (325.50)
    tqdm_pattern = re.compile(r'Training Progress:.*?\|\s*(\d+)/\d+\s+\[.*,\s*(\d+\.\d+)s/it\]')

    for line in lines:
        # 1. 优先尝试匹配进度条行（提取平滑后的时间）
        match = tqdm_pattern.search(line)
        if match:
            try:
                step = int(match.group(1))
                time_val = float(match.group(2))
                progress_data.append({
                    'step': step,
                    'time_from_progress': time_val
                })
            except ValueError:
                pass
            continue  # 如果是进度条行，就不处理为常规 metric 行了

        # 2. 处理常规 Metrics 行
        if "step:" in line and " - " in line:
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
                metrics_data.append(row)

    # 分别创建 DataFrame
    df_metrics = pd.DataFrame(metrics_data)
    df_progress = pd.DataFrame(progress_data)

    # 结果合并逻辑
    if df_metrics.empty and df_progress.empty:
        return pd.DataFrame()
    elif df_metrics.empty:
        return df_progress.sort_values(by='step')
    elif df_progress.empty:
        return df_metrics.sort_values(by='step')
    else:
        # 根据 step 进行合并 (Outer Join 以保留所有数据)
        # 注意：有时候 metrics 记录的 step 和进度条的 step 可能不是完全一一对应（比如进度条每N步打印一次）
        # 这里使用 merge 可以自动对齐
        df_final = pd.merge(df_metrics, df_progress, on='step', how='outer')
        return df_final.sort_values(by='step')


def plot_comparison(sync_file, async_file):
    print(f"Parsing Sync Log: {sync_file}...")
    df_sync = parse_log(sync_file)

    print(f"Parsing Async Log: {async_file}...")
    df_async = parse_log(async_file)

    if df_sync.empty or df_async.empty:
        print("Error: One of the dataframes is empty. Check file paths.")
        return

    # 定义要对比的公共指标
    # 注意：这里把原来的 perf/time_per_step 换成了 time_from_progress
    metrics_to_compare = [
        ('critic/rewards/mean', 'Mean Reward'),
        ('critic/score/mean', 'Mean Score'),
        ('actor/ppo_kl', 'PPO KL Divergence'),
        ('actor/pg_loss', 'Actor PG Loss'),
        ('perf/throughput', 'Throughput (tokens/s)'),
        ('time_from_progress', 'Time per Step (s/it from tqdm)')  # <--- 更改了这里
    ]

    # 筛选出两个日志中都存在的指标
    common_metrics = [
        m for m in metrics_to_compare
        if m[0] in df_sync.columns and m[0] in df_async.columns
    ]

    if not common_metrics:
        print("No common metrics found to compare. (Check if 'Training Progress' lines exist in both logs)")
        # 如果进度条提取失败，回退尝试显示原来的指标，防止画不出图
        if 'time_from_progress' not in df_async.columns and 'perf/time_per_step' in df_async.columns:
            print("Fallback: Using original 'perf/time_per_step' for timing.")
            common_metrics.append(('perf/time_per_step', 'Time per Step (Internal Metric)'))
        else:
            return

    # 绘图布局
    num_plots = len(common_metrics)
    cols = 2
    rows = (num_plots + 1) // cols
    if num_plots % cols != 0: rows += 1  # 修正行数计算

    plt.figure(figsize=(16, 5 * rows))

    for i, (key, title) in enumerate(common_metrics):
        plt.subplot(rows, cols, i + 1)

        # 绘制 Sync 曲线
        sync_data = df_sync[['step', key]].dropna()
        if not sync_data.empty:
            plt.plot(sync_data['step'], sync_data[key],
                     label='Synchronous', color='tab:blue', linestyle='-', marker='o', markersize=3, alpha=0.7)

        # 绘制 Async 曲线
        async_data = df_async[['step', key]].dropna()
        if not async_data.empty:
            plt.plot(async_data['step'], async_data[key],
                     label='Asynchronous', color='tab:red', linestyle='-', marker='x', markersize=3, alpha=0.7)

        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel(title)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

    plt.suptitle('Synchronous vs Asynchronous Training Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_file = "comparison_sync_async_tqdm.png"
    plt.savefig(output_file, dpi=150)
    print(f"Comparison chart saved to: {output_file}")
    plt.close()


# --- 运行配置 ---
# 请确保文件名正确
sync_log = "sync_training_log_20260105_171127.txt"
async_log = "async_train_log_20260106_001332.txt"

plot_comparison(sync_log, async_log)