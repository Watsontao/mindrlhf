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


def plot_reward_curves(log_file_path):
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

            # 分割键值对
            parts = seg.rsplit(':', 1)
            if len(parts) != 2:
                continue

            k, v = parts
            k = clean_key_name(k)

            try:
                # 尝试转换数值，处理可能的非数值情况
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

    # --- 定义要绘制的 Reward 相关指标 ---
    metrics_to_plot = [
        # 1. 训练集奖励 (Critic Reward) - 最核心的指标
        ('critic/rewards/mean', 'Critic Mean Reward (Training)'),

        # 2. 验证集奖励 (Validation Reward) - 真实效果指标
        ('val-aux/openai/gsm8k/reward/mean@1', 'Validation Reward (GSM8K)'),

        # 3. 验证集准确率 (可选，但很有用)
        ('val-core/openai/gsm8k/acc/mean@1', 'Validation Accuracy (GSM8K)')
    ]

    # 筛选存在的指标
    valid_metrics = [m for m in metrics_to_plot if m[0] in df.columns]

    if not valid_metrics:
        print("None of the reward metrics found.")
        return

    # 绘图布局
    num_plots = len(valid_metrics)
    cols = 1  # 竖着排，方便看清楚 Reward 变化
    rows = num_plots

    plt.figure(figsize=(10, 4 * rows))

    for i, (key, title) in enumerate(valid_metrics):
        plt.subplot(rows, cols, i + 1)

        # 提取非空数据
        sub_df = df[['step', key]].dropna()
        if sub_df.empty:
            continue

        # 绘制曲线
        # 验证集数据点较少，用带点的线绘制；训练集数据点多，用实线
        marker = 'o' if 'val-' in key else '.'
        plt.plot(sub_df['step'], sub_df[key], marker=marker, linestyle='-', alpha=0.8, label=key)

        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

    plt.suptitle(f'Reward Analysis: {os.path.basename(log_file_path)}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图片
    base_name = os.path.splitext(os.path.basename(log_file_path))[0]
    output_png = f"{base_name}_reward_curve.png"

    plt.savefig(output_png, dpi=150)
    print(f"Successfully saved reward chart to: {output_png}")
    plt.close()


# --- 执行 ---
# 使用您新上传的日志文件名
target_log_file = "async_train_log_20260104_191644.txt"
plot_reward_curves(target_log_file)