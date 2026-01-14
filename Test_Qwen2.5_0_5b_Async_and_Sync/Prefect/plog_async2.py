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
    if key.endswith('step') and '/' not in key:
        return 'step'
    return key


def plot_training_analysis(log_file_path):
    data = []

    if not os.path.exists(log_file_path):
        print(f"Error: File {log_file_path} not found.")
        return

    print(f"Reading file: {log_file_path}...")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if "step:" not in line: continue
        if " - " not in line: continue

        row = {}
        segments = line.strip().split(' - ')

        for seg in segments:
            if ':' not in seg: continue
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

    if not data:
        print("No data found.")
        return

    df = pd.DataFrame(data)
    df = df.sort_values(by='step')

    # --- 定义要绘制的指标组 (Acc, Reward, Response Length) ---
    metric_groups = [
        {
            'title': 'Accuracy (Validation)',
            'metrics': [
                ('val-core/openai/gsm8k/acc/mean@1', 'Validation Accuracy (GSM8K)')
            ]
        },
        {
            'title': 'Reward Analysis (Train vs Val)',
            'metrics': [
                ('critic/rewards/mean', 'Train Mean Reward'),
                ('val-aux/openai/gsm8k/reward/mean@1', 'Validation Mean Reward'),
                ('critic/rewards/max', 'Train Max Reward')
            ]
        },
        {
            'title': 'Response Length Statistics',
            'metrics': [
                ('response_length/mean', 'Mean Length'),
                ('response_length/max', 'Max Length'),
                ('response_length/min', 'Min Length')
            ]
        }
    ]

    # 绘图设置
    num_groups = len(metric_groups)
    plt.figure(figsize=(15, 5 * num_groups))

    for i, group in enumerate(metric_groups):
        plt.subplot(num_groups, 1, i + 1)

        has_data = False
        for key, label in group['metrics']:
            if key in df.columns:
                # 过滤掉空值
                sub_df = df[['step', key]].dropna()
                if not sub_df.empty:
                    # 样式设置：验证集用圆点强调，均值线加粗
                    marker = 'o' if 'val-' in key else '.'
                    linewidth = 2.5 if 'mean' in key or 'acc' in key else 1
                    alpha = 1.0 if 'mean' in key or 'acc' in key else 0.4
                    linestyle = '--' if 'max' in key or 'min' in key else '-'

                    plt.plot(sub_df['step'], sub_df[key],
                             label=label, marker=marker, markersize=4,
                             linestyle=linestyle, linewidth=linewidth, alpha=alpha)
                    has_data = True

        if has_data:
            plt.title(group['title'], fontsize=14, fontweight='bold')
            plt.xlabel('Step', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(fontsize=10)
        else:
            plt.text(0.5, 0.5, f"No data found for {group['title']}",
                     ha='center', va='center', transform=plt.gca().transAxes, color='gray')

    plt.suptitle(f'Training Analysis (Acc/Reward/Len): {os.path.basename(log_file_path)}', fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    output_png = f"{os.path.splitext(os.path.basename(log_file_path))[0]}_analysis.png"
    plt.savefig(output_png, dpi=150)
    print(f"Chart saved to: {output_png}")
    plt.close()


# --- 执行 ---
# 请确保文件名与您实际的日志文件一致
log_file = "async_train_log_20260106_114550.txt"
plot_training_analysis(log_file)