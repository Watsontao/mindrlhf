import matplotlib

# ★★★ 核心修改：必须在 import pyplot 之前强制指定 'Agg' 后端
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_training_metrics(log_file_path):
    data = []

    # 检查文件是否存在
    if not os.path.exists(log_file_path):
        print(f"Error: File {log_file_path} not found.")
        return

    print(f"Reading file: {log_file_path}...")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # 过滤掉不包含 step 信息或指标分隔符的行
        if "step:" not in line or " - " not in line:
            continue

        row = {}
        # 日志格式通常为: "... step:1 - key1:value1 - key2:value2 ..."
        # 使用 " - " 分割各个指标块
        segments = line.strip().split(' - ')

        for seg in segments:
            if ':' not in seg:
                continue

            # 从右侧分割一次，以兼容 key 中可能存在的冒号
            parts = seg.rsplit(':', 1)
            if len(parts) != 2:
                continue

            k, v = parts
            k = k.strip()

            # 尝试将 value 转换为浮点数
            try:
                val = float(v)
            except ValueError:
                continue

            # 清洗 step 键名
            if k == 'step' or k.endswith(' step'):
                k = 'step'
            elif k.endswith('step') and '/' not in k and '_' not in k:
                k = 'step'

            row[k] = val

        # 只有包含 step 的行才记录
        if 'step' in row:
            data.append(row)

    if not data:
        print("No valid metrics data found.")
        return

    # 转换为 DataFrame 并按 step 排序
    df = pd.DataFrame(data)
    df = df.sort_values(by='step')

    # 定义需要绘制的指标列表 (指标Key, 图表标题)
    metrics_to_plot = [
        ('critic/rewards/mean', 'Critic Mean Reward'),
        ('critic/score/mean', 'Critic Mean Score'),
        ('actor/ppo_kl', 'Actor PPO KL'),
        ('actor/entropy', 'Actor Entropy'),
        ('val-aux/openai/gsm8k/reward/mean@1', 'Validation Reward (GSM8K)')
    ]

    # 筛选出当前日志中实际存在的指标
    valid_metrics = [m for m in metrics_to_plot if m[0] in df.columns]

    if not valid_metrics:
        print("None of the target metrics found in the log.")
        return

    # 设置绘图布局
    num_plots = len(valid_metrics)
    cols = 2
    rows = (num_plots + 1) // cols

    # 动态调整图片高度
    plt.figure(figsize=(15, 5 * rows))

    for i, (key, title) in enumerate(valid_metrics):
        plt.subplot(rows, cols, i + 1)

        # 提取非空数据
        sub_df = df[['step', key]].dropna()

        if sub_df.empty:
            continue

        # 绘制曲线
        if 'val-' in key:
            plt.plot(sub_df['step'], sub_df[key], marker='o', linestyle='-', color='orange', label=key)
        else:
            plt.plot(sub_df['step'], sub_df[key], linestyle='-', alpha=0.8, label=key)

        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

    plt.suptitle(f'Training Metrics Analysis: {os.path.basename(log_file_path)}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ★★★ 修改为保存图片
    # 根据输入文件名生成输出图片名，防止重名
    base_name = os.path.splitext(os.path.basename(log_file_path))[0]
    output_png = f"{base_name}_metrics.png"

    plt.savefig(output_png, dpi=150)
    print(f"Successfully saved chart to: {output_png}")

    # 关闭当前的 figure 以释放内存，防止在循环调用时绘图重叠
    plt.close()


# --- 运行部分 ---

# 1. 绘制同步日志
# 请确保该文件在当前目录下，或者使用绝对路径
sync_log_file = "training_log_20260102_235402.txt"
plot_training_metrics(sync_log_file)

# 2. 绘制异步日志 (如需运行请取消注释)
# async_log_file = "async_train_log_20260102_170319.txt"
# plot_training_metrics(async_log_file)