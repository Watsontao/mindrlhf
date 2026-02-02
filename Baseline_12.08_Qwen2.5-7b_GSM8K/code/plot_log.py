import re
import matplotlib.pyplot as plt
import os

# 配置
LOG_FILE = 'prof_vllm_log/worker_0.log'
OUTPUT_IMAGE = 'training_curve_first_500.png'
PLOT_LIMIT = 500  

def parse_log_and_plot():
    if not os.path.exists(LOG_FILE):
        print(f"错误: 找不到日志文件 {LOG_FILE}")
        return

    steps = []
    rewards_mean = []
    rewards_max = []

    pattern = re.compile(r"Metrics of total step (\d+).*?'reward_mean': ([\d\.]+).*?'reward_max': ([\d\.]+)")

    print("正在读取日志...")
    with open(LOG_FILE, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                r_mean = float(match.group(2))
                r_max = float(match.group(3))
                
                steps.append(step)
                rewards_mean.append(r_mean)
                rewards_max.append(r_max)

    if not steps:
        print("未找到数据。")
        return

    # --- 只取前 PLOT_LIMIT 个数据 ---
    steps = steps[:PLOT_LIMIT]
    rewards_mean = rewards_mean[:PLOT_LIMIT]
    rewards_max = rewards_max[:PLOT_LIMIT]
    # ---------------------------------------

    print(f"正在绘制前 {len(steps)} 步的曲线...")

    plt.figure(figsize=(12, 6))
    
    # 绘制 Mean Reward
    plt.plot(steps, rewards_mean, label='Reward Mean', color='blue', linewidth=1.5)
    
    # 绘制 Max Reward
    plt.plot(steps, rewards_max, label='Reward Max', color='green', linestyle='--', alpha=0.6)

    plt.title(f'Training Reward Curve (First {len(steps)} Steps)')
    plt.xlabel('Step')
    plt.ylabel('Reward Score')
    plt.ylim(0.0, 1.1) 
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='lower right')
    
    plt.savefig(OUTPUT_IMAGE)
    print(f"图表已保存为: {os.path.abspath(OUTPUT_IMAGE)}")

if __name__ == "__main__":
    parse_log_and_plot()