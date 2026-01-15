import matplotlib
matplotlib.use('Agg')


import re
import matplotlib.pyplot as plt


def parse_and_plot_log(file_path):
    steps = []
    pg_losses = []
    kl_losses = []
    entropies = []

    # 定义正则模式
    # 匹配格式: step:1 - ... actor/pg_loss:-2.5e-06 ...
    step_pattern = re.compile(r"step:(\d+)")
    pg_loss_pattern = re.compile(r"actor/pg_loss:([-\d.eE]+)")
    kl_loss_pattern = re.compile(r"actor/kl_loss:([-\d.eE]+)")
    entropy_pattern = re.compile(r"actor/entropy:([-\d.eE]+)")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            # 仅处理包含 step 数据且非进度条的行
            if "step:" in line and "actor/pg_loss" in line:
                step_match = step_pattern.search(line)
                pg_match = pg_loss_pattern.search(line)
                kl_match = kl_loss_pattern.search(line)
                entropy_match = entropy_pattern.search(line)

                if step_match and pg_match and kl_match:
                    steps.append(int(step_match.group(1)))
                    pg_losses.append(float(pg_match.group(1)))
                    kl_losses.append(float(kl_match.group(1)))

                    if entropy_match:
                        entropies.append(float(entropy_match.group(1)))

        if not steps:
            print("未在文件中找到有效的日志数据，请检查日志格式。")
            return

        # 开始绘图
        plt.figure(figsize=(12, 10))

        # 子图 1: Actor PG Loss
        plt.subplot(3, 1, 1)
        plt.plot(steps, pg_losses, label='Actor PG Loss', color='blue', marker='o', markersize=3)
        plt.title('Training Loss Curves')
        plt.ylabel('PG Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 子图 2: Actor KL Loss
        plt.subplot(3, 1, 2)
        plt.plot(steps, kl_losses, label='Actor KL Loss', color='red', marker='x', markersize=3)
        plt.ylabel('KL Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 子图 3: Actor Entropy (辅助判断收敛情况)
        if entropies:
            plt.subplot(3, 1, 3)
            plt.plot(steps, entropies, label='Actor Entropy', color='green', linestyle='-')
            plt.ylabel('Entropy')
            plt.xlabel('Step')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'，请确保日志文件在当前目录下。")


if __name__ == "__main__":
    # 请将你的日志内容保存为这个文件名，或者修改这里的文件名
    log_filename = "sync_training_log_20260105_171127.txt"
    parse_and_plot_log(log_filename)
    plt.savefig("loss_curve_2.png", dpi=300)
    print("图像已保存为 loss_curve.png")
