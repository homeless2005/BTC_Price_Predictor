"""
绘制 RNN / LSTM / GRU 训练损失曲线对比图
依赖：pip install matplotlib pandas
用法：python plot_training_curves.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 解决 Matplotlib 中文显示问题
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# ============================================================

# 配置
MODELS = ['rnn', 'lstm', 'gru']
COLORS = {'rnn': '#1f77b4', 'lstm': '#ff7f0e', 'gru': '#2ca02c'}
LINESTYLES = {'train': '-', 'val': '--'}
LOSS_DIR = Path(__file__).parent / "train_loss"

def load_loss_data(model):
    csv_path = LOSS_DIR / f'btc_{model}_loss.csv'
    if not csv_path.exists():
        print(f"⚠️ 未找到 {csv_path}，跳过 {model.upper()}")
        return None
    df = pd.read_csv(csv_path)
    if 'epoch' not in df.columns:
        df['epoch'] = range(1, len(df) + 1)
    return df

def plot_combined():
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for model in MODELS:
        df = load_loss_data(model)
        if df is not None:
            plt.plot(df['epoch'], df['train_loss'],
                     label=f'{model.upper()} train',
                     color=COLORS[model], linestyle=LINESTYLES['train'])
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('训练损失对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for model in MODELS:
        df = load_loss_data(model)
        if df is not None:
            plt.plot(df['epoch'], df['val_loss'],
                     label=f'{model.upper()} val',
                     color=COLORS[model], linestyle=LINESTYLES['val'])
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('验证损失对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = LOSS_DIR / 'training_curves_comparison.png'
    plt.savefig(out_path, dpi=150)
    print(f"✅ 对比图已保存: {out_path}")
    plt.show()

def plot_individual():
    for model in MODELS:
        df = load_loss_data(model)
        if df is None:
            continue
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'{model.upper()} 训练与验证损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = LOSS_DIR / f'{model}_loss_curve.png'
        plt.savefig(out_path, dpi=150)
        print(f"✅ {model.upper()} 曲线已保存: {out_path}")
        plt.close()

if __name__ == "__main__":
    print("📊 绘制训练损失曲线对比图...")
    plot_combined()
    plot_individual()
