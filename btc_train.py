"""
比特币价格预测 — RNN / LSTM / GRU 对比训练
使用真实 BTC/USDT 1分钟线数据，预测下一分钟收盘价

用法：
    python btc_train.py                          # 默认 LSTM
    python btc_train.py --model GRU              # 使用 GRU
    python btc_train.py --model RNN --epochs 100 # 使用 RNN，100 轮
    python btc_train.py --all                    # 依次训练 RNN, GRU, LSTM
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import sys
import argparse
from pathlib import Path

# ============================================================
# 配置 — 可通过命令行参数覆盖
# ============================================================
DATA_DIR = Path(__file__).resolve().parent
DATA_FILE = DATA_DIR / "train_loss" / "btc_1min_1year.csv"


# ============================================================
# 1. 模型定义（与原始架构保持兼容）
# ============================================================
class BTCPredictor(nn.Module):
    """通用循环神经网络 — 支持 RNN / GRU / LSTM 切换"""

    def __init__(self, cell_type='LSTM', input_size=5, hidden_size=64,
                 num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.cell_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError("cell_type 必须是 'RNN', 'GRU' 或 'LSTM'")

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        if self.cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        # 只取最后一个时间步的输出做预测
        out = self.dropout_layer(out[:, -1, :])
        return self.fc(out)


# ============================================================
# 2. 数据加载与预处理
# ============================================================
class BTCDataLoader:
    """加载 BTC 分钟线，生成训练序列"""

    def __init__(self, csv_path: Path, seq_len=60, pred_col='close',
                 feature_cols=None, train_ratio=0.7, val_ratio=0.15):
        self.seq_len = seq_len
        self.pred_col = pred_col

        if feature_cols is None:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
        self.feature_cols = feature_cols

        # 加载数据
        print(f"📂 加载数据: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['open_time'])
        print(f"   原始行数: {len(df):,}")

        # 检查必要列
        missing = set(feature_cols + [pred_col]) - set(df.columns)
        if missing:
            raise ValueError(f"CSV 缺少列: {missing}")

        # 提取特征和目标
        self.raw_data = df[feature_cols].values.astype(np.float32)
        self.target_col_idx = feature_cols.index(pred_col)

        # 保存时间戳（用于后续可视化）
        self.timestamps = df['open_time'].values
        self.df = df

        # 标准化
        self.mean = self.raw_data.mean(axis=0)
        self.std = self.raw_data.std(axis=0) + 1e-8
        self.norm_data = (self.raw_data - self.mean) / self.std

        # 生成序列
        self.X, self.y = self._make_sequences(self.norm_data)
        print(f"   特征维度: {self.X.shape} (样本数, 序列长度, 特征数)")
        print(f"   目标维度: {self.y.shape}")

        # 时序切分（不打乱）
        n = len(self.X)
        self.train_end = int(n * train_ratio)
        self.val_end = int(n * (train_ratio + val_ratio))

        self.X_train = self.X[:self.train_end]
        self.y_train = self.y[:self.train_end]
        self.X_val = self.X[self.train_end:self.val_end]
        self.y_val = self.y[self.train_end:self.val_end]
        self.X_test = self.X[self.val_end:]
        self.y_test = self.y[self.val_end:]

        print(f"   训练集: {len(self.X_train):,} | 验证集: {len(self.X_val):,} | 测试集: {len(self.X_test):,}")

        # 反标准化辅助
        self.y_mean = self.mean[self.target_col_idx]
        self.y_std = self.std[self.target_col_idx]

    def _make_sequences(self, data):
        X_list, y_list = [], []
        for i in range(len(data) - self.seq_len):
            X_list.append(data[i:i + self.seq_len])           # 过去 seq_len 分钟
            y_list.append(data[i + self.seq_len, self.target_col_idx])  # 下一分钟目标列
        return np.array(X_list), np.array(y_list).reshape(-1, 1)

    def inverse_y(self, y_norm):
        """将标准化的预测值还原为真实价格"""
        return y_norm * self.y_std + self.y_mean

    def get_train_batches(self, batch_size=256, device='cpu'):
        """生成训练 batch（时序不打乱）"""
        n = len(self.X_train)
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            yield (
                torch.from_numpy(self.X_train[i:end]).to(device),
                torch.from_numpy(self.y_train[i:end]).to(device),
            )


# ============================================================
# 3. 训练主逻辑（加入早停）
# ============================================================
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, model_type, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_type': model_type,
    }
    torch.save(checkpoint, path)
    print(f"   💾 检查点已保存 → {path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    data = BTCDataLoader(
        csv_path=Path(args.data),
        seq_len=args.seq_len,
        pred_col=args.target,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    # 创建 DataLoader 用于验证和测试（分批加载，避免 OOM）
    val_dataset = TensorDataset(
        torch.from_numpy(data.X_val),
        torch.from_numpy(data.y_val)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(data.X_test),
        torch.from_numpy(data.y_test)
    )
    # 使用与训练相同的 batch_size 进行验证和测试
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    # 创建模型
    input_size = len(data.feature_cols)
    model = BTCPredictor(
        cell_type=args.model,
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 模型: {args.model} | 总参数: {total_params:,} | 可训练: {trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 断点续训
    ckpt_path = DATA_DIR / "train_model" / f"btc_{args.model.lower()}_checkpoint.pth"
    start_epoch = 0
    best_val_loss = float('inf')
    if ckpt_path.exists() and args.resume:
        print(f"🔄 恢复训练: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"   从 Epoch {start_epoch} 继续 | 最佳 Val Loss: {best_val_loss:.6f}")

    # 验证函数（分批计算）
    def validate(loader):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)  # 恢复总损失
        avg_loss = total_loss / len(loader.dataset)
        model.train()
        return avg_loss

    # 计算测试集 MAE（分批推理后合并）
    def compute_test_mae(loader):
        model.eval()
        preds_list = []
        trues_list = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                pred_batch = model(X_batch).cpu().numpy()
                preds_list.append(pred_batch)
                trues_list.append(y_batch.numpy())
        y_pred_norm = np.concatenate(preds_list, axis=0)
        y_true_norm = np.concatenate(trues_list, axis=0)
        y_pred_real = data.inverse_y(y_pred_norm)
        y_true_real = data.inverse_y(y_true_norm)
        mae = np.mean(np.abs(y_pred_real - y_true_real))
        model.train()
        return mae

    # 训练循环
    print(f"\n{'=' * 60}")
    print(f"  🚀 开始训练 {args.model} | Epochs: {args.epochs} | Batch: {args.batch}")
    print(f"  序列长度: {args.seq_len} 分钟 | 预测目标: {args.target}")
    print(f"  早停: patience={args.patience}, min_delta={args.min_delta}")
    print(f"{'=' * 60}\n")

    train_losses, val_losses = [], []
    early_stop_counter = 0          # 早停计数器
    best_model_path = DATA_DIR / "train_model" / f"btc_{args.model.lower()}_best.pth"

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in data.get_train_batches(args.batch, device):
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            val_loss = validate(val_loader)
            scheduler.step(val_loss)

            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"   ⭐ 最佳模型 (Val Loss: {val_loss:.6f}) → {best_model_path}")
                early_stop_counter = 0   # 改善则重置早停计数
            else:
                early_stop_counter += 1

            # 每个 epoch 输出（每5轮输出一次，第一轮也输出）
            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                test_mae = compute_test_mae(test_loader)
                lr_now = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1:4d}/{args.epochs} | "
                      f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                      f"Test MAE: ${test_mae:.2f} | LR: {lr_now:.2e} | "
                      f"EarlyStop: {early_stop_counter}/{args.patience}")

            # 定期存检查点
            if (epoch + 1) % 20 == 0:
                save_checkpoint(model, optimizer, epoch, avg_train_loss, val_loss,
                                args.model, ckpt_path)

            # 早停判断
            if early_stop_counter >= args.patience:
                print(f"\n🛑 早停触发: 验证损失连续 {args.patience} 轮未改善 (best: {best_val_loss:.6f})")
                # 加载之前保存的最佳模型权重
                if best_model_path.exists():
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                    print(f"   🔁 已加载最佳模型权重 → {best_model_path}")
                else:
                    print("   ⚠️ 未找到最佳模型文件，将继续使用当前模型")
                break

    except KeyboardInterrupt:
        print(f"\n⚠️  训练中断！正在保存检查点...")
        save_checkpoint(model, optimizer, epoch, avg_train_loss, val_loss,
                        args.model, ckpt_path)
        print("已安全退出。")
        return

    # 训练结束：如果正常结束（非早停），也加载最佳模型用于测试（确保评估使用最优权重）
    if early_stop_counter < args.patience:
        print("\n🏁 训练正常结束，加载最佳模型权重用于测试...")
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"   🔁 已加载最佳模型权重 → {best_model_path}")
        else:
            print("   ⚠️ 未找到最佳模型文件，将使用最终模型权重")

    # 保存最终权重（可选）
    final_path = DATA_DIR / "train_model" / f"btc_{args.model.lower()}_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ 训练完成！最终权重 → {final_path}")

    # 保存损失历史
    loss_df = pd.DataFrame({'epoch': range(1, len(train_losses) + 1),
                            'train_loss': train_losses, 'val_loss': val_losses})
    loss_df.to_csv(DATA_DIR / "train_loss" / f"btc_{args.model.lower()}_loss.csv", index=False)

    # 清理检查点
    if ckpt_path.exists():
        os.remove(ckpt_path)

    # 最终测试集评估（使用加载的最佳模型）
    print(f"\n{'=' * 60}")
    print(f"  📊 最终测试集评估（使用最佳验证模型）")
    print(f"{'=' * 60}")
    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred_batch = model(X_batch).cpu().numpy()
            preds_list.append(pred_batch)
            trues_list.append(y_batch.numpy())
    y_pred_norm = np.concatenate(preds_list, axis=0)
    y_true_norm = np.concatenate(trues_list, axis=0)
    y_pred_real = data.inverse_y(y_pred_norm)
    y_true_real = data.inverse_y(y_true_norm)

    mae = np.mean(np.abs(y_pred_real - y_true_real))
    rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))
    mape = np.mean(np.abs((y_pred_real - y_true_real) / y_true_real)) * 100

    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")

    print(f"\n  测试集时间范围: {data.timestamps[data.val_end + data.seq_len]} → {data.timestamps[-1]}")


# ============================================================
# 4. 命令行入口（支持 --all 循环训练）
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC 价格预测 — RNN/LSTM/GRU 训练")
    parser.add_argument("--model", type=str, default="LSTM",
                        choices=["RNN", "GRU", "LSTM"],
                        help="循环神经网络类型 (默认: LSTM)")
    parser.add_argument("--all", action="store_true", default=True, help="依次训练 RNN, GRU, LSTM")
    parser.add_argument("--data", type=str, default=str(DATA_FILE),
                        help=f"BTC 数据 CSV (默认: {DATA_FILE})")
    parser.add_argument("--target", type=str, default="close",
                        choices=["open", "high", "low", "close"],
                        help="预测目标列 (默认: close)")
    parser.add_argument("--seq-len", type=int, default=60,
                        help="输入序列长度，即用前 N 分钟预测 (默认: 60)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--batch", type=int, default=256,
                        help="批次大小 (默认: 128)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="隐藏层维度 (默认: 64)")
    parser.add_argument("--layers", type=int, default=2,
                        help="RNN 层数 (默认: 2)")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout 比率 (默认: 0.2)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率 (默认: 0.001)")
    parser.add_argument("--wd", type=float, default=1e-5,
                        help="权重衰减 (默认: 1e-5)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="断点续训 (默认: 开启)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="从头开始训练")
    # 早停参数
    parser.add_argument("--patience", type=int, default=50,
                        help="早停耐心值，连续多少轮验证损失未改善则停止 (默认: 10)")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="最小改善阈值，验证损失下降超过此值才视为改善 (默认: 0.0)")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"❌ 数据文件不存在: {args.data}")
        print(f"   请先运行: python download_btc_data.py")
        sys.exit(1)

    # 如果指定了 --all，则依次训练 RNN, GRU, LSTM
    if args.all:
        models_to_train = ["RNN","GRU", "LSTM"]
        print("=" * 60)
        print("🔥 启用循环训练模式：将依次训练 RNN → GRU → LSTM")
        print("=" * 60)
        for model_type in models_to_train:
            print(f"\n{'=' * 60}")
            print(f"🏁 开始训练 {model_type}")
            print(f"{'=' * 60}")
            args.model = model_type
            train(args)
        print("\n✅ 所有模型训练完成！")
    else:
        train(args)
