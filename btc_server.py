"""
BTC 价格预测 — WebSocket 实时遥测终端
加载训练好的 RNN/LSTM/GRU 模型，流式推送真实价格 + 预测对比

用法：
    python btc_server.py                            # 默认 LSTM
    python btc_server.py --model GRU --port 8765     # 指定模型和端口
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import asyncio
import json
import time
import argparse
from pathlib import Path
import websockets

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).resolve().parent
DATA_FILE = DATA_DIR / "train_loss" / "btc_1min_1year.csv"
DEFAULT_MODEL = "LSTM"      #"RNN", "GRU", "LSTM"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
SEQ_LEN = 60  # 与训练时一致


# ============================================================
# 1. 模型结构（必须与训练时完全一致）
# ============================================================
class BTCPredictor(nn.Module):
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

        out = self.dropout_layer(out[:, -1, :])
        return self.fc(out)


# ============================================================
# 2. 数据预处理（与训练时一致）
# ============================================================
def load_and_prepare_data(csv_path: Path):
    """加载 BTC 数据并做标准化"""
    feature_cols = ['open', 'high', 'low', 'close', 'volume']

    print(f"📂 加载 BTC 数据: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['open_time'])
    print(f"   共 {len(df):,} 行")

    raw_data = df[feature_cols].values.astype(np.float32)
    mean = raw_data.mean(axis=0)
    std = raw_data.std(axis=0) + 1e-8
    norm_data = (raw_data - mean) / std

    return df, norm_data, mean, std, feature_cols


# ============================================================
# 3. WebSocket 数据流
# ============================================================
async def stream_btc_data(websocket, model, df, norm_data, mean, std, feature_cols, model_type):
    """流式推送 BTC 真实价格与模型预测"""
    print(f"\n🔌 客户端已连接，开始推送 {model_type} 预测数据流...")

    device = next(model.parameters()).device
    model.eval()

    # 从测试集起点开始（最近 15% 的数据）
    test_start = int(len(norm_data) * 0.85)
    # 需要前 SEQ_LEN 个点做第一次预测
    start_idx = test_start
    if start_idx < SEQ_LEN:
        start_idx = SEQ_LEN

    total_points = len(norm_data) - start_idx

    print(f"   数据起点: {df['open_time'].iloc[start_idx]}")
    print(f"   数据终点: {df['open_time'].iloc[-1]}")
    print(f"   推送点数: {total_points:,}")

    # 用训练集统计做标准化 / 反标准化的辅助
    close_mean = mean[3]  # close 是第4列
    close_std = std[3]
    vol_mean = mean[4]
    vol_std = std[4]

    prediction_errors = []  # 累积误差用于显示

    try:
        with torch.no_grad():
            for i in range(start_idx, len(norm_data)):
                loop_start = time.time()

                # 取前 SEQ_LEN 分钟数据
                seq = norm_data[i - SEQ_LEN:i]
                input_tensor = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, seq_len, features)

                # 模型预测（标准化空间）
                pred_norm = model(input_tensor).item()

                # 反标准化
                pred_price = pred_norm * close_std + close_mean
                actual_price = float(df['close'].iloc[i])
                actual_open = float(df['open'].iloc[i])
                actual_high = float(df['high'].iloc[i])
                actual_low = float(df['low'].iloc[i])
                actual_volume = float(df['volume'].iloc[i])
                timestamp_str = str(df['open_time'].iloc[i])

                error_abs = abs(actual_price - pred_price)
                prediction_errors.append(error_abs)
                avg_error = np.mean(prediction_errors[-100:]) if prediction_errors else 0

                # 模拟边缘推理延迟
                calc_time = (time.time() - loop_start) * 1000
                simulated_latency = calc_time + np.random.uniform(0.5, 2.0)

                # 构建 JSON 消息 —— 所有数值显式转换为 Python 原生类型
                payload = {
                    "timestamp": timestamp_str,
                    "timestamp_unix": time.time() * 1000,
                    "model_type": model_type,
                    "symbol": "BTC/USDT",
                    "ch1_actual": round(float(actual_price), 2),
                    "ch2_predict": round(float(pred_price), 2),
                    "open": round(float(actual_open), 2),
                    "high": round(float(actual_high), 2),
                    "low": round(float(actual_low), 2),
                    "volume": round(float(actual_volume), 4),
                    "error_abs": round(float(error_abs), 2),
                    "error_avg_100": round(float(avg_error), 2),
                    "error_pct": round(float(error_abs / actual_price * 100), 4) if actual_price > 0 else 0,
                    "progress_pct": round(float((i - start_idx) / total_points * 100), 1),
                    "latency_ms": round(float(simulated_latency), 2),
                }

                await websocket.send(json.dumps(payload))

                # 模拟实时速率：约 30ms 一帧（≈33 Hz 刷新率，适合示波器显示）
                await asyncio.sleep(0.03)

    except websockets.exceptions.ConnectionClosed:
        print(f"\n🔌 客户端断开连接")
    except Exception as e:
        print(f"\n⚠️  异常: {e}")


async def handle_connection(websocket, model, df, norm_data, mean, std, feature_cols, model_type):
    """包装连接处理，支持断线重连"""
    await stream_btc_data(websocket, model, df, norm_data, mean, std, feature_cols, model_type)


async def main(args):
    # 加载数据
    df, norm_data, mean, std, feature_cols = load_and_prepare_data(Path(args.data))

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = len(feature_cols)
    model = BTCPredictor(
        cell_type=args.model,
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=0.0,  # 推理时关闭 dropout
    ).to(device)

    weights_path = DATA_DIR / "train_model" / f"btc_{args.model.lower()}_best.pth"
    if weights_path.exists():
        print(f"✅ 加载权重: {weights_path}")
        # 添加 weights_only=True 可消除 FutureWarning（模型为纯 PyTorch 格式时安全）
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        print(f"⚠️  未找到权重文件 {weights_path}，将使用未训练的模型（随机预测）")

    model.eval()

    # 启动 WebSocket 服务
    print(f"\n{'=' * 60}")
    print(f"  🖥️  BTC 价格预测遥测终端")
    print(f"  🧠  计算核心: {args.model} 神经网络")
    print(f"  🌐  监听地址: ws://{args.host}:{args.port}")
    print(f"  📊  数据范围: {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")
    print(f"{'=' * 60}")
    print("  等待前端监控面板接入...")

    # 使用 partial 传递参数
    async def handler(websocket):
        await handle_connection(websocket, model, df, norm_data, mean, std, feature_cols, args.model)

    async with websockets.serve(handler, args.host, args.port, max_size=2**23):
        await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC 预测 WebSocket 服务端")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=["RNN", "GRU", "LSTM"],
                        help=f"模型类型 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--data", type=str, default=str(DATA_FILE),
                        help=f"BTC 数据路径 (默认: {DATA_FILE})")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                        help=f"监听地址 (默认: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"监听端口 (默认: {DEFAULT_PORT})")
    parser.add_argument("--hidden", type=int, default=64,
                        help="隐藏层维度，需与训练时一致 (默认: 64)")
    parser.add_argument("--layers", type=int, default=2,
                        help="RNN 层数，需与训练时一致 (默认: 2)")
    args = parser.parse_args()

    asyncio.run(main(args))
