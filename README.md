# ₿ BTC Price Predictor


**使用神经网络实时进行比特币价格预测** — 使用 RNN / GRU / LSTM 预测 BTC 下一分钟收盘价，WebSocket 实时推送到网页前端显示。  

---

https://github.com/user-attachments/assets/184c8e21-c99f-4899-ad56-2d8486dce3e8

## 📁 项目结构

```
.
├── btc_train.py               # 训练脚本 (RNN/GRU/LSTM)
├── btc_server.py              # WebSocket 实时推送服务
├── btc_index.html             # 前端 K 线可视化终端
├── download_btc_data.py       # Binance 历史数据下载
├── loss.py                    # 训练损失曲线对比绘图
│
├── train_loss/                # 数据 & 损失记录
│   ├── btc_1min_1year.csv     # BTC 历史分钟线
│   ├── btc_*_loss.csv         # 各模型训练/验证损失
│   └── *_loss_curve.png       # 损失曲线图
│
└── train_model/               # 训练好的模型权重
    ├── btc_rnn_best.pth
    ├── btc_gru_best.pth
    └── btc_lstm_best.pth
```

## 🚀 快速开始
### 🔧 依赖

```
torch >= 1.10
numpy
pandas
websockets
matplotlib
requests
```
### 1. 安装依赖

```bash
pip install torch numpy pandas websockets matplotlib requests
```

### 2. 下载 BTC 历史数据（仓库内已包含历史数据）

```bash
python download_btc_data.py              # 默认最近 365 天
python download_btc_data.py --days 180   # 最近 180 天
```

> [!NOTE]
> 数据来源为 Binance 公开 API，无需注册或 API Key。下载约 52 万行 1 分钟数据，耗时约 2-3 分钟。

### 3. 训练模型（仓库内已训练一个较好的模型）

```bash
python btc_train.py                      # 默认依次训练RNN → GRU → LSTM
python btc_train.py --model LSTM         # 只训练 LSTM
python btc_train.py --model GRU --epochs 200
```

训练输出：
- `train_model/btc_{model}_best.pth` — 验证集最优权重
- `train_model/btc_{model}_final.pth` — 训练结束权重
- `train_loss/btc_{model}_loss.csv` — 逐 epoch 损失记录

> [!IMPORTANT]
> 模型结构与 `btc_server.py` 推理时需严格一致（hidden_size、num_layers 等参数保持默认即可）。

### 4. 绘制损失曲线（仓库内保存有模型的损失曲线）

```bash
python loss.py
```

生成 `train_loss/` 目录下的 PNG 对比图。

### 5. 启动实时推送服务

```bash
python btc_server.py                     # 默认 LSTM
python btc_server.py --model GRU         # 使用 GRU
```

看到 `等待前端监控面板接入...` 表示就绪。

### 6. 打开前端终端

浏览器打开 `btc_index.html`，即可看到 K 线图 + 预测曲线实时滚动。

| 操作 | 方式 |
|------|------|
| 暂停/继续 | 点击 `LIVE` 按钮 或按 `Space` |


## ⚙️ CLI 参考

### `download_btc_data.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--days` | `365` | 下载从今天往前 N 天的比特币真实数据 |
| `--output` | `train_loss/btc_1min_1year.csv` | 输出路径 |

### `btc_train.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `LSTM` | `RNN` / `GRU` / `LSTM` |
| `--all` | `True` | 依次训练 RNN → GRU → LSTM |
| `--epochs` | `150` | 训练轮数 |
| `--batch` | `256` | 批次大小 |
| `--hidden` | `64` | 隐藏层维度 |
| `--layers` | `2` | RNN 层数 |
| `--dropout` | `0.4` | Dropout 比率 |
| `--lr` | `0.001` | 学习率 |
| `--patience` | `50` | 早停轮数 |
| `--seq-len` | `60` | 输入序列长度（分钟） |

### `btc_server.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `LSTM` | 模型类型 |
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8765` | 监听端口 |
| `--hidden` | `64` | 隐藏层维度（需与训练一致） |
| `--layers` | `2` | RNN 层数（需与训练一致） |

## 🧠 模型架构

```
Input: (batch, 60 min, 5 features)
       │ open, high, low, close, volume
       ▼
┌─ RNN / GRU / LSTM ─────┐
│  hidden_size = 64      │
│  num_layers  = 2       │
│  dropout     = 0.2     │
└─────────┬──────────────┘
          │ last timestep
          ▼
┌─ FC Head ──────────────┐
│  Linear(64 → 32)       │
│  ReLU                  │
│  Linear(32 → 1)        │
└─────────┬──────────────┘
          ▼
Output: 下一分钟 close 预测值
```

三种循环单元共用同一骨架，仅 `cell_type` 切换。

## 📊 WebSocket API

**连接**: `ws://127.0.0.1:8765`

**每帧 JSON** (~30ms 间隔):

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | `str` | 数据时间戳 |
| `model_type` | `str` | 当前模型 `RNN`/`GRU`/`LSTM` |
| `ch1_actual` | `float` | 真实收盘价 |
| `ch2_predict` | `float` | 神经网络预测价 |
| `open` / `high` / `low` | `float` | OHLC 完整数据 |
| `volume` | `float` | 成交量 (BTC) |
| `error_abs` | `float` | 绝对误差 |
| `error_pct` | `float` | 误差百分比 |
| `error_avg_100` | `float` | 近 100 点平均误差 |
| `latency_ms` | `float` | 模拟推理延迟 |
| `progress_pct` | `float` | 数据回放进度 0-100% |





