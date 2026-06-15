"""
从 Binance 公开 API 下载比特币近一年 1 分钟 K 线数据
无需 API Key，自动分页 + 限速处理

用法：
    python download_btc_data.py                        # 默认最近365天
    python download_btc_data.py --days 180              # 最近180天
    python download_btc_data.py --start 2025-01-01      # 指定起始日期
"""

import pandas as pd
import requests
import time
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# 配置
# ============================================================
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
BASE_URL = "https://api.binance.com/api/v3/klines"
LIMIT = 1000  # 单次请求最大条数
DATA_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = DATA_DIR / "train_loss" / "btc_1min_1year.csv"

# 精简列名，适合训练
COLUMNS = [
    "open_time",           # 开盘时间 (UTC)
    "open",                # 开盘价
    "high",                # 最高价
    "low",                 # 最低价
    "close",               # 收盘价 (用于预测目标)
    "volume",              # 成交量 (BTC)
    "quote_volume",        # 成交额 (USDT)
    "trades",              # 成交笔数
]


def parse_klines(raw: list) -> list[dict]:
    """将 Binance 原始 K 线数据转为精简字典"""
    result = []
    for k in raw:
        result.append({
            "open_time": pd.to_datetime(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "quote_volume": float(k[7]),
            "trades": int(k[8]),
        })
    return result


def download(start_dt: datetime, end_dt: datetime, output_path: Path):
    """
    分页下载指定时间段的 K 线数据
    start_dt / end_dt: datetime 对象 (UTC)
    """
    current = start_dt
    all_data: list[dict] = []
    req_count = 0

    total_seconds = (end_dt - start_dt).total_seconds()
    print(f"📡 下载区间: {start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  预计行数: ~{int(total_seconds / 60):,}")

    while current < end_dt:
        start_ms = int(current.timestamp() * 1000)
        end_ms = int(min(current + timedelta(hours=16.6), end_dt).timestamp() * 1000)

        try:
            resp = requests.get(BASE_URL, params={
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": LIMIT,
            }, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            print(f"  ⚠️ 请求失败 @ {current.strftime('%Y-%m-%d %H:%M')}: {e}，10秒后重试...")
            time.sleep(10)
            continue

        if not raw:
            # 跳过空白区间
            current = min(current + timedelta(hours=16.6), end_dt)
            continue

        parsed = parse_klines(raw)
        all_data.extend(parsed)

        # 以最后一根 K 线收盘时间 + 1ms 推进
        last_close_ms = raw[-1][6]
        current = datetime.fromtimestamp(last_close_ms / 1000) + timedelta(milliseconds=1)

        req_count += 1
        pct = min(100, (current - start_dt).total_seconds() / total_seconds * 100)
        bar_len = int(pct / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"\r  [{bar}] {pct:5.1f}% | {len(all_data):,} 条 | 请求 #{req_count}", end="")
        sys.stdout.flush()

        # 遵守频率限制（每分钟约 20 次请求）
        if req_count % 18 == 0:
            time.sleep(1)

    print("\n")

    # 构建 DataFrame 并清洗
    df = pd.DataFrame(all_data, columns=COLUMNS)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)

    # 添加辅助列 (本地时间)
    df["open_time_local"] = df["open_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Shanghai")

    df.to_csv(output_path, index=False)

    print(f"✅ 保存完成: {output_path}")
    print(f"   行数: {len(df):,}")
    print(f"   时间范围: {df['open_time'].min()} → {df['open_time'].max()} (UTC)")
    print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   列: {', '.join(COLUMNS)}")

    # 数据预览
    print(f"\n📊 数据预览 (前5行):")
    print(df.head().to_string(max_colwidth=25))
    print(f"\n📊 数据预览 (后5行):")
    print(df.tail().to_string(max_colwidth=25))

    return df


def main():
    parser = argparse.ArgumentParser(description="下载 BTC/USDT 1分钟 K 线 (Binance)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--days", type=int, default=365,
                       help="从今天往前推 N 天 (默认: 365)")
    group.add_argument("--start", type=str, default=None,
                       help="起始日期 YYYY-MM-DD (默认: 365天前)")
    parser.add_argument("--end", type=str, default=None,
                        help="结束日期 YYYY-MM-DD (默认: 今天)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                        help=f"输出文件路径 (默认: {OUTPUT_FILE})")
    args = parser.parse_args()

    # 确定时间范围
    now = datetime.utcnow()
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_dt = now - timedelta(days=args.days)

    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_dt = now

    if start_dt >= end_dt:
        print("❌ 起始日期必须在结束日期之前！")
        sys.exit(1)

    output_path = Path(args.output)
    if output_path.exists():
        print(f"⚠️  文件已存在: {output_path}")
        print(f"   大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        ans = input("   是否覆盖？(y/N): ").strip().lower()
        if ans != 'y':
            print("已取消。")
            sys.exit(0)
        output_path.unlink()

    print("=" * 60)
    print("  比特币 1 分钟线数据下载器 (Binance Public API)")
    print("=" * 60)
    download(start_dt, end_dt, output_path)


if __name__ == "__main__":
    main()
