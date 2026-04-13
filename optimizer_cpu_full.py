import argparse
import csv
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================================================
# CONFIG / DATA STRUCTURES
# =========================================================

@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    direction: int  # 1=buy, -1=sell
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    result_points: float
    result_r: float
    exit_reason: str


@dataclass
class Metrics:
    trades: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    total_points: float
    total_r: float
    avg_r: float
    profit_factor: float
    max_dd_r: float
    expectancy_r: float


# =========================================================
# GLOBALS FOR WORKERS
# =========================================================

G_DATA = None


# =========================================================
# DATA LOADING
# =========================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    rename_map = {}

    # aceita variações comuns
    aliases = {
        "open": ["open", "abertura"],
        "high": ["high", "max", "maximum", "alta"],
        "low": ["low", "min", "minimum", "baixa"],
        "close": ["close", "fechamento"],
        "volume": ["volume", "tick_volume", "vol"]
    }

    for target, options in aliases.items():
        found = None
        for opt in options:
            if opt in cols:
                found = cols[opt]
                break
        if found:
            rename_map[found] = target

    df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0

    return df


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    # tenta converter coluna de tempo se existir
    for time_col in ["time", "datetime", "date", "timestamp"]:
        if time_col in df.columns:
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except Exception:
                pass
            break

    df = df.reset_index(drop=True)
    return df


# =========================================================
# SPLITS
# =========================================================

def split_indices(n: int, train_pct: float = 0.60, valid_pct: float = 0.20) -> Dict[str, Tuple[int, int]]:
    train_end = int(n * train_pct)
    valid_end = int(n * (train_pct + valid_pct))

    return {
        "train": (0, train_end),
        "valid": (train_end, valid_end),
        "test": (valid_end, n),
    }


# =========================================================
# STRATEGY BASE
# =========================================================
# Esta é uma estratégia-base para o otimizador funcionar já.
# Depois eu posso trocar esta parte pela sua lógica SMC completa.

def run_strategy_on_slice(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    params: Dict
) -> List[Trade]:
    """
    Estratégia-base:
    - momentum no rompimento da máxima/mínima anterior
    - filtro de candle anterior
    - stop / alvo em pontos
    - timeout
    - filtro de horário
    """

    stop_points = params["stop_points"]
    rr = params["rr"]
    target_points = stop_points * rr
    max_hold_bars = params["max_hold_bars"]
    breakout_buffer = params["breakout_buffer"]
    min_prev_body = params["min_prev_body"]
    use_session = params["use_session"]
    session_start = params["session_start"]
    session_end = params["session_end"]
    slippage_points = params["slippage_points"]

    trades: List[Trade] = []

    time_col = None
    for c in ["time", "datetime", "date", "timestamp"]:
        if c in df.columns:
            time_col = c
            break

    i = max(start_idx + 2, 2)

    while i < end_idx - 2:
        prev = df.iloc[i - 1]
        row = df.iloc[i]

        if time_col is not None and use_session:
            ts = row[time_col]
            hour = pd.Timestamp(ts).hour
            if hour < session_start or hour > session_end:
                i += 1
                continue

        prev_body = abs(float(prev["close"]) - float(prev["open"]))
        if prev_body < min_prev_body:
            i += 1
            continue

        direction = 0
        entry_price = None

        # sinal de compra
        if float(prev["close"]) > float(prev["open"]) and float(row["high"]) >= (float(prev["high"]) + breakout_buffer):
            direction = 1
            entry_price = float(prev["high"]) + breakout_buffer + slippage_points

        # sinal de venda
        elif float(prev["close"]) < float(prev["open"]) and float(row["low"]) <= (float(prev["low"]) - breakout_buffer):
            direction = -1
            entry_price = float(prev["low"]) - breakout_buffer - slippage_points

        if direction == 0 or entry_price is None:
            i += 1
            continue

        if direction == 1:
            stop_price = entry_price - stop_points
            target_price = entry_price + target_points
        else:
            stop_price = entry_price + stop_points
            target_price = entry_price - target_points

        exit_price = None
        exit_reason = None
        exit_idx = None

        last_idx = min(i + max_hold_bars, end_idx - 1)

        for j in range(i + 1, last_idx + 1):
            candle = df.iloc[j]
            high = float(candle["high"])
            low = float(candle["low"])

            if direction == 1:
                stop_hit = low <= stop_price
                target_hit = high >= target_price
                if stop_hit:
                    exit_price = stop_price
                    exit_reason = "stop"
                    exit_idx = j
                    break
                if target_hit:
                    exit_price = target_price
                    exit_reason = "target"
                    exit_idx = j
                    break
            else:
                stop_hit = high >= stop_price
                target_hit = low <= target_price
                if stop_hit:
                    exit_price = stop_price
                    exit_reason = "stop"
                    exit_idx = j
                    break
                if target_hit:
                    exit_price = target_price
                    exit_reason = "target"
                    exit_idx = j
                    break

        if exit_price is None:
            exit_idx = last_idx
            exit_price = float(df.iloc[last_idx]["close"])
            exit_reason = "timeout"

        if direction == 1:
            result_points = exit_price - entry_price
            result_r = result_points / stop_points
        else:
            result_points = entry_price - exit_price
            result_r = result_points / stop_points

        trades.append(
            Trade(
                entry_idx=i,
                exit_idx=exit_idx,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_price=stop_price,
                target_price=target_price,
                result_points=result_points,
                result_r=result_r,
                exit_reason=exit_reason,
            )
        )

        i = exit_idx + 1

    return trades


# =========================================================
# METRICS
# =========================================================

def calc_metrics(trades: List[Trade]) -> Metrics:
    if not trades:
        return Metrics(
            trades=0,
            wins=0,
            losses=0,
            timeouts=0,
            win_rate=0.0,
            total_points=0.0,
            total_r=0.0,
            avg_r=0.0,
            profit_factor=0.0,
            max_dd_r=0.0,
            expectancy_r=0.0,
        )

    results_r = np.array([t.result_r for t in trades], dtype=float)
    results_points = np.array([t.result_points for t in trades], dtype=float)

    wins = int(np.sum(results_r > 0))
    losses = int(np.sum(results_r < 0))
    timeouts = int(np.sum([1 for t in trades if t.exit_reason == "timeout"]))

    gross_profit = float(np.sum(results_r[results_r > 0])) if np.any(results_r > 0) else 0.0
    gross_loss = abs(float(np.sum(results_r[results_r < 0]))) if np.any(results_r < 0) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    equity = np.cumsum(results_r)
    peaks = np.maximum.accumulate(equity) if len(equity) else np.array([0.0])
    dd = peaks - equity if len(equity) else np.array([0.0])
    max_dd_r = float(np.max(dd)) if len(dd) else 0.0

    total_r = float(np.sum(results_r))
    avg_r = float(np.mean(results_r))
    expectancy_r = avg_r
    total_points = float(np.sum(results_points))
    win_rate = (wins / len(trades)) * 100.0

    return Metrics(
        trades=len(trades),
        wins=wins,
        losses=losses,
        timeouts=timeouts,
        win_rate=win_rate,
        total_points=total_points,
        total_r=total_r,
        avg_r=avg_r,
        profit_factor=profit_factor,
        max_dd_r=max_dd_r,
        expectancy_r=expectancy_r,
    )


# =========================================================
# PARAM SAMPLING
# =========================================================

def sample_params(rng: random.Random) -> Dict:
    return {
        "stop_points": rng.randint(20, 200),
        "rr": round(rng.uniform(0.8, 4.0), 2),
        "max_hold_bars": rng.randint(2, 80),
        "breakout_buffer": rng.randint(0, 20),
        "min_prev_body": rng.randint(0, 50),
        "use_session": rng.choice([True, False]),
        "session_start": rng.randint(8, 13),
        "session_end": rng.randint(14, 18),
        "slippage_points": rng.randint(0, 5),
    }


# =========================================================
# SCORING
# =========================================================

def score_metrics(train_m: Metrics, valid_m: Metrics, test_m: Metrics) -> float:
    # trava de amostra mínima
    if train_m.trades < 20:
        return -10000 + train_m.trades
    if valid_m.trades < 8:
        return -5000 + valid_m.trades
    if test_m.trades < 8:
        return -5000 + test_m.trades

    score = 0.0

    # validação e teste valem mais
    score += min(valid_m.profit_factor, 4.0) * 25.0
    score += min(test_m.profit_factor, 4.0) * 30.0

    score += valid_m.total_r * 2.5
    score += test_m.total_r * 3.0

    score += valid_m.avg_r * 30.0
    score += test_m.avg_r * 35.0

    score += (valid_m.win_rate / 100.0) * 5.0
    score += (test_m.win_rate / 100.0) * 5.0

    # penalidade por drawdown
    score -= valid_m.max_dd_r * 2.0
    score -= test_m.max_dd_r * 2.5

    # penalidade por overfit
    pf_gap = abs(valid_m.profit_factor - test_m.profit_factor)
    score -= pf_gap * 8.0

    r_gap = abs(valid_m.total_r - test_m.total_r)
    score -= r_gap * 1.5

    return score


# =========================================================
# WORKER INIT / EVAL
# =========================================================

def worker_init(csv_path: str):
    global G_DATA
    G_DATA = load_csv(csv_path)


def evaluate_candidate(candidate_id: int, params: Dict) -> Dict:
    global G_DATA
    df = G_DATA

    splits = split_indices(len(df), train_pct=0.60, valid_pct=0.20)

    train_trades = run_strategy_on_slice(df, *splits["train"], params)
    valid_trades = run_strategy_on_slice(df, *splits["valid"], params)
    test_trades = run_strategy_on_slice(df, *splits["test"], params)

    train_m = calc_metrics(train_trades)
    valid_m = calc_metrics(valid_trades)
    test_m = calc_metrics(test_trades)

    score = score_metrics(train_m, valid_m, test_m)

    row = {
        "candidate_id": candidate_id,
        "score": score,
        **params,

        "train_trades": train_m.trades,
        "train_pf": train_m.profit_factor,
        "train_total_r": train_m.total_r,
        "train_avg_r": train_m.avg_r,
        "train_win_rate": train_m.win_rate,
        "train_max_dd_r": train_m.max_dd_r,

        "valid_trades": valid_m.trades,
        "valid_pf": valid_m.profit_factor,
        "valid_total_r": valid_m.total_r,
        "valid_avg_r": valid_m.avg_r,
        "valid_win_rate": valid_m.win_rate,
        "valid_max_dd_r": valid_m.max_dd_r,

        "test_trades": test_m.trades,
        "test_pf": test_m.profit_factor,
        "test_total_r": test_m.total_r,
        "test_avg_r": test_m.avg_r,
        "test_win_rate": test_m.win_rate,
        "test_max_dd_r": test_m.max_dd_r,
    }

    return row


# =========================================================
# MAIN
# =========================================================

def save_csv(rows: List[Dict], path: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="CPU-heavy optimizer for OHLC strategies")
    parser.add_argument("--csv", type=str, default="wdo_m5.csv", help="Caminho do CSV")
    parser.add_argument("--iterations", type=int, default=5000, help="Número total de candidatos")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1), help="Número de processos")
    parser.add_argument("--seed", type=int, default=42, help="Seed aleatória")
    parser.add_argument("--save-every", type=int, default=100, help="Salvar parcial a cada N resultados")
    parser.add_argument("--top-k", type=int, default=100, help="Salvar top K")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV não encontrado: {args.csv}")

    print("=" * 80)
    print("CPU OPTIMIZER START")
    print(f"CSV: {args.csv}")
    print(f"Iterations: {args.iterations}")
    print(f"Workers: {args.workers}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    rng = random.Random(args.seed)

    candidates = []
    for i in range(args.iterations):
        candidates.append((i, sample_params(rng)))

    results: List[Dict] = []
    started = time.time()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(args.csv,),
    ) as ex:
        futures = [ex.submit(evaluate_candidate, cid, params) for cid, params in candidates]

        for idx, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            results.append(row)

            if idx % 10 == 0:
                elapsed = time.time() - started
                rate = idx / elapsed if elapsed > 0 else 0
                print(f"[{idx}/{args.iterations}] done | best_score={max(r['score'] for r in results):.2f} | {rate:.2f} eval/s")

            if idx % args.save_every == 0:
                save_csv(results, "optimizer_results_all.csv")
                top_df = pd.DataFrame(results).sort_values("score", ascending=False).head(args.top_k)
                top_df.to_csv("optimizer_results_top.csv", index=False)

    save_csv(results, "optimizer_results_all.csv")
    top_df = pd.DataFrame(results).sort_values("score", ascending=False).head(args.top_k)
    top_df.to_csv("optimizer_results_top.csv", index=False)

    print("\nTOP 10")
    print(top_df.head(10).to_string(index=False))
    print("\nArquivos salvos:")
    print("- optimizer_results_all.csv")
    print("- optimizer_results_top.csv")


if __name__ == "__main__":
    main()