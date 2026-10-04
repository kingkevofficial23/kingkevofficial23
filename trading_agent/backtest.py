"""Bar-by-bar backtester with no look-ahead.

Timeline for every bar t:
  1. At the open: fill any order decided at the previous close (with slippage + commission).
  2. During the bar: if long and the low touches the stop, exit at the stop
     (or at the open if the market gapped through it).
  3. At the close: mark equity to market and decide orders for bar t+1's open.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .strategy import StrategyConfig, compute_signals

TRADING_DAYS = 252


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    bars_held: int = 0
    costs: float = 0.0

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares - self.costs

    @property
    def return_pct(self) -> float:
        return self.pnl / (self.entry_price * self.shares)


@dataclass
class BacktestResult:
    config: StrategyConfig
    trades: list
    equity: pd.Series
    data: pd.DataFrame
    open_trade: Optional[Trade] = None
    pending_order: Optional[str] = None  # "BUY" / "SELL" for the next session's open
    metrics: dict = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "bars_held": t.bars_held,
                    "exit_reason": t.exit_reason,
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                }
                for t in self.trades
            ]
        )


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig = StrategyConfig(),
                 initial_capital: float = 10_000.0) -> BacktestResult:
    _validate(df)
    data = compute_signals(df, cfg)
    slip = cfg.slippage_bps / 10_000
    comm = cfg.commission_bps / 10_000

    o, h, l, c = (data[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    entry_sig = data["entry_signal"].to_numpy(bool)
    exit_sig = data["exit_signal"].to_numpy(bool)
    dates = data.index

    cash = initial_capital
    trade: Optional[Trade] = None
    pending: Optional[str] = None
    trade_exit_reason: Optional[str] = None
    trades, equity = [], np.empty(len(data))

    def close_trade(i, raw_price, reason):
        nonlocal cash, trade
        fill = raw_price * (1 - slip)
        fee = fill * trade.shares * comm
        cash += fill * trade.shares - fee
        trade.exit_date, trade.exit_price, trade.exit_reason = dates[i], fill, reason
        trade.costs += fee
        trades.append(trade)
        trade = None

    for i in range(len(data)):
        # 1. Fill orders decided at the previous close.
        if pending == "BUY":
            fill = o[i] * (1 + slip)
            shares = cash * cfg.position_fraction / (fill * (1 + comm))
            fee = fill * shares * comm
            cash -= fill * shares + fee
            trade = Trade(dates[i], fill, shares, costs=fee)
        elif pending == "SELL":
            close_trade(i, o[i], trade_exit_reason)
        pending = None

        # 2. Intrabar stop loss.
        if trade is not None and cfg.stop_loss_pct > 0:
            stop = trade.entry_price * (1 - cfg.stop_loss_pct)
            if l[i] <= stop:
                trade.bars_held += 1
                close_trade(i, min(o[i], stop), "stop")

        # 3. Decide at the close.
        if trade is not None:
            trade.bars_held += 1
            if exit_sig[i]:
                pending, trade_exit_reason = "SELL", "target"
            elif trade.bars_held >= cfg.max_hold_bars:
                pending, trade_exit_reason = "SELL", "time"
        elif i >= cfg.warmup_bars - 1 and entry_sig[i]:
            pending = "BUY"

        equity[i] = cash + (trade.shares * c[i] if trade else 0.0)

    result = BacktestResult(cfg, trades, pd.Series(equity, index=dates, name="equity"),
                            data, open_trade=trade, pending_order=pending)
    result.metrics = compute_metrics(result, initial_capital)
    return result


def compute_metrics(result: BacktestResult, initial_capital: float) -> dict:
    eq = result.equity
    pnls = np.array([t.pnl for t in result.trades])
    rets = np.array([t.return_pct for t in result.trades])
    wins, losses = rets[pnls > 0], rets[pnls <= 0]

    years = max(len(eq) / TRADING_DAYS, 1e-9)
    total_return = eq.iloc[-1] / initial_capital - 1
    daily = eq.pct_change().dropna()
    close = result.data["Close"]

    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = -losses.mean() if len(losses) else 0.0
    gross_win, gross_loss = pnls[pnls > 0].sum(), -pnls[pnls <= 0].sum()

    return {
        "trades": len(pnls),
        "win_rate": len(wins) / len(pnls) if len(pnls) else float("nan"),
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "payoff_ratio": avg_win / avg_loss if avg_loss else float("inf"),
        # The win rate you need just to break even, given the average win/loss sizes.
        "breakeven_win_rate": avg_loss / (avg_win + avg_loss) if (avg_win + avg_loss) else float("nan"),
        "expectancy_pct": rets.mean() if len(rets) else float("nan"),
        "largest_loss_pct": -rets.min() if len(rets) and rets.min() < 0 else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss else float("inf"),
        "total_return": total_return,
        "cagr": (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "sharpe": daily.mean() / daily.std() * np.sqrt(TRADING_DAYS) if daily.std() > 0 else 0.0,
        "exposure": _exposure(result),
        "buy_hold_return": close.iloc[-1] / close.iloc[0] - 1,
        "buy_hold_max_drawdown": (close / close.cummax() - 1).min(),
    }


def _exposure(result: BacktestResult) -> float:
    held = sum(t.bars_held for t in result.trades)
    if result.open_trade:
        held += result.open_trade.bars_held
    return held / len(result.equity)


def _validate(df: pd.DataFrame):
    missing = {"Open", "High", "Low", "Close"} - set(df.columns)
    if missing:
        raise ValueError(f"data is missing columns: {sorted(missing)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("data index must be sorted ascending by date")
    if df[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("data contains NaN prices")
