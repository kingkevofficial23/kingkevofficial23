"""RSI(2) pullback-in-an-uptrend strategy.

Why this design tends to produce a high win rate:
  * It only buys short, sharp dips (RSI(2) very low) while the long-term trend is up
    (close above the 200-day SMA), betting on a snap-back.
  * It exits on the *first* sign of a bounce (close back above a short SMA), so many
    trades bank a small profit quickly.
  * The stop loss is wide, so noise rarely knocks a trade out.

The flip side is the important part: winners are small and the occasional loser is
large. A high win rate says nothing about profitability on its own -- always read
profit factor, expectancy and max drawdown alongside it.
"""

from dataclasses import dataclass

import pandas as pd

from .indicators import rsi, sma


@dataclass(frozen=True)
class StrategyConfig:
    trend_sma: int = 200          # only trade long when close > this SMA
    rsi_period: int = 2
    entry_rsi: float = 10.0       # buy when RSI drops below this
    exit_sma: int = 5             # sell when close rises above this SMA
    max_hold_bars: int = 10       # time stop
    stop_loss_pct: float = 0.08   # hard stop below entry price (0 disables)
    position_fraction: float = 1.0  # fraction of equity committed per trade
    commission_bps: float = 1.0   # per side, in basis points of traded value
    slippage_bps: float = 2.0     # per side, applied against you on every fill

    def __post_init__(self):
        if not 0 < self.position_fraction <= 1:
            raise ValueError("position_fraction must be in (0, 1]")
        if not 0 <= self.stop_loss_pct < 1:
            raise ValueError("stop_loss_pct must be in [0, 1)")
        if self.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be >= 1")

    @property
    def warmup_bars(self) -> int:
        return max(self.trend_sma, self.exit_sma, self.rsi_period + 1)


def compute_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Add indicator and signal columns. Signals at bar t are acted on at bar t+1's open."""
    out = df.copy()
    close = out["Close"]
    out["trend_sma"] = sma(close, cfg.trend_sma)
    out["exit_sma"] = sma(close, cfg.exit_sma)
    out["rsi"] = rsi(close, cfg.rsi_period)
    out["entry_signal"] = (close > out["trend_sma"]) & (out["rsi"] < cfg.entry_rsi)
    out["exit_signal"] = close > out["exit_sma"]
    return out
