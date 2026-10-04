import numpy as np
import pandas as pd
import pytest

from trading_agent import StrategyConfig, run_backtest
from trading_agent.data import synthetic
from trading_agent.indicators import rsi, sma


def bars(closes, opens=None, lows=None, highs=None):
    closes = np.asarray(closes, float)
    opens = closes if opens is None else np.asarray(opens, float)
    lows = np.minimum(opens, closes) if lows is None else np.asarray(lows, float)
    highs = np.maximum(opens, closes) if highs is None else np.asarray(highs, float)
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 30, dtype=float))
    assert rsi(up, 2).dropna().eq(100).all()
    r = rsi(pd.Series(synthetic(300)["Close"]), 2).dropna()
    assert r.between(0, 100).all()


def test_sma():
    assert sma(pd.Series([1.0, 2, 3, 4]), 2).tolist()[1:] == [1.5, 2.5, 3.5]


def test_no_lookahead():
    df = synthetic(800, seed=3)
    base = run_backtest(df, StrategyConfig())
    cut = 600
    altered = df.copy()
    altered.iloc[cut:, :] *= 0.5  # crash everything after the cut
    alt = run_backtest(altered, StrategyConfig())
    closed_before = [t for t in base.trades if t.exit_date < df.index[cut]]
    assert closed_before == alt.trades[: len(closed_before)]
    pd.testing.assert_series_equal(base.equity.iloc[:cut], alt.equity.iloc[:cut])


# Steady uptrend, then a sharp one-day dip at bar 6 that stays above the 6-bar SMA.
UPTREND_DIP = [10, 12, 14, 16, 18, 20, 17]
DIP_CFG = StrategyConfig(entry_rsi=50, trend_sma=6, exit_sma=2, stop_loss_pct=0.10,
                         commission_bps=0, slippage_bps=0)


def test_entry_at_next_open_and_target_exit():
    closes = UPTREND_DIP + [19, 20]
    opens = UPTREND_DIP + [17.5, 19.5]
    res = run_backtest(bars(closes, opens), DIP_CFG)
    assert res.data["entry_signal"].iloc[6]
    t = res.trades[0]
    assert t.entry_date == res.data.index[7] and t.entry_price == 17.5  # next bar's open
    # Close 19 > SMA2 18 at bar 7 -> sell at bar 8's open.
    assert t.exit_reason == "target" and t.exit_date == res.data.index[8] and t.exit_price == 19.5
    assert res.metrics["win_rate"] == 1.0


def test_no_trade_when_below_trend():
    closes = [10, 12, 14, 16, 18, 20, 12, 13, 14]
    res = run_backtest(bars(closes), DIP_CFG)
    assert not res.data["entry_signal"].iloc[6]
    assert res.trades == []


def test_stop_loss_fills_at_stop_or_gap():
    closes = UPTREND_DIP + [16.8, 15.5, 16.0]
    opens = UPTREND_DIP + [17.0, 18.0, 15.5]
    lows = UPTREND_DIP + [16.5, 15.0, 15.5]
    res = run_backtest(bars(closes, opens, lows), DIP_CFG)
    t = res.trades[0]
    assert t.entry_price == 17.0
    assert t.exit_reason == "stop" and t.exit_price == pytest.approx(15.3)
    assert res.metrics["win_rate"] == 0.0

    opens[8], lows[8] = 14.0, 14.0  # gap straight through the stop
    t = run_backtest(bars(closes, opens, lows), DIP_CFG).trades[0]
    assert t.exit_reason == "stop" and t.exit_price == 14.0


def test_costs_reduce_pnl():
    df = synthetic(1500, seed=11)
    free = run_backtest(df, StrategyConfig(commission_bps=0, slippage_bps=0))
    costly = run_backtest(df, StrategyConfig(commission_bps=5, slippage_bps=10))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_metrics_consistent():
    res = run_backtest(synthetic(2520), StrategyConfig())
    m = res.metrics
    tf = res.trades_frame()
    assert m["trades"] == len(tf)
    assert m["win_rate"] == pytest.approx((tf["pnl"] > 0).mean())
    assert m["max_drawdown"] <= 0
    assert 0 <= m["exposure"] <= 1


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame({"Close": [1.0]}))
    with pytest.raises(ValueError):
        StrategyConfig(position_fraction=2)
