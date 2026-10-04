"""Command-line trading agent.

  python -m trading_agent backtest --ticker SPY
  python -m trading_agent sweep    --ticker SPY
  python -m trading_agent signal   --ticker SPY

`signal` tells you what the strategy would do at the next open. It never places orders.
"""

import argparse
import itertools
from dataclasses import replace

import pandas as pd

from . import data as data_mod
from .backtest import run_backtest
from .strategy import StrategyConfig


def main(argv=None):
    p = argparse.ArgumentParser(prog="trading_agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_ in [("backtest", "run a backtest and print metrics"),
                        ("sweep", "compare parameter sets in-sample vs out-of-sample"),
                        ("signal", "show the action for the next session")]:
        sp = sub.add_parser(name, help=help_)
        src = sp.add_mutually_exclusive_group()
        src.add_argument("--ticker", help="download daily bars from Yahoo Finance")
        src.add_argument("--csv", help="CSV with Date,Open,High,Low,Close")
        sp.add_argument("--start", default="2005-01-01")
        sp.add_argument("--capital", type=float, default=10_000.0)
        sp.add_argument("--entry-rsi", type=float, default=StrategyConfig.entry_rsi)
        sp.add_argument("--stop", type=float, default=StrategyConfig.stop_loss_pct,
                        help="stop loss as a fraction, e.g. 0.08 (0 disables)")
        sp.add_argument("--max-hold", type=int, default=StrategyConfig.max_hold_bars)
        sp.add_argument("--oos", type=float, default=0.3,
                        help="fraction of history held out as out-of-sample (sweep)")
        if name == "backtest":
            sp.add_argument("--trades-csv", help="write the trade list to this file")

    args = p.parse_args(argv)
    df = _load(args)
    cfg = StrategyConfig(entry_rsi=args.entry_rsi, stop_loss_pct=args.stop,
                         max_hold_bars=args.max_hold)
    {"backtest": cmd_backtest, "sweep": cmd_sweep, "signal": cmd_signal}[args.command](df, cfg, args)


def _load(args) -> pd.DataFrame:
    if args.ticker:
        return data_mod.download(args.ticker, start=args.start)
    if args.csv:
        return data_mod.load_csv(args.csv)
    print("No --ticker/--csv given: using SYNTHETIC demo data (results are meaningless "
          "for real trading).\n")
    return data_mod.synthetic()


def cmd_backtest(df, cfg, args):
    res = run_backtest(df, cfg, args.capital)
    print(f"Period: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} bars)\n")
    print(format_metrics(res.metrics))
    if res.trades:
        reasons = res.trades_frame()["exit_reason"].value_counts().to_dict()
        print(f"\nExit reasons: {reasons}")
    if args.trades_csv:
        res.trades_frame().to_csv(args.trades_csv, index=False)
        print(f"Trades written to {args.trades_csv}")


def cmd_sweep(df, cfg, args):
    split = int(len(df) * (1 - args.oos))
    # Give the out-of-sample run the warm-up history it needs, but only score OOS trades.
    is_df = df.iloc[:split]
    oos_start = df.index[split]
    rows = []
    for entry_rsi, stop, hold in itertools.product([5, 10, 15, 25], [0.0, 0.05, 0.10], [5, 10]):
        c = replace(cfg, entry_rsi=entry_rsi, stop_loss_pct=stop, max_hold_bars=hold)
        ins = run_backtest(is_df, c, args.capital).metrics
        full = run_backtest(df, c, args.capital)
        oos_trades = [t for t in full.trades if t.entry_date >= oos_start]
        oos_rets = pd.Series([t.return_pct for t in oos_trades], dtype=float)
        rows.append({
            "entry_rsi": entry_rsi, "stop": stop, "hold": hold,
            "IS_trades": ins["trades"], "IS_win%": 100 * ins["win_rate"],
            "IS_exp%": 100 * ins["expectancy_pct"], "IS_PF": ins["profit_factor"],
            "OOS_trades": len(oos_rets),
            "OOS_win%": 100 * (oos_rets > 0).mean() if len(oos_rets) else float("nan"),
            "OOS_exp%": 100 * oos_rets.mean() if len(oos_rets) else float("nan"),
        })
    table = pd.DataFrame(rows).sort_values("IS_exp%", ascending=False)
    print(f"In-sample: {df.index[0].date()} -> {df.index[split - 1].date()}   "
          f"Out-of-sample: {oos_start.date()} -> {df.index[-1].date()}\n")
    with pd.option_context("display.width", 140, "display.float_format", "{:.2f}".format):
        print(table.to_string(index=False))
    print("\nPick parameters on the in-sample columns only, then check they hold up out-of-sample."
          "\nA high win% with negative expectancy (exp%) loses money.")


def cmd_signal(df, cfg, args):
    res = run_backtest(df, cfg, args.capital)
    last = res.data.iloc[-1]
    print(f"As of close {res.data.index[-1].date()}: close={last['Close']:.2f} "
          f"RSI({cfg.rsi_period})={last['rsi']:.1f} SMA{cfg.trend_sma}={last['trend_sma']:.2f} "
          f"SMA{cfg.exit_sma}={last['exit_sma']:.2f}")
    t = res.open_trade
    if t:
        stop = t.entry_price * (1 - cfg.stop_loss_pct) if cfg.stop_loss_pct else None
        print(f"Strategy is LONG since {t.entry_date.date()} @ {t.entry_price:.2f} "
              f"({t.bars_held} bars), unrealised {last['Close'] / t.entry_price - 1:+.2%}"
              + (f", stop {stop:.2f}" if stop else ""))
    if res.pending_order == "BUY":
        print("ACTION: BUY at next open")
    elif res.pending_order == "SELL":
        print("ACTION: SELL at next open")
    else:
        print("ACTION: HOLD" if t else "ACTION: stay flat")
    print("\n(Informational only -- no orders are placed.)")


def format_metrics(m: dict) -> str:
    pct = {"win_rate", "avg_win_pct", "avg_loss_pct", "breakeven_win_rate", "expectancy_pct",
           "largest_loss_pct", "total_return", "cagr", "max_drawdown", "exposure",
           "buy_hold_return", "buy_hold_max_drawdown"}
    lines = []
    for k, v in m.items():
        val = f"{v:.2%}" if k in pct else (f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append(f"  {k:<22} {val}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
