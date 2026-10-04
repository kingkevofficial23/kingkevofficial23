# Trading Agent: RSI(2) mean reversion

A small Python trading agent with a backtester that can't see future data. It runs a
classic **high-win-rate** setup: buy sharp short-term dips in an uptrend and sell into
the first bounce.

> **Read this first.** No code can promise an 80–90% win rate on future trades, and a
> high win rate does not by itself mean you make money. A strategy that wins 90% of the
> time loses money if the average loss is more than 9× the average win. This project
> reports win rate **next to** expectancy, profit factor and drawdown so you can see the
> full picture. It is not financial advice. Paper trade before risking real money.

## The strategy

| Rule  | Default |
|-------|---------|
| Trend filter | Close > 200-day SMA (long only) |
| Entry | RSI(2) < 10 → buy at the **next** open |
| Profit exit | Close > 5-day SMA → sell at the next open |
| Time stop | Exit after 10 bars |
| Stop loss | 8% below the entry price (intrabar; fills at the open if the price gaps through) |
| Costs | 1 bp commission + 2 bp slippage per side |

**Why the win rate tends to be high:** it takes profits on the first small bounce and
uses a wide stop, so most trades close slightly green. **Why that's dangerous:** the
losses are rare but large, which is why `avg_loss_pct`, `largest_loss_pct` and
`breakeven_win_rate` are part of every report.

On liquid US index ETFs, versions of this rule set have historically won about 65–80% of
trades. Past results do not hold up reliably. Check it on your own data.

## Usage

```bash
pip install -r requirements.txt

# Backtest (downloads daily bars from Yahoo Finance)
python -m trading_agent backtest --ticker SPY --start 2005-01-01

# Use your own data: CSV with Date,Open,High,Low,Close
python -m trading_agent backtest --csv data/spy.csv --trades-csv trades.csv

# Tune parameters on in-sample data and check them on held-out data
python -m trading_agent sweep --ticker SPY --oos 0.3

# What the strategy would do at the next open (it never places orders)
python -m trading_agent signal --ticker SPY

# Override parameters
python -m trading_agent backtest --ticker QQQ --entry-rsi 5 --stop 0.1 --max-hold 5
```

With no `--ticker` or `--csv`, it runs on synthetic random data. That is only a smoke
test, and the results mean nothing for real markets.

## Reading the report

| Metric | Meaning |
|--------|---------|
| `win_rate` | % of trades that closed with a profit after costs |
| `breakeven_win_rate` | The win rate needed to break even given the average win and loss. Your edge is `win_rate − breakeven_win_rate`. |
| `expectancy_pct` | Average return per trade. **This is the number that must be positive.** |
| `profit_factor` | Gross profit ÷ gross loss. Below 1.0 loses money. |
| `max_drawdown` | Worst peak-to-trough fall in equity |
| `exposure` | Share of days spent in the market |
| `buy_hold_*` | Buying and holding the same asset, for comparison |

### Pushing the win rate higher

You can raise the win rate by lowering `--entry-rsi`, widening `--stop`, or turning the
stop off with `--stop 0`. That usually makes the losing trades much worse. Use `sweep`:
pick parameters from the **in-sample** columns only, then check that the
**out-of-sample** columns hold up. If a setting only looks good in-sample, it is
curve-fit.

## Project layout

```
trading_agent/
  indicators.py  SMA, Wilder RSI
  strategy.py    StrategyConfig + signal generation
  backtest.py    Bar-by-bar engine, trade log, metrics
  data.py        CSV / Yahoo Finance / synthetic data
  agent.py       CLI: backtest, sweep, signal
tests/           Unit tests, including a no-look-ahead test
```

Run the tests with `python -m pytest`.

## Known limitations

- Daily bars only. Intrabar order is guessed: the stop is checked before the close.
- Fractional shares, a single asset, and no leverage, shorting, dividends or taxes.
- Yahoo Finance data is free and convenient but can contain errors. Use a proper data
  vendor before trusting results.
- Before going live, connect `signal` to a broker's **paper-trading** account first and
  compare live fills to the backtest.
