"""A small, honest mean-reversion trading agent with a bias-free backtester."""

from .strategy import StrategyConfig
from .backtest import BacktestResult, run_backtest

__all__ = ["StrategyConfig", "BacktestResult", "run_backtest"]
