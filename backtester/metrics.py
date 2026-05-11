from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Metrics:
    cash: float = 0.0
    inventory: float = 0.0
    turnover: float = 0.0
    n_fills: int = 0

    ts_history: List[int] = field(default_factory=list)
    mid_history: List[float] = field(default_factory=list)
    inv_history: List[float] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)
    cash_history: List[float] = field(default_factory=list)

    def on_fill(self, side: str, price: float, size: float) -> None:
        if side == "bid":
            self.cash -= price * size
            self.inventory += size
        else:
            self.cash += price * size
            self.inventory -= size
        self.turnover += price * size
        self.n_fills += 1

    def snapshot(self, ts: int, mid: float) -> None:
        pnl = self.cash + self.inventory * mid
        self.ts_history.append(ts)
        self.mid_history.append(mid)
        self.inv_history.append(self.inventory)
        self.pnl_history.append(pnl)
        self.cash_history.append(self.cash)

    def final_pnl(self, mid: float) -> float:
        return self.cash + self.inventory * mid

    def summary(self) -> dict:
        pnl = np.asarray(self.pnl_history, dtype=float)
        inv = np.asarray(self.inv_history, dtype=float)
        if len(pnl) < 2:
            return {
                "final_pnl": float(pnl[-1]) if len(pnl) else 0.0,
                "max_drawdown": 0.0, "pnl_std": 0.0, "sharpe_like": 0.0,
                "turnover": self.turnover, "n_fills": self.n_fills,
                "mean_inventory": 0.0, "std_inventory": 0.0, "max_abs_inventory": 0.0,
            }
        d_pnl = np.diff(pnl)
        peaks = np.maximum.accumulate(pnl)
        dd = (peaks - pnl).max()
        std = float(d_pnl.std())
        sharpe = float(d_pnl.mean() / std) if std > 0 else 0.0
        return {
            "final_pnl": float(pnl[-1]),
            "max_drawdown": float(dd),
            "pnl_std": std,
            "sharpe_like": sharpe,
            "turnover": self.turnover,
            "n_fills": self.n_fills,
            "mean_inventory": float(inv.mean()),
            "std_inventory": float(inv.std()),
            "max_abs_inventory": float(np.abs(inv).max()),
        }
