from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class OrderBookSnapshot:
    ts: int
    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray

    @property
    def best_bid(self) -> Optional[float]:
        return float(self.bid_prices[0]) if len(self.bid_prices) and not np.isnan(self.bid_prices[0]) else None

    @property
    def best_ask(self) -> Optional[float]:
        return float(self.ask_prices[0]) if len(self.ask_prices) and not np.isnan(self.ask_prices[0]) else None

    @property
    def best_bid_size(self) -> float:
        return float(self.bid_sizes[0]) if len(self.bid_sizes) and not np.isnan(self.bid_sizes[0]) else 0.0

    @property
    def best_ask_size(self) -> float:
        return float(self.ask_sizes[0]) if len(self.ask_sizes) and not np.isnan(self.ask_sizes[0]) else 0.0

    @property
    def mid(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return 0.5 * (a + b)

    @property
    def spread(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return a - b

    @property
    def imbalance(self) -> Optional[float]:
        bs, as_ = self.best_bid_size, self.best_ask_size
        if bs + as_ == 0:
            return None
        return bs / (bs + as_)

    @property
    def weighted_mid(self) -> Optional[float]:
        I = self.imbalance
        if I is None or self.best_bid is None or self.best_ask is None:
            return None
        return I * self.best_ask + (1 - I) * self.best_bid


class OrderBook:
    def __init__(self, snapshots: List[OrderBookSnapshot]):
        self._snaps = snapshots

    def __len__(self) -> int:
        return len(self._snaps)

    def __getitem__(self, i: int) -> OrderBookSnapshot:
        return self._snaps[i]

    def __iter__(self):
        return iter(self._snaps)

    def find(self, ts: int) -> Optional[OrderBookSnapshot]:
        lo, hi = 0, len(self._snaps) - 1
        if hi < 0 or self._snaps[0].ts > ts:
            return None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._snaps[mid].ts <= ts:
                lo = mid
            else:
                hi = mid - 1
        return self._snaps[lo]
