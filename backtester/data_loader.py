from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .order_book import OrderBookSnapshot


_LEVEL_RE = re.compile(r"(asks|bids)\[(\d+)\]\.(price|amount)")


@dataclass
class TradeEvent:
    ts: int
    side: str
    price: float
    size: float


class DataLoader:
    def __init__(self, n_levels: int = 5):
        self.n_levels = n_levels

    def _detect_levels(self, columns) -> int:
        max_idx = -1
        for c in columns:
            m = _LEVEL_RE.match(c)
            if m:
                idx = int(m.group(2))
                if idx > max_idx:
                    max_idx = idx
        return max_idx + 1

    def load_lob(self, path: str, max_rows: Optional[int] = None) -> List[OrderBookSnapshot]:
        df = pd.read_csv(path, nrows=max_rows)
        if "local_timestamp" not in df.columns:
            raise ValueError("LOB file missing 'local_timestamp' column")

        avail = self._detect_levels(df.columns)
        n = min(self.n_levels, avail) if avail > 0 else 0
        if n == 0:
            raise ValueError("No bids/asks columns detected in LOB file")

        bid_p_cols = [f"bids[{i}].price" for i in range(n)]
        bid_s_cols = [f"bids[{i}].amount" for i in range(n)]
        ask_p_cols = [f"asks[{i}].price" for i in range(n)]
        ask_s_cols = [f"asks[{i}].amount" for i in range(n)]

        df = df.dropna(subset=["local_timestamp"]).copy()
        df = df.sort_values("local_timestamp").reset_index(drop=True)

        ts_arr = df["local_timestamp"].astype(np.int64).to_numpy()
        bid_p = df[bid_p_cols].to_numpy(dtype=np.float64)
        bid_s = df[bid_s_cols].to_numpy(dtype=np.float64)
        ask_p = df[ask_p_cols].to_numpy(dtype=np.float64)
        ask_s = df[ask_s_cols].to_numpy(dtype=np.float64)

        snaps: List[OrderBookSnapshot] = []
        for i in range(len(df)):
            snaps.append(
                OrderBookSnapshot(
                    ts=int(ts_arr[i]),
                    bid_prices=bid_p[i],
                    bid_sizes=bid_s[i],
                    ask_prices=ask_p[i],
                    ask_sizes=ask_s[i],
                )
            )
        return snaps

    def load_trades(self, path: str, max_rows: Optional[int] = None) -> List[TradeEvent]:
        df = pd.read_csv(path, nrows=max_rows)
        required = {"local_timestamp", "side", "price", "amount"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Trades file missing columns: {missing}")
        df = df.dropna(subset=["side", "price", "amount", "local_timestamp"]).copy()
        df = df.sort_values("local_timestamp").reset_index(drop=True)
        ts = df["local_timestamp"].astype(np.int64).to_numpy()
        side = df["side"].astype(str).to_numpy()
        price = df["price"].astype(np.float64).to_numpy()
        size = df["amount"].astype(np.float64).to_numpy()
        out: List[TradeEvent] = []
        for i in range(len(df)):
            out.append(TradeEvent(int(ts[i]), str(side[i]), float(price[i]), float(size[i])))
        return out


def merge_event_stream(
    snaps: List[OrderBookSnapshot],
    trades: List[TradeEvent],
) -> Iterator[Tuple[str, object]]:
    """Merge-sort LOB snapshots and trades by timestamp."""
    i = j = 0
    while i < len(snaps) and j < len(trades):
        if snaps[i].ts <= trades[j].ts:
            yield ("lob", snaps[i])
            i += 1
        else:
            yield ("trade", trades[j])
            j += 1
    while i < len(snaps):
        yield ("lob", snaps[i])
        i += 1
    while j < len(trades):
        yield ("trade", trades[j])
        j += 1
