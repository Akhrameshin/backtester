from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from backtester.data_loader import TradeEvent
from backtester.order_book import OrderBookSnapshot

NS = 1_000_000_000


@dataclass
class Calibration:
    sigma: float
    kappa: float
    A: float
    dt_seconds: float

    def as_dict(self) -> dict:
        return {"sigma": self.sigma, "kappa": self.kappa, "A": self.A, "dt_seconds": self.dt_seconds}


def estimate_sigma(snaps: List[OrderBookSnapshot]) -> Tuple[float, float]:
    """Var(dM)/dt estimator for arithmetic-Brownian vol."""
    mids, ts = [], []
    for s in snaps:
        m = s.mid
        if m is None or m != m:
            continue
        mids.append(m)
        ts.append(s.ts)
    if len(mids) < 3:
        return 0.0, 0.0
    mids = np.asarray(mids, dtype=float)
    ts = np.asarray(ts, dtype=np.int64)
    dt = np.diff(ts) / NS
    dm = np.diff(mids)
    mask = dt > 0
    if not mask.any():
        return 0.0, 0.0
    var_per_sec = float(np.var(dm[mask]) / max(np.mean(dt[mask]), 1e-12))
    sigma = float(np.sqrt(max(var_per_sec, 0.0)))
    return sigma, float(np.mean(dt[mask]))


def estimate_kappa(
    trades: List[TradeEvent],
    snaps: List[OrderBookSnapshot],
    n_bins: int = 12,
) -> Tuple[float, float]:

    if not trades or not snaps:
        return 0.0, 0.0

    snap_ts = np.array([s.ts for s in snaps], dtype=np.int64)
    snap_mid = np.array([s.mid if s.mid is not None else np.nan for s in snaps], dtype=float)
    mask = np.isnan(snap_mid)
    if mask.all():
        return 0.0, 0.0
    if mask.any():
        first = np.argmax(~mask)
        snap_mid[:first] = snap_mid[first]
        for i in range(first + 1, len(snap_mid)):
            if np.isnan(snap_mid[i]):
                snap_mid[i] = snap_mid[i - 1]

    deltas = []
    for tr in trades: #mid in moment
        idx = np.searchsorted(snap_ts, tr.ts, side="right") - 1
        if idx < 0:
            continue
        m = snap_mid[idx]
        if np.isnan(m):
            continue
        d = abs(tr.price - m)
        if d > 0:
            deltas.append(d)
    if len(deltas) < 10:
        return 0.0, 0.0

    deltas = np.asarray(deltas, dtype=float)
    total_sec = max((trades[-1].ts - trades[0].ts) / NS, 1e-9)

    hi = np.quantile(deltas, 0.97)
    if hi <= 0:
        return 0.0, 0.0
    edges = np.linspace(0.0, hi, n_bins + 1)
    counts, _ = np.histogram(deltas, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    rate = len(deltas) / total_sec
    density = counts / max(counts.sum(), 1) / max(width, 1e-12)
    lam = density * rate

    valid = lam > 0
    if valid.sum() < 3:
        return 0.0, rate
    x = centers[valid]
    y = np.log(lam[valid])
    w = counts[valid].astype(float)
    slope, intercept = np.polyfit(x, y, 1, w=w)
    kappa = float(-slope)
    A = float(np.exp(intercept))
    if not np.isfinite(kappa) or kappa <= 0:
        kappa = 0.0
    if not np.isfinite(A) or A <= 0:
        A = 0.0
    return kappa, A


def calibrate_from_data(snaps: List[OrderBookSnapshot], trades: List[TradeEvent]) -> Calibration:
    sigma, dt = estimate_sigma(snaps)
    kappa, A = estimate_kappa(trades, snaps)
    return Calibration(sigma=sigma, kappa=kappa, A=A, dt_seconds=dt)
