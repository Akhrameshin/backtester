from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from backtester.data_loader import TradeEvent
from backtester.engine import Quote
from backtester.order_book import OrderBookSnapshot


class Strategy(ABC):
    @abstractmethod
    def on_lob(self, snap: OrderBookSnapshot, inventory: float) -> Optional[Quote]:
        ...

    def on_trade(self, trade: TradeEvent, inventory: float) -> None:
        return None

NS = 1_000_000_000


@dataclass
class ASParams:
    gamma: float
    sigma: float
    kappa: float
    T_seconds: float
    order_size: float
    tick: float = 0.0
    min_half_spread: float = 0.0


class AvellanedaStoikov2008(Strategy):
    """AS-2008 approximate solution: eqs 3.10–3.12."""

    def __init__(self, params: ASParams):
        self.p = params
        self._t0_ns: Optional[int] = None

    def _time_left(self, ts_ns: int) -> float:
        if self._t0_ns is None:
            self._t0_ns = ts_ns
        elapsed = (ts_ns - self._t0_ns) / NS
        return max(self.p.T_seconds - elapsed, 0.0)

    def _round(self, p: float) -> float:
        if self.p.tick <= 0:
            return p
        return round(p / self.p.tick) * self.p.tick

    def fair_value(self, snap: OrderBookSnapshot) -> Optional[float]:
        return snap.mid

    def on_lob(self, snap: OrderBookSnapshot, inventory: float) -> Optional[Quote]:
        s = self.fair_value(snap)
        if s is None:
            return None

        tau = self._time_left(snap.ts)
        gamma = self.p.gamma
        sigma2 = self.p.sigma ** 2
        kappa = max(self.p.kappa, 1e-9)

        # r = s - q*gamma*sigma^2*(T-t)
        reservation = s - inventory * gamma * sigma2 * tau

        # delta* = (1/gamma) * ln(1 + gamma/kappa)
        half_spread = (1.0 / gamma) * math.log(1.0 + gamma / kappa)
        if half_spread < self.p.min_half_spread:
            half_spread = self.p.min_half_spread

        bid = reservation - half_spread
        ask = reservation + half_spread

        bb, ba = snap.best_bid, snap.best_ask
        if bb is not None and ask <= bid:
            ask = bid + max(2 * self.p.tick, 1e-12)
        if bb is not None and bid >= ba:
            bid = ba - max(self.p.tick, 1e-12)
        if ba is not None and ask <= bb:
            ask = bb + max(self.p.tick, 1e-12)

        return Quote(
            bid_price=self._round(bid),
            bid_size=self.p.order_size,
            ask_price=self._round(ask),
            ask_size=self.p.order_size,
        )

    def on_trade(self, trade: TradeEvent, inventory: float) -> None:
        return None
