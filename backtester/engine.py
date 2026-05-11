from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .data_loader import TradeEvent, merge_event_stream
from .metrics import Metrics
from .order_book import OrderBookSnapshot
from .orders import Order, Side, OrderStatus


@dataclass
class Quote:
    bid_price: Optional[float]
    bid_size: float
    ask_price: Optional[float]
    ask_size: float


class BacktestEngine:
    def __init__(
        self,
        strategy,
        snapshots: List[OrderBookSnapshot],
        trades: List[TradeEvent],
        liquidate_at_end: bool = True,
        max_inventory: Optional[float] = None,
    ):
        self.strategy = strategy
        self.snapshots = snapshots
        self.trades = trades
        self.metrics = Metrics()
        self.liquidate_at_end = liquidate_at_end
        self.max_inventory = float(max_inventory) if max_inventory is not None else None

        self.bid_order: Optional[Order] = None
        self.ask_order: Optional[Order] = None
        self.last_snap: Optional[OrderBookSnapshot] = None

    def _cancel(self, side: Side) -> None:
        if side == Side.BID and self.bid_order is not None:
            self.bid_order.cancel()
            self.bid_order = None
        if side == Side.ASK and self.ask_order is not None:
            self.ask_order.cancel()
            self.ask_order = None

    def _replace_quote(self, quote: Optional[Quote], ts: int) -> None:
        if quote is None:
            self._cancel(Side.BID)
            self._cancel(Side.ASK)
            return

        if quote.bid_price is None or quote.bid_size <= 0:
            self._cancel(Side.BID)
        else:
            need_replace = (
                self.bid_order is None
                or not self.bid_order.is_active
                or self.bid_order.price != quote.bid_price
                or self.bid_order.remaining != quote.bid_size
            )
            if need_replace:
                self._cancel(Side.BID)
                self.bid_order = Order(Side.BID, quote.bid_price, quote.bid_size, ts)

        if quote.ask_price is None or quote.ask_size <= 0:
            self._cancel(Side.ASK)
        else:
            need_replace = (
                self.ask_order is None
                or not self.ask_order.is_active
                or self.ask_order.price != quote.ask_price
                or self.ask_order.remaining != quote.ask_size
            )
            if need_replace:
                self._cancel(Side.ASK)
                self.ask_order = Order(Side.ASK, quote.ask_price, quote.ask_size, ts)

    def _match_trade(self, trade: TradeEvent) -> None:
        # Aggressor BUY lifts our ASK
        if trade.side == "buy" and self.ask_order is not None and self.ask_order.is_active:
            if trade.price >= self.ask_order.price:
                fill_size = self.ask_order.fill(trade.size, self.ask_order.price, trade.ts)
                if fill_size > 0:
                    self.metrics.on_fill("ask", self.ask_order.price, fill_size)
                if self.ask_order.status == OrderStatus.FILLED:
                    self.ask_order = None

        # Aggressor SELL hits our BID
        if trade.side == "sell" and self.bid_order is not None and self.bid_order.is_active:
            if trade.price <= self.bid_order.price:
                fill_size = self.bid_order.fill(trade.size, self.bid_order.price, trade.ts)
                if fill_size > 0:
                    self.metrics.on_fill("bid", self.bid_order.price, fill_size)
                if self.bid_order.status == OrderStatus.FILLED:
                    self.bid_order = None

    def _enforce_inventory_cap(self, quote: Optional[Quote]) -> Optional[Quote]:
        if self.max_inventory is None or quote is None:
            return quote
        inv = self.metrics.inventory
        # Kill the side that would worsen the position
        if inv >= self.max_inventory:
            quote = Quote(None, 0.0, quote.ask_price, quote.ask_size)
        elif inv <= -self.max_inventory:
            quote = Quote(quote.bid_price, quote.bid_size, None, 0.0)
        return quote

    def run(self) -> Metrics:
        for kind, ev in merge_event_stream(self.snapshots, self.trades):
            if kind == "lob":
                self.last_snap = ev
                quote = self.strategy.on_lob(ev, self.metrics.inventory)
                quote = self._enforce_inventory_cap(quote)
                self._replace_quote(quote, ev.ts)
                if ev.mid is not None:
                    self.metrics.snapshot(ev.ts, ev.mid)
            else:
                self.strategy.on_trade(ev, self.metrics.inventory)
                self._match_trade(ev)

        if self.last_snap is not None and self.last_snap.mid is not None:
            mid = self.last_snap.mid
            if self.liquidate_at_end and self.metrics.inventory != 0:
                px = self.last_snap.best_bid if self.metrics.inventory > 0 else self.last_snap.best_ask
                if px is not None:
                    side = "ask" if self.metrics.inventory > 0 else "bid"
                    size = abs(self.metrics.inventory)
                    self.metrics.on_fill(side, px, size)
            self.metrics.snapshot(self.last_snap.ts, mid)

        return self.metrics
