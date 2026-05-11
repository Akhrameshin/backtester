from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Optional


class Side(str, Enum):
    BID = "bid"
    ASK = "ask"


class OrderStatus(str, Enum):
    NEW = "new"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


_order_id = count(1)


@dataclass
class Order:
    side: Side
    price: float
    size: float
    ts_placed: int
    order_id: int = field(default_factory=lambda: next(_order_id))
    status: OrderStatus = OrderStatus.NEW
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    ts_filled: Optional[int] = None

    @property
    def remaining(self) -> float:
        return max(self.size - self.filled_size, 0.0)

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.OPEN, OrderStatus.PARTIAL)

    def fill(self, size: float, price: float, ts: int) -> float:
        size = min(size, self.remaining)
        if size <= 0:
            return 0.0
        new_total = self.filled_size + size
        self.avg_fill_price = (
            self.avg_fill_price * self.filled_size + price * size
        ) / new_total
        self.filled_size = new_total
        self.ts_filled = ts
        self.status = OrderStatus.FILLED if self.remaining == 0 else OrderStatus.PARTIAL
        return size

    def cancel(self) -> None:
        if self.is_active:
            self.status = OrderStatus.CANCELLED


@dataclass
class Fill:
    order_id: int
    side: Side
    price: float
    size: float
    ts: int
