from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from backtester.engine import Quote
from backtester.order_book import OrderBookSnapshot

from .avellaneda_stoikov import AvellanedaStoikov2008, ASParams

MicropriceFn = Callable[[OrderBookSnapshot], Optional[float]]


def weighted_mid_microprice(snap: OrderBookSnapshot) -> Optional[float]:
    return snap.weighted_mid


@dataclass
class MicropriceModel:
    n_imb: int
    n_spread: int
    spreads: np.ndarray
    G_star: np.ndarray
    tick_half: float


class MicropriceEstimator:
    """Stoikov-2018 finite-state Markov micro-price."""

    def __init__(self, n_imbalance_bins: int = 6, spread_levels: int = 4, tick: float = 1e-5):
        self.n_imb = n_imbalance_bins
        self.n_spread = spread_levels
        self.tick = tick
        self.tick_half = 0.5 * tick
        self.model: Optional[MicropriceModel] = None

    def _bin_imbalance(self, I: float) -> int:
        if I <= 0:
            return 0
        if I >= 1:
            return self.n_imb - 1
        return min(int(I * self.n_imb), self.n_imb - 1)

    def _bin_spread(self, S: float) -> int:
        idx = int(round(S / self.tick)) - 1
        return max(0, min(idx, self.n_spread - 1))

    def _state_idx(self, i: int, s: int) -> int:
        return s * self.n_imb + i

    def fit(self, snaps: List[OrderBookSnapshot]) -> MicropriceModel:
        n_states = self.n_imb * self.n_spread
        K = np.array([-self.tick, -self.tick_half, self.tick_half, self.tick], dtype=float)
        n_outcomes = len(K)

        Q = np.zeros((n_states, n_states), dtype=float)
        T = np.zeros((n_states, n_states), dtype=float)
        R = np.zeros((n_states, n_outcomes), dtype=float)

        prev_state, prev_mid = None, None

        for snap in snaps:
            mid, sp, imb = snap.mid, snap.spread, snap.imbalance
            if mid is None or sp is None or imb is None or sp <= 0:
                continue
            cur_state = self._state_idx(self._bin_imbalance(imb), self._bin_spread(sp))

            if prev_state is not None and prev_mid is not None:
                dM = mid - prev_mid
                if dM == 0:
                    Q[prev_state, cur_state] += 1
                else:
                    k_idx = int(np.argmin(np.abs(K - dM)))
                    R[prev_state, k_idx] += 1
                    T[prev_state, cur_state] += 1

            prev_state = cur_state
            prev_mid = mid

        # Symmetrise data so B*G1=0
        Q_sym = np.zeros_like(Q)
        T_sym = np.zeros_like(T)
        R_sym = np.zeros_like(R)
        for s in range(self.n_spread):
            for i in range(self.n_imb):
                src = self._state_idx(i, s)
                mir_src = self._state_idx(self.n_imb - 1 - i, s)
                for sp_dst in range(self.n_spread):
                    for i_dst in range(self.n_imb):
                        dst = self._state_idx(i_dst, sp_dst)
                        mir_dst = self._state_idx(self.n_imb - 1 - i_dst, sp_dst)
                        Q_sym[src, dst] += Q[src, dst] + Q[mir_src, mir_dst]
                        T_sym[src, dst] += T[src, dst] + T[mir_src, mir_dst]
                R_sym[src] += R[src] + R[mir_src][::-1]

        Q, T = Q_sym, T_sym
        absorb = R_sym.sum(axis=1)
        denom = Q.sum(axis=1) + absorb
        denom_safe = np.where(denom > 0, denom, 1.0)
        Q = Q / denom_safe[:, None]
        T = T / denom_safe[:, None]
        R = R_sym / np.where(absorb > 0, absorb, 1.0)[:, None]

        # G1 = (I-Q)^-1 * R * K,  B = (I-Q)^-1 * T
        I_mat = np.eye(n_states)
        try:
            inv_IQ = np.linalg.inv(I_mat - Q)
        except np.linalg.LinAlgError:
            inv_IQ = np.linalg.pinv(I_mat - Q)
        G1 = inv_IQ @ (R @ K)
        B = inv_IQ @ T

        # G* = G1 + B*G1 + B^2*G1 + ...
        G_star = G1.copy()
        term = G1.copy()
        for _ in range(30):
            term = B @ term
            if np.max(np.abs(term)) < 1e-9:
                break
            G_star = G_star + term

        model = MicropriceModel(
            n_imb=self.n_imb, n_spread=self.n_spread,
            spreads=np.arange(1, self.n_spread + 1) * self.tick,
            G_star=G_star, tick_half=self.tick_half,
        )
        self.model = model
        return model

    def microprice(self, snap: OrderBookSnapshot) -> Optional[float]:
        if self.model is None or snap.mid is None or snap.spread is None or snap.imbalance is None:
            return None
        i = self._bin_imbalance(snap.imbalance)
        s = self._bin_spread(snap.spread)
        idx = self._state_idx(i, s)
        return float(snap.mid + self.model.G_star[idx])


class AvellanedaStoikovMicroprice(AvellanedaStoikov2008):
    """AS-2008 but with micro-price as fair value instead of raw mid."""

    def __init__(self, params: ASParams, microprice_fn: MicropriceFn):
        super().__init__(params)
        self._fn = microprice_fn

    def fair_value(self, snap: OrderBookSnapshot) -> Optional[float]:
        mp = self._fn(snap)
        return mp if mp is not None else snap.mid

    @classmethod
    def with_weighted_mid(cls, params: ASParams) -> "AvellanedaStoikovMicroprice":
        return cls(params, weighted_mid_microprice)

    @classmethod
    def with_fitted_estimator(cls, params: ASParams, estimator: MicropriceEstimator) -> "AvellanedaStoikovMicroprice":
        return cls(params, estimator.microprice)
