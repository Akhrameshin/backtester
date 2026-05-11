from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import yaml

from analysis import build_report, calibrate_from_data
from analysis.report import compare_reports
from backtester import BacktestEngine, DataLoader
from backtester.metrics import Metrics
from strategies import (
    AvellanedaStoikov2008,
    AvellanedaStoikovMicroprice,
    MicropriceEstimator,
)
from strategies.avellaneda_stoikov import ASParams
from strategies.microprice import weighted_mid_microprice


def _deep_get(d: dict, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.lob:
        cfg.setdefault("data", {})["lob_path"] = args.lob
    if args.trades:
        cfg.setdefault("data", {})["trades_path"] = args.trades
    if args.max_rows is not None:
        cfg.setdefault("data", {})["max_rows"] = args.max_rows
    sp = cfg.setdefault("strategy_params", {})
    if args.gamma is not None:
        sp["gamma"] = args.gamma
    if args.T is not None:
        sp["T_seconds"] = args.T
    if args.order_size is not None:
        sp["order_size"] = args.order_size
    if args.out_dir:
        cfg.setdefault("output", {})["out_dir"] = args.out_dir
    return cfg


def build_strategy(cfg: Dict[str, Any], snaps, sigma: float, kappa: float):
    sp = cfg.get("strategy_params", {})
    params = ASParams(
        gamma=float(sp["gamma"]),
        sigma=sigma,
        kappa=kappa,
        T_seconds=float(sp["T_seconds"]),
        order_size=float(sp["order_size"]),
        tick=float(sp.get("tick", 0.0)),
        min_half_spread=float(sp.get("min_half_spread", 0.0)),
    )
    name = cfg.get("strategy", "as_2008")
    if name == "as_2008":
        return AvellanedaStoikov2008(params)
    if name == "as_microprice":
        mp_cfg = cfg.get("microprice", {})
        if mp_cfg.get("mode", "weighted") == "fitted":
            est = MicropriceEstimator(
                n_imbalance_bins=int(mp_cfg.get("n_imbalance_bins", 6)),
                spread_levels=int(mp_cfg.get("spread_levels", 4)),
                tick=float(mp_cfg.get("tick", 1e-5)),
            )
            est.fit(snaps)
            return AvellanedaStoikovMicroprice.with_fitted_estimator(params, est)
        return AvellanedaStoikovMicroprice.with_weighted_mid(params)
    raise ValueError(f"Unknown strategy: {name}")


def run_one(config_path: str, args: argparse.Namespace) -> Metrics:
    cfg = apply_overrides(load_config(config_path), args)

    data = cfg["data"]
    loader = DataLoader(n_levels=int(data.get("n_levels", 5)))
    snaps = loader.load_lob(data["lob_path"], max_rows=data.get("max_rows"))
    trades = loader.load_trades(data["trades_path"], max_rows=data.get("max_rows"))
    print(f"[{cfg.get('strategy')}] loaded {len(snaps)} LOB rows, {len(trades)} trades")

    cal_cfg = cfg.get("calibration", {})
    if cal_cfg.get("use_data", True):
        cal = calibrate_from_data(snaps, trades)
        sigma = cal_cfg.get("sigma_override") or cal.sigma
        kappa = cal_cfg.get("kappa_override") or cal.kappa
        print(f"  calibrated: sigma={sigma:.6g}, kappa={kappa:.6g}, A={cal.A:.6g}")
    else:
        sigma = float(cal_cfg.get("sigma_override", 1e-6))
        kappa = float(cal_cfg.get("kappa_override", 1.0))

    if sigma <= 0 or not (sigma == sigma):
        sigma = 1e-9
    if kappa <= 0 or not (kappa == kappa):
        kappa = 1.0

    strategy = build_strategy(cfg, snaps, sigma, kappa)

    eng_cfg = cfg.get("engine", {})
    engine = BacktestEngine(
        strategy=strategy,
        snapshots=snaps,
        trades=trades,
        liquidate_at_end=bool(eng_cfg.get("liquidate_at_end", True)),
        max_inventory=eng_cfg.get("max_inventory"),
    )
    metrics = engine.run()

    out_cfg = cfg.get("output", {})
    out_dir = out_cfg.get("out_dir", "results/run")
    label = out_cfg.get("label", cfg.get("strategy", "run"))
    paths = build_report(metrics, out_dir=out_dir, label=label)
    print(f"  summary: {metrics.summary()}")
    print(f"  artefacts: {paths}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="HFT backtester (AS-2008 / AS-microprice)")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--compare-with", default=None, help="Optional second config to compare PnL")
    p.add_argument("--lob", default=None, help="Override LOB CSV path")
    p.add_argument("--trades", default=None, help="Override trades CSV path")
    p.add_argument("--max-rows", type=int, default=None, help="Cap rows loaded for both files")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--T", type=float, default=None, help="Trading horizon (seconds)")
    p.add_argument("--order-size", type=float, default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    primary = run_one(args.config, args)

    if args.compare_with:
        secondary = run_one(args.compare_with, args)
        cmp_dir = os.path.join("results", "comparison")

        s1 = primary.summary()
        s2 = secondary.summary()
        print("\n" + "=" * 65)
        print(f"{'metric':<22} {'primary':>18} {'compare':>18}")
        print("-" * 65)
        for k in s1:
            v1, v2 = s1[k], s2[k]
            print(f"{k:<22} {v1:>18.6g} {v2:>18.6g}")
        print("=" * 65)

        os.makedirs(cmp_dir, exist_ok=True)
        md_path = os.path.join(cmp_dir, "comparison.md")
        with open(md_path, "w") as f:
            f.write("# Strategy comparison\n\n")
            f.write(f"| metric | primary | compare |\n|---|---|---|\n")
            for k in s1:
                f.write(f"| {k} | {s1[k]:.6g} | {s2[k]:.6g} |\n")

        path = compare_reports({"primary": primary, "compare": secondary}, cmp_dir)
        if path:
            print(f"\ncomparison plot: {path}")
        print(f"comparison table: {md_path}")


if __name__ == "__main__":
    main()
