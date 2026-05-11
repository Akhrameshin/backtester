from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from backtester.metrics import Metrics


def build_report(metrics: Metrics, out_dir: str, label: str = "strategy") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    summary = metrics.summary()

    json_path = os.path.join(out_dir, f"{label}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    csv_path = os.path.join(out_dir, f"{label}_trace.csv")
    with open(csv_path, "w") as f:
        f.write("ts,mid,inventory,cash,pnl\n")
        for ts, m, q, c, p in zip(
            metrics.ts_history, metrics.mid_history,
            metrics.inv_history, metrics.cash_history, metrics.pnl_history,
        ):
            f.write(f"{ts},{m},{q},{c},{p}\n")

    paths = {"summary_json": json_path, "trace_csv": csv_path}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = np.asarray(metrics.ts_history, dtype=np.int64)
        t_sec = (ts - ts[0]) / 1e9 if len(ts) else ts

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(t_sec, metrics.pnl_history, color="tab:blue")
        axes[0].set_ylabel("PnL")
        axes[0].grid(alpha=0.3)

        axes[1].plot(t_sec, metrics.inv_history, color="tab:orange")
        axes[1].axhline(0, color="grey", lw=0.5)
        axes[1].set_ylabel("Inventory")
        axes[1].grid(alpha=0.3)

        axes[2].plot(t_sec, metrics.mid_history, color="tab:green")
        axes[2].set_ylabel("Mid-price")
        axes[2].set_xlabel("Time (sec)")
        axes[2].grid(alpha=0.3)

        fig.suptitle(label)
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"{label}_report.png")
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        paths["report_png"] = png_path
    except Exception as e:
        paths["plot_error"] = str(e)

    md_path = os.path.join(out_dir, f"{label}_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# {label}\n\n| metric | value |\n|---|---|\n")
        for k, v in summary.items():
            f.write(f"| {k} | {v:.6g} |\n")
    paths["summary_md"] = md_path

    return paths


def compare_reports(metrics_by_label: Dict[str, Metrics], out_dir: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for label, m in metrics_by_label.items():
        ts = np.asarray(m.ts_history, dtype=np.int64)
        if len(ts) == 0:
            continue
        t_sec = (ts - ts[0]) / 1e9
        ax.plot(t_sec, m.pnl_history, label=label)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("PnL")
    ax.set_title("Strategy comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
