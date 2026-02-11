#!/usr/bin/env python3

import argparse
from pathlib import Path
from tabulate import tabulate

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


FWD_BASELINE = "Pytorch grouped Forward TFLOPS"
BWD_BASELINE = "Pytorch grouped Backward TFLOPS"


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if "TFLOPS" in c:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _make_labels(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: f"{r['Case']} B={r['B']} M={r['M']} N={r['N']} K={r['K']}",
        axis=1,
    )

def _pareto_summary(speed: pd.DataFrame) -> pd.DataFrame:
    # speed: rows=cases, cols=impls, values=speedup vs PyTorch
    # Return: cols=impls, rows=stats
    stats = {}
    for col in speed.columns:
        s = speed[col].dropna().to_numpy()
        if s.size == 0:
            stats[col] = {"median": np.nan, "p10": np.nan, "p90": np.nan, "min": np.nan, "max": np.nan}
            continue
        stats[col] = {
            "median": float(np.median(s)),
            "p10": float(np.percentile(s, 10)),
            "p90": float(np.percentile(s, 90)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
        }
    return pd.DataFrame(stats).loc[["median", "p10", "p90", "min", "max"]]


def _discover_impl_columns(df: pd.DataFrame, baseline_col: str, suffix: str) -> dict:
    impls = {}
    for c in df.columns:
        if not c.endswith(suffix):
            continue
        if c == baseline_col:
            continue
        impl_name = c[: -len(suffix)]
        impls[impl_name] = c
    return impls


def _compute_speedup(df: pd.DataFrame, baseline_col: str, impls: dict) -> pd.DataFrame:
    out = pd.DataFrame({"Label": df["Label"]})
    for impl_name, col in impls.items():
        out[impl_name] = df[col] / df[baseline_col]
    return out.set_index("Label")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Input CSV from benchmark")
    ap.add_argument("--sort-by", default="TE (CK_Tile)",
                    help="Which implementation to sort rows by (default: TE (CK_Tile))")
    ap.add_argument("--ascending", action="store_true",
                    help="Sort ascending (default is descending unless you set this).")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Cap number of rows plotted (0 = all). Consider 60-120 for readability.")
    ap.add_argument("--output", default="",
                    help="Output PNG path. If omitted, derived from CSV name.")
    ap.add_argument("--annot-fontsize", type=float, default=7.0,
                    help="Cell annotation font size (default: 7).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    df = _ensure_numeric(df)

    # Validate baselines
    missing_baselines = [c for c in [FWD_BASELINE, BWD_BASELINE] if c not in df.columns]
    if missing_baselines:
        raise RuntimeError(
            "Missing required baseline columns:\n  " + "\n  ".join(missing_baselines)
        )

    # Discover impls from CSV
    fwd_impls = _discover_impl_columns(df, FWD_BASELINE, " Forward TFLOPS")
    bwd_impls = _discover_impl_columns(df, BWD_BASELINE, " Backward TFLOPS")

    if not fwd_impls and not bwd_impls:
        raise RuntimeError(
            "No implementation TFLOPS columns found besides baselines. "
            "Expected columns like '<Impl> Forward TFLOPS' / '<Impl> Backward TFLOPS'."
        )

    df["Label"] = _make_labels(df)

    # Compute speedups vs PyTorch
    fwd_speed = _compute_speedup(df, FWD_BASELINE, fwd_impls)
    bwd_speed = _compute_speedup(df, BWD_BASELINE, bwd_impls)

    # Choose a sort reference (prefer forward if available, else backward)
    sort_key = args.sort_by
    sort_ref = fwd_speed if fwd_speed is not None else bwd_speed

    if sort_ref is not None and sort_key in sort_ref.columns:
        order = sort_ref.sort_values(sort_key, ascending=args.ascending).index
        fwd_speed = fwd_speed.loc[order]
        bwd_speed = bwd_speed.loc[order]
    else:
        # If requested sort key not found, keep CSV order
        pass

    # Optional cap
    if args.max_rows and args.max_rows > 0:
        fwd_speed = fwd_speed.iloc[: args.max_rows]
        bwd_speed = bwd_speed.iloc[: args.max_rows]

    # Figure layout: two heatmaps (forward + backward)
    panels = []
    panels.append(("Forward speedup vs PyTorch grouped", fwd_speed))
    panels.append(("Backward speedup vs PyTorch grouped", bwd_speed))

    n_panels = len(panels)
    nrows = len(panels[0][1].index) if n_panels else 0

    fig_w = 7.5 * n_panels if n_panels else 10.0
    fig_h = max(8.0, 0.24 * nrows)

    sns.set_theme(style="white", context="talk")
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=1, ncols=n_panels, wspace=1.25)

    def heatmap(ax, data, title):
        vmin = float(np.nanmin(data.to_numpy()))
        vmax = float(np.nanmax(data.to_numpy()))

        sns.heatmap(
            data,
            ax=ax,
            cmap="vlag",
            center=1.0,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": args.annot_fontsize},
            cbar_kws={"label": "Speedup vs PyTorch"},
            yticklabels=1,  # request every row
        )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Force all row labels to be shown
        ax.set_yticks(np.arange(len(data.index)) + 0.5)
        ax.set_yticklabels(list(data.index), rotation=0)

        if len(data.index) > 40:
            ax.tick_params(axis="y", labelsize=6)
        else:
            ax.tick_params(axis="y", labelsize=8)

        ax.tick_params(axis="x", labelsize=10)

    for i, (title, data) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        heatmap(ax, data, title)

    # Output
    if args.output:
        out_path = Path(args.output)
    else:
        stem = csv_path.with_suffix("").name
        out_path = csv_path.with_name(f"{stem}_heatmap_fwd_bwd_vs_pytorch.png")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Summaries
    fwd_summary = _pareto_summary(fwd_speed)
    bwd_summary = _pareto_summary(bwd_speed)

    print("\n=== Forward speedup vs PyTorch (Pareto summary) ===")
    print(tabulate(fwd_summary, headers="keys", tablefmt="github", floatfmt=".3f"))

    print("\n=== Backward speedup vs PyTorch (Pareto summary) ===")
    print(tabulate(bwd_summary, headers="keys", tablefmt="github", floatfmt=".3f"))


if __name__ == "__main__":
    main()
