#!/usr/bin/env python3
"""
Grouped GEMM TFLOPS plotting utility (Seaborn)

Usage:
    python plot_grouped_gemm.py results.csv
    python plot_grouped_gemm.py results.csv --output perf.png
    python plot_grouped_gemm.py results.csv --no-backward
"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def plot_bar(df, cols, title, output_path=None):
    """
    Plot grouped bar chart for given TFLOPS columns.
    """

    # Wide -> long
    plot_df = df[["Label"] + cols].melt(
        id_vars="Label",
        value_vars=cols,
        var_name="Implementation",
        value_name="TFLOPS",
    )

    plot_df["Implementation"] = plot_df["Implementation"].str.replace(
        " TFLOPS", "", regex=False
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="Label",
        y="TFLOPS",
        hue="Implementation",
        palette="tab10",
    )

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("TFLOPS")
    plt.title(title)
    plt.legend(title="")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Grouped GEMM TFLOPS results")
    parser.add_argument("csv", help="Path to results CSV")
    parser.add_argument(
        "--output",
        help="Output image path (e.g., perf.png). Forward/backward suffixes will be added.",
    )
    parser.add_argument(
        "--no-backward",
        action="store_true",
        help="Skip backward plot",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Load
    df = pd.read_csv(csv_path)

    # Convert TFLOPS columns to numeric
    for col in df.columns:
        if "TFLOPS" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create readable label
    df["Label"] = df.apply(
        lambda r: f"{r['Case']}\nB={r['B']} M={r['M']}",
        axis=1,
    )

    sns.set_theme(style="whitegrid", context="talk")

    # Forward columns
    fwd_cols = [c for c in df.columns if "Forward TFLOPS" in c]
    if not fwd_cols:
        raise RuntimeError("No 'Forward TFLOPS' columns found")

    fwd_out = None
    if args.output:
        fwd_out = str(Path(args.output).with_name(Path(args.output).stem + "_forward.png"))

    plot_bar(df, fwd_cols, "Grouped GEMM TFLOPS (Forward)", fwd_out)

    # Backward
    if not args.no_backward:
        bwd_cols = [c for c in df.columns if "Backward TFLOPS" in c]
        if bwd_cols:
            bwd_out = None
            if args.output:
                bwd_out = str(Path(args.output).with_name(Path(args.output).stem + "_backward.png"))

            plot_bar(df, bwd_cols, "Grouped GEMM TFLOPS (Backward)", bwd_out)


if __name__ == "__main__":
    main()
