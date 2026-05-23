#!/usr/bin/env python3
"""
main.py — Entry point for the Dimensionality Reduction project.

Usage examples:
  python main.py --dataset iris --method pca
  python main.py --dataset digits --method tsne --n_components 2
  python main.py --dataset swiss_roll --method umap --n_components 3
  python main.py --dataset iris --compare_all
  python main.py --csv my_data.csv --label_col target --method pca
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import DATASETS, load_from_csv
from reducer import reduce, explained_variance_ratio, SUPPORTED_METHODS
from evaluate import evaluate
from visualize import (
    plot_2d, plot_3d, plot_explained_variance,
    plot_correlation_heatmap, plot_comparison_grid,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BANNER = """
╔══════════════════════════════════════════════════════╗
║     Dimensionality Reduction Project — Python        ║
║  PCA · t-SNE · UMAP · LDA · KernelPCA · Isomap      ║
╚══════════════════════════════════════════════════════╝
"""


def run_single(X, y, feature_names, target_names, method, n_components, tag):
    print(f"\n  ▶ Running {method.upper()} (n_components={n_components}) …")

    X_reduced, reducer, scaler = reduce(
        X, method=method, n_components=n_components, y=y, scale=True
    )

    # ── Evaluation ──────────────────────────────────────────────────────────
    print("  ▶ Evaluating …")
    # Use scaled X for evaluation if scaler was applied
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    report = evaluate(X_scaled, X_reduced, labels=y, reducer=reducer)
    evr = explained_variance_ratio(reducer)

    print(f"\n  {'Metric':<28}{'Value':>10}")
    print("  " + "─" * 40)
    for k, v in report.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<28}{val:>10}")
    if evr is not None:
        print(f"\n  Explained variance (each component): {np.round(evr * 100, 2)} %")
        print(f"  Cumulative:                           {round(np.sum(evr) * 100, 2)} %")

    # ── Save results CSV ─────────────────────────────────────────────────────
    cols = [f"C{i+1}" for i in range(X_reduced.shape[1])]
    df_out = pd.DataFrame(X_reduced, columns=cols)
    if y is not None:
        if target_names is not None:
            df_out["label"] = [target_names[i] if i < len(target_names) else i for i in y]
        else:
            df_out["label"] = y
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}_{method}_reduced.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n  ✓ Reduced data saved → {csv_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    label_list = list(y) if y is not None else None
    if target_names is not None and label_list is not None:
        label_list = [target_names[i] if i < len(target_names) else str(i)
                      for i in label_list]

    if X_reduced.shape[1] >= 2:
        path_2d = os.path.join(OUTPUT_DIR, f"{tag}_{method}_2d.png")
        plot_2d(X_reduced, labels=label_list,
                title=f"{tag.title()} Dataset", method=method, save_path=path_2d)
        print(f"  ✓ 2-D plot saved   → {path_2d}")

    if X_reduced.shape[1] >= 3:
        path_3d = os.path.join(OUTPUT_DIR, f"{tag}_{method}_3d.png")
        plot_3d(X_reduced, labels=label_list,
                title=f"{tag.title()} Dataset", method=method, save_path=path_3d)
        print(f"  ✓ 3-D plot saved   → {path_3d}")

    if evr is not None:
        path_ev = os.path.join(OUTPUT_DIR, f"{tag}_{method}_variance.png")
        plot_explained_variance(evr, title=f"{method.upper()} Explained Variance",
                                save_path=path_ev)
        print(f"  ✓ Variance chart   → {path_ev}")

    return X_reduced, report


def run_comparison(X, y, feature_names, target_names, tag, n_components=2):
    print("\n  ▶ Running all methods for comparison …")
    results = {}
    all_reports = {}

    label_list = list(y) if y is not None else None
    if target_names is not None and label_list is not None:
        label_list = [target_names[i] if i < len(target_names) else str(i)
                      for i in label_list]

    for method in SUPPORTED_METHODS:
        try:
            X_r, _, _ = reduce(X, method=method, n_components=n_components,
                                y=y, scale=True)
            results[method] = X_r
            print(f"    ✔ {method.upper():<12} done")
        except Exception as e:
            print(f"    ✘ {method.upper():<12} failed: {e}")

    path_grid = os.path.join(OUTPUT_DIR, f"{tag}_comparison_grid.png")
    plot_comparison_grid(results, labels=label_list, save_path=path_grid)
    print(f"\n  ✓ Comparison grid  → {path_grid}")

    # summary table
    rows = []
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    for method, X_r in results.items():
        rep = evaluate(X_scaled, X_r, labels=y)
        rep["method"] = method
        rows.append(rep)

    df_summary = pd.DataFrame(rows).set_index("method")
    summary_path = os.path.join(OUTPUT_DIR, f"{tag}_comparison_summary.csv")
    df_summary.to_csv(summary_path)
    print(f"  ✓ Summary table    → {summary_path}")
    print(f"\n{df_summary.to_string()}\n")


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="Dimensionality Reduction Toolkit",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        default="iris", help="Built-in dataset to use")
    parser.add_argument("--csv", type=str, help="Path to a custom CSV file")
    parser.add_argument("--label_col", type=str, default=None,
                        help="Column name for labels in custom CSV")

    # Method
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default="pca",
                        help="Reduction method")
    parser.add_argument("--n_components", type=int, default=2,
                        help="Target number of dimensions")
    parser.add_argument("--compare_all", action="store_true",
                        help="Run all methods and produce a comparison grid")

    # Viz extras
    parser.add_argument("--heatmap", action="store_true",
                        help="Plot feature correlation heatmap of raw data")

    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.csv:
        print(f"  ▶ Loading custom CSV: {args.csv}")
        X, y, feature_names, target_names = load_from_csv(args.csv, args.label_col)
        tag = os.path.splitext(os.path.basename(args.csv))[0]
    else:
        print(f"  ▶ Loading dataset: {args.dataset}")
        X, y, feature_names, target_names = DATASETS[args.dataset]()
        tag = args.dataset

    print(f"  ✓ Shape: {X.shape}  |  Classes: {len(set(y)) if y is not None else 'N/A'}")

    if feature_names is None:
        feature_names = [f"F{i}" for i in range(X.shape[1])]

    # ── Optional heatmap ──────────────────────────────────────────────────────
    if args.heatmap and X.shape[1] <= 50:
        path_hm = os.path.join(OUTPUT_DIR, f"{tag}_correlation_heatmap.png")
        plot_correlation_heatmap(X, feature_names=feature_names,
                                  title=f"{tag.title()} — Feature Correlations",
                                  save_path=path_hm)
        print(f"  ✓ Heatmap saved    → {path_hm}")

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.compare_all:
        run_comparison(X, y, feature_names, target_names, tag, n_components=2)
    else:
        run_single(X, y, feature_names, target_names,
                   args.method, args.n_components, tag)

    print("\n  ✅ All done. Results saved to ./outputs/\n")


if __name__ == "__main__":
    main()
