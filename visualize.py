"""
Visualization utilities for dimensionality reduction results.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os


PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def _color_map(labels):
    unique = sorted(set(labels))
    color_dict = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(unique)}
    colors = [color_dict[l] for l in labels]
    return colors, color_dict, unique


def plot_2d(X_reduced, labels=None, title="2D Projection", method="",
            save_path=None, figsize=(9, 7)):
    """Scatter plot of 2-D reduced data."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    if labels is not None:
        colors, color_dict, unique = _color_map(labels)
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                             c=colors, alpha=0.75, s=40, edgecolors="none")
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=color_dict[u], markersize=9, label=str(u))
                   for u in unique]
        legend = ax.legend(handles=handles, title="Class", frameon=True,
                           facecolor="#1E1E2E", edgecolor="#444444",
                           title_fontsize=9, fontsize=8)
        plt.setp(legend.get_texts(), color="#DDDDDD")
        plt.setp(legend.get_title(), color="#FFFFFF")
    else:
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                   color=PALETTE[0], alpha=0.65, s=40, edgecolors="none")

    ax.set_title(f"{title} — {method.upper()}", color="#FFFFFF", fontsize=14, pad=12)
    ax.set_xlabel("Component 1", color="#AAAAAA")
    ax.set_ylabel("Component 2", color="#AAAAAA")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def plot_3d(X_reduced, labels=None, title="3D Projection", method="",
            save_path=None, figsize=(10, 8)):
    """3-D scatter plot of reduced data."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#0F1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0F1117")
    ax.tick_params(colors="#AAAAAA")

    if labels is not None:
        colors, color_dict, unique = _color_map(labels)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                   c=colors, alpha=0.75, s=30, edgecolors="none")
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=color_dict[u], markersize=9, label=str(u))
                   for u in unique]
        legend = ax.legend(handles=handles, title="Class", frameon=True,
                           facecolor="#1E1E2E", edgecolor="#444444",
                           title_fontsize=9, fontsize=8)
        plt.setp(legend.get_texts(), color="#DDDDDD")
        plt.setp(legend.get_title(), color="#FFFFFF")
    else:
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                   color=PALETTE[0], alpha=0.65, s=30, edgecolors="none")

    ax.set_title(f"{title} — {method.upper()}", color="#FFFFFF", fontsize=13, pad=10)
    ax.set_xlabel("C1", color="#AAAAAA", labelpad=6)
    ax.set_ylabel("C2", color="#AAAAAA", labelpad=6)
    ax.set_zlabel("C3", color="#AAAAAA", labelpad=6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def plot_explained_variance(evr, title="PCA — Explained Variance", save_path=None):
    """Bar + cumulative line chart for explained variance ratio."""
    n = len(evr)
    cumulative = np.cumsum(evr) * 100
    bar_vals = evr * 100

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0F1117")
    ax1.set_facecolor("#0F1117")

    bars = ax1.bar(range(1, n + 1), bar_vals, color=PALETTE[0],
                   alpha=0.85, edgecolor="none", label="Individual")
    ax1.set_xlabel("Principal Component", color="#AAAAAA")
    ax1.set_ylabel("Explained Variance (%)", color="#AAAAAA")
    ax1.tick_params(colors="#AAAAAA")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")

    ax2 = ax1.twinx()
    ax2.set_facecolor("#0F1117")
    ax2.plot(range(1, n + 1), cumulative, color=PALETTE[1],
             marker="o", linewidth=2, markersize=5, label="Cumulative")
    ax2.axhline(90, color="#FFFFFF", linestyle="--", linewidth=0.8, alpha=0.4)
    ax2.set_ylabel("Cumulative Variance (%)", color=PALETTE[1])
    ax2.tick_params(colors=PALETTE[1])
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                        facecolor="#1E1E2E", edgecolor="#444444")
    plt.setp(legend.get_texts(), color="#DDDDDD")

    ax1.set_title(title, color="#FFFFFF", fontsize=13, pad=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def plot_correlation_heatmap(X, feature_names=None,
                              title="Feature Correlation Matrix", save_path=None):
    """Seaborn heatmap of feature correlations (input space)."""
    import pandas as pd
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(corr) // 2), max(6, len(corr) // 2)))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    sns.heatmap(corr, ax=ax, annot=len(corr) <= 20, fmt=".2f",
                cmap="coolwarm", center=0,
                linewidths=0.3, linecolor="#333333",
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, color="#FFFFFF", fontsize=13, pad=10)
    plt.xticks(color="#AAAAAA", fontsize=8, rotation=45, ha="right")
    plt.yticks(color="#AAAAAA", fontsize=8, rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def plot_comparison_grid(results_dict, labels=None, save_path=None):
    """
    Grid of 2-D scatter plots, one per method.
    results_dict = {method_name: X_reduced_2d}
    """
    methods = list(results_dict.keys())
    n = len(methods)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    fig.patch.set_facecolor("#0F1117")
    axes = np.array(axes).flatten()

    for i, method in enumerate(methods):
        ax = axes[i]
        ax.set_facecolor("#0F1117")
        X2 = results_dict[method]

        if labels is not None:
            colors, color_dict, unique = _color_map(labels)
            ax.scatter(X2[:, 0], X2[:, 1], c=colors, alpha=0.7, s=25, edgecolors="none")
        else:
            ax.scatter(X2[:, 0], X2[:, 1], color=PALETTE[0],
                       alpha=0.7, s=25, edgecolors="none")

        ax.set_title(method.upper(), color="#FFFFFF", fontsize=12)
        ax.tick_params(colors="#AAAAAA", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Dimensionality Reduction — Method Comparison",
                 color="#FFFFFF", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig
