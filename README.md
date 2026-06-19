#  Dimensionality Reduction — Full Python Project

> A complete, production-grade toolkit for exploring, comparing, and evaluating dimensionality reduction techniques on any tabular dataset.

---

##  Table of Contents

1. [Overview](#overview)
2. [Algorithms Implemented](#algorithms-implemented)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [CLI Reference](#cli-reference)
7. [Module API](#module-api)
8. [Datasets](#datasets)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Output Files](#output-files)
11. [Examples](#examples)
12. [Theoretical Background](#theoretical-background)
13. [Extending the Project](#extending-the-project)

---

## Overview

Dimensionality reduction transforms high-dimensional data into a lower-dimensional representation while preserving as much meaningful structure as possible. This project provides:

- **6 algorithms** — linear, non-linear, manifold, and supervised
- **Rich visualizations** — 2-D/3-D scatter plots, explained-variance charts, correlation heatmaps, side-by-side comparison grids
- **Quantitative evaluation** — trustworthiness, continuity, reconstruction error, silhouette score, Davies–Bouldin index
- **Clean CLI** — run any method on any dataset with one command
- **Importable API** — use `src/` modules in your own notebooks or scripts

---

## Algorithms Implemented

| Method | Type | Key Strength | When to Use |
|---|---|---|---|
| **PCA** | Linear | Fast, interpretable, variance-based | Baseline; high-dim numeric data |
| **t-SNE** | Non-linear manifold | Clusters visible, local structure | Visualization of clusters |
| **UMAP** | Non-linear manifold | Fast, preserves global + local | Large datasets, general viz |
| **LDA** | Supervised linear | Maximizes class separability | Labeled data, classification prep |
| **Kernel PCA** | Non-linear (kernel trick) | Flexible non-linear projection | Non-linearly separable data |
| **Isomap** | Geodesic manifold | Preserves geodesic distances | Curved manifolds (Swiss roll) |

---

## Project Structure

```
dimensionality_reduction/
│
├── main.py                    # CLI entry point
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── reducer.py             # Core reduction logic (all 6 methods)
│   ├── visualize.py           # All plotting functions
│   ├── evaluate.py            # Evaluation metrics
│   └── data_loader.py         # Built-in + custom dataset loaders
│
├── outputs/                   # Auto-created; all plots & CSVs saved here
└── data/                      # Place custom CSV files here
```

---

## Installation

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/dimensionality-reduction.git
cd dimensionality-reduction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `umap-learn` requires a C compiler. If installation fails, install the rest and all other methods will still work:
> ```bash
> pip install numpy pandas scikit-learn matplotlib seaborn plotly kaleido
> ```

---

## Quick Start

```bash
# Run PCA on the Iris dataset
python main.py --dataset iris --method pca

# Run t-SNE with 3 components on MNIST digits
python main.py --dataset digits --method tsne --n_components 3

# Compare ALL methods on Wine dataset (generates a grid plot)
python main.py --dataset wine --compare_all

# Use UMAP on Swiss Roll (classic non-linear manifold)
python main.py --dataset swiss_roll --method umap --n_components 2

# Run on your own CSV file
python main.py --csv data/my_data.csv --label_col species --method pca --heatmap
```

Outputs are saved automatically to `./outputs/`.

---

## CLI Reference

```
python main.py [OPTIONS]

Dataset options:
  --dataset {iris,wine,digits,breast_cancer,swiss_roll,s_curve}
                        Built-in dataset (default: iris)
  --csv PATH            Path to a custom CSV file
  --label_col COL       Column name for class labels in custom CSV

Method options:
  --method {pca,tsne,umap,lda,kernel_pca,isomap}
                        Reduction algorithm (default: pca)
  --n_components INT    Target number of dimensions (default: 2)
  --compare_all         Run all 6 methods and produce a comparison grid

Visualization extras:
  --heatmap             Plot feature correlation heatmap of raw data
```

---

## Module API

Import and use any module directly in your own code:

```python
import numpy as np
from src.reducer import reduce, explained_variance_ratio
from src.visualize import plot_2d, plot_explained_variance
from src.evaluate import evaluate
from src.data_loader import DATASETS

# Load data
X, y, feature_names, target_names = DATASETS["iris"]()

# Reduce to 2D with PCA
X_reduced, reducer, scaler = reduce(X, method="pca", n_components=2, y=y)

# Evaluate
report = evaluate(X, X_reduced, labels=y, reducer=reducer)
print(report)

# Plot
plot_2d(X_reduced, labels=y, title="Iris PCA", method="pca", save_path="my_plot.png")

# Explained variance (PCA / KernelPCA only)
evr = explained_variance_ratio(reducer)
plot_explained_variance(evr, save_path="variance.png")
```

### `reduce()` signature

```python
X_reduced, reducer, scaler = reduce(
    X,                  # array-like (n_samples, n_features)
    method="pca",       # algorithm name
    n_components=2,     # target dimensions
    y=None,             # labels (required for LDA)
    scale=True,         # apply StandardScaler before reducing
    **kwargs            # forwarded to the underlying reducer
)
```

### Supported `**kwargs` per method

| Method | Key kwargs |
|---|---|
| PCA | `svd_solver`, `random_state` |
| t-SNE | `perplexity` (default 30), `random_state` |
| UMAP | `n_neighbors` (default 15), `min_dist` (default 0.1), `random_state` |
| LDA | — |
| Kernel PCA | `kernel` (default `"rbf"`), `gamma` |
| Isomap | `n_neighbors` (default 5) |

---

## Datasets

| Dataset | Samples | Features | Classes | Description |
|---|---|---|---|---|
| `iris` | 150 | 4 | 3 | Classic flower classification |
| `wine` | 178 | 13 | 3 | Wine chemical attributes |
| `digits` | 1 797 | 64 | 10 | Handwritten digit images |
| `breast_cancer` | 569 | 30 | 2 | Tumor classification |
| `swiss_roll` | 1 500 | 3 | — | Classic 3-D manifold |
| `s_curve` | 1 500 | 3 | — | S-shaped manifold |

All built-in datasets come from `sklearn.datasets`. Custom CSV files are supported via `--csv` and `--label_col`.

---

## Evaluation Metrics

| Metric | Range | Interpretation |
|---|---|---|
| **Trustworthiness** | 0–1 | Are low-dim neighbours true neighbours in high-dim? 1 = perfect |
| **Continuity** | 0–1 | Are high-dim neighbours preserved in low-dim? 1 = perfect |
| **Reconstruction Error** | ≥ 0 | MSE of inverse-projection (PCA only); lower = better |
| **Silhouette Score** | −1 – 1 | Cluster separation in reduced space; higher = better |
| **Davies–Bouldin Index** | ≥ 0 | Cluster compactness & separation; lower = better |

---

## Output Files

After each run, the following files appear in `./outputs/`:

| File | Description |
|---|---|
| `{dataset}_{method}_reduced.csv` | Reduced coordinates + class labels |
| `{dataset}_{method}_2d.png` | 2-D scatter plot |
| `{dataset}_{method}_3d.png` | 3-D scatter plot (if n_components ≥ 3) |
| `{dataset}_{method}_variance.png` | Explained variance chart (PCA/KernelPCA) |
| `{dataset}_correlation_heatmap.png` | Feature correlation heatmap (with `--heatmap`) |
| `{dataset}_comparison_grid.png` | Side-by-side grid of all methods |
| `{dataset}_comparison_summary.csv` | Evaluation metrics table for all methods |

---

## Examples

### Example 1 — PCA on Iris with heatmap

```bash
python main.py --dataset iris --method pca --heatmap
```

Output:
```
Metric                       Value
──────────────────────────────────────────
trustworthiness              1.0000
continuity                   1.0000
reconstruction_error         0.0419
silhouette                   0.4014
davies_bouldin               0.9555

Explained variance: [72.96  22.85] %
Cumulative:         95.81 %
```

### Example 2 — All methods on Wine

```bash
python main.py --dataset wine --compare_all
```

Generates `wine_comparison_grid.png` and prints:

```
             trustworthiness  continuity  silhouette  davies_bouldin
method
pca                      1.0         1.0    0.526154        0.639202
tsne                     1.0         1.0    0.563446        0.562250
umap                     1.0         1.0    0.587318        0.547391
lda                      1.0         1.0    0.663170        0.448286
kernel_pca               1.0         1.0    0.610434        0.536626
isomap                   1.0         1.0    0.557016        0.568302
```

### Example 3 — Custom CSV

```bash
python main.py --csv data/customers.csv --label_col segment --method umap --n_components 3
```

---

## Theoretical Background

### PCA (Principal Component Analysis)
PCA finds orthogonal directions (principal components) that maximise variance. It solves the eigenvalue problem of the covariance matrix. Best for linear structure; fast and deterministic.

### t-SNE (t-Distributed Stochastic Neighbour Embedding)
t-SNE models pairwise similarities as probabilities in high and low dimensions, minimising their KL-divergence. Excellent for revealing cluster structure but non-convex and slow on large datasets.

### UMAP (Uniform Manifold Approximation and Projection)
UMAP constructs a topological representation of the data manifold and optimises a low-dimensional layout. Faster than t-SNE, preserves both local and global structure.

### LDA (Linear Discriminant Analysis)
A supervised method that finds the linear combination of features maximising between-class scatter relative to within-class scatter. Requires labels; n_components ≤ n_classes − 1.

### Kernel PCA
Applies PCA in a kernel-induced feature space, allowing non-linear projections. Common kernels: RBF, polynomial, sigmoid.

### Isomap
Constructs a neighbourhood graph and computes geodesic distances via shortest paths, then applies MDS. Good at unfolding curved manifolds.

---

## Extending the Project

### Add a new method

1. Open `src/reducer.py`
2. Add a branch in `get_reducer()`:
   ```python
   elif method == "my_method":
       return MyReducer(n_components=n_components, **kwargs)
   ```
3. Append `"my_method"` to `SUPPORTED_METHODS`.

### Add a new dataset

Open `src/data_loader.py`, add a loader function and register it in `DATASETS`:

```python
def load_my_data():
    X, y = ...
    return X, y, feature_names, class_names

DATASETS["my_data"] = load_my_data
```


