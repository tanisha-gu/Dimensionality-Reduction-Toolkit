"""
Dimensionality Reduction Pipeline
Supports: PCA, t-SNE, UMAP, LDA, Kernel PCA, Isomap, Autoencoder
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

SUPPORTED_METHODS = ["pca", "tsne", "umap", "lda", "kernel_pca", "isomap"]


def get_reducer(method: str, n_components: int = 2, **kwargs):
    """Return the appropriate reducer object for a given method."""
    method = method.lower()
    if method == "pca":
        return PCA(n_components=n_components, **kwargs)
    elif method == "tsne":
        perplexity = kwargs.pop("perplexity", 30)
        random_state = kwargs.pop("random_state", 42)
        return TSNE(n_components=n_components, perplexity=perplexity,
                    random_state=random_state, **kwargs)
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is not installed.")
        n_neighbors = kwargs.pop("n_neighbors", 15)
        min_dist = kwargs.pop("min_dist", 0.1)
        random_state = kwargs.pop("random_state", 42)
        return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                         min_dist=min_dist, random_state=random_state, **kwargs)
    elif method == "lda":
        return LDA(n_components=n_components, **kwargs)
    elif method == "kernel_pca":
        kernel = kwargs.pop("kernel", "rbf")
        return KernelPCA(n_components=n_components, kernel=kernel, **kwargs)
    elif method == "isomap":
        n_neighbors = kwargs.pop("n_neighbors", 5)
        return Isomap(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: {SUPPORTED_METHODS}")


def reduce(X: np.ndarray, method: str = "pca", n_components: int = 2,
           y=None, scale: bool = True, **kwargs) -> np.ndarray:
    """
    Fit and transform data using the chosen dimensionality reduction method.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
    method : one of SUPPORTED_METHODS
    n_components : target number of dimensions
    y : labels (required for LDA)
    scale : whether to StandardScale X before reducing
    **kwargs : extra arguments forwarded to the reducer

    Returns
    -------
    X_reduced : np.ndarray (n_samples, n_components)
    reducer  : fitted reducer object
    scaler   : fitted scaler (or None)
    """
    X = np.array(X, dtype=float)
    scaler = None

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    reducer = get_reducer(method, n_components, **kwargs)

    if method == "lda":
        if y is None:
            raise ValueError("LDA requires labels `y`.")
        # LDA max components = n_classes - 1
        n_classes = len(np.unique(y))
        max_comp = n_classes - 1
        if n_components > max_comp:
            print(f"[LDA] n_components capped to {max_comp} (n_classes - 1).")
            reducer = LDA(n_components=max_comp)
        X_reduced = reducer.fit_transform(X, y)
    else:
        X_reduced = reducer.fit_transform(X)

    return X_reduced, reducer, scaler


def explained_variance_ratio(reducer) -> np.ndarray | None:
    """Return explained variance ratio if available (PCA / KernelPCA)."""
    if hasattr(reducer, "explained_variance_ratio_"):
        return reducer.explained_variance_ratio_
    return None
