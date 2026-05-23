"""
Evaluation metrics for dimensionality reduction quality.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score


def trustworthiness(X_high, X_low, n_neighbors=5) -> float:
    """
    Trustworthiness measures how well local neighbourhoods are preserved.
    Score in [0, 1]; 1 = perfect preservation.
    """
    n = len(X_high)
    nbrs_high = NearestNeighbors(n_neighbors=n_neighbors).fit(X_high)
    nbrs_low = NearestNeighbors(n_neighbors=n_neighbors).fit(X_low)

    _, indices_high = nbrs_high.kneighbors(X_high)
    _, indices_low = nbrs_low.kneighbors(X_low)

    # rank matrix in high-dim space
    rank_high = np.zeros((n, n), dtype=int)
    for i in range(n):
        for rank, j in enumerate(indices_high[i], start=1):
            rank_high[i, j] = rank

    t = 0.0
    for i in range(n):
        for j in indices_low[i]:
            if rank_high[i, j] > n_neighbors:
                t += rank_high[i, j] - n_neighbors

    normaliser = n * n_neighbors * (2 * n - 3 * n_neighbors - 1) / 2
    return float(1 - t / normaliser)


def continuity(X_high, X_low, n_neighbors=5) -> float:
    """
    Continuity (complement of trustworthiness direction):
    measures how well low-dim neighbours are true neighbours in high-dim.
    """
    n = len(X_high)
    nbrs_high = NearestNeighbors(n_neighbors=n_neighbors).fit(X_high)
    nbrs_low = NearestNeighbors(n_neighbors=n_neighbors).fit(X_low)

    _, indices_high = nbrs_high.kneighbors(X_high)
    _, indices_low = nbrs_low.kneighbors(X_low)

    rank_low = np.zeros((n, n), dtype=int)
    for i in range(n):
        for rank, j in enumerate(indices_low[i], start=1):
            rank_low[i, j] = rank

    c = 0.0
    for i in range(n):
        for j in indices_high[i]:
            if rank_low[i, j] > n_neighbors:
                c += rank_low[i, j] - n_neighbors

    normaliser = n * n_neighbors * (2 * n - 3 * n_neighbors - 1) / 2
    return float(1 - c / normaliser)


def reconstruction_error(X_original, X_reduced, reducer) -> float | None:
    """
    Reconstruction error (MSE) for methods with an inverse_transform (e.g. PCA).
    Returns None if inverse_transform is not available.
    """
    if not hasattr(reducer, "inverse_transform"):
        return None
    X_reconstructed = reducer.inverse_transform(X_reduced)
    return float(np.mean((X_original - X_reconstructed) ** 2))


def cluster_quality(X_reduced, labels) -> dict:
    """
    Silhouette score and Davies–Bouldin index to measure cluster separation
    in the reduced space.
    """
    if len(set(labels)) < 2:
        return {"silhouette": None, "davies_bouldin": None}
    return {
        "silhouette": float(silhouette_score(X_reduced, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_reduced, labels)),
    }


def evaluate(X_original, X_reduced, labels=None, reducer=None,
             n_neighbors=5) -> dict:
    """
    Full evaluation report for a single reduction result.
    """
    report = {}
    report["trustworthiness"] = trustworthiness(X_original, X_reduced, n_neighbors)
    report["continuity"] = continuity(X_original, X_reduced, n_neighbors)
    report["reconstruction_error"] = reconstruction_error(X_original, X_reduced, reducer)
    if labels is not None:
        report.update(cluster_quality(X_reduced, labels))
    return report
