"""
Dataset loaders used throughout the project.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_digits, load_wine, load_breast_cancer,
    make_swiss_roll, make_s_curve, fetch_openml
)


def load_iris_data():
    data = load_iris()
    return data.data, data.target, data.feature_names, data.target_names


def load_digits_data(n_samples=1000):
    data = load_digits()
    idx = np.random.RandomState(0).choice(len(data.data), size=min(n_samples, len(data.data)), replace=False)
    return data.data[idx], data.target[idx], None, [str(i) for i in range(10)]


def load_wine_data():
    data = load_wine()
    return data.data, data.target, data.feature_names, data.target_names


def load_breast_cancer_data():
    data = load_breast_cancer()
    return data.data, data.target, list(data.feature_names), data.target_names


def load_swiss_roll(n_samples=1500, noise=0.1):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)
    labels = (t / t.max() * 4).astype(int)
    return X, labels, ["x", "y", "z"], None


def load_s_curve(n_samples=1500, noise=0.1):
    X, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=42)
    labels = (t / t.max() * 4).astype(int)
    return X, labels, ["x", "y", "z"], None


def load_from_csv(path: str, label_col: str = None):
    df = pd.read_csv(path)
    if label_col and label_col in df.columns:
        y = df[label_col].values
        X = df.drop(columns=[label_col]).values
        feature_names = [c for c in df.columns if c != label_col]
    else:
        y = None
        X = df.values
        feature_names = list(df.columns)
    return X, y, feature_names, None


DATASETS = {
    "iris": load_iris_data,
    "wine": load_wine_data,
    "digits": load_digits_data,
    "breast_cancer": load_breast_cancer_data,
    "swiss_roll": load_swiss_roll,
    "s_curve": load_s_curve,
}
