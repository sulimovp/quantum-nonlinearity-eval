"""Real dataset loaders for QNLEP evaluation.

All loaders return (X_train, X_test, y_train, y_test) as numpy arrays.
Classification: features scaled to ``[-pi, pi]`` for angle encoding.
Regression: features ``X`` stay in ``[-pi, pi]`` where needed for encoding;
targets ``y`` are scaled to ``[-1, 1]`` (train-set min–max) so they match a
typical single-qubit Pauli-Z readout range used in VQCs.

Datasets:
  - Iris binary: real 4-feature classification (classes 0 vs 1 and 1 vs 2)
  - Moons / Circles: sklearn non-linear synthetic classification benchmarks
  - Mackey-Glass: chaotic time-series regression (standard QRC benchmark)
  - Digits binary: 8x8 pixel images downsampled for quantum encoding
"""

import numpy as np
from sklearn.datasets import (
    load_iris,
    make_moons,
    make_circles,
    load_digits,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_to_pi(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Scale each feature to [-pi, pi] using training-set statistics."""
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test


def _pca_reduce(X_train: np.ndarray, X_test: np.ndarray, n_components: int) -> tuple:
    """PCA reduction fit on training data, applied to both splits."""
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test  = pca.transform(X_test)
    return X_train, X_test


# ---------------------------------------------------------------------------
# Classification datasets
# ---------------------------------------------------------------------------

def load_iris_binary(
    classes: tuple = (0, 1),
    test_size: float = 0.3,
    n_features: int = 4,
    random_state: int = 42,
) -> tuple:
    """Iris dataset, binary subset.

    Args:
        classes: pair of class indices to keep (default 0 vs 1 — linearly
                 separable; use (1, 2) for the harder, non-linear case).
        test_size: fraction for test split.
        n_features: number of features after optional PCA reduction.
                    Iris has 4 native features; set n_features < 4 to apply PCA.
        random_state: seed for train/test split.

    Returns:
        X_train, X_test — shape (n, n_features), scaled to [-pi, pi]
        y_train, y_test — binary labels {0, 1}
    """
    iris = load_iris()
    mask = np.isin(iris.target, classes)
    X = iris.data[mask]
    y = (iris.target[mask] == classes[1]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if n_features < X_train.shape[1]:
        X_train, X_test = _pca_reduce(X_train, X_test, n_features)
    X_train, X_test = _scale_to_pi(X_train, X_test)
    return X_train, X_test, y_train.astype(float), y_test.astype(float)


def load_moons(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple:
    """Sklearn make_moons — non-linearly separable binary classification.

    2 features, naturally suited for 2-qubit circuits.
    Tests whether a method can learn a non-linear decision boundary.

    Returns:
        X_train, X_test — shape (n, 2), scaled to [-pi, pi]
        y_train, y_test — binary labels {0.0, 1.0}
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = _scale_to_pi(X_train, X_test)
    return X_train, X_test, y_train.astype(float), y_test.astype(float)


def load_circles(
    n_samples: int = 200,
    noise: float = 0.05,
    factor: float = 0.5,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple:
    """Sklearn make_circles — concentric circles binary classification.

    Strictly non-linearly separable: requires non-linear decision boundary.
    2 features, naturally suited for 2-qubit circuits.

    Returns:
        X_train, X_test — shape (n, 2), scaled to [-pi, pi]
        y_train, y_test — binary labels {0.0, 1.0}
    """
    X, y = make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = _scale_to_pi(X_train, X_test)
    return X_train, X_test, y_train.astype(float), y_test.astype(float)


def load_digits_binary(
    digits: tuple = (0, 1),
    n_components: int = 4,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple:
    """Sklearn digits dataset, binary subset with PCA reduction.

    8×8 pixel images (64 features) reduced via PCA to n_components features
    for quantum angle encoding.

    Args:
        digits: pair of digit classes to keep.
        n_components: PCA output dimensionality (= number of qubits needed).

    Returns:
        X_train, X_test — shape (n, n_components), scaled to [-pi, pi]
        y_train, y_test — binary labels {0.0, 1.0}
    """
    d = load_digits()
    mask = np.isin(d.target, digits)
    X = d.data[mask]
    y = (d.target[mask] == digits[1]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = _pca_reduce(X_train, X_test, n_components)
    X_train, X_test = _scale_to_pi(X_train, X_test)
    return X_train, X_test, y_train.astype(float), y_test.astype(float)


# ---------------------------------------------------------------------------
# Regression / time-series dataset
# ---------------------------------------------------------------------------

def load_mackey_glass(
    n_points: int = 2000,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_power: int = 10,
    transient: int = 500,
    n_train: int = 200,
    n_test: int = 80,
    random_state: int = 42,
) -> tuple:
    """Mackey-Glass chaotic time-series regression.

    Standard benchmark for reservoir computing. The target is one-step-ahead
    prediction: given x(t), predict x(t+1).

    Dynamics (discretised): x(t+1) = x(t) + dt*(beta*x(t-tau)/(1+x(t-tau)^n) - gamma*x(t))

    Args:
        tau: delay parameter (17 = standard chaotic regime).
        n_train, n_test: number of consecutive time steps for each split.

    Returns:
        X_train, X_test — shape (n, 1), x(t) scaled to [-pi, pi]
        y_train, y_test — shape (n,), x(t+1) scaled to [-1, 1] using train y only
    """
    rng = np.random.default_rng(random_state)
    dt = 0.1
    x = np.zeros(n_points + tau + transient)
    x[:tau] = 0.9 + 0.1 * rng.random(tau)   # random initialisation near 0.9

    for t in range(tau, len(x) - 1):
        x[t + 1] = (
            x[t]
            + dt * (beta * x[t - tau] / (1 + x[t - tau] ** n_power) - gamma * x[t])
        )

    x = x[transient:]   # discard transient
    x = x[:n_points]

    # Scale to [-pi, pi]
    x_min, x_max = x.min(), x.max()
    x_scaled = (x - x_min) / (x_max - x_min) * (2 * np.pi) - np.pi

    start = 100
    X_train = x_scaled[start : start + n_train].reshape(-1, 1)
    y_train = x_scaled[start + 1 : start + n_train + 1]

    X_test = x_scaled[start + n_train : start + n_train + n_test].reshape(-1, 1)
    y_test = x_scaled[start + n_train + 1 : start + n_train + n_test + 1]

    # Align regression targets with typical Z readout in [-1, 1] (train-only stats)
    ymin, ymax = float(y_train.min()), float(y_train.max())
    span = max(ymax - ymin, 1e-6)
    y_train = (2.0 * (y_train - ymin) / span - 1.0).astype(np.float64)
    y_test = (2.0 * (y_test - ymin) / span - 1.0).astype(np.float64)

    return X_train, X_test, y_train, y_test


def load_regression_1d(
    n_train: int = 60,
    n_test: int = 40,
    noise: float = 0.05,
    random_state: int = 42,
) -> tuple:
    """1D non-linear regression: f(x) = x*sin(x) + eps.

    More complex than sin(x): the amplitude grows with |x|, requiring
    multi-scale non-linearity to approximate.

    Returns:
        X_train, X_test — shape (n, 1), x in [-pi, pi]
        y_train, y_test — shape (n,), f(x) + noise, scaled to [-1, 1] (train y stats)
    """
    rng = np.random.default_rng(random_state)
    x_all = np.linspace(-np.pi, np.pi, n_train + n_test)
    rng.shuffle(x_all)

    x_tr, x_te = x_all[:n_train], x_all[n_train:]
    y_tr = x_tr * np.sin(x_tr) + noise * rng.standard_normal(n_train)
    y_te = x_te * np.sin(x_te) + noise * rng.standard_normal(n_test)

    ymin, ymax = float(y_tr.min()), float(y_tr.max())
    span = max(ymax - ymin, 1e-6)
    y_tr = (2.0 * (y_tr - ymin) / span - 1.0).astype(np.float64)
    y_te = (2.0 * (y_te - ymin) / span - 1.0).astype(np.float64)

    return x_tr.reshape(-1, 1), x_te.reshape(-1, 1), y_tr, y_te
