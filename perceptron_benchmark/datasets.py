"""Dataset loading, generation, and transformation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DatasetName = Literal["linear", "nonlinear", "xor"]


@dataclass
class DatasetMeta:
    """Metadata about a loaded dataset."""

    name: str
    n_samples: int
    n_features: int
    n_classes: int
    transformed: bool
    transform_type: str | None


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 'Unnamed: 0' or similar index columns from DataFrame."""
    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed")]
    return df.drop(columns=cols_to_drop, errors="ignore")


def _ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    """Ensure labels are in {0, 1}."""
    unique = np.unique(y)
    if set(unique) == {-1, 1}:
        return np.where(y == -1, 0, 1)
    if set(unique) == {0, 1}:
        return y.astype(int)
    if len(unique) == 2:
        mapping = {unique[0]: 0, unique[1]: 1}
        return np.array([mapping[v] for v in y], dtype=int)
    raise ValueError(f"Expected binary labels, got unique values: {unique}")


def _apply_nonlinear_transform(X: np.ndarray) -> np.ndarray:
    """Apply nonlinear transformation: add x1^2 + x2^2 as third feature."""
    if X.shape[1] < 2:
        raise ValueError("Nonlinear transform requires at least 2 features")
    third_feature = X[:, 0] ** 2 + X[:, 1] ** 2
    return np.column_stack((X, third_feature))


def _apply_xor_transform(X: np.ndarray) -> np.ndarray:
    """Apply XOR transformation: add x1 * x2 as third feature."""
    if X.shape[1] < 2:
        raise ValueError("XOR transform requires at least 2 features")
    third_feature = X[:, 0] * X[:, 1]
    return np.column_stack((X, third_feature))


def load_dataset(
    name: DatasetName,
    data_dir: Path | str = Path("Datasets"),
) -> tuple[np.ndarray, np.ndarray, DatasetMeta]:
    """Load a dataset by name from CSV files.

    Args:
        name: Dataset name ('linear', 'nonlinear', or 'xor')
        data_dir: Directory containing the CSV files

    Returns:
        Tuple of (X, y, metadata) where X is features, y is labels

    Raises:
        FileNotFoundError: If required CSV files are missing
        ValueError: If dataset name is unknown
    """
    data_dir = Path(data_dir)

    file_mapping = {
        "linear": ("X.csv", "y.csv"),
        "nonlinear": ("Xnonlinear.csv", "ynonlinear.csv"),
        "xor": ("XXOR.csv", "YXOR.csv"),
    }

    if name not in file_mapping:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(file_mapping.keys())}")

    x_file, y_file = file_mapping[name]
    x_path = data_dir / x_file
    y_path = data_dir / y_file

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found: {x_path}, {y_path}. "
            "Run 'python scripts/make_datasets.py' to generate them."
        )

    X_df = _drop_unnamed_columns(pd.read_csv(x_path))
    y_df = _drop_unnamed_columns(pd.read_csv(y_path))

    X = X_df.values.astype(float)

    if "label" in y_df.columns:
        y = y_df["label"].values.flatten()
    elif "Output" in y_df.columns:
        y = y_df["Output"].values.flatten()
    elif "0" in y_df.columns:
        y = y_df["0"].values.flatten()
    else:
        y = y_df.iloc[:, 0].values.flatten()

    y = _ensure_binary_labels(y)

    transformed = False
    transform_type = None

    if name == "nonlinear":
        X = _apply_nonlinear_transform(X)
        transformed = True
        transform_type = "x1^2 + x2^2"
    elif name == "xor":
        X = _apply_xor_transform(X)
        transformed = True
        transform_type = "x1 * x2"

    meta = DatasetMeta(
        name=name,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_classes=2,
        transformed=transformed,
        transform_type=transform_type,
    )

    return X, y, meta


def generate_dataset(
    name: DatasetName,
    n_samples: int = 100,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, DatasetMeta]:
    """Generate a synthetic dataset.

    Args:
        name: Dataset type to generate
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y, metadata)
    """
    if seed is not None:
        np.random.seed(seed)

    if name == "linear":
        X, y = _generate_linear(n_samples)
        transformed = False
        transform_type = None
    elif name == "nonlinear":
        X, y = _generate_nonlinear(n_samples)
        X = _apply_nonlinear_transform(X)
        transformed = True
        transform_type = "x1^2 + x2^2"
    elif name == "xor":
        X, y = _generate_xor(n_samples)
        X = _apply_xor_transform(X)
        transformed = True
        transform_type = "x1 * x2"
    else:
        raise ValueError(f"Unknown dataset type: {name}")

    meta = DatasetMeta(
        name=name,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_classes=2,
        transformed=transformed,
        transform_type=transform_type,
    )

    return X, y, meta


def _generate_linear(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate linearly separable 2D data."""
    n_per_class = n_samples // 2

    X0 = np.random.randn(n_per_class, 2) + np.array([-2, -2])
    X1 = np.random.randn(n_per_class, 2) + np.array([2, 2])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx].astype(int)


def _generate_nonlinear(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate non-linearly separable 2D data (concentric circles)."""
    n_per_class = n_samples // 2

    theta_inner = np.random.uniform(0, 2 * np.pi, n_per_class)
    r_inner = np.random.uniform(0, 1.5, n_per_class)
    X0 = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])

    theta_outer = np.random.uniform(0, 2 * np.pi, n_per_class)
    r_outer = np.random.uniform(2.5, 4, n_per_class)
    X1 = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx].astype(int)


def _generate_xor(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate XOR-like 2D data."""
    n_per_quadrant = n_samples // 4

    X_q1 = np.random.uniform(0.5, 2, (n_per_quadrant, 2))
    X_q2 = np.random.uniform(-2, -0.5, (n_per_quadrant, 2))
    X_q2[:, 1] = np.random.uniform(0.5, 2, n_per_quadrant)
    X_q3 = np.random.uniform(-2, -0.5, (n_per_quadrant, 2))
    X_q4 = np.random.uniform(0.5, 2, (n_per_quadrant, 2))
    X_q4[:, 1] = np.random.uniform(-2, -0.5, n_per_quadrant)

    X = np.vstack([X_q1, X_q2, X_q3, X_q4])
    y = np.hstack([
        np.ones(n_per_quadrant),   # Q1: class 1
        np.ones(n_per_quadrant),   # Q2: class 1
        np.zeros(n_per_quadrant),  # Q3: class 0
        np.zeros(n_per_quadrant),  # Q4: class 0
    ])

    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx].astype(int)


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    name: DatasetName,
    data_dir: Path | str = Path("Datasets"),
) -> None:
    """Save a dataset to CSV files.

    Args:
        X: Feature matrix (before any transforms)
        y: Label vector
        name: Dataset name for file naming
        data_dir: Directory to save files
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    file_mapping = {
        "linear": ("X.csv", "y.csv"),
        "nonlinear": ("Xnonlinear.csv", "ynonlinear.csv"),
        "xor": ("XXOR.csv", "YXOR.csv"),
    }

    x_file, y_file = file_mapping[name]

    X_df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=["label"])

    X_df.to_csv(data_dir / x_file, index=False)
    y_df.to_csv(data_dir / y_file, index=False)
