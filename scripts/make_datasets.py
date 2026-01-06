#!/usr/bin/env python3
"""Generate synthetic datasets for perceptron benchmarking.

This script creates CSV files for linear, nonlinear, and XOR datasets
that can be used by the benchmark tool.

Usage:
    python scripts/make_datasets.py
    python scripts/make_datasets.py --output-dir custom/path --n-samples 200 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_linear_dataset(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate linearly separable 2D dataset.

    Creates two clusters of points that can be separated by a line.

    Args:
        n_samples: Total number of samples
        seed: Random seed

    Returns:
        Tuple of (X, y) where X is (n_samples, 2) and y is (n_samples,)
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2

    # Class 0: centered at (-2, -2)
    X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-2, -2])

    # Class 1: centered at (2, 2)
    X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([2, 2])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx].astype(int)


def generate_nonlinear_dataset(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate non-linearly separable 2D dataset (concentric circles).

    Creates inner and outer circles that require a radial feature for separation.

    Args:
        n_samples: Total number of samples
        seed: Random seed

    Returns:
        Tuple of (X, y) where X is (n_samples, 2) and y is (n_samples,)
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2

    # Inner circle (class 0)
    theta_inner = np.random.uniform(0, 2 * np.pi, n_per_class)
    r_inner = np.random.uniform(0, 1.5, n_per_class)
    X0 = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])

    # Outer ring (class 1)
    theta_outer = np.random.uniform(0, 2 * np.pi, n_per_class)
    r_outer = np.random.uniform(2.5, 4, n_per_class)
    X1 = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx].astype(int)


def generate_xor_dataset(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate XOR-like 2D dataset.

    Creates four quadrants where opposite quadrants share the same class.

    Args:
        n_samples: Total number of samples
        seed: Random seed

    Returns:
        Tuple of (X, y) where X is (n_samples, 2) and y is (n_samples,)
    """
    np.random.seed(seed)
    n_per_quadrant = n_samples // 4

    # Q1 (++): class 1
    X_q1 = np.random.uniform(0.5, 2, (n_per_quadrant, 2))

    # Q2 (-+): class 0
    X_q2 = np.column_stack([
        np.random.uniform(-2, -0.5, n_per_quadrant),
        np.random.uniform(0.5, 2, n_per_quadrant),
    ])

    # Q3 (--): class 1
    X_q3 = np.random.uniform(-2, -0.5, (n_per_quadrant, 2))

    # Q4 (+-): class 0
    X_q4 = np.column_stack([
        np.random.uniform(0.5, 2, n_per_quadrant),
        np.random.uniform(-2, -0.5, n_per_quadrant),
    ])

    X = np.vstack([X_q1, X_q2, X_q3, X_q4])
    y = np.hstack([
        np.ones(n_per_quadrant),   # Q1
        np.zeros(n_per_quadrant),  # Q2
        np.ones(n_per_quadrant),   # Q3
        np.zeros(n_per_quadrant),  # Q4
    ])

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx].astype(int)


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    x_filename: str,
    y_filename: str,
    output_dir: Path,
) -> None:
    """Save dataset to CSV files.

    Args:
        X: Feature matrix
        y: Labels
        x_filename: Filename for features CSV
        y_filename: Filename for labels CSV
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X_df = pd.DataFrame(X, columns=["x1", "x2"])
    y_df = pd.DataFrame(y, columns=["label"])

    X_df.to_csv(output_dir / x_filename, index=False)
    y_df.to_csv(output_dir / y_filename, index=False)

    print(f"  Saved: {output_dir / x_filename} ({len(X)} samples)")
    print(f"  Saved: {output_dir / y_filename}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for perceptron benchmarking",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Datasets"),
        help="Output directory for CSV files (default: Datasets/)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per dataset (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print(f"Generating datasets with {args.n_samples} samples each (seed={args.seed})")
    print(f"Output directory: {args.output_dir}")
    print()

    # Generate and save linear dataset
    print("Linear dataset:")
    X, y = generate_linear_dataset(args.n_samples, args.seed)
    save_dataset(X, y, "X.csv", "y.csv", args.output_dir)
    print()

    # Generate and save nonlinear dataset
    print("Nonlinear dataset:")
    X, y = generate_nonlinear_dataset(args.n_samples, args.seed + 1)
    save_dataset(X, y, "Xnonlinear.csv", "ynonlinear.csv", args.output_dir)
    print()

    # Generate and save XOR dataset
    print("XOR dataset:")
    X, y = generate_xor_dataset(args.n_samples, args.seed + 2)
    save_dataset(X, y, "XXOR.csv", "YXOR.csv", args.output_dir)
    print()

    print("Done! All datasets generated successfully.")


if __name__ == "__main__":
    main()
