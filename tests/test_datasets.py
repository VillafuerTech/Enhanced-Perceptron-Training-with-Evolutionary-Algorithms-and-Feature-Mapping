"""Tests for dataset loading and generation."""

import numpy as np
import pytest

from perceptron_benchmark.datasets import (
    DatasetMeta,
    _apply_nonlinear_transform,
    _apply_xor_transform,
    _ensure_binary_labels,
    generate_dataset,
)


class TestGenerateDataset:
    """Tests for generate_dataset function."""

    @pytest.mark.parametrize("name", ["linear", "nonlinear", "xor"])
    def test_generate_returns_correct_shapes(self, name: str) -> None:
        """Generated datasets have correct shapes."""
        X, y, meta = generate_dataset(name, n_samples=100, seed=42)

        assert X.shape[0] == y.shape[0]
        assert X.shape[0] >= 96  # Allow for rounding in quadrant-based generation
        assert isinstance(meta, DatasetMeta)

    @pytest.mark.parametrize("name", ["linear", "nonlinear", "xor"])
    def test_generate_binary_labels(self, name: str) -> None:
        """Generated datasets have binary labels {0, 1}."""
        X, y, meta = generate_dataset(name, n_samples=100, seed=42)

        unique_labels = set(np.unique(y))
        assert unique_labels.issubset({0, 1})

    def test_generate_linear_2d(self) -> None:
        """Linear dataset has 2 features (no transform)."""
        X, y, meta = generate_dataset("linear", n_samples=100, seed=42)

        assert X.shape[1] == 2
        assert meta.transformed is False
        assert meta.n_features == 2

    def test_generate_nonlinear_3d(self) -> None:
        """Nonlinear dataset has 3 features (with transform)."""
        X, y, meta = generate_dataset("nonlinear", n_samples=100, seed=42)

        assert X.shape[1] == 3
        assert meta.transformed is True
        assert meta.transform_type == "x1^2 + x2^2"

    def test_generate_xor_3d(self) -> None:
        """XOR dataset has 3 features (with transform)."""
        X, y, meta = generate_dataset("xor", n_samples=100, seed=42)

        assert X.shape[1] == 3
        assert meta.transformed is True
        assert meta.transform_type == "x1 * x2"

    def test_generate_deterministic_with_seed(self) -> None:
        """Same seed produces same dataset."""
        X1, y1, _ = generate_dataset("linear", n_samples=50, seed=123)
        X2, y2, _ = generate_dataset("linear", n_samples=50, seed=123)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_generate_different_with_different_seeds(self) -> None:
        """Different seeds produce different datasets."""
        X1, y1, _ = generate_dataset("linear", n_samples=50, seed=123)
        X2, y2, _ = generate_dataset("linear", n_samples=50, seed=456)

        assert not np.allclose(X1, X2)

    def test_generate_unknown_name_raises(self) -> None:
        """Unknown dataset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            generate_dataset("unknown", n_samples=50)


class TestTransforms:
    """Tests for feature transformations."""

    def test_nonlinear_transform(self) -> None:
        """Nonlinear transform adds x1^2 + x2^2 as third feature."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
        X_transformed = _apply_nonlinear_transform(X)

        assert X_transformed.shape == (3, 3)
        expected_third = np.array([1 + 4, 9 + 16, 0])
        np.testing.assert_array_almost_equal(X_transformed[:, 2], expected_third)

    def test_xor_transform(self) -> None:
        """XOR transform adds x1 * x2 as third feature."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 2.0]])
        X_transformed = _apply_xor_transform(X)

        assert X_transformed.shape == (3, 3)
        expected_third = np.array([2, 12, -2])
        np.testing.assert_array_almost_equal(X_transformed[:, 2], expected_third)


class TestEnsureBinaryLabels:
    """Tests for label normalization."""

    def test_already_binary_01(self) -> None:
        """Labels already in {0, 1} are unchanged."""
        y = np.array([0, 1, 0, 1, 1])
        result = _ensure_binary_labels(y)
        np.testing.assert_array_equal(result, y)

    def test_convert_minus1_1_to_01(self) -> None:
        """Labels in {-1, 1} are converted to {0, 1}."""
        y = np.array([-1, 1, -1, 1, 1])
        result = _ensure_binary_labels(y)
        expected = np.array([0, 1, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_non_binary_raises(self) -> None:
        """Non-binary labels with >2 classes raise ValueError."""
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="Expected binary labels"):
            _ensure_binary_labels(y)
