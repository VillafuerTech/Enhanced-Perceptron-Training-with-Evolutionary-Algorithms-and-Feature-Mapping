"""Smoke tests for all optimizers."""

import numpy as np
import pytest

from perceptron_benchmark.datasets import generate_dataset
from perceptron_benchmark.optimizers import (
    BatchOptimizer,
    GeneticAlgorithmOptimizer,
    GreyWolfOptimizer,
    SequentialOptimizer,
    TrainingHistory,
    get_optimizer,
)


@pytest.fixture
def linear_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate small linear dataset for testing."""
    X, y, _ = generate_dataset("linear", n_samples=50, seed=42)
    return X, y


@pytest.fixture
def nonlinear_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate small nonlinear dataset for testing."""
    X, y, _ = generate_dataset("nonlinear", n_samples=50, seed=42)
    return X, y


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""

    def test_append_and_to_dict(self) -> None:
        """History records can be appended and converted to dict."""
        history = TrainingHistory()
        history.append(0, 10, 0.8)
        history.append(1, 5, 0.9)

        assert len(history.epoch) == 2
        assert history.errors == [10, 5]
        assert history.accuracy == [0.8, 0.9]

        d = history.to_dict()
        assert d["epoch"] == [0, 1]
        assert d["errors"] == [10, 5]


class TestSequentialOptimizer:
    """Tests for SequentialOptimizer."""

    def test_fit_returns_correct_types(self, linear_data: tuple) -> None:
        """fit returns (weights, bias, history) with correct types."""
        X, y = linear_data
        optimizer = SequentialOptimizer()

        weights, bias, history = optimizer.fit(X, y, epochs=10, learning_rate=0.1, seed=42)

        assert isinstance(weights, np.ndarray)
        assert weights.shape == (X.shape[1],)
        assert isinstance(bias, float)
        assert isinstance(history, TrainingHistory)
        assert len(history.epoch) == 10

    def test_deterministic_with_seed(self, linear_data: tuple) -> None:
        """Same seed produces same results."""
        X, y = linear_data
        optimizer = SequentialOptimizer()

        w1, b1, _ = optimizer.fit(X, y, epochs=10, seed=123)
        w2, b2, _ = optimizer.fit(X, y, epochs=10, seed=123)

        np.testing.assert_array_almost_equal(w1, w2)
        assert b1 == b2


class TestBatchOptimizer:
    """Tests for BatchOptimizer."""

    def test_fit_returns_correct_types(self, linear_data: tuple) -> None:
        """fit returns correct types."""
        X, y = linear_data
        optimizer = BatchOptimizer()

        weights, bias, history = optimizer.fit(X, y, epochs=10, learning_rate=0.1, seed=42)

        assert isinstance(weights, np.ndarray)
        assert weights.shape == (X.shape[1],)
        assert isinstance(bias, float)
        assert len(history.epoch) == 10

    def test_deterministic_with_seed(self, linear_data: tuple) -> None:
        """Same seed produces same results."""
        X, y = linear_data
        optimizer = BatchOptimizer()

        w1, b1, _ = optimizer.fit(X, y, epochs=10, seed=123)
        w2, b2, _ = optimizer.fit(X, y, epochs=10, seed=123)

        np.testing.assert_array_almost_equal(w1, w2)
        assert b1 == b2


class TestGeneticAlgorithmOptimizer:
    """Tests for GeneticAlgorithmOptimizer."""

    def test_fit_returns_correct_types(self, linear_data: tuple) -> None:
        """fit returns correct types."""
        X, y = linear_data
        optimizer = GeneticAlgorithmOptimizer()

        weights, bias, history = optimizer.fit(X, y, generations=5, population_size=20, seed=42)

        assert isinstance(weights, np.ndarray)
        assert weights.shape == (X.shape[1],)
        assert isinstance(bias, float)
        assert len(history.epoch) == 5

    def test_history_tracks_best_fitness(self, linear_data: tuple) -> None:
        """History errors should be non-increasing (best so far)."""
        X, y = linear_data
        optimizer = GeneticAlgorithmOptimizer()

        _, _, history = optimizer.fit(X, y, generations=10, population_size=30, seed=42)

        # Best error should be non-increasing
        for i in range(1, len(history.errors)):
            assert history.errors[i] <= history.errors[i - 1]

    def test_mostly_deterministic_with_seed(self, linear_data: tuple) -> None:
        """Same seed produces same results (GA uses both np and random)."""
        X, y = linear_data
        optimizer = GeneticAlgorithmOptimizer()

        w1, b1, h1 = optimizer.fit(X, y, generations=5, population_size=20, seed=42)
        w2, b2, h2 = optimizer.fit(X, y, generations=5, population_size=20, seed=42)

        np.testing.assert_array_almost_equal(w1, w2)
        assert b1 == b2


class TestGreyWolfOptimizer:
    """Tests for GreyWolfOptimizer."""

    def test_fit_returns_correct_types(self, linear_data: tuple) -> None:
        """fit returns correct types."""
        X, y = linear_data
        optimizer = GreyWolfOptimizer()

        weights, bias, history = optimizer.fit(
            X, y, epochs=5, n_wolves=10, lb=-5.0, ub=5.0, seed=42
        )

        assert isinstance(weights, np.ndarray)
        assert weights.shape == (X.shape[1],)
        assert isinstance(bias, float)
        assert len(history.epoch) == 5

    def test_bounds_scalar_expansion(self, linear_data: tuple) -> None:
        """Scalar bounds are expanded to match dimensions."""
        X, y = linear_data
        optimizer = GreyWolfOptimizer()

        # Should not raise
        weights, bias, _ = optimizer.fit(X, y, epochs=3, n_wolves=5, lb=-10, ub=10, seed=42)

        assert weights is not None

    def test_mostly_deterministic_with_seed(self, linear_data: tuple) -> None:
        """Same seed produces same results."""
        X, y = linear_data
        optimizer = GreyWolfOptimizer()

        w1, b1, _ = optimizer.fit(X, y, epochs=5, n_wolves=10, seed=42)
        w2, b2, _ = optimizer.fit(X, y, epochs=5, n_wolves=10, seed=42)

        np.testing.assert_array_almost_equal(w1, w2)
        assert b1 == b2


class TestGetOptimizer:
    """Tests for get_optimizer factory function."""

    @pytest.mark.parametrize(
        "name,expected_class",
        [
            ("sequential", SequentialOptimizer),
            ("batch", BatchOptimizer),
            ("ga", GeneticAlgorithmOptimizer),
            ("gwo", GreyWolfOptimizer),
        ],
    )
    def test_returns_correct_optimizer(self, name: str, expected_class: type) -> None:
        """get_optimizer returns correct optimizer class."""
        optimizer = get_optimizer(name)
        assert isinstance(optimizer, expected_class)

    def test_unknown_optimizer_raises(self) -> None:
        """Unknown optimizer name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer("unknown")


class TestOptimizersOnNonlinearData:
    """Test all optimizers can handle 3D transformed data."""

    @pytest.mark.parametrize("optimizer_name", ["sequential", "batch", "ga", "gwo"])
    def test_optimizer_on_3d_data(self, nonlinear_data: tuple, optimizer_name: str) -> None:
        """All optimizers work with 3D feature-mapped data."""
        X, y = nonlinear_data
        optimizer = get_optimizer(optimizer_name)

        if optimizer_name == "sequential":
            weights, bias, history = optimizer.fit(X, y, epochs=5, seed=42)
        elif optimizer_name == "batch":
            weights, bias, history = optimizer.fit(X, y, epochs=5, seed=42)
        elif optimizer_name == "ga":
            weights, bias, history = optimizer.fit(X, y, generations=5, population_size=20, seed=42)
        else:  # gwo
            weights, bias, history = optimizer.fit(X, y, epochs=5, n_wolves=10, seed=42)

        assert weights.shape == (3,)  # 3D after transform
        assert isinstance(bias, float)
        assert len(history.epoch) == 5
