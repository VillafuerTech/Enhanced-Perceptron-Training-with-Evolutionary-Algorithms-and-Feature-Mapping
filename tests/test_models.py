"""Tests for Perceptron model."""

import numpy as np
import pytest

from perceptron_benchmark.models import Perceptron


class TestPerceptron:
    """Tests for Perceptron class."""

    def test_set_params(self) -> None:
        """set_params stores weights and bias correctly."""
        model = Perceptron()
        weights = np.array([1.0, 2.0, 3.0])
        bias = 0.5

        model.set_params(weights, bias)

        np.testing.assert_array_equal(model.weights, weights)
        assert model.bias == bias
        assert model.n_features == 3

    def test_decision_function(self) -> None:
        """decision_function computes w·x + b correctly."""
        model = Perceptron()
        model.set_params(np.array([1.0, 2.0]), 0.5)

        X = np.array([[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]])
        scores = model.decision_function(X)

        expected = np.array([3.5, 0.5, -2.5])
        np.testing.assert_array_almost_equal(scores, expected)

    def test_decision_function_without_fit_raises(self) -> None:
        """decision_function raises if model not fitted."""
        model = Perceptron()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.decision_function(np.array([[1.0, 2.0]]))

    def test_predict_binary(self) -> None:
        """predict returns binary labels based on threshold 0."""
        model = Perceptron()
        model.set_params(np.array([1.0, 0.0]), 0.0)

        X = np.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
        predictions = model.predict(X)

        expected = np.array([1, 1, 0])  # scores: 1.0, 0.0, -1.0
        np.testing.assert_array_equal(predictions, expected)

    def test_score_accuracy(self) -> None:
        """score computes accuracy correctly."""
        model = Perceptron()
        model.set_params(np.array([1.0, 0.0]), 0.0)

        X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0], [-0.5, 0.0]])
        y = np.array([1, 0, 1, 0])

        accuracy = model.score(X, y)
        assert accuracy == 1.0

    def test_error_count(self) -> None:
        """error_count returns number of misclassifications."""
        model = Perceptron()
        model.set_params(np.array([1.0, 0.0]), 0.0)

        X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0], [-0.5, 0.0]])
        y = np.array([0, 0, 1, 1])  # 2 errors: first and last

        errors = model.error_count(X, y)
        assert errors == 2

    def test_to_dict_and_from_dict(self) -> None:
        """Model can be serialized and deserialized."""
        model = Perceptron()
        model.set_params(np.array([1.5, -2.5, 3.0]), 0.75)

        data = model.to_dict()
        restored = Perceptron.from_dict(data)

        np.testing.assert_array_almost_equal(restored.weights, model.weights)
        assert restored.bias == model.bias
        assert restored.n_features == model.n_features

    def test_predict_single_sample(self) -> None:
        """predict handles single sample correctly."""
        model = Perceptron()
        model.set_params(np.array([1.0, 1.0]), 0.0)

        X = np.array([1.0, 1.0])
        prediction = model.predict(X)

        assert prediction[0] == 1
