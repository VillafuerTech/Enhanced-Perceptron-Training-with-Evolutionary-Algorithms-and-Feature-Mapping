"""Perceptron model implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Perceptron:
    """Binary perceptron classifier.

    Attributes:
        weights: Weight vector (set after fitting)
        bias: Bias term (set after fitting)
        n_features: Number of features (set after fitting)
    """

    weights: np.ndarray | None = field(default=None, repr=False)
    bias: float = 0.0
    n_features: int = 0

    def set_params(self, weights: np.ndarray, bias: float) -> None:
        """Set model parameters.

        Args:
            weights: Weight vector
            bias: Bias term
        """
        self.weights = np.asarray(weights)
        self.bias = float(bias)
        self.n_features = len(self.weights)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute raw decision scores (w·x + b).

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Decision scores of shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call set_params() first.")
        X = np.atleast_2d(X)
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted labels (0 or 1) of shape (n_samples,)
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Accuracy as a float in [0, 1]
        """
        predictions = self.predict(X)
        return float(np.mean(predictions == y))

    def error_count(self, X: np.ndarray, y: np.ndarray) -> int:
        """Count misclassified samples.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Number of misclassifications
        """
        predictions = self.predict(X)
        return int(np.sum(predictions != y))

    def to_dict(self) -> dict:
        """Export model parameters as dictionary."""
        return {
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": self.bias,
            "n_features": self.n_features,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Perceptron:
        """Create model from dictionary."""
        model = cls()
        if data.get("weights") is not None:
            model.set_params(np.array(data["weights"]), data["bias"])
        return model
