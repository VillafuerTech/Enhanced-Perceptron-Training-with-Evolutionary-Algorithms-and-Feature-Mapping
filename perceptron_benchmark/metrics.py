"""Metrics computation for classification evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationMetrics:
    """Container for classification metrics.

    Attributes:
        accuracy: Classification accuracy [0, 1]
        error_count: Number of misclassifications
        error_rate: Fraction of misclassifications [0, 1]
        n_samples: Total number of samples
        true_positives: Number of true positives
        true_negatives: Number of true negatives
        false_positives: Number of false positives
        false_negatives: Number of false negatives
    """

    accuracy: float
    error_count: int
    error_rate: float
    n_samples: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "n_samples": self.n_samples,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }

    @property
    def precision(self) -> float:
        """Compute precision (TP / (TP + FP))."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Compute recall (TP / (TP + FN))."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Compute F1 score (harmonic mean of precision and recall)."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        ClassificationMetrics instance with all computed metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    n_samples = len(y_true)
    correct = np.sum(y_true == y_pred)
    error_count = n_samples - correct

    # Confusion matrix elements
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return ClassificationMetrics(
        accuracy=float(correct / n_samples) if n_samples > 0 else 0.0,
        error_count=int(error_count),
        error_rate=float(error_count / n_samples) if n_samples > 0 else 0.0,
        n_samples=n_samples,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute binary confusion matrix.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        2x2 confusion matrix as numpy array:
        [[TN, FP],
         [FN, TP]]
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return np.array([[tn, fp], [fn, tp]])
