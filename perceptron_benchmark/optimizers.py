"""Perceptron training optimizers with unified interface."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrainingHistory:
    """Training history container.

    Attributes:
        epoch: List of epoch/generation numbers
        errors: List of error counts per epoch
        accuracy: List of accuracy values per epoch
    """

    epoch: list[int] = field(default_factory=list)
    errors: list[int] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)

    def append(self, epoch: int, errors: int, accuracy: float) -> None:
        """Append a training step record."""
        self.epoch.append(epoch)
        self.errors.append(errors)
        self.accuracy.append(accuracy)

    def to_dict(self) -> dict[str, list]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "errors": self.errors,
            "accuracy": self.accuracy,
        }


class BaseOptimizer(ABC):
    """Abstract base class for perceptron optimizers."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, float, TrainingHistory]:
        """Train perceptron on data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels of shape (n_samples,)
            **kwargs: Optimizer-specific parameters

        Returns:
            Tuple of (weights, bias, history)
        """
        pass

    @staticmethod
    def _compute_predictions(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
        """Compute binary predictions."""
        scores = np.dot(X, weights) + bias
        return np.where(scores >= 0, 1, 0)

    @staticmethod
    def _compute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> int:
        """Count misclassifications."""
        return int(np.sum(y_true != y_pred))

    @staticmethod
    def _compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(y_true == y_pred))


class SequentialOptimizer(BaseOptimizer):
    """Sequential (online) perceptron training.

    Updates weights after each misclassified sample.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.1,
        seed: int | None = None,
    ) -> tuple[np.ndarray, float, TrainingHistory]:
        """Train using sequential updates.

        Args:
            X: Feature matrix
            y: Labels
            epochs: Number of training epochs
            learning_rate: Learning rate for weight updates
            seed: Random seed for weight initialization

        Returns:
            Tuple of (weights, bias, history)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        n_samples, n_features = X.shape
        weights = np.random.rand(n_features)
        bias = np.random.rand()
        history = TrainingHistory()

        for epoch in range(epochs):
            total_error = 0
            for i in range(n_samples):
                y_hat = np.dot(weights, X[i]) + bias
                prediction = 1 if y_hat >= 0 else 0

                error = y[i] - prediction
                if error != 0:
                    weights += learning_rate * error * X[i]
                    bias += learning_rate * error
                    total_error += abs(error)

            predictions = self._compute_predictions(X, weights, bias)
            accuracy = self._compute_accuracy(y, predictions)
            history.append(epoch, int(total_error), accuracy)

        return weights, float(bias), history


class BatchOptimizer(BaseOptimizer):
    """Batch perceptron training.

    Accumulates updates over all samples before applying.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.1,
        seed: int | None = None,
    ) -> tuple[np.ndarray, float, TrainingHistory]:
        """Train using batch updates.

        Args:
            X: Feature matrix
            y: Labels
            epochs: Number of training epochs
            learning_rate: Learning rate for weight updates
            seed: Random seed for weight initialization

        Returns:
            Tuple of (weights, bias, history)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        n_samples, n_features = X.shape
        weights = np.random.rand(n_features)
        bias = np.random.rand()
        history = TrainingHistory()

        for epoch in range(epochs):
            weights_update = np.zeros(n_features)
            bias_update = 0.0
            total_error = 0

            for i in range(n_samples):
                y_hat = np.dot(weights, X[i]) + bias
                prediction = 1 if y_hat >= 0 else 0

                error = y[i] - prediction
                if error != 0:
                    weights_update += learning_rate * error * X[i]
                    bias_update += learning_rate * error
                    total_error += abs(error)

            weights += weights_update
            bias += bias_update

            predictions = self._compute_predictions(X, weights, bias)
            accuracy = self._compute_accuracy(y, predictions)
            history.append(epoch, int(total_error), accuracy)

        return weights, float(bias), history


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Genetic Algorithm optimizer for perceptron training.

    Uses evolutionary selection, crossover, and mutation to find optimal weights.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        generations: int = 100,
        population_size: int = 100,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.7,
        mutation_step: float = 0.1,
        selection_ratio: float = 0.3,
        seed: int | None = None,
    ) -> tuple[np.ndarray, float, TrainingHistory]:
        """Train using genetic algorithm.

        Args:
            X: Feature matrix
            y: Labels
            generations: Number of generations
            population_size: Population size
            mutation_rate: Probability of gene mutation
            crossover_rate: Probability of crossover
            mutation_step: Standard deviation for mutation
            selection_ratio: Fraction of population to select as parents
            seed: Random seed

        Returns:
            Tuple of (weights, bias, history)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        n_features = X.shape[1]
        chromosome_length = n_features + 1  # weights + bias

        # Initialize population
        population = [np.random.uniform(-1, 1, chromosome_length) for _ in range(population_size)]

        best_individual = None
        best_fitness = float("inf")
        history = TrainingHistory()

        for generation in range(generations):
            # Evaluate fitness for each individual
            fitness_scores = []
            for individual in population:
                weights = individual[:-1]
                bias = individual[-1]
                predictions = self._compute_predictions(X, weights, bias)
                errors = self._compute_errors(y, predictions)
                fitness_scores.append(errors)

                if errors < best_fitness:
                    best_fitness = errors
                    best_individual = individual.copy()

            # Record history using best individual
            best_weights = best_individual[:-1]
            best_bias = best_individual[-1]
            best_predictions = self._compute_predictions(X, best_weights, best_bias)
            accuracy = self._compute_accuracy(y, best_predictions)
            history.append(generation, int(best_fitness), accuracy)

            # Selection: select top individuals
            n_selected = max(2, int(population_size * selection_ratio))
            sorted_indices = np.argsort(fitness_scores)
            selected = [population[i].copy() for i in sorted_indices[:n_selected]]

            # Create new population through crossover
            new_population = []
            while len(new_population) < population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)

                if random.random() < crossover_rate:
                    crossover_point = random.randint(1, chromosome_length - 1)
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                    new_population.append(child1)
                    if len(new_population) < population_size:
                        new_population.append(child2)
                else:
                    new_population.append(parent1.copy())

            # Mutation
            for individual in new_population:
                for gene_idx in range(chromosome_length):
                    if random.random() < mutation_rate:
                        individual[gene_idx] += np.random.normal(0, mutation_step)
                        individual[gene_idx] = np.clip(individual[gene_idx], -10, 10)

            population = new_population

        # Extract best weights and bias
        weights = best_individual[:-1]
        bias = float(best_individual[-1])

        return weights, bias, history


class GreyWolfOptimizer(BaseOptimizer):
    """Grey Wolf Optimizer for perceptron training.

    Uses wolf pack hierarchy (alpha, beta, delta) to guide search.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        n_wolves: int = 30,
        lb: float | np.ndarray = -5.0,
        ub: float | np.ndarray = 5.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, float, TrainingHistory]:
        """Train using grey wolf optimizer.

        Args:
            X: Feature matrix
            y: Labels
            epochs: Number of iterations
            n_wolves: Pack size
            lb: Lower bound for search space (scalar or per-dimension)
            ub: Upper bound for search space (scalar or per-dimension)
            seed: Random seed

        Returns:
            Tuple of (weights, bias, history)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        n_features = X.shape[1]
        dim = n_features + 1  # weights + bias

        # Expand bounds to dimension
        if np.isscalar(lb):
            lb = np.full(dim, lb)
        else:
            lb = np.asarray(lb)
        if np.isscalar(ub):
            ub = np.full(dim, ub)
        else:
            ub = np.asarray(ub)

        # Initialize wolf population
        population = [lb + (ub - lb) * np.random.rand(dim) for _ in range(n_wolves)]

        # Sort by fitness
        population = self._sort_population(population, X, y)

        alpha = population[0].copy()
        beta = population[1].copy() if len(population) > 1 else alpha.copy()
        delta = population[2].copy() if len(population) > 2 else beta.copy()

        history = TrainingHistory()

        for epoch in range(epochs):
            # Linearly decrease a from 2 to 0
            a = 2 - epoch * (2 / epochs)

            # Update each wolf position
            new_population = []
            for wolf in population:
                new_wolf = self._update_wolf_position(wolf, alpha, beta, delta, a, lb, ub)
                new_population.append(new_wolf)

            population = self._sort_population(new_population, X, y)

            # Update hierarchy
            alpha = population[0].copy()
            beta = population[1].copy() if len(population) > 1 else alpha.copy()
            delta = population[2].copy() if len(population) > 2 else beta.copy()

            # Record history
            alpha_weights = alpha[:-1]
            alpha_bias = alpha[-1]
            predictions = self._compute_predictions(X, alpha_weights, alpha_bias)
            errors = self._compute_errors(y, predictions)
            accuracy = self._compute_accuracy(y, predictions)
            history.append(epoch, errors, accuracy)

        # Return best solution (alpha)
        weights = alpha[:-1]
        bias = float(alpha[-1])

        return weights, bias, history

    def _fitness(self, wolf: np.ndarray, X: np.ndarray, y: np.ndarray) -> int:
        """Compute fitness (error count) for a wolf."""
        weights = wolf[:-1]
        bias = wolf[-1]
        predictions = self._compute_predictions(X, weights, bias)
        return self._compute_errors(y, predictions)

    def _sort_population(
        self, population: list[np.ndarray], X: np.ndarray, y: np.ndarray
    ) -> list[np.ndarray]:
        """Sort population by fitness (ascending error)."""
        return sorted(population, key=lambda w: self._fitness(w, X, y))

    def _update_wolf_position(
        self,
        wolf: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        delta: np.ndarray,
        a: float,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        """Update wolf position based on alpha, beta, delta."""
        new_wolf = np.zeros_like(wolf)

        for j in range(len(wolf)):
            # Alpha influence
            r1, r2 = random.random(), random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha[j] - wolf[j])
            X1 = alpha[j] - A1 * D_alpha

            # Beta influence
            r1, r2 = random.random(), random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta[j] - wolf[j])
            X2 = beta[j] - A2 * D_beta

            # Delta influence
            r1, r2 = random.random(), random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta[j] - wolf[j])
            X3 = delta[j] - A3 * D_delta

            new_wolf[j] = np.clip((X1 + X2 + X3) / 3, lb[j], ub[j])

        return new_wolf


def get_optimizer(name: str) -> BaseOptimizer:
    """Get optimizer by name.

    Args:
        name: Optimizer name ('sequential', 'batch', 'ga', 'gwo')

    Returns:
        Optimizer instance
    """
    optimizers = {
        "sequential": SequentialOptimizer,
        "batch": BatchOptimizer,
        "ga": GeneticAlgorithmOptimizer,
        "gwo": GreyWolfOptimizer,
    }

    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from: {list(optimizers.keys())}")

    return optimizers[name]()
