"""Perceptron Optimizers Benchmark Tool.

A benchmark tool for comparing perceptron training algorithms including
sequential, batch, genetic algorithm (GA), and grey wolf optimizer (GWO).
"""

from perceptron_benchmark.datasets import generate_dataset, load_dataset
from perceptron_benchmark.models import Perceptron
from perceptron_benchmark.optimizers import (
    BatchOptimizer,
    GeneticAlgorithmOptimizer,
    GreyWolfOptimizer,
    SequentialOptimizer,
)

__version__ = "1.0.0"
__author__ = "Perceptron Benchmark Contributors"

__all__ = [
    "Perceptron",
    "load_dataset",
    "generate_dataset",
    "SequentialOptimizer",
    "BatchOptimizer",
    "GeneticAlgorithmOptimizer",
    "GreyWolfOptimizer",
]
