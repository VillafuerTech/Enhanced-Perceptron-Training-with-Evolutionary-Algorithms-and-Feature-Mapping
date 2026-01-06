"""Perceptron Optimizers Benchmark Tool.

A benchmark tool for comparing perceptron training algorithms including
sequential, batch, genetic algorithm (GA), and grey wolf optimizer (GWO).
"""

__version__ = "1.0.0"
__author__ = "Perceptron Benchmark Contributors"

from perceptron_benchmark.models import Perceptron
from perceptron_benchmark.datasets import load_dataset, generate_dataset
from perceptron_benchmark.optimizers import (
    SequentialOptimizer,
    BatchOptimizer,
    GeneticAlgorithmOptimizer,
    GreyWolfOptimizer,
)

__all__ = [
    "Perceptron",
    "load_dataset",
    "generate_dataset",
    "SequentialOptimizer",
    "BatchOptimizer",
    "GeneticAlgorithmOptimizer",
    "GreyWolfOptimizer",
]
