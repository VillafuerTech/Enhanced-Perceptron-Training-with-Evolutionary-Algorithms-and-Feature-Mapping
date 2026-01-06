# Perceptron Optimizers Benchmark

A benchmark tool for comparing perceptron training algorithms including classical gradient-based methods and metaheuristic optimizers. This tool evaluates **Sequential Perceptron**, **Batch Perceptron**, **Genetic Algorithm (GA)**, and **Grey Wolf Optimizer (GWO)** on linearly separable, non-linear (with feature mapping), and XOR datasets.

## Features

- **Four Optimizers**: Sequential, Batch, Genetic Algorithm, Grey Wolf Optimizer
- **Three Dataset Types**: Linear, Non-linear (radial), XOR (product features)
- **Feature Mapping**: Automatic transformation for non-linearly separable problems
- **Reproducibility**: Deterministic seeding across numpy and random
- **CLI Interface**: Config-driven experiments with full parameter control
- **Result Persistence**: JSON configs, metrics, CSV curves, and PNG figures per run
- **Aggregated Results**: Append-only `results.csv` for cross-run comparison

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/perceptron-optimizers-benchmark.git
cd perceptron-optimizers-benchmark

# Install dependencies
pip install -r requirements.txt

# Or install as editable package (recommended)
pip install -e .

# For development (includes pytest, ruff, black)
pip install -e ".[dev]"
```

## Quickstart

### 1. Generate Datasets

If you don't have CSV datasets, generate synthetic ones:

```bash
python scripts/make_datasets.py --n-samples 100 --seed 42
```

This creates `Datasets/X.csv`, `Datasets/y.csv`, etc.

### 2. Run a Benchmark

**Sequential perceptron on linear data:**
```bash
python -m perceptron_benchmark.run \
    --dataset linear \
    --optimizer sequential \
    --epochs 100 \
    --lr 0.1 \
    --seed 42 \
    --save
```

**Genetic Algorithm on XOR data:**
```bash
python -m perceptron_benchmark.run \
    --dataset xor \
    --optimizer ga \
    --generations 200 \
    --pop 300 \
    --seed 42 \
    --save
```

**Grey Wolf Optimizer on non-linear data:**
```bash
python -m perceptron_benchmark.run \
    --dataset nonlinear \
    --optimizer gwo \
    --epochs 100 \
    --wolves 50 \
    --seed 42 \
    --save
```

### 3. Run Repeated Experiments

Run 5 experiments with different seeds for statistical analysis:

```bash
python -m perceptron_benchmark.run \
    --dataset linear \
    --optimizer batch \
    --epochs 100 \
    --seed 0 \
    --repeat 5 \
    --save
```

### 4. Use Generated Data Mode

If dataset files are missing, use `--generate`:

```bash
python -m perceptron_benchmark.run \
    --dataset linear \
    --optimizer sequential \
    --epochs 50 \
    --generate \
    --n-samples 100 \
    --seed 42
```

## CLI Reference

```
usage: perceptron_benchmark [-h] --dataset {linear,nonlinear,xor}
                            --optimizer {sequential,batch,ga,gwo}
                            [--seed INT] [--epochs INT] [--lr FLOAT]
                            [--generations INT] [--pop INT]
                            [--mutation-rate FLOAT] [--crossover-rate FLOAT]
                            [--mutation-step FLOAT] [--wolves INT]
                            [--lb FLOAT] [--ub FLOAT]
                            [--save] [--no-save] [--show]
                            [--reports-dir PATH] [--data-dir PATH]
                            [--tag STRING] [--repeat N]
                            [--generate] [--n-samples INT]
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset: `linear`, `nonlinear`, `xor` |
| `--optimizer` | Optimizer: `sequential`, `batch`, `ga`, `gwo` |
| `--seed` | Random seed (default: 42) |
| `--epochs` | Training epochs for sequential/batch/gwo |
| `--lr` | Learning rate for sequential/batch |
| `--generations` | Generations for GA |
| `--pop` | Population size for GA |
| `--wolves` | Pack size for GWO |
| `--repeat` | Repeat experiment N times with different seeds |
| `--save` | Save results to disk (default) |
| `--show` | Display plots interactively |
| `--generate` | Generate synthetic data if files missing |

## Output Structure

```
reports/
├── results.csv              # Aggregated results (one row per run)
└── runs/
    └── 20240115_143022_a1b2c3d4/
        ├── config.json      # Run configuration
        ├── metrics.json     # Final metrics (accuracy, errors, etc.)
        ├── curves.csv       # Training history (epoch, errors, accuracy)
        ├── model.json       # Trained weights and bias
        └── figures/
            ├── training_curve_errors.png
            ├── training_curve_accuracy.png
            └── decision_boundary.png  # or decision_plane.png for 3D
```

### results.csv Columns

| Column | Description |
|--------|-------------|
| `run_id` | Unique run identifier |
| `dataset` | Dataset name |
| `optimizer` | Optimizer used |
| `seed` | Random seed |
| `n_samples` | Number of samples |
| `n_features` | Number of features (after transform) |
| `final_accuracy` | Final classification accuracy |
| `final_error_count` | Final misclassification count |
| `best_accuracy` | Best accuracy during training |
| `runtime_seconds` | Training time |
| `tag` | Optional user tag |

## Datasets

### Linear
Two linearly separable clusters. No feature transformation needed.

### Non-linear
Concentric circles (inner class 0, outer class 1). Transformed with:
```
x3 = x1² + x2²
```

### XOR
Four quadrants with opposite corners sharing classes. Transformed with:
```
x3 = x1 × x2
```

## Optimizers

### Sequential Perceptron
Online learning: updates weights after each misclassified sample.

Parameters: `--epochs`, `--lr`

### Batch Perceptron
Accumulates weight updates over entire dataset before applying.

Parameters: `--epochs`, `--lr`

### Genetic Algorithm (GA)
Evolutionary optimization using selection, crossover, and mutation.

Parameters: `--generations`, `--pop`, `--mutation-rate`, `--crossover-rate`, `--mutation-step`

### Grey Wolf Optimizer (GWO)
Swarm intelligence inspired by grey wolf hunting behavior (alpha, beta, delta hierarchy).

Parameters: `--epochs`, `--wolves`, `--lb`, `--ub`

## Reproducibility

All runs are deterministic when using the same seed:

```python
# Seeds are set for both numpy and random
np.random.seed(seed)
random.seed(seed)
```

Full run metadata is logged in `config.json` including all parameters.

## Adding a New Optimizer

1. Create a class inheriting from `BaseOptimizer` in `perceptron_benchmark/optimizers.py`:

```python
class MyOptimizer(BaseOptimizer):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        my_param: int = 10,
        seed: int | None = None,
    ) -> tuple[np.ndarray, float, TrainingHistory]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize weights
        weights = np.random.rand(X.shape[1])
        bias = np.random.rand()
        history = TrainingHistory()

        # Training loop
        for epoch in range(my_param):
            # ... your optimization logic ...
            predictions = self._compute_predictions(X, weights, bias)
            errors = self._compute_errors(y, predictions)
            accuracy = self._compute_accuracy(y, predictions)
            history.append(epoch, errors, accuracy)

        return weights, float(bias), history
```

2. Register in `get_optimizer()`:
```python
optimizers = {
    "sequential": SequentialOptimizer,
    "batch": BatchOptimizer,
    "ga": GeneticAlgorithmOptimizer,
    "gwo": GreyWolfOptimizer,
    "myopt": MyOptimizer,  # Add here
}
```

3. Add CLI arguments in `run.py` if needed.

## Adding a New Dataset

1. Add generation function in `perceptron_benchmark/datasets.py`:

```python
def _generate_my_dataset(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    # Generate X and y
    return X, y
```

2. Update `generate_dataset()` to handle the new name.

3. Add file mapping in `load_dataset()` if using CSV files.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=perceptron_benchmark

# Run specific test file
pytest tests/test_optimizers_smoke.py -v
```

## Code Quality

```bash
# Lint with ruff
ruff check .

# Format with black
black .

# Check formatting
black --check .
```

## Example Figures

After running benchmarks, find generated figures in `reports/runs/<run_id>/figures/`:

- **Decision Boundary (2D)**: Shows classification regions for linear datasets
- **Decision Plane (3D)**: Shows hyperplane for feature-mapped datasets
- **Training Curves**: Error count and accuracy over epochs/generations

<!--
Example figure placeholders (replace with actual figures after running):
![Decision Boundary](docs/figures/decision_boundary_example.png)
![Training Curve](docs/figures/training_curve_example.png)
-->

## Project Structure

```
.
├── perceptron_benchmark/
│   ├── __init__.py          # Package exports
│   ├── datasets.py          # Data loading and generation
│   ├── models.py            # Perceptron model class
│   ├── optimizers.py        # All optimizer implementations
│   ├── metrics.py           # Classification metrics
│   ├── viz.py               # Visualization utilities
│   ├── io.py                # File I/O utilities
│   └── run.py               # CLI entry point
├── scripts/
│   └── make_datasets.py     # Dataset generator script
├── tests/
│   ├── test_datasets.py
│   ├── test_models.py
│   └── test_optimizers_smoke.py
├── Datasets/                # CSV data files
├── reports/                 # Generated outputs (gitignored)
├── pyproject.toml           # Project configuration
├── requirements.txt         # Dependencies
├── LICENSE                  # MIT License
└── README.md
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this benchmark tool in your research, please cite:

```bibtex
@software{perceptron_optimizers_benchmark,
  title = {Perceptron Optimizers Benchmark},
  year = {2024},
  url = {https://github.com/yourusername/perceptron-optimizers-benchmark}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `ruff check .` and `black --check .` pass
5. Submit a pull request
