"""CLI entry point and experiment orchestrator."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from perceptron_benchmark.datasets import generate_dataset, load_dataset
from perceptron_benchmark.io import (
    append_to_results_csv,
    create_run_directory,
    generate_run_id,
    save_config,
    save_history,
    save_metrics,
    save_model,
)
from perceptron_benchmark.metrics import compute_metrics
from perceptron_benchmark.models import Perceptron
from perceptron_benchmark.optimizers import (
    BatchOptimizer,
    GeneticAlgorithmOptimizer,
    GreyWolfOptimizer,
    SequentialOptimizer,
    get_optimizer,
)
from perceptron_benchmark.viz import (
    plot_decision_boundary_2d,
    plot_decision_plane_3d,
    plot_training_curve,
)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def build_optimizer_kwargs(args: argparse.Namespace, optimizer_name: str) -> dict[str, Any]:
    """Build optimizer-specific kwargs from CLI args."""
    kwargs: dict[str, Any] = {"seed": args.seed}

    if optimizer_name in ("sequential", "batch"):
        kwargs["epochs"] = args.epochs
        kwargs["learning_rate"] = args.lr
    elif optimizer_name == "ga":
        kwargs["generations"] = args.generations
        kwargs["population_size"] = args.pop
        kwargs["mutation_rate"] = args.mutation_rate
        kwargs["crossover_rate"] = args.crossover_rate
        kwargs["mutation_step"] = args.mutation_step
    elif optimizer_name == "gwo":
        kwargs["epochs"] = args.epochs
        kwargs["n_wolves"] = args.wolves
        kwargs["lb"] = args.lb
        kwargs["ub"] = args.ub

    return kwargs


def run_single_experiment(
    dataset_name: str,
    optimizer_name: str,
    optimizer_kwargs: dict[str, Any],
    data_dir: Path,
    reports_dir: Path,
    save: bool = True,
    show: bool = False,
    tag: str | None = None,
    generate_data: bool = False,
    n_samples: int = 100,
) -> dict[str, Any]:
    """Run a single benchmark experiment.

    Args:
        dataset_name: Name of dataset ('linear', 'nonlinear', 'xor')
        optimizer_name: Name of optimizer
        optimizer_kwargs: Optimizer parameters
        data_dir: Directory with dataset files
        reports_dir: Directory to save results
        save: Whether to save results to disk
        show: Whether to display plots
        tag: Optional run tag
        generate_data: Generate synthetic data if files not found
        n_samples: Number of samples for generated data

    Returns:
        Dictionary with run results
    """
    seed = optimizer_kwargs.get("seed")
    if seed is not None:
        set_seeds(seed)

    # Load or generate dataset
    try:
        X, y, meta = load_dataset(dataset_name, data_dir)
    except FileNotFoundError:
        if generate_data:
            print(f"Dataset files not found. Generating synthetic {dataset_name} dataset...")
            X, y, meta = generate_dataset(dataset_name, n_samples=n_samples, seed=seed)
        else:
            raise

    # Get optimizer and train
    optimizer = get_optimizer(optimizer_name)
    start_time = time.time()
    weights, bias, history = optimizer.fit(X, y, **optimizer_kwargs)
    runtime = time.time() - start_time

    # Create model and compute metrics
    model = Perceptron()
    model.set_params(weights, bias)
    predictions = model.predict(X)
    metrics = compute_metrics(y, predictions)

    # Build config dictionary
    config = {
        "dataset": dataset_name,
        "optimizer": optimizer_name,
        "seed": seed,
        "tag": tag,
        **{k: v for k, v in optimizer_kwargs.items() if k != "seed"},
    }

    # Build result row
    history_dict = history.to_dict()
    best_accuracy = max(history_dict["accuracy"]) if history_dict["accuracy"] else metrics.accuracy

    result = {
        "run_id": generate_run_id(config, seed),
        "dataset": dataset_name,
        "optimizer": optimizer_name,
        "seed": seed,
        "n_samples": meta.n_samples,
        "n_features": meta.n_features,
        "final_accuracy": metrics.accuracy,
        "final_error_count": metrics.error_count,
        "best_accuracy": best_accuracy,
        "runtime_seconds": round(runtime, 4),
        "tag": tag or "",
    }

    if save:
        run_dir = create_run_directory(reports_dir, result["run_id"])

        # Save all artifacts
        save_config(run_dir, config)
        save_metrics(run_dir, metrics.to_dict())
        save_history(run_dir, history_dict)
        save_model(run_dir, weights, bias)

        # Generate and save plots
        figures_dir = run_dir / "figures"

        # Training curve
        plot_training_curve(
            history_dict,
            metric="errors",
            title=f"{optimizer_name.upper()} Training - {dataset_name} (Errors)",
            save_path=figures_dir / "training_curve_errors.png",
            show=show,
        )

        plot_training_curve(
            history_dict,
            metric="accuracy",
            title=f"{optimizer_name.upper()} Training - {dataset_name} (Accuracy)",
            save_path=figures_dir / "training_curve_accuracy.png",
            show=show,
        )

        # Decision boundary/plane
        if meta.n_features == 2:
            plot_decision_boundary_2d(
                X, y, weights, bias,
                title=f"{optimizer_name.upper()} - {dataset_name}",
                save_path=figures_dir / "decision_boundary.png",
                show=show,
            )
        elif meta.n_features == 3:
            plot_decision_plane_3d(
                X, y, weights, bias,
                title=f"{optimizer_name.upper()} - {dataset_name}",
                save_path=figures_dir / "decision_plane.png",
                show=show,
            )

        # Append to results CSV
        append_to_results_csv(reports_dir, result)

        print(f"Results saved to: {run_dir}")

    return result


def run_repeated_experiments(
    dataset_name: str,
    optimizer_name: str,
    base_optimizer_kwargs: dict[str, Any],
    data_dir: Path,
    reports_dir: Path,
    n_repeats: int = 5,
    base_seed: int = 0,
    save: bool = True,
    show: bool = False,
    tag: str | None = None,
    generate_data: bool = False,
) -> list[dict[str, Any]]:
    """Run multiple experiments with different seeds.

    Args:
        dataset_name: Dataset name
        optimizer_name: Optimizer name
        base_optimizer_kwargs: Base optimizer parameters
        data_dir: Dataset directory
        reports_dir: Reports directory
        n_repeats: Number of repetitions
        base_seed: Starting seed value
        save: Save results
        show: Show plots
        tag: Run tag
        generate_data: Generate data if missing

    Returns:
        List of result dictionaries
    """
    results = []

    for i in range(n_repeats):
        seed = base_seed + i
        kwargs = {**base_optimizer_kwargs, "seed": seed}

        print(f"\n[{i + 1}/{n_repeats}] Running {optimizer_name} on {dataset_name} (seed={seed})...")

        result = run_single_experiment(
            dataset_name=dataset_name,
            optimizer_name=optimizer_name,
            optimizer_kwargs=kwargs,
            data_dir=data_dir,
            reports_dir=reports_dir,
            save=save,
            show=show,
            tag=tag,
            generate_data=generate_data,
        )
        results.append(result)

        print(f"  Accuracy: {result['final_accuracy']:.4f}, Errors: {result['final_error_count']}")

    # Print summary
    accuracies = [r["final_accuracy"] for r in results]
    print(f"\nSummary ({n_repeats} runs):")
    print(f"  Mean accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    print(f"  Best accuracy: {max(accuracies):.4f}")

    return results


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="perceptron_benchmark",
        description="Benchmark perceptron training optimizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m perceptron_benchmark.run --dataset linear --optimizer sequential --epochs 100 --lr 0.1 --seed 42 --save
  python -m perceptron_benchmark.run --dataset xor --optimizer ga --generations 200 --pop 300 --seed 42 --save
  python -m perceptron_benchmark.run --dataset nonlinear --optimizer gwo --epochs 100 --wolves 30 --seed 0 --repeat 5
        """,
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["linear", "nonlinear", "xor"],
        required=True,
        help="Dataset to use",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Datasets"),
        help="Directory containing dataset CSV files (default: Datasets/)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic dataset if files not found",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for generated datasets (default: 100)",
    )

    # Optimizer arguments
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sequential", "batch", "ga", "gwo"],
        required=True,
        help="Optimizer to use",
    )

    # Common training arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs for sequential/batch/gwo (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for sequential/batch (default: 0.1)",
    )

    # GA-specific arguments
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Generations for GA (default: 100)",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=100,
        help="Population size for GA (default: 100)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.01,
        help="Mutation rate for GA (default: 0.01)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="Crossover rate for GA (default: 0.7)",
    )
    parser.add_argument(
        "--mutation-step",
        type=float,
        default=0.1,
        help="Mutation step size for GA (default: 0.1)",
    )

    # GWO-specific arguments
    parser.add_argument(
        "--wolves",
        type=int,
        default=30,
        help="Number of wolves for GWO (default: 30)",
    )
    parser.add_argument(
        "--lb",
        type=float,
        default=-5.0,
        help="Lower bound for GWO search space (default: -5.0)",
    )
    parser.add_argument(
        "--ub",
        type=float,
        default=5.0,
        help="Upper bound for GWO search space (default: 5.0)",
    )

    # Output arguments
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to disk (default: True)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for output reports (default: reports/)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag for this run",
    )

    # Repeat experiments
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat experiment with different seeds (default: 1)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle save flag
    save = args.save and not args.no_save

    # Build optimizer kwargs
    optimizer_kwargs = build_optimizer_kwargs(args, args.optimizer)

    print(f"Perceptron Benchmark")
    print(f"=" * 40)
    print(f"Dataset: {args.dataset}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Seed: {args.seed}")

    try:
        if args.repeat > 1:
            results = run_repeated_experiments(
                dataset_name=args.dataset,
                optimizer_name=args.optimizer,
                base_optimizer_kwargs=optimizer_kwargs,
                data_dir=args.data_dir,
                reports_dir=args.reports_dir,
                n_repeats=args.repeat,
                base_seed=args.seed,
                save=save,
                show=args.show,
                tag=args.tag,
                generate_data=args.generate,
            )
        else:
            result = run_single_experiment(
                dataset_name=args.dataset,
                optimizer_name=args.optimizer,
                optimizer_kwargs=optimizer_kwargs,
                data_dir=args.data_dir,
                reports_dir=args.reports_dir,
                save=save,
                show=args.show,
                tag=args.tag,
                generate_data=args.generate,
            )
            print(f"\nResults:")
            print(f"  Accuracy: {result['final_accuracy']:.4f}")
            print(f"  Errors: {result['final_error_count']}")
            print(f"  Runtime: {result['runtime_seconds']:.4f}s")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Use --generate flag to create synthetic datasets.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
