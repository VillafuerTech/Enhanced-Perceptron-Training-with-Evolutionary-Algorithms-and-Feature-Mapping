"""Input/output utilities for saving experiment results."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def generate_run_id(config: dict, seed: int | None = None) -> str:
    """Generate a unique run ID based on config and timestamp.

    Args:
        config: Run configuration dictionary
        seed: Random seed used

    Returns:
        Unique run ID string (timestamp + hash prefix)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_str = json.dumps(config, sort_keys=True) + str(seed)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    return f"{timestamp}_{config_hash}"


def create_run_directory(
    reports_dir: Path | str,
    run_id: str,
) -> Path:
    """Create directory structure for a run.

    Args:
        reports_dir: Base reports directory
        run_id: Unique run identifier

    Returns:
        Path to the run directory
    """
    reports_dir = Path(reports_dir)
    run_dir = reports_dir / "runs" / run_id
    figures_dir = run_dir / "figures"

    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def save_config(run_dir: Path, config: dict) -> Path:
    """Save run configuration to JSON.

    Args:
        run_dir: Run directory path
        config: Configuration dictionary

    Returns:
        Path to saved config file
    """
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return config_path


def save_metrics(run_dir: Path, metrics: dict) -> Path:
    """Save metrics to JSON.

    Args:
        run_dir: Run directory path
        metrics: Metrics dictionary

    Returns:
        Path to saved metrics file
    """
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_serializer)
    return metrics_path


def save_history(run_dir: Path, history: dict) -> Path:
    """Save training history to CSV.

    Args:
        run_dir: Run directory path
        history: Training history dictionary

    Returns:
        Path to saved curves file
    """
    curves_path = run_dir / "curves.csv"
    df = pd.DataFrame(history)
    df.to_csv(curves_path, index=False)
    return curves_path


def save_model(run_dir: Path, weights: Any, bias: float) -> Path:
    """Save model parameters to JSON.

    Args:
        run_dir: Run directory path
        weights: Weight vector
        bias: Bias term

    Returns:
        Path to saved model file
    """
    import numpy as np

    model_path = run_dir / "model.json"
    model_data = {
        "weights": weights.tolist() if isinstance(weights, np.ndarray) else list(weights),
        "bias": float(bias),
    }
    with open(model_path, "w") as f:
        json.dump(model_data, f, indent=2)
    return model_path


def append_to_results_csv(
    reports_dir: Path | str,
    result_row: dict,
) -> Path:
    """Append a result row to the main results CSV.

    Args:
        reports_dir: Base reports directory
        result_row: Dictionary with result data

    Returns:
        Path to results CSV
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_path = reports_dir / "results.csv"

    # Check if file exists to determine if we need headers
    file_exists = results_path.exists()

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

    return results_path


def load_results_csv(reports_dir: Path | str) -> pd.DataFrame:
    """Load results CSV as DataFrame.

    Args:
        reports_dir: Base reports directory

    Returns:
        DataFrame with all results
    """
    results_path = Path(reports_dir) / "results.csv"
    if not results_path.exists():
        return pd.DataFrame()
    return pd.read_csv(results_path)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for objects not serializable by default."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
