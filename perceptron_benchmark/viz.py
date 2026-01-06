"""Visualization utilities for perceptron decision boundaries and training curves."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary_2d(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    title: str = "Decision Boundary",
    save_path: Path | str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot 2D decision boundary for perceptron.

    Args:
        X: Feature matrix (n_samples, 2)
        y: Labels
        weights: Weight vector (at least 2 elements)
        bias: Bias term
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02),
    )

    # Compute decision boundary
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights[:2]) + bias
    Z = np.where(Z >= 0, 1, 0)
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5)

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolor="k", s=50)

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label="Class")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_decision_plane_3d(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    title: str = "Decision Plane",
    save_path: Path | str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot 3D decision plane for perceptron.

    Args:
        X: Feature matrix (n_samples, 3)
        y: Labels
        weights: Weight vector (3 elements)
        bias: Bias term
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot data points
    colors = ["blue" if label == 0 else "red" for label in y]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, edgecolor="k", s=50)

    # Create decision plane mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 20),
        np.linspace(y_min, y_max, 20),
    )

    # Compute z values for the plane: w1*x + w2*y + w3*z + b = 0
    # => z = (-w1*x - w2*y - b) / w3
    if abs(weights[2]) > 1e-10:
        zz = (-weights[0] * xx - weights[1] * yy - bias) / weights[2]

        # Clip z to reasonable range
        z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
        zz = np.clip(zz, z_min, z_max)

        ax.plot_surface(xx, yy, zz, alpha=0.3, color="yellow", edgecolor="gray")

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Class 0"
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Class 1"
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_training_curve(
    history: dict | list,
    metric: Literal["errors", "accuracy"] = "errors",
    title: str = "Training Curve",
    save_path: Path | str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot training curve.

    Args:
        history: Training history dict with 'epoch', 'errors', 'accuracy' keys
                 or list of error values
        metric: Which metric to plot ('errors' or 'accuracy')
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(history, list):
        epochs = list(range(len(history)))
        values = history
        ylabel = "Errors"
    else:
        epochs = history.get("epoch", list(range(len(history.get(metric, [])))))
        values = history.get(metric, [])
        ylabel = metric.capitalize()

    ax.plot(epochs, values, marker="o", markersize=2, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Epoch / Generation")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if metric == "accuracy":
        ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_data_2d(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Dataset",
    save_path: Path | str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot 2D dataset.

    Args:
        X: Feature matrix (n_samples, 2)
        y: Labels
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolor="k", s=50)

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label="Class")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_data_3d(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Dataset",
    save_path: Path | str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot 3D dataset.

    Args:
        X: Feature matrix (n_samples, 3)
        y: Labels
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    colors = ["blue" if label == 0 else "red" for label in y]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, edgecolor="k", s=50)

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Class 0"
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Class 1"
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
