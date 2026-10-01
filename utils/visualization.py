"""Visualization utilities for continual learning experiments."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_accuracy_matrix(acc_matrix: np.ndarray, save_path: str = None):
    """Plot accuracy matrix as a heatmap.

    acc_matrix[i, j] = accuracy on task j after training up to task i.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        acc_matrix, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=[f"Task {i+1}" for i in range(acc_matrix.shape[1])],
        yticklabels=[f"After {i+1}" for i in range(acc_matrix.shape[0])],
        ax=ax,
    )
    ax.set_xlabel("Evaluated on")
    ax.set_ylabel("Trained up to")
    ax.set_title("Task Accuracy Matrix")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_forgetting_curve(results: dict, save_path: str = None):
    """Plot per-task accuracy over training progression."""
    mat = np.array(results["accuracy_matrix"])
    n_tasks = mat.shape[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    for task_id in range(n_tasks):
        accs = mat[task_id:, task_id]
        steps = list(range(task_id, n_tasks))
        ax.plot(steps, accs, marker="o", label=f"Task {task_id + 1}")

    ax.set_xlabel("Tasks Learned")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Task Accuracy Over Training")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_method_comparison(
    method_results: dict[str, dict], metric: str = "avg_accuracy",
    save_path: str = None,
):
    """Bar chart comparing methods on a given metric."""
    names = list(method_results.keys())
    values = [r[metric] for r in method_results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=sns.color_palette("Set2", len(names)))
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Method Comparison: {metric.replace('_', ' ').title()}")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10,
        )

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
