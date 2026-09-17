"""Continual learning evaluation metrics.

Tracks the full T×T accuracy matrix and computes:
  - Average Accuracy (AA)
  - Last Accuracy (LA)
  - Average Forgetting (AF)
  - Backward Transfer (BWT)
  - Forward Transfer (FWT)
"""

import json
import numpy as np
from pathlib import Path


class MetricTracker:
    """Track accuracy matrix and compute CL metrics."""

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        # acc_matrix[i, j] = accuracy on task j after training up to task i
        self.acc_matrix = np.zeros((n_tasks, n_tasks))

    def update(self, train_task: int, eval_task: int, accuracy: float):
        self.acc_matrix[train_task, eval_task] = accuracy

    def compute_metrics(self) -> dict:
        T = self.n_tasks
        mat = self.acc_matrix

        # Last Accuracy: average accuracy across all tasks after final training
        last_accuracy = mat[T - 1, :T].mean()

        # Average Incremental Accuracy
        avg_accuracy = 0.0
        for t in range(T):
            avg_accuracy += mat[t, :t + 1].mean()
        avg_accuracy /= T

        # Average Forgetting
        forgetting = 0.0
        count = 0
        for j in range(T - 1):
            best_before_final = mat[:T - 1, j].max() if T > 1 else mat[0, j]
            forgetting += max(0, best_before_final - mat[T - 1, j])
            count += 1
        avg_forgetting = forgetting / count if count > 0 else 0.0

        # Backward Transfer: how much later tasks hurt earlier ones
        bwt = 0.0
        bwt_count = 0
        for j in range(T - 1):
            bwt += mat[T - 1, j] - mat[j, j]
            bwt_count += 1
        bwt = bwt / bwt_count if bwt_count > 0 else 0.0

        # Forward Transfer: how much earlier tasks help later ones
        # FWT_j = acc on task j before training on it (using model trained on tasks 1..j-1)
        fwt = 0.0
        fwt_count = 0
        for j in range(1, T):
            if mat[j - 1, j] > 0:
                fwt += mat[j - 1, j]
                fwt_count += 1
        fwt = fwt / fwt_count if fwt_count > 0 else 0.0

        return {
            "avg_accuracy": float(avg_accuracy),
            "last_accuracy": float(last_accuracy),
            "avg_forgetting": float(avg_forgetting),
            "backward_transfer": float(bwt),
            "forward_transfer": float(fwt),
            "accuracy_matrix": self.acc_matrix.tolist(),
        }

    def __str__(self) -> str:
        results = self.compute_metrics()
        lines = [
            "=" * 50,
            "Continual Learning Results",
            "=" * 50,
            f"  Average Accuracy:    {results['avg_accuracy']:6.2f}%",
            f"  Last Accuracy:       {results['last_accuracy']:6.2f}%",
            f"  Average Forgetting:  {results['avg_forgetting']:6.2f}%",
            f"  Backward Transfer:   {results['backward_transfer']:+6.2f}%",
            f"  Forward Transfer:    {results['forward_transfer']:6.2f}%",
            "-" * 50,
            "Accuracy Matrix (row=after training, col=task):",
        ]

        T = self.n_tasks
        header = "       " + "".join(f"{'T'+str(j+1):>7}" for j in range(T))
        lines.append(header)
        for i in range(T):
            row = f"  A{i+1:>2}: " + "".join(f"{self.acc_matrix[i, j]:7.1f}" for j in range(T))
            lines.append(row)
        lines.append("=" * 50)

        return "\n".join(lines)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results = self.compute_metrics()
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
