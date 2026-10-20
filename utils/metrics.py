"""
Streaming Metrics Computation for Semantic Segmentation.

Provides StreamingConfusionMatrix, which accumulates predictions across batches
or entire multi-tile aerial datasets (e.g., 6000x6000 ISPRS Potsdam tiles) without
storing high-resolution prediction arrays in memory.

Metrics Computed:
- Mean Intersection-over-Union (mIoU)
- Per-class Intersection-over-Union (IoU)
- Overall Accuracy (OA)
- Macro F1-Score
- Per-class F1-Score (Dice coefficient)
- Per-class Precision and Recall
"""

from typing import Any, Sequence

import numpy as np
import torch


class StreamingConfusionMatrix:
    """
    Streaming confusion matrix for evaluating semantic segmentation models.

    Maintains an accumulated (num_classes, num_classes) integer confusion matrix
    where rows correspond to Ground Truth categories and columns correspond to
    Model Predictions:
        matrix[t, p] = count of pixels with true class t and predicted class p.

    Supports online accumulation across batches and sliding-window tiles,
    properly ignores void labels (ignore_index=255), and computes standard
    benchmarks (mIoU, OA, F1).

    Args:
        num_classes: Total number of valid semantic classes.
        class_names: Optional sequence of human-readable class names.
                     If None, defaults to ['Class_0', 'Class_1', ...].
        ignore_index: Integer label value to ignore in metrics computation (default: 255).
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Sequence[str] | None = None,
        ignore_index: int = 255,
    ) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)

        if class_names is not None:
            if len(class_names) != num_classes:
                raise ValueError(
                    f"Length of class_names ({len(class_names)}) does not match num_classes ({num_classes})"
                )
            self.class_names = list(class_names)
        else:
            self.class_names = [f"Class_{i}" for i in range(num_classes)]

        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Clears accumulated confusion matrix counts."""
        self.matrix.fill(0)

    def update(
        self,
        pred: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
    ) -> None:
        """
        Accumulates batch predictions and targets into the confusion matrix.

        Args:
            pred: Predicted class indices or class logits/probabilities.
                  Can be a PyTorch Tensor or NumPy array of shapes:
                  - [B, C, H, W] (class logits/probabilities -> argmax along dim 1)
                  - [C, H, W] (class logits -> argmax along dim 0)
                  - [B, H, W] or [H, W] (class indices)
            target: Ground truth integer class labels.
                    Can be a PyTorch Tensor or NumPy array of shapes:
                    - [B, H, W] or [H, W]
                    - [B, 1, H, W] (channel dim squeezed)
        """
        # Convert prediction tensor/array to integer class index map
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu()
            if pred.dim() == 4:
                # Shape: [B, C, H, W]
                pred = torch.argmax(pred, dim=1)
            elif pred.dim() == 3 and pred.size(0) == self.num_classes:
                # Shape: [C, H, W]
                pred = torch.argmax(pred, dim=0)
            pred_arr = pred.numpy().astype(np.int64)
        elif isinstance(pred, np.ndarray):
            if pred.ndim == 4:
                pred_arr = np.argmax(pred, axis=1).astype(np.int64)
            elif pred.ndim == 3 and pred.shape[0] == self.num_classes:
                pred_arr = np.argmax(pred, axis=0).astype(np.int64)
            else:
                pred_arr = pred.astype(np.int64)
        else:
            raise TypeError(f"Unsupported prediction type: {type(pred)}")

        # Convert target tensor/array to integer class index map
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu()
            if target.dim() == 4 and target.size(1) == 1:
                target = target.squeeze(1)
            target_arr = target.numpy().astype(np.int64)
        elif isinstance(target, np.ndarray):
            if target.ndim == 4 and target.shape[1] == 1:
                target_arr = np.squeeze(target, axis=1).astype(np.int64)
            else:
                target_arr = target.astype(np.int64)
        else:
            raise TypeError(f"Unsupported target type: {type(target)}")

        # Flatten arrays to 1D
        p_flat = pred_arr.reshape(-1)
        t_flat = target_arr.reshape(-1)

        if p_flat.shape[0] != t_flat.shape[0]:
            raise ValueError(
                f"Element count mismatch between prediction ({p_flat.shape[0]}) and target ({t_flat.shape[0]})"
            )

        # Filter valid pixels: exclude ignore_index and any out-of-range indices
        valid_mask = (
            (t_flat != self.ignore_index)
            & (t_flat >= 0)
            & (t_flat < self.num_classes)
            & (p_flat >= 0)
            & (p_flat < self.num_classes)
        )

        p_valid = p_flat[valid_mask]
        t_valid = t_flat[valid_mask]

        if len(t_valid) == 0:
            return

        # Fast 2D histogram via 1D bincount: index = num_classes * true + pred
        bin_indices = self.num_classes * t_valid + p_valid
        batch_counts = np.bincount(
            bin_indices,
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

        self.matrix += batch_counts

    def compute_per_class_iou(self) -> dict[str, float]:
        """
        Computes the Intersection-over-Union (IoU) for each class.

        Formula for class i:
            IoU_i = TP_i / (TP_i + FP_i + FN_i)
                  = matrix[i, i] / (row_sum_i + col_sum_i - matrix[i, i])

        Returns:
            Dictionary mapping class name to IoU float value (or float('nan') if union is 0).
        """
        ious: dict[str, float] = {}
        row_sums = self.matrix.sum(axis=1)  # Ground truth pixel counts per class
        col_sums = self.matrix.sum(axis=0)  # Prediction pixel counts per class

        for i, name in enumerate(self.class_names):
            tp = self.matrix[i, i]
            union = row_sums[i] + col_sums[i] - tp
            if union > 0:
                ious[name] = float(tp / union)
            else:
                ious[name] = float("nan")

        return ious

    def compute_miou(self, exclude_classes: Optional[Sequence[Union[int, str]]] = None) -> float:
        """
        Computes the Mean Intersection-over-Union (mIoU) across valid classes.

        Args:
            exclude_classes: Optional sequence of class names or class indices to exclude
                             from the average (e.g. ['Building']).

        Returns:
            Mean IoU as a float in [0.0, 1.0]. Returns 0.0 if no valid pixels observed.
        """
        per_class_iou = self.compute_per_class_iou()
        excluded = set()
        if exclude_classes:
            for c in exclude_classes:
                if isinstance(c, int) and 0 <= c < len(self.class_names):
                    excluded.add(self.class_names[c])
                else:
                    excluded.add(str(c))

        valid_ious = [
            v for name, v in per_class_iou.items()
            if name not in excluded and not np.isnan(v)
        ]
        if len(valid_ious) == 0:
            return 0.0
        return float(np.mean(valid_ious))

    def compute_overall_accuracy(self, exclude_classes: Optional[Sequence[Union[int, str]]] = None) -> float:
        """
        Computes the Overall Accuracy (OA) across evaluated pixels, optionally excluding specified classes.

        Args:
            exclude_classes: Optional sequence of class names or class indices to exclude.

        Returns:
            Overall Accuracy as a float in [0.0, 1.0].
        """
        excluded_indices = set()
        if exclude_classes:
            for c in exclude_classes:
                if isinstance(c, int):
                    excluded_indices.add(c)
                elif c in self.class_names:
                    excluded_indices.add(self.class_names.index(c))

        keep_mask = np.ones(self.num_classes, dtype=bool)
        for idx in excluded_indices:
            if 0 <= idx < self.num_classes:
                keep_mask[idx] = False

        sub_matrix = self.matrix[np.ix_(keep_mask, keep_mask)]
        total_correct = np.trace(sub_matrix)
        total_pixels = np.sum(sub_matrix)
        if total_pixels == 0:
            return 0.0
        return float(total_correct / total_pixels)

    compute_oa = compute_overall_accuracy

    def compute_per_class_f1(self) -> dict[str, float]:
        """
        Computes the F1-Score (Dice coefficient) for each class.

        Formula for class i:
            F1_i = (2 * TP_i) / (2 * TP_i + FP_i + FN_i)
                 = (2 * matrix[i, i]) / (row_sum_i + col_sum_i)

        Returns:
            Dictionary mapping class name to F1-Score (or float('nan') if denominator is 0).
        """
        f1_scores: dict[str, float] = {}
        row_sums = self.matrix.sum(axis=1)
        col_sums = self.matrix.sum(axis=0)

        for i, name in enumerate(self.class_names):
            tp = self.matrix[i, i]
            denom = row_sums[i] + col_sums[i]
            if denom > 0:
                f1_scores[name] = float(2.0 * tp / denom)
            else:
                f1_scores[name] = float("nan")

        return f1_scores

    def compute_f1(self, exclude_classes: Optional[Sequence[Union[int, str]]] = None) -> float:
        """
        Computes the Macro F1-Score across valid classes, optionally excluding specified classes.
        """
        per_class_f1 = self.compute_per_class_f1()
        excluded = set()
        if exclude_classes:
            for c in exclude_classes:
                if isinstance(c, int) and 0 <= c < len(self.class_names):
                    excluded.add(self.class_names[c])
                else:
                    excluded.add(str(c))

        valid_f1s = [
            v for name, v in per_class_f1.items()
            if name not in excluded and not np.isnan(v)
        ]
        if len(valid_f1s) == 0:
            return 0.0
        return float(np.mean(valid_f1s))

    def compute_per_class_precision(self) -> dict[str, float]:
        """
        Computes Precision for each class.

        Formula for class i:
            Precision_i = TP_i / (TP_i + FP_i) = matrix[i, i] / col_sum_i
        """
        precisions: dict[str, float] = {}
        col_sums = self.matrix.sum(axis=0)

        for i, name in enumerate(self.class_names):
            tp = self.matrix[i, i]
            denom = col_sums[i]
            precisions[name] = float(tp / denom) if denom > 0 else float("nan")

        return precisions

    def compute_per_class_recall(self) -> dict[str, float]:
        """
        Computes Recall for each class.

        Formula for class i:
            Recall_i = TP_i / (TP_i + FN_i) = matrix[i, i] / row_sum_i
        """
        recalls: dict[str, float] = {}
        row_sums = self.matrix.sum(axis=1)

        for i, name in enumerate(self.class_names):
            tp = self.matrix[i, i]
            denom = row_sums[i]
            recalls[name] = float(tp / denom) if denom > 0 else float("nan")

        return recalls

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Returns a copy of the accumulated confusion matrix.

        Returns:
            np.ndarray of shape (num_classes, num_classes) and dtype np.int64.
        """
        return self.matrix.copy()

    def get_summary(self) -> dict[str, Any]:
        """
        Compiles a comprehensive summary dictionary of all evaluation metrics.

        Returns:
            Dict containing 'miou', 'oa', 'f1', 'per_class_iou', 'per_class_f1',
            'per_class_precision', and 'per_class_recall'.
        """
        return {
            "miou": self.compute_miou(),
            "oa": self.compute_overall_accuracy(),
            "f1": self.compute_f1(),
            "per_class_iou": self.compute_per_class_iou(),
            "per_class_f1": self.compute_per_class_f1(),
            "per_class_precision": self.compute_per_class_precision(),
            "per_class_recall": self.compute_per_class_recall(),
        }
