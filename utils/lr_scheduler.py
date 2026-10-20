"""
Polynomial Learning Rate Schedulers for Semantic Segmentation.

Standard deep learning frameworks and SOTA segmentation architectures (e.g., SegFormer,
DeepLabV3+, PSPNet) utilize polynomial learning rate decay with power=0.9, as it provides
a smoother decay schedule compared to step decay, enabling optimal convergence on dense
prediction tasks.
"""

from typing import Any, Optional, Dict, List, Union

import torch
from torch.optim import Optimizer


class PolyLRScheduler:
    """
    Polynomial Learning Rate Scheduler.

    Updates the learning rate according to the polynomial decay formula:
        lr = max(min_lr, base_lr * (1 - epoch / max_epochs) ** power)

    Args:
        optimizer: Wrapped PyTorch optimizer.
        max_epochs: Total number of training epochs.
        power: Exponent of the polynomial function (default: 0.9, standard for semantic segmentation).
        min_lr: Minimum learning rate floor (default: 1e-6).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: Optional[int] = None,
        power: float = 0.9,
        min_lr: float = 1e-6,
        **kwargs: Any
    ) -> None:
        actual_max = max_epochs if max_epochs is not None else kwargs.get("max_iters", 100)
        if actual_max <= 0:
            raise ValueError(f"max_epochs/max_iters must be positive, got {actual_max}")
        if power <= 0:
            raise ValueError(f"power must be positive, got {power}")
        if min_lr < 0:
            raise ValueError(f"min_lr cannot be negative, got {min_lr}")

        self.optimizer = optimizer
        self.max_epochs = int(actual_max)
        self.power = float(power)
        self.min_lr = float(min_lr)

        # Store initial learning rate for each parameter group
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch: int = 0

    def step(self, epoch: int | None = None) -> None:
        """
        Updates the learning rate of each parameter group for the given epoch.

        Args:
            epoch: Optional 0-indexed epoch index. If None, uses internal counter
                   and increments it by 1.
        """
        if epoch is None:
            epoch = self.current_epoch
            self.current_epoch += 1
        else:
            self.current_epoch = epoch + 1

        # Compute polynomial decay factor
        progress = float(epoch) / float(self.max_epochs)
        decay_factor = max(0.0, 1.0 - progress) ** self.power

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            new_lr = max(self.min_lr, base_lr * decay_factor)
            group["lr"] = new_lr

    def get_lr(self) -> list[float]:
        """
        Retrieves the current learning rate for each parameter group.

        Returns:
            List of learning rates for all parameter groups.
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    @property
    def current_lr(self) -> float:
        """Convenience property returning the learning rate of the first parameter group."""
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict[str, Any]:
        """Returns the scheduler state as a dictionary for checkpointing."""
        return {
            "max_epochs": self.max_epochs,
            "power": self.power,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the scheduler state from a checkpoint dictionary."""
        self.max_epochs = state_dict["max_epochs"]
        self.power = state_dict["power"]
        self.min_lr = state_dict["min_lr"]
        self.base_lrs = state_dict["base_lrs"]
        self.current_epoch = state_dict["current_epoch"]


class WarmupPolyLR:
    """
    Polynomial Learning Rate Scheduler with Linear Warmup.

    Applies linear learning rate warmup from warmup_start_lr to base_lr over
    the first `warmup_epochs` epochs, followed by polynomial decay over the
    remaining (max_epochs - warmup_epochs) epochs.

    Warmup Phase (epoch < warmup_epochs):
        lr = warmup_start_lr + ((epoch + 1) / warmup_epochs) * (base_lr - warmup_start_lr)

    Decay Phase (epoch >= warmup_epochs):
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        lr = max(min_lr, base_lr * (1 - progress) ** power)

    Args:
        optimizer: Wrapped PyTorch optimizer.
        max_epochs: Total number of training epochs.
        warmup_epochs: Number of epochs dedicated to linear warmup (default: 5).
        power: Exponent of the polynomial decay (default: 0.9).
        min_lr: Minimum learning rate floor (default: 1e-6).
        warmup_start_lr: Initial learning rate at start of warmup (default: 1e-7).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        warmup_epochs: int = 5,
        power: float = 0.9,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs cannot be negative, got {warmup_epochs}")
        if min_lr < 0:
            raise ValueError(f"min_lr cannot be negative, got {min_lr}")

        self.optimizer = optimizer
        self.max_epochs = int(max_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.power = float(power)
        self.min_lr = float(min_lr)
        self.warmup_start_lr = float(warmup_start_lr)

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch: int = 0

        # If warmup is enabled, initialize learning rates to warmup_start_lr
        if self.warmup_epochs > 0:
            for group in self.optimizer.param_groups:
                group["lr"] = self.warmup_start_lr

    def step(self, epoch: int | None = None) -> None:
        """
        Updates learning rates for the given epoch according to the warmup + poly schedule.

        Args:
            epoch: Optional 0-indexed epoch index. If None, uses internal counter.
        """
        if epoch is None:
            epoch = self.current_epoch
            self.current_epoch += 1
        else:
            self.current_epoch = epoch + 1

        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            # Linear warmup: progress from 0 -> 1
            alpha = float(epoch + 1) / float(self.warmup_epochs)
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
                new_lr = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                group["lr"] = max(self.min_lr, new_lr)
        else:
            # Polynomial decay over remaining epochs
            total_decay_epochs = max(1, self.max_epochs - self.warmup_epochs)
            decay_epoch = max(0, epoch - self.warmup_epochs)
            progress = min(1.0, float(decay_epoch) / float(total_decay_epochs))
            decay_factor = max(0.0, 1.0 - progress) ** self.power

            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
                new_lr = max(self.min_lr, base_lr * decay_factor)
                group["lr"] = new_lr

    def get_lr(self) -> list[float]:
        """
        Retrieves the current learning rate for each parameter group.

        Returns:
            List of learning rates for all parameter groups.
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    @property
    def current_lr(self) -> float:
        """Convenience property returning the learning rate of the first parameter group."""
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict[str, Any]:
        """Returns the scheduler state as a dictionary for checkpointing."""
        return {
            "max_epochs": self.max_epochs,
            "warmup_epochs": self.warmup_epochs,
            "power": self.power,
            "min_lr": self.min_lr,
            "warmup_start_lr": self.warmup_start_lr,
            "base_lrs": self.base_lrs,
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the scheduler state from a checkpoint dictionary."""
        self.max_epochs = state_dict["max_epochs"]
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.power = state_dict["power"]
        self.min_lr = state_dict["min_lr"]
        self.warmup_start_lr = state_dict["warmup_start_lr"]
        self.base_lrs = state_dict["base_lrs"]
        self.current_epoch = state_dict["current_epoch"]
