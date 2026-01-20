import math
from typing import Dict, Any, Optional
from torch.optim import Optimizer
from torch.optim import lr_scheduler


def build_lr_scheduler(
    optimizer: Optimizer,
    *,
    base_lr: float,
    train_steps: int,
    warmup_steps: int = 0,
    scheduler_cfg: Optional[Dict[str, Any]] = None,
):
    """
    Build a learning-rate scheduler from config.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    base_lr : float
        Peak / initial learning rate.
    train_steps : int
        Total training steps.
    warmup_steps : int
        Linear warmup steps.
    scheduler_cfg : dict
        Example:
        {
          "type": "cosine",
          "min_lr": 5e-06,
          "num_cycles": 0.5
        }
    """
    scheduler_cfg = scheduler_cfg or {"type": "constant"}

    # Backward compatibility: allow string
    if isinstance(scheduler_cfg, str):
        scheduler_cfg = {"type": scheduler_cfg}

    sched_type = scheduler_cfg.get("type", "constant")
    min_lr = float(scheduler_cfg.get("min_lr", 0.0))
    num_cycles = float(scheduler_cfg.get("num_cycles", 0.5))

    if sched_type == "constant" and warmup_steps == 0:
        return None

    def lr_lambda(step: int) -> float:
        # --- Warmup ---
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        # --- After warmup ---
        t = step - warmup_steps
        T = max(1, train_steps - warmup_steps)
        progress = min(1.0, t / T)

        if sched_type == "constant":
            return 1.0

        if sched_type == "linear":
            lr = min_lr + (base_lr - min_lr) * (1.0 - progress)
            return lr / base_lr

        if sched_type == "cosine":
            cosine = 0.5 * (1.0 + math.cos(2.0 * math.pi * num_cycles * progress))
            lr = min_lr + (base_lr - min_lr) * cosine
            return lr / base_lr

        raise ValueError(f"Unknown lr_scheduler type: {sched_type}")

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)
