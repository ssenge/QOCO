from __future__ import annotations


def num_trainable_params(module) -> int:
    """Return number of trainable (requires_grad) parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

