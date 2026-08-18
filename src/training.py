"""Canonical supervised-training API for external training jobs.

The array-backed trainer is the supported implementation because it batches
the dataset and handles device placement. ``rfb_training`` remains available
for legacy sample-dictionary workflows, but new code should import from this
module.
"""

from src.dataset_generation import (
    train_bubble_on_dataset,
    train_multi_bubble_on_dataset,
)

__all__ = ["train_bubble_on_dataset", "train_multi_bubble_on_dataset"]
