"""
patch_mix.py — Patch-Mix augmentation for spectrograms.

Randomly replaces 20–40% of patches in a spectrogram with patches from
a different-class sample. The label becomes a soft convex combination.

This forces the AST to build per-patch discriminative features rather
than relying on global statistics of the whole spectrogram.

Usage (in training loop, applied to a full batch after collation):
    mixer = PatchMixBatch(patch_size=16, min_mix=0.2, max_mix=0.4, prob=0.5)
    fine_mixed, coarse_mixed, labels_mixed = mixer(fine_specs, coarse_specs, labels)
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn.functional as F


class PatchMixBatch:
    """
    Patch-Mix applied to an entire batch.

    Pairs each sample with a random sample from a different class.
    Replaces a random subset of patches from sample A with patches from sample B.
    Returns soft mixed labels.

    Args:
        patch_size:  Patch size in pixels (should match AST patch_size=16).
        min_mix:     Minimum fraction of patches to replace (default 0.2).
        max_mix:     Maximum fraction of patches to replace (default 0.4).
        prob:        Probability of applying mix to each sample (default 0.5).
        num_classes: Number of output classes (default 4).
    """

    def __init__(
        self,
        patch_size: int = 16,
        min_mix: float = 0.20,
        max_mix: float = 0.40,
        prob: float = 0.50,
        num_classes: int = 4,
    ):
        self.patch_size = patch_size
        self.min_mix = min_mix
        self.max_mix = max_mix
        self.prob = prob
        self.num_classes = num_classes

    def __call__(
        self,
        fine_specs: torch.Tensor,    # (B, 1, H, W)
        coarse_specs: torch.Tensor,  # (B, 1, H, W)
        labels: torch.Tensor,        # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fine_mixed:   (B, 1, H, W)
            coarse_mixed: (B, 1, H, W)
            soft_labels:  (B, num_classes) float — one-hot if no mix, soft if mixed
        """
        B = fine_specs.shape[0]
        # Start with hard one-hot labels
        soft_labels = F.one_hot(labels, num_classes=self.num_classes).float()

        fine_out   = fine_specs.clone()
        coarse_out = coarse_specs.clone()

        for i in range(B):
            if random.random() > self.prob:
                continue

            # Find a sample from a different class
            current_label = labels[i].item()
            candidates = [j for j in range(B) if labels[j].item() != current_label]
            if not candidates:
                continue
            j = random.choice(candidates)

            # Determine mix ratio
            lam = random.uniform(self.min_mix, self.max_mix)

            # Apply patch mixing to fine spec
            fine_out[i]   = self._mix_patches(fine_specs[i], fine_specs[j], lam)
            coarse_out[i] = self._mix_patches(coarse_specs[i], coarse_specs[j], lam)

            # Soft label: (1-lam) * label_A + lam * label_B
            label_j_onehot = F.one_hot(labels[j], num_classes=self.num_classes).float()
            soft_labels[i] = (1.0 - lam) * soft_labels[i] + lam * label_j_onehot

        return fine_out, coarse_out, soft_labels

    def _mix_patches(
        self,
        spec_a: torch.Tensor,   # (1, H, W)
        spec_b: torch.Tensor,   # (1, H, W)
        mix_ratio: float,
    ) -> torch.Tensor:
        """Replace mix_ratio fraction of patches in spec_a with patches from spec_b."""
        _, H, W = spec_a.shape
        P = self.patch_size
        n_h = H // P
        n_w = W // P
        n_patches = n_h * n_w
        n_replace = max(1, int(n_patches * mix_ratio))

        patch_indices = random.sample(range(n_patches), n_replace)
        spec_out = spec_a.clone()

        for idx in patch_indices:
            ph = (idx // n_w) * P
            pw = (idx % n_w) * P
            spec_out[:, ph:ph + P, pw:pw + P] = spec_b[:, ph:ph + P, pw:pw + P]

        return spec_out