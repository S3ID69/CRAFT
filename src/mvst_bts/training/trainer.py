"""
trainer.py — Trainer class encapsulating the full training loop.

Handles:
  - Train / val loop with ASAM two-step update
  - PatchMix applied per-batch
  - RepAugment applied on fused embeddings
  - Early stopping on sensitivity (primary) or ICBHI score
  - Checkpointing: best_sensitivity.pt, best_icbhi.pt, latest.pt
  - TensorBoard logging
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mvst_bts.training.losses import build_loss, SoftFocalLoss
from mvst_bts.training.asam import build_optimizer
from mvst_bts.training.scheduler import build_scheduler
from mvst_bts.augmentation.patch_mix import PatchMixBatch
from mvst_bts.augmentation.rep_augment import RepAugment
from mvst_bts.utils.metrics import compute_icbhi_metrics, format_metrics
from mvst_bts.utils.logging import get_logger, TBLogger


class Trainer:
    """
    Full training loop for MVST-BTS+.

    Args:
        model:       The MVSTBTSPlus model.
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation/test split.
        cfg:          Full OmegaConf config.
        run_dir:      Directory to save checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        run_dir: str | Path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # ── Loss ───────────────────────────────────────
        self.criterion = build_loss(cfg).to(self.device)
        self.use_soft_labels = cfg.augmentation.patch_mix.enabled

        # ── Optimizer / ASAM ───────────────────────────
        self.base_optimizer, self.asam = build_optimizer(model, cfg)

        # ── Scheduler ─────────────────────────────────
        total_steps = cfg.training.epochs * len(train_loader)
        self.scheduler = build_scheduler(self.base_optimizer, cfg, total_steps)

        # ── Augmentations (batch-level) ────────────────
        aug_cfg = cfg.augmentation
        self.patch_mix = (
            PatchMixBatch(
                patch_size=cfg.model.ast.patch_size,
                min_mix=aug_cfg.patch_mix.min_mix_ratio,
                max_mix=aug_cfg.patch_mix.max_mix_ratio,
                prob=aug_cfg.patch_mix.prob,
            ) if aug_cfg.patch_mix.enabled else None
        )
        self.rep_augment = (
            RepAugment(
                mask_rate=aug_cfg.rep_augment.mask_rate,
                gen_alpha=aug_cfg.rep_augment.gen_alpha,
            ) if aug_cfg.rep_augment.enabled else None
        )
        if self.rep_augment:
            self.rep_augment.train()

        # ── Logging ────────────────────────────────────
        self.logger = get_logger("trainer", self.run_dir / "train.log")
        self.tb = TBLogger(self.run_dir / "tb")

        # ── Early stopping state ───────────────────────
        self.patience = cfg.training.early_stopping_patience
        self.monitor  = cfg.training.early_stopping_metric  # "sensitivity" or "icbhi_score"
        self.best_monitored = -1.0
        self.best_icbhi     = -1.0
        self.no_improve     = 0
        self.global_step    = 0

    # ────────────────────────────────────────────────────
    # Public interface
    # ────────────────────────────────────────────────────

    def fit(self) -> dict:
        """Run the full training loop. Returns the best metrics dict."""
        self.logger.info(f"Training on {self.device} for {self.cfg.training.epochs} epochs")
        self.logger.info(f"Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")

        best_metrics = {}
        for epoch in range(1, self.cfg.training.epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(epoch)
            val_metrics = self._val_epoch(epoch)

            elapsed = time.time() - t0
            self.logger.info(
                f"Epoch {epoch:3d}/{self.cfg.training.epochs} | "
                f"loss={train_loss:.4f} | "
                f"ICBHI={val_metrics['icbhi_score']:.2f}% | "
                f"sens={val_metrics['sensitivity']:.2f}% | "
                f"spec={val_metrics['specificity']:.2f}% | "
                f"time={elapsed:.1f}s"
            )
            self.logger.info(format_metrics(val_metrics))

            # TensorBoard
            self.tb.log_metrics({"loss": train_loss}, step=epoch, prefix="train")
            self.tb.log_metrics(val_metrics, step=epoch, prefix="val")

            # Checkpoint
            self._save_checkpoint("latest.pt")
            monitored = val_metrics[self.monitor]
            if monitored > self.best_monitored:
                self.best_monitored = monitored
                self._save_checkpoint(f"best_{self.monitor}.pt")
                best_metrics = dict(val_metrics)
                self.no_improve = 0
                self.logger.info(f"  ✓ New best {self.monitor}: {monitored:.2f}%")
            else:
                self.no_improve += 1

            if val_metrics["icbhi_score"] > self.best_icbhi:
                self.best_icbhi = val_metrics["icbhi_score"]
                self._save_checkpoint("best_icbhi.pt")

            # Early stopping
            if self.no_improve >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        self.tb.close()
        return best_metrics

    # ────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        if self.rep_augment:
            self.rep_augment.train()

        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f"Train {epoch}", leave=False):
            fine   = batch["fine_spec"].to(self.device)
            coarse = batch["coarse_spec"].to(self.device)
            meta   = {k: v.to(self.device) for k, v in batch["metadata"].items()}
            labels = batch["label"].to(self.device)

            # ── PatchMix (batch-level) ─────────────────
            soft_labels = None
            if self.patch_mix is not None:
                fine, coarse, soft_labels = self.patch_mix(fine, coarse, labels)
                soft_labels = soft_labels.to(self.device)

            # ── ASAM ascent step ───────────────────────
            # First forward pass at original weights
            outputs = self.model(fine, coarse, meta)
            fused   = outputs["fused"]

            # Apply RepAugment to fused embedding
            if self.rep_augment is not None:
                fused, labels_aug = self.rep_augment(fused, labels)
            else:
                labels_aug = labels

            # Recompute logits from (possibly augmented) fused embeddings
            logits = self.model.classifier(fused)

            if soft_labels is not None and isinstance(self.criterion, SoftFocalLoss):
                # Extend soft_labels if RepAugment added synthetic samples
                if logits.shape[0] > soft_labels.shape[0]:
                    extra = logits.shape[0] - soft_labels.shape[0]
                    extra_labels = torch.nn.functional.one_hot(
                        labels_aug[-extra:], num_classes=4
                    ).float()
                    soft_labels = torch.cat([soft_labels, extra_labels], dim=0)
                loss = self.criterion(logits, soft_labels)
            else:
                loss = self.criterion(logits, labels_aug)

            loss.backward()

            if self.asam is not None:
                self.asam.ascent_step()
                # Second forward pass at perturbed weights
                outputs2 = self.model(fine, coarse, meta)
                fused2   = outputs2["fused"]
                if self.rep_augment is not None:
                    fused2, labels_aug2 = self.rep_augment(fused2, labels)
                else:
                    labels_aug2 = labels
                logits2 = self.model.classifier(fused2)
                if isinstance(self.criterion, SoftFocalLoss):
                    soft_labels2 = torch.nn.functional.one_hot(
                        labels_aug2, num_classes=4
                    ).float()
                    loss2 = self.criterion(logits2, soft_labels2)
                else:
                    loss2 = self.criterion(logits2, labels_aug2)
                loss2.backward()
                self.asam.descent_step()
            else:
                self.base_optimizer.step()
                self.base_optimizer.zero_grad()

            self.scheduler.step()
            total_loss += loss.item()
            self.global_step += 1

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()

        all_preds, all_labels = [], []
        for batch in tqdm(self.val_loader, desc=f"Val   {epoch}", leave=False):
            fine   = batch["fine_spec"].to(self.device)
            coarse = batch["coarse_spec"].to(self.device)
            meta   = {k: v.to(self.device) for k, v in batch["metadata"].items()}
            labels = batch["label"].to(self.device)

            outputs = self.model(fine, coarse, meta)
            preds   = outputs["logits"].argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        return compute_icbhi_metrics(all_labels, all_preds)

    def _save_checkpoint(self, filename: str) -> None:
        path = self.run_dir / filename
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.base_optimizer.state_dict(),
                "best_monitored":       self.best_monitored,
                "best_icbhi":           self.best_icbhi,
                "global_step":          self.global_step,
            },
            path,
        )