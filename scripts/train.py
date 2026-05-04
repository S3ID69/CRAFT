"""
train.py — End-to-end training script for MVST-BTS+.

Usage:
    python scripts/train.py --config configs/mvst_bts_plus.yaml

    # Override specific values:
    python scripts/train.py --config configs/mvst_bts_plus.yaml \
        --override training.batch_size=16 optimizer.base_lr=1e-4

    # Kaggle example:
    python scripts/train.py \
        --config configs/mvst_bts_plus.yaml \
        --metadata_csv /kaggle/working/data/metadata.csv \
        --cycles_dir   /kaggle/working/data/cycles \
        --run_dir      /kaggle/working/experiments/run_01 \
        --override training.batch_size=16 training.num_workers=2
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mvst_bts.data.dataset import ICBHIDataset, build_weighted_sampler, collate_fn
from mvst_bts.augmentation.spec_augment import SpecAugment
from mvst_bts.models.mvst_bts_plus import MVSTBTSPlus
from mvst_bts.training.trainer import Trainer
from mvst_bts.utils.config import load_config, save_config
from mvst_bts.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train MVST-BTS+")
    parser.add_argument("--config", type=str, default="configs/mvst_bts_plus.yaml",
                        help="Path to the model config YAML")
    parser.add_argument("--metadata_csv", type=str, default=None,
                        help="Override data.metadata_csv from config")
    parser.add_argument("--cycles_dir", type=str, default=None,
                        help="Override data.cycles_dir from config")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Override training.checkpoint_dir from config")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Dotlist overrides, e.g. training.batch_size=16")
    return parser.parse_args()


def build_dataloaders(cfg, metadata_csv: str, cycles_dir: str):
    aug_cfg = cfg.augmentation
    audio_cfg = OmegaConf.to_container(cfg.audio, resolve=True)
    spec_cfg  = OmegaConf.to_container(cfg.spectrogram, resolve=True)

    train_augs = []
    if aug_cfg.spec_augment.enabled:
        train_augs.append(SpecAugment(
            time_mask_max_ratio=aug_cfg.spec_augment.time_mask_max_ratio,
            freq_mask_max_ratio=aug_cfg.spec_augment.freq_mask_max_ratio,
            num_time_masks=aug_cfg.spec_augment.num_time_masks,
            num_freq_masks=aug_cfg.spec_augment.num_freq_masks,
        ))

    train_ds = ICBHIDataset(
        metadata_csv=metadata_csv,
        cycles_dir=cycles_dir,
        split="train",
        audio_cfg=audio_cfg,
        spec_cfg=spec_cfg,
        augmentations=train_augs if train_augs else None,
    )
    val_ds = ICBHIDataset(
        metadata_csv=metadata_csv,
        cycles_dir=cycles_dir,
        split="test",
        audio_cfg=audio_cfg,
        spec_cfg=spec_cfg,
        augmentations=None,
    )

    sampler = build_weighted_sampler(train_ds)

    num_workers = cfg.training.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")
    return train_loader, val_loader


def main():
    args = parse_args()

    cfg = load_config(args.config, overrides=args.override if args.override else None)

    # CLI path overrides
    if args.metadata_csv:
        OmegaConf.update(cfg, "data.metadata_csv", args.metadata_csv)
    if args.cycles_dir:
        OmegaConf.update(cfg, "data.cycles_dir", args.cycles_dir)

    # Run directory — timestamped subfolder
    base_run_dir = args.run_dir or cfg.training.checkpoint_dir
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.data.seed)

    print("=" * 60)
    print(f"  MVST-BTS+ Training")
    print(f"  Config  : {args.config}")
    print(f"  Run dir : {run_dir}")
    print(f"  Device  : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    save_config(cfg, run_dir / "config.yaml")

    metadata_csv = cfg.data.metadata_csv
    cycles_dir   = cfg.data.cycles_dir

    print("\nBuilding datasets...")
    train_loader, val_loader = build_dataloaders(cfg, metadata_csv, cycles_dir)

    print("\nBuilding model...")
    model = MVSTBTSPlus(cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params / 1e6:.1f}M")

    print("\nStarting training...\n")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        run_dir=run_dir,
    )
    best_metrics = trainer.fit()

    print("\n" + "=" * 60)
    print("  Training complete. Best results:")
    for k, v in best_metrics.items():
        print(f"    {k:25s}: {v:.2f}")
    print(f"\n  Checkpoints saved to: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
