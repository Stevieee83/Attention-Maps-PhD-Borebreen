#!/usr/bin/env python3
"""
Swin Transformer Fine-Tuning Script
=====================================
Fine-tunes swin_base_patch4_window12_384 on a 46-class custom dataset.

Workflow
--------
1. [Optional] Hyperparameter search  (--tune)
   Uses Optuna to find the best lr, weight_decay, drop_path_rate,
   label_smoothing, and batch_size on the train+val splits.

2. Full training  (--train)
   Trains with the best (or manually supplied) hyperparameters.
   Saves two checkpoints:
     <output_dir>/checkpoints/best.pt   — lowest validation loss
     <output_dir>/checkpoints/last.pt   — final epoch

3. Test evaluation  (--test)
   Loads best.pt, runs on the test split, and reports:
     • Accuracy, Precision, Recall, F1  (macro)
     • Per-class breakdown table
     • AUROC curve plot  (one-vs-rest, macro-average)
     • Confusion matrix plot

Dataset layout expected (ImageFolder-compatible)
-----------------------------------------------
    <data_root>/
        train/   class_A/  img1.png ...
                 class_B/  ...
        val/     class_A/  ...
        test/    class_A/  ...

Usage examples
--------------
    # Full pipeline in one command:
    python swin_finetune.py --data_root ./data --output_dir ./runs/exp1 \\
        --tune --n_trials 30 --train --test

    # Skip tuning, train with manual HPs, then test:
    python swin_finetune.py --data_root ./data --output_dir ./runs/exp1 \\
        --train --test --lr 3e-5 --weight_decay 1e-2 --epochs 30

    # Only run test on an existing checkpoint:
    python swin_finetune.py --data_root ./data --output_dir ./runs/exp1 --test
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import argparse
import json
import os
import time
from pathlib import Path

# ── third-party ────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import timm
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import optuna
from optuna.samplers import TPESampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import label_binarize

# ═══════════════════════════════════════════════════════════════════════════
# 1.  GLOBAL DEFAULTS  (all overridable via CLI)
# ═══════════════════════════════════════════════════════════════════════════
NUM_CLASSES    = 46
IMG_SIZE       = 384
MODEL_NAME     = "swin_base_patch4_window12_384"

# Default hyperparameters (used when --tune is not requested)
DEFAULT_LR            = 3e-5
DEFAULT_WEIGHT_DECAY  = 1e-2
DEFAULT_DROP_PATH     = 0.2
DEFAULT_LABEL_SMOOTH  = 0.1
DEFAULT_BATCH_SIZE    = 16
DEFAULT_EPOCHS        = 30
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_MIN_LR        = 1e-7

# Optuna
DEFAULT_N_TRIALS      = 30
TUNE_EPOCHS           = 10     # epochs per trial during HP search

# DataLoader workers
NUM_WORKERS = min(8, os.cpu_count() or 4)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  DATA TRANSFORMS & LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def build_transforms(split: str, img_size: int = IMG_SIZE):
    """
    Build augmentation pipeline for a given split.
    train  : RandAugment + RandomErasing for strong regularisation
    val/test: deterministic centre-crop only
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=(0.7, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.05,
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(
                int(img_size * 1.15),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def build_loaders(data_root: str, batch_size: int):
    """Return (train_loader, val_loader, test_loader, class_names)."""
    root = Path(data_root)

    train_ds = datasets.ImageFolder(root / "train",
                                    transform=build_transforms("train"))
    val_ds   = datasets.ImageFolder(root / "val",
                                    transform=build_transforms("val"))
    test_ds  = datasets.ImageFolder(root / "test",
                                    transform=build_transforms("test"))

    assert len(train_ds.classes) == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes, found {len(train_ds.classes)} "
        f"in {root / 'train'}"
    )

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
        )

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds,   shuffle=False),
        _loader(test_ds,  shuffle=False),
        train_ds.classes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_model(drop_path_rate: float = DEFAULT_DROP_PATH,
                checkpoint: str = None) -> nn.Module:
    """
    Load pretrained Swin, replace the head for NUM_CLASSES outputs.
    Optionally load weights from *checkpoint*.
    """
    model = timm.create_model(
        MODEL_NAME,
        pretrained=(checkpoint is None),
        num_classes=NUM_CLASSES,
        drop_path_rate=drop_path_rate,
    )

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        # support both raw state-dict and our wrapped checkpoint format
        if "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        print(f"  Loaded weights from {checkpoint}")

    return model.to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  OPTIMISER & SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════

def build_optimiser(model: nn.Module, lr: float,
                    weight_decay: float) -> torch.optim.Optimizer:
    """
    AdamW with layer-wise lr decay.
    Backbone parameters use 0.1x the base lr; head uses full lr.
    Bias and LayerNorm parameters are excluded from weight decay.
    """
    decay_params, no_decay_params = [], []
    head_params, head_no_decay    = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = name.startswith("head") or name.startswith("fc")
        if any(nd in name for nd in ("bias", "norm", "LayerNorm")):
            (head_no_decay if is_head else no_decay_params).append(param)
        else:
            (head_params if is_head else decay_params).append(param)

    return torch.optim.AdamW([
        {"params": decay_params,    "lr": lr * 0.1, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr * 0.1, "weight_decay": 0.0},
        {"params": head_params,     "lr": lr,        "weight_decay": weight_decay},
        {"params": head_no_decay,   "lr": lr,        "weight_decay": 0.0},
    ])


def build_scheduler(optimiser, epochs: int, warmup_epochs: int,
                    steps_per_epoch: int, min_lr: float):
    """
    Linear warm-up for *warmup_epochs*, then cosine decay to *min_lr*.
    Returns a step-level scheduler (call .step() every batch).
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-6, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        # scale so the floor is min_lr / base_lr
        base_lr  = optimiser.param_groups[-1]["lr"]
        floor    = min_lr / base_lr if base_lr > 0 else 0.0
        return max(floor, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  ONE EPOCH OF TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimiser, scheduler,
                    mixup_fn, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimiser.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        # for accuracy, use hard labels even when mixup was applied
        hard_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels
        correct     += (logits.argmax(dim=1) == hard_labels).sum().item()
        total       += images.size(0)

    return total_loss / total, correct / total


# ═══════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION (val or test)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, criterion):
    """Returns (loss, accuracy, all_preds, all_labels, all_probs)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)

        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7.  CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, path: Path, meta: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **(meta or {})}, path)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  FULL TRAINING RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_training(data_root, output_dir, hparams, trial=None):
    """
    Train for hparams['epochs'] epochs.
    Saves best.pt (lowest val loss) and last.pt.
    Returns best_val_loss (used by Optuna).
    """
    lr            = hparams["lr"]
    weight_decay  = hparams["weight_decay"]
    drop_path     = hparams["drop_path_rate"]
    label_smooth  = hparams["label_smoothing"]
    batch_size    = hparams["batch_size"]
    epochs        = hparams["epochs"]
    warmup_epochs = hparams.get("warmup_epochs", DEFAULT_WARMUP_EPOCHS)
    min_lr        = hparams.get("min_lr", DEFAULT_MIN_LR)

    train_loader, val_loader, _, _ = build_loaders(data_root, batch_size)

    model     = build_model(drop_path_rate=drop_path)
    optimiser = build_optimiser(model, lr, weight_decay)
    scheduler = build_scheduler(
        optimiser, epochs, warmup_epochs,
        len(train_loader), min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # Mixup / CutMix for stronger regularisation
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0,
        prob=0.8, switch_prob=0.5,
        label_smoothing=label_smooth,
        num_classes=NUM_CLASSES,
    )
    # Use SoftTargetCrossEntropy when mixup is active (labels are soft)
    train_criterion = SoftTargetCrossEntropy()
    eval_criterion  = nn.CrossEntropyLoss(
        label_smoothing=label_smooth
    )

    ckpt_dir      = Path(output_dir) / "checkpoints"
    best_val_loss = float("inf")
    history       = []

    print(f"\n{'─'*65}")
    print(f"  Training   lr={lr:.2e}  wd={weight_decay:.2e}  "
          f"dp={drop_path:.2f}  ls={label_smooth:.2f}  bs={batch_size}")
    print(f"  Epochs={epochs}  Warmup={warmup_epochs}  "
          f"Device={DEVICE}  Workers={NUM_WORKERS}")
    print(f"{'─'*65}")
    header = (f"{'Epoch':>6} | {'Train Loss':>10} {'Train Acc':>10} | "
              f"{'Val Loss':>10} {'Val Acc':>10} | {'LR':>10} | {'Time':>7}")
    print(header)
    print("─" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_criterion,
            optimiser, scheduler, mixup_fn, scaler,
        )
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, eval_criterion,
        )

        elapsed = time.time() - t0
        cur_lr  = optimiser.param_groups[-1]["lr"]

        print(f"{epoch:>6} | {train_loss:>10.4f} {train_acc:>10.4f} | "
              f"{val_loss:>10.4f} {val_acc:>10.4f} | "
              f"{cur_lr:>10.2e} | {elapsed:>6.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_acc": train_acc, "val_loss": val_loss,
            "val_acc": val_acc, "lr": cur_lr,
        })

        # ── save best checkpoint ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, ckpt_dir / "best.pt", {
                "epoch": epoch, "val_loss": val_loss,
                "val_acc": val_acc, "hparams": hparams,
            })
            print(f"         ↳ best.pt saved  (val_loss={val_loss:.4f})")

        # ── Optuna pruning ────────────────────────────────────────────────
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # ── save last checkpoint ──────────────────────────────────────────────
    save_checkpoint(model, ckpt_dir / "last.pt", {
        "epoch": epochs, "val_loss": val_loss,
        "val_acc": val_acc, "hparams": hparams,
    })
    print(f"\n  last.pt saved  (epoch={epochs})")
    print(f"  Best val loss : {best_val_loss:.4f}")

    # ── save training history ─────────────────────────────────────────────
    hist_path = Path(output_dir) / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    _plot_training_curves(history, Path(output_dir))

    return best_val_loss


# ═══════════════════════════════════════════════════════════════════════════
# 9.  HYPERPARAMETER TUNING  (Optuna)
# ═══════════════════════════════════════════════════════════════════════════

def run_tuning(data_root, output_dir, n_trials):
    """
    Run Optuna TPE search.  Returns best hparams dict.
    Each trial trains for TUNE_EPOCHS epochs and returns val_loss.
    """
    print(f"\n{'═'*65}")
    print(f"  Hyperparameter search  ({n_trials} trials × {TUNE_EPOCHS} epochs)")
    print(f"{'═'*65}\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                            n_warmup_steps=3),
    )

    def objective(trial):
        hp = {
            "lr":             trial.suggest_float("lr",           1e-6, 1e-3, log=True),
            "weight_decay":   trial.suggest_float("weight_decay", 1e-4, 0.1,  log=True),
            "drop_path_rate": trial.suggest_float("drop_path_rate", 0.0, 0.5),
            "label_smoothing":trial.suggest_float("label_smoothing", 0.0, 0.3),
            "batch_size":     trial.suggest_categorical("batch_size", [8, 16, 32]),
            "epochs":         TUNE_EPOCHS,
            "warmup_epochs":  2,
            "min_lr":         DEFAULT_MIN_LR,
        }
        trial_dir = Path(output_dir) / "tuning" / f"trial_{trial.number:03d}"
        return run_training(data_root, trial_dir, hp, trial=trial)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_hp = study.best_params
    best_hp["epochs"]        = DEFAULT_EPOCHS
    best_hp["warmup_epochs"] = DEFAULT_WARMUP_EPOCHS
    best_hp["min_lr"]        = DEFAULT_MIN_LR

    # save results
    out = Path(output_dir) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "best_hparams.json", "w") as f:
        json.dump(best_hp, f, indent=2)

    print(f"\n{'─'*65}")
    print("  Best hyperparameters:")
    for k, v in best_hp.items():
        print(f"    {k:<22} = {v}")
    print(f"  Best val loss : {study.best_value:.4f}")
    print(f"{'─'*65}\n")

    # Optuna visualisations (if plotly available)
    try:
        import plotly
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(str(out / "param_importances.png"))
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(str(out / "optimization_history.png"))
    except Exception:
        pass

    return best_hp


# ═══════════════════════════════════════════════════════════════════════════
# 10. TEST EVALUATION & METRICS
# ═══════════════════════════════════════════════════════════════════════════

def run_test(data_root, output_dir, batch_size, class_names):
    """
    Load best.pt, evaluate on the test split, save all metric artefacts.
    """
    ckpt = Path(output_dir) / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"best.pt not found at {ckpt} — run --train first."
        )

    state    = torch.load(ckpt, map_location="cpu")
    dp_rate  = state.get("hparams", {}).get("drop_path_rate", DEFAULT_DROP_PATH)
    model    = build_model(drop_path_rate=dp_rate, checkpoint=str(ckpt))

    _, _, test_loader, loaded_classes = build_loaders(data_root, batch_size)
    names = class_names or loaded_classes

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels, probs = evaluate(
        model, test_loader, criterion
    )

    # ── scalar metrics ────────────────────────────────────────────────────
    prec   = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1     = f1_score(labels, preds, average="macro", zero_division=0)

    # AUROC (one-vs-rest, macro)
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    try:
        auroc = roc_auc_score(labels_bin, probs,
                               multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")

    metrics = {
        "test_loss":  round(float(test_loss), 4),
        "accuracy":   round(float(test_acc),  4),
        "precision":  round(float(prec),      4),
        "recall":     round(float(recall),    4),
        "f1_macro":   round(float(f1),        4),
        "auroc_macro":round(float(auroc),     4) if not np.isnan(auroc) else None,
    }

    test_out = Path(output_dir) / "test_results"
    test_out.mkdir(parents=True, exist_ok=True)

    # ── print summary ─────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  TEST RESULTS")
    print(f"{'─'*65}")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print(f"{'─'*65}")

    # ── per-class breakdown ───────────────────────────────────────────────
    report = classification_report(
        labels, preds,
        target_names=names, zero_division=0, digits=4,
    )
    print("\nPer-class report:\n")
    print(report)
    with open(test_out / "classification_report.txt", "w") as f:
        f.write(report)

    with open(test_out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── confusion matrix ──────────────────────────────────────────────────
    _plot_confusion_matrix(labels, preds, names, test_out)

    # ── AUROC ─────────────────────────────────────────────────────────────
    _plot_auroc(labels_bin, probs, names, auroc, test_out)

    print(f"\n  Artefacts saved to: {test_out}/")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 11. PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _plot_training_curves(history, out_dir: Path):
    epochs     = [h["epoch"]     for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    train_acc  = [h["train_acc"]  for h in history]
    val_acc    = [h["val_acc"]    for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss,   label="Val Loss",   linewidth=2)
    best_ep = epochs[int(np.argmin(val_loss))]
    ax1.axvline(best_ep, color="red", linestyle="--",
                linewidth=1, label=f"Best epoch ({best_ep})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss curves"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in train_acc],
             label="Train Acc", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in val_acc],
             label="Val Acc",   linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy curves"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(labels, preds, class_names, out_dir: Path):
    cm   = confusion_matrix(labels, preds)
    n    = len(class_names)
    size = max(12, n * 0.45)

    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # annotate cells
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    tick_locs = np.arange(n)
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)
    short = [c[:18] for c in class_names]   # truncate long names
    ax.set_xticklabels(short, rotation=90, fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.set_title("Confusion Matrix (Test Set)", fontsize=13)

    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Confusion matrix saved.")


def _plot_auroc(labels_bin, probs, class_names, macro_auc, out_dir: Path):
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(10, 8))

    # per-class curves (thin, semi-transparent)
    for i, cname in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            class_auc   = auc(fpr, tpr)
            ax.plot(fpr, tpr, alpha=0.25, linewidth=0.8,
                    label=f"{cname[:20]} ({class_auc:.2f})")
        except ValueError:
            pass

    # macro-average curve
    all_fpr = np.unique(np.concatenate([
        roc_curve(labels_bin[:, i], probs[:, i])[0]
        for i in range(NUM_CLASSES)
        if len(np.unique(labels_bin[:, i])) > 1
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        if len(np.unique(labels_bin[:, i])) > 1:
            fpr_i, tpr_i, _ = roc_curve(labels_bin[:, i], probs[:, i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= NUM_CLASSES

    ax.plot(all_fpr, mean_tpr, color="navy", linewidth=2.5,
            label=f"Macro-avg ROC (AUC={macro_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"AUROC Curve — Test Set  (macro={macro_auc:.4f})",
                 fontsize=13)

    # legend: two columns for readability with 46 classes
    ax.legend(loc="lower right", fontsize=6,
              ncol=2, framealpha=0.7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "auroc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  AUROC plot saved.")


# ═══════════════════════════════════════════════════════════════════════════
# 12. CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Swin Transformer on a 46-class custom dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── paths ───────────────────────────────────────────────────────────
    p.add_argument("--data_root",  "-d", required=True,
                   help="Root directory containing train/ val/ test/ folders.")
    p.add_argument("--output_dir", "-o", default="./runs/exp1",
                   help="Directory for checkpoints, logs, and plots.")

    # ── pipeline stages ─────────────────────────────────────────────────
    p.add_argument("--tune",  action="store_true",
                   help="Run Optuna hyperparameter search before training.")
    p.add_argument("--train", action="store_true",
                   help="Run full training.")
    p.add_argument("--test",  action="store_true",
                   help="Evaluate best.pt on the test split.")

    # ── training hyperparameters (ignored when --tune is used) ──────────
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--drop_path",    type=float, default=DEFAULT_DROP_PATH,
                   dest="drop_path_rate")
    p.add_argument("--label_smooth", type=float, default=DEFAULT_LABEL_SMOOTH,
                   dest="label_smoothing")
    p.add_argument("--batch_size",   type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--warmup_epochs",type=int,   default=DEFAULT_WARMUP_EPOCHS)
    p.add_argument("--min_lr",       type=float, default=DEFAULT_MIN_LR)

    # ── tuning ───────────────────────────────────────────────────────────
    p.add_argument("--n_trials",     type=int,   default=DEFAULT_N_TRIALS,
                   help="Number of Optuna trials (only used with --tune).")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 13. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if not (args.tune or args.train or args.test):
        print("Nothing to do — pass at least one of --tune, --train, --test.")
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  Swin Transformer Fine-Tuning")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Classes : {NUM_CLASSES}")
    print(f"  Device  : {DEVICE}")
    print(f"  Data    : {args.data_root}")
    print(f"  Output  : {out}")
    print(f"{'═'*65}\n")

    # ── build final hparams dict ─────────────────────────────────────────
    hparams = {
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "drop_path_rate": args.drop_path_rate,
        "label_smoothing":args.label_smoothing,
        "batch_size":     args.batch_size,
        "epochs":         args.epochs,
        "warmup_epochs":  args.warmup_epochs,
        "min_lr":         args.min_lr,
    }

    # ── stage 1: HP tuning ───────────────────────────────────────────────
    if args.tune:
        best_hp = run_tuning(args.data_root, str(out), args.n_trials)
        # override epochs/warmup with full-run values
        best_hp["epochs"]        = args.epochs
        best_hp["warmup_epochs"] = args.warmup_epochs
        best_hp["min_lr"]        = args.min_lr
        hparams = best_hp
        with open(out / "final_hparams.json", "w") as f:
            json.dump(hparams, f, indent=2)

    # ── stage 2: full training ───────────────────────────────────────────
    if args.train:
        print(f"\n{'═'*65}")
        print("  FULL TRAINING RUN")
        print(f"{'═'*65}")
        run_training(args.data_root, str(out), hparams)

    # ── stage 3: test evaluation ─────────────────────────────────────────
    if args.test:
        print(f"\n{'═'*65}")
        print("  TEST EVALUATION")
        print(f"{'═'*65}")
        # load class names from train split
        train_ds   = datasets.ImageFolder(
            Path(args.data_root) / "train",
            transform=build_transforms("test"),
        )
        class_names = train_ds.classes
        run_test(args.data_root, str(out), args.batch_size, class_names)


if __name__ == "__main__":
    main()
