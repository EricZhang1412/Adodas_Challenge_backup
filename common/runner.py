#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .data.dataset import FeatureConfig, ITEM_COLS, A1_COLS
from .data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from .models.mtcn_backbone import BackboneConfig, MTCNBackbone
from .models.heads import (
    A1Head,
    A2OrdinalHead,
    a1_loss,
    a2_combined_loss,
    dass21_consistency_loss,
)
from .models.grouped_model import AuxFiLM, AuxHead, GroupedModel, CORALHead
from .data.dataset import AUX_NAMES
from .utils.seed import seed_everything
from .utils.metrics import binary_f1, macro_auroc, per_class_f1, mean_qwk, mean_mae, per_item_qwk
from .utils.ckpt import save_checkpoint, load_checkpoint
from .utils.run_naming import build_run_name, setup_run_dirs
from .utils.run_metadata import RunMetadata

log = logging.getLogger("train_grouped")


class _RealtimeFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()
        if self.stream is None:
            return
        try:
            os.fsync(self.stream.fileno())
        except OSError:
            pass

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["a1", "a2", "joint"])
    p.add_argument("--config", type=str, default="configs/default.yaml")

    p.add_argument("--feature_root", type=str, default=None)
    p.add_argument("--manifest_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)

    # K-fold CV. When --fold is given, train.csv/val.csv are replaced with
    # manifest_dir/cv{n_folds}/train_fold_{fold}.csv and val_fold_{fold}.csv.
    # Generate the cv{N}/ directory first via scripts/make_cv_folds.py.
    p.add_argument("--fold", type=int, default=None,
                   help="CV fold index (0-based). Requires --n_folds.")
    p.add_argument("--n_folds", type=int, default=None,
                   help="CV total fold count (matches the cv{N} subdirectory).")

    p.add_argument("--audio_features", nargs="+", default=None)
    p.add_argument("--video_features", nargs="+", default=None)
    p.add_argument("--core_audio", nargs="+", default=None)
    p.add_argument("--core_video", nargs="+", default=None)
    p.add_argument("--audio_ssl_model_tag", type=str, default=None)
    p.add_argument("--video_ssl_model_tag", type=str, default=None)

    p.add_argument("--mask_policy", type=str, default=None, choices=['or', 'and_core', 'require_k'])

    p.add_argument("--d_adapter", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--tcn_layers", type=int, default=None)
    p.add_argument("--tcn_kernel_size", type=int, default=None)
    p.add_argument("--asp_alpha", type=float, default=None)
    p.add_argument("--asp_beta", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--d_shared", type=int, default=None)

    p.add_argument("--aggregator", type=str, default=None, choices=["mean", "mlp", "attention"])
    p.add_argument("--session_loss_weight", type=float, default=None)
    p.add_argument("--session_type_loss_weight", type=float, default=None)
    p.add_argument("--use_coral", type=int, default=None, help="1=use CORAL head for A2")

    p.add_argument("--submission_level", type=str, default=None,
                    choices=["session", "participant"], help="Use participant-level preds for submission")
    p.add_argument("--decode_method", type=str, default=None,
                    choices=["auto", "argmax", "expectation", "monotonic"],
                    help="A2 decode: auto-select on val, or use argmax / expectation / monotonic")
    p.add_argument("--label_smoothing", type=float, default=None, help="Label smoothing factor")
    p.add_argument("--feature_noise_std", type=float, default=None, help="Gaussian noise std on features during training")
    p.add_argument("--session_drop_prob", type=float, default=None, help="Prob of dropping a session during training")
    p.add_argument("--early_stop_metric", type=str, default=None,
                    choices=["primary", "val_loss"], help="Metric for early stopping")

    p.add_argument("--a2_soft_qwk_weight", type=float, default=None,
                   help="Weight for differentiable soft-QWK loss on top of CORAL BCE (A2)")
    p.add_argument("--a2_emd_weight", type=float, default=None,
                   help="Weight for ordinal Wasserstein-1 / EMD auxiliary loss (A2)")
    p.add_argument("--a2_expectation_loss_weight", type=float, default=None,
                   help="Weight for SmoothL1(E[y], y) auxiliary loss (A2)")
    p.add_argument("--a2_expectation_distance_aware", type=int, default=None,
                   help="1=distance-aware expectation auxiliary (off-by-3 > off-by-1)")
    p.add_argument("--a2_expectation_distance_power", type=float, default=None,
                   help="Distance-aware power p in (1 + |E[y]-y|^p), typical 1 or 2")

    p.add_argument("--use_aux_supervision", type=int, default=None,
                    help="1=enable participant-level auxiliary attribute supervision (training only)")
    p.add_argument("--aux_loss_weight", type=float, default=None,
                    help="Global lambda multiplied onto the summed aux CE loss")

    # Joint A1+A2 multi-task knobs (only active when --task joint).
    p.add_argument("--a1_loss_weight", type=float, default=None,
                   help="Weight on A1 BCE in joint mode")
    p.add_argument("--a2_loss_weight", type=float, default=None,
                   help="Weight on A2 combined (CORAL+aux) loss in joint mode")
    p.add_argument("--consistency_loss_weight", type=float, default=None,
                   help="Weight on the DASS-21 consistency loss anchoring A1 to A2-derived targets")
    p.add_argument("--consistency_temperature", type=float, default=None,
                   help="Temperature tau of the soft DASS-21 sigmoid; higher = softer targets")
    p.add_argument("--joint_primary_metric", type=str, default=None,
                   choices=["mean", "geomean", "qwk", "f1"],
                   help="How to combine A1 F1 and A2 QWK into the joint primary metric")

    # Concept-bottleneck FiLM injection of demographic aux labels (joint-only).
    # When enabled, AuxHead must also be on; AuxFiLM modulates participant_repr
    # and session_reprs before A1Head/A2Head, using GT labels in training
    # (with scheduled sampling to predicted) and predicted labels at inference.
    p.add_argument("--use_aux_film", type=int, default=None,
                   help="1=enable AuxFiLM injection (joint-only; requires use_aux_supervision=1)")
    p.add_argument("--aux_film_d_embed", type=int, default=None,
                   help="Per-attr embedding dim for AuxFiLM")
    p.add_argument("--aux_film_hidden", type=int, default=None,
                   help="Hidden width inside AuxFiLM MLP")
    p.add_argument("--aux_film_dropout", type=float, default=None,
                   help="Dropout inside AuxFiLM MLP")
    p.add_argument("--aux_teacher_p_start", type=float, default=None,
                   help="Scheduled sampling: initial probability of using GT aux labels")
    p.add_argument("--aux_teacher_p_end", type=float, default=None,
                   help="Scheduled sampling: final probability of using GT aux labels")
    p.add_argument("--aux_teacher_anneal_until_frac", type=float, default=None,
                   help="Fraction of total epochs over which p_gt linearly anneals from start to end")
    p.add_argument("--aux_sampling_mode", type=str, default=None,
                   choices=["per_attr", "per_sample", "batch"],
                   help="Granularity of scheduled sampling")

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--amp", type=int, default=None)
    p.add_argument("--preload", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--use_pos_weight", type=int, default=None)
    p.add_argument("--run_inference_after_train", type=int, default=None)

    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
    cfg = cfg or {}
    feature_selection = cfg.pop("feature_selection", {}) or {}
    if not isinstance(feature_selection, dict):
        raise TypeError("feature_selection must be a mapping in the config YAML")
    cfg.update(feature_selection)
    for k, v in vars(args).items():
        if k == "config":
            continue
        if v is not None:
            cfg[k] = v
    return cfg



def setup_logging(log_dir: Path, task: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_grouped_{task}_{ts}.log"
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = _RealtimeFileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass

    root.addHandler(ch)
    root.addHandler(fh)
    log.info(f"Logging to {log_file}")


class EarlyStopping:
    def __init__(self, patience: int = 6, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  
        self.best_score: float | None = None
        self.counter = 0

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

    def step(self, score: float) -> bool:
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _aux_p_gt(epoch: int, total_epochs: int, p_start: float, p_end: float, anneal_frac: float) -> float:
    """Linear anneal of teacher-forcing probability p_gt over [1, total*frac]."""
    horizon = max(1.0, float(total_epochs) * float(anneal_frac))
    progress = min(1.0, max(0.0, (epoch - 1) / horizon))
    return float(p_start) + (float(p_end) - float(p_start)) * progress


def _sample_aux_input(
    aux_logits_dict: dict[str, torch.Tensor],
    aux_film: AuxFiLM,
    aux_labels: torch.Tensor,
    aux_masks: torch.Tensor,
    p_gt: float,
    mode: str,
) -> torch.Tensor:
    """Choose aux_input_indices (B, n_aux) by per-attr/per-sample/batch scheduled sampling.

    When use_gt is False or aux_masks is False, falls back to argmax(aux_logits).
    Returns long tensor of valid indices in [0, n_classes) for each attr.
    """
    device = aux_labels.device
    B, n_aux = aux_labels.shape
    pred_idx = torch.stack(
        [aux_logits_dict[name].argmax(dim=-1) for name in aux_film.names], dim=-1
    ).long()

    if mode == "per_sample":
        sample_draw = torch.rand(B, 1, device=device) < p_gt
        use_gt = sample_draw.expand(-1, n_aux)
    elif mode == "batch":
        batch_draw = (torch.rand((), device=device) < p_gt).item()
        use_gt = torch.full((B, n_aux), bool(batch_draw), device=device)
    else:  # per_attr (default)
        use_gt = torch.rand(B, n_aux, device=device) < p_gt

    use_gt = use_gt & aux_masks.bool()
    aux_in = torch.where(use_gt, aux_labels.long(), pred_idx)
    return aux_in


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _build_scheduler(optimizer, warmup_epochs, total_epochs):
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)


def _flatten_valid_session_mask(session_valid: torch.Tensor) -> torch.Tensor:
    return session_valid.reshape(-1).bool()


def _normalize_decode_method(decode_method: str | None) -> str:
    if decode_method is None:
        return "argmax"

    method = str(decode_method).strip().lower()
    valid_methods = {"auto", "argmax", "expectation", "monotonic"}
    if method not in valid_methods:
        raise ValueError(
            f"Unsupported decode_method: {decode_method!r}. "
            f"Expected one of {sorted(valid_methods)}"
        )
    return method


def _decode_a2_logits(task_head: nn.Module, logits: torch.Tensor, decode_method: str = "expectation") -> torch.Tensor:
    method = _normalize_decode_method(decode_method)
    if method == "auto":
        raise ValueError("decode_method='auto' is selection-only; pass a concrete decode method")

    if method == "expectation":
        decode_name = "predict_expectation"
    elif method == "monotonic":
        decode_name = "predict_int_monotonic"
    else:
        decode_name = "predict_int"

    decode_fn = getattr(task_head, decode_name, None)
    if decode_fn is None:
        decode_fn = getattr(A2OrdinalHead, decode_name)
    return decode_fn(logits.float())


def _evaluate_a2_decode_candidates(
    task_head: nn.Module,
    logits: torch.Tensor,
    labels: np.ndarray,
    decode_methods: list[str],
    offsets: np.ndarray | None = None,
) -> dict[str, dict[str, float | np.ndarray | str]]:
    logits_f = logits.float()
    if offsets is not None:
        logits_f = logits_f + torch.as_tensor(offsets, device=logits_f.device, dtype=torch.float32)

    results: dict[str, dict[str, float | np.ndarray | str]] = {}
    for method in decode_methods:
        preds = _decode_a2_logits(task_head, logits_f, decode_method=method).cpu().numpy()
        qwk = mean_qwk(preds, labels)
        mae = mean_mae(preds, labels)
        results[method] = {
            "preds": preds,
            "qwk": qwk,
            "mae": mae,
            "decode_method": method,
        }
    return results


def _select_best_a2_result(results: dict[str, dict[str, float | np.ndarray | str]]) -> tuple[str, dict[str, float | np.ndarray | str]]:
    best_name = max(
        results,
        key=lambda name: (
            float(results[name]["qwk"]),
            -float(results[name]["mae"]),
        ),
    )
    return best_name, results[best_name]


def _compute_pos_weight_a1(manifest_path: Path) -> list[float]:
    df = pd.read_csv(manifest_path)
    weights = []
    for col in ["y_D", "y_A", "y_S"]:
        n_pos = df[col].sum()
        n_neg = len(df) - n_pos
        w = float(np.sqrt(n_neg / max(n_pos, 1)))
        w = max(1.0, min(w, 4.0))
        weights.append(w)
    return weights


def _compute_bias_init_a1(manifest_path: Path) -> list[float]:
    df = pd.read_csv(manifest_path)
    biases = []
    for col in ["y_D", "y_A", "y_S"]:
        rate = df[col].mean()
        rate = max(min(rate, 0.99), 0.01)
        biases.append(math.log(rate / (1 - rate)))
    return biases


def compute_a2_pos_weight(manifest_path: Path, n_items=21, n_thresholds=3):
    df = pd.read_csv(manifest_path)
    item_cols = [f"d{i:02d}" for i in range(1, n_items + 1)]
    pw = np.ones((n_items, n_thresholds), dtype=np.float32)
    for j, col in enumerate(item_cols):
        vals = df[col].values.astype(int)
        for k in range(n_thresholds):
            p = max(np.mean(vals >= (k + 1)), 1e-6)
            pw[j, k] = np.clip(np.sqrt((1 - p) / p), 1.0, 10.0)
    return torch.from_numpy(pw).unsqueeze(0)

def train_one_epoch_grouped(
    grouped_model: GroupedModel,
    task_head: nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    epoch: int,
    epochs: int,
    scaler=None,
    use_amp: bool = False,
    pos_weight=None,
    grad_clip: float = 1.0,
    session_loss_weight: float = 0.5,
    session_type_loss_weight: float = 0.15,
    best_metric: float = -1.0,
    label_smoothing: float = 0.0,
    feature_noise_std: float = 0.0,
    a2_soft_qwk_weight: float = 0.0,
    a2_emd_weight: float = 0.0,
    a2_expectation_loss_weight: float = 0.0,
    a2_expectation_distance_aware: bool = False,
    a2_expectation_distance_power: float = 1.0,
    aux_head: AuxHead | None = None,
    aux_loss_weight: float = 0.0,
    aux_per_task_weights: dict[str, float] | None = None,
    a1_head: nn.Module | None = None,
    a2_head: nn.Module | None = None,
    pos_weight_a1=None,
    pos_weight_a2=None,
    a1_loss_weight: float = 1.0,
    a2_loss_weight: float = 1.0,
    consistency_loss_weight: float = 0.3,
    consistency_temperature: float = 1.0,
    aux_film: AuxFiLM | None = None,
    aux_sampling_mode: str = "per_attr",
    p_gt: float = 1.0,
) -> float:
    grouped_model.train()
    if task == "joint":
        assert a1_head is not None and a2_head is not None, "joint training needs both heads"
        a1_head.train()
        a2_head.train()
    else:
        assert task_head is not None, f"task_head required for task={task!r}"
        task_head.train()
    if aux_head is not None:
        aux_head.train()
    if aux_film is not None:
        aux_film.train()
    total_loss = 0.0
    n_batches = 0
    aux_running: dict[str, float] = {n: 0.0 for n in AUX_NAMES}
    aux_running_total = 0.0
    cons_running = 0.0
    a1_running = 0.0
    a2_running = 0.0

    desc = f"Train {epoch}/{epochs}"
    if best_metric >= 0:
        desc += f" [best={best_metric:.4f}]"
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for batch in pbar:
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        session_types = batch["session_types"].to(device)
        B = batch["n_participants"]

        if feature_noise_std > 0.0:
            noise_mask = (~flat_batch["pad_mask"]).unsqueeze(-1).float()
            for key in ("audio_groups", "video_groups"):
                for name in flat_batch[key]:
                    flat_batch[key][name] = flat_batch[key][name] + torch.randn_like(
                        flat_batch[key][name]
                    ) * feature_noise_std * noise_mask

        if task == "a1":
            targets = batch["participant_y_a1"].to(device)
        elif task == "a2":
            targets = batch["participant_y_a2"].to(device).long()
        else:  # joint
            targets_a1 = batch["participant_y_a1"].to(device)
            targets_a2 = batch["participant_y_a2"].to(device).long()

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            valid_session_mask = _flatten_valid_session_mask(session_valid)
            has_valid_sessions = bool(valid_session_mask.any().item())

            if task != "joint":
                p_logits = task_head(out["participant_repr"])
                if task == "a1":
                    main_loss = a1_loss(p_logits, targets, pos_weight=pos_weight, label_smoothing=label_smoothing)
                else:
                    main_loss = a2_combined_loss(
                        p_logits,
                        targets,
                        pos_weight=pos_weight,
                        label_smoothing=label_smoothing,
                        soft_qwk_weight=a2_soft_qwk_weight,
                        emd_weight=a2_emd_weight,
                        expectation_weight=a2_expectation_loss_weight,
                        expectation_distance_aware=a2_expectation_distance_aware,
                        expectation_distance_power=a2_expectation_distance_power,
                    )

                if has_valid_sessions:
                    s_logits = task_head(out["session_reprs"])[valid_session_mask]
                    if task == "a1":
                        s_targets = targets.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)[valid_session_mask]
                        sess_loss = a1_loss(s_logits, s_targets, pos_weight=pos_weight, label_smoothing=label_smoothing)
                    else:
                        s_targets = targets.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 21)[valid_session_mask]
                        sess_loss = a2_combined_loss(
                            s_logits,
                            s_targets,
                            pos_weight=pos_weight,
                            label_smoothing=label_smoothing,
                            soft_qwk_weight=a2_soft_qwk_weight,
                            emd_weight=a2_emd_weight,
                            expectation_weight=a2_expectation_loss_weight,
                            expectation_distance_aware=a2_expectation_distance_aware,
                            expectation_distance_power=a2_expectation_distance_power,
                        )

                    type_loss = F.cross_entropy(
                        out["session_type_logits"][valid_session_mask],
                        session_types[valid_session_mask],
                    )
                else:
                    sess_loss = p_logits.new_zeros(())
                    type_loss = p_logits.new_zeros(())

                loss = main_loss + session_loss_weight * sess_loss + session_type_loss_weight * type_loss
            else:
                # Joint A1+A2. When AuxFiLM is enabled, fuse demographic
                # conditioning into participant_repr / session_reprs *before*
                # the A1/A2 heads. The same aux indices broadcast to all 4
                # sessions of the same participant.
                aux_logits_dict_cache: dict[str, torch.Tensor] | None = None
                fused_p = out["participant_repr"]
                fused_s = out["session_reprs"]
                if aux_film is not None:
                    aux_logits_dict_cache = aux_head(out["participant_repr"])
                    aux_labels_b = batch["participant_aux_labels"].to(device)
                    aux_masks_b = batch["participant_aux_masks"].to(device)
                    aux_in = _sample_aux_input(
                        aux_logits_dict_cache, aux_film,
                        aux_labels_b, aux_masks_b, p_gt, aux_sampling_mode,
                    )
                    fused_p = aux_film(out["participant_repr"], aux_in)
                    aux_in_s = aux_in.unsqueeze(1).expand(-1, 4, -1).reshape(B * 4, -1)
                    fused_s = aux_film(out["session_reprs"], aux_in_s)

                a1_p_logits = a1_head(fused_p)
                a2_p_logits = a2_head(fused_p)

                main_a1 = a1_loss(
                    a1_p_logits, targets_a1,
                    pos_weight=pos_weight_a1, label_smoothing=label_smoothing,
                )
                main_a2 = a2_combined_loss(
                    a2_p_logits, targets_a2,
                    pos_weight=pos_weight_a2, label_smoothing=label_smoothing,
                    soft_qwk_weight=a2_soft_qwk_weight,
                    emd_weight=a2_emd_weight,
                    expectation_weight=a2_expectation_loss_weight,
                    expectation_distance_aware=a2_expectation_distance_aware,
                    expectation_distance_power=a2_expectation_distance_power,
                )
                cons = dass21_consistency_loss(
                    a1_p_logits, a2_p_logits, temperature=consistency_temperature
                )

                if has_valid_sessions:
                    a1_s_logits = a1_head(fused_s)[valid_session_mask]
                    a2_s_logits = a2_head(fused_s)[valid_session_mask]
                    a1_s_targets = targets_a1.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)[valid_session_mask]
                    a2_s_targets = targets_a2.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 21)[valid_session_mask]
                    a1_sess_loss = a1_loss(
                        a1_s_logits, a1_s_targets,
                        pos_weight=pos_weight_a1, label_smoothing=label_smoothing,
                    )
                    a2_sess_loss = a2_combined_loss(
                        a2_s_logits, a2_s_targets,
                        pos_weight=pos_weight_a2, label_smoothing=label_smoothing,
                        soft_qwk_weight=a2_soft_qwk_weight,
                        emd_weight=a2_emd_weight,
                        expectation_weight=a2_expectation_loss_weight,
                        expectation_distance_aware=a2_expectation_distance_aware,
                        expectation_distance_power=a2_expectation_distance_power,
                    )
                    type_loss = F.cross_entropy(
                        out["session_type_logits"][valid_session_mask],
                        session_types[valid_session_mask],
                    )
                else:
                    a1_sess_loss = a1_p_logits.new_zeros(())
                    a2_sess_loss = a1_p_logits.new_zeros(())
                    type_loss = a1_p_logits.new_zeros(())

                loss = (
                    a1_loss_weight * main_a1
                    + a2_loss_weight * main_a2
                    + consistency_loss_weight * cons
                    + session_loss_weight * (a1_sess_loss + a2_sess_loss)
                    + session_type_loss_weight * type_loss
                )

                cons_running += float(cons.detach().item())
                a1_running += float(main_a1.detach().item())
                a2_running += float(main_a2.detach().item())
                p_logits = a1_p_logits  # for aux block below (only used to .new_zeros)

            if aux_head is not None and aux_loss_weight > 0.0:
                aux_labels = batch["participant_aux_labels"].to(device)
                aux_masks = batch["participant_aux_masks"].to(device)
                # In joint+AuxFiLM we already called aux_head once above; reuse
                # its logits instead of paying a second forward.
                if task == "joint" and aux_film is not None and aux_logits_dict_cache is not None:
                    aux_logits_dict = aux_logits_dict_cache
                else:
                    aux_logits_dict = aux_head(out["participant_repr"])
                aux_loss, aux_per_task = aux_head.compute_loss(
                    aux_logits_dict, aux_labels, aux_masks, aux_per_task_weights
                )
                if aux_loss.requires_grad:
                    loss = loss + aux_loss_weight * aux_loss
                aux_running_total += float(aux_loss.detach().item())
                for k, v in aux_per_task.items():
                    aux_running[k] += v

        optimizer.zero_grad()
        if task == "joint":
            clip_params = (
                list(grouped_model.parameters())
                + list(a1_head.parameters())
                + list(a2_head.parameters())
            )
        else:
            clip_params = list(grouped_model.parameters()) + list(task_head.parameters())
        if aux_head is not None:
            clip_params = clip_params + list(aux_head.parameters())
        if aux_film is not None:
            clip_params = clip_params + list(aux_film.parameters())

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(clip_params, max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(clip_params, max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix_str(f"{loss.item():.4f}")

    pbar.close()
    if aux_head is not None and aux_loss_weight > 0.0 and n_batches > 0:
        avg_aux_total = aux_running_total / n_batches
        per_task_avg = {k: v / n_batches for k, v in aux_running.items()}
        per_task_str = " ".join(f"{k}={v:.3f}" for k, v in per_task_avg.items())
        log.info(f"    aux loss avg={avg_aux_total:.4f} | {per_task_str}")
    if task == "joint" and n_batches > 0:
        log.info(
            f"    joint loss avg: a1={a1_running / n_batches:.4f}  "
            f"a2={a2_running / n_batches:.4f}  "
            f"cons={cons_running / n_batches:.4f}"
        )
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_grouped(
    grouped_model: GroupedModel,
    task_head: nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    task: str,
    epoch: int,
    epochs: int,
    use_amp: bool = False,
    pos_weight=None,
    decode_method: str = "expectation",
    a2_soft_qwk_weight: float = 0.0,
    a2_emd_weight: float = 0.0,
    aux_head: AuxHead | None = None,
    a1_head: nn.Module | None = None,
    a2_head: nn.Module | None = None,
    pos_weight_a1=None,
    pos_weight_a2=None,
    a1_loss_weight: float = 1.0,
    a2_loss_weight: float = 1.0,
    consistency_loss_weight: float = 0.3,
    consistency_temperature: float = 1.0,
    joint_primary_metric: str = "mean",
    aux_film: AuxFiLM | None = None,
):
    """Validate grouped model. Returns metrics dict."""
    grouped_model.eval()
    if task == "joint":
        assert a1_head is not None and a2_head is not None, "joint validate needs both heads"
        a1_head.eval()
        a2_head.eval()
    else:
        assert task_head is not None, f"task_head required for task={task!r}"
        task_head.eval()
    if aux_head is not None:
        aux_head.eval()
    if aux_film is not None:
        aux_film.eval()
    decode_method = _normalize_decode_method(decode_method)
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    all_logits = []
    all_sess_preds = []
    # Joint-only buffers (kept empty for single-task paths)
    all_a1_probs: list[np.ndarray] = []
    all_a1_logits: list[np.ndarray] = []
    all_a1_labels: list[np.ndarray] = []
    all_a2_logits_joint: list[torch.Tensor] = []
    all_a2_labels_joint: list[np.ndarray] = []
    aux_correct: dict[str, int] = {n: 0 for n in AUX_NAMES}
    aux_seen: dict[str, int] = {n: 0 for n in AUX_NAMES}

    for batch in tqdm(loader, desc=f"Val {epoch}/{epochs}", leave=False, dynamic_ncols=True):
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]

        if task == "a1":
            targets = batch["participant_y_a1"].to(device)
        elif task == "a2":
            targets = batch["participant_y_a2"].to(device).long()
        else:  # joint
            targets_a1 = batch["participant_y_a1"].to(device)
            targets_a2 = batch["participant_y_a2"].to(device).long()

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            if task != "joint":
                p_logits = task_head(out["participant_repr"])
                if task == "a1":
                    loss = a1_loss(p_logits, targets, pos_weight=pos_weight)
                else:
                    loss = a2_combined_loss(
                        p_logits,
                        targets,
                        pos_weight=pos_weight,
                        soft_qwk_weight=a2_soft_qwk_weight,
                        emd_weight=a2_emd_weight,
                    )
                s_logits = task_head(out["session_reprs"])
            else:
                # Always use predicted aux at validation (mirrors inference).
                fused_p_val = out["participant_repr"]
                aux_logits_dict_val: dict[str, torch.Tensor] | None = None
                if aux_film is not None:
                    aux_logits_dict_val = aux_head(out["participant_repr"])
                    pred_idx_val = torch.stack(
                        [aux_logits_dict_val[n].argmax(dim=-1) for n in aux_film.names],
                        dim=-1,
                    )
                    fused_p_val = aux_film(out["participant_repr"], pred_idx_val)
                a1_p_logits = a1_head(fused_p_val)
                a2_p_logits = a2_head(fused_p_val)
                main_a1 = a1_loss(a1_p_logits, targets_a1, pos_weight=pos_weight_a1)
                main_a2 = a2_combined_loss(
                    a2_p_logits, targets_a2,
                    pos_weight=pos_weight_a2,
                    soft_qwk_weight=a2_soft_qwk_weight,
                    emd_weight=a2_emd_weight,
                )
                cons = dass21_consistency_loss(
                    a1_p_logits, a2_p_logits, temperature=consistency_temperature
                )
                loss = (
                    a1_loss_weight * main_a1
                    + a2_loss_weight * main_a2
                    + consistency_loss_weight * cons
                )

            if aux_head is not None:
                aux_labels = batch["participant_aux_labels"].to(device)
                aux_masks = batch["participant_aux_masks"].to(device)
                if task == "joint" and aux_film is not None and aux_logits_dict_val is not None:
                    aux_logits_dict = aux_logits_dict_val
                else:
                    aux_logits_dict = aux_head(out["participant_repr"])
                for i, name in enumerate(AUX_NAMES):
                    m = aux_masks[:, i]
                    n_valid = int(m.sum().item())
                    if n_valid == 0:
                        continue
                    pred = aux_logits_dict[name][m].argmax(dim=-1)
                    aux_correct[name] += int((pred == aux_labels[m, i]).sum().item())
                    aux_seen[name] += n_valid

        if task == "a1":
            logits_np = p_logits.float().cpu().numpy()
            probs = torch.sigmoid(p_logits.float()).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(targets.cpu().numpy())
            all_logits.append(logits_np)

            s_probs = torch.sigmoid(s_logits.float()).cpu().numpy()
            all_sess_preds.append(s_probs)
        elif task == "a2":
            if decode_method == "auto":
                all_logits.append(p_logits.float().cpu())
            else:
                preds = _decode_a2_logits(task_head, p_logits, decode_method=decode_method)
                all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
        else:  # joint
            a1_logits_f = a1_p_logits.float()
            all_a1_probs.append(torch.sigmoid(a1_logits_f).cpu().numpy())
            all_a1_logits.append(a1_logits_f.cpu().numpy())
            all_a1_labels.append(targets_a1.cpu().numpy())
            all_a2_logits_joint.append(a2_p_logits.float().cpu())
            all_a2_labels_joint.append(targets_a2.cpu().numpy())

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    if aux_head is not None and any(aux_seen.values()):
        parts = []
        for name in AUX_NAMES:
            if aux_seen[name] > 0:
                parts.append(f"{name}={aux_correct[name] / aux_seen[name]:.3f}")
            else:
                parts.append(f"{name}=na")
        log.info(f"    aux acc: {' '.join(parts)}")

    if task == "joint":
        a1_probs_np = np.concatenate(all_a1_probs)
        a1_labels_np = np.concatenate(all_a1_labels)
        a1_logits_np = np.concatenate(all_a1_logits)
        mf1 = binary_f1(a1_probs_np, a1_labels_np, threshold=0.5)
        auroc = macro_auroc(a1_probs_np, a1_labels_np)
        pcf1 = per_class_f1(a1_probs_np, a1_labels_np, threshold=0.5)
        cal_biases, cal_pcf1 = calibrate_a1_bias(a1_logits_np, a1_labels_np)
        cal_logits_np = a1_logits_np + cal_biases.reshape(1, -1)
        cal_probs_np = 1.0 / (1.0 + np.exp(-cal_logits_np))
        cal_mf1 = binary_f1(cal_probs_np, a1_labels_np, threshold=0.5)
        a1_primary = float(max(mf1, cal_mf1))

        a2_labels_np = np.concatenate(all_a2_labels_joint)
        a2_logits_t = torch.cat(all_a2_logits_joint, dim=0)
        a2_decode_used = None
        if decode_method == "auto":
            raw_results = _evaluate_a2_decode_candidates(
                a2_head, a2_logits_t, a2_labels_np,
                decode_methods=["argmax", "monotonic", "expectation"],
            )
            a2_decode_used, best_result = _select_best_a2_result(raw_results)
            a2_preds = best_result["preds"]
            a2_primary = float(best_result["qwk"])
            a2_mae = float(best_result["mae"])
        else:
            a2_preds = _decode_a2_logits(a2_head, a2_logits_t, decode_method=decode_method).numpy()
            a2_decode_used = decode_method
            a2_primary = mean_qwk(a2_preds, a2_labels_np)
            a2_mae = mean_mae(a2_preds, a2_labels_np)

        if joint_primary_metric == "qwk":
            combined = a2_primary
        elif joint_primary_metric == "f1":
            combined = a1_primary
        elif joint_primary_metric == "geomean":
            combined = float(np.sqrt(max(a1_primary, 0.0) * max(a2_primary, 0.0)))
        else:  # mean
            combined = 0.5 * (a1_primary + a2_primary)

        log.info(
            f"    A1: F1={mf1:.4f} (cal={cal_mf1:.4f}) AUROC={auroc:.4f} "
            f"per_class=[{pcf1[0]:.3f}/{pcf1[1]:.3f}/{pcf1[2]:.3f}] | "
            f"A2: QWK={a2_primary:.4f} MAE={a2_mae:.4f} decode={a2_decode_used} | "
            f"primary({joint_primary_metric})={combined:.4f}"
        )

        return {
            "loss": avg_loss,
            "mean_f1": mf1,
            "mean_f1_calibrated": cal_mf1,
            "auroc": auroc,
            "pcf1": pcf1,
            "calibration_biases": cal_biases.tolist(),
            "mean_qwk": a2_primary,
            "mean_mae": a2_mae,
            "selected_decode_method": a2_decode_used,
            "primary_metric": combined,
            "a1_primary": a1_primary,
            "a2_primary": a2_primary,
            "joint_primary_metric": joint_primary_metric,
        }

    if task == "a1":
        probs_np = np.concatenate(all_preds)
        labels_np = np.concatenate(all_labels)
        logits_np = np.concatenate(all_logits)
        mf1 = binary_f1(probs_np, labels_np, threshold=0.5)
        auroc = macro_auroc(probs_np, labels_np)
        pcf1 = per_class_f1(probs_np, labels_np, threshold=0.5)
        cal_biases, cal_pcf1 = calibrate_a1_bias(logits_np, labels_np)
        cal_logits_np = logits_np + cal_biases.reshape(1, -1)
        cal_probs_np = 1.0 / (1.0 + np.exp(-cal_logits_np))
        cal_mf1 = binary_f1(cal_probs_np, labels_np, threshold=0.5)
        selection_source = "calibrated" if cal_mf1 > mf1 else "raw"

        task_names = ["D", "A", "S"]
        for t, name in enumerate(task_names):
            gt = labels_np[:, t]
            pr = (probs_np[:, t] > 0.5).astype(int)
            gt_rate = gt.mean()
            pred_rate = pr.mean()
            p_mean = probs_np[:, t].mean()
            tp = ((pr == 1) & (gt == 1)).sum()
            prec = tp / max(pr.sum(), 1)
            rec = tp / max(gt.sum(), 1)
            log.info(
                f"    {name}: gt_pos={gt_rate:.3f} pred_pos={pred_rate:.3f} "
                f"p_mean={p_mean:.3f} P={prec:.3f} R={rec:.3f} F1={pcf1[t]:.3f}"
            )

        if all_sess_preds:
            sess_probs = np.concatenate(all_sess_preds)
            n_sess = sess_probs.shape[0]
            if n_sess % 4 == 0:
                n_part = n_sess // 4
                sess_grid = sess_probs.reshape(n_part, 4, 3)
                sess_var = np.mean(np.var(sess_grid, axis=1))
                log.info(f"    Session-level variance (collapse metric): {sess_var:.6f}")

        log.info(
            f"    calibrated F1={cal_mf1:.4f} via biases "
            f"D={cal_biases[0]:+.2f} A={cal_biases[1]:+.2f} S={cal_biases[2]:+.2f} "
            f"(selected={selection_source})"
        )

        return {
            "loss": avg_loss, "mean_f1": mf1, "auroc": auroc,
            "pcf1": pcf1,
            "mean_f1_calibrated": cal_mf1,
            "pcf1_calibrated": cal_pcf1,
            "calibration_biases": cal_biases.tolist(),
            "primary_metric": max(mf1, cal_mf1),
            "selection_source": selection_source,
        }
    else:
        labels_np = np.concatenate(all_labels)
        auto_selected_decode = None
        if decode_method == "auto":
            logits_t = torch.cat(all_logits, dim=0)
            raw_results = _evaluate_a2_decode_candidates(
                task_head,
                logits_t,
                labels_np,
                decode_methods=["argmax", "monotonic", "expectation"],
            )
            auto_selected_decode, best_result = _select_best_a2_result(raw_results)
            preds_np = best_result["preds"]
            log.info(
                f"    auto decode selected: {auto_selected_decode} "
                f"(QWK={float(best_result['qwk']):.4f}, MAE={float(best_result['mae']):.4f})"
            )
        else:
            preds_np = np.concatenate(all_preds)
        mqwk = mean_qwk(preds_np, labels_np)
        mmae = mean_mae(preds_np, labels_np)

        total = preds_np.size
        dist = [np.sum(preds_np == v) / total * 100 for v in range(4)]
        gt_dist = [np.sum(labels_np == v) / total * 100 for v in range(4)]
        log.info(f"    pred dist: 0={dist[0]:.1f}% 1={dist[1]:.1f}% 2={dist[2]:.1f}% 3={dist[3]:.1f}%")
        log.info(f"    GT   dist: 0={gt_dist[0]:.1f}% 1={gt_dist[1]:.1f}% 2={gt_dist[2]:.1f}% 3={gt_dist[3]:.1f}%")

        item_qwk = per_item_qwk(preds_np, labels_np)
        ranked = sorted(range(21), key=lambda i: item_qwk[i], reverse=True)
        top3 = " ".join(f"d{r+1:02d}={item_qwk[r]:.3f}" for r in ranked[:3])
        bot3 = " ".join(f"d{r+1:02d}={item_qwk[r]:.3f}" for r in ranked[-3:])
        log.info(f"    top3: {top3}  |  bot3: {bot3}")

        return {
            "loss": avg_loss, "mean_qwk": mqwk, "mean_mae": mmae,
            "primary_metric": mqwk, "selected_decode_method": auto_selected_decode,
        }



@torch.no_grad()
def _aux_fused_repr(
    out: dict[str, torch.Tensor],
    aux_head: AuxHead | None,
    aux_film: AuxFiLM | None,
    submission_level: str,
    n_participants: int,
) -> torch.Tensor:
    """Pick the correct (possibly aux-FiLM-modulated) repr for downstream heads.

    Always uses predicted aux indices (argmax of AuxHead) when aux_film is set.
    """
    if submission_level == "participant":
        base = out["participant_repr"]
    else:
        base = out["session_reprs"]
    if aux_head is None or aux_film is None:
        return base
    aux_logits = aux_head(out["participant_repr"])
    pred_idx = torch.stack(
        [aux_logits[name].argmax(dim=-1) for name in aux_film.names], dim=-1
    )
    if submission_level == "session":
        pred_idx = pred_idx.unsqueeze(1).expand(-1, 4, -1).reshape(n_participants * 4, -1)
    return aux_film(base, pred_idx)


def generate_submission_grouped(
    grouped_model: GroupedModel,
    task_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    use_amp: bool = False,
    desc: str = "Submit",
    submission_level: str = "participant",
    a1_biases: np.ndarray | None = None,
    decode_method: str = "expectation",
    a2_threshold_offsets: np.ndarray | None = None,
    aux_head: AuxHead | None = None,
    aux_film: AuxFiLM | None = None,
):
    grouped_model.eval()
    task_head.eval()
    if aux_head is not None:
        aux_head.eval()
    if aux_film is not None:
        aux_film.eval()
    decode_method = _normalize_decode_method(decode_method)
    if submission_level not in {"participant", "session"}:
        raise ValueError("submission_level must be 'participant' or 'session'")

    all_pids = []
    all_sessions = []
    all_preds = []
    a1_biases_t = None if a1_biases is None else torch.as_tensor(a1_biases, device=device, dtype=torch.float32)
    a2_offsets_t = (
        None if a2_threshold_offsets is None
        else torch.as_tensor(a2_threshold_offsets, device=device, dtype=torch.float32)
    )

    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            repr_for_head = _aux_fused_repr(out, aux_head, aux_film, submission_level, B)
            logits = task_head(repr_for_head)

        if task == "a1":
            logits_f = logits.float()
            if a1_biases_t is not None:
                logits_f = logits_f + a1_biases_t
            preds = torch.sigmoid(logits_f).cpu().numpy()
        else:
            logits_f = logits.float()
            if a2_offsets_t is not None:
                logits_f = logits_f + a2_offsets_t
            preds = _decode_a2_logits(task_head, logits_f, decode_method=decode_method).cpu().numpy()

        if submission_level == "participant":
            participant_ids = [str(pid) for pid in batch["anon_pids"]]
            all_pids.extend(participant_ids)
            all_sessions.extend(["participant"] * len(participant_ids))
        else:
            all_pids.extend(batch["flat_pids"])
            all_sessions.extend(batch["flat_sessions"])
        all_preds.append(preds)

    return all_pids, all_sessions, np.concatenate(all_preds)



@torch.no_grad()
def collect_val_logits_grouped_a1(grouped_model, task_head, loader, device, use_amp,
                                   submission_level="participant",
                                   aux_head: AuxHead | None = None,
                                   aux_film: AuxFiLM | None = None):
    grouped_model.eval()
    task_head.eval()
    if aux_head is not None:
        aux_head.eval()
    if aux_film is not None:
        aux_film.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            repr_for_head = _aux_fused_repr(out, aux_head, aux_film, submission_level, B)
            if submission_level == "participant":
                logits = task_head(repr_for_head).float().cpu().numpy()
                labels = batch["participant_y_a1"].numpy()
            else:
                valid_session_mask = _flatten_valid_session_mask(session_valid).cpu().numpy()
                logits = task_head(repr_for_head).float().cpu().numpy()[valid_session_mask]
                labels = batch["participant_y_a1"].unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3).numpy()
                labels = labels[valid_session_mask]
        all_logits.append(logits)
        all_labels.append(labels)
    return np.concatenate(all_logits), np.concatenate(all_labels)


@torch.no_grad()
def collect_val_logits_grouped_a2(grouped_model, task_head, loader, device, use_amp,
                                   submission_level="participant",
                                   aux_head: AuxHead | None = None,
                                   aux_film: AuxFiLM | None = None):
    """Collect A2 logits and labels from validation set for calibration."""
    grouped_model.eval()
    task_head.eval()
    if aux_head is not None:
        aux_head.eval()
    if aux_film is not None:
        aux_film.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            repr_for_head = _aux_fused_repr(out, aux_head, aux_film, submission_level, B)
            if submission_level == "participant":
                logits = task_head(repr_for_head).float().cpu().numpy()
                labels = batch["participant_y_a2"].numpy()
            else:
                valid_session_mask = _flatten_valid_session_mask(session_valid).cpu().numpy()
                logits = task_head(repr_for_head).float().cpu().numpy()[valid_session_mask]
                labels = batch["participant_y_a2"].unsqueeze(1).expand(-1, 4, -1).reshape(-1, 21).numpy()
                labels = labels[valid_session_mask]
        all_logits.append(logits)
        all_labels.append(labels)
    return np.concatenate(all_logits), np.concatenate(all_labels)


def calibrate_a2_thresholds(logits, labels, n_items=21, n_thresholds=3,
                             grid_min=-2.0, grid_max=2.0, grid_step=0.1,
                             decode_method: str = "expectation"):
    import warnings
    from sklearn.metrics import cohen_kappa_score
    decode_method = _normalize_decode_method(decode_method)
    decode_head = A2OrdinalHead(1)
    grid = np.arange(grid_min, grid_max + grid_step, grid_step)
    offsets = np.zeros((n_items, n_thresholds), dtype=np.float64)
    item_qwks = []

    for j in range(n_items):
        best_qwk = -1.0
        best_offset = np.zeros(n_thresholds)

        # Single shared offset per item (simpler, less overfitting)
        for b in grid:
            shifted = logits[:, j, :] + b  # (N, 3)
            shifted_t = torch.from_numpy(shifted).float().unsqueeze(0)
            preds = _decode_a2_logits(task_head=decode_head, logits=shifted_t, decode_method=decode_method)
            preds = preds.squeeze(0).cpu().numpy().astype(int)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    qwk = cohen_kappa_score(labels[:, j].astype(int), preds, weights="quadratic")
                if not np.isfinite(qwk):
                    qwk = 0.0
            except Exception:
                qwk = 0.0
            if qwk > best_qwk:
                best_qwk = qwk
                best_offset = np.full(n_thresholds, b)

        offsets[j] = best_offset
        item_qwks.append(best_qwk)

    return offsets, item_qwks


def calibrate_a1_bias(logits, labels, grid_min=-3.0, grid_max=3.0, grid_step=0.1):
    from sklearn.metrics import f1_score as skf1
    grid = np.arange(grid_min, grid_max + grid_step, grid_step)
    biases = np.zeros(3, dtype=np.float64)
    best_f1s = []
    for t in range(3):
        best_f1 = -1.0
        best_b = 0.0
        for b in grid:
            probs = 1.0 / (1.0 + np.exp(-(logits[:, t] + b)))
            preds = (probs > 0.5).astype(int)
            f1 = skf1(labels[:, t], preds, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_b = b
        biases[t] = best_b
        best_f1s.append(best_f1)
    return biases, best_f1s



def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    task = cfg["task"]

    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = Path(cfg.get("output_dir", "/media/k3nwong/Data1/test/train/output"))
    manifest_dir = Path(cfg.get("manifest_dir", "/media/k3nwong/Data1/test/outputs/data"))

    # Resolve train/val manifest paths. With CV: cv{N}/train_fold_{k}.csv etc.
    # Without CV (default): the official train.csv / val.csv.
    fold = cfg.get("fold")
    n_folds = cfg.get("n_folds")
    if (fold is None) ^ (n_folds is None):
        raise ValueError(
            "--fold and --n_folds must be specified together (or both omitted). "
            f"Got fold={fold}, n_folds={n_folds}."
        )
    if fold is not None:
        if fold < 0 or fold >= n_folds:
            raise ValueError(f"--fold must be in [0, {n_folds}), got {fold}.")
        cv_dir = manifest_dir / f"cv{n_folds}"
        train_manifest = cv_dir / f"train_fold_{fold}.csv"
        val_manifest = cv_dir / f"val_fold_{fold}.csv"
        if not train_manifest.exists() or not val_manifest.exists():
            raise FileNotFoundError(
                f"CV manifests not found:\n  {train_manifest}\n  {val_manifest}\n"
                f"Generate them first:\n"
                f"  python scripts/make_cv_folds.py --manifest_dir {manifest_dir} "
                f"--n_folds {n_folds}"
            )
    else:
        train_manifest = manifest_dir / "train.csv"
        val_manifest = manifest_dir / "val.csv"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = build_run_name(cfg, task, timestamp, training_mode="grouped_participant")
    run_dirs = setup_run_dirs(output_root, run_name)

    setup_logging(run_dirs["logs"], task)
    log.info(f"Device: {device}")
    log.info(f"Task: {task}")
    log.info(f"Run name: {run_name}")
    log.info(f"Config: {cfg}")

    meta = RunMetadata(run_dirs["root"], cfg, task, run_name)

    _defaults = FeatureConfig()
    feat_cfg = FeatureConfig(
        feature_root=cfg.get("feature_root", _defaults.feature_root),
        audio_features=cfg.get("audio_features", _defaults.audio_features),
        video_features=cfg.get("video_features", _defaults.video_features),
        audio_ssl_model_tag=cfg.get("audio_ssl_model_tag", _defaults.audio_ssl_model_tag),
        video_ssl_model_tag=cfg.get("video_ssl_model_tag", _defaults.video_ssl_model_tag),
        ssl_group_tags=cfg.get("ssl_group_tags", {}),
        vision_ssl_group_tags=cfg.get("vision_ssl_group_tags", {}),
        mask_policy=cfg.get("mask_policy", _defaults.mask_policy),
        core_audio=cfg.get("core_audio", _defaults.core_audio),
        core_video=cfg.get("core_video", _defaults.core_video),
    )
    log.info(f"Mask policy: {feat_cfg.mask_policy}")

    if fold is not None:
        log.info(
            f"CV mode: fold {fold}/{n_folds}  "
            f"train={train_manifest.relative_to(manifest_dir)}  "
            f"val={val_manifest.relative_to(manifest_dir)}"
        )

    train_ds = GroupedParticipantDataset(
        train_manifest, feat_cfg, split="train",
        session_drop_prob=cfg.get("session_drop_prob", 0.1),
    )
    val_ds = GroupedParticipantDataset(val_manifest, feat_cfg, split="val")

    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 8)
    log.info(f"Train: {len(train_ds)} participants, Val: {len(val_ds)} participants")

    preload = bool(cfg.get("preload", True))
    if preload:
        log.info("Preloading data into RAM ...")
        t_pre = time.time()
        train_gb = train_ds.preload(desc="Preload train")
        val_gb = val_ds.preload(desc="Preload val")
        log.info(f"Preload done: {train_gb:.1f}G + {val_gb:.1f}G = {train_gb + val_gb:.1f}G, "
                 f"took {_fmt_duration(time.time() - t_pre)}")
        num_workers = 0

    log.info(f"batch_size={batch_size}, num_workers={num_workers}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=grouped_collate_fn,
        pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=grouped_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    dims = train_ds.feature_dims
    audio_group_dims = {n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims}
    audio_pooled_group_dims = {n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims}
    video_group_dims = {n: dims[n] for n in feat_cfg.video_features if n in dims}

    bb_cfg = BackboneConfig(
        audio_group_dims=audio_group_dims,
        audio_pooled_group_dims=audio_pooled_group_dims,
        video_group_dims=video_group_dims,
        d_adapter=cfg.get("d_adapter", 64),
        d_model=cfg.get("d_model", 256),
        tcn_layers=cfg.get("tcn_layers", 6),
        tcn_kernel_size=cfg.get("tcn_kernel_size", 3),
        asp_alpha=cfg.get("asp_alpha", 0.5),
        asp_beta=cfg.get("asp_beta", 0.5),
        dropout=cfg.get("dropout", 0.2),
        d_shared=cfg.get("d_shared", 256),
        n_sessions=cfg.get("n_sessions", 4),
        d_session=cfg.get("d_session", 16),
        use_cross_modal_attn=bool(cfg.get("use_cross_modal_attn", False)),
        cross_modal_heads=cfg.get("cross_modal_heads", 4),
    )

    backbone = MTCNBackbone(bb_cfg)
    grouped_model = GroupedModel(
        backbone=backbone,
        d_shared=bb_cfg.d_shared,
        aggregator_method=cfg.get("aggregator", "mlp"),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    use_coral = bool(cfg.get("use_coral", False))
    task_head: nn.Module | None = None
    a1_head: nn.Module | None = None
    a2_head: nn.Module | None = None
    if task == "a1":
        bias_init = _compute_bias_init_a1(train_manifest)
        task_head = A1Head(bb_cfg.d_shared, bias_init=bias_init).to(device)
    elif task == "a2":
        if use_coral:
            task_head = CORALHead(bb_cfg.d_shared).to(device)
            log.info("Using CORAL head for A2")
        else:
            task_head = A2OrdinalHead(bb_cfg.d_shared).to(device)
    else:  # joint
        bias_init = _compute_bias_init_a1(train_manifest)
        a1_head = A1Head(bb_cfg.d_shared, bias_init=bias_init).to(device)
        if use_coral:
            a2_head = CORALHead(bb_cfg.d_shared).to(device)
            log.info("Joint mode: A1Head + CORALHead")
        else:
            a2_head = A2OrdinalHead(bb_cfg.d_shared).to(device)
            log.info("Joint mode: A1Head + A2OrdinalHead")

    use_aux_supervision = bool(cfg.get("use_aux_supervision", False))
    aux_loss_weight = float(cfg.get("aux_loss_weight", 0.0))
    aux_per_task_weights = cfg.get("aux_per_task_weights", {}) or {}
    aux_head: AuxHead | None = None
    if use_aux_supervision and aux_loss_weight > 0.0:
        aux_head = AuxHead(bb_cfg.d_shared, dropout=cfg.get("dropout", 0.2)).to(device)
        log.info(
            f"Aux supervision ON: lambda={aux_loss_weight}, "
            f"per_task_weights={aux_per_task_weights or '{all 1.0}'}"
        )

    use_aux_film = (task == "joint") and bool(cfg.get("use_aux_film", False))
    aux_film: AuxFiLM | None = None
    aux_sampling_mode = str(cfg.get("aux_sampling_mode", "per_attr"))
    aux_teacher_p_start = float(cfg.get("aux_teacher_p_start", 1.0))
    aux_teacher_p_end = float(cfg.get("aux_teacher_p_end", 0.3))
    aux_teacher_anneal_until_frac = float(cfg.get("aux_teacher_anneal_until_frac", 0.7))
    if use_aux_film:
        if aux_head is None:
            raise ValueError(
                "use_aux_film=true requires use_aux_supervision=true and aux_loss_weight>0"
            )
        aux_film = AuxFiLM(
            d_shared=bb_cfg.d_shared,
            d_embed=int(cfg.get("aux_film_d_embed", 16)),
            hidden=int(cfg.get("aux_film_hidden", 128)),
            dropout=float(cfg.get("aux_film_dropout", 0.2)),
        ).to(device)
        log.info(
            f"AuxFiLM ON: d_embed={cfg.get('aux_film_d_embed', 16)} "
            f"hidden={cfg.get('aux_film_hidden', 128)} "
            f"sampling={aux_sampling_mode} p_gt={aux_teacher_p_start}->{aux_teacher_p_end} "
            f"anneal_until_frac={aux_teacher_anneal_until_frac}"
        )

    n_params = sum(p.numel() for p in grouped_model.parameters())
    if task == "joint":
        n_params += sum(p.numel() for p in a1_head.parameters())
        n_params += sum(p.numel() for p in a2_head.parameters())
    else:
        n_params += sum(p.numel() for p in task_head.parameters())
    if aux_head is not None:
        n_params += sum(p.numel() for p in aux_head.parameters())
    if aux_film is not None:
        n_params += sum(p.numel() for p in aux_film.parameters())
    log.info(f"Model params: {n_params:,}")

    use_amp = bool(cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        log.info("AMP enabled (BF16)")

    grad_clip = cfg.get("grad_clip", 1.0)
    pos_weight_t = None
    pos_weight_a1_t = None
    pos_weight_a2_t = None
    if cfg.get("use_pos_weight", True):
        if task == "a1":
            pw = _compute_pos_weight_a1(train_manifest)
            pos_weight_t = torch.tensor(pw, dtype=torch.float32, device=device)
            log.info(f"pos_weight [D/A/S]: {pw[0]:.2f} / {pw[1]:.2f} / {pw[2]:.2f}")
        elif task == "a2":
            pos_weight_t = compute_a2_pos_weight(train_manifest).to(device)
            log.info(f"A2 pos_weight shape: {pos_weight_t.shape}")
        else:  # joint
            pw = _compute_pos_weight_a1(train_manifest)
            pos_weight_a1_t = torch.tensor(pw, dtype=torch.float32, device=device)
            pos_weight_a2_t = compute_a2_pos_weight(train_manifest).to(device)
            log.info(f"Joint pos_weight A1 [D/A/S]: {pw[0]:.2f} / {pw[1]:.2f} / {pw[2]:.2f}")
            log.info(f"Joint pos_weight A2 shape: {pos_weight_a2_t.shape}")

    if task == "joint":
        params = (
            list(grouped_model.parameters())
            + list(a1_head.parameters())
            + list(a2_head.parameters())
        )
    else:
        params = list(grouped_model.parameters()) + list(task_head.parameters())
    if aux_head is not None:
        params = params + list(aux_head.parameters())
    if aux_film is not None:
        params = params + list(aux_film.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-2)
    )
    epochs = cfg.get("epochs", 20)
    warmup_epochs = cfg.get("warmup_epochs", 3)
    scheduler = _build_scheduler(optimizer, warmup_epochs, epochs)
    log.info(f"Scheduler: warmup={warmup_epochs} -> cosine, total={epochs}")
    log.info(f"Grad clip: {grad_clip}")

    session_loss_weight = cfg.get("session_loss_weight", 0.5)
    session_type_loss_weight = cfg.get("session_type_loss_weight", 0.15)
    log.info(f"Session loss weight: {session_loss_weight}")
    log.info(f"Session type loss weight: {session_type_loss_weight}")

    patience = cfg.get("patience", 8)
    early_stop_metric = cfg.get("early_stop_metric", "val_loss")
    es_mode = "min" if early_stop_metric == "val_loss" else "max"
    early_stop = EarlyStopping(patience=patience, mode=es_mode)
    log.info(f"EarlyStopping: patience={patience}, metric={early_stop_metric}, mode={es_mode}")

    label_smoothing = cfg.get("label_smoothing", 0.05)
    feature_noise_std = cfg.get("feature_noise_std", 0.01)
    session_drop_prob = cfg.get("session_drop_prob", 0.1)
    log.info(f"Label smoothing: {label_smoothing}")
    log.info(f"Feature noise std: {feature_noise_std}")
    log.info(f"Session drop prob: {session_drop_prob}")

    a2_soft_qwk_weight = float(cfg.get("a2_soft_qwk_weight", 0.0))
    a2_emd_weight = float(cfg.get("a2_emd_weight", 0.0))
    if task in ("a2", "joint"):
        log.info(f"A2 aux losses: soft_QWK weight={a2_soft_qwk_weight}, EMD weight={a2_emd_weight}")

    # Joint-mode loss weights (no-op for single-task).
    a1_loss_weight = float(cfg.get("a1_loss_weight", 1.0))
    a2_loss_weight = float(cfg.get("a2_loss_weight", 1.0))
    consistency_loss_weight = float(cfg.get("consistency_loss_weight", 0.3))
    consistency_temperature = float(cfg.get("consistency_temperature", 1.0))
    joint_primary_metric = str(cfg.get("joint_primary_metric", "mean"))
    if joint_primary_metric not in {"mean", "geomean", "qwk", "f1"}:
        raise ValueError(f"joint_primary_metric must be one of mean/geomean/qwk/f1, got {joint_primary_metric!r}")
    if task == "joint":
        log.info(
            f"Joint loss weights: a1={a1_loss_weight} a2={a2_loss_weight} "
            f"cons={consistency_loss_weight} tau={consistency_temperature} | "
            f"primary={joint_primary_metric}"
        )

    best_metric = -1.0
    if task == "a1":
        metric_name = "F1"
    elif task == "a2":
        metric_name = "QWK"
    else:
        metric_name = f"joint-{joint_primary_metric}"
    t_start = time.time()

    log.info("=" * 90)
    if task == "a1":
        log.info("  Epoch  |    LR     | Train Loss | Val Loss | F1 raw | F1 sel |  AUROC | F1[D/A/S]       | Time")
    elif task == "a2":
        log.info("  Epoch  |    LR     | Train Loss | Val Loss | mean QWK | mean MAE | Time")
    else:
        log.info("  Epoch  |    LR     | Train Loss | Val Loss | F1 (cal) | QWK   | MAE   | primary | Time")
    log.info("=" * 90)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        cur_p_gt = _aux_p_gt(
            epoch, epochs,
            aux_teacher_p_start, aux_teacher_p_end, aux_teacher_anneal_until_frac,
        )
        if aux_film is not None:
            log.info(f"  [Epoch {epoch}] aux scheduled-sampling p_gt={cur_p_gt:.3f}")

        train_loss = train_one_epoch_grouped(
            grouped_model, task_head, train_loader, optimizer, device,
            task, epoch, epochs, scaler, use_amp,
            pos_weight=pos_weight_t, grad_clip=grad_clip,
            session_loss_weight=session_loss_weight,
            session_type_loss_weight=session_type_loss_weight,
            best_metric=best_metric,
            label_smoothing=label_smoothing,
            feature_noise_std=feature_noise_std,
            a2_soft_qwk_weight=a2_soft_qwk_weight,
            a2_emd_weight=a2_emd_weight,
            aux_head=aux_head,
            aux_loss_weight=aux_loss_weight,
            aux_per_task_weights=aux_per_task_weights,
            a1_head=a1_head,
            a2_head=a2_head,
            pos_weight_a1=pos_weight_a1_t,
            pos_weight_a2=pos_weight_a2_t,
            a1_loss_weight=a1_loss_weight,
            a2_loss_weight=a2_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            consistency_temperature=consistency_temperature,
            aux_film=aux_film,
            aux_sampling_mode=aux_sampling_mode,
            p_gt=cur_p_gt,
        )

        val_metrics = validate_grouped(
            grouped_model, task_head, val_loader, device,
            task, epoch, epochs, use_amp, pos_weight=pos_weight_t,
            decode_method=cfg.get("decode_method", "expectation"),
            a2_soft_qwk_weight=a2_soft_qwk_weight,
            a2_emd_weight=a2_emd_weight,
            aux_head=aux_head,
            a1_head=a1_head,
            a2_head=a2_head,
            pos_weight_a1=pos_weight_a1_t,
            pos_weight_a2=pos_weight_a2_t,
            a1_loss_weight=a1_loss_weight,
            a2_loss_weight=a2_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            consistency_temperature=consistency_temperature,
            joint_primary_metric=joint_primary_metric,
            aux_film=aux_film,
        )
        scheduler.step()

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / epoch) * (epochs - epoch)
        lr_now = optimizer.param_groups[0]["lr"]
        vram_gb = torch.cuda.max_memory_allocated() / 1024**3

        primary = val_metrics["primary_metric"]
        is_best = primary > best_metric
        marker = " *" if is_best else ""

        if task == "a1":
            pcf1 = val_metrics.get("pcf1", [0, 0, 0])
            selected_f1 = val_metrics["primary_metric"]
            log.info(
                f"  {epoch:3d}/{epochs:3d} | {lr_now:.2e} |   {train_loss:.4f}   |  {val_metrics['loss']:.4f}  | "
                f"{val_metrics['mean_f1']:.4f} | {selected_f1:.4f} | {val_metrics['auroc']:.4f} | "
                f"{pcf1[0]:.3f}/{pcf1[1]:.3f}/{pcf1[2]:.3f} | "
                f"{_fmt_duration(elapsed)} ETA {_fmt_duration(eta)} VRAM {vram_gb:.1f}G{marker}"
            )
        elif task == "a2":
            log.info(
                f"  {epoch:3d}/{epochs:3d} | {lr_now:.2e} |   {train_loss:.4f}   |  {val_metrics['loss']:.4f}  | "
                f" {val_metrics['mean_qwk']:.4f}  |  {val_metrics['mean_mae']:.4f}  | "
                f"{_fmt_duration(elapsed)} ETA {_fmt_duration(eta)} VRAM {vram_gb:.1f}G{marker}"
            )
        else:
            cal_f1 = val_metrics.get("mean_f1_calibrated", val_metrics["mean_f1"])
            log.info(
                f"  {epoch:3d}/{epochs:3d} | {lr_now:.2e} |   {train_loss:.4f}   |  {val_metrics['loss']:.4f}  | "
                f"{val_metrics['mean_f1']:.4f}({cal_f1:.4f}) | {val_metrics['mean_qwk']:.4f} | "
                f"{val_metrics['mean_mae']:.4f} | {primary:.4f} | "
                f"{_fmt_duration(elapsed)} ETA {_fmt_duration(eta)} VRAM {vram_gb:.1f}G{marker}"
            )

        if is_best:
            best_metric = primary
            if task == "joint":
                ckpt_extra = {
                    "head_state_dict": {
                        "a1": a1_head.state_dict(),
                        "a2": a2_head.state_dict(),
                    }
                }
            else:
                ckpt_extra = {"head_state_dict": task_head.state_dict()}
            if aux_head is not None:
                ckpt_extra["aux_head_state_dict"] = aux_head.state_dict()
            if aux_film is not None:
                ckpt_extra["aux_film_state_dict"] = aux_film.state_dict()
            save_checkpoint(
                run_dirs["checkpoints"] / "best.pt",
                grouped_model, optimizer, epoch, best_metric,
                extra=ckpt_extra,
            )
            log.info(f"  >>> New best {metric_name}={best_metric:.4f} saved at epoch {epoch}.")
            meta.update_best(epoch, val_metrics)

        es_value = val_metrics["loss"] if early_stop_metric == "val_loss" else primary
        if early_stop.step(es_value):
            log.info(f"  EarlyStopping triggered at epoch {epoch} (patience={patience}, metric={early_stop_metric})")
            break

    log.info("=" * 90)
    total_time = time.time() - t_start
    log.info(f"Training complete. Best {metric_name}={best_metric:.4f}, time={_fmt_duration(total_time)}")

    log.info("Loading best checkpoint for submission generation ...")
    state = load_checkpoint(run_dirs["checkpoints"] / "best.pt", grouped_model, optimizer=None)
    if task == "joint":
        head_sd = state["head_state_dict"]
        if not (isinstance(head_sd, dict) and {"a1", "a2"} <= set(head_sd.keys())):
            raise RuntimeError("Joint checkpoint missing 'a1'/'a2' head_state_dict subkeys.")
        a1_head.load_state_dict(head_sd["a1"])
        a2_head.load_state_dict(head_sd["a2"])
        a1_head.to(device)
        a2_head.to(device)
    else:
        task_head.load_state_dict(state["head_state_dict"])
        task_head.to(device)
    if aux_head is not None and "aux_head_state_dict" in state:
        aux_head.load_state_dict(state["aux_head_state_dict"])
        aux_head.to(device)
    if aux_film is not None:
        if "aux_film_state_dict" not in state:
            raise RuntimeError(
                "use_aux_film=true but checkpoint is missing aux_film_state_dict; "
                "the saved best.pt was produced without AuxFiLM."
            )
        aux_film.load_state_dict(state["aux_film_state_dict"])
        aux_film.to(device)
    grouped_model.to(device)

    submission_level = cfg.get("submission_level", "participant")
    decode_method = _normalize_decode_method(cfg.get("decode_method", "expectation"))
    log.info(f"Submission level: {submission_level}")
    log.info(f"Decode method: {decode_method}")

    a1_biases = None
    a2_offsets = None
    selected_decode_method = decode_method

    # Pick the head to use for each calibration: in joint mode we have two heads;
    # in single-task mode the loop only runs for the matching task.
    a1_calib_head = a1_head if task == "joint" else task_head
    a2_calib_head = a2_head if task == "joint" else task_head

    if task in ("a1", "joint"):
        log.info("Calibrating per-task bias offsets on val ...")
        val_logits, val_labels = collect_val_logits_grouped_a1(
            grouped_model, a1_calib_head, val_loader, device, use_amp,
            submission_level=submission_level,
            aux_head=aux_head if task == "joint" else None,
            aux_film=aux_film if task == "joint" else None,
        )
        biases, cal_f1s = calibrate_a1_bias(val_logits, val_labels)
        for t, name in enumerate(["D", "A", "S"]):
            log.info(f"  {name}: bias={biases[t]:+.2f}  F1_cal={cal_f1s[t]:.4f}")
        cal_mean_f1 = float(np.mean(cal_f1s))
        if task == "a1":
            best_raw_f1 = float(meta.meta.get("best_metrics", {}).get("mean_f1", best_metric))
            best_selected_f1 = float(meta.meta.get("best_metrics", {}).get("primary_metric", best_metric))
        else:
            best_raw_f1 = float(meta.meta.get("best_metrics", {}).get("mean_f1", 0.0))
            best_selected_f1 = best_raw_f1
        log.info(
            f"  Mean calibrated F1: {cal_mean_f1:.4f} "
            f"(vs selected best: {best_selected_f1:.4f}, raw best: {best_raw_f1:.4f})"
        )
        a1_biases = biases
        final_a1_metric = max(best_raw_f1, cal_mean_f1)
        final_a1_strategy = "bias_calibrated" if cal_mean_f1 >= best_raw_f1 else "raw"
        if task == "a1":
            meta.set_extra("final_selected_strategy", final_a1_strategy)
            meta.set_extra("final_selected_metrics", {
                "mean_f1": final_a1_metric,
                "mean_f1_raw": best_raw_f1,
                "mean_f1_calibrated": cal_mean_f1,
                "auroc": meta.meta.get("best_metrics", {}).get("auroc"),
            })
        else:
            meta.set_extra("final_a1_strategy", final_a1_strategy)
            meta.set_extra("final_a1_metrics", {
                "mean_f1": final_a1_metric,
                "mean_f1_raw": best_raw_f1,
                "mean_f1_calibrated": cal_mean_f1,
                "auroc": meta.meta.get("best_metrics", {}).get("auroc"),
            })

        cal_data = {"biases": biases.tolist(), "cal_f1": cal_f1s, "mean_cal_f1": cal_mean_f1}
        with open(run_dirs["calibration"] / "a1_bias_grouped.json", "w") as f:
            json.dump(cal_data, f, indent=2)

    if task in ("a2", "joint"):
        log.info("Calibrating and selecting A2 decode strategy on val ...")
        val_logits, val_labels = collect_val_logits_grouped_a2(
            grouped_model, a2_calib_head, val_loader, device, use_amp,
            submission_level=submission_level,
            aux_head=aux_head if task == "joint" else None,
            aux_film=aux_film if task == "joint" else None,
        )
        val_labels_int = val_labels.astype(int)
        raw_results = _evaluate_a2_decode_candidates(
            a2_calib_head,
            torch.from_numpy(val_logits).float(),
            val_labels_int,
            decode_methods=["argmax", "monotonic", "expectation"],
        )
        calibrated_results = {}
        for method in ("argmax", "monotonic", "expectation"):
            offsets, item_qwks = calibrate_a2_thresholds(
                val_logits,
                val_labels_int,
                decode_method=method,
            )
            preds = _decode_a2_logits(
                a2_calib_head,
                torch.from_numpy(val_logits).float() + torch.as_tensor(offsets, dtype=torch.float32),
                decode_method=method,
            ).cpu().numpy()
            calibrated_results[f"calibrated_{method}"] = {
                "preds": preds,
                "qwk": mean_qwk(preds, val_labels_int),
                "mae": mean_mae(preds, val_labels_int),
                "decode_method": method,
                "offsets": offsets,
                "item_qwks": item_qwks,
            }

        strategy_results = {**raw_results, **calibrated_results}
        best_strategy, best_result = _select_best_a2_result(strategy_results)
        selected_decode_method = str(best_result["decode_method"])
        a2_offsets = best_result.get("offsets")

        log.info("  A2 decode comparison on val:")
        for name in ("argmax", "monotonic", "expectation", "calibrated_argmax", "calibrated_monotonic", "calibrated_expectation"):
            result = strategy_results[name]
            preds = result["preds"]
            total = preds.size
            dist = [np.sum(preds == v) / total * 100 for v in range(4)]
            log.info(
                f"    {name:<22} QWK={float(result['qwk']):.4f} MAE={float(result['mae']):.4f} "
                f"| 0={dist[0]:.1f}% 1={dist[1]:.1f}% 2={dist[2]:.1f}% 3={dist[3]:.1f}%"
            )

        log.info(
            f"  Selected A2 strategy: {best_strategy} "
            f"(decode={selected_decode_method}, QWK={float(best_result['qwk']):.4f}, MAE={float(best_result['mae']):.4f})"
        )

        if task == "a2":
            meta.set_extra("final_selected_strategy", best_strategy)
            meta.set_extra("final_selected_metrics", {
                "mean_qwk": float(best_result["qwk"]),
                "mean_mae": float(best_result["mae"]),
                "decode_method": selected_decode_method,
            })
        else:
            meta.set_extra("final_a2_strategy", best_strategy)
            meta.set_extra("final_a2_metrics", {
                "mean_qwk": float(best_result["qwk"]),
                "mean_mae": float(best_result["mae"]),
                "decode_method": selected_decode_method,
            })

        cal_data = {
            "selected_strategy": best_strategy,
            "selected_decode_method": selected_decode_method,
            "selected_qwk": float(best_result["qwk"]),
            "selected_mae": float(best_result["mae"]),
            "strategies": {
                name: {
                    "decode_method": str(result["decode_method"]),
                    "qwk": float(result["qwk"]),
                    "mae": float(result["mae"]),
                    **({"offsets": result["offsets"].tolist()} if "offsets" in result else {}),
                    **({"item_qwks": result["item_qwks"]} if "item_qwks" in result else {}),
                }
                for name, result in strategy_results.items()
            },
        }
        with open(run_dirs["calibration"] / "a2_threshold_offsets_grouped.json", "w") as f:
            json.dump(cal_data, f, indent=2)

    if bool(cfg.get("run_inference_after_train", False)):
        run_dirs["submissions"].mkdir(parents=True, exist_ok=True)
        # In joint mode we emit one CSV per head; otherwise just the single
        # task's head. The tuple lists (head_task_label, head_module).
        if task == "joint":
            inference_heads = [("a1", a1_head), ("a2", a2_head)]
        else:
            inference_heads = [(task, task_head)]
        for split_name in ("val", "test_hidden"):
            # In CV mode, "val" means this fold's held-out participants, not
            # the official val.csv (which mixes participants across folds and
            # would leak through the training set).
            if split_name == "val" and fold is not None:
                manifest_path = val_manifest
            else:
                manifest_path = manifest_dir / f"{split_name}.csv"
            if not manifest_path.exists():
                continue
            ds = GroupedParticipantDataset(manifest_path, feat_cfg, split=split_name)
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=grouped_collate_fn,
            )

            for head_task, head_module in inference_heads:
                pids, sessions, preds = generate_submission_grouped(
                    grouped_model, head_module, loader, device, head_task, use_amp,
                    desc=f"Submit {split_name} ({head_task})",
                    submission_level=submission_level,
                    a1_biases=a1_biases,
                    decode_method=selected_decode_method,
                    a2_threshold_offsets=a2_offsets,
                    aux_head=aux_head if task == "joint" else None,
                    aux_film=aux_film if task == "joint" else None,
                )

                manifest_df = pd.read_csv(manifest_path)
                file_ids = []
                filtered_preds = []
                if submission_level == "participant":
                    pid_to_info = {}
                    for _, row in manifest_df.iterrows():
                        pid = str(row["anon_pid"])
                        pid_to_info.setdefault(pid, (str(row["anon_school"]), str(row["anon_class"])))

                    for pid, pred in zip(pids, preds):
                        pid_str = str(pid)
                        info = pid_to_info.get(pid_str)
                        if info is None:
                            continue
                        school, cls = info
                        file_ids.append(f"{school}_{cls}_{pid_str}")
                        filtered_preds.append(pred)
                    expected_rows = int(manifest_df["anon_pid"].astype(str).nunique())
                else:
                    pid_to_info = {}
                    for _, row in manifest_df.iterrows():
                        pid_to_info[(str(row["anon_pid"]), str(row["session"]))] = (
                            str(row["anon_school"]), str(row["anon_class"])
                        )

                    for pid, sess, pred in zip(pids, sessions, preds):
                        key = (str(pid), str(sess))
                        info = pid_to_info.get(key)
                        if info is None:
                            continue
                        school, cls = info
                        file_ids.append(f"{school}_{cls}_{key[0]}_{key[1]}")
                        filtered_preds.append(pred)
                    expected_rows = len(manifest_df)

                if filtered_preds:
                    preds = np.asarray(filtered_preds)
                elif head_task == "a1":
                    preds = np.zeros((0, 3), dtype=np.float32)
                else:
                    preds = np.zeros((0, 21), dtype=np.int64)
                if len(file_ids) != expected_rows:
                    log.warning(
                        f"Submission row count mismatch for {split_name} ({head_task}): "
                        f"expected={expected_rows} generated={len(file_ids)}"
                    )

                if head_task == "a1":
                    sub = pd.DataFrame({
                        "file_id": file_ids,
                        "p_D": preds[:, 0],
                        "p_A": preds[:, 1],
                        "p_S": preds[:, 2],
                    })
                else:
                    item_cols = [f"d{i:02d}" for i in range(1, 22)]
                    sub = pd.DataFrame({"file_id": file_ids})
                    for j, col in enumerate(item_cols):
                        sub[col] = preds[:, j]

                out_path = run_dirs["submissions"] / f"submission_{head_task}_{split_name}.csv"
                sub.to_csv(out_path, index=False)
                log.info(f"Wrote {len(sub)} rows to {out_path}")
    else:
        log.info("Skipping submission generation after training; use infer.py for release inference.")

    meta.finish("completed")
    log.info(f"Run complete: {run_name}")
    log.info(f"Output dir: {run_dirs['root']}")


if __name__ == "__main__":
    main()
