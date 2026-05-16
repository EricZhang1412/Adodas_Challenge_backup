#!/usr/bin/env python3
"""Stage-2 trainer: AuxPredictor over a frozen GT-input main model.

Trains a 5-head classifier on top of the frozen participant_repr produced by a
stage-1 joint checkpoint, so inference can substitute its softmax probs for
the manifest GT aux labels (via AuxFiLM.forward_soft).

Usage:
    python scripts/train_aux_predictor.py \
        --from_main_ckpt output/.../checkpoints/best.pt \
        [--config tasks/aux_predictor/default.yaml] \
        [--epochs 20] [--lr 1e-3] [--out_ckpt_name best_with_predictor.pt]

Outputs ``<main run_dir>/checkpoints/<out_ckpt_name>`` containing every
field from the stage-1 ckpt plus ``aux_predictor_state_dict``,
``aux_predictor_config`` and ``aux_predictor_val_metrics``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data.dataset import AUX_NAMES, AUX_SCHEMA, FeatureConfig  # noqa: E402
from common.data.grouped_dataset import (  # noqa: E402
    GroupedParticipantDataset,
    grouped_collate_fn,
)
from common.models.grouped_model import (  # noqa: E402
    AuxFiLM,
    AuxPredictor,
    CORALHead,
    GroupedModel,
)
from common.models.heads import A1Head, A2OrdinalHead  # noqa: E402
from common.models.mtcn_backbone import BackboneConfig, MTCNBackbone  # noqa: E402
from common.runner import (  # noqa: E402
    EarlyStopping,
    _RealtimeFileHandler,
    _build_scheduler,
    _fmt_duration,
    _to_device,
    setup_logging,
    validate_grouped,
)
from common.utils.seed import seed_everything  # noqa: E402

log = logging.getLogger("aux_predictor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--from_main_ckpt", type=str, required=True,
                   help="Path to stage-1 best.pt (joint, with aux_film_state_dict)")
    p.add_argument("--config", type=str,
                   default="tasks/aux_predictor/default.yaml",
                   help="Predictor training config")
    p.add_argument("--main_config", type=str, default=None,
                   help="Override path to stage-1 config_used.yaml "
                        "(default: <ckpt run_dir>/config_used.yaml)")
    p.add_argument("--out_ckpt_name", type=str, default=None,
                   help="Override output ckpt filename (default from yaml)")

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    feature_selection = cfg.pop("feature_selection", {}) or {}
    if not isinstance(feature_selection, dict):
        raise TypeError("feature_selection must be a mapping in the config YAML")
    cfg.update(feature_selection)
    return cfg


def merge_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    for k in ("epochs", "lr", "batch_size", "seed", "num_workers", "patience"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.out_ckpt_name is not None:
        cfg["out_ckpt_name"] = args.out_ckpt_name
    return cfg


def resolve_manifests(main_cfg: dict, main_ckpt: Path) -> tuple[Path, Path, Path]:
    """Return (manifest_dir, train_manifest, val_manifest) mirroring runner.main."""
    manifest_dir = Path(main_cfg.get("manifest_dir", ""))
    if not manifest_dir.exists():
        raise FileNotFoundError(
            f"manifest_dir from stage-1 config does not exist: {manifest_dir}"
        )
    fold = main_cfg.get("fold")
    n_folds = main_cfg.get("n_folds")
    if (fold is None) ^ (n_folds is None):
        raise ValueError("stage-1 cfg has only one of fold/n_folds set")
    if fold is not None:
        cv_dir = manifest_dir / f"cv{n_folds}"
        train_manifest = cv_dir / f"train_fold_{fold}.csv"
        val_manifest = cv_dir / f"val_fold_{fold}.csv"
    else:
        train_manifest = manifest_dir / "train.csv"
        val_manifest = manifest_dir / "val.csv"
    if not train_manifest.exists():
        raise FileNotFoundError(f"train manifest not found: {train_manifest}")
    if not val_manifest.exists():
        raise FileNotFoundError(f"val manifest not found: {val_manifest}")
    return manifest_dir, train_manifest, val_manifest


def build_feat_cfg(main_cfg: dict) -> FeatureConfig:
    defaults = FeatureConfig()
    return FeatureConfig(
        feature_root=main_cfg.get("feature_root", defaults.feature_root),
        audio_features=main_cfg.get("audio_features", defaults.audio_features),
        video_features=main_cfg.get("video_features", defaults.video_features),
        audio_ssl_model_tag=main_cfg.get("audio_ssl_model_tag", defaults.audio_ssl_model_tag),
        video_ssl_model_tag=main_cfg.get("video_ssl_model_tag", defaults.video_ssl_model_tag),
        ssl_group_tags=main_cfg.get("ssl_group_tags", {}),
        vision_ssl_group_tags=main_cfg.get("vision_ssl_group_tags", {}),
        mask_policy=main_cfg.get("mask_policy", defaults.mask_policy),
        core_audio=main_cfg.get("core_audio", defaults.core_audio),
        core_video=main_cfg.get("core_video", defaults.core_video),
    )


def build_backbone(main_cfg: dict, feat_cfg: FeatureConfig, train_ds: GroupedParticipantDataset) -> BackboneConfig:
    dims = train_ds.feature_dims
    return BackboneConfig(
        audio_group_dims={n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims},
        audio_pooled_group_dims={n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims},
        video_group_dims={n: dims[n] for n in feat_cfg.video_features if n in dims},
        d_adapter=main_cfg.get("d_adapter", 64),
        d_model=main_cfg.get("d_model", 256),
        tcn_layers=main_cfg.get("tcn_layers", 6),
        tcn_kernel_size=main_cfg.get("tcn_kernel_size", 3),
        asp_alpha=main_cfg.get("asp_alpha", 0.5),
        asp_beta=main_cfg.get("asp_beta", 0.5),
        dropout=main_cfg.get("dropout", 0.2),
        d_shared=main_cfg.get("d_shared", 256),
        n_sessions=main_cfg.get("n_sessions", 4),
        d_session=main_cfg.get("d_session", 16),
        use_cross_modal_attn=bool(main_cfg.get("use_cross_modal_attn", False)),
        cross_modal_heads=main_cfg.get("cross_modal_heads", 4),
    )


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


@torch.no_grad()
def collect_participant_reprs(
    grouped_model: GroupedModel, loader: DataLoader, device: torch.device, use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the frozen backbone once over a loader, cache (repr, aux_labels, aux_masks)."""
    reprs, labels, masks = [], [], []
    for batch in tqdm(loader, desc="Cache repr", leave=False, dynamic_ncols=True):
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
        reprs.append(out["participant_repr"].float().cpu())
        labels.append(batch["participant_aux_labels"])
        masks.append(batch["participant_aux_masks"])
    return torch.cat(reprs), torch.cat(labels), torch.cat(masks)


def predictor_eval(
    aux_predictor: AuxPredictor,
    reprs: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict:
    """Per-attr accuracy and macro F1 on cached val reprs."""
    aux_predictor.eval()
    correct = {n: 0 for n in AUX_NAMES}
    seen = {n: 0 for n in AUX_NAMES}
    all_pred: dict[str, list[int]] = {n: [] for n in AUX_NAMES}
    all_true: dict[str, list[int]] = {n: [] for n in AUX_NAMES}
    n = reprs.size(0)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            x = reprs[s:s + batch_size].to(device)
            logits = aux_predictor(x)
            for i, name in enumerate(AUX_NAMES):
                m = masks[s:s + batch_size, i].bool()
                if not bool(m.any()):
                    continue
                pred = logits[name][m].argmax(dim=-1).cpu().numpy()
                tgt = labels[s:s + batch_size, i][m].cpu().numpy()
                correct[name] += int((pred == tgt).sum())
                seen[name] += int(m.sum().item())
                all_pred[name].extend(pred.tolist())
                all_true[name].extend(tgt.tolist())
    out: dict = {"per_attr_acc": {}, "per_attr_macro_f1": {}, "per_attr_seen": seen}
    f1s = []
    for name in AUX_NAMES:
        if seen[name] == 0:
            out["per_attr_acc"][name] = float("nan")
            out["per_attr_macro_f1"][name] = float("nan")
            continue
        out["per_attr_acc"][name] = correct[name] / seen[name]
        f1 = f1_score(all_true[name], all_pred[name], average="macro", zero_division=0.0)
        out["per_attr_macro_f1"][name] = float(f1)
        f1s.append(float(f1))
    out["mean_macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    return out


def main() -> None:
    args = parse_args()
    pred_cfg = merge_overrides(load_yaml(Path(args.config)), args)

    main_ckpt_path = Path(args.from_main_ckpt).resolve()
    if not main_ckpt_path.exists():
        raise FileNotFoundError(f"main ckpt not found: {main_ckpt_path}")
    run_dir = main_ckpt_path.parent.parent  # .../runs/<run_name>/
    ckpt_dir = main_ckpt_path.parent

    main_cfg_path = Path(args.main_config) if args.main_config else run_dir / "config_used.yaml"
    main_cfg = load_yaml(main_cfg_path)

    if not bool(main_cfg.get("use_aux_film", False)):
        raise RuntimeError("stage-1 cfg has use_aux_film=false; predictor stage requires AuxFiLM")
    if str(main_cfg.get("task", "joint")) != "joint":
        raise RuntimeError("stage-2 predictor only supported when stage-1 task==joint")

    out_name = pred_cfg.get("out_ckpt_name", "best_with_predictor.pt")
    out_ckpt_path = ckpt_dir / out_name
    log_dir = run_dir / "logs"
    setup_logging(log_dir, "aux_predictor")
    log.info(f"main ckpt: {main_ckpt_path}")
    log.info(f"main cfg : {main_cfg_path}")
    log.info(f"output    : {out_ckpt_path}")

    seed_everything(int(pred_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, train_manifest, val_manifest = resolve_manifests(main_cfg, main_ckpt_path)
    log.info(f"train manifest: {train_manifest}")
    log.info(f"val manifest  : {val_manifest}")

    feat_cfg = build_feat_cfg(main_cfg)
    train_ds = GroupedParticipantDataset(train_manifest, feat_cfg, split="train",
                                         session_drop_prob=0.0)
    val_ds = GroupedParticipantDataset(val_manifest, feat_cfg, split="val")
    log.info(f"train participants: {len(train_ds)}, val participants: {len(val_ds)}")

    batch_size = int(pred_cfg.get("batch_size", 32))
    num_workers = int(pred_cfg.get("num_workers", 8))
    if bool(pred_cfg.get("preload", True)):
        log.info("preloading features ...")
        t_pre = time.time()
        train_ds.preload(desc="Preload train")
        val_ds.preload(desc="Preload val")
        log.info(f"preload took {_fmt_duration(time.time() - t_pre)}")
        num_workers = 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=grouped_collate_fn, pin_memory=True,
        drop_last=False, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=grouped_collate_fn, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    bb_cfg = build_backbone(main_cfg, feat_cfg, train_ds)
    grouped_model = GroupedModel(
        backbone=MTCNBackbone(bb_cfg),
        d_shared=bb_cfg.d_shared,
        aggregator_method=main_cfg.get("aggregator", "mlp"),
        dropout=main_cfg.get("dropout", 0.2),
    ).to(device)
    a1_head = A1Head(bb_cfg.d_shared).to(device)
    use_coral = bool(main_cfg.get("use_coral", False))
    a2_head = (CORALHead if use_coral else A2OrdinalHead)(bb_cfg.d_shared).to(device)
    aux_film = AuxFiLM(
        d_shared=bb_cfg.d_shared,
        d_embed=int(main_cfg.get("aux_film_d_embed", 16)),
        hidden=int(main_cfg.get("aux_film_hidden", 128)),
        dropout=float(main_cfg.get("aux_film_dropout", 0.2)),
    ).to(device)

    state = torch.load(main_ckpt_path, map_location="cpu", weights_only=False)
    grouped_model.load_state_dict(state["model_state_dict"])
    head_sd = state["head_state_dict"]
    if not (isinstance(head_sd, dict) and {"a1", "a2"} <= set(head_sd.keys())):
        raise RuntimeError("main ckpt head_state_dict must contain 'a1' and 'a2' sub-keys")
    a1_head.load_state_dict(head_sd["a1"])
    a2_head.load_state_dict(head_sd["a2"])
    if "aux_film_state_dict" not in state:
        raise RuntimeError("main ckpt missing aux_film_state_dict")
    aux_film.load_state_dict(state["aux_film_state_dict"])

    freeze(grouped_model)
    freeze(a1_head)
    freeze(a2_head)
    freeze(aux_film)

    predictor_hidden = int(pred_cfg.get("predictor_hidden", 128))
    predictor_dropout = float(pred_cfg.get("predictor_dropout", 0.2))
    aux_predictor = AuxPredictor(
        d_in=bb_cfg.d_shared,
        hidden=predictor_hidden,
        dropout=predictor_dropout,
    ).to(device)
    log.info(
        f"AuxPredictor: d_in={bb_cfg.d_shared} hidden={predictor_hidden} "
        f"dropout={predictor_dropout}  "
        f"params={sum(p.numel() for p in aux_predictor.parameters()):,}"
    )

    use_amp = bool(pred_cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    epochs = int(pred_cfg.get("epochs", 20))
    warmup_epochs = int(pred_cfg.get("warmup_epochs", 2))
    optimizer = torch.optim.AdamW(
        aux_predictor.parameters(),
        lr=float(pred_cfg.get("lr", 1e-3)),
        weight_decay=float(pred_cfg.get("weight_decay", 1e-2)),
    )
    scheduler = _build_scheduler(optimizer, warmup_epochs, epochs)
    per_task_weights = pred_cfg.get("predictor_per_task_weights", {}) or {}
    grad_clip = float(pred_cfg.get("grad_clip", 1.0))

    es_metric = str(pred_cfg.get("early_stop_metric", "macro_f1"))
    es_mode = "max" if es_metric == "macro_f1" else "min"
    early_stop = EarlyStopping(patience=int(pred_cfg.get("patience", 6)), mode=es_mode)

    log.info("caching val participant_reprs from frozen backbone ...")
    val_reprs, val_aux_labels, val_aux_masks = collect_participant_reprs(
        grouped_model, val_loader, device, use_amp,
    )
    log.info(f"val cache: reprs={tuple(val_reprs.shape)}")

    best_score: float | None = None
    best_state: dict | None = None
    best_epoch = -1
    best_val_metrics: dict = {}
    t_start = time.time()
    log.info("=" * 90)
    log.info(" Epoch |    LR     | TrainLoss | Val macroF1 (mean) | per-attr F1            | Time")
    log.info("=" * 90)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        aux_predictor.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False, dynamic_ncols=True):
            flat_batch = _to_device(batch["flat_batch"], device)
            session_valid = batch["session_valid"].to(device)
            B = batch["n_participants"]
            aux_labels = batch["participant_aux_labels"].to(device)
            aux_masks = batch["participant_aux_masks"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                with torch.no_grad():
                    out = grouped_model(flat_batch, B, session_valid)
                logits = aux_predictor(out["participant_repr"])
                loss, _per_task = aux_predictor.compute_loss(
                    logits, aux_labels, aux_masks, per_task_weights,
                )

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(aux_predictor.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(aux_predictor.parameters(), max_norm=grad_clip)
                optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        val_metrics = predictor_eval(
            aux_predictor, val_reprs, val_aux_labels, val_aux_masks, device, batch_size,
        )
        mean_f1 = val_metrics["mean_macro_f1"]
        per_attr_f1 = " ".join(
            f"{n}={val_metrics['per_attr_macro_f1'][n]:.3f}" for n in AUX_NAMES
        )
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        score = mean_f1 if es_metric == "macro_f1" else train_loss
        is_best = (best_score is None) or (
            (score > best_score) if es_mode == "max" else (score < best_score)
        )
        marker = " *" if is_best else ""
        log.info(
            f"  {epoch:3d}/{epochs:3d} | {lr_now:.2e} |  {train_loss:.4f}  | "
            f"{mean_f1:.4f}            | {per_attr_f1} | "
            f"{_fmt_duration(elapsed)}{marker}"
        )

        if is_best:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in aux_predictor.state_dict().items()}
            best_epoch = epoch
            best_val_metrics = val_metrics

        if early_stop.step(score):
            log.info(f"  EarlyStopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("no best checkpoint captured — training did not run any epoch")
    log.info(f"best epoch: {best_epoch}  best {es_metric}: {best_score:.4f}")
    log.info(f"per-attr acc:  " + " ".join(
        f"{n}={best_val_metrics['per_attr_acc'][n]:.3f}" for n in AUX_NAMES
    ))
    log.info(f"per-attr F1:   " + " ".join(
        f"{n}={best_val_metrics['per_attr_macro_f1'][n]:.3f}" for n in AUX_NAMES
    ))

    aux_predictor.load_state_dict(best_state)

    log.info("end-to-end val: GT-input baseline vs predictor soft-embedding ...")
    val_gt = validate_grouped(
        grouped_model, None, val_loader, device,
        task="joint", epoch=0, epochs=0, use_amp=use_amp,
        decode_method=main_cfg.get("decode_method", "expectation"),
        a2_soft_qwk_weight=float(main_cfg.get("a2_soft_qwk_weight", 0.0)),
        a2_emd_weight=float(main_cfg.get("a2_emd_weight", 0.0)),
        a1_head=a1_head, a2_head=a2_head,
        a1_loss_weight=float(main_cfg.get("a1_loss_weight", 1.0)),
        a2_loss_weight=float(main_cfg.get("a2_loss_weight", 1.0)),
        consistency_loss_weight=float(main_cfg.get("consistency_loss_weight", 0.3)),
        consistency_temperature=float(main_cfg.get("consistency_temperature", 1.0)),
        joint_primary_metric=str(main_cfg.get("joint_primary_metric", "mean")),
        aux_film=aux_film,
        aux_predictor=None,
        aux_input_source="gt",
    )
    val_pred = validate_grouped(
        grouped_model, None, val_loader, device,
        task="joint", epoch=0, epochs=0, use_amp=use_amp,
        decode_method=main_cfg.get("decode_method", "expectation"),
        a2_soft_qwk_weight=float(main_cfg.get("a2_soft_qwk_weight", 0.0)),
        a2_emd_weight=float(main_cfg.get("a2_emd_weight", 0.0)),
        a1_head=a1_head, a2_head=a2_head,
        a1_loss_weight=float(main_cfg.get("a1_loss_weight", 1.0)),
        a2_loss_weight=float(main_cfg.get("a2_loss_weight", 1.0)),
        consistency_loss_weight=float(main_cfg.get("consistency_loss_weight", 0.3)),
        consistency_temperature=float(main_cfg.get("consistency_temperature", 1.0)),
        joint_primary_metric=str(main_cfg.get("joint_primary_metric", "mean")),
        aux_film=aux_film,
        aux_predictor=aux_predictor,
        aux_input_source="predictor",
    )

    e2e = {
        "gt":        {"f1": float(val_gt["mean_f1"]),   "qwk": float(val_gt["mean_qwk"]),
                      "primary": float(val_gt["primary_metric"])},
        "predictor": {"f1": float(val_pred["mean_f1"]), "qwk": float(val_pred["mean_qwk"]),
                      "primary": float(val_pred["primary_metric"])},
    }
    e2e["drop"] = {
        "f1": e2e["gt"]["f1"] - e2e["predictor"]["f1"],
        "qwk": e2e["gt"]["qwk"] - e2e["predictor"]["qwk"],
        "primary": e2e["gt"]["primary"] - e2e["predictor"]["primary"],
    }
    log.info(
        "End-to-end val comparison\n"
        f"  GT-input   : F1={e2e['gt']['f1']:.4f}  QWK={e2e['gt']['qwk']:.4f}  "
        f"primary={e2e['gt']['primary']:.4f}\n"
        f"  Predictor  : F1={e2e['predictor']['f1']:.4f}  QWK={e2e['predictor']['qwk']:.4f}  "
        f"primary={e2e['predictor']['primary']:.4f}\n"
        f"  Drop       : F1={e2e['drop']['f1']:+.4f}  QWK={e2e['drop']['qwk']:+.4f}  "
        f"primary={e2e['drop']['primary']:+.4f}"
    )

    new_state = dict(state)
    new_state["aux_predictor_state_dict"] = best_state
    new_state["aux_predictor_config"] = {
        "hidden": predictor_hidden,
        "dropout": predictor_dropout,
        "d_in": bb_cfg.d_shared,
    }
    new_state["aux_predictor_val_metrics"] = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "early_stop_metric": es_metric,
        "per_attr_acc": best_val_metrics["per_attr_acc"],
        "per_attr_macro_f1": best_val_metrics["per_attr_macro_f1"],
        "per_attr_seen": best_val_metrics["per_attr_seen"],
        "mean_macro_f1": best_val_metrics["mean_macro_f1"],
        "end_to_end": e2e,
        "main_ckpt": str(main_ckpt_path),
        "trained_at": datetime.now().isoformat(),
    }
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state, out_ckpt_path)
    log.info(f"Saved: {out_ckpt_path}")
    log.info(f"Total time: {_fmt_duration(time.time() - t_start)}")


if __name__ == "__main__":
    main()
