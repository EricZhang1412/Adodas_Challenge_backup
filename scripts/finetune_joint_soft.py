#!/usr/bin/env python3
"""Stage-3 fine-tune: teach AuxFiLM to accept soft predictor input.

Loads stage-2 best_with_predictor.pt (which already has main backbone + heads
+ AuxFiLM + AuxPredictor weights), freezes the backbone and the AuxPredictor,
then continues training AuxFiLM + A1Head + A2Head with a dual-path forward:

    loss = alpha * loss_gt_one_hot + (1 - alpha) * loss_predictor_softmax

where both paths share the same downstream heads / losses. After fine-tuning,
the GT path keeps its QWK while the predictor path (used at inference) catches
up dramatically because AuxFiLM no longer sees soft input as out-of-distribution.

Output ckpt mirrors stage-2 layout plus a `finetune_val_metrics` field. The
existing infer.py path (auto / predictor) consumes it without changes.

Usage:
    uv run scripts/finetune_joint_soft.py \\
        --from_ckpt output/.../checkpoints/best_with_predictor.pt \\
        [--config tasks/joint_finetune_soft/default.yaml] \\
        [--epochs 12] [--lr 1.6e-4] [--alpha 0.3]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
from common.models.heads import (  # noqa: E402
    A1Head,
    A2OrdinalHead,
    a1_loss,
    a2_combined_loss,
    dass21_consistency_loss,
)
from common.models.mtcn_backbone import MTCNBackbone  # noqa: E402
from common.runner import (  # noqa: E402
    EarlyStopping,
    _build_scheduler,
    _flatten_valid_session_mask,
    _fmt_duration,
    _to_device,
    setup_logging,
    validate_grouped,
    compute_a2_pos_weight,
    _compute_pos_weight_a1,
)
from common.utils.seed import seed_everything  # noqa: E402

from train_aux_predictor import (  # noqa: E402  (sibling script under scripts/)
    build_backbone,
    build_feat_cfg,
    freeze,
    load_yaml,
    resolve_manifests,
)

log = logging.getLogger("finetune_soft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--from_ckpt", type=str, required=True,
                   help="Stage-2 ckpt (best_with_predictor.pt)")
    p.add_argument("--config", type=str,
                   default="tasks/joint_finetune_soft/default.yaml",
                   help="Stage-3 fine-tune config")
    p.add_argument("--main_config", type=str, default=None,
                   help="Override path to stage-1 config_used.yaml")
    p.add_argument("--out_ckpt_name", type=str, default=None,
                   help="Override output ckpt filename (default from yaml)")

    p.add_argument("--alpha", type=float, default=None,
                   help="Weight on GT path; predictor path gets 1-alpha")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    return p.parse_args()


def merge_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    for k in ("alpha", "epochs", "lr", "batch_size", "seed", "num_workers", "patience"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.out_ckpt_name is not None:
        cfg["out_ckpt_name"] = args.out_ckpt_name
    return cfg


def compute_joint_loss(
    fused_p: torch.Tensor,
    fused_s: torch.Tensor,
    targets_a1: torch.Tensor,
    targets_a2: torch.Tensor,
    a1_head: nn.Module,
    a2_head: nn.Module,
    valid_session_mask: torch.Tensor,
    has_valid_sessions: bool,
    *,
    pos_weight_a1: torch.Tensor | None,
    pos_weight_a2: torch.Tensor | None,
    label_smoothing: float,
    a2_soft_qwk_weight: float,
    a2_emd_weight: float,
    a2_expectation_loss_weight: float,
    a2_expectation_distance_aware: bool,
    a2_expectation_distance_power: float,
    a1_loss_weight: float,
    a2_loss_weight: float,
    consistency_loss_weight: float,
    consistency_temperature: float,
    session_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reproduces the joint loss block from runner.train_one_epoch_grouped.

    session_type_loss is intentionally omitted: backbone (and its
    session_type_head) is frozen during stage-3, so that term has no gradient.
    """
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
        a1_p_logits, a2_p_logits, temperature=consistency_temperature,
    )

    if has_valid_sessions:
        a1_s_logits = a1_head(fused_s)[valid_session_mask]
        a2_s_logits = a2_head(fused_s)[valid_session_mask]
        a1_s_targets = (
            targets_a1.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)[valid_session_mask]
        )
        a2_s_targets = (
            targets_a2.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 21)[valid_session_mask]
        )
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
    else:
        a1_sess_loss = a1_p_logits.new_zeros(())
        a2_sess_loss = a1_p_logits.new_zeros(())

    total = (
        a1_loss_weight * main_a1
        + a2_loss_weight * main_a2
        + consistency_loss_weight * cons
        + session_loss_weight * (a1_sess_loss + a2_sess_loss)
    )
    parts = {
        "a1": float(main_a1.detach().item()),
        "a2": float(main_a2.detach().item()),
        "cons": float(cons.detach().item()),
        "sess": float((a1_sess_loss + a2_sess_loss).detach().item()),
    }
    return total, parts


def main() -> None:
    args = parse_args()
    ft_cfg = merge_overrides(load_yaml(Path(args.config)), args)

    main_ckpt_path = Path(args.from_ckpt).resolve()
    if not main_ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {main_ckpt_path}")
    run_dir = main_ckpt_path.parent.parent
    ckpt_dir = main_ckpt_path.parent

    main_cfg_path = Path(args.main_config) if args.main_config else run_dir / "config_used.yaml"
    main_cfg = load_yaml(main_cfg_path)
    if not bool(main_cfg.get("use_aux_film", False)):
        raise RuntimeError("stage-1 cfg has use_aux_film=false; stage-3 requires AuxFiLM")
    if str(main_cfg.get("task", "joint")) != "joint":
        raise RuntimeError("stage-3 only supported for joint task")

    out_name = ft_cfg.get("out_ckpt_name", "best_finetuned_soft.pt")
    out_ckpt_path = ckpt_dir / out_name
    log_dir = run_dir / "logs"
    setup_logging(log_dir, "finetune_soft")
    log.info(f"stage-2 ckpt: {main_ckpt_path}")
    log.info(f"main cfg    : {main_cfg_path}")
    log.info(f"output      : {out_ckpt_path}")

    seed_everything(int(ft_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, train_manifest, val_manifest = resolve_manifests(main_cfg, main_ckpt_path)
    log.info(f"train manifest: {train_manifest}")
    log.info(f"val manifest  : {val_manifest}")

    feat_cfg = build_feat_cfg(main_cfg)
    # Disable session_drop_prob during fine-tune: backbone is frozen so the
    # augmentation doesn't add signal, and we want stable participant_reprs
    # across epochs.
    train_ds = GroupedParticipantDataset(
        train_manifest, feat_cfg, split="train", session_drop_prob=0.0,
    )
    val_ds = GroupedParticipantDataset(val_manifest, feat_cfg, split="val")
    log.info(f"train: {len(train_ds)} participants, val: {len(val_ds)} participants")

    batch_size = int(ft_cfg.get("batch_size", 32))
    num_workers = int(ft_cfg.get("num_workers", 8))
    if bool(ft_cfg.get("preload", True)):
        log.info("preloading features ...")
        t_pre = time.time()
        train_ds.preload(desc="Preload train")
        val_ds.preload(desc="Preload val")
        log.info(f"preload took {_fmt_duration(time.time() - t_pre)}")
        num_workers = 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=grouped_collate_fn, pin_memory=True,
        drop_last=True, persistent_workers=num_workers > 0,
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
        raise RuntimeError("stage-2 ckpt head_state_dict missing 'a1' / 'a2'")
    a1_head.load_state_dict(head_sd["a1"])
    a2_head.load_state_dict(head_sd["a2"])
    if "aux_film_state_dict" not in state:
        raise RuntimeError("stage-2 ckpt missing aux_film_state_dict")
    aux_film.load_state_dict(state["aux_film_state_dict"])

    if "aux_predictor_state_dict" not in state:
        raise RuntimeError(
            "stage-2 ckpt missing aux_predictor_state_dict — "
            "run scripts/train_aux_predictor.py first."
        )
    pred_cfg = state.get("aux_predictor_config") or {}
    aux_predictor = AuxPredictor(
        d_in=int(pred_cfg.get("d_in", bb_cfg.d_shared)),
        hidden=int(pred_cfg.get("hidden", 128)),
        dropout=float(pred_cfg.get("dropout", 0.2)),
    ).to(device)
    aux_predictor.load_state_dict(state["aux_predictor_state_dict"])

    freeze(grouped_model)
    freeze(aux_predictor)
    a1_head.train()
    a2_head.train()
    aux_film.train()

    trainable_params = (
        list(aux_film.parameters())
        + list(a1_head.parameters())
        + list(a2_head.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    log.info(f"Trainable params: {n_trainable:,} "
             f"(aux_film + a1_head + a2_head; backbone & predictor frozen)")

    use_amp = bool(ft_cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    pos_weight_a1_t = None
    pos_weight_a2_t = None
    if bool(main_cfg.get("use_pos_weight", True)):
        pw = _compute_pos_weight_a1(train_manifest)
        pos_weight_a1_t = torch.tensor(pw, dtype=torch.float32, device=device)
        pos_weight_a2_t = compute_a2_pos_weight(train_manifest).to(device)
        log.info(f"pos_weight A1: {pw}")
        log.info(f"pos_weight A2 shape: {pos_weight_a2_t.shape}")

    epochs = int(ft_cfg.get("epochs", 12))
    warmup_epochs = int(ft_cfg.get("warmup_epochs", 1))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(ft_cfg.get("lr", 1.6e-4)),
        weight_decay=float(ft_cfg.get("weight_decay", 0.05)),
    )
    scheduler = _build_scheduler(optimizer, warmup_epochs, epochs)
    grad_clip = float(ft_cfg.get("grad_clip", 1.0))
    alpha = float(ft_cfg.get("alpha", 0.3))
    log.info(f"alpha={alpha}  lr={float(ft_cfg.get('lr', 1.6e-4))}  "
             f"epochs={epochs}  warmup={warmup_epochs}")

    label_smoothing = float(main_cfg.get("label_smoothing", 0.0))
    a2_soft_qwk_weight = float(main_cfg.get("a2_soft_qwk_weight", 0.0))
    a2_emd_weight = float(main_cfg.get("a2_emd_weight", 0.0))
    a2_expectation_loss_weight = float(main_cfg.get("a2_expectation_loss_weight", 0.0))
    a2_expectation_distance_aware = bool(main_cfg.get("a2_expectation_distance_aware", False))
    a2_expectation_distance_power = float(main_cfg.get("a2_expectation_distance_power", 1.0))
    a1_loss_weight = float(main_cfg.get("a1_loss_weight", 1.0))
    a2_loss_weight = float(main_cfg.get("a2_loss_weight", 1.0))
    consistency_loss_weight = float(main_cfg.get("consistency_loss_weight", 0.3))
    consistency_temperature = float(main_cfg.get("consistency_temperature", 1.0))
    session_loss_weight = float(main_cfg.get("session_loss_weight", 0.4))
    joint_primary_metric = str(main_cfg.get("joint_primary_metric", "mean"))
    decode_method = str(main_cfg.get("decode_method", "expectation"))

    loss_kw = dict(
        pos_weight_a1=pos_weight_a1_t, pos_weight_a2=pos_weight_a2_t,
        label_smoothing=label_smoothing,
        a2_soft_qwk_weight=a2_soft_qwk_weight,
        a2_emd_weight=a2_emd_weight,
        a2_expectation_loss_weight=a2_expectation_loss_weight,
        a2_expectation_distance_aware=a2_expectation_distance_aware,
        a2_expectation_distance_power=a2_expectation_distance_power,
        a1_loss_weight=a1_loss_weight,
        a2_loss_weight=a2_loss_weight,
        consistency_loss_weight=consistency_loss_weight,
        consistency_temperature=consistency_temperature,
        session_loss_weight=session_loss_weight,
    )
    validate_kw = dict(
        decode_method=decode_method,
        a2_soft_qwk_weight=a2_soft_qwk_weight,
        a2_emd_weight=a2_emd_weight,
        a1_head=a1_head, a2_head=a2_head,
        pos_weight_a1=pos_weight_a1_t, pos_weight_a2=pos_weight_a2_t,
        a1_loss_weight=a1_loss_weight,
        a2_loss_weight=a2_loss_weight,
        consistency_loss_weight=consistency_loss_weight,
        consistency_temperature=consistency_temperature,
        joint_primary_metric=joint_primary_metric,
        aux_film=aux_film,
    )

    early_stop = EarlyStopping(patience=int(ft_cfg.get("patience", 5)), mode="max")

    best_score: float | None = None
    best_epoch = -1
    best_states: dict | None = None
    best_metrics: dict = {}
    t_start = time.time()

    log.info("=" * 100)
    log.info(" Epoch |    LR    | Train(α/1-α) | val(gt) F1/QWK/pri | val(pred) F1/QWK/pri | Time")
    log.info("=" * 100)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        a1_head.train()
        a2_head.train()
        aux_film.train()
        run_loss = 0.0
        run_loss_gt = 0.0
        run_loss_pr = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Train {epoch}/{epochs}",
                          leave=False, dynamic_ncols=True):
            flat_batch = _to_device(batch["flat_batch"], device)
            session_valid = batch["session_valid"].to(device)
            B = batch["n_participants"]
            targets_a1 = batch["participant_y_a1"].to(device)
            targets_a2 = batch["participant_y_a2"].to(device).long()
            aux_labels = batch["participant_aux_labels"].to(device)
            aux_masks = batch["participant_aux_masks"].to(device)
            valid_session_mask = _flatten_valid_session_mask(session_valid)
            has_valid_sessions = bool(valid_session_mask.any().item())

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                with torch.no_grad():
                    out = grouped_model(flat_batch, B, session_valid)
                    probs = aux_predictor.predict_probs(out["participant_repr"])
                # Detach extra-safe (predict_probs already inside no_grad).
                probs = {n: p.detach() for n, p in probs.items()}
                participant_repr = out["participant_repr"].detach()
                session_reprs = out["session_reprs"].detach()

                # GT one-hot path
                aux_in = aux_film.fill_missing(aux_labels, aux_masks)
                aux_in_s = aux_in.unsqueeze(1).expand(-1, 4, -1).reshape(B * 4, -1)
                fused_p_gt = aux_film(participant_repr, aux_in)
                fused_s_gt = aux_film(session_reprs, aux_in_s)
                loss_gt, _parts_gt = compute_joint_loss(
                    fused_p_gt, fused_s_gt, targets_a1, targets_a2,
                    a1_head, a2_head, valid_session_mask, has_valid_sessions,
                    **loss_kw,
                )

                # Predictor softmax path
                probs_s = {
                    n: p.unsqueeze(1).expand(-1, 4, -1).reshape(B * 4, -1)
                    for n, p in probs.items()
                }
                fused_p_pr = aux_film.forward_soft(participant_repr, probs)
                fused_s_pr = aux_film.forward_soft(session_reprs, probs_s)
                loss_pr, _parts_pr = compute_joint_loss(
                    fused_p_pr, fused_s_pr, targets_a1, targets_a2,
                    a1_head, a2_head, valid_session_mask, has_valid_sessions,
                    **loss_kw,
                )

                loss = alpha * loss_gt + (1.0 - alpha) * loss_pr

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                optimizer.step()

            run_loss += float(loss.item())
            run_loss_gt += float(loss_gt.detach().item())
            run_loss_pr += float(loss_pr.detach().item())
            n_batches += 1
        scheduler.step()

        avg_loss = run_loss / max(n_batches, 1)
        avg_loss_gt = run_loss_gt / max(n_batches, 1)
        avg_loss_pr = run_loss_pr / max(n_batches, 1)

        vm_gt = validate_grouped(
            grouped_model, None, val_loader, device,
            task="joint", epoch=epoch, epochs=epochs, use_amp=use_amp,
            aux_predictor=None, aux_input_source="gt", **validate_kw,
        )
        vm_pr = validate_grouped(
            grouped_model, None, val_loader, device,
            task="joint", epoch=epoch, epochs=epochs, use_amp=use_amp,
            aux_predictor=aux_predictor, aux_input_source="predictor", **validate_kw,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        score = float(vm_pr["primary_metric"])
        is_best = (best_score is None) or (score > best_score)
        marker = " *" if is_best else ""
        log.info(
            f"  {epoch:3d}/{epochs:3d} | {lr_now:.2e} | "
            f"{avg_loss_gt:.3f}/{avg_loss_pr:.3f} | "
            f"{vm_gt['mean_f1']:.3f}/{vm_gt['mean_qwk']:.3f}/{vm_gt['primary_metric']:.3f} | "
            f"{vm_pr['mean_f1']:.3f}/{vm_pr['mean_qwk']:.3f}/{vm_pr['primary_metric']:.3f} | "
            f"{_fmt_duration(elapsed)}{marker}"
        )

        if is_best:
            best_score = score
            best_epoch = epoch
            best_states = {
                "a1": {k: v.detach().cpu().clone() for k, v in a1_head.state_dict().items()},
                "a2": {k: v.detach().cpu().clone() for k, v in a2_head.state_dict().items()},
                "aux_film": {k: v.detach().cpu().clone() for k, v in aux_film.state_dict().items()},
            }
            best_metrics = {
                "gt": {
                    "mean_f1": float(vm_gt["mean_f1"]),
                    "mean_qwk": float(vm_gt["mean_qwk"]),
                    "primary": float(vm_gt["primary_metric"]),
                },
                "predictor": {
                    "mean_f1": float(vm_pr["mean_f1"]),
                    "mean_qwk": float(vm_pr["mean_qwk"]),
                    "primary": float(vm_pr["primary_metric"]),
                },
                "alpha": alpha,
                "best_epoch": epoch,
                "early_stop_metric": f"predictor.{joint_primary_metric}",
            }

        if early_stop.step(score):
            log.info(f"  EarlyStopping at epoch {epoch}")
            break

    if best_states is None:
        raise RuntimeError("no best ckpt captured — training did not progress")

    log.info(f"best epoch: {best_epoch}  best predictor primary: {best_score:.4f}")
    log.info(
        "End-to-end val (best epoch)\n"
        f"  GT-input   : F1={best_metrics['gt']['mean_f1']:.4f}  "
        f"QWK={best_metrics['gt']['mean_qwk']:.4f}  primary={best_metrics['gt']['primary']:.4f}\n"
        f"  Predictor  : F1={best_metrics['predictor']['mean_f1']:.4f}  "
        f"QWK={best_metrics['predictor']['mean_qwk']:.4f}  "
        f"primary={best_metrics['predictor']['primary']:.4f}\n"
        f"  Drop       : F1={best_metrics['gt']['mean_f1'] - best_metrics['predictor']['mean_f1']:+.4f}  "
        f"QWK={best_metrics['gt']['mean_qwk'] - best_metrics['predictor']['mean_qwk']:+.4f}"
    )

    new_state = dict(state)
    new_state["head_state_dict"] = {"a1": best_states["a1"], "a2": best_states["a2"]}
    new_state["aux_film_state_dict"] = best_states["aux_film"]
    # Keep aux_predictor_state_dict / aux_predictor_config from stage-2 unchanged.
    new_state["finetune_val_metrics"] = {
        **best_metrics,
        "trained_at": datetime.now().isoformat(),
        "from_ckpt": str(main_ckpt_path),
    }
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state, out_ckpt_path)
    log.info(f"Saved: {out_ckpt_path}")
    log.info(f"Total time: {_fmt_duration(time.time() - t_start)}")


if __name__ == "__main__":
    main()
