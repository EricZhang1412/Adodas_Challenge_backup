#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

import numpy as np
from tqdm import tqdm

from common.data.dataset import FeatureConfig
from common.data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from common.models.grouped_model import CORALHead, GroupedModel
from common.models.heads import A1Head, A2OrdinalHead
from common.models.mtcn_backbone import BackboneConfig, MTCNBackbone
from common.runner import (
    _decode_a2_logits,
    _normalize_decode_method,
    _to_device,
    generate_submission_grouped,
    setup_logging,
)
from common.utils.ckpt import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["a1", "a2"])
    parser.add_argument(
        "--checkpoint", default=None,
        help="Single checkpoint path or directory containing best.pt. "
             "Mutually exclusive with --checkpoints.",
    )
    parser.add_argument(
        "--checkpoints", default=None,
        help="Comma-separated list of checkpoint paths/dirs for K-fold ensemble "
             "inference. Raw logits are averaged across folds before decoding. "
             "Calibration (decode method + A2 threshold offsets) is taken from "
             "the FIRST checkpoint's run_dir.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test_hidden")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.checkpoints is None):
        parser.error("Pass exactly one of --checkpoint or --checkpoints.")
    return args


def load_config(config_path: str | None, checkpoint_path: Path) -> dict:
    if config_path is None:
        candidate = checkpoint_path.parent.parent / "config_used.yaml"
        config_path = str(candidate)
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    feature_selection = cfg.pop("feature_selection", {}) or {}
    if not isinstance(feature_selection, dict):
        raise TypeError("feature_selection must be a mapping in the config YAML")
    cfg.update(feature_selection)
    return cfg


def load_calibration(run_dir: Path, task: str) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    calibration_dir = run_dir / "calibration"
    if task == "a1":
        path = calibration_dir / "a1_bias_grouped.json"
        if not path.exists():
            return None, None, "expectation"
        with open(path) as f:
            data = json.load(f)
        biases = torch.tensor(data.get("biases", []), dtype=torch.float32) if data.get("biases") else None
        return biases, None, "expectation"

    path = calibration_dir / "a2_threshold_offsets_grouped.json"
    if not path.exists():
        return None, None, _normalize_decode_method("expectation")
    with open(path) as f:
        data = json.load(f)
    selected_method = _normalize_decode_method(data.get("selected_decode_method", "expectation"))
    strategies = data.get("strategies", {})
    selected_strategy = data.get("selected_strategy", "")
    offsets = None
    if selected_strategy in strategies and "offsets" in strategies[selected_strategy]:
        offsets = torch.tensor(strategies[selected_strategy]["offsets"], dtype=torch.float32)
    return None, offsets, selected_method


def resolve_checkpoint_path(checkpoint_arg: str) -> Path:
    p = Path(checkpoint_arg).resolve()
    if p.is_dir():
        # Prefer best.pt, then fall back to the most recently modified .pt file
        best = p / "best.pt"
        if best.exists():
            return best
        candidates = sorted(p.glob("*.pt"), key=lambda x: x.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No .pt checkpoint files found in directory: {p}")
        return candidates[-1]
    return p


@torch.no_grad()
def _gather_inference_logits(
    grouped_model: GroupedModel,
    task_head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    submission_level: str,
    desc: str,
) -> tuple[list[str], list[str], torch.Tensor]:
    """Forward pass returning (pids, sessions, raw_logits) without decoding.

    Used for K-fold ensemble: we average logits across models *before* decoding
    so the ordinal threshold calibration is applied on the smoothed signal.
    Mirrors generate_submission_grouped's collection logic but stops short of
    the decode step.
    """
    grouped_model.eval()
    task_head.eval()

    all_pids: list[str] = []
    all_sessions: list[str] = []
    all_logits: list[torch.Tensor] = []

    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            if submission_level == "participant":
                logits = task_head(out["participant_repr"])
            else:
                logits = task_head(out["session_reprs"])

        all_logits.append(logits.float().cpu())

        if submission_level == "participant":
            participant_ids = [str(pid) for pid in batch["anon_pids"]]
            all_pids.extend(participant_ids)
            all_sessions.extend(["participant"] * len(participant_ids))
        else:
            all_pids.extend(batch["flat_pids"])
            all_sessions.extend(batch["flat_sessions"])

    return all_pids, all_sessions, torch.cat(all_logits, dim=0)


def main() -> None:
    args = parse_args()

    # Resolve one or more checkpoint paths. The FIRST checkpoint owns the
    # config, calibration, and log destination — its run_dir is the canonical
    # one for this submission. All checkpoints must be config-compatible
    # (same architecture); we don't validate this beyond letting torch.load
    # complain on shape mismatch.
    if args.checkpoint is not None:
        checkpoint_paths = [resolve_checkpoint_path(args.checkpoint)]
    else:
        checkpoint_paths = [
            resolve_checkpoint_path(p.strip())
            for p in args.checkpoints.split(",")
            if p.strip()
        ]
    if not checkpoint_paths:
        raise ValueError("No checkpoints to evaluate.")

    primary_checkpoint = checkpoint_paths[0]
    cfg = load_config(args.config, primary_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = primary_checkpoint.parent.parent
    setup_logging(run_dir / "logs", f"infer_{args.task}")
    if len(checkpoint_paths) > 1:
        print(f"Ensemble mode: averaging logits across {len(checkpoint_paths)} checkpoints")
        for i, p in enumerate(checkpoint_paths):
            print(f"  [{i}] {p}")
        print(f"Calibration source: {primary_checkpoint.parent.parent}")

    manifest_dir = Path(cfg.get("manifest_dir", "/media/k3nwong/Data1/test/outputs/data"))
    manifest_path = Path(args.manifest) if args.manifest else manifest_dir / f"{args.split}.csv"
    if not manifest_path.exists():
        # 尝试从特征目录自动扫描生成 manifest
        feature_root = Path(cfg.get("feature_root", FeatureConfig.feature_root))
        split_dir = feature_root / args.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Also tried auto-scanning feature directory but it does not exist: {split_dir}\n"
                f"Please provide --manifest or ensure features are extracted to {split_dir}"
            )
        print(f"[infer] Manifest not found, auto-scanning feature directory: {split_dir}")
        _known_sessions = {"A01", "B01", "B02", "B03"}
        seen: set[tuple[str, str, str, str]] = set()
        for seq_path in split_dir.rglob("sequence.npz"):
            rel_parts = seq_path.relative_to(split_dir).parts
            if len(rel_parts) < 7:
                continue
            session = rel_parts[-2]
            if session not in _known_sessions:
                continue
            school, cls, pid = rel_parts[0], rel_parts[1], rel_parts[2]
            seen.add((school, cls, pid, session))
        if not seen:
            raise FileNotFoundError(
                f"No sequence.npz files found under {split_dir}. "
                f"Please extract test features there or provide --manifest."
            )
        auto_df = pd.DataFrame(
            sorted(seen),
            columns=["anon_school", "anon_class", "anon_pid", "session"],
        )
        auto_df.insert(0, "split", args.split)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        auto_df.to_csv(manifest_path, index=False)
        n_pid = auto_df["anon_pid"].nunique()
        print(f"[infer] Auto-generated manifest: {n_pid} participants, {len(auto_df)} sessions -> {manifest_path}")

    defaults = FeatureConfig()
    feat_cfg = FeatureConfig(
        feature_root=cfg.get("feature_root", defaults.feature_root),
        audio_features=cfg.get("audio_features", defaults.audio_features),
        video_features=cfg.get("video_features", defaults.video_features),
        audio_ssl_model_tag=cfg.get("audio_ssl_model_tag", defaults.audio_ssl_model_tag),
        video_ssl_model_tag=cfg.get("video_ssl_model_tag", defaults.video_ssl_model_tag),
        mask_policy=cfg.get("mask_policy", defaults.mask_policy),
        core_audio=cfg.get("core_audio", defaults.core_audio),
        core_video=cfg.get("core_video", defaults.core_video),
    )

    ds = GroupedParticipantDataset(manifest_path, feat_cfg, split=args.split)
    preload = bool(cfg.get("preload", True))
    num_workers = int(cfg.get("num_workers", 8))
    if preload:
        ds.preload(desc=f"Preload {args.split}")
        num_workers = 0

    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=grouped_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    dims = ds.feature_dims
    bb_cfg = BackboneConfig(
        audio_group_dims={n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims},
        audio_pooled_group_dims={n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims},
        video_group_dims={n: dims[n] for n in feat_cfg.video_features if n in dims},
        d_adapter=cfg.get("d_adapter", 64),
        d_model=cfg.get("d_model", 256),
        tcn_layers=cfg.get("tcn_layers", 6),
        tcn_kernel_size=cfg.get("tcn_kernel_size", 3),
        asp_alpha=cfg.get("asp_alpha", 0.5),
        asp_beta=cfg.get("asp_beta", 0.5),
        dropout=cfg.get("dropout", 0.2),
        d_shared=cfg.get("d_shared", 256),
        use_cross_modal_attn=cfg.get("use_cross_modal_attn", False),
        cross_modal_heads=cfg.get("cross_modal_heads", 4),
    )
    grouped_model = GroupedModel(
        backbone=MTCNBackbone(bb_cfg),
        d_shared=bb_cfg.d_shared,
        aggregator_method=cfg.get("aggregator", "mlp"),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    if args.task == "a1":
        task_head = A1Head(bb_cfg.d_shared).to(device)
    else:
        if bool(cfg.get("use_coral", False)):
            task_head = CORALHead(bb_cfg.d_shared).to(device)
        else:
            task_head = A2OrdinalHead(bb_cfg.d_shared).to(device)

    a1_biases, a2_offsets, selected_decode_method = load_calibration(run_dir, args.task)
    use_amp = bool(cfg.get("amp", True))
    submission_level = cfg.get("submission_level", "participant")

    if len(checkpoint_paths) == 1:
        # Single-model path — unchanged from before, decode happens inside
        # generate_submission_grouped.
        state = load_checkpoint(checkpoint_paths[0], grouped_model, optimizer=None)
        task_head.load_state_dict(state["head_state_dict"])

        pids, sessions, preds = generate_submission_grouped(
            grouped_model=grouped_model,
            task_head=task_head,
            loader=loader,
            device=device,
            task=args.task,
            use_amp=use_amp,
            desc=f"Infer {args.split}",
            submission_level=submission_level,
            a1_biases=None if a1_biases is None else a1_biases.to(device),
            decode_method=selected_decode_method,
            a2_threshold_offsets=None if a2_offsets is None else a2_offsets.to(device),
        )
    else:
        # Ensemble path — gather raw logits from each checkpoint, average,
        # then decode once. ID order is determined by the loader (deterministic
        # since shuffle=False), so per-fold logits align row-by-row.
        logits_per_fold: list[torch.Tensor] = []
        pids: list[str] = []
        sessions: list[str] = []
        for k, ckpt_path in enumerate(checkpoint_paths):
            state = load_checkpoint(ckpt_path, grouped_model, optimizer=None)
            task_head.load_state_dict(state["head_state_dict"])
            fold_pids, fold_sessions, fold_logits = _gather_inference_logits(
                grouped_model=grouped_model,
                task_head=task_head,
                loader=loader,
                device=device,
                use_amp=use_amp,
                submission_level=submission_level,
                desc=f"Infer fold {k}/{len(checkpoint_paths)}",
            )
            if not pids:
                pids, sessions = fold_pids, fold_sessions
            else:
                # Cross-check that row order is consistent across folds. The
                # loader is deterministic (shuffle=False), so this should hold;
                # fail loudly if it doesn't (e.g., dataset filtering changed).
                if fold_pids != pids:
                    raise RuntimeError(
                        f"Checkpoint {ckpt_path} produced a different participant "
                        f"ordering than the first checkpoint — refusing to ensemble."
                    )
            logits_per_fold.append(fold_logits)

        avg_logits = torch.stack(logits_per_fold, dim=0).mean(dim=0).to(device)

        if args.task == "a1":
            if a1_biases is not None:
                avg_logits = avg_logits + a1_biases.to(device)
            preds = torch.sigmoid(avg_logits).cpu().numpy()
        else:
            if a2_offsets is not None:
                avg_logits = avg_logits + a2_offsets.to(device)
            preds_t = _decode_a2_logits(
                task_head, avg_logits, decode_method=selected_decode_method
            )
            preds = preds_t.cpu().numpy()

    manifest_df = pd.read_csv(manifest_path)
    out_schools = []
    out_classes = []
    out_pids = []
    out_sessions = []
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
            out_schools.append(school)
            out_classes.append(cls)
            out_pids.append(pid_str)
            filtered_preds.append(pred)
    else:
        pid_to_info = {
            (str(row["anon_pid"]), str(row["session"])): (
                str(row["anon_school"]),
                str(row["anon_class"]),
            )
            for _, row in manifest_df.iterrows()
        }

        for pid, sess, pred in zip(pids, sessions, preds):
            key = (str(pid), str(sess))
            info = pid_to_info.get(key)
            if info is None:
                continue
            school, cls = info
            out_schools.append(school)
            out_classes.append(cls)
            out_pids.append(key[0])
            out_sessions.append(key[1])
            filtered_preds.append(pred)

    if args.task == "a1":
        sub = pd.DataFrame(
            {
                "anon_school": out_schools,
                "anon_class": out_classes,
                "anon_pid": out_pids,
                "p_D": [float(pred[0]) for pred in filtered_preds],
                "p_A": [float(pred[1]) for pred in filtered_preds],
                "p_S": [float(pred[2]) for pred in filtered_preds],
            }
        )
    else:
        sub = pd.DataFrame({
            "anon_school": out_schools,
            "anon_class": out_classes,
            "anon_pid": out_pids,
        })
        if submission_level == "session":
            sub["session"] = out_sessions
        for idx, col in enumerate([f"d{i:02d}" for i in range(1, 22)]):
            sub[col] = [int(pred[idx]) for pred in filtered_preds]

    output_path = Path(args.output) if args.output else run_dir / "submissions" / f"submission_{args.task}_{args.split}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
