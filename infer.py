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
from common.models.grouped_model import AuxFiLM, CORALHead, GroupedModel
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
    parser.add_argument("--task", required=True, choices=["a1", "a2", "joint"])
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


def _load_a1_biases(calibration_dir: Path) -> torch.Tensor | None:
    path = calibration_dir / "a1_bias_grouped.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    biases = data.get("biases")
    if not biases:
        return None
    return torch.tensor(biases, dtype=torch.float32)


def _load_a2_offsets_and_decode(calibration_dir: Path) -> tuple[torch.Tensor | None, str]:
    path = calibration_dir / "a2_threshold_offsets_grouped.json"
    if not path.exists():
        return None, _normalize_decode_method("expectation")
    with open(path) as f:
        data = json.load(f)
    selected_method = _normalize_decode_method(data.get("selected_decode_method", "expectation"))
    strategies = data.get("strategies", {})
    selected_strategy = data.get("selected_strategy", "")
    offsets = None
    if selected_strategy in strategies and "offsets" in strategies[selected_strategy]:
        offsets = torch.tensor(strategies[selected_strategy]["offsets"], dtype=torch.float32)
    return offsets, selected_method


def load_calibration(run_dir: Path, task: str) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    calibration_dir = run_dir / "calibration"
    if task == "a1":
        return _load_a1_biases(calibration_dir), None, "expectation"
    if task == "a2":
        offsets, method = _load_a2_offsets_and_decode(calibration_dir)
        return None, offsets, method
    # joint
    offsets, method = _load_a2_offsets_and_decode(calibration_dir)
    return _load_a1_biases(calibration_dir), offsets, method


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


@torch.no_grad()
def _gather_inference_logits_joint(
    grouped_model: GroupedModel,
    a1_head: torch.nn.Module,
    a2_head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    submission_level: str,
    desc: str,
    aux_film: AuxFiLM | None = None,
) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor]:
    """Joint ensemble helper: one backbone forward, both heads tapped.

    When ``aux_film`` is provided, the repr fed to a1/a2 heads is FiLM-modulated
    by the GT aux indices from the manifest (MISSING slot for masked attrs).
    """
    grouped_model.eval()
    a1_head.eval()
    a2_head.eval()
    if aux_film is not None:
        aux_film.eval()

    all_pids: list[str] = []
    all_sessions: list[str] = []
    a1_chunks: list[torch.Tensor] = []
    a2_chunks: list[torch.Tensor] = []

    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            if aux_film is not None:
                aux_labels_b = batch["participant_aux_labels"].to(device)
                aux_masks_b = batch["participant_aux_masks"].to(device)
                aux_in = aux_film.fill_missing(aux_labels_b, aux_masks_b)
                if submission_level == "participant":
                    repr_for_head = aux_film(out["participant_repr"], aux_in)
                else:
                    aux_in_s = aux_in.unsqueeze(1).expand(-1, 4, -1).reshape(B * 4, -1)
                    repr_for_head = aux_film(out["session_reprs"], aux_in_s)
            else:
                repr_key = "participant_repr" if submission_level == "participant" else "session_reprs"
                repr_for_head = out[repr_key]
            a1_logits = a1_head(repr_for_head)
            a2_logits = a2_head(repr_for_head)

        a1_chunks.append(a1_logits.float().cpu())
        a2_chunks.append(a2_logits.float().cpu())

        if submission_level == "participant":
            participant_ids = [str(pid) for pid in batch["anon_pids"]]
            all_pids.extend(participant_ids)
            all_sessions.extend(["participant"] * len(participant_ids))
        else:
            all_pids.extend(batch["flat_pids"])
            all_sessions.extend(batch["flat_sessions"])

    return all_pids, all_sessions, torch.cat(a1_chunks, dim=0), torch.cat(a2_chunks, dim=0)


def _write_submission_csv(
    head_task: str,
    pids: list[str],
    sessions: list[str],
    preds: np.ndarray,
    manifest_df: pd.DataFrame,
    submission_level: str,
    output_dir: Path,
    split: str,
    split_explicit_output: Path | None = None,
) -> Path:
    """Build the per-task CSV with the existing column schema and write it.

    Mirrors the post-train submission writer in runner.py so single-task and
    joint inference produce identical layouts.
    """
    out_schools: list[str] = []
    out_classes: list[str] = []
    out_pids: list[str] = []
    out_sessions: list[str] = []
    filtered: list = []
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
            filtered.append(pred)
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
            filtered.append(pred)

    if head_task == "a1":
        sub = pd.DataFrame({
            "anon_school": out_schools,
            "anon_class": out_classes,
            "anon_pid": out_pids,
            "p_D": [float(p[0]) for p in filtered],
            "p_A": [float(p[1]) for p in filtered],
            "p_S": [float(p[2]) for p in filtered],
        })
    else:
        sub = pd.DataFrame({
            "anon_school": out_schools,
            "anon_class": out_classes,
            "anon_pid": out_pids,
        })
        if submission_level == "session":
            sub["session"] = out_sessions
        for idx, col in enumerate([f"d{i:02d}" for i in range(1, 22)]):
            sub[col] = [int(p[idx]) for p in filtered]

    if split_explicit_output is not None:
        output_path = split_explicit_output
    else:
        output_path = output_dir / f"submission_{head_task}_{split}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    return output_path


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

    use_coral = bool(cfg.get("use_coral", False))
    task_head = None
    a1_head = None
    a2_head = None
    if args.task == "a1":
        task_head = A1Head(bb_cfg.d_shared).to(device)
    elif args.task == "a2":
        task_head = (CORALHead if use_coral else A2OrdinalHead)(bb_cfg.d_shared).to(device)
    else:  # joint
        a1_head = A1Head(bb_cfg.d_shared).to(device)
        a2_head = (CORALHead if use_coral else A2OrdinalHead)(bb_cfg.d_shared).to(device)

    # AuxFiLM (joint-only). Instantiated when configured in the YAML; the
    # checkpoint must contain matching aux_film_state_dict. GT aux indices
    # from the manifest drive FiLM modulation at inference time, with the
    # MISSING slot covering any masked attributes.
    aux_film: AuxFiLM | None = None
    if args.task == "joint" and bool(cfg.get("use_aux_film", False)):
        aux_film = AuxFiLM(
            d_shared=bb_cfg.d_shared,
            d_embed=int(cfg.get("aux_film_d_embed", 16)),
            hidden=int(cfg.get("aux_film_hidden", 128)),
            dropout=float(cfg.get("aux_film_dropout", 0.2)),
        ).to(device)

    a1_biases, a2_offsets, selected_decode_method = load_calibration(run_dir, args.task)
    use_amp = bool(cfg.get("amp", True))
    submission_level = cfg.get("submission_level", "participant")
    manifest_df = pd.read_csv(manifest_path)
    submissions_dir = run_dir / "submissions"

    def _load_head_state(state: dict, target_task: str) -> None:
        """Resolve the head state_dict from a checkpoint and load it.

        Joint checkpoints store a {"a1": ..., "a2": ...} dict; single-task
        checkpoints store the flat state_dict directly. A joint ckpt can be
        used for single-task inference by pulling out the matching sub-head;
        a flat single-task ckpt cannot satisfy --task joint and errors out.
        """
        head_sd = state["head_state_dict"]
        joint_layout = isinstance(head_sd, dict) and {"a1", "a2"} <= set(head_sd.keys())
        if target_task == "joint":
            if not joint_layout:
                raise RuntimeError(
                    "--task joint requires a joint checkpoint (head_state_dict with 'a1'/'a2' subkeys); "
                    "got a flat single-task state_dict."
                )
            a1_head.load_state_dict(head_sd["a1"])
            a2_head.load_state_dict(head_sd["a2"])
        else:
            sub_sd = head_sd[target_task] if joint_layout else head_sd
            task_head.load_state_dict(sub_sd)

        if aux_film is not None:
            if "aux_film_state_dict" not in state:
                raise RuntimeError(
                    "use_aux_film=true but checkpoint missing aux_film_state_dict; "
                    "the saved best.pt was trained without AuxFiLM."
                )
            aux_film.load_state_dict(state["aux_film_state_dict"])

    if len(checkpoint_paths) == 1:
        # Single-model path — decode happens inside generate_submission_grouped.
        state = load_checkpoint(checkpoint_paths[0], grouped_model, optimizer=None)
        _load_head_state(state, args.task)

        if args.task == "joint":
            head_jobs = [
                ("a1", a1_head, a1_biases, None),
                ("a2", a2_head, None, a2_offsets),
            ]
        else:
            head_jobs = [(args.task, task_head, a1_biases, a2_offsets)]

        outputs: list[Path] = []
        for head_task, head_module, biases, offsets in head_jobs:
            pids, sessions, preds = generate_submission_grouped(
                grouped_model=grouped_model,
                task_head=head_module,
                loader=loader,
                device=device,
                task=head_task,
                use_amp=use_amp,
                desc=f"Infer {args.split} ({head_task})",
                submission_level=submission_level,
                a1_biases=None if biases is None else biases.to(device),
                decode_method=selected_decode_method,
                a2_threshold_offsets=None if offsets is None else offsets.to(device),
                aux_film=aux_film if args.task == "joint" else None,
            )
            # Honor --output only when running a single head.
            explicit_out = Path(args.output) if args.output and len(head_jobs) == 1 else None
            out_path = _write_submission_csv(
                head_task=head_task,
                pids=pids,
                sessions=sessions,
                preds=preds,
                manifest_df=manifest_df,
                submission_level=submission_level,
                output_dir=submissions_dir,
                split=args.split,
                split_explicit_output=explicit_out,
            )
            outputs.append(out_path)
        for p in outputs:
            print(p)
    else:
        # Ensemble path — average raw logits across folds, then decode once.
        # Per-fold loader is deterministic (shuffle=False), so row ordering
        # matches across folds.
        if args.task == "joint":
            a1_logits_per_fold: list[torch.Tensor] = []
            a2_logits_per_fold: list[torch.Tensor] = []
            pids: list[str] = []
            sessions: list[str] = []
            for k, ckpt_path in enumerate(checkpoint_paths):
                state = load_checkpoint(ckpt_path, grouped_model, optimizer=None)
                _load_head_state(state, args.task)
                fold_pids, fold_sessions, fold_a1, fold_a2 = _gather_inference_logits_joint(
                    grouped_model=grouped_model,
                    a1_head=a1_head,
                    a2_head=a2_head,
                    loader=loader,
                    device=device,
                    use_amp=use_amp,
                    submission_level=submission_level,
                    desc=f"Infer fold {k}/{len(checkpoint_paths)}",
                    aux_film=aux_film,
                )
                if not pids:
                    pids, sessions = fold_pids, fold_sessions
                elif fold_pids != pids:
                    raise RuntimeError(
                        f"Checkpoint {ckpt_path} produced a different participant "
                        f"ordering than the first checkpoint — refusing to ensemble."
                    )
                a1_logits_per_fold.append(fold_a1)
                a2_logits_per_fold.append(fold_a2)

            avg_a1 = torch.stack(a1_logits_per_fold, dim=0).mean(dim=0).to(device)
            avg_a2 = torch.stack(a2_logits_per_fold, dim=0).mean(dim=0).to(device)
            if a1_biases is not None:
                avg_a1 = avg_a1 + a1_biases.to(device)
            a1_preds = torch.sigmoid(avg_a1).cpu().numpy()
            if a2_offsets is not None:
                avg_a2 = avg_a2 + a2_offsets.to(device)
            a2_preds = _decode_a2_logits(a2_head, avg_a2, decode_method=selected_decode_method).cpu().numpy()

            outputs = [
                _write_submission_csv(
                    "a1", pids, sessions, a1_preds, manifest_df,
                    submission_level, submissions_dir, args.split, None,
                ),
                _write_submission_csv(
                    "a2", pids, sessions, a2_preds, manifest_df,
                    submission_level, submissions_dir, args.split, None,
                ),
            ]
            for p in outputs:
                print(p)
        else:
            logits_per_fold: list[torch.Tensor] = []
            pids: list[str] = []
            sessions: list[str] = []
            for k, ckpt_path in enumerate(checkpoint_paths):
                state = load_checkpoint(ckpt_path, grouped_model, optimizer=None)
                _load_head_state(state, args.task)
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
                elif fold_pids != pids:
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
                preds = _decode_a2_logits(task_head, avg_logits, decode_method=selected_decode_method).cpu().numpy()

            explicit_out = Path(args.output) if args.output else None
            out_path = _write_submission_csv(
                head_task=args.task,
                pids=pids,
                sessions=sessions,
                preds=preds,
                manifest_df=manifest_df,
                submission_level=submission_level,
                output_dir=submissions_dir,
                split=args.split,
                split_explicit_output=explicit_out,
            )
            print(out_path)


if __name__ == "__main__":
    main()
