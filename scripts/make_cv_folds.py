#!/usr/bin/env python3
"""Generate K-fold CV splits at the participant level.

Reads `train.csv` + `val.csv` from `manifest_dir`, combines them into a
single pool, splits the *participants* (not sessions) into K folds with
stratification on the y_a2 total-score quantile, and writes session-level
CSVs to `manifest_dir/cv{K}/train_fold_{k}.csv` and `val_fold_{k}.csv`.

Participant-level grouping is enforced by deduplicating on `anon_pid` before
splitting; the resulting fold IDs then filter the original session-level rows.
This guarantees no participant ever appears in both train and val of the same
fold (which would be a leak — same person, different sessions).

Stratification uses the total y_a2 score (sum of d01..d21) bucketed into
quantiles, so each fold sees a similar mix of low/medium/high-severity cases.

Usage:
    python scripts/make_cv_folds.py --manifest_dir /path/to/manifests --n_folds 5
    python scripts/make_cv_folds.py --manifest_dir /path/to/manifests --n_folds 10 --seed 0

Output layout:
    manifest_dir/
        train.csv                     (untouched)
        val.csv                       (untouched)
        cv5/
            train_fold_0.csv          (~80% participants, all their sessions)
            val_fold_0.csv            (~20% participants, all their sessions)
            train_fold_1.csv
            ...
            split_summary.json        (fold sizes + stratification distribution)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ITEM_COLS = [f"d{i:02d}" for i in range(1, 22)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--manifest_dir", type=str, required=True,
        help="Directory containing train.csv and val.csv (the official splits).",
    )
    p.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of folds (K). 5 is the typical sweet spot; 10 if compute is cheap.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling participants before stratified splitting.",
    )
    p.add_argument(
        "--n_strata", type=int, default=4,
        help="Number of quantile buckets for y_a2 total. Default 4 (quartiles). "
             "Lower this if K * n_strata exceeds the number of labeled participants.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing cv{K} directory instead of erroring.",
    )
    return p.parse_args()


def load_pool(manifest_dir: Path) -> pd.DataFrame:
    train_path = manifest_dir / "train.csv"
    val_path = manifest_dir / "val.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}")

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    pool = pd.concat([train, val], ignore_index=True)

    n_train_pids = train["anon_pid"].nunique()
    n_val_pids = val["anon_pid"].nunique()
    n_pool_pids = pool["anon_pid"].nunique()
    print(
        f"Loaded train.csv: {len(train)} rows / {n_train_pids} pids   "
        f"val.csv: {len(val)} rows / {n_val_pids} pids   "
        f"pool: {len(pool)} rows / {n_pool_pids} pids"
    )
    if n_pool_pids != n_train_pids + n_val_pids:
        # Same participant in both train.csv and val.csv would be unusual but
        # not necessarily wrong (e.g., re-released data). Just warn.
        print(
            f"  WARNING: train and val share {n_train_pids + n_val_pids - n_pool_pids} "
            f"participant id(s). Using deduplicated pool."
        )
    return pool


def participant_table(pool: pd.DataFrame, n_strata: int) -> pd.DataFrame:
    """Reduce session-level rows to one row per participant + a stratification label.

    Stratum = quantile bucket of y_a2 total score (sum of d01..d21).
    Participants with any unlabeled item (NaN) get bucket -1 so they form their
    own "unlabeled" stratum (kept together to avoid leaking labels via stratum
    assignment).
    """
    parts = pool.drop_duplicates(subset="anon_pid", keep="first").copy()

    a2_present = [c for c in ITEM_COLS if c in parts.columns]
    if len(a2_present) < len(ITEM_COLS):
        print(
            f"  WARNING: only {len(a2_present)}/21 A2 item columns present in manifest; "
            f"stratification will use whatever is available."
        )
    if not a2_present:
        # No A2 labels at all — everyone goes to one stratum, splits are random
        parts = parts[["anon_pid"]].copy()
        parts["stratum"] = 0
        parts["sum_score"] = 0
        return parts

    y = parts[a2_present].apply(pd.to_numeric, errors="coerce")
    is_labeled = y.notna().all(axis=1) & (y >= 0).all(axis=1)
    sum_score = y.where(y >= 0, 0).sum(axis=1)

    strata = pd.Series(-1, index=parts.index, dtype=int)
    if is_labeled.sum() >= n_strata:
        # Rank-based qcut to be robust to heavy ties at sum_score=0
        ranked = sum_score[is_labeled].rank(method="first")
        try:
            strata.loc[is_labeled] = pd.qcut(
                ranked, q=n_strata, labels=False, duplicates="drop"
            ).astype(int)
        except ValueError:
            # If qcut still collapses to <2 buckets, single stratum
            strata.loc[is_labeled] = 0
    else:
        strata.loc[is_labeled] = 0

    return pd.DataFrame({
        "anon_pid": parts["anon_pid"].astype(str).values,
        "stratum": strata.values,
        "sum_score": sum_score.values,
    })


def build_folds(
    parts: pd.DataFrame, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns list of (train_pids, val_pids) numpy arrays for each fold."""
    pids = parts["anon_pid"].values
    strata = parts["stratum"].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(pids, strata):
        folds.append((pids[train_idx], pids[val_idx]))
    return folds


def write_folds(
    pool: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_pid_str = pool["anon_pid"].astype(str)
    summary = {"n_folds": len(folds), "folds": []}

    for k, (train_pids, val_pids) in enumerate(folds):
        train_set = set(map(str, train_pids))
        val_set = set(map(str, val_pids))

        # Sanity: no overlap before writing
        overlap = train_set & val_set
        if overlap:
            raise RuntimeError(
                f"Internal error: fold {k} has {len(overlap)} overlapping pids"
            )

        train_mask = pool_pid_str.isin(train_set)
        val_mask = pool_pid_str.isin(val_set)

        train_rows = pool[train_mask]
        val_rows = pool[val_mask]

        train_path = out_dir / f"train_fold_{k}.csv"
        val_path = out_dir / f"val_fold_{k}.csv"
        train_rows.to_csv(train_path, index=False)
        val_rows.to_csv(val_path, index=False)

        info = {
            "fold": k,
            "train_pids": len(train_set),
            "train_rows": int(train_mask.sum()),
            "val_pids": len(val_set),
            "val_rows": int(val_mask.sum()),
            "train_csv": train_path.name,
            "val_csv": val_path.name,
        }
        summary["folds"].append(info)
        print(
            f"  fold {k}: train={info['train_pids']} pids / {info['train_rows']} rows   "
            f"val={info['val_pids']} pids / {info['val_rows']} rows"
        )

    return summary


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {manifest_dir}")

    out_dir = manifest_dir / f"cv{args.n_folds}"
    if out_dir.exists() and not args.force:
        existing = sorted(p.name for p in out_dir.iterdir())
        raise FileExistsError(
            f"{out_dir} already exists with {len(existing)} entries. "
            f"Pass --force to overwrite."
        )

    pool = load_pool(manifest_dir)
    parts = participant_table(pool, n_strata=args.n_strata)

    stratum_counts = parts["stratum"].value_counts().sort_index().to_dict()
    print(f"\nStratum distribution (-1 = unlabeled): {stratum_counts}")
    print(f"y_a2 sum_score: min={parts['sum_score'].min():.0f} "
          f"median={parts['sum_score'].median():.0f} "
          f"max={parts['sum_score'].max():.0f}")

    folds = build_folds(parts, n_folds=args.n_folds, seed=args.seed)

    print(f"\nWriting {args.n_folds} folds to {out_dir}/")
    summary = write_folds(pool, folds, out_dir)
    summary["seed"] = args.seed
    summary["n_strata"] = args.n_strata
    summary["stratum_distribution"] = {int(k): int(v) for k, v in stratum_counts.items()}

    summary_path = out_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")

    # Final sanity loop using the actual files we just wrote
    print("\nVerifying no participant leaks within any fold...")
    for k in range(args.n_folds):
        t = pd.read_csv(out_dir / f"train_fold_{k}.csv")
        v = pd.read_csv(out_dir / f"val_fold_{k}.csv")
        leak = set(t["anon_pid"].astype(str)) & set(v["anon_pid"].astype(str))
        assert not leak, f"Fold {k} train/val share {len(leak)} pid(s) — bug!"
    print("OK: every fold is participant-disjoint.\n")

    print("Next steps:")
    print(f"  # Train each fold:")
    print(f"  for k in $(seq 0 {args.n_folds - 1}); do")
    print(f"    uv run python train.py --task a2 --config tasks/a2/default.yaml \\")
    print(f"        --fold $k --n_folds {args.n_folds}")
    print(f"  done")
    print()
    print(f"  # Ensemble inference (after all folds finish):")
    print(f"  uv run python infer.py --task a2 --split test_hidden \\")
    print(f"      --checkpoints output/.../fold_0/.../best.pt,output/.../fold_1/.../best.pt,...")


if __name__ == "__main__":
    main()
