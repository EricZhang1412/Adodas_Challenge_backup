#!/usr/bin/env python3
"""Step 0: Evaluate A2 model on A1 task by deterministic post-processing.

Takes one or more A2 submission CSVs (with file_id + d01..d21 item predictions)
and applies the DASS-21 rule to derive y_D / y_A / y_S, then reports F1 against
the manifest's ground-truth labels.

What this measures
------------------
"If we used the A2 model to predict A1 via the deterministic DASS-21 rule
 (instead of training a separate A1 head), what would the val F1 be?"

Decision logic
--------------
Compare the resulting mean F1 against your current A1 best.pt's val F1:
  mean_f1(A2-derived)  >=  mean_f1(A1-direct):
      A2 features already carry the binary signal. Joint training mainly
      benefits A2 (extra binary supervision regularizes the encoder).
  mean_f1(A2-derived)  <   mean_f1(A1-direct):
      A1 head learned something beyond the item-level recipe. Joint training
      with consistency loss has real upside.

Usage
-----
1. Generate the A2 val submission (if not already done):

    python infer.py --task a2 \\
        --checkpoint output/a2-4mlp-v5/runs/<RUN>/checkpoints/best.pt \\
        --split val

2. Run this script on the resulting submission CSV:

    python tasks/joint/eval_a2_to_a1.py \\
        --manifest /data/.../manifests/val.csv \\
        --submission output/a2-4mlp-v5/runs/<RUN>/submissions/submission_a2_val.csv

You can pass multiple --submission paths to compare runs side by side.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


GROUPS = {
    "D": ([3, 5, 10, 13, 16, 17, 21], 10),
    "A": ([2, 4, 7, 9, 15, 19, 20],    8),
    "S": ([1, 6, 8, 11, 12, 14, 18],  15),
}
ITEM_COLS = [f"d{i:02d}" for i in range(1, 22)]


def _cols(idxs):
    return [f"d{i:02d}" for i in idxs]


def _join(sub: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    m = (
        manifest
        .drop_duplicates(subset=["anon_school", "anon_class", "anon_pid"], keep="first")
        .copy()
    )
    m["file_id"] = (
        m["anon_school"].astype(str) + "_" +
        m["anon_class"].astype(str) + "_" +
        m["anon_pid"].astype(str)
    )
    return sub.merge(m[["file_id", "y_D", "y_A", "y_S"]], on="file_id", how="inner")


def evaluate_one(sub_path: Path, manifest: pd.DataFrame) -> dict:
    sub = pd.read_csv(sub_path)
    missing = [c for c in (["file_id"] + ITEM_COLS) if c not in sub.columns]
    if missing:
        print(f"  ERROR: submission missing columns {missing[:3]}{'...' if len(missing) > 3 else ''}")
        return {}

    merged = _join(sub, manifest)
    n_sub, n_merged = len(sub), len(merged)
    note = f"  (WARNING: {n_sub - n_merged} unmatched)" if n_merged < n_sub else ""
    print(f"  Rows: submission={n_sub}, after-join={n_merged}{note}")
    if n_merged == 0:
        print("  No matched rows; check file_id format consistency between submission and manifest.")
        return {}

    flat = merged[ITEM_COLS].values.flatten()
    dist = [(flat == v).mean() * 100 for v in range(4)]
    print(f"  Item-pred dist: 0={dist[0]:.1f}% 1={dist[1]:.1f}% 2={dist[2]:.1f}% 3={dist[3]:.1f}%")

    preds = np.zeros((n_merged, 3), dtype=int)
    labels = np.zeros((n_merged, 3), dtype=int)
    for k, (name, (items, cutoff)) in enumerate(GROUPS.items()):
        score = 2 * merged[_cols(items)].sum(axis=1)
        preds[:, k] = (score >= cutoff).astype(int).values
        labels[:, k] = merged[f"y_{name}"].astype(int).values

    pcf1 = [float(f1_score(labels[:, k], preds[:, k], zero_division=0.0)) for k in range(3)]
    mf1 = float(np.mean(pcf1))

    print("  Per-class metrics:")
    for k, name in enumerate(("D", "A", "S")):
        tp = int(((preds[:, k] == 1) & (labels[:, k] == 1)).sum())
        fp = int(((preds[:, k] == 1) & (labels[:, k] == 0)).sum())
        fn = int(((preds[:, k] == 0) & (labels[:, k] == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(
            f"    y_{name}: F1={pcf1[k]:.4f}  P={prec:.3f}  R={rec:.3f}  "
            f"pos_rate true={labels[:, k].mean():.3f} pred={preds[:, k].mean():.3f}  "
            f"(TP={tp} FP={fp} FN={fn})"
        )
    print(f"  >>> mean F1 (D+A+S)/3 = {mf1:.4f}")
    return {"submission": str(sub_path), "mean_f1": mf1, "pcf1": pcf1}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="Manifest CSV (val.csv) with y_D/y_A/y_S")
    p.add_argument("--submission", nargs="+", required=True,
                   help="One or more A2 submission CSVs (with file_id + d01..d21)")
    args = p.parse_args()

    manifest = pd.read_csv(args.manifest)
    results = []
    for sub_path in args.submission:
        path = Path(sub_path)
        print(f"\n=== {path} ===")
        if not path.exists():
            print("  NOT FOUND")
            continue
        r = evaluate_one(path, manifest)
        if r:
            results.append(r)

    if len(results) > 1:
        print("\n" + "=" * 80)
        print("Summary (sorted by mean F1, descending):")
        for r in sorted(results, key=lambda x: -x["mean_f1"]):
            d, a, s = r["pcf1"]
            run_tag = Path(r["submission"]).parent.parent.name
            run_tag = run_tag if len(run_tag) <= 70 else run_tag[:67] + "..."
            print(f"  mean_F1={r['mean_f1']:.4f}  D/A/S={d:.3f}/{a:.3f}/{s:.3f}  {run_tag}")


if __name__ == "__main__":
    main()
