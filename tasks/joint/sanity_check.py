#!/usr/bin/env python3
"""DASS-21 label consistency check.

Verifies whether y_D / y_A / y_S in the manifest are the deterministic
DASS-21 function of the 21 item scores under the standard "non-normal" cutoffs
(D>=10, A>=8, S>=15 on the doubled raw sum).

Decision rule for joint training:
  agree == 1.0 on standard cutoffs
      => A1 fully determined by A2; consistency loss is well-motivated.
  agree <  1.0
      => A1 has independent label noise; prefer single-direction A2->A1 soft
         distillation at low weight, or skip the consistency term entirely.

Usage:
    python tasks/joint/sanity_check.py train.csv [val.csv ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# DASS-21 standard groupings (1-indexed item positions) and "non-normal" cutoffs
# applied to 2 * sum(items) (the DASS-42-equivalent score).
GROUPS = {
    "D": ([3, 5, 10, 13, 16, 17, 21], 10),
    "A": ([2, 4, 7, 9, 15, 19, 20],    8),
    "S": ([1, 6, 8, 11, 12, 14, 18],  15),
}


def _cols(idxs):
    return [f"d{i:02d}" for i in idxs]


def evaluate(path: Path) -> None:
    df = pd.read_csv(path)

    group_cols = [c for c in ("anon_school", "anon_class", "anon_pid") if c in df.columns]
    if group_cols:
        n_rows = len(df)
        df = df.drop_duplicates(subset=group_cols, keep="first").reset_index(drop=True)
        print(f"  Rows: {n_rows} session-rows -> {len(df)} participants")

    items_present = all(f"d{i:02d}" in df.columns for i in range(1, 22))
    labels_present = all(c in df.columns for c in ("y_D", "y_A", "y_S"))
    if not (items_present and labels_present):
        print("  SKIP: manifest missing item columns d01..d21 or label columns y_D/y_A/y_S")
        return

    scores = {n: 2 * df[_cols(items)].sum(axis=1) for n, (items, _) in GROUPS.items()}

    print("\n  Standard cutoffs:")
    all_perfect = True
    for name, (_, cutoff) in GROUPS.items():
        derived = (scores[name] >= cutoff).astype(int)
        true = df[f"y_{name}"].astype(int)
        agree = (true == derived).mean()
        fp = int(((derived == 1) & (true == 0)).sum())
        fn = int(((derived == 0) & (true == 1)).sum())
        print(
            f"    y_{name} (cut={cutoff:>2}): agree={agree:.4f}  "
            f"d+/l-:{fp:>3}  d-/l+:{fn:>3}  "
            f"pos_rate true={true.mean():.3f} derived={derived.mean():.3f}"
        )
        if agree < 1.0:
            all_perfect = False

    print("\n  Best cutoff via grid search (0..42):")
    for name, (_, std_cut) in GROUPS.items():
        score = scores[name].values
        true = df[f"y_{name}"].astype(int).values
        best_cut, best_agree = std_cut, -1.0
        for cut in range(0, 43):
            a = ((score >= cut).astype(int) == true).mean()
            if a > best_agree:
                best_agree, best_cut = a, cut
        marker = "" if best_cut == std_cut else f"  (!= standard {std_cut})"
        print(f"    y_{name}: best_cut={best_cut:>2}  agree={best_agree:.4f}{marker}")

    if not all_perfect:
        print("\n  Sample disagreement rows (first 5):")
        derived = {n: (scores[n] >= cut).astype(int) for n, (_, cut) in GROUPS.items()}
        diff_mask = pd.Series(False, index=df.index)
        for n in GROUPS:
            diff_mask |= (derived[n] != df[f"y_{n}"].astype(int))
        for idx in df.index[diff_mask][:5]:
            pid = df.loc[idx].get("anon_pid", "<?>")
            sd, sa, ss = int(scores["D"][idx]), int(scores["A"][idx]), int(scores["S"][idx])
            yd, ya, ys = int(df.loc[idx, "y_D"]), int(df.loc[idx, "y_A"]), int(df.loc[idx, "y_S"])
            dd, da_, ds = int(derived["D"][idx]), int(derived["A"][idx]), int(derived["S"][idx])
            print(
                f"    {pid}: score(D/A/S)=({sd:>2}/{sa:>2}/{ss:>2})"
                f"  y_true=({yd}/{ya}/{ys})"
                f"  y_derived=({dd}/{da_}/{ds})"
            )

    msg = (
        "A1 labels EXACTLY determined by DASS-21 rule. Consistency loss is well-motivated."
        if all_perfect
        else "A1 labels DEVIATE from standard rule. See best-cut and disagreement samples above."
    )
    print(f"\n  Conclusion: {msg}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifests", nargs="+", help="One or more manifest CSV paths (e.g. train.csv val.csv)")
    args = p.parse_args()
    for m in args.manifests:
        path = Path(m)
        print(f"\n=== {path} ===")
        if not path.exists():
            print("  NOT FOUND")
            continue
        evaluate(path)


if __name__ == "__main__":
    main()
