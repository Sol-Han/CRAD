"""
Minimal loader for the CRAD manual roll/pitch/height labels.

No CROCS/CRACCS dependency: only numpy/pandas (standard library json also works
if you don't need pandas). Run from anywhere, or import the functions directly.

Usage:
    python load_labels.py                       # prints a quick summary
    python load_labels.py --seq pohang03         # summary for one sequence
"""
import argparse
import json
import os

LABELS_DIR = os.path.join(os.path.dirname(__file__), "..", "labels")


def load_sequence_json(seq: str) -> dict:
    """Load one sequence's labels as {frame_id (int): {roll_deg, pitch_deg, height_m}}."""
    path = os.path.join(LABELS_DIR, seq, "gt_rph.json")
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_all_labels_csv():
    """Load the consolidated table across all six sequences as a pandas DataFrame
    with columns: seq, frame_id, roll_deg, pitch_deg, height_m."""
    import pandas as pd
    return pd.read_csv(os.path.join(LABELS_DIR, "gt_labels.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None, help="e.g. pohang03; omit for all sequences")
    args = ap.parse_args()

    df = load_all_labels_csv()
    if args.seq:
        df = df[df["seq"] == args.seq]
        if df.empty:
            raise SystemExit(f"no labels found for seq={args.seq}")

    print(f"{len(df)} labeled frames"
          + (f" in {args.seq}" if args.seq else " across all sequences"))
    print(df[["roll_deg", "pitch_deg", "height_m"]].describe().loc[["mean", "std", "min", "max"]])
