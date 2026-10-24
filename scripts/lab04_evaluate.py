"""Evaluate Lab 04 screening reports by whole recording.

  sweep     DEVELOPMENT split only. For one feature, show how candidate thresholds trade window
            alarms on normal recordings against window alarms on fault recordings, plus the
            recording-level outcome. Use it to choose a rule, then freeze that rule.
  evaluate  Apply the `alarm` column that `screen` wrote (with frozen thresholds) to one split
            and report recording-level TP/FP/TN/FN, defect recall, and false-alarm rate.

A recording is flagged when at least --min-fraction of its windows alarm. The labels CSV has
columns recording_id,label,split with label in {normal, fault} and split in {dev, test}.
Standard library only.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

FEATURES = ("rms", "peak", "crest")


def read_labels(path):
    labels = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            recording_id = row["recording_id"].strip()
            label, split = row["label"].strip(), row["split"].strip()
            if label not in ("normal", "fault") or split not in ("dev", "test"):
                raise SystemExit(f"{path}: invalid label or split for {recording_id!r}.")
            if recording_id in labels:
                raise SystemExit(f"{path}: duplicate recording {recording_id!r}.")
            labels[recording_id] = (label, split)
    return labels


def read_reports(paths):
    windows = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                windows.setdefault(row["recording_id"], []).append(row)
    return windows


def select(labels, windows, split):
    unlabeled = sorted(set(windows) - set(labels))
    if unlabeled:
        print("Ignoring recordings without labels: " + ", ".join(unlabeled), file=sys.stderr)
    selected = []
    for recording_id, (label, recording_split) in sorted(labels.items()):
        if recording_split != split:
            continue
        if recording_id not in windows:
            raise SystemExit(f"No report rows for labeled recording {recording_id!r}.")
        selected.append((recording_id, label, windows[recording_id]))
    if not selected:
        raise SystemExit(f"No labeled recordings in split {split!r}.")
    return selected


def rate(numerator, denominator):
    return "undefined" if denominator == 0 else f"{numerator / denominator:.3f}"


def confusion(decisions):
    """decisions: (label, flagged) pairs with faults as positives."""
    tp = sum(1 for label, flagged in decisions if label == "fault" and flagged)
    fn = sum(1 for label, flagged in decisions if label == "fault" and not flagged)
    fp = sum(1 for label, flagged in decisions if label == "normal" and flagged)
    tn = sum(1 for label, flagged in decisions if label == "normal" and not flagged)
    return tp, fp, tn, fn


def evaluate(args):
    recordings = select(read_labels(args.labels), read_reports(args.reports), args.split)
    decisions = []
    print("recording_id,label,windows,alarmed,alarm_fraction,flagged")
    for recording_id, label, rows in recordings:
        alarmed = sum(int(row["alarm"]) for row in rows)
        fraction = alarmed / len(rows)
        flagged = fraction >= args.min_fraction
        decisions.append((label, flagged))
        print(f"{recording_id},{label},{len(rows)},{alarmed},{fraction:.4f},{int(flagged)}")
    tp, fp, tn, fn = confusion(decisions)
    print(f"\nSplit {args.split!r}; a recording is flagged when its alarm fraction >= {args.min_fraction}.")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}  defect_recall={rate(tp, tp + fn)}  "
          f"false_alarm_rate={rate(fp, fp + tn)}  recordings={len(decisions)}")
    return 0


def sweep(args):
    recordings = select(read_labels(args.labels), read_reports(args.reports), "dev")
    values = {recording_id: [float(row[args.feature]) for row in rows] for recording_id, _, rows in recordings}
    normal = sorted(v for recording_id, label, _ in recordings if label == "normal" for v in values[recording_id])
    fault = [v for recording_id, label, _ in recordings if label == "fault" for v in values[recording_id]]
    if not normal or not fault:
        raise SystemExit("The dev split needs both normal and fault recordings.")
    print(f"Development split only. Window alarm: {args.feature} > threshold; recording flagged when "
          f"alarm fraction >= {args.min_fraction}.")
    print("normal_quantile,threshold,normal_window_alarm_rate,fault_window_alarm_rate,TP,FP,TN,FN")
    for quantile in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0):
        index = min(len(normal) - 1, max(0, math.ceil(quantile * len(normal)) - 1))
        threshold = normal[index]
        decisions = []
        for recording_id, label, _ in recordings:
            alarmed = sum(1 for v in values[recording_id] if v > threshold)
            decisions.append((label, alarmed / len(values[recording_id]) >= args.min_fraction))
        tp, fp, tn, fn = confusion(decisions)
        normal_rate = sum(1 for v in normal if v > threshold) / len(normal)
        fault_rate = sum(1 for v in fault if v > threshold) / len(fault)
        print(f"{quantile},{threshold:.6g},{normal_rate:.4f},{fault_rate:.4f},{tp},{fp},{tn},{fn}")
    print("\nFreeze one rule, rerun `screen` on every recording with it, then run `evaluate --split test` once.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Lab 04 screening reports by recording.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("sweep", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--labels", required=True, type=Path)
        command.add_argument("--min-fraction", type=float, default=0.1,
                             help="fraction of alarmed windows that flags a recording (default 0.1)")
        command.add_argument("reports", nargs="+", type=Path, help="CSV reports written by screen")
    commands.choices["sweep"].add_argument("--feature", choices=FEATURES, required=True)
    commands.choices["evaluate"].add_argument("--split", choices=("dev", "test"), required=True)
    args = parser.parse_args()
    if not 0.0 < args.min_fraction <= 1.0:
        parser.error("--min-fraction must be in (0, 1].")
    return sweep(args) if args.command == "sweep" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
