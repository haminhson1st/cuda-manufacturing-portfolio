"""Calibrate and evaluate the Lab 07 inspection score.

  calibrate  From scores of the development split (score_images --split dev), choose the image
             threshold whose false-reject rate on the GOOD images is at most --false-reject
             (default 0.05). Freeze it before looking at the test split.
  evaluate   Apply a frozen --threshold to held-out scores (score_images --split test) and report
             TP/FP/TN/FN with defects as positives, defect recall, false-reject rate, recall per
             defect type, the worst misses and false rejects, and images near the threshold.

Standard library only.
"""
import argparse
import csv
import math


def read_scores(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no rows.")
    for row in rows:
        if row["label"] not in ("good", "defect"):
            raise SystemExit(f"{path}: label must be good or defect (image {row['image_id']}).")
        row["score"] = float(row["score"])
    return rows


def rate(numerator, denominator):
    return "undefined" if denominator == 0 else f"{numerator / denominator:.3f}"


def calibrate(args):
    rows = read_scores(args.scores)
    if any(row["split"] == "test" for row in rows):
        raise SystemExit("calibrate refuses test-split rows: score the dev split only.")
    good = sorted((row["score"] for row in rows if row["label"] == "good"), reverse=True)
    if not good:
        raise SystemExit("No good images to calibrate on.")
    allowed = math.floor(args.false_reject * len(good))  # good images allowed above the threshold
    threshold = good[min(allowed, len(good) - 1)]
    rejected = sum(1 for score in good if score > threshold)
    print(f"Good images: {len(good)}; scores min {good[-1]:.6g}, median {good[len(good) // 2]:.6g}, max {good[0]:.6g}.")
    print(f"Threshold {threshold:.9g}: dev false-reject rate {rate(rejected, len(good))} (target <= {args.false_reject}).")
    defects = [row for row in rows if row["label"] == "defect"]
    if defects:
        found = sum(1 for row in defects if row["score"] > threshold)
        print(f"Warning: this file also holds {len(defects)} defect images (recall {rate(found, len(defects))}). "
              "Images that influence the threshold must be excluded from the final evaluation.")
    print("Freeze the threshold, score the test split once with --image-threshold, then run evaluate.")
    return 0


def evaluate(args):
    rows = read_scores(args.scores)
    threshold = args.threshold
    tp = fp = tn = fn = disagreements = 0
    by_type = {}
    for row in rows:
        predicted = row["score"] > threshold
        recorded = (row.get("predicted_defect") or "").strip()
        if recorded and int(recorded) != int(predicted):
            disagreements += 1
        if row["label"] == "defect":
            found, total = by_type.get(row["defect_type"], (0, 0))
            by_type[row["defect_type"]] = (found + predicted, total + 1)
            tp += predicted
            fn += not predicted
        else:
            fp += predicted
            tn += not predicted
    splits = ", ".join(sorted({row["split"] for row in rows}))
    print(f"Threshold {threshold:.9g} on {len(rows)} images (split: {splits}); defects are positives.")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}  defect_recall={rate(tp, tp + fn)}  false_reject_rate={rate(fp, fp + tn)}")
    if disagreements:
        print(f"Warning: {disagreements} predicted_defect values disagree with --threshold. "
              "Was score_images run with a different --image-threshold?")
    if by_type:
        print("Recall by defect type:")
        for defect_type, (found, total) in sorted(by_type.items()):
            print(f"  {defect_type}: {found}/{total}")
    misses = sorted((row for row in rows if row["label"] == "defect" and row["score"] <= threshold),
                    key=lambda row: row["score"])
    rejects = sorted((row for row in rows if row["label"] == "good" and row["score"] > threshold),
                     key=lambda row: row["score"], reverse=True)
    near = [row for row in rows if abs(row["score"] - threshold) <= args.near * max(abs(threshold), 1e-12)]
    for title, items in (("Worst misses (defects scored lowest)", misses),
                         ("Worst false rejects (good images scored highest)", rejects),
                         (f"Within {args.near:.0%} of the threshold", near)):
        if items:
            print(f"{title}:")
            for row in items[:args.show]:
                print(f"  {row['image_id']}  {row['defect_type']}  score={row['score']:.6g}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Calibrate and evaluate Lab 07 inspection scores.")
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = commands.add_parser("calibrate")
    calibrate_parser.add_argument("--scores", required=True, help="score_images output for the dev split")
    calibrate_parser.add_argument("--false-reject", type=float, default=0.05,
                                  help="largest acceptable false-reject rate on good dev images")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--scores", required=True, help="score_images output for the test split")
    evaluate_parser.add_argument("--threshold", type=float, required=True, help="the frozen image threshold")
    evaluate_parser.add_argument("--near", type=float, default=0.1, help="relative band reported as near the threshold")
    evaluate_parser.add_argument("--show", type=int, default=5, help="images listed per failure category")
    args = parser.parse_args()
    if args.command == "calibrate":
        if not 0.0 <= args.false_reject < 1.0:
            parser.error("--false-reject must be in [0, 1).")
        return calibrate(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
