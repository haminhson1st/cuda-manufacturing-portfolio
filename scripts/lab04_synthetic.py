"""Write synthetic vibration recordings for Lab 04 tests and rehearsal.

Each recording is little-endian float32 samples (<id>.f32) plus a JSON sidecar (<id>.json), and
labels.csv assigns whole recordings to dev/test splits. Normal recordings contain two shaft
tones and broadband noise; fault recordings add periodic decaying impacts, a simplified stand-in
for a localized bearing defect. They exercise the pipeline; they are not evidence about real
bearings. The edge/ subdirectory holds 1- and 3-sample recordings for boundary tests.

Standard library only.
"""
import argparse
import array
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

SAMPLE_RATE_HZ = 12000.0  # the CWRU 12 kHz drive-end rate
SHAFT_HZ = 29.95  # about 1797 rpm
IMPACT_HZ = 107.4  # illustrative fault repetition rate
RESONANCE_HZ = 3100.0  # structural resonance excited by each impact
DECAY_PER_S = 900.0


def gaussian(rng):
    """Box-Muller from rng.random(), which is reproducible for a fixed seed."""
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_recording(seconds, fault, rng):
    n = int(seconds * SAMPLE_RATE_HZ) + rng.randrange(0, 997)  # not a multiple of typical windows
    noise = 0.02 + 0.01 * rng.random()
    impact = 0.6 + 0.4 * rng.random()
    phase = rng.random() * 2.0 * math.pi
    next_impact = rng.random() / IMPACT_HZ
    last_impact = None
    values = array.array("f")
    for i in range(n):
        t = i / SAMPLE_RATE_HZ
        v = 0.05 * math.sin(2.0 * math.pi * SHAFT_HZ * t + phase) + 0.02 * math.sin(4.0 * math.pi * SHAFT_HZ * t)
        v += noise * gaussian(rng)
        if fault:
            if t >= next_impact:
                last_impact = next_impact
                next_impact += (1.0 + 0.02 * gaussian(rng)) / IMPACT_HZ
            if last_impact is not None:
                dt = t - last_impact
                v += impact * math.exp(-DECAY_PER_S * dt) * math.sin(2.0 * math.pi * RESONANCE_HZ * dt)
        values.append(v)
    return values


def write_recording(directory, recording_id, values, metadata):
    if sys.byteorder != "little":
        values.byteswap()
    path = directory / f"{recording_id}.f32"
    path.write_bytes(values.tobytes())
    sidecar = dict(metadata)
    sidecar.update(
        recording_id=recording_id,
        samples=len(values),
        sample_rate_hz=SAMPLE_RATE_HZ,
        format="float32 little-endian",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        generator="scripts/lab04_synthetic.py",
    )
    (directory / f"{recording_id}.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Write synthetic Lab 04 recordings and labels.csv.")
    parser.add_argument("--output", required=True, type=Path, help="directory to create or reuse")
    parser.add_argument("--seconds", type=float, default=2.0, help="approximate length of each recording")
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--per-class", type=int, default=4, help="recordings per class; half dev, half test")
    args = parser.parse_args()
    if args.seconds <= 0 or args.per_class < 2:
        parser.error("--seconds must be positive and --per-class at least 2.")

    args.output.mkdir(parents=True, exist_ok=True)
    edge = args.output / "edge"
    edge.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for label in ("normal", "fault"):
        for k in range(1, args.per_class + 1):
            recording_id = f"syn-{label}-{k:02d}"
            split = "dev" if k <= args.per_class // 2 else "test"
            values = make_recording(args.seconds, label == "fault", rng)
            write_recording(args.output, recording_id, values,
                            {"label": label, "split": split, "seed": args.seed, "synthetic": True})
            rows.append((recording_id, label, split))
    with open(args.output / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["recording_id", "label", "split"])
        writer.writerows(rows)
    write_recording(edge, "one-sample", array.array("f", [0.5]), {"synthetic": True, "purpose": "edge case"})
    write_recording(edge, "three-samples", array.array("f", [1.0, -2.0, 3.0]),
                    {"synthetic": True, "purpose": "edge case"})
    print(f"Wrote {len(rows)} recordings, labels.csv, and edge cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
