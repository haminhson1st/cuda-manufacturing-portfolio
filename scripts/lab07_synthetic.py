"""Write a synthetic inspection image set for Lab 07 tests and rehearsal.

Good images contain an illumination gradient, a low-frequency texture, and mild noise. Defect
images add a thin bright scratch or a small dark spot with sharp edges, which the residual
detector should notice. Images are binary PGM (P5), and images.csv lists
image_id,file,split,label,defect_type. The edge/ subdirectory holds tiny, thin, and RGB (PPM)
images plus a malformed file for boundary tests. These images exercise the pipeline; they are
not evidence about real inspection data. Standard library only.
"""
import argparse
import csv
import math
import random
from pathlib import Path


def good_values(width, height, rng):
    values = []
    for y in range(height):
        row = []
        for x in range(width):
            v = 90 + 60 * x / max(1, width - 1) + 30 * y / max(1, height - 1)
            v += 12 * math.sin(2 * math.pi * x / 23.0) * math.sin(2 * math.pi * y / 31.0)
            v += rng.uniform(-4.0, 4.0)
            row.append(v)
        values.append(row)
    return values


def add_scratch(values, rng):
    height, width = len(values), len(values[0])
    x0, y0 = rng.uniform(0.25, 0.75) * width, rng.uniform(0.25, 0.75) * height
    angle = rng.uniform(0.0, math.pi)
    half = 0.2 * min(width, height)
    steps = max(1, int(4 * half))
    pixels = set()
    for k in range(steps + 1):
        s = -half + 2 * half * k / steps
        pixels.add((int(round(x0 + s * math.cos(angle))), int(round(y0 + s * math.sin(angle)))))
    for x, y in pixels:
        if 0 <= x < width and 0 <= y < height:
            values[y][x] += 80


def add_spot(values, rng):
    height, width = len(values), len(values[0])
    cx, cy = rng.uniform(0.25, 0.75) * width, rng.uniform(0.25, 0.75) * height
    for y in range(height):
        for x in range(width):
            if (x - cx) ** 2 + (y - cy) ** 2 <= 4.0:
                values[y][x] -= 90


def make_values(width, height, rng, kind):
    values = good_values(width, height, rng)
    if kind == "scratch":
        add_scratch(values, rng)
    elif kind == "spot":
        add_spot(values, rng)
    return values


def write_pnm(path, values, channels):
    height, width = len(values), len(values[0])
    data = bytearray()
    for row in values:
        for v in row:
            g = min(255, max(0, int(round(v))))
            if channels == 1:
                data.append(g)
            else:  # R, G, B with different gains, so the gray conversion matters
                data.extend((g, min(255, int(g * 0.9) + 10), min(255, int(g * 0.8) + 20)))
    magic = "P5" if channels == 1 else "P6"
    path.write_bytes(f"{magic}\n{width} {height}\n255\n".encode("ascii") + bytes(data))


def write_list(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "file", "split", "label", "defect_type"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Write a synthetic Lab 07 image set.")
    parser.add_argument("--output", required=True, type=Path, help="directory to create or reuse")
    parser.add_argument("--size", type=int, default=96, help="width and height of the main images")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.size < 8:
        parser.error("--size must be at least 8.")

    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, kind, count in (("train", "good", 8), ("dev", "good", 6), ("test", "good", 6),
                               ("test", "scratch", 3), ("test", "spot", 3)):
        for k in range(count):
            image_id = f"syn-{split}-{kind}-{k:02d}"
            write_pnm(args.output / f"{image_id}.pgm", make_values(args.size, args.size, rng, kind), 1)
            rows.append((image_id, f"{image_id}.pgm", split, "good" if kind == "good" else "defect", kind))
    write_list(args.output / "images.csv", rows)

    edge = args.output / "edge"
    edge.mkdir(exist_ok=True)
    edge_rows = []
    for image_id, width, height, channels, kind in (("edge-one", 1, 1, 1, "good"),
                                                    ("edge-rgb-tiny", 3, 2, 3, "good"),
                                                    ("edge-rgb-scratch", 37, 29, 3, "scratch"),
                                                    ("edge-wide", 300, 7, 1, "spot")):
        file = f"{image_id}.{'pgm' if channels == 1 else 'ppm'}"
        write_pnm(edge / file, make_values(width, height, rng, kind), channels)
        edge_rows.append((image_id, file, "test", "good" if kind == "good" else "defect", kind))
    write_list(edge / "images.csv", edge_rows)
    (edge / "bad-header.pgm").write_bytes(b"P2\n2 2\n255\n1 2 3 4\n")  # ASCII PGM: rejected by read_pnm
    write_list(edge / "bad.csv", [("edge-bad", "bad-header.pgm", "test", "good", "good")])
    print(f"Wrote {len(rows)} images and images.csv, plus edge cases, to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
