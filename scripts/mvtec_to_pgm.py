"""Convert one MVTec AD category to binary PGM/PPM images plus images.csv for Lab 07.

Expects the official layout <category>/train/good/*.png and <category>/test/<type>/*.png, where
<type> is good or a defect name. Grayscale PNGs become PGM (P5); color PNGs become PPM (P6) in
R, G, B order, without resampling. The last --dev-fraction of the sorted train/good images form
the "dev" split for threshold calibration. The official test images form "test" and stay
untouched until the rule is frozen. Requires Pillow.

MVTec AD is licensed CC BY-NC-SA 4.0 (noncommercial, attribution, share-alike). Keep the raw and
converted images out of Git, and review the terms before publishing any example image.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Convert one MVTec AD category for Lab 07.")
    parser.add_argument("category_dir", type=Path, help="for example data/raw/mvtec/bottle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.2,
                        help="fraction of train/good images held out for calibration (default 0.2)")
    args = parser.parse_args()
    if not 0.0 < args.dev_fraction < 1.0:
        parser.error("--dev-fraction must be in (0, 1).")
    try:
        from PIL import Image
    except ImportError as error:
        raise SystemExit(f"Pillow is required ({error}). Load a Python module that provides it.")

    category = args.category_dir.name
    train = sorted((args.category_dir / "train" / "good").glob("*.png"))
    test_root = args.category_dir / "test"
    test_dirs = sorted(path for path in test_root.iterdir() if path.is_dir()) if test_root.is_dir() else []
    if len(train) < 2 or not test_dirs:
        raise SystemExit(f"{args.category_dir} does not look like an MVTec AD category (train/good, test/<type>).")
    dev_count = min(len(train) - 1, max(1, round(args.dev_fraction * len(train))))
    plan = [(path, "train", "good", "good") for path in train[:-dev_count]]
    plan += [(path, "dev", "good", "good") for path in train[-dev_count:]]
    for directory in test_dirs:
        label = "good" if directory.name == "good" else "defect"
        plan += [(path, "test", label, directory.name) for path in sorted(directory.glob("*.png"))]

    args.output.mkdir(parents=True, exist_ok=True)
    rows, files = [], []
    for source, split, label, defect_type in plan:
        image = Image.open(source)
        if image.mode == "LA":
            image = image.convert("L")
        elif image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        elif image.mode not in ("L", "RGB"):
            raise SystemExit(f"{source}: unsupported image mode {image.mode}.")
        folder = "test" if split == "test" else "train"
        image_id = f"{category}-{folder}-{defect_type}-{source.stem}"
        file = f"{image_id}.{'pgm' if image.mode == 'L' else 'ppm'}"
        image.save(args.output / file, format="PPM")  # Pillow writes P5 for L and P6 for RGB
        source_digest = sha256(source)
        rows.append((image_id, file, split, label, defect_type, source_digest))
        files.append({
            "relative_path": file,
            "sha256": sha256(args.output / file),
            "source_id": source.relative_to(args.category_dir).as_posix(),
            "source_sha256": source_digest,
            "split": split,
            "label": label,
            "defect_type": defect_type,
            "width": image.width,
            "height": image.height,
            "channels": 1 if image.mode == "L" else 3,
        })

    with open(args.output / "images.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "file", "split", "label", "defect_type", "source_sha256"])
        writer.writerows(rows)
    manifest = {
        "status": "converted",
        "dataset_name": f"MVTec AD / {category}",
        "source_url": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "license_and_terms": "CC BY-NC-SA 4.0: noncommercial use with attribution",
        "retrieved_date": None,
        "selection_rule": f"every image of {category}; the last {dev_count} sorted train/good images form dev",
        "split_unit": "independent image",
        "development_items": [row[0] for row in rows if row[2] == "dev"],
        "evaluation_items": [row[0] for row in rows if row[2] == "test"],
        "files": files,
        "preprocessing": ["PNG decoded with Pillow; LA -> L and RGBA/P -> RGB; saved as binary PGM/PPM, no resampling"],
        "converted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "converter": "scripts/mvtec_to_pgm.py",
        "notes": "Fill in retrieved_date and the archive checksum you downloaded.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {split: sum(1 for row in rows if row[2] == split) for split in ("train", "dev", "test")}
    print(f"Wrote {len(rows)} images ({counts}), images.csv, and manifest.json to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
