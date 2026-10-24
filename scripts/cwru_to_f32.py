"""Convert one CWRU bearing-data .mat file to a Lab 04 recording (.f32 plus a JSON sidecar).

CWRU files hold arrays such as X097_DE_time (drive end), X097_FE_time (fan end), X097_BA_time
(base), and X097RPM. The sampling rate, fault description, and motor load are documented on the
CWRU Bearing Data Center pages rather than reliably inside every file, so pass them explicitly
and record where you read them. Requires NumPy and SciPy (scipy.io.loadmat).
"""
import argparse
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
    parser = argparse.ArgumentParser(description="Convert one CWRU .mat file to float32 for Lab 04.")
    parser.add_argument("mat_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", choices=("DE", "FE", "BA"), default="DE")
    parser.add_argument("--variable", help="exact variable name when the file holds several candidates")
    parser.add_argument("--sample-rate", type=float, required=True, help="Hz from the CWRU page, e.g. 12000")
    parser.add_argument("--label", choices=("normal", "fault"), required=True)
    parser.add_argument("--condition", default="", help="e.g. 'inner race, 0.007 in, 1 hp' as documented")
    parser.add_argument("--recording-id", help="default: <file stem>-<channel>")
    args = parser.parse_args()
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive.")
    try:
        import numpy as np
        from scipy.io import loadmat
    except ImportError as error:
        raise SystemExit(f"NumPy and SciPy are required ({error}). Load a Python module that provides them.")

    mat = loadmat(args.mat_file)
    suffix = f"_{args.channel}_time"
    candidates = sorted(key for key in mat if not key.startswith("__") and key.endswith(suffix))
    if args.variable:
        if args.variable not in mat:
            raise SystemExit(f"{args.mat_file}: no variable {args.variable!r}.")
        variable = args.variable
    elif len(candidates) == 1:
        variable = candidates[0]
    else:
        raise SystemExit(f"Expected one *{suffix} variable, found {candidates or 'none'}; pass --variable.")

    data = np.asarray(mat[variable], dtype=np.float64).ravel()
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise SystemExit(f"{variable} is empty or contains non-finite values.")
    samples = data.astype("<f4")
    rpm_keys = sorted(key for key in mat if key.endswith("RPM"))
    rpm = float(np.asarray(mat[rpm_keys[0]]).ravel()[0]) if rpm_keys else None

    recording_id = args.recording_id or f"{args.mat_file.stem}-{args.channel}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{recording_id}.f32"
    samples.tofile(output)
    sidecar = {
        "recording_id": recording_id,
        "source_file": args.mat_file.name,
        "source_sha256": sha256(args.mat_file),
        "variable": variable,
        "channel": args.channel,
        "rpm_in_file": rpm,
        "sample_rate_hz": args.sample_rate,
        "samples": int(samples.size),
        "duration_s": samples.size / args.sample_rate,
        "label": args.label,
        "condition": args.condition,
        "format": "float32 little-endian",
        "float32_max_abs_rounding": float(np.max(np.abs(samples.astype(np.float64) - data))),
        "output_sha256": sha256(output),
        "converted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "converter": "scripts/cwru_to_f32.py",
        "source": "CWRU Bearing Data Center, https://engineering.case.edu/bearingdatacenter",
    }
    sidecar_path = args.output_dir / f"{recording_id}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({samples.size} samples) and {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
