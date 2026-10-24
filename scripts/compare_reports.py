"""Compare CSV reports and check run records (Labs 11 and 12).

  same A.csv B.csv --key COLS --columns COLS        the listed columns hold identical text
  close A.csv B.csv --key COLS --columns COLS [--rel R] [--abs A]
                                                     numeric columns agree within abs + rel * |b|
  threshold REPORT.csv --threshold T                 predicted_defect == (score > T) on every row
  record RUN.json KEY=VALUE ...                      dotted keys of a run record hold these values

COLS are comma-separated. Rows are matched by their key columns, and both reports must contain
the same keys. Standard library only; exits with status 1 when a check fails.
"""
import argparse
import csv
import json


def read_rows(path, key_columns):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    table = {}
    for row in rows:
        missing = [column for column in key_columns if column not in row]
        if missing:
            raise SystemExit(f"{path}: missing key column(s) {', '.join(missing)}.")
        key = tuple(row[column] for column in key_columns)
        if key in table:
            raise SystemExit(f"{path}: duplicate key {key}.")
        table[key] = row
    if not table:
        raise SystemExit(f"{path} has no rows.")
    return table


def compare(args, numeric):
    keys = args.key.split(",")
    columns = args.columns.split(",")
    a, b = read_rows(args.a, keys), read_rows(args.b, keys)
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        print(f"Keys differ: {len(only_a)} only in {args.a} (e.g. {only_a[:3]}), "
              f"{len(only_b)} only in {args.b} (e.g. {only_b[:3]}).")
        return 1
    differences = 0
    for key in sorted(a):
        for column in columns:
            if column not in a[key] or column not in b[key]:
                raise SystemExit(f"Column {column!r} is missing from one of the reports.")
            x, y = a[key][column], b[key][column]
            if numeric:
                ok = abs(float(x) - float(y)) <= args.abs + args.rel * abs(float(y))
            else:
                ok = x == y
            if not ok:
                differences += 1
                if differences <= 10:
                    print(f"  {dict(zip(keys, key))} {column}: {x!r} vs {y!r}")
    print(f"{len(a)} rows x {len(columns)} columns compared: {differences} difference(s).")
    return 1 if differences else 0


def threshold(args):
    with open(args.report, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{args.report} has no rows.")
    mismatches = boundary = 0
    for row in rows:
        recorded = (row.get("predicted_defect") or "").strip()
        if recorded == "":
            raise SystemExit(f"{args.report}: predicted_defect is empty; run with --image-threshold.")
        score = float(row["score"])
        boundary += score == args.threshold
        if int(recorded) != int(score > args.threshold):
            mismatches += 1
            print(f"  {row['image_id']}: score {score!r}, predicted_defect {recorded}")
    print(f"{len(rows)} rows: {mismatches} decision(s) disagree with score > {args.threshold}; "
          f"{boundary} row(s) exactly on the threshold.")
    return 1 if mismatches else 0


def record(args):
    with open(args.record, encoding="utf-8") as handle:
        data = json.load(handle)
    problems = 0
    for expectation in args.expect:
        path, sep, wanted = expectation.partition("=")
        if not sep:
            raise SystemExit(f"Expected KEY=VALUE, got {expectation!r}.")
        value = data
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                value = KeyError
                break
            value = value[part]
        if value is KeyError:
            ok = False
        elif value is None:
            ok = wanted == "null"
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                ok = float(wanted) == float(value)
            except ValueError:
                ok = False
        else:
            ok = str(value) == wanted
        if not ok:
            problems += 1
            print(f"  {path}: expected {wanted!r}, found {value if value is not KeyError else 'nothing'!r}")
    print(f"{len(args.expect)} expectation(s) checked: {problems} problem(s).")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description="Compare reports and check run records.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("same", "close"):
        command = commands.add_parser(name)
        command.add_argument("a")
        command.add_argument("b")
        command.add_argument("--key", required=True, help="comma-separated key columns")
        command.add_argument("--columns", required=True, help="comma-separated columns to compare")
        if name == "close":
            command.add_argument("--rel", type=float, default=1e-5)
            command.add_argument("--abs", type=float, default=0.0)
    threshold_parser = commands.add_parser("threshold")
    threshold_parser.add_argument("report")
    threshold_parser.add_argument("--threshold", type=float, required=True)
    record_parser = commands.add_parser("record")
    record_parser.add_argument("record")
    record_parser.add_argument("expect", nargs="+", help="dotted.key=value")
    args = parser.parse_args()
    if args.command == "same":
        return compare(args, numeric=False)
    if args.command == "close":
        return compare(args, numeric=True)
    if args.command == "threshold":
        return threshold(args)
    return record(args)


if __name__ == "__main__":
    raise SystemExit(main())
