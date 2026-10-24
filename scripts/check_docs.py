"""Check repository-local Markdown links and portable text files; no dependencies."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def main():
    root = Path(__file__).resolve().parents[1]
    ignored = {".git", "runs", ".venv", "__pycache__"}
    errors = []
    checked = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in ignored or part.startswith("build") for part in relative.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix in {".sh", ".sbatch", ".py", ".md", ".cu", ".cuh", ".cpp", ".hpp", ".yml", ".conf"} or path.name == "Makefile":
            raw = path.read_bytes()
            if b"\r" in raw:
                errors.append(f"{relative}: expected LF line endings")
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        if text.count("```") % 2:
            errors.append(f"{relative}: unbalanced fenced code blocks")
        # This repository uses simple inline links; skip code-fenced examples.
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", prose):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            if not destination.is_relative_to(root):
                errors.append(f"{relative}: link escapes repository: {target}")
            elif not destination.exists():
                errors.append(f"{relative}: missing link target: {target}")
            checked += 1
    for error in errors:
        print(error)
    print(f"Checked {checked} local links; {len(errors)} error(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
