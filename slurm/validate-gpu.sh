#!/bin/bash
set -euo pipefail
# Run from the repository root INSIDE a GPU allocation: builds everything, then runs every test
# and sanitizer target. SOLUTION=1 checks the reference solutions instead of your src/ code.
SOLUTION="${SOLUTION:-0}"
make -j 4 SOLUTION="$SOLUTION" all
make SOLUTION="$SOLUTION" test
make SOLUTION="$SOLUTION" sanitize
