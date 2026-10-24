#!/bin/bash
set -euo pipefail
# Run from the repository root INSIDE a GPU allocation, after building.
ctest --test-dir build-gpu --output-on-failure
for n in 1 255 256 257 1000003; do
    compute-sanitizer --tool memcheck --error-exitcode 3 \
        ./build-gpu/calibrate_cuda "$n"
done
# For later shared-memory lessons, also run racecheck and synccheck on their targets.
