# Validation status

This file separates generated material from executed evidence.

| Check | Status |
| --- | --- |
| Internal documentation links, LF line endings, balanced code fences | `scripts/check_docs.py`, run locally on October 23, 2026; see the development log for the result |
| Lab 04 Python tools (`lab04_synthetic.py`, `lab04_evaluate.py`) | Ran locally on synthetic data with a Python stand-in for `screen`'s report; `cwru_to_f32.py` only checked for argument parsing (needs SciPy and real CWRU files) |
| Lab 07 Python tools (`lab07_synthetic.py`, `lab07_evaluate.py`, `mvtec_to_pgm.py`) | Ran locally: synthetic images scored by a Python stand-in for `score_images` that applies the same integer rules, and the converter run on a fake category of RGB, RGBA, and gray PNGs with Pillow 12.3. Not run on real MVTec AD data |
| Lab 11–12 Python tools (`compare_reports.py`, `lab12_report.py`) | Ran locally: comparisons on Lab 07 score files (agreeing and deliberately mismatched inputs), record checks on a sample JSON, and report generation from a placeholder `release_bench` directory with invented numbers (formatting only, discarded) |
| GitHub workflow YAML | Supplied; no remote run claimed |
| CPU C++ build and `make test-cpu` | Not run; no C++ compiler or make in the authoring environment |
| CUDA build of Labs 01–12 (starter and `SOLUTION=1`) | Not run; no CUDA toolkit in the authoring environment. The `cuda-compile` CI job will compile both once the repository is pushed. Lab 08's Thrust/CUB calls and Lab 09's NVTX header are the most toolkit-version-sensitive code |
| Lab 09 profiling targets (`make profile-lab09`) | Not run; need Nsight Systems and Nsight Compute in a GPU allocation |
| GPU tests (`make test-gpu`, `make SOLUTION=1 test-gpu`) | Not run; requires an allocated GPU session |
| Compute Sanitizer (`make sanitize`) | Not run; requires an allocated GPU session |
| Manufacturing performance and accuracy | Not measured |

**First thing to do on Sol:** `make SOLUTION=1 -j 4 && make SOLUTION=1 test-gpu && make SOLUTION=1 sanitize`. That is the first compilation and execution of the Lab 02–12 harnesses and solutions. Record the result here with the environment and source commit. Report any failure as a course defect, not as a defect in your own kernels.

A passing documentation check does not validate the C++ or CUDA programs.
