# Validation status

This file separates generated material from executed evidence.

| Check | Status |
| --- | --- |
| Internal documentation links and file structure | Passed: 77 local links; LF text and balanced code fences checked |
| Python documentation checker syntax and dataset-template JSON | Passed local parsing |
| GitHub workflow YAML | Passed local parsing and expected job/trigger checks |
| Git whitespace check | Passed for the staged scaffold |
| CPU C++ build and CTest | Not run; C++ compiler/CMake not available in the authoring shell |
| CUDA build and GPU CTest | Not run; CUDA toolkit/GPU session not available in the authoring shell |
| Compute Sanitizer | Not run; requires allocated GPU session |
| GitHub Actions | Workflow supplied; no remote run claimed |
| Manufacturing performance/accuracy | Not measured |

Run [Sol setup](sol-setup.md), retain the results, then update this table with environment and source commit. A passing documentation check does not validate the C++ or CUDA program.

Local checks were performed on October 22, 2026. Existing WSL/Docker runtimes could not be inspected from the restricted authoring session. No compiler was installed and no GPU measurements were generated.
