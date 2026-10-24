# Twelve weeks of CUDA for manufacturing

Use 6–8 hours per week: roughly one hour of reading, four hours of implementation, and one to three hours of verification and explanation. Finish a lab's acceptance criteria before adding its extension.

| Week | Tutorial | Deliverable |
| --- | --- | --- |
| 1 | [First GPU program](01-first-gpu.md) | Explain, run, and modify calibration reference |
| 2 | [Multiple sensor channels](02-sensor-calibration.md) | Layout-aware calibration and threshold flags |
| 3 | [Window statistics](03-window-statistics.md) | Correct mean, RMS, and peak reductions |
| 4 | [Sensor screening case study](04-sensor-project.md) | Real-recording analysis and timing report |
| 5 | [Pixels and image layout](05-image-indexing.md) | Correct per-pixel pipeline |
| 6 | [Shared-memory filtering](06-tiled-filter.md) | Direct/tiled filter comparison |
| 7 | [Inspection case study](07-image-project.md) | Defect scores, quality metrics, failure cases |
| 8 | [Use a library well](08-library-comparison.md) | Custom versus library reduction |
| 9 | [Profile and improve](09-profiling.md) | One evidence-backed optimization |
| 10 | [Batching and streams](10-streaming.md) | Throughput/latency comparison |
| 11 | [Build the capstone](11-capstone.md) | Reproducible command-line application |
| 12 | [Publish engineering evidence](12-portfolio-release.md) | Case study, demonstration, release |

Only Lab 01 currently includes implementation code. Later lessons describe code you will write. Keep your first working version before optimizing so you retain a comparison.

## The weekly routine

1. Read the business task and define the output.
2. Run or inspect the linked tutorial and explain its mapping in your own words.
3. Implement the smallest correct CPU/GPU versions.
4. Test edges and run the applicable sanitizer checks.
5. Change one variable and collect measurements.
6. Write a short [notebook entry](../templates/lab-notebook.md), commit your work, and update the project status with a link to evidence.

Your prior C/C++ skills transfer to input handling and host code. Your architecture background is useful for reasoning about locality and bandwidth, while your verification experience is useful for edge cases and invariants. The new work is expressing parallelism correctly and measuring its effect.

For setup, begin with [Tutorial 00](../docs/sol-setup.md). For publishing, use the [GitHub walkthrough](../docs/github-publishing.md).
