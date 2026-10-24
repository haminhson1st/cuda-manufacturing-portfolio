# Project 1 — Sensor-window screening

**Status:** planned; implementation exercises are Labs 02–04 (starter code and harnesses supplied, kernels not yet implemented by the owner).

## Problem

A manufacturing team has recorded vibration readings and wants to summarize many windows before investigating unusual recordings. The educational objective is a correct, efficient screening pipeline with explicit detection limitations.

## Scope

Input: synthetic signals for correctness, then a documented subset of CWRU recordings. Output: mean, RMS, peak, and an alarm flag per window. Implement CPU and CUDA paths with the same calibration and window rules.

Do not claim remaining-life prediction or a production maintenance recommendation from this baseline. Use complete held-out recordings for evaluation; adjacent windows must not leak across development/evaluation splits.

## Acceptance criteria

- [ ] Deterministic reference signals and edge cases pass.
- [ ] Partial-window denominator and empty-input behavior are documented.
- [ ] CUDA matches a higher-precision reference within stated tolerances.
- [ ] Dataset manifest records files, source, hashes, sampling, and operating conditions.
- [ ] Alarm threshold selection and held-out performance are separate.
- [ ] Timing report includes windows/s and the relevant transfer/I/O costs.
- [ ] A fresh run reproduces the CSV and report.

## Evidence to add

Link your implementation, dataset manifest, tests, run command, and [case study](../../templates/case-study.md) after completion. No measurements exist yet.

Begin with [Lab 02](../../tutorials/02-sensor-calibration.md); finish with [Lab 04](../../tutorials/04-sensor-project.md).
