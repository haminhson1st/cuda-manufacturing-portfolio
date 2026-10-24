# Project 2 — Visual defect screening

**Status:** planned; implementation exercises are Labs 05–07 (starter code and data tools for Labs 05–07 supplied, kernels not yet implemented by the owner).

## Problem

A quality team needs to route images with possible defects for further inspection. Speed matters only if the detector's missed defects and false rejects are understood.

## Scope

Start with synthetic images, then one MVTec AD category. Implement grayscale/normalization, local filtering, residual or difference scoring, a threshold mask, and reduction to an image score. Select a detector suitable for the category and document its assumptions.

The first detector is a simple educational baseline. It is allowed to fail; investigate and report failure modes. Keep raw dataset images outside the repository and respect the source's noncommercial terms.

## Acceptance criteria

- [ ] CPU/GPU outputs agree for tiny, padded-stride, and non-multiple images.
- [ ] Boundary, channel-order, and rounding rules are explicit.
- [ ] Applicable memory and synchronization checks pass.
- [ ] Thresholds are fixed before evaluating the final held-out set.
- [ ] Report includes defect recall, false-reject rate, and example counts.
- [ ] Timing distinguishes decode, transfers, kernels, and full pipeline.
- [ ] At least three failures are analyzed.

## Evidence to add

Link your implementation, permitted output examples, parameter configuration, tests, and a [case study](../../templates/case-study.md). No accuracy or performance measurements exist yet.

Begin with [Lab 05](../../tutorials/05-image-indexing.md); finish with [Lab 07](../../tutorials/07-image-project.md).
