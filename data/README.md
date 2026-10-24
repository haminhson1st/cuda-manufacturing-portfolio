# Data acquisition

No datasets are included. Use synthetic arrays/images first, then selected CWRU recordings and one MVTec AD category. See [sources and terms](../docs/references.md).

Create `data/raw/` for original files and `data/processed/` for converted inputs when storage policy permits. Both are ignored by Git. Large Sol datasets should use storage appropriate to your allocation; configure their paths rather than hardcoding a personal home directory.

Store a completed [manifest](../templates/dataset-manifest.json) next to the published report. Record source IDs, hashes, conversions, partitions, and operating conditions. Do not include credentials or confidential manufacturing data.

Keep raw images and recordings outside Git. Review dataset terms before publishing example images or derived visualizations.
