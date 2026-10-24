# Data acquisition

No datasets are included. Use synthetic arrays/images first, then selected CWRU recordings and one MVTec AD category. See [sources and terms](../docs/references.md).

Create `data/raw/` for original files and `data/processed/` for converted inputs when storage policy permits. Both are ignored by Git. Large Sol datasets should use storage appropriate to your allocation; configure their paths rather than hardcoding a personal home directory.

Convert originals once with the course tools: [cwru_to_f32.py](../scripts/cwru_to_f32.py) for Lab 04 (writes `.f32` recordings with JSON sidecars) and [mvtec_to_pgm.py](../scripts/mvtec_to_pgm.py) for Lab 07 (writes PGM/PPM images, `images.csv`, and `manifest.json`). Both record source hashes.

Store a completed [manifest](../templates/dataset-manifest.json) next to the published report. Record source IDs, hashes, conversions, partitions, and operating conditions. Do not include credentials or confidential manufacturing data.

Keep raw images and recordings outside Git. Review dataset terms before publishing example images or derived visualizations.
