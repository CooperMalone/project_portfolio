# Cooper Malone · Project Portfolio

Data science, statistics and machine learning projects from my work at the University of Nebraska–Lincoln. Each folder has the source code, the results and a write-up. Start with the project README; the full paper or report is linked from it.

| Project | What it is | Headline result | Stack |
|---|---|---|---|
| [**Vision Zero Crash Analysis**](./vision-zero-crash-analysis) | Checks whether the patterns behind Lincoln, NE's traffic-safety plan still hold, using all 6,776 crashes from 2025. Work for the City of Lincoln. | Pedestrians were in **20.6%** of fatal/serious-injury crashes, nearly double the plan's 11.0% (p = 0.003). Also found **8 high-risk intersections** that a raw crash count misses. | Python, pandas, SciPy, Folium, Matplotlib |
| [**Lane Marker Detection**](./lane-marker-detection) | A Spatial CNN (ResNet-18 backbone) that segments lane markings in CULane driving images, trained on a GPU cluster. | Best validation **F1 = 0.437**, mIoU = 0.295, over 96 epochs (the last 48 on the full 88k-image set). | PyTorch, CUDA/AMP, SLURM, OpenCV |
| [**Adaptive Chunk-Based Compression**](./adaptive-chunk-compression) | Integer codecs for search-engine posting lists (vByte, Elias gamma, bit packing), plus an adaptive scheme that picks the best codec for each chunk. | The adaptive codec placed **first or second on every data distribution tested**, and up to **77% smaller** than vByte. | Python, pytest (98 tests), uv |

---

## Vision Zero Crash Analysis
**Team of 3 · client: City of Lincoln · in progress**

<a href="./vision-zero-crash-analysis"><img src="./vision-zero-crash-analysis/figures/adt_excess_top20.png" alt="Intersections with the most crashes above what their traffic predicts" width="720"></a>

The City of Lincoln gave our team its full 2025 crash records. The question was whether its *Vision Zero Action Plan*, built on 2014–2023 data, still describes the city's streets. I wrote the analysis pipeline:

- **Hypothesis tests** (binomial and Poisson) of seven of the plan's headline statistics against 2025.
- **Safety performance functions** fit by maximum likelihood to a zero-truncated negative binomial, then **empirical Bayes** estimates to rank 925 intersections by excess crashes for their traffic volume.
- **Singular value decomposition** of crash conditions (weather, light, time of day) at the 47 highest-crash locations.
- An **[interactive crash map](https://coopermalone.github.io/project_portfolio/vision-zero-crash-analysis/figures/crash_map.html)** for City staff, filterable by severity, travel mode, date, light and time of day.

## Lane Marker Detection
**Individual project**

<a href="./lane-marker-detection"><img src="./lane-marker-detection/plots/metrics_overall.png" alt="SCNN validation metrics across epochs" width="560"></a>

- Implemented a Spatial CNN, which passes messages row by row and column by column so that thin, continuous lane lines survive occlusion.
- Trained in two phases on the university's GPU cluster with SLURM and mixed precision: 48 epochs on a 2k-image subset to debug the pipeline, then 48 more on the full 88k-image CULane training set.
- Analyzed overfitting on the small subset and the failure cases (windshield glare, occluding trucks).

## Adaptive Chunk-Based Compression
**Individual project**

- Implemented vByte, Elias gamma and bit-packing encoders and decoders from scratch, using bit-level buffered writes.
- Built an adaptive compressor that encodes each chunk of a posting list with every codec, keeps the smallest, and stores a codec ID. The ID costs about 1–2% in metadata.
- Benchmarked the codecs on Shakespeare, a newsgroups corpus and four synthetic gap distributions. Each codec wins somewhere; the adaptive one is the only codec that never finishes worse than second.

---

## Skills shown

- **Statistics:** hypothesis testing, count models (Poisson, negative binomial), maximum likelihood, empirical Bayes, SVD/PCA, multiple-comparison control
- **Machine learning:** CNN segmentation, transfer learning, mixed-precision training, model evaluation (mIoU, F1)
- **Engineering:** reproducible script pipelines, unit testing, HPC job scheduling, algorithm implementation at the bit level
- **Communication:** technical reports, interactive maps and charts for a non-technical client

## Contact

Cooper Malone · [coopercruz2004@gmail.com](mailto:coopercruz2004@gmail.com)
