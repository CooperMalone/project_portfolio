# Lane Marker Identification via Deep Learning

An end-to-end computer vision pipeline that implements a Spatial Convolutional Neural Network (SCNN) with a ResNet-18 backbone to perform robust semantic segmentation of continuous lane boundaries in real-world driving scenes.

---

## Project Overview

Detecting lanes in real-world environments is a foundational task for autonomous vehicle localization, path planning, and Advanced Driver-Assistance Systems (ADAS). However, standard encoder-decoder segmentation architectures often struggle to maintain the fine, continuous geometry of lines when faced with traffic occlusions, shadows, or harsh lighting conditions.

This project implements **Spatial CNN (SCNN)**, which addresses these limitations by introducing sequential message passing across spatial dimensions (rows and columns). This enables the network to propagate structural information across pixel boundaries, ensuring highly coherent lane boundary delineation.

### Key Achievements:
*   Processed and engineered data channels for the **88,000-image CULane dataset**.
*   Trained a semantic segmentation network using **Automatic Mixed Precision (AMP)** on a high-performance **GPU cluster**.
*   Configured and managed distributed training workflows via **SLURM batch scheduling**.
*   Achieved stable convergence and a validation **F1-score of 0.43**, maintaining lane continuity across diverse traffic and traffic conditions.

---

## Tech Stack & Tools

*   **Deep Learning Frameworks:** PyTorch / torchvision (ResNet-18 backbone)
*   **Computer Vision & Data Libraries:** OpenCV, NumPy, Scikit-Learn
*   **Infrastructure & MLOps:** SLURM Batch Scheduling, CUDA, Automatic Mixed Precision (AMP)
*   **Visualization:** Matplotlib, Seaborn
---

## Experimental Results & Model Evaluation

### Quantitative Metrics
The model was evaluated using standard semantic segmentation benchmarks, focusing on mean Intersection-over-Union (mIoU) and pixel-level F1-score.

| Phase / Configuration | Epochs | Dataset Scale | Validation mIoU | Validation F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Phase 1: Pipeline Validation** | 1–48 | Subset (~2k images) | 0.290 | 0.430 |
| **Phase 2: Full-Scale Run** | 49–96 | Full CULane (88k images) | **0.320** | **0.437** |

*Extended training through 96 epochs over a 2-week period demonstrated strong structural convergence, stabilizing validation metrics in the low-0.30s for mIoU and mid-0.40s for the F1-score.*

### Qualitative Insights
*   **Success Cases:** SCNN effectively preserves line topology and handles perspective scaling beautifully on standard highways and dense urban straights.
*   **Edge Cases & Limitations:** Model degradation was primarily observed in scenes with severe physical occlusions (e.g., large commercial vehicles completely covering markers) or high-exposure windshield glare which saturates pixel channels.

---

## Repository Structure

```text
├── datasets/
│   ├── __init__.py
│   └── tusimple.py        # Dataset loading and binarization logic
├── config.py              # Centralized hyperparameter and training config
├── train.py               # Main cluster training script with AMP support
├── eval.py                # Validation and metric evaluation loop
├── predict_single.py      # Single-frame inference and overlay script
├── run_train_48.sbatch    # SLURM cluster execution script
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
