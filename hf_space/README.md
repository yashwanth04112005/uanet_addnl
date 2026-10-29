---
title: UANet Remote Sensing Building Extraction
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# UANet: Uncertainty-Aware Network for Building Extraction from Remote Sensing Images

Interactive demonstration of **UANet** (IEEE Transactions on Geoscience and Remote Sensing 2024, IEEE GRSS Data Fusion Contest Champion).

### Key Capabilities Demonstrated:
- **Interactive Split Comparison**: Compare raw aerial satellite imagery with predicted building footprints.
- **Uncertainty Heatmap Visualization**: Inspect pixel-wise prediction uncertainty ($0.5 - |0.5 - \text{Sigmoid}(M)|$).
- **Multi-Level Progressive Refinement**: Explore how the Uncertainty-Aware Fusion Module (UAFM) refines feature maps from $M_5$ down to $M_1$.
- **Quantitative Benchmark Performance**: Full IoU (80.49%), F1-Score (89.19%), Precision, Recall, and Accuracy scores on WHU Building Dataset.

