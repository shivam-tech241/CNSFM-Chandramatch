# CNSFM-ChandraMatch

Standalone research reproduction and transfer experiment of the lunar feature matching method:

> **"Robust Feature Matching of Multi-Illumination Lunar Orbiter Images Based on Crater Neighborhood Structure"**  
> *Bin Xie, Bin Liu, Kaichang Di, Wai-Chung Liu, Yuke Kou, Yutong Jia, Yifan Zhang* (Remote Sensing, 2025).

---

## 1. Research Objective

This project evaluates whether **CNSFM** (Crater Neighborhood Structure Feature Matching) can provide reliable crater correspondences under severe cross-sensor, cross-resolution, and multi-illumination conditions between Chandrayaan-2 imagery:
- **OHRC**: ~0.25 m/pixel (calibrated, 8-bit)
- **TMC-2**: ~5.40 m/pixel (calibrated, 16-bit little-endian)
- **Illumination Condition**: ~160° solar azimuth difference, ~35° incidence difference

**Core Question:**  
*Does CNSFM produce sufficiently reliable crater correspondences on actual Chandrayaan-2 OHRC ↔ TMC-2 imagery to justify future integration into lunar registration pipelines?*

---

## 2. Project Architecture

```
CNSFM-ChandraMatch/
├── configs/               # Experiment configurations (YAML)
│   └── default.yaml
├── data/                  # Local data markers / documentation (raw images are external)
├── experiments/           # Experiment definitions and run scripts
├── results/               # Saved run logs, metrics, correspondence outputs
├── src/
│   ├── io/                # Region-based and memory-efficient image loaders
│   ├── preprocessing/     # Radiometric scaling, normalizations
│   ├── crater_detection/  # Detection interfaces and implementations
│   ├── cnsf/              # Crater Neighborhood Structure Feature construction
│   ├── matching/          # Similarity metrics and correspondence search
│   ├── mcr/               # Mismatched CNSF Removal (MCR)
│   ├── registration/      # Verification and transformation estimation
│   ├── evaluation/        # Quantitative evaluation metrics
│   └── utils/             # Config parsing, logging, and geometry utilities
├── tests/                 # Unit and regression tests
├── main.py                # Pipeline driver
├── requirements.txt       # Dependencies
└── .gitignore
```

---

## 3. External Data References

Large lunar orbital images are **never** stored inside this repository. Data paths are managed through configuration (`configs/default.yaml`):

- **Root Dataset**: `D:\ChandraData\triplet_1\`
  - **OHRC**: `D:\ChandraData\triplet_1\OHRC\data\calibrated\20210405\`
  - **TMC-2**: `D:\ChandraData\triplet_1\TMC\data\calibrated\20250807\`
  - *(IIRS is deferred to future work)*

---

## 4. Incremental Roadmap

1. **Chunk 1: Project Skeleton & Configuration System** (Current)
2. **Chunk 2: Memory-Efficient Data I/O & Region Crop Loader**
3. **Chunk 3: Crater Detection Interface & Adaptation**
4. **Chunk 4: CNSF Construction (K-Nearest Neighbors & Descriptors)**
5. **Chunk 5: CNSF Similarity Matching**
6. **Chunk 6: Mismatched CNSF Removal (MCR)**
7. **Chunk 7: Verification & Transformation Estimation**
8. **Chunk 8: Quantitative Evaluation & Metrics Reporting**
