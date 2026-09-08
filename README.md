# AMBER Dataset — Example Subset

> **Note:** This repository provides a size-limited example subset for review purposes only, not the complete dataset. It includes three of the seven architectures (all from the CRNN family) and the corresponding label rows, so that the data format, label schema, and detection pipeline can be inspected within GitHub's repository size constraints. The full AMBER dataset (seven architectures across four model families, 2,724 models, ~72 GB) follows the identical construction procedure and label schema demonstrated here, and will be released publicly on Hugging Face upon acceptance of the accompanying paper.

**AMBER** (**A**rchitecture **M**utation-based **B**ackdoor **E**valuation and **R**esearch) is an open-source benchmark dataset of backdoored deep neural network models targeting malware detection systems. It covers four model architecture families, seven concrete architectures, and 2,724 model samples annotated with a three-level stealthiness taxonomy (L0–L3).

This repository (`AMBER_example`) is a reduced example subset of the full AMBER dataset. It contains **three of the seven architectures** (all from the CRNN family) and the corresponding rows of the unified label file, together with a self-contained example detection pipeline. The full seven-architecture dataset (2,724 models, ~72 GB) is described in the project's main documentation.

## Example Subset Overview

| Item | Value |
|------|-------|
| Architectures included | 3 (MalConvBase, MalConvPlus, RCNN) |
| Architecture family | CRNN |
| Total models | 1,200 |
| Benign (L0) | 485 |
| Backdoored (L1–L3) | 715 |
| Model format | ONNX (.onnx) + PyTorch (.pt); detection pipeline uses ONNX only |
| Label file | `backdoor_dataset_all.csv` (filtered to these three architectures) |

### Stealthiness Level Distribution (subset)

| Level | Count | Description |
|-------|-------|-------------|
| L0 | 485 | Benign (no backdoor) |
| L1 | 244 | Explicit — interleaved any-integration (B=3), or shared+targeted (B=2 AND C=1); clear traces in code size / component count / data-flow |
| L2 | 234 | Medium-risk/Semi-covert — separate/shared propagation with untargeted integration; a single flaw in Detection or Propagation |
| L3 | 237 | Covert — operator trigger (A=2) + untargeted noise (C=2); optimal on both structural overhead and statistical audit camouflage |

### Architecture Breakdown (subset)

| Architecture | Family | Models | Benign (L0) | L1 | L2 | L3 |
|---|---|---|---|---|---|---|
| MalConvBase | CRNN | 424 | 176 | 83 | 82 | 83 |
| MalConvPlus | CRNN | 421 | 177 | 83 | 79 | 82 |
| RCNN | CRNN | 355 | 132 | 78 | 73 | 72 |
| **Total** | | **1,200** | **485** | **244** | **234** | **237** |

## Directory Structure

```
AMBER_example/
├── README.md
├── backdoor_dataset_all.csv       # Label file filtered to the three included architectures
│
├── MalConvBase/                   # ONNX + PyTorch model files (one per sample)
├── MalConvPlus/
├── RCNN/
│
└── Detect/                        # Example detection code (OC-SVM, --mode feature)
    ├── detect.py                  # Main detection script
    ├── project_config.py          # Path configuration (pre-configured for this directory)
    ├── requirements.txt           # Python dependencies
    └── models/
        ├── __init__.py
        ├── dataset.py             # ONNX → PyG graph dataset loader
        └── features.py            # 23-dimensional structural feature extractor
```

## Label File Format

`backdoor_dataset_all.csv` contains one row per model with the following columns (identical schema to the full dataset; here filtered to the three included architectures):

| Column | Description |
|--------|-------------|
| `sample_id` | Model ID; corresponds to ONNX filename (e.g., `0` → `model0.onnx`) |
| `label` | `0` = benign ($L_0$ control), `1` = backdoor ($L_1$–$L_3$) |
| `model_family` | High-level architecture family (`CRNN` for all rows in this subset) |
| `model_name` | Concrete architecture name (used as `family_filter` in detection code) |
| `param_scale` | Parameter scale category (`Small` / `Medium` / `Large`) |
| `detector_type` | Trigger detection type: `constant` / `operator` / `None` |
| `propagation_path` | Path coupling: `separated` / `shared` / `interleaved` / `None` |
| `attack_type` | Output manipulation type: `targeted` / `stochastic-untargeted` / `None` |
| `asr` | Attack success rate of the embedded backdoor |
| `accuracy` | Clean accuracy of the model |
| `precision` | Precision on clean inputs |
| `acc_triggered` | Accuracy when the trigger is present |
| `A_level` / `B_level` / `C_level` | Raw stealthiness scores per dimension (A∈{0,1,2}, B∈{0,1,2,3}, C∈{0,1,2}; 0=clean/benign). Three attacker-controlled config axes with cardinality 2/3/2 = 12 backdoor combinations |
| `stealthiness_level` | Composite stealthiness level: `L0` (benign) / `L1`–`L3` (backdoor) |

## Example Detection Code

The `Detect/` directory provides an implementation of **OC-SVM (`--mode feature`)**, a training-free backdoor detection method based on 23-dimensional structural features extracted from ONNX computational graphs. The detector is **one-class**: it trains only on benign models and flags anomalous models as backdoored. No pretraining or GPU is required.

This serves as a ready-to-run example for researchers who want to evaluate detection methods on the AMBER dataset.

### Environment Setup

```bash
conda create -n amber_detect python=3.10
conda activate amber_detect
pip install scikit-learn numpy onnx torch torch_geometric
```

### Run Detection

```bash
cd AMBER_example/Detect

# Detect backdoors in one architecture (5-fold cross-validation, results saved automatically)
python detect.py --project MalConvBase --mode feature --kernel rbf --nu 0.1 --n_folds 5
python detect.py --project MalConvPlus --mode feature --kernel rbf --nu 0.1 --n_folds 5
python detect.py --project RCNN --mode feature --kernel rbf --nu 0.1 --n_folds 5
```

Results are automatically saved to `Detect/results_{architecture}.txt`.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--project` | required | Architecture name (`MalConvBase` / `MalConvPlus` / `RCNN` in this subset) |
| `--mode` | `feature` | Structural features extracted from ONNX computational graphs |
| `--kernel` | `rbf` | SVM kernel |
| `--nu` | `0.1` | OC-SVM outlier fraction |
| `--n_folds` | `5` | Number of cross-validation folds |
| `--train_ratio` | `0.7` | Fraction of benign models used for training |
| `--gamma` | `scale` | SVM kernel coefficient (`scale`, `auto`, or float) |
| `--seed` | `42` | Random seed for reproducibility |