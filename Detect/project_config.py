"""
Project configuration for OC-SVM (feature) backdoor detection on the AMBER
example subset.

Data layout (relative to this file):
  ../                        ← AMBER_example root
  ├── backdoor_dataset_all.csv
  ├── MalConvBase/           ← ONNX files for each architecture
  ├── MalConvPlus/
  └── RCNN/

This is a reduced example subset of the full AMBER dataset, containing three
of the seven architectures (all CRNN family). See ../README.md for details.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # AMBER_example/
MODEL_OUTPUT_DIR = BASE_DIR
MODEL_OUTPUT_CSV = MODEL_OUTPUT_DIR / "backdoor_dataset_all.csv"

PROJECT_CONFIGS = {
    "MalConvBase": {
        "root": MODEL_OUTPUT_DIR / "MalConvBase",
        "data_subdir": ".",
        "suffixes": [".onnx"],
        "label_csv": str(MODEL_OUTPUT_CSV),
        "family_filter": "MalConvBase",
    },
    "MalConvPlus": {
        "root": MODEL_OUTPUT_DIR / "MalConvPlus",
        "data_subdir": ".",
        "suffixes": [".onnx"],
        "label_csv": str(MODEL_OUTPUT_CSV),
        "family_filter": "MalConvPlus",
    },
    "RCNN": {
        "root": MODEL_OUTPUT_DIR / "RCNN",
        "data_subdir": ".",
        "suffixes": [".onnx"],
        "label_csv": str(MODEL_OUTPUT_CSV),
        "family_filter": "RCNN",
    },
}


def get_project_config(project: str) -> dict:
    if project not in PROJECT_CONFIGS:
        raise KeyError(
            f"Unknown project '{project}'. "
            f"Available: {', '.join(PROJECT_CONFIGS.keys())}"
        )
    return PROJECT_CONFIGS[project]


def get_data_dir(project: str) -> Path:
    cfg = get_project_config(project)
    return cfg["root"] / cfg["data_subdir"]


def get_label_csv(project: str) -> Path:
    cfg = get_project_config(project)
    return cfg["root"] / cfg["label_csv"]


def get_family_filter(project: str):
    cfg = get_project_config(project)
    return cfg.get("family_filter")
