"""Graph feature extraction and label loading for OC-SVM detection."""

import csv
import numpy as np
import torch

# Stealthiness level names for per-level detection statistics
# FP-aligned taxonomy (Langford et al., arXiv 2402.06957): 2x3x2=12 configs, L1-L3
# Derived dimension D (old D=f(B,C)) removed; composite level decided by (A,B,C) risk table.
LEVEL_NAMES = {
    "L0": "Benign (clean model)",
    "L1": "High-risk/Explicit — interleaved any-goal (B=3) or shared+targeted (B=2,C=1); clear traces in code size / component count / data-flow",
    "L2": "Medium-risk/Semi-covert — separate/shared propagation with untargeted goal; a single flaw in Detection or Propagation",
    "L3": "Low-risk/Covert — operator trigger (A=2) + untargeted noise (C=2); optimal on both structural overhead and statistical audit camouflage",
}


def extract_graph_features(data) -> np.ndarray:
    """Extract hand-crafted structural features from a PyG Data object.

    Features: num_nodes, num_edges, avg_degree, density,
    node feature stats, graph-level weight statistics, structural ratios.
    """
    x = data.x.numpy()
    edge_index = data.edge_index.numpy()
    num_nodes = data.num_nodes
    num_edges = data.num_edges

    avg_degree = num_edges / max(num_nodes, 1)
    density = num_edges / max(num_nodes * (num_nodes - 1), 1)

    feat_mean = x.mean()
    feat_std = x.std() if x.size > 1 else 0.0
    feat_min = x.min()
    feat_max = x.max()
    feat_norm = np.linalg.norm(x)

    node_norms = np.linalg.norm(x, axis=1)
    node_norm_mean = node_norms.mean()
    node_norm_std = node_norms.std() if len(node_norms) > 1 else 0.0
    node_norm_max = node_norms.max()

    feat_var_across_nodes = x.var(axis=0).mean()

    if num_edges > 0:
        src, dst = edge_index
        out_degrees = np.bincount(src, minlength=num_nodes)[:num_nodes]
        in_degrees = np.bincount(dst, minlength=num_nodes)[:num_nodes]
        out_deg_mean = out_degrees.mean()
        out_deg_std = out_degrees.std() if len(out_degrees) > 1 else 0.0
        in_deg_mean = in_degrees.mean()
    else:
        out_deg_mean = out_deg_std = in_deg_mean = 0.0

    edge_node_ratio = num_edges / max(num_nodes, 1)

    stats_cols = x[:, :8]
    weight_mean = stats_cols[:, 0].mean()
    weight_std = stats_cols[:, 1].mean()
    weight_L2 = stats_cols[:, 4].mean()

    n_special = (x[:, 1] == -1.0).sum()
    n_weighted = num_nodes - n_special
    special_ratio = n_special / max(num_nodes, 1)

    return np.array([
        num_nodes, num_edges, avg_degree, density,
        feat_mean, feat_std, feat_min, feat_max, feat_norm,
        node_norm_mean, node_norm_std, node_norm_max,
        feat_var_across_nodes, out_deg_mean, out_deg_std, in_deg_mean,
        edge_node_ratio, weight_mean, weight_std, weight_L2,
        n_weighted, n_special, special_ratio,
    ], dtype=np.float32)


FEAT_NAMES = [
    "num_nodes", "num_edges", "avg_degree", "density",
    "feat_mean", "feat_std", "feat_min", "feat_max", "feat_norm",
    "node_norm_mean", "node_norm_std", "node_norm_max",
    "feat_var", "out_deg_mean", "out_deg_std", "in_deg_mean",
    "edge_node_ratio", "weight_mean", "weight_std", "weight_L2",
    "n_weighted", "n_special", "special_ratio",
]


def load_labels(csv_path: str) -> dict:
    """Load model_id -> label mapping from CSV. 0=benign, 1=backdoor."""
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row["sample_id"].strip()
            label_str = row["label"].strip()
            if not model_id or not label_str:
                continue
            labels[model_id] = int(label_str)
    return labels


def load_stealthiness(csv_path: str, family_filter: str = None) -> dict:
    """Load model_id -> stealth_level mapping from CSV. Returns dict with string keys.

    If family_filter is provided, only rows whose `model_name` column
    matches the filter are loaded (for per-architecture projects).
    """
    stealth_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row["sample_id"].strip()
            if not model_id:
                continue
            if family_filter:
                arch = row.get("model_name", "").strip()
                if arch != family_filter:
                    continue
            stealth = row["stealthiness_level"].strip()
            stealth_map[model_id] = stealth
    return stealth_map