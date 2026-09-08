"""
ONNX/PT model -> PyG Graph Dataset for backdoor detection.

Extracts computational graph structure from ONNX files and converts
each model into a PyG Data object with:
  - Node features: compressed weight statistics (fixed-dim vectors)
  - Edges: data-flow connections between operators
  - Labels: benign (0) or backdoor (1) from backdoor_dataset_all.csv

Each project is a single-family directory under the AMBER_example root.
"""

import os
import re
import csv
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

try:
    import onnx
    from onnx import numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# ---------- weight compression utilities ----------

FEATURE_DIM = 128

OP_TYPE_MAP = {
    "Constant": 0, "Clip": 1, "Cast": 2, "Gather": 3,
    "Transpose": 4, "Slice": 5, "Conv": 6, "Sigmoid": 7,
    "Mul": 8, "MaxPool": 9, "Flatten": 10, "Gemm": 11,
    "Add": 12, "Sub": 13, "Div": 14, "Equal": 15,
    "Greater": 16, "GreaterOrEqual": 17, "And": 18,
    "ReduceSum": 19, "ReduceMax": 20, "Unsqueeze": 21,
    "Relu": 22, "Abs": 23, "RandomUniformLike": 24,
    "RandomNormalLike": 25, "Reshape": 26, "Concat": 27,
    "Pad": 28, "Softmax": 29, "Squeeze": 30,
    "LSTM": 31, "Erf": 32, "MatMul": 33, "Where": 34,
    "ReduceMean": 35, "ConstantOfShape": 36, "Shape": 37,
    "Identity": 38, "Pow": 39, "Sqrt": 40,
}


def compress_weight(tensor: np.ndarray, target_dim: int = FEATURE_DIM) -> torch.Tensor:
    """Compress an arbitrary-shape weight tensor into a fixed-dim feature vector."""
    flat = tensor.flatten().astype(np.float32)
    stats = np.array([
        flat.mean(),
        flat.std() if flat.size > 1 else 0.0,
        flat.min(),
        flat.max(),
        np.linalg.norm(flat),
        np.percentile(flat, 25) if flat.size > 0 else 0.0,
        np.percentile(flat, 50) if flat.size > 0 else 0.0,
        np.percentile(flat, 75) if flat.size > 0 else 0.0,
    ], dtype=np.float32)

    remaining = target_dim - len(stats)
    if remaining <= 0:
        return torch.from_numpy(stats[:target_dim])

    if flat.size >= remaining:
        indices = np.linspace(0, flat.size - 1, remaining, dtype=int)
        sampled = flat[indices]
    else:
        sampled = np.zeros(remaining, dtype=np.float32)
        sampled[:flat.size] = flat

    feat = np.concatenate([stats, sampled])
    return torch.from_numpy(feat[:target_dim])


def compress_empty_node(op_type: str, target_dim: int = FEATURE_DIM) -> torch.Tensor:
    """Feature vector for a node that has no learnable weight."""
    feat = torch.zeros(target_dim, dtype=torch.float32)
    type_id = OP_TYPE_MAP.get(op_type, len(OP_TYPE_MAP))
    feat[0] = float(type_id)
    feat[1] = -1.0  # marker: no weight
    return feat


# ---------- ONNX graph extraction ----------

class ONNXGraphExtractor:
    """Extract a computational graph from an ONNX model file."""

    SKIP_OPS = {"Constant"}
    BIAS_SUFFIX = (".bias",)
    # Small initializers (scalar thresholds, tiny detection kernels, etc.)
    # are not real "learnable weights" and pollute weight statistics.
    # Treat them as empty nodes instead.
    MIN_WEIGHT_SIZE = 10

    def __init__(self, onnx_path: str):
        if not HAS_ONNX:
            raise RuntimeError("onnx package is required. Install with: pip install onnx")
        self.model = onnx.load(onnx_path)
        self.path = onnx_path

        self.init_dict = {}
        for init in self.model.graph.initializer:
            self.init_dict[init.name] = numpy_helper.to_array(init)

        self.output_to_node = {}
        for idx, node in enumerate(self.model.graph.node):
            for out_name in node.output:
                self.output_to_node[out_name] = idx

    def extract(self) -> dict:
        """Extract graph components."""
        nodes = list(self.model.graph.node)
        kept_indices = [i for i, n in enumerate(nodes) if n.op_type not in self.SKIP_OPS]
        onnx_idx_to_graph_idx = {}
        for gid, onnx_idx in enumerate(kept_indices):
            onnx_idx_to_graph_idx[onnx_idx] = gid

        node_features = []
        node_types = []

        for onnx_idx in kept_indices:
            node = nodes[onnx_idx]
            weight_found = False
            for inp_name in node.input:
                if inp_name in self.init_dict:
                    w = self.init_dict[inp_name]
                    if not inp_name.endswith(self.BIAS_SUFFIX):
                        flat = w.flatten()
                        # Skip small initializers (scalar thresholds, detection
                        # kernels, etc.) — they are not real learnable weights
                        # and their extreme values pollute weight statistics.
                        if flat.size < self.MIN_WEIGHT_SIZE:
                            continue
                        feat = compress_weight(w)
                        node_features.append(feat)
                        node_types.append(node.op_type)
                        weight_found = True
                        break

            if not weight_found:
                feat = compress_empty_node(node.op_type)
                node_features.append(feat)
                node_types.append(node.op_type)

        edges = []
        for onnx_idx in kept_indices:
            node = nodes[onnx_idx]
            dst_gid = onnx_idx_to_graph_idx[onnx_idx]
            for inp_name in node.input:
                if inp_name in self.output_to_node:
                    src_onnx_idx = self.output_to_node[inp_name]
                    if src_onnx_idx in onnx_idx_to_graph_idx:
                        src_gid = onnx_idx_to_graph_idx[src_onnx_idx]
                        if src_gid != dst_gid:
                            edges.append((src_gid, dst_gid))

        return {
            "node_features": node_features,
            "node_types": node_types,
            "edges": edges,
        }



# ---------- PyG Dataset ----------


class ModelDataset(Dataset):
    """Dataset converting ONNX models to PyG graph data.

    Each project is a single-family directory under the AMBER_example root.
    """

    def __init__(self, data_dir: str, label_csv: str = None,
                 suffixes: list = None, project: str = None,
                 feature_dim: int = FEATURE_DIM,
                 family_filter: str = None):
        super().__init__()
        self.data_dir = data_dir
        self.feature_dim = feature_dim
        self.project = project or ""
        self.suffixes = suffixes or [".onnx"]
        self.family_filter = family_filter
        self._cache = {}  # idx -> Data, avoids re-parsing ONNX every epoch

        # Scan for model files. Only .onnx files are usable (ONNXGraphExtractor
        # cannot parse .pt/.pth). Deduplicate: prefer .onnx when both exist.
        # Order by numeric model ID (model2 before model10), with file-path
        # lexicographic tiebreak so ordering is deterministic and reproducible
        # regardless of filesystem inode order. Files whose ID cannot be parsed
        # as an integer sort last (stable), keeping numeric IDs contiguous.
        base_to_path = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith(".onnx"):
                continue
            path = os.path.join(data_dir, f)
            base = self._extract_base_name(f)
            base_to_path[base] = path

        def _sort_key(path):
            mid = self._get_model_id(path)
            try:
                return (0, int(mid), path)
            except (ValueError, TypeError):
                return (1, 0, path)

        all_files = sorted(base_to_path.values(), key=_sort_key)

        self.data_path = all_files
        self._samples = [(p, None, self._get_model_id(p)) for p in all_files]

        # Load labels (string keys)
        if label_csv and os.path.exists(label_csv):
            self.labels = {}
            with open(label_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model_id = row["sample_id"].strip()
                    label_str = row["label"].strip()
                    if not model_id or not label_str:
                        continue
                    if self.family_filter:
                        arch = row.get("model_name", "").strip()
                        if arch != self.family_filter:
                            continue
                    label = int(label_str)
                    self.labels[model_id] = label
        else:
            self.labels = None

    def _extract_base_name(self, filename: str) -> str:
        """Extract the model base name for deduplication.

        Examples: model0.onnx -> model0, model13.pt -> model13
        """
        # Remove known suffixes
        for s in (".onnx", ".pt", ".pth"):
            if filename.endswith(s):
                return filename[:-len(s)]
        return filename

    def _get_model_id(self, path: str) -> str:
        """Extract model ID from filename.

        Returns string ID.
        """
        basename = os.path.basename(path)

        # Default: model{N}.onnx
        m = re.match(r"^model(\d+)\.(?:onnx|pt)$", basename)
        if m:
            return m.group(1)

        # Fallback: first sequence of digits
        m = re.search(r'(\d+)', basename)
        if m:
            return m.group(1)
        return "-1"

    def _get_label(self, sample_key: str) -> int:
        """Get label for a model. 0=benign, 1=backdoor."""
        if self.labels:
            if sample_key in self.labels:
                return self.labels[sample_key]
        return 0  # default benign

    def get(self, idx: int) -> Data:
        if idx in self._cache:
            return self._cache[idx]

        path, _, sample_key = self._samples[idx]
        extractor = ONNXGraphExtractor(path)
        info = extractor.extract()

        x = torch.stack(info["node_features"])

        if info["edges"]:
            edge_index = torch.tensor(info["edges"], dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        y = torch.LongTensor([self._get_label(sample_key)])

        data = Data(x=x, edge_index=edge_index, y=y)
        self._cache[idx] = data
        return data

    def len(self) -> int:
        return len(self._samples)
