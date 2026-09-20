"""
OC-SVM feature-based backdoor detection for ONNX model graphs.

Extracts 23-dimensional structural features from ONNX computational graphs and uses
One-Class SVM for detection. Only ONNX files are used.

Usage:
    python detect.py --project MalConvBase --mode feature --kernel rbf --nu 0.1 --n_folds 5
    python detect.py --project MalConvPlus --mode feature --kernel rbf --nu 0.1 --n_folds 5
    python detect.py --project RCNN --mode feature --kernel rbf --nu 0.1 --n_folds 5
"""

import argparse
import os
import sys
import numpy as np
from collections import defaultdict

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from project_config import get_data_dir, get_label_csv, get_project_config, get_family_filter, PROJECT_CONFIGS
from models import ModelDataset
from models.features import extract_graph_features, load_stealthiness, LEVEL_NAMES, FEAT_NAMES


class _Tee:
    def __init__(self, *files):
        self._files = files

    def write(self, data):
        for f in self._files:
            f.write(data)

    def flush(self):
        for f in self._files:
            f.flush()


def _run_detection(X, y, all_model_ids, samples_meta, cfg, args, stealth_map):
    """Core OC-SVM detection logic.

    Args:
        X: feature matrix (N, D)
        y: labels (N,) — 0=benign, 1=backdoor
        all_model_ids: list of model ID strings
        samples_meta: list of (path, family, sample_key) for test indices
        cfg: project config dict
        args: parsed CLI args
        stealth_map: dict mapping model IDs to stealth level strings
    """
    print(f"\n{'='*60}")
    print(f"OC-SVM Detection")
    print(f"{'='*60}")
    print(f"Models: {len(X)} (benign={int((y==0).sum())}, backdoor={int((y==1).sum())})")

    # NOTE: Standardization is fit ONLY on the benign training partition of each
    # split, then applied to train+test. Fitting on the full matrix (benign +
    # backdoor) before splitting leaks test/backdoor statistics into the scaler
    # mean/variance and systematically inflates detection rates.

    # Feature importance hints — only show discriminating features (ratio far from 1)
    print(f"\nKey discriminating features (ratio < 0.5 or > 1.5):")
    discriminating = []
    for j, name in enumerate(FEAT_NAMES):
        if j >= X.shape[1]:
            break
        b_mean = X[y == 0, j].mean() if (y == 0).sum() > 0 else 0
        m_mean = X[y == 1, j].mean() if (y == 1).sum() > 0 else 0
        ratio = m_mean / b_mean if b_mean != 0 else float('inf')
        if ratio < 0.5 or ratio > 1.5:
            discriminating.append((name, b_mean, m_mean, ratio))
    if discriminating:
        # Sort by how far ratio is from 1.0 (most discriminating first)
        discriminating.sort(key=lambda x: abs(x[3] - 1.0), reverse=True)
        for name, b_mean, m_mean, ratio in discriminating:
            print(f"  {name:20s}: benign={b_mean:10.2f}  backdoor={m_mean:10.2f}  ratio={ratio:.2f}")
    else:
        print("  (No strongly discriminating features found)")

    # Cross-validation
    if args.n_folds > 0:
        n_benign = int((y == 0).sum())
        n_backdoor = int((y == 1).sum())
        min_per_fold_benign = n_benign // args.n_folds
        min_per_fold_bd = n_backdoor // args.n_folds
        if min_per_fold_benign < 1 or min_per_fold_bd < 1:
            print(f"  Skipping cross-validation: too few samples "
                  f"(benign={n_benign}, backdoor={n_backdoor}, folds={args.n_folds})")
        else:
            print(f"\nCross-validation ({args.n_folds} folds):")
            aucs, accs, f1s = [], [], []
            tprs, tnrs, balaccs = [], [], []

            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                benign_train_mask = y[train_idx] == 0
                ocsvm_train_idx = train_idx[benign_train_mask]

                # Fit scaler on benign training fold only, then transform.
                fold_scaler = StandardScaler()
                fold_scaler.fit(X[ocsvm_train_idx])
                X_train = fold_scaler.transform(X[ocsvm_train_idx])
                X_test = fold_scaler.transform(X[test_idx])
                y_test = y[test_idx]

                clf = OneClassSVM(kernel=args.kernel, nu=args.nu, gamma=args.gamma)
                clf.fit(X_train)

                preds_raw = clf.predict(X_test)
                preds = (preds_raw == -1).astype(int)
                scores = -clf.decision_function(X_test)

                auc = roc_auc_score(y_test, scores)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="binary")
                aucs.append(auc)
                accs.append(acc)
                f1s.append(f1)
                # Paper-aligned metrics: TPR, TNR, BalAcc (see Sec. 4).
                tpr_f = (((y_test == 1) & (preds == 1)).sum() /
                         (y_test == 1).sum()) if (y_test == 1).sum() > 0 else 0.0
                tnr_f = (((y_test == 0) & (preds == 0)).sum() /
                         (y_test == 0).sum()) if (y_test == 0).sum() > 0 else 0.0
                balacc_f = (tpr_f + tnr_f) / 2.0
                tprs.append(tpr_f)
                tnrs.append(tnr_f)
                balaccs.append(balacc_f)

            print(f"  Avg AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
                  f"ACC={np.mean(accs):.4f}±{np.std(accs):.4f}  "
                  f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
            print(f"  Paper metrics: "
                  f"TPR={np.mean(tprs):.4f}±{np.std(tprs):.4f}  "
                  f"TNR={np.mean(tnrs):.4f}±{np.std(tnrs):.4f}  "
                  f"BalAcc={np.mean(balaccs):.4f}±{np.std(balaccs):.4f}")

    # Final train/test split
    print(f"\n{'='*60}")
    print(f"Final train/test evaluation")
    print(f"{'='*60}")

    benign_idx = np.where(y == 0)[0]
    backdoor_idx = np.where(y == 1)[0]

    n_train_benign = int(len(benign_idx) * args.train_ratio)
    perm = np.random.permutation(benign_idx)
    train_benign = perm[:n_train_benign]
    test_idx = np.concatenate([perm[n_train_benign:], backdoor_idx])

    # Fit scaler on benign training partition only, then transform train+test.
    final_scaler = StandardScaler()
    final_scaler.fit(X[train_benign])
    X_train = final_scaler.transform(X[train_benign])
    X_test = final_scaler.transform(X[test_idx])
    y_test = y[test_idx]
    test_ids = [all_model_ids[i] for i in test_idx]

    clf = OneClassSVM(kernel=args.kernel, nu=args.nu, gamma=args.gamma)
    clf.fit(X_train)

    preds_raw = clf.predict(X_test)
    preds = (preds_raw == -1).astype(int)
    scores = -clf.decision_function(X_test)

    auc = roc_auc_score(y_test, scores)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="binary")

    print(f"Train (benign): {len(train_benign)}, Test: {len(test_idx)} "
          f"({(y_test==0).sum()} benign + {(y_test==1).sum()} backdoor)")
    print(f"AUC={auc:.4f}, ACC={acc:.4f}, F1={f1:.4f}")
    benign_correct = ((y_test == 0) & (preds == 0)).sum()
    benign_total = (y_test == 0).sum()
    bd_detected = ((y_test == 1) & (preds == 1)).sum()
    bd_total = (y_test == 1).sum()
    b_prec = benign_correct / (y_test == 0).sum() if benign_total > 0 else 0
    b_rec = benign_correct / benign_total if benign_total > 0 else 0
    m_prec = bd_detected / (preds == 1).sum() if (preds == 1).sum() > 0 else 0
    m_rec = bd_detected / bd_total if bd_total > 0 else 0
    print(f"Benign: P={b_prec:.2f} R={b_rec:.2f}  Backdoor: P={m_prec:.2f} R={m_rec:.2f}")
    # Metrics aligned with the paper (Sec. 4): TPR (backdoor recall), TNR (benign
    # recall), and balanced accuracy BalAcc = (TPR + TNR) / 2. These are the
    # metrics reported in the main benchmark; AUC/ACC/F1 above are kept for
    # additional reference.
    tnr = benign_correct / benign_total if benign_total > 0 else 0.0
    tpr = bd_detected / bd_total if bd_total > 0 else 0.0
    balacc = (tpr + tnr) / 2.0
    print(f"Paper metrics: TPR={tpr:.4f}, TNR={tnr:.4f}, BalAcc={balacc:.4f}")

    # Per-model results — show all models, mark misclassifications
    print(f"\nPer-model results:")
    for j, mid in enumerate(test_ids):
        match = "OK" if preds[j] == y_test[j] else "MISS"
        stealth = stealth_map.get(str(mid), "?")
        if stealth == "?" and y_test[j] == 0:
            stealth = "L0"
        print(f"  {mid}: true={y_test[j]}, pred={preds[j]}, "
              f"stealth={stealth}, score={scores[j]:.4f} {match}")

    # Per-stealthiness-level detection statistics — only count backdoor models
    print(f"\nDetection statistics by stealthiness level (backdoor only):")

    level_total = defaultdict(int)
    level_detected = defaultdict(int)

    for j, mid in enumerate(test_ids):
        if y_test[j] == 1:
            stealth = stealth_map.get(str(mid), "?")
            level_total[stealth] += 1
            if preds[j] == 1:
                level_detected[stealth] += 1

    all_levels = sorted(set(level_total.keys()))
    if all_levels:
        print(f"  {'Level':8s}  {'Meaning':20s}  {'Total':>6s}  {'Detected':>8s}  {'Rate':>8s}")
        print(f"  {'-'*8}  {'-'*20}  {'-'*6}  {'-'*8}  {'-'*8}")
        for lv in all_levels:
            total = level_total[lv]
            detected = level_detected[lv]
            rate = detected / total if total > 0 else 0.0
            name = LEVEL_NAMES.get(lv, "")
            print(f"  {lv:8s}  {name:20s}  {total:6d}  {detected:8d}  {rate:8.2%}")
        grand_total = sum(level_total.values())
        grand_detected = sum(level_detected.values())
        grand_rate = grand_detected / grand_total if grand_total > 0 else 0.0
        print(f"  {'Total':8s}  {'':20s}  {grand_total:6d}  {grand_detected:8d}  {grand_rate:8.2%}")
    else:
        print("  (No backdoor models in test set)")

    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "n_benign_train": len(train_benign),
        "n_test": len(test_idx),
        "dr": bd_detected / bd_total if bd_total > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="OC-SVM feature-based backdoor detection")
    parser.add_argument("--project", type=str, default="MalConvBase",
                        choices=list(PROJECT_CONFIGS.keys()))
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="Override label CSV path")
    parser.add_argument("--mode", type=str, default="feature",
                        choices=["feature"],
                        help="Detection mode: structural features extracted from ONNX graphs")
    parser.add_argument("--kernel", type=str, default="rbf",
                        choices=["rbf", "linear", "poly"])
    parser.add_argument("--nu", type=float, default=0.1,
                        help="OC-SVM nu (upper bound on training outliers). "
                             "0.1 is the benchmark setting for feature mode (nu=0.3 is "
                             "used for hybrid mode, not available in this example).")
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    cfg = get_project_config(args.project)
    data_dir = args.data_dir or str(get_data_dir(args.project))
    label_csv = args.label_csv or str(get_label_csv(args.project))
    family_filter = get_family_filter(args.project)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, f"results_{args.project}.txt")

    with open(out_path, "w") as _out_file:
        sys.stdout = _Tee(sys.__stdout__, _out_file)
        try:
            np.random.seed(args.seed)

            stealth_map = load_stealthiness(label_csv, family_filter=family_filter)

            dataset = ModelDataset(
                data_dir=data_dir,
                label_csv=label_csv,
                suffixes=cfg["suffixes"],
                project=args.project,
                family_filter=family_filter,
            )
            print(f"Project: {args.project}")
            print(f"Total models: {len(dataset)}")

            print("Extracting graph features...")
            all_features = []
            all_labels = []
            all_model_ids = []

            for i in range(len(dataset)):
                data = dataset.get(i)
                _, _, sample_key = dataset._samples[i]
                feat = extract_graph_features(data)
                all_features.append(feat)
                all_labels.append(data.y.item())
                all_model_ids.append(sample_key)

            X = np.stack(all_features)
            y = np.array(all_labels)
            print(f"Feature matrix shape: {X.shape}")
            print(f"Benign: {(y == 0).sum()}, Backdoor: {(y == 1).sum()}")

            samples_meta = dataset._samples
            _run_detection(X, y, all_model_ids, samples_meta, cfg, args, stealth_map)
        finally:
            sys.stdout = sys.__stdout__

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
