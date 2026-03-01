"""
Step 5: MLP Deep Learning baseline
===================================

Non-linear baseline using a 3-layer MLP (PyTorch).
Uses the best preprocessing from step4: Z-score per region + region means.

Grid: 2 atlases (S200, S300) x 3 activations (ReLU, GELU, LeakyReLU)
       x 3 architectures (shallow/medium/deep)
       x 6 labels (Sex, Age, BMI, BPDiastolic, Education, Race)

Same 5-fold CV as step4 for fair comparison.

Author: Dan Abergel
Master's Thesis - Hebrew University of Jerusalem
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")
CLEAN_SUBJECTS_JSON = HCP_ROOT / "subjects_clean.json"
HCP_SUBJECTS_CSV = HCP_ROOT / "metadata" / "HCP_YA_subjects.csv"

SCHAEFER_REGIONS = [200, 300]

N_SPLITS = 5
RANDOM_STATE = 42

# MLP hyperparameters
ARCHITECTURES = {
    "Shallow [256]":      [256],
    "Medium [512,256]":   [512, 256],
    "Deep [512,256,128]": [512, 256, 128],
}
DROPOUT = 0.3
BATCH_SIZE = 64
MAX_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
EARLY_STOP_PATIENCE = 20

ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "LeakyReLU": nn.LeakyReLU,
}

LABELS = {
    "Sex": {
        "column": "Gender",
        "type": "classification",
        "transform": lambda x: 1 if x == "M" else 0,
        "class_names": {1: "Male", 0: "Female"},
    },
    "Age": {
        "column": "Age_in_Yrs",
        "type": "regression",
        "transform": float,
    },
    "BMI": {
        "column": "BMI",
        "type": "regression",
        "transform": float,
    },
    "BPDiastolic": {
        "column": "BPDiastolic",
        "type": "regression",
        "transform": float,
    },
    "Education": {
        "column": "SSAGA_Educ",
        "type": "regression",
        "transform": float,
    },
    "Race": {
        "column": "Race",
        "type": "classification",
        "transform": lambda x: 1 if x == "White" else 0,
        "class_names": {1: "White", 0: "Non-White"},
    },
}

# Step4 best baselines (Z-score per region + means)
STEP4_BASELINES = {
    "Sex": {"metric": "ROC-AUC", "best": 0.908, "atlas": "S200"},
    "Age": {"metric": "MAE", "best": 3.129, "atlas": "S300"},
    "BMI": {"metric": "MAE", "best": 3.532, "atlas": "S300"},
    "BPDiastolic": {"metric": "MAE", "best": 9.253, "atlas": "S300"},
    "Education": {"metric": "MAE", "best": 1.559, "atlas": "S300"},
    "Race": {"metric": "ROC-AUC", "best": 0.753, "atlas": "S200"},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# NORMALIZATION + FEATURE EXTRACTION
# =============================================================================

def extract_cov_upper(ts: np.ndarray) -> np.ndarray:
    """Extract upper triangle of the covariance matrix."""
    cov = np.cov(ts, rowvar=False)
    idx = np.triu_indices(cov.shape[0], k=1)
    return cov[idx].astype(np.float32)


def normalize_and_extract(ts: np.ndarray) -> np.ndarray:
    """Z-score per region + region means (best condition from step4)."""
    region_means = ts.mean(axis=0).astype(np.float32)
    mean = ts.mean(axis=0, keepdims=True)
    std = ts.std(axis=0, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    ts_norm = (ts - mean) / std
    return np.concatenate([extract_cov_upper(ts_norm), region_means])


# =============================================================================
# DATA LOADING
# =============================================================================

def load_subject(subject_id: str, n_regions: int) -> np.ndarray | None:
    """Load Schaefer time series for one subject."""
    npy_path = (
        HCP_ROOT / f"subject_{subject_id}" / "MNINonLinear" / "Results"
        / "rfMRI_REST1_LR" / f"rfMRI_REST1_LR_schaefer{n_regions}.npy"
    )
    if not npy_path.exists():
        return None
    return np.load(npy_path)


def load_dataset(n_regions: int, verbose: bool = True):
    """Load clean subjects and all labels from HCP CSV."""
    hcp_df = pd.read_csv(HCP_SUBJECTS_CSV)
    hcp_df["Subject"] = hcp_df["Subject"].astype(str)
    hcp_df = hcp_df.set_index("Subject")

    with open(CLEAN_SUBJECTS_JSON, "r") as f:
        clean_data = json.load(f)
    clean_ids = clean_data["subject_ids"]

    subjects = []
    rows = []

    iterator = tqdm(clean_ids, desc=f"Loading Schaefer {n_regions}") if verbose else clean_ids

    for subject_id in iterator:
        ts = load_subject(subject_id, n_regions)
        if ts is None:
            continue
        if subject_id not in hcp_df.index:
            continue

        subjects.append((subject_id, ts))
        rows.append(hcp_df.loc[subject_id])

    labels_df = pd.DataFrame(rows)
    return subjects, labels_df


def get_label_array(labels_df: pd.DataFrame, label_name: str):
    """Extract and transform a label column, dropping NaN subjects."""
    label_cfg = LABELS[label_name]
    col = label_cfg["column"]

    valid_indices = []
    y_list = []

    for i, (_, row) in enumerate(labels_df.iterrows()):
        val = row.get(col)
        if pd.isna(val):
            continue
        try:
            y_list.append(label_cfg["transform"](val))
            valid_indices.append(i)
        except (ValueError, TypeError):
            continue

    return valid_indices, np.array(y_list)


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class BrainDataset(Dataset):
    """Simple dataset wrapping numpy feature and label arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# MLP MODEL
# =============================================================================

class MLP(nn.Module):
    """MLP with batch norm and dropout (variable depth)."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float,
                 activation_cls):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                activation_cls(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# TRAINING
# =============================================================================

def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_type: str,
    activation_name: str,
    hidden_dims: list,
) -> dict:
    """Train MLP on one fold with early stopping. Return val score."""

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = BrainDataset(X_train, y_train)
    val_ds = BrainDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    activation_cls = ACTIVATIONS[activation_name]
    model = MLP(input_dim, hidden_dims, DROPOUT, activation_cls).to(DEVICE)

    if label_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # --- Train ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                val_losses.append(criterion(preds, y_batch).item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

    # --- Evaluate best model ---
    model.load_state_dict(best_state)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    if label_type == "classification":
        probs = 1.0 / (1.0 + np.exp(-preds))  # sigmoid
        score = roc_auc_score(targets, probs)
    else:
        score = mean_absolute_error(targets, preds)

    return {
        "score": score,
        "best_epoch": epoch - patience_counter + 1,
        "total_epochs": epoch + 1,
    }


# =============================================================================
# EVALUATION (5-fold CV)
# =============================================================================

def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    label_type: str,
    activation_name: str,
    hidden_dims: list,
) -> dict:
    """Run 5-fold CV for one configuration."""

    if label_type == "classification":
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=N_SPLITS, shuffle=True,
                   random_state=RANDOM_STATE)

    fold_scores = []
    fold_epochs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        result = train_one_fold(
            X_train, y_train, X_val, y_val,
            label_type, activation_name, hidden_dims,
        )
        fold_scores.append(result["score"])
        fold_epochs.append(result["total_epochs"])

    scores = np.array(fold_scores)
    return {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
        "avg_epochs": np.mean(fold_epochs),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(all_results: list, output_dir: Path):
    """Generate one heatmap per label (arch x activation rows, atlas columns)."""

    labels_in_results = list(dict.fromkeys(r["label"] for r in all_results))

    for label_name in labels_in_results:
        label_results = [r for r in all_results if r["label"] == label_name]
        label_type = LABELS[label_name]["type"]
        metric_name = "MAE" if label_type == "regression" else "ROC-AUC"
        cmap = "RdYlGn_r" if label_type == "regression" else "RdYlGn"

        atlas_sizes = sorted(set(r["n_regions"] for r in label_results))
        arch_names = list(dict.fromkeys(r["arch"] for r in label_results))
        act_names = list(dict.fromkeys(r["activation"] for r in label_results))

        # Rows: (arch, activation) combos
        row_labels = []
        for arch in arch_names:
            for act in act_names:
                row_labels.append(f"{arch} / {act}")

        matrix = np.zeros((len(row_labels), len(atlas_sizes)))
        for r in label_results:
            row_key = f"{r['arch']} / {r['activation']}"
            i = row_labels.index(row_key)
            j = atlas_sizes.index(r["n_regions"])
            matrix[i, j] = r["mean"]

        fig, ax = plt.subplots(figsize=(8, max(4, len(row_labels) * 0.5 + 1)))
        im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                       vmin=matrix.min() - 0.02, vmax=matrix.max() + 0.02)

        ax.set_xticks(range(len(atlas_sizes)))
        ax.set_xticklabels([f"Schaefer {n}" for n in atlas_sizes])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)

        for i in range(len(row_labels)):
            for j in range(len(atlas_sizes)):
                val = matrix[i, j]
                color = "white" if val < matrix.mean() else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")

        baseline = STEP4_BASELINES[label_name]
        ax.set_title(
            f"{label_name} — MLP ({metric_name})\n"
            f"Step4 best: {baseline['best']:.3f} ({baseline['atlas']})",
            fontsize=12, fontweight="bold",
        )
        plt.colorbar(im, ax=ax, label=metric_name)
        plt.tight_layout()

        path = output_dir / f"mlp_{label_name.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    print("=" * 70)
    print("STEP 5: MLP DEEP LEARNING BASELINE")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Atlas resolutions: {SCHAEFER_REGIONS}")
    print(f"Architectures: {list(ARCHITECTURES.keys())}")
    print(f"Activations: {list(ACTIVATIONS.keys())}")
    print(f"Labels: {list(LABELS.keys())}")
    print(f"Dropout: {DROPOUT}, LR: {LR}, Batch: {BATCH_SIZE}")
    print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
    n_combos = (len(SCHAEFER_REGIONS) * len(ARCHITECTURES)
                * len(ACTIVATIONS) * len(LABELS))
    print(f"Total combinations: {n_combos}")

    all_results = []
    start_time = time.time()

    for n_regions in SCHAEFER_REGIONS:
        print(f"\n{'='*70}")
        print(f"SCHAEFER {n_regions} REGIONS")
        print(f"{'='*70}")

        subjects, labels_df = load_dataset(n_regions, verbose=True)

        if len(subjects) == 0:
            print(f"  No subjects found for Schaefer {n_regions}")
            continue

        print(f"Subjects loaded: {len(subjects)}")

        # Pre-extract features for this atlas (shared across all configs)
        print("Extracting features (Z-score per region + means)...")
        X_all = np.stack([
            normalize_and_extract(ts) for _, ts in
            tqdm(subjects, desc="Features")
        ])
        n_bad = np.isnan(X_all).sum() + np.isinf(X_all).sum()
        if n_bad > 0:
            print(f"  WARNING: {n_bad} NaN/Inf values - replacing with 0")
            X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Feature matrix: {X_all.shape}")

        for label_name, label_cfg in LABELS.items():
            valid_idx, y = get_label_array(labels_df, label_name)

            if len(y) < 50:
                print(f"\n  Skipping {label_name}: only {len(y)} valid subjects")
                continue

            X = X_all[valid_idx]
            label_type = label_cfg["type"]
            metric_name = "MAE" if label_type == "regression" else "ROC-AUC"
            baseline = STEP4_BASELINES[label_name]

            print(f"\n  --- {label_name} ({label_type}, n={len(y)}) ---")
            print(f"      Step4 baseline: {baseline['best']:.3f} {baseline['metric']}"
                  f" ({baseline['atlas']})")

            for arch_name, hidden_dims in ARCHITECTURES.items():
                print(f"    [{arch_name}]")
                for act_name in ACTIVATIONS:
                    t0 = time.time()
                    result = evaluate_model(X, y, label_type, act_name,
                                            hidden_dims)
                    elapsed = time.time() - t0

                    result["n_regions"] = n_regions
                    result["arch"] = arch_name
                    result["activation"] = act_name
                    result["label"] = label_name
                    result["n_features"] = X.shape[1]
                    all_results.append(result)

                    # Compare with step4
                    diff = result["mean"] - baseline["best"]
                    if label_type == "regression":
                        better = diff < 0
                    else:
                        better = diff > 0
                    marker = "^" if better else "v"

                    print(f"      {act_name:<12} {metric_name}:"
                          f" {result['mean']:.3f} +/- {result['std']:.3f}"
                          f"  [{diff:+.3f} {marker}]"
                          f"  (avg {result['avg_epochs']:.0f} epochs,"
                          f" {elapsed:.0f}s)")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total training time: {total_time/60:.1f} min")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY — MLP vs Step4 Linear Baselines")
    print(f"{'='*70}")

    for label_name in LABELS:
        label_results = [r for r in all_results if r["label"] == label_name]
        if not label_results:
            continue

        label_type = LABELS[label_name]["type"]
        metric_name = "MAE" if label_type == "regression" else "ROC-AUC"
        lower_is_better = label_type == "regression"
        results_sorted = sorted(label_results, key=lambda x: x["mean"],
                                reverse=not lower_is_better)
        baseline = STEP4_BASELINES[label_name]

        print(f"\n  {label_name} ({metric_name})"
              f"  |  Step4 best: {baseline['best']:.3f} ({baseline['atlas']})")
        print(f"  {'Rank':<5} {'Atlas':<6} {'Architecture':<22}"
              f" {'Activation':<12} {'Score':<18} {'vs Step4'}")
        print(f"  {'-'*85}")

        for i, r in enumerate(results_sorted):
            score_str = f"{r['mean']:.3f} +/- {r['std']:.3f}"
            diff = r["mean"] - baseline["best"]
            if lower_is_better:
                better = diff < 0
            else:
                better = diff > 0
            marker = ">>>" if i == 0 else "   "
            flag = "^" if better else "v"
            print(f"  {marker}{i+1:<2} S{r['n_regions']:<5}"
                  f" {r['arch']:<22} {r['activation']:<12}"
                  f" {score_str:<18} {diff:+.3f} {flag}")

    # =========================================================================
    # SAVE JSON
    # =========================================================================
    output_path = HCP_ROOT / "mlp_results.json"
    output_data = {
        "description": "Step 5 - MLP deep learning baseline",
        "preprocessing": "Z-score per region + region means",
        "architectures": {k: v for k, v in ARCHITECTURES.items()},
        "hyperparameters": {
            "dropout": DROPOUT,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "scheduler_factor": SCHEDULER_FACTOR,
        },
        "device": str(DEVICE),
        "schaefer_regions": SCHAEFER_REGIONS,
        "activations": list(ACTIVATIONS.keys()),
        "labels": list(LABELS.keys()),
        "step4_baselines": STEP4_BASELINES,
        "results": [
            {
                "n_regions": r["n_regions"],
                "arch": r["arch"],
                "activation": r["activation"],
                "label": r["label"],
                "mean": float(r["mean"]),
                "std": float(r["std"]),
                "fold_scores": r["scores"].tolist(),
                "n_features": r["n_features"],
                "avg_epochs": float(r["avg_epochs"]),
            }
            for r in all_results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved: {output_path}")

    # =========================================================================
    # PLOTS
    # =========================================================================
    print("\nGenerating plots...")
    plot_results(all_results, HCP_ROOT)

    print("\nDone.")


if __name__ == "__main__":
    main()
