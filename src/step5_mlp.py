"""
Step 5: MLP Deep Learning baseline
===================================

Non-linear baseline using a multi-layer MLP (PyTorch).
Compares 4 feature extraction strategies from Schaefer atlas:
  1. Raw time series  — flattened T x R values
  2. Means only       — R region means
  3. Cov only         — upper triangle of covariance matrix
  4. Cov + means      — upper triangle + region means

Grid: 2 atlases (S200, S300) x 4 feature sets x 3 architectures x 3 activations
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

# Feature conditions to compare
FEATURE_CONDITIONS = [
    "Raw time series",
    "Means only",
    "Cov only",
    "Cov + means",
]

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

# Step4 best baselines (Z-score per region + means, linear model)
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
# FEATURE EXTRACTION
# =============================================================================

def _extract_cov_upper(ts: np.ndarray) -> np.ndarray:
    """Upper triangle of covariance matrix (z-scored per region first)."""
    mean = ts.mean(axis=0, keepdims=True)
    std = ts.std(axis=0, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    ts_norm = (ts - mean) / std
    cov = np.cov(ts_norm, rowvar=False)
    idx = np.triu_indices(cov.shape[0], k=1)
    return cov[idx].astype(np.float32)


def _region_means(ts: np.ndarray) -> np.ndarray:
    """Mean activation per region."""
    return ts.mean(axis=0).astype(np.float32)


def extract_features(ts: np.ndarray, condition: str) -> np.ndarray:
    """
    Extract features from a (T, R) time series according to a condition.

    Conditions
    ----------
    Raw time series : flatten the T x R matrix              -> T*R features
    Means only      : mean activation per region             -> R features
    Cov only        : upper triangle of covariance (z-scored)-> R*(R-1)/2
    Cov + means     : cov upper triangle + region means      -> R*(R-1)/2 + R
    """
    if condition == "Raw time series":
        return ts.flatten().astype(np.float32)
    elif condition == "Means only":
        return _region_means(ts)
    elif condition == "Cov only":
        return _extract_cov_upper(ts)
    elif condition == "Cov + means":
        return np.concatenate([_extract_cov_upper(ts), _region_means(ts)])
    else:
        raise ValueError(f"Unknown condition: {condition}")


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
        "preds": preds,
        "targets": targets,
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
    # Out-of-fold predictions (each subject predicted once, on its val fold)
    oof_preds = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        result = train_one_fold(
            X_train, y_train, X_val, y_val,
            label_type, activation_name, hidden_dims,
        )
        fold_scores.append(result["score"])
        fold_epochs.append(result["total_epochs"])
        oof_preds[val_idx] = result["preds"]

    scores = np.array(fold_scores)
    return {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
        "avg_epochs": np.mean(fold_epochs),
        "oof_preds": oof_preds,
        "y_true": y,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_predictions(all_results: list, output_dir: Path):
    """For each label, find the best config and plot true vs predicted.

    Regression: scatter plot with identity line, R, R², MAE.
    Classification: ROC curve + predicted probability histograms per class.
    """
    from sklearn.metrics import roc_curve, auc

    labels_in_results = list(dict.fromkeys(r["label"] for r in all_results))

    for label_name in labels_in_results:
        label_results = [r for r in all_results if r["label"] == label_name]
        label_type = LABELS[label_name]["type"]
        lower_is_better = label_type == "regression"

        if lower_is_better:
            best = min(label_results, key=lambda r: r["mean"])
        else:
            best = max(label_results, key=lambda r: r["mean"])

        y_true = best["y_true"]
        oof_preds = best["oof_preds"]
        config_str = (f"S{best['n_regions']} / {best['condition']} / "
                      f"{best['arch']} / {best['activation']}")
        baseline = STEP4_BASELINES[label_name]

        if label_type == "regression":
            y_pred = oof_preds

            # Stats computed on ALL subjects
            r_corr = np.corrcoef(y_true, y_pred)[0, 1]
            r2 = r_corr ** 2
            mae = mean_absolute_error(y_true, y_pred)

            # For the plot: pick one random subject per unique label value
            # to avoid overlapping points (e.g. many subjects aged 28)
            rng = np.random.RandomState(RANDOM_STATE)
            unique_vals = np.unique(y_true)
            plot_idx = []
            for val in unique_vals:
                candidates = np.where(y_true == val)[0]
                plot_idx.append(rng.choice(candidates))
            plot_idx = np.array(plot_idx)
            y_true_plot = y_true[plot_idx]
            y_pred_plot = y_pred[plot_idx]

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(y_true_plot, y_pred_plot, alpha=0.6, s=30,
                       edgecolors="none", c="#4C72B0")

            # Identity line
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    "k--", linewidth=1, label="Perfect prediction")
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

            ax.set_xlabel("True value", fontsize=12)
            ax.set_ylabel("Predicted value", fontsize=12)
            ax.set_title(f"{label_name} — True vs Predicted (out-of-fold)\n"
                         f"{config_str}", fontsize=11, fontweight="bold")

            stats_text = (f"MAE = {mae:.3f}\n"
                          f"r = {r_corr:.3f}\n"
                          f"R² = {r2:.3f}\n"
                          f"n = {len(y_true)} subjects"
                          f" ({len(unique_vals)} shown)\n"
                          f"Step4 MAE = {baseline['best']:.3f}")
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat",
                              alpha=0.8))
            ax.legend(loc="lower right")
            ax.set_aspect("equal")
            plt.tight_layout()

        else:
            # Classification: ROC curve + probability histogram
            probs = 1.0 / (1.0 + np.exp(-oof_preds))
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            class_names = LABELS[label_name].get(
                "class_names", {1: "class 1", 0: "class 0"})

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

            # ROC curve
            ax1.plot(fpr, tpr, color="#4C72B0", linewidth=2,
                     label=f"MLP (AUC = {roc_auc:.3f})")
            ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
            ax1.set_xlabel("False Positive Rate", fontsize=12)
            ax1.set_ylabel("True Positive Rate", fontsize=12)
            ax1.set_title("ROC Curve", fontsize=12, fontweight="bold")
            ax1.legend(loc="lower right", fontsize=10)
            ax1.set_aspect("equal")

            # Probability histogram per class
            mask_pos = y_true == 1
            ax2.hist(probs[mask_pos], bins=30, alpha=0.6, color="#4C72B0",
                     label=f"{class_names[1]} (n={mask_pos.sum()})",
                     density=True)
            ax2.hist(probs[~mask_pos], bins=30, alpha=0.6, color="#DD8452",
                     label=f"{class_names[0]} (n={(~mask_pos).sum()})",
                     density=True)
            ax2.set_xlabel("Predicted probability", fontsize=12)
            ax2.set_ylabel("Density", fontsize=12)
            ax2.set_title("Predicted probabilities by class",
                          fontsize=12, fontweight="bold")
            ax2.legend(fontsize=10)

            fig.suptitle(f"{label_name} — {config_str}",
                         fontsize=11, fontweight="bold", y=1.02)
            plt.tight_layout()

        path = output_dir / f"mlp_pred_{label_name.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {path}")
        plt.close(fig)


def plot_results(all_results: list, output_dir: Path):
    """One heatmap per label: rows = (condition / arch), cols = atlas.

    Shows the best activation for each (condition, arch, atlas) combo
    since activations showed minimal impact in earlier runs.
    """

    labels_in_results = list(dict.fromkeys(r["label"] for r in all_results))

    for label_name in labels_in_results:
        label_results = [r for r in all_results if r["label"] == label_name]
        label_type = LABELS[label_name]["type"]
        metric_name = "MAE" if label_type == "regression" else "ROC-AUC"
        lower_is_better = label_type == "regression"
        cmap = "RdYlGn_r" if lower_is_better else "RdYlGn"

        atlas_sizes = sorted(set(r["n_regions"] for r in label_results))
        cond_names = list(dict.fromkeys(r["condition"] for r in label_results))
        arch_names = list(dict.fromkeys(r["arch"] for r in label_results))

        # Rows: (condition, arch) — pick best activation per combo
        row_labels = []
        for cond in cond_names:
            for arch in arch_names:
                row_labels.append(f"{cond}  |  {arch}")

        matrix = np.full((len(row_labels), len(atlas_sizes)), np.nan)
        for i, (cond, arch) in enumerate(
            (c, a) for c in cond_names for a in arch_names
        ):
            for j, atlas in enumerate(atlas_sizes):
                matches = [
                    r for r in label_results
                    if r["condition"] == cond
                    and r["arch"] == arch
                    and r["n_regions"] == atlas
                ]
                if matches:
                    if lower_is_better:
                        best = min(matches, key=lambda r: r["mean"])
                    else:
                        best = max(matches, key=lambda r: r["mean"])
                    matrix[i, j] = best["mean"]

        fig, ax = plt.subplots(figsize=(8, max(5, len(row_labels) * 0.45 + 2)))
        im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                       vmin=np.nanmin(matrix) - 0.02,
                       vmax=np.nanmax(matrix) + 0.02)

        ax.set_xticks(range(len(atlas_sizes)))
        ax.set_xticklabels([f"Schaefer {n}" for n in atlas_sizes])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)

        # Horizontal lines between feature conditions
        n_arch = len(arch_names)
        for k in range(1, len(cond_names)):
            ax.axhline(y=k * n_arch - 0.5, color="black", linewidth=1.5)

        for i in range(len(row_labels)):
            for j in range(len(atlas_sizes)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val < np.nanmean(matrix) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=color, fontsize=8, fontweight="bold")

        baseline = STEP4_BASELINES[label_name]
        ax.set_title(
            f"{label_name} — MLP best per (features, arch) — {metric_name}\n"
            f"Step4 linear baseline: {baseline['best']:.3f} ({baseline['atlas']})",
            fontsize=11, fontweight="bold",
        )
        plt.colorbar(im, ax=ax, label=metric_name)
        plt.tight_layout()

        path = output_dir / f"mlp_{label_name.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {path}")
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
    print(f"  Device           : {DEVICE}")
    print(f"  Atlases          : {SCHAEFER_REGIONS}")
    print(f"  Feature sets     : {FEATURE_CONDITIONS}")
    print(f"  Architectures    : {list(ARCHITECTURES.keys())}")
    print(f"  Activations      : {list(ACTIVATIONS.keys())}")
    print(f"  Labels           : {list(LABELS.keys())}")
    print(f"  Dropout={DROPOUT}  LR={LR}  Batch={BATCH_SIZE}"
          f"  EarlyStop={EARLY_STOP_PATIENCE}")
    n_combos = (len(SCHAEFER_REGIONS) * len(FEATURE_CONDITIONS)
                * len(ARCHITECTURES) * len(ACTIVATIONS) * len(LABELS))
    print(f"  Total combinations: {n_combos}")
    print("=" * 70)

    all_results = []
    start_time = time.time()
    labels_printed = set()  # print label stats only once

    for n_regions in SCHAEFER_REGIONS:
        print(f"\n{'#'*70}")
        print(f"#  SCHAEFER {n_regions} REGIONS")
        print(f"{'#'*70}")

        subjects, labels_df = load_dataset(n_regions, verbose=True)

        if len(subjects) == 0:
            print(f"  No subjects found for Schaefer {n_regions}")
            continue

        n_subjects = len(subjects)
        ts_shape = subjects[0][1].shape  # (T, R)
        print(f"  Subjects loaded: {n_subjects}")
        print(f"  Time series shape per subject: {ts_shape}")

        for condition in FEATURE_CONDITIONS:
            # ----- Extract features for this condition -----
            print(f"\n  ╔{'═'*60}╗")
            print(f"  ║  Feature set: {condition:<45}║")

            X_all = np.stack([
                extract_features(ts, condition) for _, ts in
                tqdm(subjects, desc=f"  {condition}", leave=False)
            ])

            n_bad = np.isnan(X_all).sum() + np.isinf(X_all).sum()
            if n_bad > 0:
                print(f"  ║  WARNING: {n_bad} NaN/Inf -> replaced with 0"
                      f"{' '*(28-len(str(n_bad)))}║")
                X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

            n_features = X_all.shape[1]
            mem_mb = X_all.nbytes / 1e6
            print(f"  ║  Dimensions: {n_subjects} subjects x"
                  f" {n_features:,} features ({mem_mb:.0f} MB)  ║")
            print(f"  ╚{'═'*60}╝")

            if n_features > 100_000:
                print(f"  ⚠  Large feature space — training will be slow")

            for label_name, label_cfg in LABELS.items():
                valid_idx, y = get_label_array(labels_df, label_name)

                if len(y) < 50:
                    print(f"\n    Skipping {label_name}: only {len(y)} valid")
                    continue

                X = X_all[valid_idx]
                label_type = label_cfg["type"]
                metric_name = "MAE" if label_type == "regression" else "ROC-AUC"
                baseline = STEP4_BASELINES[label_name]

                print(f"\n    --- {label_name} ({label_type}, n={len(y)})"
                      f"  |  Step4: {baseline['best']:.3f} {baseline['metric']}"
                      f" ({baseline['atlas']}) ---")

                # Print label distribution once (first time we see this label)
                if label_name not in labels_printed:
                    labels_printed.add(label_name)
                    if label_type == "regression":
                        p50, p75, p90, p95 = np.percentile(y, [50, 75, 90, 95])
                        dummy_mae = np.mean(np.abs(y - np.mean(y)))
                        print(f"        Label distribution:")
                        print(f"          mean={y.mean():.2f}  std={y.std():.2f}"
                              f"  min={y.min():.2f}  max={y.max():.2f}")
                        print(f"          P50={p50:.2f}  P75={p75:.2f}"
                              f"  P90={p90:.2f}  P95={p95:.2f}")
                        print(f"          Dummy baseline (predict mean): MAE={dummy_mae:.2f}")
                        print(f"          Interpretation: MAE={baseline['best']:.2f}"
                              f" means avg error is ~{baseline['best']/y.std():.1%} of 1 std")
                    else:
                        n_pos = int(y.sum())
                        n_neg = len(y) - n_pos
                        class_names = label_cfg.get("class_names",
                                                    {1: "class 1", 0: "class 0"})
                        print(f"        Label distribution:")
                        print(f"          {class_names[1]}: {n_pos}"
                              f" ({n_pos/len(y)*100:.1f}%)  |"
                              f"  {class_names[0]}: {n_neg}"
                              f" ({n_neg/len(y)*100:.1f}%)")
                        print(f"          Dummy baseline (random): ROC-AUC=0.500")

                for arch_name, hidden_dims in ARCHITECTURES.items():
                    print(f"      [{arch_name}]")
                    for act_name in ACTIVATIONS:
                        t0 = time.time()
                        result = evaluate_model(X, y, label_type, act_name,
                                                hidden_dims)
                        elapsed = time.time() - t0

                        result["n_regions"] = n_regions
                        result["condition"] = condition
                        result["arch"] = arch_name
                        result["activation"] = act_name
                        result["label"] = label_name
                        result["n_features"] = n_features
                        all_results.append(result)

                        diff = result["mean"] - baseline["best"]
                        if label_type == "regression":
                            better = diff < 0
                        else:
                            better = diff > 0
                        flag = "+" if better else "-"

                        print(f"        {act_name:<12} {metric_name}:"
                              f" {result['mean']:.3f} +/- {result['std']:.3f}"
                              f"  [{diff:+.3f} {flag}]"
                              f"  ({result['avg_epochs']:.0f}ep, {elapsed:.0f}s)")

            # Free memory before next condition
            del X_all

    total_time = time.time() - start_time

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  RESULTS SUMMARY — MLP vs Step4 Linear Baselines")
    print(f"  Total training time: {total_time/60:.1f} min")
    print(f"{'='*90}")

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
        print(f"  {'#':<4} {'Atlas':<6} {'Features':<18} {'Architecture':<22}"
              f" {'Activation':<12} {'Score':<18} {'vs Step4'}")
        print(f"  {'-'*95}")

        for i, r in enumerate(results_sorted[:15]):
            score_str = f"{r['mean']:.3f} +/- {r['std']:.3f}"
            diff = r["mean"] - baseline["best"]
            if lower_is_better:
                better = diff < 0
            else:
                better = diff > 0
            rank = f">>>{i+1}" if i == 0 else f"   {i+1}"
            flag = "+" if better else "-"
            print(f"  {rank:<4} S{r['n_regions']:<5}"
                  f" {r['condition']:<18} {r['arch']:<22}"
                  f" {r['activation']:<12} {score_str:<18} {diff:+.3f} {flag}")

        if len(results_sorted) > 15:
            print(f"  ... ({len(results_sorted) - 15} more rows)")

    # =========================================================================
    # BEST PER FEATURE CONDITION (concise comparison table)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  BEST SCORE PER FEATURE CONDITION (across arch/activation/atlas)")
    print(f"{'='*90}")
    print(f"  {'Label':<14} {'Condition':<18} {'Score':<18}"
          f" {'Config':<35} {'vs Step4'}")
    print(f"  {'-'*90}")

    for label_name in LABELS:
        label_results = [r for r in all_results if r["label"] == label_name]
        if not label_results:
            continue
        label_type = LABELS[label_name]["type"]
        lower_is_better = label_type == "regression"
        baseline = STEP4_BASELINES[label_name]

        for condition in FEATURE_CONDITIONS:
            cond_results = [r for r in label_results
                            if r["condition"] == condition]
            if not cond_results:
                continue
            if lower_is_better:
                best = min(cond_results, key=lambda r: r["mean"])
            else:
                best = max(cond_results, key=lambda r: r["mean"])

            diff = best["mean"] - baseline["best"]
            if lower_is_better:
                better = diff < 0
            else:
                better = diff > 0
            flag = "+" if better else "-"
            config = f"S{best['n_regions']} {best['arch']} {best['activation']}"

            print(f"  {label_name:<14} {condition:<18}"
                  f" {best['mean']:.3f} +/- {best['std']:.3f}  "
                  f" {config:<35} {diff:+.3f} {flag}")

    # =========================================================================
    # SAVE JSON
    # =========================================================================
    output_path = HCP_ROOT / "mlp_results.json"
    output_data = {
        "description": "Step 5 - MLP baseline with feature condition comparison",
        "feature_conditions": FEATURE_CONDITIONS,
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
                "condition": r["condition"],
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
    print(f"\n  Results saved: {output_path}")

    # =========================================================================
    # PLOTS
    # =========================================================================
    print("\n  Generating heatmaps...")
    plot_results(all_results, HCP_ROOT)

    print("\n  Generating true vs predicted plots (best config per label)...")
    plot_predictions(all_results, HCP_ROOT)

    print(f"\n  Done. Total time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
