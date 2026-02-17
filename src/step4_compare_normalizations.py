"""
Step 4: Compare normalization axes across multiple labels
=========================================================

Each normalization removes certain means. We save those means and add them
back as extra features alongside the covariance matrix.

Conditions:
1. No normalization         -> cov only (nothing was removed)
2. Z-score per region       -> cov + region means (R values)
3. Z-score per timepoint    -> cov + timepoint means (T values)
4. Z-score both             -> cov + region means + timepoint means (R+T values)

Atlas: Schaefer 100, 200, 300
Labels: Sex, Age, BMI, BPDiastolic, Education, Race

Author: Dan Abergel
Master's Thesis - Hebrew University of Jerusalem
"""

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score


# =============================================================================
# CONFIGURATION
# =============================================================================
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")
CLEAN_SUBJECTS_JSON = HCP_ROOT / "subjects_clean.json"
HCP_SUBJECTS_CSV = HCP_ROOT / "metadata" / "HCP_YA_subjects.csv"

SCHAEFER_REGIONS = [100, 200, 300]

N_SPLITS = 5
RANDOM_STATE = 42

LABELS = {
    "Sex": {
        "column": "Gender",
        "type": "classification",
        "transform": lambda x: 1 if x == "M" else 0,
        "scoring": "roc_auc",
    },
    "Age": {
        "column": "Age_in_Yrs",
        "type": "regression",
        "transform": float,
        "scoring": "r2",
    },
    "BMI": {
        "column": "BMI",
        "type": "regression",
        "transform": float,
        "scoring": "r2",
    },
    "BPDiastolic": {
        "column": "BPDiastolic",
        "type": "regression",
        "transform": float,
        "scoring": "r2",
    },
    "Education": {
        "column": "SSAGA_Educ",
        "type": "regression",
        "transform": float,
        "scoring": "r2",
    },
    "Race": {
        "column": "Race",
        "type": "classification",
        "transform": lambda x: 1 if x == "White" else 0,
        "scoring": "roc_auc",
    },
}


# =============================================================================
# NORMALIZATION + FEATURE EXTRACTION
# =============================================================================

def extract_cov_upper(ts: np.ndarray) -> np.ndarray:
    """Extract upper triangle of the covariance matrix."""
    cov = np.cov(ts, rowvar=False)
    idx = np.triu_indices(cov.shape[0], k=1)
    return cov[idx].astype(np.float32)


def normalize_and_extract(ts: np.ndarray, condition: str) -> np.ndarray:
    """
    Normalize time series and extract features.
    The means removed during normalization are saved and appended as features.

    ts: (T, R) raw time series

    Returns: feature vector
    """
    if condition == "No normalization":
        # No normalization -> covariance only
        return extract_cov_upper(ts)

    elif condition == "Z-score per region":
        # Save region means (R values), then z-score per region
        region_means = ts.mean(axis=0).astype(np.float32)  # (R,)
        mean = ts.mean(axis=0, keepdims=True)
        std = ts.std(axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        ts_norm = (ts - mean) / std
        cov_features = extract_cov_upper(ts_norm)
        return np.concatenate([cov_features, region_means])

    elif condition == "Z-score per timepoint":
        # Save timepoint means (T values), then z-score per timepoint
        timepoint_means = ts.mean(axis=1).astype(np.float32)  # (T,)
        mean = ts.mean(axis=1, keepdims=True)
        std = ts.std(axis=1, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        ts_norm = (ts - mean) / std
        cov_features = extract_cov_upper(ts_norm)
        return np.concatenate([cov_features, timepoint_means])

    elif condition == "Z-score both":
        # Save region means (R values) from step 1
        region_means = ts.mean(axis=0).astype(np.float32)  # (R,)
        mean_r = ts.mean(axis=0, keepdims=True)
        std_r = ts.std(axis=0, keepdims=True)
        std_r = np.where(std_r < 1e-10, 1.0, std_r)
        ts_norm = (ts - mean_r) / std_r

        # Save timepoint means (T values) from step 2 (after region z-score)
        timepoint_means = ts_norm.mean(axis=1).astype(np.float32)  # (T,)
        mean_t = ts_norm.mean(axis=1, keepdims=True)
        std_t = ts_norm.std(axis=1, keepdims=True)
        std_t = np.where(std_t < 1e-10, 1.0, std_t)
        ts_norm = (ts_norm - mean_t) / std_t

        cov_features = extract_cov_upper(ts_norm)
        return np.concatenate([cov_features, region_means, timepoint_means])

    else:
        raise ValueError(f"Unknown condition: {condition}")


CONDITIONS = [
    "No normalization",
    "Z-score per region",
    "Z-score per timepoint",
    "Z-score both",
]


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
# EVALUATION
# =============================================================================

def evaluate_condition(
    subjects: list,
    y: np.ndarray,
    condition: str,
    label_type: str,
    scoring: str,
) -> dict:
    """Evaluate one condition."""

    X_list = []
    for subject_id, ts in subjects:
        features = normalize_and_extract(ts, condition)
        X_list.append(features)

    X = np.stack(X_list)

    n_bad = np.isnan(X).sum() + np.isinf(X).sum()
    if n_bad > 0:
        print(f"    WARNING: {n_bad} NaN/Inf values - replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if label_type == "classification":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l2", C=1.0, max_iter=5000,
                class_weight="balanced", solver="lbfgs", n_jobs=-1
            ))
        ])
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    else:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0))
        ])
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

    return {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
        "n_features": X.shape[1],
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(all_results: list, output_dir: Path):
    """Generate one heatmap per label."""

    labels_in_results = list(dict.fromkeys(r["label"] for r in all_results))

    for label_name in labels_in_results:
        label_results = [r for r in all_results if r["label"] == label_name]
        label_cfg = LABELS[label_name]
        metric_name = label_cfg["scoring"].upper()

        atlas_sizes = sorted(set(r["n_regions"] for r in label_results))
        conditions = list(dict.fromkeys(r["condition"] for r in label_results))

        matrix = np.zeros((len(conditions), len(atlas_sizes)))
        for r in label_results:
            i = conditions.index(r["condition"])
            j = atlas_sizes.index(r["n_regions"])
            matrix[i, j] = r["mean"]

        fig, ax = plt.subplots(figsize=(10, max(4, len(conditions) * 0.8 + 1)))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                       vmin=matrix.min() - 0.02, vmax=matrix.max() + 0.02)

        ax.set_xticks(range(len(atlas_sizes)))
        ax.set_xticklabels([f'Schaefer {n}' for n in atlas_sizes])
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)

        for i in range(len(conditions)):
            for j in range(len(atlas_sizes)):
                val = matrix[i, j]
                color = 'white' if val < matrix.mean() else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        color=color, fontsize=10, fontweight='bold')

        ax.set_title(f'{label_name} prediction ({metric_name})', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label=metric_name)
        plt.tight_layout()

        path = output_dir / f'comparison_{label_name.lower()}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("NORMALIZATION x LABELS COMPARISON")
    print("=" * 70)
    print(f"\nAtlas resolutions: {SCHAEFER_REGIONS}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Labels: {list(LABELS.keys())}")
    print(f"Total combinations: {len(SCHAEFER_REGIONS) * len(CONDITIONS) * len(LABELS)}")

    all_results = []

    for n_regions in SCHAEFER_REGIONS:
        print(f"\n{'='*70}")
        print(f"SCHAEFER {n_regions} REGIONS")
        print(f"{'='*70}")

        subjects, labels_df = load_dataset(n_regions, verbose=True)

        if len(subjects) == 0:
            print(f"  No subjects found for Schaefer {n_regions}")
            continue

        print(f"Subjects loaded: {len(subjects)}")

        for label_name, label_cfg in LABELS.items():
            valid_idx, y = get_label_array(labels_df, label_name)

            if len(y) < 50:
                print(f"\n  Skipping {label_name}: only {len(y)} valid subjects")
                continue

            valid_subjects = [subjects[i] for i in valid_idx]
            print(f"\n  --- {label_name} ({label_cfg['type']}, n={len(y)}) ---")

            for condition in CONDITIONS:
                print(f"    > {condition}")

                result = evaluate_condition(
                    valid_subjects, y,
                    condition,
                    label_cfg["type"], label_cfg["scoring"],
                )

                result["n_regions"] = n_regions
                result["condition"] = condition
                result["label"] = label_name

                all_results.append(result)

                metric = label_cfg["scoring"].upper()
                print(f"      {metric}: {result['mean']:.3f} +/- {result['std']:.3f}"
                      f"  ({result['n_features']} features)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    for label_name in LABELS:
        label_results = [r for r in all_results if r["label"] == label_name]
        if not label_results:
            continue

        metric = LABELS[label_name]["scoring"].upper()
        results_sorted = sorted(label_results, key=lambda x: x["mean"], reverse=True)

        print(f"\n  {label_name} ({metric}):")
        print(f"  {'Rank':<5} {'Atlas':<10} {'Condition':<25} {'Score':<18} {'Features'}")
        print(f"  {'-'*75}")

        for i, r in enumerate(results_sorted[:5]):
            score_str = f"{r['mean']:.3f} +/- {r['std']:.3f}"
            marker = ">>>" if i == 0 else "   "
            print(f"  {marker}{i+1:<2} {r['n_regions']:<10} {r['condition']:<25} {score_str:<18} {r['n_features']}")

    # =========================================================================
    # SAVE
    # =========================================================================
    output_path = HCP_ROOT / "normalization_labels_comparison.json"
    output_data = {
        "description": "Normalization with preserved means x labels comparison",
        "schaefer_regions": SCHAEFER_REGIONS,
        "conditions": CONDITIONS,
        "labels": list(LABELS.keys()),
        "results": [
            {
                "n_regions": r["n_regions"],
                "condition": r["condition"],
                "label": r["label"],
                "mean": float(r["mean"]),
                "std": float(r["std"]),
                "fold_scores": r["scores"].tolist(),
                "n_features": r["n_features"],
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
