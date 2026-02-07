"""
Step 4: Compare normalization axes for connectivity matrices
============================================================

Question: Before computing the covariance matrix, should we z-score the
time series, and along which axis?

Conditions (all use covariance as the connectivity measure):
1. No normalization         -> raw covariance
2. Z-score per region       -> covariance (= correlation)
3. Z-score per timepoint    -> covariance
4. Z-score both             -> covariance (per region, then per timepoint)

Atlas: Schaefer 100, 200, 300

Author: Dan Abergel
Master's Thesis - Hebrew University of Jerusalem
"""

import json
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


# =============================================================================
# CONFIGURATION
# =============================================================================
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")
LABELS_JSON = HCP_ROOT / "model_input" / "imageID_to_labels.json"
CLEAN_SUBJECTS_JSON = HCP_ROOT / "subjects_clean.json"

SCHAEFER_REGIONS = [100, 200, 300]

N_SPLITS = 5
RANDOM_STATE = 42


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def no_normalization(ts: np.ndarray) -> np.ndarray:
    """No normalization - raw data."""
    return ts


def zscore_per_region(ts: np.ndarray) -> np.ndarray:
    """
    Z-score along time axis (axis=0), independently for each region.
    Each region gets mean=0, std=1 across time.
    Result: covariance of this = correlation matrix.

    ts: (T, R)
    """
    mean = ts.mean(axis=0, keepdims=True)  # (1, R)
    std = ts.std(axis=0, keepdims=True)    # (1, R)
    std = np.where(std < 1e-10, 1.0, std)
    return (ts - mean) / std


def zscore_per_timepoint(ts: np.ndarray) -> np.ndarray:
    """
    Z-score along region axis (axis=1), independently for each timepoint.
    Each timepoint gets mean=0, std=1 across regions.
    Removes global signal fluctuations at each time point.

    ts: (T, R)
    """
    mean = ts.mean(axis=1, keepdims=True)  # (T, 1)
    std = ts.std(axis=1, keepdims=True)    # (T, 1)
    std = np.where(std < 1e-10, 1.0, std)
    return (ts - mean) / std


def zscore_both(ts: np.ndarray) -> np.ndarray:
    """
    Z-score per region first (axis=0), then per timepoint (axis=1).
    Double normalization.

    ts: (T, R)
    """
    # First: per region (temporal)
    ts = zscore_per_region(ts)
    # Then: per timepoint (spatial)
    ts = zscore_per_timepoint(ts)
    return ts


# =============================================================================
# FEATURE EXTRACTION (always covariance)
# =============================================================================

def extract_covariance(ts: np.ndarray) -> np.ndarray:
    """
    Extract upper triangle of the covariance matrix.

    ts: (T, R) -> vector of size R*(R-1)/2
    """
    cov = np.cov(ts, rowvar=False)  # (R, R)
    idx = np.triu_indices(cov.shape[0], k=1)
    return cov[idx].astype(np.float32)


# =============================================================================
# CONDITIONS
# =============================================================================

CONDITIONS = [
    {
        "name": "No normalization",
        "normalize": no_normalization,
    },
    {
        "name": "Z-score per region",
        "normalize": zscore_per_region,
    },
    {
        "name": "Z-score per timepoint",
        "normalize": zscore_per_timepoint,
    },
    {
        "name": "Z-score both",
        "normalize": zscore_both,
    },
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
    """Load clean subjects and labels for a given Schaefer resolution."""
    with open(LABELS_JSON, "r") as f:
        labels_dict = json.load(f)

    with open(CLEAN_SUBJECTS_JSON, "r") as f:
        clean_data = json.load(f)
    clean_ids = clean_data["subject_ids"]

    subjects = []
    labels = []

    iterator = tqdm(clean_ids, desc=f"Loading Schaefer {n_regions}") if verbose else clean_ids

    for subject_id in iterator:
        ts = load_subject(subject_id, n_regions)
        if ts is None:
            continue

        scan_key = f"{subject_id}_REST1_LR"
        if scan_key not in labels_dict:
            continue

        subjects.append((subject_id, ts))
        labels.append(labels_dict[scan_key]["Sex_Binary"])

    return subjects, np.array(labels)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_condition(
    subjects: list,
    y: np.ndarray,
    normalize_fn: Callable,
    condition_name: str
) -> dict:
    """Evaluate one normalization condition with covariance features."""

    X_list = []
    for subject_id, ts in subjects:
        ts_norm = normalize_fn(ts)
        features = extract_covariance(ts_norm)
        X_list.append(features)

    X = np.stack(X_list)

    # Check for NaN/Inf
    n_bad = np.isnan(X).sum() + np.isinf(X).sum()
    if n_bad > 0:
        print(f"  WARNING: {n_bad} NaN/Inf values detected - replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1
        ))
    ])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    return {
        "name": condition_name,
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(all_results: list, output_dir: Path):
    """Generate comparison plots."""

    atlas_sizes = sorted(set(r["n_regions"] for r in all_results))
    norm_names = list(dict.fromkeys(r["normalization"] for r in all_results))

    # Build results matrix
    results_matrix = np.zeros((len(norm_names), len(atlas_sizes)))
    std_matrix = np.zeros((len(norm_names), len(atlas_sizes)))

    for r in all_results:
        i = norm_names.index(r["normalization"])
        j = atlas_sizes.index(r["n_regions"])
        results_matrix[i, j] = r["mean"]
        std_matrix[i, j] = r["std"]

    # === Figure 1: Heatmap ===
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    im = ax1.imshow(results_matrix, cmap='RdYlGn', aspect='auto',
                    vmin=results_matrix.min() - 0.02,
                    vmax=results_matrix.max() + 0.02)

    ax1.set_xticks(range(len(atlas_sizes)))
    ax1.set_xticklabels([f'Schaefer {n}' for n in atlas_sizes])
    ax1.set_yticks(range(len(norm_names)))
    ax1.set_yticklabels(norm_names)

    for i in range(len(norm_names)):
        for j in range(len(atlas_sizes)):
            val = results_matrix[i, j]
            color = 'white' if val < results_matrix.mean() else 'black'
            ax1.text(j, i, f'{val:.3f}', ha='center', va='center',
                     color=color, fontsize=11, fontweight='bold')

    ax1.set_xlabel('Atlas')
    ax1.set_ylabel('Normalization')
    ax1.set_title('ROC-AUC: Normalization axis x Atlas resolution',
                  fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='ROC-AUC')
    plt.tight_layout()
    fig1.savefig(output_dir / 'comparison_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'comparison_heatmap.png'}")
    plt.close(fig1)

    # === Figure 2: Line plot ===
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

    for i, norm in enumerate(norm_names):
        means = results_matrix[i, :]
        stds = std_matrix[i, :]
        ax2.plot(atlas_sizes, means, 'o-', color=colors[i], label=norm,
                 linewidth=2, markersize=8)
        ax2.fill_between(atlas_sizes, means - stds, means + stds,
                         color=colors[i], alpha=0.1)

    ax2.set_xlabel('Number of Schaefer regions')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('Performance vs Atlas resolution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(atlas_sizes)
    plt.tight_layout()
    fig2.savefig(output_dir / 'comparison_lines.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'comparison_lines.png'}")
    plt.close(fig2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("NORMALIZATION AXIS COMPARISON")
    print("=" * 70)
    print(f"\nAtlas resolutions: {SCHAEFER_REGIONS}")
    print(f"Normalization conditions: {len(CONDITIONS)}")
    print(f"Feature extraction: covariance (always)")
    print(f"Total combinations: {len(SCHAEFER_REGIONS) * len(CONDITIONS)}")

    all_results = []

    for n_regions in SCHAEFER_REGIONS:
        print(f"\n{'='*70}")
        print(f"SCHAEFER {n_regions} REGIONS")
        print(f"{'='*70}")

        n_features = n_regions * (n_regions - 1) // 2
        print(f"Features: {n_features} (upper triangle {n_regions}x{n_regions})")

        subjects, y = load_dataset(n_regions, verbose=True)

        if len(subjects) == 0:
            print(f"  No subjects found for Schaefer {n_regions}")
            continue

        print(f"Subjects loaded: {len(subjects)}")

        for condition in CONDITIONS:
            print(f"\n  > {condition['name']}")

            result = evaluate_condition(
                subjects, y,
                condition["normalize"],
                condition["name"]
            )

            result["n_regions"] = n_regions
            result["n_features"] = n_features
            result["normalization"] = condition["name"]

            all_results.append(result)
            print(f"    ROC-AUC: {result['mean']:.3f} +/- {result['std']:.3f}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    results_sorted = sorted(all_results, key=lambda x: x["mean"], reverse=True)

    print(f"\n{'Rank':<6} {'Atlas':<12} {'Normalization':<25} {'ROC-AUC':<15}")
    print("-" * 60)

    for i, r in enumerate(results_sorted):
        score_str = f"{r['mean']:.3f} +/- {r['std']:.3f}"
        marker = ">>> " if i == 0 else "    "
        print(f"{marker}{i+1:<2} {r['n_regions']:<12} {r['normalization']:<25} {score_str}")

    # Best per atlas
    print(f"\n{'-'*60}")
    print("BEST NORMALIZATION PER ATLAS")
    print(f"{'-'*60}")

    for n in sorted(set(r["n_regions"] for r in all_results)):
        best = max([r for r in all_results if r["n_regions"] == n], key=lambda x: x["mean"])
        print(f"  Schaefer {n:<5} -> {best['normalization']:<25} {best['mean']:.3f} +/- {best['std']:.3f}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_path = HCP_ROOT / "normalization_axis_comparison.json"
    output_data = {
        "description": "Comparison of z-score normalization axes before covariance computation",
        "conditions": [c["name"] for c in CONDITIONS],
        "schaefer_regions": SCHAEFER_REGIONS,
        "results": [
            {
                "n_regions": r["n_regions"],
                "normalization": r["normalization"],
                "mean_auc": float(r["mean"]),
                "std_auc": float(r["std"]),
                "fold_scores": r["scores"].tolist()
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
