"""
ÉTAPE 4: Comparer les méthodes de normalisation ET les atlas Schaefer
=====================================================================

Questions:
1. Quelle normalisation du signal donne les meilleurs résultats?
2. Quel nombre de régions Schaefer est optimal?

Dimensions testées:
- Normalisation: raw, z-score temporal, PSC, mean-centered, z-score spatial
- Features: covariance, corrélation
- Atlas Schaefer: 100, 200, 300, 400 régions

Auteur: Dan Abergel
Thèse de Master - Hebrew University of Jerusalem
"""

import json
from pathlib import Path
from typing import Callable
from itertools import product

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

# Atlas Schaefer à tester (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
# Note: il faut que les fichiers .npy existent pour chaque résolution
SCHAEFER_REGIONS = [100, 200, 300]

N_SPLITS = 5
RANDOM_STATE = 42


# =============================================================================
# MÉTHODES DE NORMALISATION DU SIGNAL
# =============================================================================

def no_normalization(ts: np.ndarray) -> np.ndarray:
    """
    Aucune normalisation - données brutes.

    Args:
        ts: (T, R) array - T timepoints, R ROIs

    Returns:
        ts inchangé
    """
    return ts


def zscore_temporal(ts: np.ndarray) -> np.ndarray:
    """
    Z-score sur le temps (par ROI).

    Chaque ROI aura moyenne=0 et std=1 sur le temps.
    Formule: z_{t,r} = (x_{t,r} - mean_r) / std_r

    Args:
        ts: (T, R) array

    Returns:
        ts normalisé par ROI
    """
    mean = ts.mean(axis=0, keepdims=True)  # (1, R)
    std = ts.std(axis=0, keepdims=True)    # (1, R)

    # Éviter division par zéro
    std = np.where(std < 1e-10, 1.0, std)

    return (ts - mean) / std


def zscore_spatial(ts: np.ndarray) -> np.ndarray:
    """
    Z-score sur les ROIs (par timepoint).

    Chaque timepoint aura moyenne=0 et std=1 sur les ROIs.
    Formule: z_{t,r} = (x_{t,r} - mean_t) / std_t

    Args:
        ts: (T, R) array

    Returns:
        ts normalisé par timepoint
    """
    mean = ts.mean(axis=1, keepdims=True)  # (T, 1)
    std = ts.std(axis=1, keepdims=True)    # (T, 1)

    # Éviter division par zéro
    std = np.where(std < 1e-10, 1.0, std)

    return (ts - mean) / std


def percent_signal_change(ts: np.ndarray) -> np.ndarray:
    """
    Percent Signal Change (PSC).

    Exprime chaque valeur comme pourcentage de changement par rapport
    à la moyenne de la ROI.
    Formule: PSC_{t,r} = (x_{t,r} - mean_r) / mean_r * 100

    Args:
        ts: (T, R) array

    Returns:
        ts en pourcentage de changement
    """
    mean = ts.mean(axis=0, keepdims=True)  # (1, R)

    # Éviter division par zéro
    mean = np.where(np.abs(mean) < 1e-10, 1.0, mean)

    return (ts - mean) / mean * 100


def mean_centering(ts: np.ndarray) -> np.ndarray:
    """
    Centrage sur la moyenne (sans normalisation de la variance).

    Enlève le niveau de base mais garde les différences de variance.
    Formule: x'_{t,r} = x_{t,r} - mean_r

    Args:
        ts: (T, R) array

    Returns:
        ts centré
    """
    mean = ts.mean(axis=0, keepdims=True)  # (1, R)
    return ts - mean


# =============================================================================
# MÉTHODES D'EXTRACTION DE FEATURES
# =============================================================================

def extract_covariance(ts: np.ndarray) -> np.ndarray:
    """
    Extrait le triangle supérieur de la matrice de covariance.

    Covariance: mesure comment deux variables varient ensemble.
    Cov(X,Y) = E[(X - μX)(Y - μY)]

    Args:
        ts: (T, R) array normalisé ou non

    Returns:
        Vecteur (R*(R-1)/2,) = (19900,) pour R=200
    """
    cov = np.cov(ts, rowvar=False)  # (R, R)
    idx = np.triu_indices(cov.shape[0], k=1)
    return cov[idx].astype(np.float32)


def extract_correlation(ts: np.ndarray) -> np.ndarray:
    """
    Extrait le triangle supérieur de la matrice de corrélation.

    Corrélation: covariance normalisée entre -1 et 1.
    Corr(X,Y) = Cov(X,Y) / (σX * σY)

    Args:
        ts: (T, R) array

    Returns:
        Vecteur (R*(R-1)/2,) = (19900,) pour R=200
    """
    corr = np.corrcoef(ts, rowvar=False)  # (R, R)
    idx = np.triu_indices(corr.shape[0], k=1)
    return corr[idx].astype(np.float32)


# =============================================================================
# DÉFINITION DES CONDITIONS EXPÉRIMENTALES
# =============================================================================

CONDITIONS = [
    {
        "name": "1. Raw + Covariance (baseline)",
        "normalize": no_normalization,
        "extract": extract_covariance,
        "description": "Aucune normalisation, matrice de covariance"
    },
    {
        "name": "2. Z-score temporal + Covariance",
        "normalize": zscore_temporal,
        "extract": extract_covariance,
        "description": "Z-score par ROI sur le temps, puis covariance"
    },
    {
        "name": "3. Z-score temporal + Correlation",
        "normalize": zscore_temporal,
        "extract": extract_correlation,
        "description": "Z-score par ROI, puis corrélation (devrait = condition 2)"
    },
    {
        "name": "4. Raw + Correlation",
        "normalize": no_normalization,
        "extract": extract_correlation,
        "description": "Pas de normalisation, mais corrélation (normalise implicitement)"
    },
    {
        "name": "5. PSC + Covariance",
        "normalize": percent_signal_change,
        "extract": extract_covariance,
        "description": "Percent Signal Change, puis covariance"
    },
    {
        "name": "6. Mean-centered + Covariance",
        "normalize": mean_centering,
        "extract": extract_covariance,
        "description": "Centrage sans normaliser variance, puis covariance"
    },
    {
        "name": "7. Z-score spatial + Covariance",
        "normalize": zscore_spatial,
        "extract": extract_covariance,
        "description": "Z-score par timepoint (rare), puis covariance"
    },
]


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_subject(subject_id: str, n_regions: int) -> np.ndarray | None:
    """Charge les time series Schaefer d'un sujet pour une résolution donnée."""
    npy_path = (
        HCP_ROOT / f"subject_{subject_id}" / "MNINonLinear" / "Results"
        / "rfMRI_REST1_LR" / f"rfMRI_REST1_LR_schaefer{n_regions}.npy"
    )
    if not npy_path.exists():
        return None
    return np.load(npy_path)


def load_dataset(n_regions: int, verbose: bool = True):
    """
    Charge les sujets propres et leurs labels pour une résolution Schaefer donnée.

    Args:
        n_regions: Nombre de régions Schaefer (100, 200, 300, ...)
        verbose: Afficher la barre de progression

    Returns:
        subjects: Liste de (subject_id, timeseries)
        labels: Array des labels
    """

    # Labels
    with open(LABELS_JSON, "r") as f:
        labels_dict = json.load(f)

    # Sujets propres
    with open(CLEAN_SUBJECTS_JSON, "r") as f:
        clean_data = json.load(f)
    clean_ids = clean_data["subject_ids"]

    # Charger les données
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
# ÉVALUATION D'UNE CONDITION
# =============================================================================

def evaluate_condition(
    subjects: list,
    y: np.ndarray,
    normalize_fn: Callable,
    extract_fn: Callable,
    condition_name: str
) -> dict:
    """
    Évalue une condition expérimentale.

    Args:
        subjects: Liste de (subject_id, timeseries)
        y: Labels
        normalize_fn: Fonction de normalisation du signal
        extract_fn: Fonction d'extraction de features
        condition_name: Nom pour l'affichage

    Returns:
        dict avec scores et statistiques
    """

    # Extraire les features pour tous les sujets
    X_list = []
    for subject_id, ts in subjects:
        # 1. Normaliser le signal
        ts_norm = normalize_fn(ts)

        # 2. Extraire les features
        features = extract_fn(ts_norm)
        X_list.append(features)

    X = np.stack(X_list)

    # Vérifier les NaN/Inf
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  ⚠️  {n_nan} NaN, {n_inf} Inf détectés - remplacement par 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Pipeline ML
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

    # Cross-validation
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    return {
        "name": condition_name,
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
        "feature_mean": np.abs(X).mean(),
        "feature_std": X.std(),
        "feature_range": (X.min(), X.max())
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_results(all_results: list, output_dir: Path):
    """
    Crée les visualisations comparatives des résultats.

    Génère 4 figures:
    1. Heatmap: Atlas × Normalisation
    2. Line plot: Performance vs Nombre de régions
    3. Bar chart: Comparaison des normalisations (moyennée sur atlas)
    4. Box plot: Variance des scores par condition

    Args:
        all_results: Liste des résultats de toutes les conditions
        output_dir: Dossier où sauvegarder les figures
    """

    # Extraire les données uniques
    atlas_sizes = sorted(set(r["n_regions"] for r in all_results))
    normalizations = []
    for r in all_results:
        if r["normalization"] not in normalizations:
            normalizations.append(r["normalization"])

    # Créer une matrice de résultats
    results_matrix = np.zeros((len(normalizations), len(atlas_sizes)))
    std_matrix = np.zeros((len(normalizations), len(atlas_sizes)))

    for r in all_results:
        i = normalizations.index(r["normalization"])
        j = atlas_sizes.index(r["n_regions"])
        results_matrix[i, j] = r["mean"]
        std_matrix[i, j] = r["std"]

    # =========================================================================
    # Figure 1: Heatmap Atlas × Normalisation
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    im = ax1.imshow(results_matrix, cmap='RdYlGn', aspect='auto',
                    vmin=results_matrix.min() - 0.02,
                    vmax=results_matrix.max() + 0.02)

    # Labels
    ax1.set_xticks(range(len(atlas_sizes)))
    ax1.set_xticklabels([f'Schaefer\n{n}' for n in atlas_sizes])
    ax1.set_yticks(range(len(normalizations)))
    # Nettoyer les noms de normalisation
    norm_labels = [n.split('. ')[1] if '. ' in n else n for n in normalizations]
    ax1.set_yticklabels(norm_labels)

    # Ajouter les valeurs dans les cellules
    for i in range(len(normalizations)):
        for j in range(len(atlas_sizes)):
            val = results_matrix[i, j]
            color = 'white' if val < results_matrix.mean() else 'black'
            ax1.text(j, i, f'{val:.3f}', ha='center', va='center',
                     color=color, fontsize=10, fontweight='bold')

    ax1.set_xlabel('Atlas Schaefer (nombre de régions)', fontsize=12)
    ax1.set_ylabel('Méthode de normalisation', fontsize=12)
    ax1.set_title('ROC-AUC par combinaison Atlas × Normalisation', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax1, label='ROC-AUC')
    plt.tight_layout()

    fig1.savefig(output_dir / 'comparison_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"✓ Figure sauvegardée: {output_dir / 'comparison_heatmap.png'}")
    plt.close(fig1)

    # =========================================================================
    # Figure 2: Line plot - Performance vs Nombre de régions
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(normalizations)))

    for i, norm in enumerate(normalizations):
        means = results_matrix[i, :]
        stds = std_matrix[i, :]
        label = norm.split('. ')[1] if '. ' in norm else norm

        ax2.plot(atlas_sizes, means, 'o-', color=colors[i], label=label,
                 linewidth=2, markersize=8)
        ax2.fill_between(atlas_sizes, means - stds, means + stds,
                         color=colors[i], alpha=0.1)

    ax2.set_xlabel('Nombre de régions Schaefer', fontsize=12)
    ax2.set_ylabel('ROC-AUC', fontsize=12)
    ax2.set_title('Performance vs Résolution de l\'atlas', fontsize=14, fontweight='bold')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(atlas_sizes)

    # Ligne de référence baseline
    ax2.axhline(y=0.849, color='red', linestyle='--', alpha=0.7, label='Baseline (0.849)')

    plt.tight_layout()
    fig2.savefig(output_dir / 'comparison_lines.png', dpi=150, bbox_inches='tight')
    print(f"✓ Figure sauvegardée: {output_dir / 'comparison_lines.png'}")
    plt.close(fig2)

    # =========================================================================
    # Figure 3: Bar chart - Comparaison des normalisations (moyenne sur atlas)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    # Moyenne sur tous les atlas
    norm_means = results_matrix.mean(axis=1)
    norm_stds = results_matrix.std(axis=1)

    # Trier par performance
    sorted_idx = np.argsort(norm_means)[::-1]

    y_pos = np.arange(len(normalizations))
    bars = ax3.barh(y_pos, norm_means[sorted_idx], xerr=norm_stds[sorted_idx],
                    color='steelblue', alpha=0.8, capsize=5)

    # Colorer la meilleure en vert
    bars[0].set_color('green')
    bars[0].set_alpha(1.0)

    ax3.set_yticks(y_pos)
    sorted_labels = [norm_labels[i] for i in sorted_idx]
    ax3.set_yticklabels(sorted_labels)
    ax3.set_xlabel('ROC-AUC (moyenne sur tous les atlas)', fontsize=12)
    ax3.set_title('Performance par méthode de normalisation', fontsize=14, fontweight='bold')

    # Ligne baseline
    ax3.axvline(x=0.849, color='red', linestyle='--', alpha=0.7, label='Baseline')

    # Ajouter les valeurs
    for i, (mean, std) in enumerate(zip(norm_means[sorted_idx], norm_stds[sorted_idx])):
        ax3.text(mean + std + 0.005, i, f'{mean:.3f}', va='center', fontsize=10)

    ax3.legend()
    plt.tight_layout()
    fig3.savefig(output_dir / 'comparison_normalizations.png', dpi=150, bbox_inches='tight')
    print(f"✓ Figure sauvegardée: {output_dir / 'comparison_normalizations.png'}")
    plt.close(fig3)

    # =========================================================================
    # Figure 4: Box plot - Distribution des scores par atlas
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    # Grouper les scores par atlas
    data_by_atlas = []
    for n in atlas_sizes:
        scores = []
        for r in all_results:
            if r["n_regions"] == n:
                scores.extend(r["scores"].tolist())
        data_by_atlas.append(scores)

    bp = ax4.boxplot(data_by_atlas, labels=[f'{n}' for n in atlas_sizes],
                     patch_artist=True)

    # Colorer les boîtes
    colors_box = plt.cm.Blues(np.linspace(0.3, 0.9, len(atlas_sizes)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)

    ax4.set_xlabel('Nombre de régions Schaefer', fontsize=12)
    ax4.set_ylabel('ROC-AUC (tous folds, toutes normalisations)', fontsize=12)
    ax4.set_title('Distribution des scores par résolution d\'atlas', fontsize=14, fontweight='bold')
    ax4.axhline(y=0.849, color='red', linestyle='--', alpha=0.7, label='Baseline')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig4.savefig(output_dir / 'comparison_boxplot.png', dpi=150, bbox_inches='tight')
    print(f"✓ Figure sauvegardée: {output_dir / 'comparison_boxplot.png'}")
    plt.close(fig4)

    # =========================================================================
    # Figure 5: Résumé - Top 10 combinaisons
    # =========================================================================
    fig5, ax5 = plt.subplots(figsize=(12, 6))

    # Trier tous les résultats
    sorted_results = sorted(all_results, key=lambda x: x["mean"], reverse=True)[:10]

    labels = [f"Schaefer {r['n_regions']} + {r['normalization'].split('. ')[1][:20]}"
              for r in sorted_results]
    means = [r["mean"] for r in sorted_results]
    stds = [r["std"] for r in sorted_results]

    y_pos = np.arange(len(labels))
    bars = ax5.barh(y_pos, means, xerr=stds, color='steelblue', alpha=0.8, capsize=5)

    # Colorer le meilleur
    bars[0].set_color('gold')
    bars[0].set_alpha(1.0)

    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(labels)
    ax5.set_xlabel('ROC-AUC', fontsize=12)
    ax5.set_title('Top 10 combinaisons Atlas × Normalisation', fontsize=14, fontweight='bold')
    ax5.axvline(x=0.849, color='red', linestyle='--', alpha=0.7, label='Baseline (0.849)')

    # Ajouter les valeurs
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax5.text(mean + std + 0.003, i, f'{mean:.3f}', va='center', fontsize=10, fontweight='bold')

    ax5.legend()
    ax5.invert_yaxis()
    plt.tight_layout()
    fig5.savefig(output_dir / 'comparison_top10.png', dpi=150, bbox_inches='tight')
    print(f"✓ Figure sauvegardée: {output_dir / 'comparison_top10.png'}")
    plt.close(fig5)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("COMPARAISON: NORMALISATION × ATLAS SCHAEFER")
    print("=" * 70)
    print(f"\nAtlas testés: {SCHAEFER_REGIONS}")
    print(f"Conditions de normalisation: {len(CONDITIONS)}")
    print(f"Total combinaisons: {len(SCHAEFER_REGIONS) * len(CONDITIONS)}")

    # Stocker tous les résultats
    all_results = []

    # Pour chaque résolution d'atlas
    for n_regions in SCHAEFER_REGIONS:
        print("\n" + "=" * 70)
        print(f"ATLAS SCHAEFER {n_regions} RÉGIONS")
        print("=" * 70)

        # Calculer le nombre de features
        n_features = n_regions * (n_regions - 1) // 2
        print(f"Features: {n_features} (triangle supérieur {n_regions}×{n_regions})")

        # Charger les données pour cette résolution
        subjects, y = load_dataset(n_regions, verbose=True)

        if len(subjects) == 0:
            print(f"⚠️  Aucun sujet trouvé pour Schaefer {n_regions}")
            print(f"   Les fichiers rfMRI_REST1_LR_schaefer{n_regions}.npy existent-ils?")
            continue

        print(f"Sujets chargés: {len(subjects)}")

        # Évaluer chaque condition de normalisation
        for condition in CONDITIONS:
            condition_name = f"Schaefer{n_regions} | {condition['name']}"
            print(f"\n  ▶ {condition['name'][:50]}")

            result = evaluate_condition(
                subjects, y,
                condition["normalize"],
                condition["extract"],
                condition["name"]
            )

            # Ajouter les métadonnées
            result["n_regions"] = n_regions
            result["n_features"] = n_features
            result["full_name"] = condition_name
            result["normalization"] = condition["name"]

            all_results.append(result)

            print(f"     ROC-AUC: {result['mean']:.3f} ± {result['std']:.3f}")

    # =========================================================================
    # TABLEAU RÉCAPITULATIF GLOBAL
    # =========================================================================
    print("\n" + "=" * 70)
    print("RÉSULTATS GLOBAUX (toutes combinaisons)")
    print("=" * 70)

    # Trier par score
    results_sorted = sorted(all_results, key=lambda x: x["mean"], reverse=True)

    print(f"\n{'Rang':<5} {'Atlas':<8} {'Normalisation':<35} {'ROC-AUC':<15}")
    print("-" * 70)

    for i, r in enumerate(results_sorted[:15]):  # Top 15
        norm_short = r["normalization"][:33]
        score_str = f"{r['mean']:.3f} ± {r['std']:.3f}"
        marker = "🏆" if i == 0 else "  "
        print(f"{marker} {i+1:<3} {r['n_regions']:<8} {norm_short:<35} {score_str:<15}")

    if len(results_sorted) > 15:
        print(f"... et {len(results_sorted) - 15} autres combinaisons")

    # =========================================================================
    # TABLEAU PAR ATLAS (meilleure normalisation pour chaque atlas)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MEILLEURE NORMALISATION PAR ATLAS")
    print("-" * 70)

    best_by_atlas = {}
    for r in all_results:
        n = r["n_regions"]
        if n not in best_by_atlas or r["mean"] > best_by_atlas[n]["mean"]:
            best_by_atlas[n] = r

    print(f"\n{'Atlas':<10} {'Meilleure normalisation':<35} {'ROC-AUC':<15} {'Features':<10}")
    print("-" * 70)

    for n in sorted(best_by_atlas.keys()):
        r = best_by_atlas[n]
        norm_short = r["normalization"][:33]
        score_str = f"{r['mean']:.3f} ± {r['std']:.3f}"
        print(f"{n:<10} {norm_short:<35} {score_str:<15} {r['n_features']:<10}")

    # =========================================================================
    # TABLEAU PAR NORMALISATION (meilleur atlas pour chaque normalisation)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MEILLEUR ATLAS PAR NORMALISATION")
    print("-" * 70)

    best_by_norm = {}
    for r in all_results:
        norm = r["normalization"]
        if norm not in best_by_norm or r["mean"] > best_by_norm[norm]["mean"]:
            best_by_norm[norm] = r

    print(f"\n{'Normalisation':<40} {'Atlas':<8} {'ROC-AUC':<15}")
    print("-" * 70)

    for norm in sorted(best_by_norm.keys()):
        r = best_by_norm[norm]
        norm_short = norm[:38]
        score_str = f"{r['mean']:.3f} ± {r['std']:.3f}"
        print(f"{norm_short:<40} {r['n_regions']:<8} {score_str:<15}")

    # =========================================================================
    # COMPARAISON AVEC BASELINE
    # =========================================================================
    baseline_score = 0.849
    best = results_sorted[0]
    diff = best["mean"] - baseline_score

    print("\n" + "=" * 70)
    print("COMPARAISON AVEC BASELINE")
    print("=" * 70)
    print(f"Baseline précédente:     0.849 (Schaefer 200, Raw + Covariance)")
    print(f"Meilleure combinaison:   {best['mean']:.3f} ({best['full_name']})")
    print(f"Différence:              {diff:+.3f} ({diff/baseline_score*100:+.1f}%)")

    # =========================================================================
    # SAUVEGARDER LES RÉSULTATS
    # =========================================================================
    output_path = HCP_ROOT / "normalization_atlas_comparison.json"
    output_data = {
        "schaefer_regions_tested": SCHAEFER_REGIONS,
        "n_conditions": len(CONDITIONS),
        "all_results": [
            {
                "n_regions": r["n_regions"],
                "n_features": r["n_features"],
                "normalization": r["normalization"],
                "mean_auc": float(r["mean"]),
                "std_auc": float(r["std"]),
                "scores": r["scores"].tolist()
            }
            for r in all_results
        ],
        "best_overall": {
            "n_regions": best["n_regions"],
            "normalization": best["normalization"],
            "mean_auc": float(best["mean"]),
            "std_auc": float(best["std"])
        },
        "best_by_atlas": {
            str(n): {
                "normalization": r["normalization"],
                "mean_auc": float(r["mean"])
            }
            for n, r in best_by_atlas.items()
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✓ Résultats sauvegardés: {output_path}")

    # =========================================================================
    # VISUALISATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)

    plot_results(all_results, HCP_ROOT)

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    print("\nFichiers générés:")
    print(f"  - {HCP_ROOT / 'normalization_atlas_comparison.json'}")
    print(f"  - {HCP_ROOT / 'comparison_heatmap.png'}")
    print(f"  - {HCP_ROOT / 'comparison_lines.png'}")
    print(f"  - {HCP_ROOT / 'comparison_normalizations.png'}")
    print(f"  - {HCP_ROOT / 'comparison_boxplot.png'}")
    print(f"  - {HCP_ROOT / 'comparison_top10.png'}")


if __name__ == "__main__":
    main()
