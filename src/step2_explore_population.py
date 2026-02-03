"""
ÉTAPE 2: Explorer la population (tous les sujets)
=================================================

Après avoir exploré UN sujet (step1), on analyse maintenant TOUS les sujets
pour comprendre la variabilité inter-sujets et identifier les outliers.

Questions clés:
- Combien de sujets sont disponibles?
- Les statistiques (moyenne, variance) sont-elles cohérentes entre sujets?
- Y a-t-il des sujets problématiques à exclure?
- Quelle est la balance des classes (H/F)?

Auteur: Dan Abergel
Thèse de Master - Hebrew University of Jerusalem
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")
LABELS_JSON = HCP_ROOT / "model_input" / "imageID_to_labels.json"
OUTPUT_DIR = HCP_ROOT  # Où sauvegarder les figures
N_REGIONS = 200


# =============================================================================
# FONCTIONS DE CHARGEMENT
# =============================================================================
def load_subject(subject_id: str) -> np.ndarray | None:
    """Charge le fichier schaefer200.npy d'un sujet."""
    npy_path = (
        HCP_ROOT / f"subject_{subject_id}" / "MNINonLinear" / "Results"
        / "rfMRI_REST1_LR" / f"rfMRI_REST1_LR_schaefer{N_REGIONS}.npy"
    )
    if not npy_path.exists():
        return None
    return np.load(npy_path)


def load_labels() -> dict:
    """Charge les labels depuis le JSON."""
    with open(LABELS_JSON, "r") as f:
        return json.load(f)


# =============================================================================
# COLLECTE DES STATISTIQUES
# =============================================================================
def collect_population_stats() -> dict:
    """
    Collecte les statistiques de tous les sujets.

    Returns:
        dict avec:
        - subject_ids: liste des IDs
        - means: moyenne globale par sujet
        - stds: écart-type global par sujet
        - variances_mean: variance moyenne des ROIs par sujet
        - variances_max: variance max des ROIs par sujet
        - n_timepoints: nombre de timepoints par sujet
        - sex: label de sexe (si disponible)
    """
    labels_dict = load_labels()

    stats = {
        "subject_ids": [],
        "means": [],
        "stds": [],
        "variances_mean": [],
        "variances_max": [],
        "variances_min": [],
        "n_timepoints": [],
        "sex": [],
    }

    # Trouver tous les sujets
    subject_dirs = sorted(HCP_ROOT.glob("subject_*"))
    print(f"Found {len(subject_dirs)} subject directories")

    for subject_dir in tqdm(subject_dirs, desc="Loading subjects"):
        subject_id = subject_dir.name.replace("subject_", "")

        # Charger les données
        ts = load_subject(subject_id)
        if ts is None:
            continue

        # Calculer les stats
        roi_var = ts.var(axis=0)  # Variance par ROI

        stats["subject_ids"].append(subject_id)
        stats["means"].append(ts.mean())
        stats["stds"].append(ts.std())
        stats["variances_mean"].append(roi_var.mean())
        stats["variances_max"].append(roi_var.max())
        stats["variances_min"].append(roi_var.min())
        stats["n_timepoints"].append(ts.shape[0])

        # Label de sexe
        scan_key = f"{subject_id}_REST1_LR"
        if scan_key in labels_dict:
            stats["sex"].append(labels_dict[scan_key].get("Sex_Binary", -1))
        else:
            stats["sex"].append(-1)  # -1 = pas de label

    # Convertir en arrays numpy
    for key in stats:
        if key != "subject_ids":
            stats[key] = np.array(stats[key])

    return stats


# =============================================================================
# DÉTECTION DES OUTLIERS
# =============================================================================
def detect_outliers(values: np.ndarray, n_std: float = 3.0) -> np.ndarray:
    """
    Détecte les outliers avec la méthode des écarts-types.

    Args:
        values: array de valeurs
        n_std: nombre d'écarts-types pour définir un outlier

    Returns:
        Masque booléen (True = outlier)
    """
    mean = values.mean()
    std = values.std()
    return np.abs(values - mean) > n_std * std


def identify_problematic_subjects(stats: dict, n_std: float = 3.0) -> dict:
    """
    Identifie les sujets problématiques selon plusieurs critères.

    Returns:
        dict avec les indices des sujets problématiques par critère
    """
    problematic = {
        "high_variance": np.where(detect_outliers(stats["variances_max"], n_std))[0],
        "low_variance": np.where(stats["variances_min"] < 1.0)[0],
        "extreme_mean": np.where(detect_outliers(stats["means"], n_std))[0],
        "wrong_timepoints": np.where(stats["n_timepoints"] != stats["n_timepoints"][0])[0],
    }

    return problematic


# =============================================================================
# VISUALISATION
# =============================================================================
def plot_population_stats(stats: dict, save_path: Path = None):
    """Crée les visualisations pour la population."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- Plot 1: Distribution des moyennes par sujet ---
    ax1 = axes[0, 0]
    ax1.hist(stats["means"], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(stats["means"].mean(), color='red', linestyle='--',
                label=f'Mean: {stats["means"].mean():.1f}')
    ax1.set_xlabel('Moyenne du signal')
    ax1.set_ylabel('Nombre de sujets')
    ax1.set_title('Distribution des moyennes par sujet')
    ax1.legend()

    # --- Plot 2: Distribution des variances moyennes par sujet ---
    ax2 = axes[0, 1]
    ax2.hist(stats["variances_mean"], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(stats["variances_mean"].mean(), color='red', linestyle='--',
                label=f'Mean: {stats["variances_mean"].mean():.1f}')
    ax2.set_xlabel('Variance moyenne des ROIs')
    ax2.set_ylabel('Nombre de sujets')
    ax2.set_title('Distribution des variances par sujet')
    ax2.legend()

    # --- Plot 3: Variance max par sujet (pour détecter outliers) ---
    ax3 = axes[0, 2]
    sorted_idx = np.argsort(stats["variances_max"])
    ax3.bar(range(len(stats["variances_max"])), stats["variances_max"][sorted_idx],
            alpha=0.7, color='steelblue')
    threshold = stats["variances_max"].mean() + 3 * stats["variances_max"].std()
    ax3.axhline(threshold, color='red', linestyle='--', label=f'3σ threshold: {threshold:.0f}')
    ax3.set_xlabel('Sujets (triés)')
    ax3.set_ylabel('Variance max')
    ax3.set_title('Variance max par sujet (triée)')
    ax3.legend()

    # --- Plot 4: Moyenne vs Variance (scatter) ---
    ax4 = axes[1, 0]
    colors = ['blue' if s == 0 else 'red' if s == 1 else 'gray' for s in stats["sex"]]
    ax4.scatter(stats["means"], stats["variances_mean"], c=colors, alpha=0.5, s=20)
    ax4.set_xlabel('Moyenne du signal')
    ax4.set_ylabel('Variance moyenne')
    ax4.set_title('Moyenne vs Variance par sujet')
    # Légende manuelle
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Male (0)'),
                       Patch(facecolor='red', label='Female (1)'),
                       Patch(facecolor='gray', label='No label')]
    ax4.legend(handles=legend_elements, loc='upper right')

    # --- Plot 5: Balance des classes ---
    ax5 = axes[1, 1]
    sex_counts = np.bincount(stats["sex"][stats["sex"] >= 0])
    labels = ['Male (0)', 'Female (1)']
    colors_bar = ['steelblue', 'coral']
    bars = ax5.bar(labels[:len(sex_counts)], sex_counts, color=colors_bar[:len(sex_counts)],
                   edgecolor='black')
    ax5.set_ylabel('Nombre de sujets')
    ax5.set_title('Balance des classes (sexe)')
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, sex_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontweight='bold')
    # Ratio
    if len(sex_counts) == 2:
        ratio = sex_counts[0] / sex_counts[1]
        ax5.text(0.5, 0.9, f'Ratio M/F: {ratio:.2f}', transform=ax5.transAxes,
                 ha='center', fontsize=12)

    # --- Plot 6: Distribution du nombre de timepoints ---
    ax6 = axes[1, 2]
    unique_tp, counts_tp = np.unique(stats["n_timepoints"], return_counts=True)
    ax6.bar([str(t) for t in unique_tp], counts_tp, color='steelblue', edgecolor='black')
    ax6.set_xlabel('Nombre de timepoints')
    ax6.set_ylabel('Nombre de sujets')
    ax6.set_title('Distribution des timepoints')
    for i, (tp, c) in enumerate(zip(unique_tp, counts_tp)):
        ax6.text(i, c + 1, str(c), ha='center', va='bottom')

    # Titre global
    n_subjects = len(stats["subject_ids"])
    fig.suptitle(f'Exploration de la population - {n_subjects} sujets',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    else:
        plt.show()

    plt.close()


def print_population_summary(stats: dict, problematic: dict):
    """Affiche un résumé textuel de la population."""

    n = len(stats["subject_ids"])

    print("\n" + "=" * 60)
    print("RÉSUMÉ DE LA POPULATION")
    print("=" * 60)

    print(f"\n📊 DONNÉES:")
    print(f"   Nombre de sujets: {n}")
    print(f"   Timepoints: {stats['n_timepoints'][0]} (mode)")
    print(f"   ROIs: {N_REGIONS}")

    print(f"\n📈 STATISTIQUES DU SIGNAL:")
    print(f"   Moyenne globale: {stats['means'].mean():.2f} ± {stats['means'].std():.2f}")
    print(f"   Plage des moyennes: [{stats['means'].min():.2f}, {stats['means'].max():.2f}]")
    print(f"   Variance moyenne: {stats['variances_mean'].mean():.2f} ± {stats['variances_mean'].std():.2f}")

    print(f"\n⚖️  BALANCE DES CLASSES:")
    sex_valid = stats["sex"][stats["sex"] >= 0]
    if len(sex_valid) > 0:
        n_male = (sex_valid == 0).sum()
        n_female = (sex_valid == 1).sum()
        print(f"   Male (0):   {n_male} ({100*n_male/len(sex_valid):.1f}%)")
        print(f"   Female (1): {n_female} ({100*n_female/len(sex_valid):.1f}%)")
        print(f"   Sans label: {(stats['sex'] == -1).sum()}")

    print(f"\n⚠️  SUJETS PROBLÉMATIQUES:")
    total_problematic = set()
    for criterion, indices in problematic.items():
        if len(indices) > 0:
            print(f"   {criterion}: {len(indices)} sujets")
            total_problematic.update(indices)
            # Montrer les premiers
            if len(indices) <= 5:
                ids = [stats["subject_ids"][i] for i in indices]
                print(f"      → {ids}")

    if len(total_problematic) == 0:
        print("   Aucun sujet problématique détecté")
    else:
        print(f"\n   TOTAL: {len(total_problematic)} sujets uniques à vérifier")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("EXPLORATION DE LA POPULATION HCP")
    print(f"Racine: {HCP_ROOT}\n")

    # Collecter les stats
    stats = collect_population_stats()

    if len(stats["subject_ids"]) == 0:
        print("Aucun sujet trouvé!")
    else:
        # Identifier les problèmes
        problematic = identify_problematic_subjects(stats)

        # Afficher le résumé
        print_population_summary(stats, problematic)

        # Visualisation
        save_path = OUTPUT_DIR / "population_exploration.png"
        plot_population_stats(stats, save_path=save_path)

        print(f"\n→ Regarde le fichier: {save_path}")
