"""
ÉTAPE 1: Explorer les données HCP (sujet par sujet)
===================================================

On charge les fichiers .npy individuels par sujet.
Chaque sujet a: rfMRI_REST1_LR_schaefer200.npy

But: Comprendre l'échelle et la distribution du signal fMRI.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")
OUTPUT_DIR = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")  # Où sauvegarder les figures


# =============================================================================
# CHARGER UN SUJET
# =============================================================================
def load_subject(subject_id):
    """Charge le fichier schaefer200.npy d'un sujet."""
    npy_path = (
        HCP_ROOT / f"subject_{subject_id}" / "MNINonLinear" / "Results"
        / "rfMRI_REST1_LR" / "rfMRI_REST1_LR_schaefer200.npy"
    )
    if not npy_path.exists():
        return None
    return np.load(npy_path)


def find_first_subject():
    """Trouve le premier sujet disponible."""
    for subject_dir in sorted(HCP_ROOT.glob("subject_*")):
        subject_id = subject_dir.name.replace("subject_", "")
        ts = load_subject(subject_id)
        if ts is not None:
            return subject_id, ts
    return None, None


# =============================================================================
# VISUALISATION
# =============================================================================
def plot_signal(ts, subject_id, save_path=None):
    """
    Crée 3 visualisations du signal fMRI:
    1. Quelques ROIs au cours du temps
    2. Heatmap complète (timepoints x ROIs)
    3. Distribution des valeurs
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Plot 1: Quelques ROIs au cours du temps ---
    ax1 = axes[0, 0]
    n_rois_to_show = 5
    for i in range(n_rois_to_show):
        ax1.plot(ts[:, i], label=f'ROI {i}', alpha=0.7)
    ax1.set_xlabel('Timepoints')
    ax1.set_ylabel('Signal')
    ax1.set_title(f'Signal de {n_rois_to_show} ROIs au cours du temps')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Heatmap complète ---
    ax2 = axes[0, 1]
    im = ax2.imshow(ts.T, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Timepoints')
    ax2.set_ylabel('ROIs')
    ax2.set_title('Heatmap: Signal (timepoints × ROIs)')
    plt.colorbar(im, ax=ax2, label='Valeur')

    # --- Plot 3: Distribution des valeurs ---
    ax3 = axes[1, 0]
    ax3.hist(ts.flatten(), bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.set_xlabel('Valeur du signal')
    ax3.set_ylabel('Fréquence')
    ax3.set_title('Distribution de toutes les valeurs')
    ax3.axvline(ts.mean(), color='red', linestyle='--', label=f'Mean: {ts.mean():.2f}')
    ax3.legend()

    # --- Plot 4: Variance par ROI ---
    ax4 = axes[1, 1]
    roi_variance = ts.var(axis=0)
    ax4.bar(range(len(roi_variance)), roi_variance, alpha=0.7, color='steelblue')
    ax4.set_xlabel('ROI index')
    ax4.set_ylabel('Variance')
    ax4.set_title('Variance par ROI')
    ax4.axhline(roi_variance.mean(), color='red', linestyle='--', label=f'Mean: {roi_variance.mean():.2f}')
    ax4.legend()

    # Titre global
    fig.suptitle(f'Sujet {subject_id} - Shape: {ts.shape}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Sauvegarder
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# STATS TEXTUELLES
# =============================================================================
def print_stats(ts, subject_id):
    """Affiche les statistiques d'un sujet."""

    print(f"\n{'=' * 50}")
    print(f"SUJET: {subject_id}")
    print(f"{'=' * 50}")
    print(f"Shape: {ts.shape} (timepoints, ROIs)")
    print(f"\nStatistiques globales:")
    print(f"  Min:  {ts.min():.4f}")
    print(f"  Max:  {ts.max():.4f}")
    print(f"  Mean: {ts.mean():.4f}")
    print(f"  Std:  {ts.std():.4f}")

    roi_var = ts.var(axis=0)
    print(f"\nVariance par ROI:")
    print(f"  Min:  {roi_var.min():.6f}")
    print(f"  Max:  {roi_var.max():.6f}")
    print(f"  Mean: {roi_var.mean():.6f}")

    # ROIs à faible variance
    n_low = (roi_var < 1e-6).sum()
    if n_low > 0:
        print(f"  ⚠️  {n_low} ROIs avec variance < 1e-6")
    else:
        print(f"  ✓ Aucun ROI avec variance < 1e-6")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("EXPLORATION DES DONNÉES HCP")
    print(f"Racine: {HCP_ROOT}\n")

    # Trouver et charger le premier sujet
    subject_id, ts = find_first_subject()

    if ts is None:
        print("❌ Aucun sujet trouvé!")
    else:
        # Stats textuelles
        print_stats(ts, subject_id)

        # Graphique
        save_path = OUTPUT_DIR / f"signal_exploration_{subject_id}.png"
        plot_signal(ts, subject_id, save_path=save_path)

        print(f"\n→ Regarde le fichier: {save_path}")
