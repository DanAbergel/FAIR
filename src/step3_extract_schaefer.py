"""
ÉTAPE 3: Extraire les séries temporelles pour différentes résolutions Schaefer
==============================================================================

Ce script extrait les séries temporelles des ROIs pour plusieurs résolutions
de l'atlas Schaefer (100, 200, 300, 400 régions).

Pour chaque sujet et chaque résolution, il crée un fichier .npy:
    subject_XXXXX/.../rfMRI_REST1_LR_schaefer100.npy
    subject_XXXXX/.../rfMRI_REST1_LR_schaefer200.npy
    subject_XXXXX/.../rfMRI_REST1_LR_schaefer300.npy
    subject_XXXXX/.../rfMRI_REST1_LR_schaefer400.npy

Pipeline:
    fMRI 4D (X,Y,Z,T) → Atlas Schaefer → Moyenne par ROI → (T, N_regions)

Auteur: Dan Abergel
Thèse de Master - Hebrew University of Jerusalem
"""

import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker


# =============================================================================
# CONFIGURATION
# =============================================================================
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")
CLEAN_SUBJECTS_JSON = HCP_ROOT / "subjects_clean.json"

# Résolutions Schaefer à extraire
# Options disponibles: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
SCHAEFER_RESOLUTIONS = [100, 300]  # 200 déjà fait

# Résolution spatiale de l'atlas (doit correspondre aux données fMRI)
ATLAS_RESOLUTION_MM = 2

# Si True, skip les sujets qui ont déjà le fichier .npy
SKIP_EXISTING = True


# =============================================================================
# FONCTIONS
# =============================================================================

def get_nifti_path(subject_id: str) -> Path | None:
    """Retourne le chemin vers le fichier fMRI NIfTI d'un sujet."""
    nifti_path = (
        HCP_ROOT / f"subject_{subject_id}" / "MNINonLinear" / "Results"
        / "rfMRI_REST1_LR" / "rfMRI_REST1_LR.nii.gz"
    )
    return nifti_path if nifti_path.exists() else None


def get_output_path(subject_id: str, n_regions: int) -> Path:
    """Retourne le chemin de sortie pour les time series extraites."""
    return (
        HCP_ROOT / f"subject_{subject_id}" / "MNINonLinear" / "Results"
        / "rfMRI_REST1_LR" / f"rfMRI_REST1_LR_schaefer{n_regions}.npy"
    )


def load_schaefer_atlas(n_regions: int):
    """
    Charge l'atlas Schaefer et crée le masker.

    Args:
        n_regions: Nombre de régions (100, 200, 300, ...)

    Returns:
        NiftiLabelsMasker configuré
    """
    print(f"  Loading Schaefer {n_regions} atlas...")
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_regions,
        resolution_mm=ATLAS_RESOLUTION_MM
    )

    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=False,  # On garde les données brutes
        verbose=0
    )

    return masker


def extract_timeseries(nifti_path: Path, masker: NiftiLabelsMasker) -> np.ndarray:
    """
    Extrait les séries temporelles des ROIs à partir du fMRI 4D.

    Args:
        nifti_path: Chemin vers le fichier NIfTI
        masker: NiftiLabelsMasker configuré

    Returns:
        Array (T, N_regions) des séries temporelles
    """
    timeseries = masker.fit_transform(str(nifti_path))
    return timeseries.astype(np.float32)


def process_subjects(n_regions: int, subject_ids: list) -> dict:
    """
    Extrait les time series pour une résolution Schaefer donnée.

    Args:
        n_regions: Nombre de régions Schaefer
        subject_ids: Liste des IDs de sujets à traiter

    Returns:
        dict avec statistiques (processed, skipped, failed)
    """
    stats = {"processed": 0, "skipped": 0, "failed": 0, "errors": []}

    # Charger l'atlas une seule fois
    masker = load_schaefer_atlas(n_regions)

    # Traiter chaque sujet
    for subject_id in tqdm(subject_ids, desc=f"Schaefer {n_regions}"):

        output_path = get_output_path(subject_id, n_regions)

        # Skip si déjà extrait
        if SKIP_EXISTING and output_path.exists():
            stats["skipped"] += 1
            continue

        # Trouver le fichier NIfTI
        nifti_path = get_nifti_path(subject_id)
        if nifti_path is None:
            stats["failed"] += 1
            stats["errors"].append(f"{subject_id}: NIfTI not found")
            continue

        try:
            # Extraire les time series
            timeseries = extract_timeseries(nifti_path, masker)

            # Sauvegarder
            np.save(output_path, timeseries)
            stats["processed"] += 1

        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append(f"{subject_id}: {str(e)[:50]}")

    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EXTRACTION DES SÉRIES TEMPORELLES SCHAEFER")
    print("=" * 70)

    # Charger la liste des sujets propres
    print("\nChargement de la liste des sujets...")
    with open(CLEAN_SUBJECTS_JSON, "r") as f:
        clean_data = json.load(f)
    subject_ids = clean_data["subject_ids"]
    print(f"Sujets à traiter: {len(subject_ids)}")

    print(f"\nRésolutions à extraire: {SCHAEFER_RESOLUTIONS}")
    print(f"Skip existing: {SKIP_EXISTING}")

    # Traiter chaque résolution
    all_stats = {}

    for n_regions in SCHAEFER_RESOLUTIONS:
        print("\n" + "-" * 70)
        print(f"SCHAEFER {n_regions} RÉGIONS")
        print("-" * 70)

        n_features = n_regions * (n_regions - 1) // 2
        print(f"  → Produira {n_features} features de connectivité")

        stats = process_subjects(n_regions, subject_ids)
        all_stats[n_regions] = stats

        print(f"\n  Résultat:")
        print(f"    Processed: {stats['processed']}")
        print(f"    Skipped:   {stats['skipped']}")
        print(f"    Failed:    {stats['failed']}")

        if stats["errors"] and len(stats["errors"]) <= 5:
            print(f"    Erreurs:")
            for err in stats["errors"]:
                print(f"      - {err}")

    # Résumé final
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    print(f"\n{'Résolution':<15} {'Processed':<12} {'Skipped':<12} {'Failed':<12}")
    print("-" * 50)

    for n_regions, stats in all_stats.items():
        print(f"Schaefer {n_regions:<5} {stats['processed']:<12} {stats['skipped']:<12} {stats['failed']:<12}")

    # Vérifier ce qui est disponible maintenant
    print("\n" + "-" * 70)
    print("FICHIERS DISPONIBLES")
    print("-" * 70)

    all_resolutions = [100, 200, 300, 400, 500]  # Résolutions à vérifier
    for n_regions in all_resolutions:
        # Compter combien de sujets ont ce fichier
        count = sum(
            1 for sid in subject_ids
            if get_output_path(sid, n_regions).exists()
        )
        status = "✓" if count == len(subject_ids) else "⚠️" if count > 0 else "✗"
        print(f"  {status} Schaefer {n_regions}: {count}/{len(subject_ids)} sujets")

    print("\n✓ Extraction terminée!")
    print("  Tu peux maintenant lancer: python -m src.step4_compare_normalizations")


if __name__ == "__main__":
    main()
