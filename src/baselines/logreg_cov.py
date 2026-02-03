"""
HCP REST1_LR → Schaefer 200 → Covariance → Logistic Regression
=============================================================

Ce script est une BASELINE (modèle de référence) pour la classification
de données fMRI. Il prédit le sexe d'un sujet à partir de la connectivité
fonctionnelle de son cerveau.

PIPELINE:
=========
1. Charger les données fMRI (4D NIfTI: X, Y, Z, Time)
2. Appliquer l'atlas Schaefer pour extraire 200 régions d'intérêt (ROIs)
3. Obtenir une série temporelle par ROI: matrice (T x 200)
4. Calculer la matrice de covariance entre ROIs: (200 x 200)
5. Vectoriser le triangle supérieur: vecteur de 19,900 features
6. Entraîner une régression logistique avec cross-validation

POURQUOI CETTE APPROCHE?
========================
- La COVARIANCE entre deux régions mesure comment elles "co-varient" dans le temps
- Si deux régions ont une covariance élevée, elles sont fonctionnellement connectées
- La matrice de covariance capture donc la "connectivité fonctionnelle" du cerveau
- C'est une approche classique et interprétable en neuroimagerie

POURQUOI UNE BASELINE?
======================
- Avant d'utiliser des modèles complexes (deep learning, transformers...),
  il faut établir une référence avec un modèle simple
- Si un modèle complexe ne bat pas la baseline, il n'est pas utile
- La régression logistique est simple, rapide, et interprétable

RÉSULTATS (HCP 1200, Schaefer 200, 5-fold CV)
=============================================

Sans filtrage (967 sujets, USE_CLEAN_SUBJECTS=False):
    ROC-AUC per fold: [0.825, 0.832, 0.817, 0.857, 0.841]
    Mean ROC-AUC: 0.834 ± 0.014

Avec filtrage outliers (945 sujets, USE_CLEAN_SUBJECTS=True):
    ROC-AUC per fold: [à mesurer]
    Mean ROC-AUC: [à mesurer]

→ C'est la référence à battre pour les modèles plus complexes.

Auteur: Dan Abergel
Thèse de Master - Hebrew University of Jerusalem
Superviseur: Ariel Yaffe
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

# nilearn: bibliothèque spécialisée pour l'analyse de neuroimagerie
# - datasets: télécharge automatiquement des atlas cérébraux
# - maskers: extrait des signaux de régions d'intérêt
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# scikit-learn: bibliothèque de machine learning classique
# - Pipeline: enchaîne preprocessing + modèle (évite les fuites de données)
# - StandardScaler: normalise les features (moyenne=0, std=1)
# - LogisticRegression: classificateur linéaire pour problèmes binaires
# - StratifiedKFold: découpe les données en K folds en gardant les proportions de classes
# - cross_val_score: évalue le modèle sur chaque fold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


# =============================================================================
# CONFIGURATION
# =============================================================================
# Ces constantes définissent les paramètres de l'expérience.
# Tu peux les modifier pour tester différentes configurations.

# Chemin vers les données HCP sur le cluster
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")

# Fichier JSON contenant les labels (sexe) pour chaque sujet
LABELS_JSON = HCP_ROOT / "model_input" / "imageID_to_labels.json"

# Fichier JSON contenant la liste des sujets propres (sans outliers)
# Généré par step2_explore_population.py
CLEAN_SUBJECTS_JSON = HCP_ROOT / "subjects_clean.json"

# =============================================================================
# OPTION: Filtrage des sujets
# =============================================================================
# True  = Utilise uniquement les 945 sujets propres (subjects_clean.json)
#         → Exclut les 22 sujets avec variance/moyenne extrême
# False = Utilise tous les 967 sujets (comportement original)
#         → Résultat: ROC-AUC = 0.834 ± 0.014
USE_CLEAN_SUBJECTS = True

# Nombre de régions dans l'atlas Schaefer
# Options: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
# Plus de régions = plus de précision spatiale mais plus de features
# 200 régions → 200*(200-1)/2 = 19,900 features (triangle supérieur)
N_REGIONS = 200

# Résolution de l'atlas en mm (doit correspondre à tes données fMRI)
ATLAS_RESOLUTION_MM = 2

# Nombre de folds pour la cross-validation
# 5-fold CV: on découpe les données en 5 parties
# On entraîne sur 4 parties, on teste sur 1, et on répète 5 fois
# Cela donne une estimation plus robuste de la performance
N_SPLITS = 5

# Graine aléatoire pour la reproductibilité
# Avec la même graine, tu obtiendras toujours les mêmes résultats
RANDOM_STATE = 42

# Si True, charge les time series pré-calculées (.npy) au lieu de
# recalculer à partir des NIfTI (beaucoup plus rapide après le 1er run)
USE_CACHED_SCHAEFER = True


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_rest1_lr_nifti(subject_dir: Path) -> Path | None:
    """
    Trouve le fichier NIfTI du scan REST1_LR pour un sujet.

    Structure attendue des données HCP:
        subject_XXXXX/
            MNINonLinear/
                Results/
                    rfMRI_REST1_LR/
                        rfMRI_REST1_LR.nii.gz  <-- Ce fichier

    Args:
        subject_dir: Chemin vers le dossier du sujet (ex: /path/subject_100206)

    Returns:
        Path vers le fichier NIfTI, ou None s'il n'existe pas

    Note:
        REST1_LR = Resting-state scan 1, Left-to-Right phase encoding
        HCP a 4 scans resting-state: REST1_LR, REST1_RL, REST2_LR, REST2_RL
    """
    nifti_path = (
        subject_dir
        / "MNINonLinear"
        / "Results"
        / "rfMRI_REST1_LR"
        / "rfMRI_REST1_LR.nii.gz"
    )
    return nifti_path if nifti_path.exists() else None


def save_schaefer_timeseries(timeseries: np.ndarray, nifti_path: Path) -> Path:
    """
    Sauvegarde les séries temporelles Schaefer en format .npy.

    Pourquoi sauvegarder?
    - L'extraction des time series depuis un NIfTI 4D est LENTE
    - En sauvegardant, on peut recharger instantanément lors des runs suivants
    - Format .npy: format binaire numpy, très rapide à charger

    Args:
        timeseries: Array de shape (T, N_REGIONS) où T = nombre de timepoints
        nifti_path: Chemin du NIfTI original (pour savoir où sauvegarder)

    Returns:
        Path vers le fichier .npy créé
    """
    out_path = nifti_path.parent / f"rfMRI_REST1_LR_schaefer{N_REGIONS}.npy"

    # float32 au lieu de float64 pour économiser de l'espace disque
    # Précision suffisante pour notre application
    np.save(out_path, timeseries.astype(np.float32))

    return out_path


def load_schaefer_timeseries(nifti_path: Path) -> np.ndarray | None:
    """
    Charge les séries temporelles Schaefer pré-calculées si elles existent.

    Args:
        nifti_path: Chemin du NIfTI original

    Returns:
        Array (T, N_REGIONS) si le cache existe, sinon None
    """
    schaefer_path = nifti_path.parent / f"rfMRI_REST1_LR_schaefer{N_REGIONS}.npy"

    if schaefer_path.exists():
        return np.load(schaefer_path)

    return None


def covariance_features(ts: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de covariance et extrait le triangle supérieur.

    ÉTAPES:
    1. Calcul de la covariance entre toutes les paires de ROIs
    2. Extraction du triangle supérieur (sans la diagonale)

    POURQUOI LE TRIANGLE SUPÉRIEUR?
    - La matrice de covariance est SYMÉTRIQUE: cov(A,B) = cov(B,A)
    - La diagonale contient les VARIANCES (pas les covariances)
    - Donc seul le triangle supérieur contient l'information utile
    - Pour N=200 régions: N*(N-1)/2 = 19,900 features uniques

    COVARIANCE vs CORRÉLATION:
    - Covariance: mesure brute de co-variation (dépend de l'échelle)
    - Corrélation: covariance normalisée entre -1 et 1
    - On utilise la covariance ici, mais la corrélation marche aussi

    Args:
        ts: Séries temporelles de shape (T, N_REGIONS)
            T = nombre de timepoints (ex: 1200 pour HCP)
            N_REGIONS = nombre de ROIs (ex: 200)

    Returns:
        Vecteur 1D de shape (N_REGIONS*(N_REGIONS-1)/2,) = (19900,) pour N=200

    Exemple visuel pour N=4 régions:
        Matrice de covariance:
        [a  b  c  d]
        [b  e  f  g]     Triangle supérieur (k=1): [b, c, d, f, g, h]
        [c  f  h  i]     → 6 valeurs = 4*3/2
        [d  g  i  j]
    """
    # np.cov calcule la matrice de covariance
    # rowvar=False signifie que les colonnes sont les variables (ROIs)
    # et les lignes sont les observations (timepoints)
    cov = np.cov(ts, rowvar=False)

    # np.triu_indices retourne les indices du triangle supérieur
    # k=1 signifie qu'on exclut la diagonale (k=0 inclurait la diagonale)
    idx = np.triu_indices(cov.shape[0], k=1)

    # Extraction et conversion en float32 pour économiser la mémoire
    return cov[idx].astype(np.float32)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main() -> None:
    """
    Point d'entrée du script.

    Exécute le pipeline complet:
    1. Chargement des labels
    2. Chargement de l'atlas Schaefer
    3. Construction du dataset (X, y)
    4. Entraînement et évaluation par cross-validation
    """

    # -------------------------------------------------------------------------
    # ÉTAPE 1: Charger les labels
    # -------------------------------------------------------------------------
    # Le fichier JSON contient pour chaque scan:
    # - "SUBJECTID_REST1_LR": {"Sex_Binary": 0 ou 1, ...}

    print("Loading labels...")
    with open(LABELS_JSON, "r") as f:
        labels_dict = json.load(f)

    # -------------------------------------------------------------------------
    # ÉTAPE 2: Charger l'atlas Schaefer
    # -------------------------------------------------------------------------
    # L'atlas Schaefer parcellise le cerveau en N régions fonctionnelles.
    # nilearn le télécharge automatiquement depuis internet (une seule fois).
    #
    # POURQUOI SCHAEFER?
    # - Atlas basé sur la connectivité fonctionnelle (pas juste anatomique)
    # - Très utilisé dans la recherche en neuroimagerie
    # - Disponible en plusieurs résolutions (100 à 1000 régions)
    # - Organisé en 7 réseaux fonctionnels (visual, somatomotor, attention, etc.)

    print(f"Loading Schaefer {N_REGIONS} atlas...")
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_REGIONS,
        resolution_mm=ATLAS_RESOLUTION_MM
    )

    # NiftiLabelsMasker: outil pour extraire les signaux moyens de chaque ROI
    # Pour chaque ROI, il calcule la moyenne des voxels à chaque timepoint
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,  # Image NIfTI de l'atlas
        standardize=False,       # On standardise plus tard dans le pipeline
        verbose=0                # Pas de messages pendant l'extraction
    )

    # -------------------------------------------------------------------------
    # ÉTAPE 3: Construire le dataset
    # -------------------------------------------------------------------------
    # Pour chaque sujet:
    # - Charger (ou extraire) les séries temporelles des ROIs
    # - Calculer les features de covariance
    # - Récupérer le label (sexe)

    X_list = []  # Liste des vecteurs de features
    y_list = []  # Liste des labels

    # Trouver les sujets à utiliser
    if USE_CLEAN_SUBJECTS:
        # Charger la liste des sujets propres (sans outliers)
        if not CLEAN_SUBJECTS_JSON.exists():
            raise FileNotFoundError(
                f"Fichier {CLEAN_SUBJECTS_JSON} introuvable.\n"
                "Lance d'abord: python -m src.step2_explore_population"
            )
        with open(CLEAN_SUBJECTS_JSON, "r") as f:
            clean_data = json.load(f)
        clean_ids = set(clean_data["subject_ids"])
        subject_dirs = [
            HCP_ROOT / f"subject_{sid}"
            for sid in clean_data["subject_ids"]
            if (HCP_ROOT / f"subject_{sid}").exists()
        ]
        print(f"Mode: CLEAN SUBJECTS (USE_CLEAN_SUBJECTS=True)")
        print(f"Found {len(subject_dirs)} clean subject directories (excluded {967 - len(subject_dirs)} outliers)")
    else:
        # Utiliser tous les sujets
        subject_dirs = sorted(HCP_ROOT.glob("subject_*"))
        clean_ids = None  # Pas de filtrage
        print(f"Mode: ALL SUBJECTS (USE_CLEAN_SUBJECTS=False)")
        print(f"Found {len(subject_dirs)} subject directories")

    # tqdm affiche une barre de progression
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):

        # Extraire l'ID du sujet depuis le nom du dossier
        # "subject_100206" → "100206"
        subject_id = subject_dir.name.replace("subject_", "")

        # Clé pour chercher dans le dictionnaire de labels
        scan_key = f"{subject_id}_REST1_LR"

        # Vérifier que ce scan a un label
        if scan_key not in labels_dict:
            continue

        # Vérifier que le fichier NIfTI existe
        nifti_path = get_rest1_lr_nifti(subject_dir)
        if nifti_path is None:
            continue

        try:
            # Essayer de charger les time series depuis le cache
            timeseries = None
            if USE_CACHED_SCHAEFER:
                timeseries = load_schaefer_timeseries(nifti_path)

            # Si pas de cache, extraire depuis le NIfTI
            if timeseries is None:
                # fit_transform: applique l'atlas et extrait les signaux
                # Résultat: array (T, N_REGIONS) = (1200, 200) pour HCP
                timeseries = masker.fit_transform(nifti_path)

                # Sauvegarder pour les prochains runs
                save_schaefer_timeseries(timeseries, nifti_path)

            # Calculer les features de covariance
            features = covariance_features(timeseries)

            # Récupérer le label (0 = Male, 1 = Female, ou l'inverse selon ton JSON)
            sex = labels_dict[scan_key]["Sex_Binary"]

            X_list.append(features)
            y_list.append(sex)

        except Exception as e:
            # En cas d'erreur (fichier corrompu, etc.), on skip ce sujet
            print(f"[ERROR] {subject_dir.name}: {e}")
            continue

    # Convertir les listes en arrays numpy
    # X: shape (n_subjects, n_features) = (N, 19900)
    # y: shape (n_subjects,) = (N,)
    X = np.stack(X_list)
    y = np.array(y_list)

    # Afficher les statistiques du dataset
    print("=" * 50)
    print(f"Dataset: X = {X.shape}, y = {y.shape}")
    print(f"Subjects: {len(y)}")
    print(f"Features: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # ÉTAPE 4: Machine Learning
    # -------------------------------------------------------------------------

    # PIPELINE SKLEARN
    # ================
    # Un Pipeline enchaîne plusieurs étapes de manière propre.
    # Avantage crucial: le scaler est fit uniquement sur les données d'entraînement
    # de chaque fold, évitant ainsi les "fuites de données" (data leakage).
    #
    # DATA LEAKAGE: erreur commune où de l'information du test set "fuit"
    # dans l'entraînement. Par exemple, si on normalise TOUT le dataset
    # avant de splitter, le test set influence la normalisation.

    pipeline = Pipeline([
        # Étape 1: StandardScaler
        # Normalise chaque feature pour avoir moyenne=0 et std=1
        # Important car la régression logistique est sensible à l'échelle
        ("scaler", StandardScaler()),

        # Étape 2: Régression Logistique
        ("classifier", LogisticRegression(
            penalty="l2",           # Régularisation L2 (Ridge) pour éviter l'overfitting
            C=1.0,                  # Force de régularisation (1/lambda). Plus petit = plus de régularisation
            max_iter=5000,          # Nombre max d'itérations pour converger
            class_weight="balanced", # Ajuste les poids pour gérer le déséquilibre de classes
            solver="lbfgs",         # Algorithme d'optimisation (bon pour données moyennes)
            n_jobs=-1               # Utilise tous les CPU disponibles
        ))
    ])

    # CROSS-VALIDATION STRATIFIÉE
    # ===========================
    # Pourquoi "stratifiée"? Elle maintient la proportion de classes dans chaque fold.
    # Si tu as 60% hommes et 40% femmes, chaque fold aura ~60/40.
    # Sans stratification, un fold pourrait avoir 80/20 par malchance.

    cv = StratifiedKFold(
        n_splits=N_SPLITS,      # 5 folds
        shuffle=True,           # Mélange les données avant de splitter
        random_state=RANDOM_STATE  # Pour la reproductibilité
    )

    # ÉVALUATION
    # ==========
    # cross_val_score fait tout automatiquement:
    # 1. Pour chaque fold i:
    #    a. Entraîne le pipeline sur les folds != i
    #    b. Prédit sur le fold i
    #    c. Calcule le score (ici ROC-AUC)
    # 2. Retourne les 5 scores

    print(f"\nRunning {N_SPLITS}-fold cross-validation...")
    scores = cross_val_score(
        pipeline,
        X, y,
        cv=cv,
        scoring="roc_auc"  # ROC-AUC: mesure la capacité à distinguer les classes
                           # 0.5 = hasard, 1.0 = parfait
    )

    # RÉSULTATS
    # =========
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    mode_str = "CLEAN (945 sujets)" if USE_CLEAN_SUBJECTS else "ALL (967 sujets)"
    print(f"Mode: {mode_str}")
    print(f"ROC-AUC per fold: {np.round(scores, 3)}")
    print(f"Mean ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    print("=" * 50)

    # INTERPRÉTATION
    # ==============
    # - ROC-AUC ~ 0.5: Le modèle ne fait pas mieux que le hasard
    # - ROC-AUC ~ 0.7: Performance correcte
    # - ROC-AUC ~ 0.8: Bonne performance
    # - ROC-AUC ~ 0.9+: Excellente performance
    #
    # Pour la prédiction du sexe à partir de la connectivité fMRI,
    # on s'attend généralement à ROC-AUC ~ 0.8-0.9


if __name__ == "__main__":
    main()
