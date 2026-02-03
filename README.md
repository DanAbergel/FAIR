# FAIR - fMRI Analysis for Individual Recognition

**Thèse de Master en Data Science**
Hebrew University of Jerusalem (HUJI)
Auteur: Dan Abergel
Superviseur: Ariel Yaffe

---

## Objectif du projet

Prédire des attributs individuels (ex: sexe) à partir de la **connectivité fonctionnelle cérébrale** mesurée par IRMf (fMRI).

Ce projet commence par une **baseline simple** (régression logistique) et évoluera vers des modèles plus complexes (deep learning, transformers).

---

## Concepts clés

### Qu'est-ce que la connectivité fonctionnelle?

Quand tu fais un scan fMRI au repos (resting-state), ton cerveau n'est pas "éteint" - différentes régions continuent de communiquer entre elles. La **connectivité fonctionnelle** mesure la corrélation/covariance entre les signaux de différentes régions cérébrales.

```
Région A: ~~~~∿∿∿~~~~∿∿∿~~~~
Région B: ~~~~∿∿∿~~~~∿∿∿~~~~  → Forte covariance = connectées
Région C: ∿∿∿~~~~∿∿∿~~~~∿∿∿  → Faible covariance = pas connectées
```

### Qu'est-ce qu'un atlas cérébral?

Un atlas divise le cerveau en régions d'intérêt (ROIs). On utilise l'atlas **Schaefer 2018** qui définit 200 régions basées sur la connectivité fonctionnelle (pas juste l'anatomie).

### Pipeline de la baseline

```
fMRI 4D (X,Y,Z,T)
       ↓
   Atlas Schaefer 200
       ↓
Séries temporelles (T × 200)
       ↓
   Matrice de covariance (200 × 200)
       ↓
Triangle supérieur → 19,900 features
       ↓
   Régression logistique
       ↓
    Prédiction (sexe)
```

---

## Structure du projet

```
FAIR/
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
├── src/
│   └── baselines/
│       └── logreg_cov.py    # Baseline régression logistique
├── data/                     # Données (ignoré par git)
├── outputs/                  # Résultats (ignoré par git)
├── configs/                  # Configurations
└── notebooks/                # Exploration Jupyter
```

---

## Installation

```bash
# Cloner le repo
git clone <url> FAIR
cd FAIR

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Données attendues

Le script s'attend à la structure HCP (Human Connectome Project):

```
HCP_ROOT/
├── model_input/
│   └── imageID_to_labels.json    # Labels (sexe, âge, etc.)
└── subject_XXXXX/
    └── MNINonLinear/
        └── Results/
            └── rfMRI_REST1_LR/
                └── rfMRI_REST1_LR.nii.gz   # Scan fMRI 4D
```

### Format du fichier de labels

```json
{
  "100206_REST1_LR": {
    "Sex_Binary": 0
  },
  "100307_REST1_LR": {
    "Sex_Binary": 1
  }
}
```

---

## Configuration

Édite les constantes au début de `src/baselines/logreg_cov.py`:

```python
# Chemin vers tes données HCP
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")

# Nombre de régions (100, 200, 300, ... 1000)
N_REGIONS = 200

# Nombre de folds pour la cross-validation
N_SPLITS = 5

# Utiliser le cache des time series (True après le 1er run)
USE_CACHED_SCHAEFER = True
```

---

## Exécution

```bash
python -m src.baselines.logreg_cov
```

### Output attendu

```
Loading labels...
Loading Schaefer 200 atlas...
Found 1000 subject directories
Processing subjects: 100%|██████████| 1000/1000

==================================================
Dataset: X = (950, 19900), y = (950,)
Subjects: 950
Features: 19900
Class distribution: [450 500]
==================================================

Running 5-fold cross-validation...

==================================================
RESULTS
==================================================
ROC-AUC per fold: [0.82  0.85  0.83  0.81  0.84]
Mean ROC-AUC: 0.830 ± 0.015
==================================================
```

---

## Interprétation des résultats

| ROC-AUC | Interprétation |
|---------|----------------|
| 0.50 | Hasard (le modèle ne prédit rien) |
| 0.60-0.70 | Faible |
| 0.70-0.80 | Correct |
| 0.80-0.90 | Bon |
| 0.90+ | Excellent |

Pour la prédiction du sexe à partir de la connectivité fMRI, on s'attend à **ROC-AUC ~ 0.80-0.90** avec cette baseline.

---

## Glossaire

| Terme | Définition |
|-------|------------|
| **fMRI** | Functional Magnetic Resonance Imaging - mesure l'activité cérébrale via le flux sanguin |
| **Resting-state** | Scan fMRI où le sujet ne fait rien (yeux fermés, au repos) |
| **ROI** | Region of Interest - une zone du cerveau |
| **Covariance** | Mesure de co-variation entre deux variables |
| **Cross-validation** | Technique pour évaluer un modèle en le testant sur plusieurs splits |
| **ROC-AUC** | Area Under the ROC Curve - métrique de classification (0.5=hasard, 1.0=parfait) |
| **Baseline** | Modèle simple de référence pour comparer les modèles plus complexes |
| **Data leakage** | Erreur où l'info du test set fuit dans l'entraînement |

---

## Prochaines étapes

- [ ] Ajouter d'autres scans (REST1_RL, REST2_LR, REST2_RL)
- [ ] Tester corrélation vs covariance
- [ ] Hyperparameter tuning (C, nombre de régions)
- [ ] Modèle deep learning (MLP, CNN sur matrices)
- [ ] Transformer sur les séries temporelles
- [ ] Prédiction d'autres variables (âge, QI, etc.)

---

## Références

- Schaefer et al. (2018) - "Local-Global Parcellation of the Human Cerebral Cortex"
- Human Connectome Project (HCP) - https://www.humanconnectome.org/
- nilearn documentation - https://nilearn.github.io/
