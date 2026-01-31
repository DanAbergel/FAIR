# Thesis workspace

This folder is a clean workspace for the thesis. It starts with a baseline
logistic regression on covariance features from Schaefer-parcellated fMRI
(HCP 1200), and is structured to grow later.

## Structure
- `data/` raw or preprocessed inputs (ignored by git)
- `outputs/` model artifacts, metrics, plots (ignored by git)
- `src/` code
- `configs/` run configs (optional)
- `notebooks/` exploration

## Baseline: Logistic regression on covariance features

### Expected input
- Per-subject 4D fMRI NIfTI `(X, Y, Z, T)` or precomputed ROI time-series `(T, N)`.
- A Schaefer atlas NIfTI if using 4D fMRI.
- A labels CSV containing subject IDs and target label (e.g. sex).

### Example
```bash
python -m src.baselines.logreg_cov \
  --data-dir data/timeseries \
  --labels-csv data/labels.csv \
  --id-col subject_id \
  --label-col sex \
  --file-pattern "{id}.nii.gz" \
  --atlas-nifti data/atlas/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz \
  --test-size 0.2 \
  --seed 42
```

This will:
1) Load each subject time series
2) Compute covariance matrix (ROIs x ROIs)
3) Vectorize the upper triangle
4) Train logistic regression and report accuracy/AUC

## Next steps (later)
- Add transformer model and bottleneck head
- Add standardized preprocessing
- Add train/val/test splits and CV
