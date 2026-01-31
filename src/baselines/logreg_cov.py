"""Logistic regression baseline on covariance features from ROI time series."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_timeseries(path: str, atlas_nifti: str | None) -> np.ndarray:
    """Load ROI time series of shape (T, N) from npy/npz/csv or 4D NIfTI."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".nii", ".gz"} or path.endswith(".nii.gz"):
        if not atlas_nifti:
            raise ValueError("atlas_nifti is required for 4D NIfTI inputs")
        img = nib.load(path)
        atlas_img = nib.load(atlas_nifti)
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
        ts = masker.fit_transform(img)
    elif ext == ".npy":
        ts = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        if "ts" in data:
            ts = data["ts"]
        elif "timeseries" in data:
            ts = data["timeseries"]
        else:
            raise ValueError(f"{path}: expected key 'ts' or 'timeseries' in npz")
    elif ext == ".csv":
        ts = np.genfromtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if ts.ndim != 2:
        raise ValueError(f"{path}: expected 2D array, got shape {ts.shape}")
    return ts


def covariance_features(ts: np.ndarray, use_corr: bool = False) -> np.ndarray:
    """Compute covariance (or correlation) and vectorize upper triangle."""
    if use_corr:
        mat = np.corrcoef(ts, rowvar=False)
    else:
        mat = np.cov(ts, rowvar=False)

    n = mat.shape[0]
    tri = np.triu_indices(n, k=1)
    return mat[tri].astype(np.float32, copy=False)


def read_labels(labels_csv: str, id_col: str, label_col: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj_id = str(row[id_col]).strip()
            label = str(row[label_col]).strip()
            if subj_id:
                labels[subj_id] = label
    return labels


def build_dataset(
    data_dir: str,
    labels: Dict[str, str],
    file_pattern: str,
    use_corr: bool,
    atlas_nifti: str | None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    used_ids: List[str] = []

    for subj_id, label in labels.items():
        fname = file_pattern.format(id=subj_id)
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        ts = load_timeseries(path, atlas_nifti)
        feat = covariance_features(ts, use_corr=use_corr)
        X_list.append(feat)
        y_list.append(label)
        used_ids.append(subj_id)

    if not X_list:
        raise RuntimeError("No subject files matched the labels")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y, used_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--id-col", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--file-pattern", default="{id}.npy")
    parser.add_argument("--atlas-nifti", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-corr", action="store_true")
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    labels = read_labels(args.labels_csv, args.id_col, args.label_col)
    X, y, used_ids = build_dataset(
        args.data_dir, labels, args.file_pattern, args.use_corr, args.atlas_nifti
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=args.max_iter,
        class_weight=None if args.class_weight.lower() == "none" else args.class_weight,
        n_jobs=1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Subjects used: {len(used_ids)}")
    print(f"Features: {X.shape[1]}")
    print(f"Accuracy: {acc:.4f}")

    if len(np.unique(y)) == 2:
        y_proba = clf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"ROC-AUC: {auc:.4f}")
        except ValueError:
            pass


if __name__ == "__main__":
    main()
