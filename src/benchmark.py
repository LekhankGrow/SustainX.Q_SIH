from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import load_dataset
from src.metrics import binary_metrics

RANDOM_STATE = 26139
ARTIFACT_DIR = Path("artifacts")


def classical_estimators() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "RBF SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
        ]),
    }


def grouped_folds(y: pd.Series | np.ndarray, groups: pd.Series | np.ndarray, n_splits: int = 5):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def evaluate_classical(n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    X, y, groups, profile = load_dataset()
    folds = grouped_folds(y, groups, n_splits)
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for fold_index, (train_indices, test_indices) in enumerate(folds, start=1):
        train_subjects = set(groups.iloc[train_indices])
        test_subjects = set(groups.iloc[test_indices])
        if train_subjects.intersection(test_subjects):
            raise AssertionError("Subject leakage: a subject appears in both sides of a fold")
        if y.iloc[train_indices].nunique() < 2 or y.iloc[test_indices].nunique() < 2:
            raise ValueError(f"Fold {fold_index} does not contain both target classes")
        for model_name, estimator in classical_estimators().items():
            started = time.perf_counter()
            estimator.fit(X.iloc[train_indices], y.iloc[train_indices])
            training_seconds = time.perf_counter() - started
            inference_started = time.perf_counter()
            scores = positive_scores(estimator, X.iloc[test_indices])
            inference_seconds = time.perf_counter() - inference_started
            importance = permutation_importance(
                estimator,
                X.iloc[test_indices],
                y.iloc[test_indices],
                scoring="roc_auc",
                n_repeats=5,
                random_state=RANDOM_STATE + fold_index,
                n_jobs=1,
            )
            importance_rows.extend(
                {"model": model_name, "feature": feature_name, "fold": fold_index, "importance_mean": float(mean)}
                for feature_name, mean in zip(X.columns, importance.importances_mean, strict=True)
            )
            fold_predictions = pd.DataFrame({
                "subject": groups.iloc[test_indices].to_numpy(),
                "truth": y.iloc[test_indices].to_numpy(),
                "score": scores,
            }).groupby("subject", as_index=False).agg(truth=("truth", "first"), score=("score", "mean"))
            metrics = binary_metrics(fold_predictions["truth"].to_numpy(), fold_predictions["score"].to_numpy())
            fold_rows.append({
                "model": model_name,
                "fold": fold_index,
                **metrics,
                "training_seconds": training_seconds,
                "inference_ms_per_recording": inference_seconds * 1000 / len(test_indices),
                "train_subjects": len(train_subjects),
                "test_subjects": len(test_subjects),
            })
            for index, score in zip(test_indices, scores, strict=True):
                prediction_rows.append({
                    "model": model_name,
                    "fold": fold_index,
                    "subject": groups.iloc[index],
                    "truth": int(y.iloc[index]),
                    "score": float(score),
                })

    folds_frame = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summaries: dict[str, Any] = {}
    for model_name, model_rows in predictions.groupby("model"):
        subject_predictions = model_rows.groupby("subject", as_index=False).agg(truth=("truth", "first"), score=("score", "mean"))
        overall = binary_metrics(subject_predictions["truth"].to_numpy(), subject_predictions["score"].to_numpy())
        per_fold = folds_frame[folds_frame["model"] == model_name]
        summaries[model_name] = {
            **overall,
            "training_seconds_mean": float(per_fold["training_seconds"].mean()),
            "inference_ms_per_recording_mean": float(per_fold["inference_ms_per_recording"].mean()),
            "fold_metrics_mean": {metric: _finite_mean(per_fold[metric]) for metric in overall},
            "fold_metrics_std": {metric: _finite_std(per_fold[metric]) for metric in overall},
        }
    metadata = {
        "dataset": profile.__dict__,
        "evaluation": {
            "method": "5-fold stratified group cross-validation; all recordings from a person remain in one fold; metrics computed after averaging recording scores per person",
            "random_state": RANDOM_STATE,
            "folds": len(folds),
            "subject_disjoint": True,
            "probability_note": "SVM decision scores are monotonically rescaled for comparison; none of the model outputs are clinically calibrated probabilities.",
        },
        "classical_models": summaries,
        "explainability": {
            "method": "held-out-fold permutation importance using ROC-AUC",
            "feature_importance_mean": (
                pd.DataFrame(importance_rows)
                .groupby(["model", "feature"])["importance_mean"]
                .mean()
                .reset_index()
                .sort_values(["model", "importance_mean"], ascending=[True, False])
                .to_dict(orient="records")
            ),
            "caveat": "Predictive associations on this small dataset are not causal or biological explanations.",
        },
    }
    return predictions, folds_frame, metadata


def _finite_mean(values: pd.Series) -> float | None:
    finite = values.dropna()
    return float(finite.mean()) if not finite.empty else None


def _finite_std(values: pd.Series) -> float | None:
    finite = values.dropna()
    return float(finite.std(ddof=1)) if len(finite) > 1 else None


def positive_scores(estimator, features: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    decision = np.asarray(estimator.decision_function(features), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -40.0, 40.0)))


def save_classical_results(predictions: pd.DataFrame, folds: pd.DataFrame, metadata: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(ARTIFACT_DIR / "classical_predictions.csv", index=False)
    folds.to_csv(ARTIFACT_DIR / "classical_folds.csv", index=False)
    (ARTIFACT_DIR / "benchmark.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")


def assess_quantum_utility(classical: pd.DataFrame, quantum: pd.DataFrame, bootstrap_samples: int = 2000) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    def by_subject(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.groupby("subject", as_index=False).agg(truth=("truth", "first"), score=("score", "mean"))

    classical_subjects = by_subject(classical[classical["model"] == "RBF SVM"]).set_index("subject").sort_index()
    quantum_subjects = by_subject(quantum).set_index("subject").sort_index()
    if not classical_subjects.index.equals(quantum_subjects.index):
        return {"status": "Insufficient evidence", "reason": "The model outputs do not cover identical subjects."}
    if not np.array_equal(classical_subjects["truth"].to_numpy(), quantum_subjects["truth"].to_numpy()):
        return {"status": "Insufficient evidence", "reason": "The evaluated subject labels do not match."}
    labels = classical_subjects["truth"].to_numpy(dtype=int)
    if np.unique(labels).size != 2 or min(np.bincount(labels)) < 2:
        return {"status": "Insufficient evidence", "reason": "Too few subjects from one class for paired ROC-AUC assessment."}

    classical_scores = classical_subjects["score"].to_numpy(dtype=float)
    quantum_scores = quantum_subjects["score"].to_numpy(dtype=float)
    delta = float(roc_auc_score(labels, quantum_scores) - roc_auc_score(labels, classical_scores))
    rng = np.random.default_rng(RANDOM_STATE)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    deltas: list[float] = []
    for _ in range(bootstrap_samples):
        sample = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in class_indices])
        deltas.append(float(roc_auc_score(labels[sample], quantum_scores[sample]) - roc_auc_score(labels[sample], classical_scores[sample])))
    lower, upper = (float(value) for value in np.quantile(deltas, [0.025, 0.975]))
    practical_margin = 0.05
    if delta >= practical_margin and lower > 0.0:
        status = "Measurable benefit observed"
        reason = "The paired subject bootstrap interval is above zero and the observed ROC-AUC gain meets the pre-set 0.05 margin."
    elif upper < practical_margin:
        status = "No measurable benefit observed"
        reason = "The paired subject bootstrap interval does not reach the pre-set 0.05 ROC-AUC improvement margin."
    else:
        status = "Insufficient evidence"
        reason = "The paired subject bootstrap interval does not resolve whether the pre-set 0.05 ROC-AUC margin is met."
    return {
        "status": status,
        "reason": reason,
        "comparator": "RBF SVM",
        "metric": "paired subject-level ROC-AUC difference (quantum minus classical)",
        "observed_delta": delta,
        "bootstrap_95_percent_interval": [lower, upper],
        "practical_margin": practical_margin,
        "bootstrap_samples": bootstrap_samples,
        "subject_count": int(len(labels)),
        "caveat": "Exploratory cross-validation comparison on a small dataset; bootstrap interval does not include full model-training or external-validation uncertainty.",
    }