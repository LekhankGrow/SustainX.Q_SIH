from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from src.metrics import binary_metrics
from src.research_data import ResearchDataset, load_alzheimer_dataset, load_als_dataset, load_depression_dataset, load_essential_tremor_dataset, load_fall_event_dataset, load_stroke_history_dataset

ARTIFACT_ROOT = Path("artifacts/research")
RANDOM_STATE = 26139
CLASSIFICATION_LOADERS = {
    "alzheimer": load_alzheimer_dataset,
    "depression": load_depression_dataset,
    "stroke_history": load_stroke_history_dataset,
    "fall_event": load_fall_event_dataset,
    "als": load_als_dataset,
}


def _preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_columns))
    if not transformers:
        raise ValueError("Research dataset has no usable feature columns")
    return ColumnTransformer(transformers, remainder="drop")


def _classification_estimators(features: pd.DataFrame) -> dict[str, Pipeline]:
    preprocessor = _preprocessor(features)
    # Gene-expression input is wide relative to its cohort; selection is trained inside each fold.
    selector: Any = SelectKBest(f_classif, k=min(200, features.shape[1])) if features.shape[1] > 1000 else "passthrough"
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessor),
            ("select", selector),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "RBF SVM": Pipeline([
            ("preprocess", _preprocessor(features)),
            ("select", SelectKBest(f_classif, k=min(200, features.shape[1])) if features.shape[1] > 1000 else "passthrough"),
            ("model", SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", _preprocessor(features)),
            ("select", SelectKBest(f_classif, k=min(200, features.shape[1])) if features.shape[1] > 1000 else "passthrough"),
            ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE)),
        ]),
    }


def _folds(dataset: ResearchDataset, n_splits: int = 5):
    if dataset.task == "classification":
        subject_labels = dataset.y.groupby(dataset.groups).agg(lambda values: values.mode().iloc[0])
        minority_subjects = int(subject_labels.value_counts().min())
        effective_splits = min(n_splits, dataset.groups.nunique(), minority_subjects)
        if effective_splits < 2:
            raise ValueError(f"{dataset.module_id} has too few labeled subjects for grouped classification")
        splitter = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(splitter.split(dataset.X, dataset.y, dataset.groups))
    return list(GroupKFold(n_splits=min(n_splits, dataset.groups.nunique())).split(dataset.X, dataset.y, dataset.groups))


def _positive_score(estimator, features: pd.DataFrame) -> np.ndarray:
    model = estimator.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    decisions = np.asarray(estimator.decision_function(features), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(decisions, -40, 40)))


def _regression_estimators(features: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "Ridge": Pipeline([
            ("preprocess", _preprocessor(features)),
            ("model", Ridge(alpha=10.0)),
        ]),
        "RBF SVR": Pipeline([
            ("preprocess", _preprocessor(features)),
            ("model", SVR(C=10.0, epsilon=5.0, kernel="rbf")),
        ]),
        "Random Forest Regressor": Pipeline([
            ("preprocess", _preprocessor(features)),
            ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate_classification(dataset: ResearchDataset, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    folds = _folds(dataset, n_splits)
    estimator_factories = _classification_estimators
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (train_indices, test_indices) in enumerate(folds, start=1):
        train_groups = set(dataset.groups.iloc[train_indices])
        test_groups = set(dataset.groups.iloc[test_indices])
        if train_groups.intersection(test_groups):
            raise AssertionError(f"Group leakage in {dataset.module_id} fold {fold_index}")
        y_train = dataset.y.iloc[train_indices]
        y_test = dataset.y.iloc[test_indices]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise ValueError(f"{dataset.module_id} fold {fold_index} lacks a target class")
        for model_name, estimator in estimator_factories(dataset.X).items():
            started = time.perf_counter()
            estimator.fit(dataset.X.iloc[train_indices], y_train)
            training_seconds = time.perf_counter() - started
            inference_started = time.perf_counter()
            scores = _positive_score(estimator, dataset.X.iloc[test_indices])
            inference_seconds = time.perf_counter() - inference_started
            metrics = binary_metrics(y_test.to_numpy(dtype=int), scores)
            fold_rows.append({
                "model": model_name,
                "fold": fold_index,
                **metrics,
                "training_seconds": float(training_seconds),
                "inference_ms_per_sample": float(inference_seconds * 1000 / len(test_indices)),
                "train_subjects": len(train_groups),
                "test_subjects": len(test_groups),
            })
            for row_index, score in zip(test_indices, scores, strict=True):
                prediction_rows.append({
                    "module_id": dataset.module_id,
                    "model": model_name,
                    "fold": fold_index,
                    "subject": str(dataset.groups.iloc[row_index]),
                    "truth": int(dataset.y.iloc[row_index]),
                    "score": float(score),
                })
    predictions = pd.DataFrame(prediction_rows)
    fold_frame = pd.DataFrame(fold_rows)
    scores_by_model = {}
    for model_name, model_predictions in predictions.groupby("model"):
        # These cohorts contain one row per subject. Fall event detection intentionally retains one score per event.
        scores_by_model[model_name] = binary_metrics(model_predictions.truth.to_numpy(), model_predictions.score.to_numpy())
        per_fold = fold_frame[fold_frame.model == model_name]
        scores_by_model[model_name].update({
            "training_seconds_mean": float(per_fold.training_seconds.mean()),
            "training_seconds_std": float(per_fold.training_seconds.std(ddof=1)),
            "inference_ms_per_sample_mean": float(per_fold.inference_ms_per_sample.mean()),
            "fold_metrics_mean": {metric: _mean(per_fold[metric]) for metric in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")},
            "fold_metrics_std": {metric: _std(per_fold[metric]) for metric in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")},
        })
    best_model = max(scores_by_model, key=lambda name: scores_by_model[name]["roc_auc"] or -1)
    return predictions, fold_frame, {
        "dataset": dataset.profile,
        "task": dataset.task,
        "evaluation": {
            "method": f"{len(folds)}-fold stratified group cross-validation by participant",
            "seed": RANDOM_STATE,
            "group_disjoint": True,
            "metrics_are_exploratory": True,
            "no_hyperparameter_search": True,
        },
        "models": scores_by_model,
        "best_classical_model": best_model,
        "explainability": _fit_explanation(dataset, _classification_estimators(dataset.X)[best_model]),
    }


def evaluate_regression(dataset: ResearchDataset, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    folds = _folds(dataset, n_splits)
    predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (train_indices, test_indices) in enumerate(folds, start=1):
        train_groups = set(dataset.groups.iloc[train_indices])
        test_groups = set(dataset.groups.iloc[test_indices])
        if train_groups.intersection(test_groups):
            raise AssertionError(f"Group leakage in {dataset.module_id} fold {fold_index}")
        for model_name, estimator in _regression_estimators(dataset.X).items():
            started = time.perf_counter()
            estimator.fit(dataset.X.iloc[train_indices], dataset.y.iloc[train_indices])
            training_seconds = time.perf_counter() - started
            inference_started = time.perf_counter()
            values = estimator.predict(dataset.X.iloc[test_indices])
            inference_seconds = time.perf_counter() - inference_started
            y_test = dataset.y.iloc[test_indices].to_numpy(dtype=float)
            fold_rows.append({
                "model": model_name,
                "fold": fold_index,
                "mae": float(mean_absolute_error(y_test, values)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, values))),
                "r2": float(r2_score(y_test, values)) if len(y_test) > 1 else None,
                "training_seconds": float(training_seconds),
                "inference_ms_per_sample": float(inference_seconds * 1000 / len(test_indices)),
                "train_subjects": len(train_groups),
                "test_subjects": len(test_groups),
            })
            for row_index, value in zip(test_indices, values, strict=True):
                predictions.append({
                    "module_id": dataset.module_id,
                    "model": model_name,
                    "fold": fold_index,
                    "subject": str(dataset.groups.iloc[row_index]),
                    "truth": float(dataset.y.iloc[row_index]),
                    "prediction": float(value),
                })
    prediction_frame = pd.DataFrame(predictions)
    fold_frame = pd.DataFrame(fold_rows)
    models: dict[str, Any] = {}
    for model_name, model_predictions in prediction_frame.groupby("model"):
        y_true, y_pred = model_predictions.truth.to_numpy(), model_predictions.prediction.to_numpy()
        per_fold = fold_frame[fold_frame.model == model_name]
        models[model_name] = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "training_seconds_mean": float(per_fold.training_seconds.mean()),
            "inference_ms_per_sample_mean": float(per_fold.inference_ms_per_sample.mean()),
            "fold_metrics_mean": {metric: _mean(per_fold[metric]) for metric in ("mae", "rmse", "r2")},
            "fold_metrics_std": {metric: _std(per_fold[metric]) for metric in ("mae", "rmse", "r2")},
        }
    best_model = min(models, key=lambda name: models[name]["mae"])
    return prediction_frame, fold_frame, {
        "dataset": dataset.profile,
        "task": dataset.task,
        "evaluation": {
            "method": f"{len(folds)}-fold group cross-validation by participant",
            "seed": RANDOM_STATE,
            "group_disjoint": True,
            "metrics_are_exploratory": True,
            "no_hyperparameter_search": True,
        },
        "models": models,
        "best_classical_model": best_model,
        "explainability": _fit_explanation(dataset, _regression_estimators(dataset.X)[best_model], regression=True),
    }


def _mean(series: pd.Series) -> float | None:
    valid = series.dropna()
    return float(valid.mean()) if len(valid) else None


def _std(series: pd.Series) -> float | None:
    valid = series.dropna()
    return float(valid.std(ddof=1)) if len(valid) > 1 else None


def _fit_explanation(dataset: ResearchDataset, estimator: Pipeline, regression: bool = False) -> dict[str, Any]:
    estimator.fit(dataset.X, dataset.y)
    preprocessor = estimator.named_steps["preprocess"]
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    selector = estimator.named_steps.get("select", "passthrough")
    if selector != "passthrough":
        support = selector.get_support()
        feature_names = feature_names[support]
    model = estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        method = "Random Forest impurity feature importance"
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_).reshape(-1)
        method = "absolute linear model coefficient"
    else:
        return {"method": "Model has no direct feature importance", "features": [], "caveat": "No causal inference is supported."}
    if len(values) != len(feature_names):
        return {"method": method, "features": [], "caveat": "Transformed feature names did not match the fitted model coefficients."}
    ranking = np.argsort(np.abs(values))[::-1][:20]
    return {
        "method": method,
        "features": [{"feature": str(feature_names[index]), "importance": float(abs(values[index]))} for index in ranking],
        "caveat": "Associations in this research dataset are not biological causes or person-specific explanations.",
    }


def evaluate_quantum(dataset: ResearchDataset, n_splits: int = 5, qubits: int = 4, layers: int = 2, epochs: int = 5):
    from src.quantum import fit_vqc, quantum_design_matrix, vqc_scores

    if dataset.task != "classification":
        return None, None, {"status": "not_applicable", "reason": "The available task is continuous FTM severity regression; no quantum regression experiment was run."}
    subject_labels = dataset.y.groupby(dataset.groups).agg(lambda values: values.mode().iloc[0])
    effective_splits = min(n_splits, dataset.groups.nunique(), int(subject_labels.value_counts().min()))
    if effective_splits < 2:
        return None, None, {"status": "not_applicable", "reason": "Too few labeled subjects to make at least two stratified group folds."}
    splitter = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    configurations: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(dataset.X, dataset.y, dataset.groups), start=1):
        train_groups = set(dataset.groups.iloc[train_indices])
        test_groups = set(dataset.groups.iloc[test_indices])
        if train_groups.intersection(test_groups):
            raise AssertionError(f"Quantum group leakage in {dataset.module_id} fold {fold}")
        train_x, test_x, configuration = quantum_design_matrix(dataset.X.iloc[train_indices], dataset.X.iloc[test_indices], qubits)
        started = time.perf_counter()
        circuit, weights, losses = fit_vqc(train_x, dataset.y.iloc[train_indices].to_numpy(), qubits=qubits, layers=layers, epochs=epochs, seed=RANDOM_STATE + fold)
        training_seconds = time.perf_counter() - started
        inference_started = time.perf_counter()
        scores = vqc_scores(circuit, weights, test_x)
        inference_seconds = time.perf_counter() - inference_started
        metrics = binary_metrics(dataset.y.iloc[test_indices].to_numpy(dtype=int), scores)
        fold_rows.append({
            "model": f"PennyLane VQC ({qubits} qubits)", "fold": fold, **metrics,
            "training_seconds": float(training_seconds),
            "inference_ms_per_sample": float(inference_seconds * 1000 / len(test_indices)),
            "final_training_loss": losses[-1],
            "train_subjects": len(train_groups), "test_subjects": len(test_groups),
        })
        configurations.append({"fold": fold, **configuration})
        for index, score in zip(test_indices, scores, strict=True):
            prediction_rows.append({
                "module_id": dataset.module_id, "model": f"PennyLane VQC ({qubits} qubits)",
                "fold": fold, "subject": str(dataset.groups.iloc[index]),
                "truth": int(dataset.y.iloc[index]), "score": float(score),
            })
    prediction_frame = pd.DataFrame(prediction_rows)
    fold_frame = pd.DataFrame(fold_rows)
    summary = binary_metrics(prediction_frame.truth.to_numpy(), prediction_frame.score.to_numpy())
    quantum_report = {
        "status": "completed",
        "summary": summary,
        "configuration": {
            "type": "Variational quantum classifier",
            "simulator": "PennyLane default.qubit statevector",
            "encoding": "Y-angle embedding",
            "ansatz": "StronglyEntanglingLayers",
            "qubits": qubits,
            "layers": layers,
            "epochs": epochs,
            "optimizer": "Adam (0.025)",
            "fold_configurations": configurations,
            "score_note": "Rescaled circuit expectation, not a calibrated probability.",
        },
        "training_seconds_mean": float(fold_frame.training_seconds.mean()),
        "inference_ms_per_sample_mean": float(fold_frame.inference_ms_per_sample.mean()),
        "fold_metrics_mean": {metric: _mean(fold_frame[metric]) for metric in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")},
        "fold_metrics_std": {metric: _std(fold_frame[metric]) for metric in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")},
    }
    return prediction_frame, fold_frame, quantum_report


def quantum_utility(classical: pd.DataFrame, quantum: pd.DataFrame, dataset: ResearchDataset, bootstrap_samples: int = 1000) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    baseline = classical[classical.model == "RBF SVM"].copy()
    qframe = quantum.copy()
    if baseline.empty or qframe.empty:
        return {"status": "Insufficient evidence", "reason": "Comparable classical and quantum outputs are unavailable."}
    merge_columns = ["subject", "truth"]
    if dataset.module_id == "fall_event":
        baseline = baseline.reset_index(drop=True)
        qframe = qframe.reset_index(drop=True)
        if len(baseline) != len(qframe) or not np.array_equal(baseline.truth, qframe.truth):
            return {"status": "Insufficient evidence", "reason": "Out-of-fold event records do not align."}
        cluster_values = baseline.subject.astype(str).to_numpy()
        labels = baseline.truth.to_numpy(dtype=int)
        baseline_scores = baseline.score.to_numpy(dtype=float)
        quantum_scores = qframe.score.to_numpy(dtype=float)
    else:
        baseline = baseline.set_index("subject").sort_index()
        qframe = qframe.set_index("subject").sort_index()
        if not baseline.index.equals(qframe.index) or not np.array_equal(baseline.truth, qframe.truth):
            return {"status": "Insufficient evidence", "reason": "The paired subject predictions do not match."}
        cluster_values = baseline.index.astype(str).to_numpy()
        labels = baseline.truth.to_numpy(dtype=int)
        baseline_scores = baseline.score.to_numpy(dtype=float)
        quantum_scores = qframe.score.to_numpy(dtype=float)
    if np.unique(labels).size != 2:
        return {"status": "Insufficient evidence", "reason": "Out-of-fold scores do not include both classes."}
    try:
        delta = float(roc_auc_score(labels, quantum_scores) - roc_auc_score(labels, baseline_scores))
    except ValueError:
        return {"status": "Insufficient evidence", "reason": "ROC-AUC could not be calculated from paired out-of-fold scores."}
    rng = np.random.default_rng(RANDOM_STATE)
    unique_groups = np.unique(cluster_values)
    grouped_indices = {group: np.flatnonzero(cluster_values == group) for group in unique_groups}
    target_by_group = {group: int(labels[index[0]]) for group, index in grouped_indices.items() if len(np.unique(labels[index])) == 1}
    if len(target_by_group) == len(unique_groups):
        strata = {class_value: np.array([group for group, value in target_by_group.items() if value == class_value]) for class_value in (0, 1)}
        strata_groups = [groups for groups in strata.values() if len(groups)]
    else:
        strata_groups = [unique_groups]
    deltas = []
    for _ in range(bootstrap_samples):
        sampled_groups = np.concatenate([rng.choice(groups, size=len(groups), replace=True) for groups in strata_groups])
        sampled_indices = np.concatenate([grouped_indices[group] for group in sampled_groups])
        if np.unique(labels[sampled_indices]).size < 2:
            continue
        deltas.append(float(roc_auc_score(labels[sampled_indices], quantum_scores[sampled_indices]) - roc_auc_score(labels[sampled_indices], baseline_scores[sampled_indices])))
    if not deltas:
        return {"status": "Insufficient evidence", "reason": "The paired subject bootstrap could not form both classes."}
    lower, upper = (float(value) for value in np.quantile(deltas, [0.025, 0.975]))
    margin = 0.05
    if delta >= margin and lower > 0:
        status = "Measurable benefit observed"
    elif upper < margin:
        status = "No measurable benefit observed"
    else:
        status = "Insufficient evidence"
    return {
        "status": status,
        "comparator": "RBF SVM",
        "metric": "paired out-of-fold ROC-AUC difference (quantum minus SVM)",
        "observed_delta": delta,
        "bootstrap_95_percent_interval": [lower, upper],
        "practical_margin": margin,
        "bootstrap_samples": len(deltas),
        "independent_subjects": int(len(unique_groups)),
        "reason": "Exploratory cross-validation comparison on a small public cohort; the bootstrap interval does not include all model-training or external-validation uncertainty.",
    }


def train_module(module_id: str, n_splits: int = 5, run_quantum: bool = True, qubits: int = 4, layers: int = 2, epochs: int = 5) -> dict[str, Any]:
    if module_id not in {"alzheimer", "depression", "stroke_history", "fall_event", "essential_tremor", "als"}:
        raise ValueError(f"Unknown research module: {module_id}")
    if module_id == "als":
        dataset = load_als_dataset()
        profiles = dataset.profile
        artifacts_dir = ARTIFACT_ROOT / module_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "module_id": module_id,
            "status": "blocked_by_dataset_confounding",
            "dataset": profiles,
            "reason": "ALS status is perfectly confounded with GEO study/batch (45 ALS all Study_1; 15 controls all Study_2/4/5). A classifier would measure batch effects, so no ALS disease model is trained.",
            "models": {},
            "quantum": {"status": "not_applicable", "reason": "No disease model is fit because of complete batch confounding."},
        }
        (artifacts_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        return report

    dataset = {
        "alzheimer": load_alzheimer_dataset,
        "depression": load_depression_dataset,
        "stroke_history": load_stroke_history_dataset,
        "fall_event": load_fall_event_dataset,
        "essential_tremor": load_essential_tremor_dataset,
    }[module_id]()
    if dataset.task == "classification":
        predictions, fold_frame, report = evaluate_classification(dataset, n_splits=n_splits)
    else:
        predictions, fold_frame, report = evaluate_regression(dataset, n_splits=min(n_splits, dataset.groups.nunique()))
    report["module_id"] = module_id
    report["status"] = "completed"
    report["quantum"] = {"status": "not_run"}
    artifacts_dir = ARTIFACT_ROOT / module_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(artifacts_dir / "out_of_fold_predictions.csv", index=False)
    fold_frame.to_csv(artifacts_dir / "fold_metrics.csv", index=False)

    if run_quantum and dataset.task == "classification":
        try:
            q_predictions, q_folds, q_report = evaluate_quantum(dataset, n_splits=n_splits, qubits=qubits, layers=layers, epochs=epochs)
            q_predictions.to_csv(artifacts_dir / "quantum_predictions.csv", index=False)
            q_folds.to_csv(artifacts_dir / "quantum_fold_metrics.csv", index=False)
            report["quantum"] = q_report
            report["quantum"]["utility"] = quantum_utility(predictions, q_predictions, dataset)
        except Exception as error:
            report["quantum"] = {"status": "failed", "error_type": type(error).__name__, "message": str(error)}
    elif dataset.task == "regression":
        report["quantum"] = {"status": "not_applicable", "reason": "Tiny within-ET regression cohort; a binary quantum classifier would not match the dataset task."}

    if dataset.task == "classification":
        estimators = _classification_estimators(dataset.X)
    else:
        estimators = _regression_estimators(dataset.X)
    fitted = {name: estimator.fit(dataset.X, dataset.y) for name, estimator in estimators.items()}
    joblib.dump(fitted, artifacts_dir / "classical_models.joblib")
    if report.get("quantum", {}).get("status") == "completed":
        report["quantum_artifact"] = _fit_quantum_artifact(dataset, qubits, layers, epochs, artifacts_dir)
    report["artifact_path"] = artifacts_dir.as_posix()
    report["model_version"] = "research-v1"
    (artifacts_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def _fit_quantum_artifact(dataset: ResearchDataset, qubits: int, layers: int, epochs: int, artifacts_dir: Path) -> str:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from src.quantum import fit_vqc

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=min(qubits, dataset.X.shape[1], len(dataset.X) - 1), random_state=RANDOM_STATE)
    encoded = pca.fit_transform(scaler.fit_transform(imputer.fit_transform(dataset.X)))
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi), clip=True)
    encoded = angle_scaler.fit_transform(encoded)
    circuit, weights, losses = fit_vqc(encoded, dataset.y.to_numpy(dtype=int), qubits=qubits, layers=layers, epochs=epochs, seed=RANDOM_STATE)
    del circuit
    artifact = {
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "angle_scaler": angle_scaler,
        "weights": np.asarray(weights),
        "feature_columns": list(dataset.X.columns),
        "qubits": qubits,
        "layers": layers,
        "epochs": epochs,
        "training_loss": losses[-1],
        "simulator": "PennyLane default.qubit",
        "warning": "Refit on all dataset participants after grouped CV; not an external validation result.",
    }
    path = artifacts_dir / "quantum_model.joblib"
    joblib.dump(artifact, path)
    return path.name