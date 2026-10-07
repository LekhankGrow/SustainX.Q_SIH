from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.benchmark import positive_scores
from src.research_benchmark import ARTIFACT_ROOT


def load_research_bundle(module_id: str) -> tuple[dict, dict]:
    directory = ARTIFACT_ROOT / module_id
    report_path = directory / "report.json"
    model_path = directory / "classical_models.joblib"
    if not report_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"No completed model artifact is registered for {module_id}")
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError(f"Module {module_id} has status {report.get('status')}; inference is unavailable")
    return joblib.load(model_path), report


def predict_research_input(module_id: str, features: pd.DataFrame) -> dict:
    models, report = load_research_bundle(module_id)
    expected = list(report["dataset"]["feature_schema"])
    if list(features.columns) != expected:
        missing = sorted(set(expected).difference(features.columns))
        extra = sorted(set(features.columns).difference(expected))
        raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}, or columns are not in training order")
    if len(features) != 1:
        raise ValueError("Upload exactly one feature row for a research input check")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Feature row contains non-finite values")

    if report["task"] == "regression":
        scores = {name: float(model.predict(numeric)[0]) for name, model in models.items()}
        return {
            "task": "regression",
            "outputs": scores,
            "target_description": report["dataset"]["target_definition"],
            "note": "Exploratory within-cohort regression outputs; not a diagnosis.",
        }

    scores = {name: float(positive_scores(model, numeric)[0]) for name, model in models.items()}
    signals = {name: value >= 0.5 for name, value in scores.items()}
    return {
        "task": "classification",
        "outputs": scores,
        "threshold_signals": signals,
        "agreement_fraction": max(sum(signals.values()), len(signals) - sum(signals.values())) / len(signals),
        "note": "Uncalibrated research scores; thresholds and scores are not validated clinical probabilities.",
        "target_description": report["dataset"]["task"],
    }