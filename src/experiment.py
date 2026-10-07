from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from src.benchmark import ARTIFACT_DIR, assess_quantum_utility, evaluate_classical


def run_experiment(folds: int = 5, qubits: int = 4, layers: int = 2, epochs: int = 20) -> dict:
    predictions, fold_results, report = evaluate_classical(n_splits=folds)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(ARTIFACT_DIR / "classical_predictions.csv", index=False)
    fold_results.to_csv(ARTIFACT_DIR / "classical_folds.csv", index=False)
    report["quantum_status"] = "running"
    _write_report(report)

    try:
        from src.quantum import evaluate_quantum

        quantum_predictions, quantum_folds, quantum_report = evaluate_quantum(
            n_splits=folds,
            qubits=qubits,
            layers=layers,
            epochs=epochs,
        )
        quantum_predictions.to_csv(ARTIFACT_DIR / "quantum_predictions.csv", index=False)
        quantum_folds.to_csv(ARTIFACT_DIR / "quantum_folds.csv", index=False)
        pd.concat([predictions, quantum_predictions], ignore_index=True).to_csv(
            ARTIFACT_DIR / "all_predictions.csv", index=False
        )
        report["quantum"] = quantum_report
        report["quantum_utility"] = assess_quantum_utility(predictions, quantum_predictions)
        report["quantum_status"] = "completed"
    except Exception as error:
        report["quantum_status"] = "unavailable"
        report["quantum_failure"] = {"type": type(error).__name__, "message": str(error)}
        report["quantum_traceback"] = traceback.format_exc()
    try:
        from src.artifacts import save_full_data_artifacts

        report["model_artifacts"] = save_full_data_artifacts(qubits=qubits, layers=layers, epochs=epochs)
        report["model_artifact_status"] = "saved"
    except Exception as error:
        report["model_artifact_status"] = "unavailable"
        report["model_artifact_failure"] = {"type": type(error).__name__, "message": str(error)}
    _write_report(report)
    return report


def load_report() -> dict | None:
    path = ARTIFACT_DIR / "benchmark.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_predictions() -> pd.DataFrame | None:
    path = ARTIFACT_DIR / "all_predictions.csv"
    if not path.exists():
        path = ARTIFACT_DIR / "classical_predictions.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None


def _write_report(report: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "benchmark.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )