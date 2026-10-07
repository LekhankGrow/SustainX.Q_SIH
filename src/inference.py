from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.benchmark import ARTIFACT_DIR, positive_scores
from src.quantum import build_quantum_circuit, vqc_scores


def load_model_bundle(artifact_dir: Path = ARTIFACT_DIR) -> tuple[dict, dict]:
    manifest_path = artifact_dir / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Model manifest is not available; run the benchmark first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classical_path = artifact_dir / manifest["classical_model_file"]
    quantum_path = artifact_dir / manifest["quantum_model_file"]
    if not classical_path.exists() or not quantum_path.exists():
        raise FileNotFoundError("One or more research model artifacts are missing")
    classical_models = joblib.load(classical_path)
    quantum_model = joblib.load(quantum_path)
    return {"classical": classical_models, "quantum": quantum_model}, manifest


def predict_compatible_features(features: pd.DataFrame, bundle: dict, manifest: dict) -> dict:
    expected = list(manifest["feature_columns"])
    received = list(features.columns)
    missing = sorted(set(expected).difference(received))
    extra = sorted(set(received).difference(expected))
    if missing or extra:
        raise ValueError(f"Feature schema mismatch. Missing: {missing}; unexpected: {extra}.")
    if len(features) != 1:
        raise ValueError("Upload exactly one feature row for this research-only inference check.")
    ordered = features.loc[:, expected].apply(pd.to_numeric, errors="coerce")
    values = ordered.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature values must all be finite numeric values.")

    scores: dict[str, float] = {}
    for name, model in bundle["classical"].items():
        scores[name] = float(positive_scores(model, ordered)[0])

    quantum = bundle["quantum"]
    quantum_input = quantum["imputer"].transform(ordered)
    quantum_input = quantum["scaler"].transform(quantum_input)
    quantum_input = quantum["pca"].transform(quantum_input)
    quantum_input = quantum["angle_scaler"].transform(quantum_input)
    circuit = build_quantum_circuit(quantum["qubits"], quantum["layers"])
    scores[f"PennyLane VQC ({quantum['qubits']} qubits)"] = float(
        vqc_scores(circuit, quantum["weights"], quantum_input)[0]
    )
    votes = {name: score >= 0.5 for name, score in scores.items()}
    agreement = max(sum(votes.values()), len(votes) - sum(votes.values())) / len(votes)
    return {
        "model_scores": scores,
        "threshold_signals": votes,
        "model_agreement_fraction": float(agreement),
        "score_note": "Uncalibrated research model scores, not calibrated probabilities. The 0.5 threshold is not medically validated.",
        "validation_note": "Artifacts were refit on all available UCI records after cross-validation. This single input has no independent validation guarantee.",
        "feature_columns": expected,
    }