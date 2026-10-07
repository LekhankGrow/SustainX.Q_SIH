from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.benchmark import ARTIFACT_DIR, classical_estimators
from src.data import load_dataset
from src.quantum import fit_vqc


def save_full_data_artifacts(qubits: int = 4, layers: int = 2, epochs: int = 15) -> dict[str, Any]:
    X, y, _, profile = load_dataset(download=False)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    classical_models = classical_estimators()
    for estimator in classical_models.values():
        estimator.fit(X, y)
    joblib.dump(classical_models, ARTIFACT_DIR / "classical_models.joblib")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    component_count = min(qubits, X.shape[1], len(X) - 1)
    pca = PCA(n_components=component_count, random_state=26139)
    quantum_features = pca.fit_transform(scaler.fit_transform(imputer.fit_transform(X)))
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi), clip=True)
    quantum_features = angle_scaler.fit_transform(quantum_features)
    circuit, weights, history = fit_vqc(
        quantum_features,
        y.to_numpy(),
        qubits=qubits,
        layers=layers,
        epochs=epochs,
        seed=26139,
    )
    del circuit
    quantum_bundle = {
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "angle_scaler": angle_scaler,
        "weights": np.asarray(weights),
        "feature_columns": list(X.columns),
        "qubits": qubits,
        "layers": layers,
        "epochs": epochs,
        "final_training_loss": history[-1],
        "simulator": "PennyLane default.qubit",
        "training_data_sha256": profile.sha256,
        "note": "Refit on all available subjects for research reproducibility; not an independent validation artifact and not connected to microphone inference.",
    }
    joblib.dump(quantum_bundle, ARTIFACT_DIR / "quantum_model.joblib")
    manifest = {
        "dataset_sha256": profile.sha256,
        "dataset_rows": profile.rows,
        "training_subjects": profile.subjects,
        "feature_columns": list(X.columns),
        "classical_model_file": "classical_models.joblib",
        "quantum_model_file": "quantum_model.joblib",
        "quantum_refit": {"qubits": qubits, "layers": layers, "epochs": epochs},
        "purpose": "Reproducible research artifacts. Evaluation results come only from subject-disjoint out-of-fold predictions.",
        "microphone_inference": "Not supported: microphone features have not been shown compatible with the UCI model schema.",
    }
    (ARTIFACT_DIR / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest