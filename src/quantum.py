from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pennylane as qml
import pennylane.numpy as pnp
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.benchmark import RANDOM_STATE
from src.metrics import binary_metrics


def build_quantum_circuit(qubits: int = 4, layers: int = 2):
    device = qml.device("default.qubit", wires=qubits)

    @qml.qnode(device, interface="autograd", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def quantum_design_matrix(train: pd.DataFrame, test: pd.DataFrame, qubits: int):
    component_count = min(qubits, train.shape[1], len(train) - 1)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=component_count, random_state=RANDOM_STATE)
    train_values = pca.fit_transform(scaler.fit_transform(imputer.fit_transform(train)))
    test_values = pca.transform(scaler.transform(imputer.transform(test)))
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi), clip=True)
    train_encoded = angle_scaler.fit_transform(train_values)
    test_encoded = angle_scaler.transform(test_values)
    return train_encoded, test_encoded, {
        "original_feature_count": int(train.shape[1]),
        "selected_feature_count": int(component_count),
        "quantum_input_count": int(component_count),
        "qubit_count": int(qubits),
        "feature_representation": "training-fold PCA components",
        "angle_scaling": "training-fold-fitted min-max to [-pi, pi]",
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def fit_vqc(train_x: np.ndarray, train_y: np.ndarray, qubits: int = 4, layers: int = 2, epochs: int = 30, seed: int = RANDOM_STATE):
    circuit = build_quantum_circuit(qubits, layers)
    rng = np.random.default_rng(seed)
    weights = pnp.array(rng.normal(0, 0.08, size=(layers, qubits, 3)), requires_grad=True)
    targets = pnp.array(np.where(train_y > 0, 1.0, -1.0))
    inputs = pnp.array(train_x)
    optimizer = qml.AdamOptimizer(stepsize=0.025)

    def loss(current_weights):
        values = circuit(inputs, current_weights)
        return pnp.mean((values - targets) ** 2)

    history: list[float] = []
    for _ in range(epochs):
        weights, current_loss = optimizer.step_and_cost(loss, weights)
        history.append(float(current_loss))
    return circuit, weights, history


def vqc_scores(circuit, weights, data: np.ndarray) -> np.ndarray:
    expectations = np.asarray(circuit(pnp.array(data), weights), dtype=float).reshape(-1)
    return (expectations + 1.0) / 2.0


def evaluate_quantum(n_splits: int = 5, qubits: int = 4, layers: int = 2, epochs: int = 30):
    from src.data import load_dataset

    X, y, groups, profile = load_dataset()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_configurations: list[dict[str, Any]] = []
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(X, y, groups), start=1):
        if set(groups.iloc[train_indices]).intersection(groups.iloc[test_indices]):
            raise AssertionError("Subject leakage detected in quantum fold")
        train_x, test_x, configuration = quantum_design_matrix(X.iloc[train_indices], X.iloc[test_indices], qubits)
        started = time.perf_counter()
        circuit, weights, loss_history = fit_vqc(
            train_x,
            y.iloc[train_indices].to_numpy(),
            qubits=qubits,
            layers=layers,
            epochs=epochs,
            seed=RANDOM_STATE + fold_index,
        )
        training_seconds = time.perf_counter() - started
        inference_started = time.perf_counter()
        scores = vqc_scores(circuit, weights, test_x)
        inference_seconds = time.perf_counter() - inference_started
        fold_predictions = pd.DataFrame({
            "subject": groups.iloc[test_indices].to_numpy(),
            "truth": y.iloc[test_indices].to_numpy(),
            "score": scores,
        }).groupby("subject", as_index=False).agg(truth=("truth", "first"), score=("score", "mean"))
        metrics = binary_metrics(fold_predictions["truth"].to_numpy(), fold_predictions["score"].to_numpy())
        fold_rows.append({
            "model": f"PennyLane VQC ({qubits} qubits)",
            "fold": fold_index,
            **metrics,
            "training_seconds": training_seconds,
            "inference_ms_per_recording": inference_seconds * 1000 / len(test_indices),
            "train_subjects": int(groups.iloc[train_indices].nunique()),
            "test_subjects": int(groups.iloc[test_indices].nunique()),
            "final_training_loss": loss_history[-1],
        })
        fold_configurations.append({"fold": fold_index, **configuration})
        for index, score in zip(test_indices, scores, strict=True):
            prediction_rows.append({
                "model": f"PennyLane VQC ({qubits} qubits)",
                "fold": fold_index,
                "subject": groups.iloc[index],
                "truth": int(y.iloc[index]),
                "score": float(score),
            })

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    subject_predictions = predictions.groupby("subject", as_index=False).agg(truth=("truth", "first"), score=("score", "mean"))
    summary_metrics = binary_metrics(subject_predictions["truth"].to_numpy(), subject_predictions["score"].to_numpy())
    metadata = {
        "dataset": profile.__dict__,
        "configuration": {
            "model": "Variational quantum classifier (VQC)",
            "simulator": "PennyLane default.qubit (statevector)",
            "encoding": "AngleEmbedding with Y rotations",
            "ansatz": "StronglyEntanglingLayers",
            "layers": layers,
            "qubits": qubits,
            "epochs": epochs,
            "optimizer": "Adam, step size 0.025",
            "fold_configurations": fold_configurations,
            "score_note": "Rescaled expectation value; not a calibrated probability.",
        },
        "summary": summary_metrics,
        "training_seconds_mean": float(folds["training_seconds"].mean()),
        "inference_ms_per_recording_mean": float(folds["inference_ms_per_recording"].mean()),
        "fold_metrics_mean": {key: float(folds[key].mean()) if folds[key].notna().any() else None for key in summary_metrics},
        "fold_metrics_std": {key: float(folds[key].std(ddof=1)) if folds[key].count() > 1 else None for key in summary_metrics},
    }
    return predictions, folds, metadata