from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.baseline import BASELINE_FEATURES, compare_to_baseline, make_entry
from src.benchmark import assess_quantum_utility, grouped_folds
from src.data import DATA_PATH, load_dataset
from src.inference import load_model_bundle, predict_compatible_features
from src.metrics import binary_metrics
from src.quality import assess_audio


def make_wav(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_grouped_folds_never_split_a_subject() -> None:
    labels = np.repeat([0, 1], 10)
    subjects = np.array([f"subject-{index}" for index in range(20)])
    row_labels = np.repeat(labels, 2)
    row_subjects = np.repeat(subjects, 2)
    folds = grouped_folds(row_labels, row_subjects, n_splits=5)
    assert len(folds) == 5
    for train_indices, test_indices in folds:
        assert set(row_subjects[train_indices]).isdisjoint(set(row_subjects[test_indices]))


def test_real_dataset_schema_and_subject_labels_when_available() -> None:
    if not DATA_PATH.exists():
        pytest.skip("Run the benchmark command to download the public UCI dataset first")
    X, y, groups, profile = load_dataset(download=False)
    assert X.shape == (profile.rows, 22)
    assert len(y) == len(groups) == profile.rows
    assert "name" not in X.columns and "status" not in X.columns
    assert groups.nunique() == profile.subjects
    assert set(y.unique()) == {0, 1}
    assert profile.missing_cells == 0


def test_full_model_artifact_loads_and_accepts_exact_feature_schema() -> None:
    if not (DATA_PATH.parent.parent / "artifacts/model_manifest.json").exists():
        pytest.skip("Run the benchmark to generate model artifacts")
    bundle, manifest = load_model_bundle()
    X, _, _, _ = load_dataset(download=False)
    result = predict_compatible_features(X.iloc[[0]], bundle, manifest)
    assert len(result["model_scores"]) == 4
    assert all(np.isfinite(score) for score in result["model_scores"].values())
    assert "not calibrated probabilities" in result["score_note"]


def test_research_inference_rejects_incompatible_feature_schema() -> None:
    expected = ["feature_a", "feature_b"]
    bundle = {"classical": {}, "quantum": {}}
    manifest = {"feature_columns": expected}
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        predict_compatible_features(pd.DataFrame({"feature_a": [1.0], "feature_c": [2.0]}), bundle, manifest)


def test_binary_metrics_include_specificity_and_auc() -> None:
    result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.9]))
    assert result["accuracy"] == 0.75
    assert result["specificity"] == 0.5
    assert result["sensitivity"] == 1.0
    assert result["roc_auc"] == 0.75


def test_quality_gate_accepts_clean_supported_audio() -> None:
    time = np.arange(16000 * 4) / 16000
    samples = (0.12 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    result = assess_audio(make_wav(samples))
    assert result.passed
    assert result.duration_seconds == pytest.approx(4.0, abs=0.01)
    assert "spectral_centroid_hz" in result.features


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(16000 * 4, dtype=np.float32),
        np.concatenate([np.ones(16000, dtype=np.float32), np.zeros(16000 * 3, dtype=np.float32)]),
        np.ones(16000 * 4, dtype=np.float32) * 0.6,
    ],
)
def test_quality_gate_rejects_unusable_or_clipped_audio(samples: np.ndarray) -> None:
    assert not assess_audio(make_wav(samples)).passed


def test_quality_gate_rejects_too_short_audio() -> None:
    short = make_wav(np.ones(16000, dtype=np.float32) * 0.1)
    assessment = assess_audio(short)
    assert not assessment.passed
    assert any("3 and 15 seconds" in reason for reason in assessment.reasons)


def test_personal_baseline_create_compare_and_validate() -> None:
    first = {feature: float(index + 1) for index, feature in enumerate(BASELINE_FEATURES)}
    second = {feature: value * 1.1 for feature, value in first.items()}
    entries = [make_entry(first, "2026-01-01T00:00:00+00:00")]
    change = compare_to_baseline(second, entries)
    assert change["rms_mean"]["relative_change_percent"] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        compare_to_baseline(second, [])


def test_quantum_utility_uses_actual_paired_subject_predictions() -> None:
    subjects = [f"s{index}" for index in range(12)]
    truth = [0] * 6 + [1] * 6
    classical = pd.DataFrame({
        "subject": subjects,
        "model": "RBF SVM",
        "truth": truth,
        "score": [0.45, 0.55, 0.4, 0.6, 0.3, 0.7, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7],
    })
    quantum = pd.DataFrame({
        "subject": subjects,
        "model": "VQC",
        "truth": truth,
        "score": [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
    })
    result = assess_quantum_utility(classical, quantum, bootstrap_samples=100)
    assert result["status"] == "Measurable benefit observed"
    assert result["subject_count"] == 12


def test_quantum_utility_refuses_mismatched_subjects() -> None:
    classical = pd.DataFrame({"subject": ["a", "b"], "model": "RBF SVM", "truth": [0, 1], "score": [0.2, 0.8]})
    quantum = pd.DataFrame({"subject": ["a", "c"], "model": "VQC", "truth": [0, 1], "score": [0.2, 0.8]})
    result = assess_quantum_utility(classical, quantum, bootstrap_samples=20)
    assert result["status"] == "Insufficient evidence"


def test_quantum_circuit_and_training_smoke() -> None:
    from src.quantum import fit_vqc, vqc_scores

    features = np.array([
        [-0.2, 0.1, 0.3, -0.1],
        [0.2, -0.1, -0.3, 0.1],
    ])
    circuit, weights, history = fit_vqc(features, np.array([0, 1]), qubits=4, layers=1, epochs=1)
    scores = vqc_scores(circuit, weights, np.zeros((1, 4)))
    assert np.isfinite(history).all()
    assert scores.shape == (1,)
    assert np.isfinite(scores).all()