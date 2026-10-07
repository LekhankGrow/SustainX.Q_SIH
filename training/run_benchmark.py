from __future__ import annotations

import argparse
import json

from src.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-disjoint Parkinson voice model benchmarks.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--qubits", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    report = run_experiment(
        folds=arguments.folds,
        qubits=arguments.qubits,
        layers=arguments.layers,
        epochs=arguments.epochs,
    )
    print(json.dumps({
        "dataset": report.get("dataset"),
        "classical_models": report.get("classical_models"),
        "quantum_status": report.get("quantum_status"),
        "quantum_utility": report.get("quantum_utility"),
        "model_artifact_status": report.get("model_artifact_status"),
    }, indent=2))
    return 0 if report.get("classical_models") else 1


if __name__ == "__main__":
    raise SystemExit(main())