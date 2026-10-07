from __future__ import annotations

import argparse
import json
import traceback

from src.disease_registry import MODULE_METADATA, RESEARCH_ARTIFACTS, save_research_registry
from src.research_benchmark import train_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reproducible non-Parkinson SustainX.Q research modules.")
    parser.add_argument("--modules", nargs="+", choices=[*MODULE_METADATA], default=list(MODULE_METADATA))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--qubits", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--skip-quantum", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    summaries = []
    for module_id in arguments.modules:
        try:
            report = train_module(
                module_id,
                n_splits=arguments.folds,
                run_quantum=not arguments.skip_quantum,
                qubits=arguments.qubits,
                layers=arguments.layers,
                epochs=arguments.epochs,
            )
        except Exception as error:
            report = {
                "module_id": module_id,
                "status": "training_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            module_dir = RESEARCH_ARTIFACTS / module_id
            module_dir.mkdir(parents=True, exist_ok=True)
            (module_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summaries.append({
            "module_id": module_id,
            "status": report.get("status"),
            "samples": report.get("dataset", {}).get("samples"),
            "subjects": report.get("dataset", {}).get("subjects"),
            "best_classical_model": report.get("best_classical_model"),
            "classical_models": report.get("models", {}),
            "quantum": report.get("quantum", {}),
        })
        print(json.dumps(summaries[-1], indent=2, allow_nan=False))
    registry = save_research_registry()
    print(json.dumps({"registry": str(RESEARCH_ARTIFACTS / "registry.json"), "module_statuses": {module["id"]: module["status"] for module in registry["modules"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())