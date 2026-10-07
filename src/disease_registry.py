from __future__ import annotations

import json
from pathlib import Path

from src.experiment import load_report as load_parkinson_report

RESEARCH_ARTIFACTS = Path("artifacts/research")

MODULE_METADATA = {
    "alzheimer": {
        "name": "Alzheimer's Disease",
        "input_modality": "Online handwriting-derived numeric features",
        "purpose": "Research classification of the UCI DARWIN Alzheimer's group versus healthy controls. MCI is not represented in this dataset.",
        "limitation": "174 participants from one dataset; exploratory, no external validation, and not a clinical diagnosis.",
    },
    "depression": {
        "name": "Depression Symptom Screening",
        "input_modality": "Demographic survey features; target derived from PHQ-9 questionnaire responses",
        "purpose": "Explore demographic correlates of an elevated self-reported PHQ-9 score (>=10). PHQ-9 items are never model predictors.",
        "limitation": "This predicts a questionnaire threshold in the NHANES sample, not a depression diagnosis; survey weights are not applied.",
    },
    "stroke_history": {
        "name": "Stroke History",
        "input_modality": "Demographic survey features; target is self-reported prior stroke history",
        "purpose": "Explore retrospective classification of reported stroke history in NHANES.",
        "limitation": "Does not detect acute stroke or predict recovery/future stroke; imbalanced self-reported US survey cohort.",
    },
    "fall_event": {
        "name": "Fall Event Detection",
        "input_modality": "Wearable accelerometer and gyroscope event recordings",
        "purpose": "Distinguish scripted fall events from activities of daily living in SisFall recordings.",
        "limitation": "Controlled scripted events and healthy volunteers; event detection is not prospective fall-risk prediction.",
    },
    "essential_tremor": {
        "name": "Essential Tremor Severity Research",
        "input_modality": "Hand accelerometry under rest and posture conditions",
        "purpose": "Exploratory within-ET regression of the FTM total severity score.",
        "limitation": "Only 26 of 29 participants have usable FTM labels in the downloaded cohort; no controls, so this is not ET diagnosis.",
    },
    "als": {
        "name": "ALS Molecular Signal Research",
        "input_modality": "Skin-fibroblast gene-expression features from GEO GSE233881",
        "purpose": "Exploratory ALS-versus-control transcriptomic research task using the small validated public fibroblast cohort.",
        "limitation": "Only 12 participants (9 sporadic ALS, 3 controls) are available in the validated public matrix. This is very small, highly imbalanced, and not clinically validated.",
    },
}


def _module_report(module_id: str) -> dict | None:
    path = RESEARCH_ARTIFACTS / module_id / "report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_research_registry() -> dict:
    modules: list[dict] = []
    for module_id, metadata in MODULE_METADATA.items():
        report = _module_report(module_id)
        status = report.get("status", "not_trained") if report else "not_trained"
        if status == "completed":
            status = "active_research"
        modules.append({
            "id": module_id,
            **metadata,
            "status": status,
            "report": report,
        })
    parkinson_report = load_parkinson_report()
    modules.insert(0, {
        "id": "parkinson",
        "name": "Parkinson's Disease",
        "input_modality": "Voice/acoustic engineered features; microphone flow is quality/baseline only",
        "purpose": "Flagship subject-disjoint classical and PennyLane research benchmark.",
        "limitation": "32 unique subjects in the validated file; exploratory, not clinically validated; live microphone inference is not feature-compatible.",
        "status": "active_research" if parkinson_report else "not_trained",
        "report": parkinson_report,
    })
    modules.append({
        "id": "skin_future",
        "name": "Skin Lesion Screening",
        "input_modality": "Future image-based research module",
        "purpose": "Planned future image-feature and questionnaire research direction.",
        "limitation": "Not currently implemented. No dataset, model, metric, or prediction.",
        "status": "future_update",
        "report": None,
    })
    return {
        "product": "SustainX.Q",
        "updated_at": "generated from current local model reports",
        "modules": modules,
    }


def save_research_registry() -> dict:
    registry = get_research_registry()
    RESEARCH_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (RESEARCH_ARTIFACTS / "registry.json").write_text(json.dumps(registry, indent=2, allow_nan=False), encoding="utf-8")
    return registry


def get_module(module_id: str) -> dict | None:
    return next((item for item in get_research_registry()["modules"] if item["id"] == module_id), None)