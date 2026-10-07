from __future__ import annotations

from datetime import datetime, timezone

BASELINE_FEATURES = (
    "rms_mean",
    "rms_std",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_flatness",
    "speech_activity_proxy",
)


def make_entry(features: dict[str, float], captured_at: str | None = None) -> dict:
    missing = set(BASELINE_FEATURES).difference(features)
    if missing:
        raise ValueError(f"Missing baseline features: {sorted(missing)}")
    return {
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": {name: float(features[name]) for name in BASELINE_FEATURES},
    }


def compare_to_baseline(current: dict[str, float], entries: list[dict]) -> dict[str, dict[str, float]]:
    if not entries:
        raise ValueError("Create a baseline before comparing measurements")
    baseline = {
        name: sum(entry["features"][name] for entry in entries) / len(entries)
        for name in BASELINE_FEATURES
    }
    return {
        name: {
            "baseline": baseline[name],
            "current": float(current[name]),
            "absolute_change": float(current[name]) - baseline[name],
            "relative_change_percent": 100.0 * (float(current[name]) - baseline[name]) / max(abs(baseline[name]), 1e-8),
        }
        for name in BASELINE_FEATURES
    }