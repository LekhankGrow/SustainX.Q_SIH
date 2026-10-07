from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/174/parkinsons.zip"
DATA_PATH = Path("data/parkinsons.data")
SUBJECT_PATTERN = re.compile(r"_S(\d+)_")
TARGET_COLUMN = "status"
ID_COLUMN = "name"


@dataclass(frozen=True)
class DatasetProfile:
    source_url: str
    citation: str
    license: str
    rows: int
    subjects: int
    features: int
    labels: dict[int, int]
    subject_labels: dict[int, int]
    missing_cells: int
    sha256: str


def download_dataset(destination: Path = DATA_PATH) -> Path:
    """Download the public UCI archive and extract only its feature table."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next((name for name in archive.namelist() if name.endswith("parkinsons.data")), None)
        if member is None:
            raise ValueError("UCI archive did not contain parkinsons.data")
        destination.write_bytes(archive.read(member))
    return destination


def load_dataset(path: Path = DATA_PATH, download: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.Series, DatasetProfile]:
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Dataset not found: {path}")
        download_dataset(path)

    raw = pd.read_csv(path)
    required = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if raw.empty:
        raise ValueError("Dataset is empty")

    subject_matches = raw[ID_COLUMN].astype(str).str.extract(SUBJECT_PATTERN, expand=False)
    if subject_matches.isna().any():
        raise ValueError("Could not parse a subject identifier from every recording name")
    groups = subject_matches.map(lambda value: f"S{value}")

    labels = pd.to_numeric(raw[TARGET_COLUMN], errors="raise").astype(int)
    if not set(labels.unique()).issubset({0, 1}) or labels.nunique() != 2:
        raise ValueError("Expected the binary UCI status labels 0 (healthy) and 1 (Parkinson's)")
    if raw.assign(_subject=groups, _target=labels).groupby("_subject")["_target"].nunique().max() != 1:
        raise ValueError("A subject has inconsistent health labels across recordings")
    features = raw.drop(columns=[ID_COLUMN, TARGET_COLUMN]).apply(pd.to_numeric, errors="raise")
    if features.shape[1] != 22:
        raise ValueError(f"Expected 22 engineered voice features; found {features.shape[1]}")

    subject_status = pd.DataFrame({"subject": groups, "status": labels}).groupby("subject")["status"].first()
    profile = DatasetProfile(
        source_url=DATA_URL,
        citation="Little, M. (2007). Parkinsons. UCI Machine Learning Repository. https://doi.org/10.24432/C59C74",
        license="CC BY 4.0",
        rows=len(raw),
        subjects=groups.nunique(),
        features=features.shape[1],
        labels={int(key): int(value) for key, value in labels.value_counts().sort_index().items()},
        subject_labels={int(key): int(value) for key, value in subject_status.value_counts().sort_index().items()},
        missing_cells=int(raw.isna().sum().sum()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return features, labels, groups, profile