from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import zipfile
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch

RESEARCH_DATA = Path("data/research")
RAW_DATA = RESEARCH_DATA / "raw"
PROCESSED_DATA = RESEARCH_DATA / "processed"

SOURCES = {
    "alzheimer": {
        "dataset": "UCI DARWIN",
        "url": "https://archive.ics.uci.edu/dataset/732/darwin",
        "download": "https://archive.ics.uci.edu/static/public/732/darwin.zip",
        "citation": "Fontanella, F. (2022). DARWIN. UCI Machine Learning Repository. https://doi.org/10.24432/C55D0K",
        "license": "CC BY 4.0",
        "modality": "Online handwriting-derived features",
        "task": "Alzheimer's disease vs healthy controls; MCI is not included.",
    },
    "depression": {
        "dataset": "NHANES 2015-2016 Depression Screener (DPQ) + Demographics (DEMO)",
        "url": "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015",
        "download": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DPQ_I.XPT",
        "citation": "National Center for Health Statistics. NHANES 2015-2016. https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015",
        "license": "Public-use federal survey data; cite CDC/NCHS. Review current NCHS terms before redistribution.",
        "modality": "Questionnaire-derived demographic features; PHQ-9 responses define the research target and are excluded from predictors.",
        "task": "Demographic correlates of an elevated self-reported PHQ-9 score (>=10), not depression diagnosis.",
    },
    "stroke_history": {
        "dataset": "NHANES 2015-2016 Medical Conditions (MCQ) + Demographics (DEMO)",
        "url": "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015",
        "download": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/MCQ_I.XPT",
        "citation": "National Center for Health Statistics. NHANES 2015-2016. https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015",
        "license": "Public-use federal survey data; cite CDC/NCHS. Review current NCHS terms before redistribution.",
        "modality": "Demographic questionnaire features; MCQ160F self-report defines target and is excluded from predictors.",
        "task": "Classify reported history of being told by a health professional that the participant had a stroke; not acute stroke detection or recovery prediction.",
    },
    "fall_event": {
        "dataset": "SisFall: A Fall and Movement Dataset",
        "url": "https://doi.org/10.5281/zenodo.22212281",
        "download": "https://zenodo.org/api/records/22212281/files/SisFall.zip/content",
        "citation": "Sucerquia, A., López, J. D., & Vargas-Bonilla, J. F. (2017). SisFall: A Fall and Movement Dataset. Sensors, 17(4), 833. https://doi.org/10.3390/s17040833",
        "license": "CC BY 4.0 (Zenodo v1 record).",
        "modality": "Wearable accelerometer and gyroscope time series, summarized per recorded event.",
        "task": "Fall event vs activity-of-daily-living event classification; not future fall-risk prediction.",
    },
    "essential_tremor": {
        "dataset": "Accelerometry recordings from essential tremor patients",
        "url": "https://doi.org/10.5281/zenodo.19130599",
        "download": "https://zenodo.org/api/records/19130599/files/acc_signal_database.mat/content",
        "citation": "Pardo-Valencia, J., Ammann, C., & Foffani, G. (2026). Accelerometry recordings from essential tremor patients. Zenodo. https://doi.org/10.5281/zenodo.19130599",
        "license": "CC BY 4.0.",
        "modality": "5000 Hz hand accelerometry under rest and posture conditions.",
        "task": "Within-essential-tremor Fahn-Tolosa-Marin (FTM) total-score regression; no healthy controls, so this is not an ET diagnostic classifier.",
    },
    "als": {
        "dataset": "GEO GSE233881 (sporadic ALS fibroblast expression)",
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE233881",
        "download": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE233nnn/GSE233881/matrix/GSE233881_series_matrix.txt.gz",
        "citation": "NCBI Gene Expression Omnibus GSE233881. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE233881",
        "license": "GEO public data; check original publication and GEO terms for reuse/attribution.",
        "modality": "Skin-fibroblast microarray gene-expression matrix.",
        "task": "Sporadic ALS vs healthy control gene-expression classification; not a speech or symptom progression model.",
    },
}

DEPRESSION_FEATURES = ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"]
STROKE_FEATURES = ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"]
PHQ_ITEMS = [f"DPQ0{index}0" for index in range(1, 10)]


@dataclass
class ResearchDataset:
    module_id: str
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    task: str
    profile: dict[str, Any]


def download_if_missing(url: str, path: Path, expected_size: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        return path
    request = Request(url, method="HEAD", headers={"User-Agent": "SustainX.Q research prototype"})
    with urlopen(request, timeout=30) as response:
        content_length = int(response.headers.get("Content-Length", "0"))
    size = expected_size or content_length
    if not size or (expected_size is not None and content_length and content_length != expected_size):
        raise RuntimeError(f"Invalid content length for {path.name}: {content_length}")
    part_path = path.with_suffix(path.suffix + ".part")
    if size <= 8 * 1024 * 1024:
        with urlopen(Request(url, headers={"User-Agent": "SustainX.Q research prototype"}), timeout=60) as response:
            payload = response.read()
        if len(payload) != size:
            raise RuntimeError(f"Incomplete download for {path.name}: {len(payload)} / {size} bytes")
        part_path.write_bytes(payload)
    else:
        chunk_size = 4 * 1024 * 1024
        starts = list(range(0, size, chunk_size))

        def fetch_chunk(start: int) -> tuple[int, bytes]:
            end = min(start + chunk_size, size) - 1
            last_error: Exception | None = None
            for _ in range(4):
                try:
                    chunk_request = Request(
                        url,
                        headers={"Range": f"bytes={start}-{end}", "User-Agent": "SustainX.Q research prototype"},
                    )
                    with urlopen(chunk_request, timeout=60) as response:
                        payload = response.read()
                        expected_range = f"bytes {start}-{end}/{size}"
                        if response.status != 206 or len(payload) != end - start + 1 or response.headers.get("Content-Range") != expected_range:
                            raise RuntimeError(f"Unexpected range response for {path.name} at byte {start}")
                        return start, payload
                except Exception as error:
                    last_error = error
            raise RuntimeError(f"Could not download {path.name} byte range {start}-{end}: {last_error}")

        part_path.unlink(missing_ok=True)
        with part_path.open("wb") as output:
            output.truncate(size)
            with ThreadPoolExecutor(max_workers=5) as executor:
                for future in as_completed([executor.submit(fetch_chunk, start) for start in starts]):
                    offset, payload = future.result()
                    output.seek(offset)
                    output.write(payload)
        if part_path.stat().st_size != size:
            part_path.unlink(missing_ok=True)
            raise RuntimeError(f"Incomplete ranged download for {path.name}")
    os.replace(part_path, path)
    return path


def _profile(module_id: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series, **extra) -> dict[str, Any]:
    return {
        "module_id": module_id,
        "dataset": SOURCES[module_id]["dataset"],
        "source_url": SOURCES[module_id]["url"],
        "download_url": SOURCES[module_id]["download"],
        "citation": SOURCES[module_id]["citation"],
        "license": SOURCES[module_id]["license"],
        "modality": SOURCES[module_id]["modality"],
        "task": SOURCES[module_id]["task"],
        "samples": int(len(X)),
        "subjects": int(groups.nunique()),
        "features": int(X.shape[1]),
        "feature_schema": list(X.columns),
        "missing_feature_cells": int(X.isna().sum().sum()),
        "target_distribution": {str(key): int(value) for key, value in y.value_counts(dropna=False).sort_index().items()},
        **extra,
    }


def load_alzheimer_dataset(raw_dir: Path = RAW_DATA) -> ResearchDataset:
    source = SOURCES["alzheimer"]
    archive_path = download_if_missing(source["download"], raw_dir / "darwin.zip", 1080804)
    with zipfile.ZipFile(archive_path) as archive:
        raw = pd.read_csv(io.BytesIO(archive.read("data.csv")))
    required = {"ID", "class"}
    if not required.issubset(raw.columns) or len(raw) != 174 or raw["ID"].nunique() != 174:
        raise ValueError("DARWIN schema/participant validation failed")
    labels = raw.pop("class").map({"H": 0, "P": 1})
    groups = raw.pop("ID").astype(str)
    if labels.isna().any() or labels.nunique() != 2:
        raise ValueError("DARWIN class labels must be H and P")
    X = raw.apply(pd.to_numeric, errors="coerce")
    profile = _profile(
        "alzheimer", X, labels.astype(int), groups,
        participants=groups.nunique(), modality_detail="Online handwriting; UCI DARWIN 174-participant dataset.",
        demographic_limitation="Dataset-level documentation reports 174 participants; population diversity/generalization is limited.",
    )
    return ResearchDataset("alzheimer", X, labels.astype(int), groups, "classification", profile)


def _nhanes_frames(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/"
    demo_path = download_if_missing(root + "DEMO_I.XPT", raw_dir / "nhanes_2015_2016_demo.xpt", 3756480)
    dpq_path = download_if_missing(root + "DPQ_I.XPT", raw_dir / "nhanes_2015_2016_dpq.xpt", 507040)
    mcq_path = download_if_missing(root + "MCQ_I.XPT", raw_dir / "nhanes_2015_2016_mcq.xpt", 6907360)
    return (
        pd.read_sas(demo_path, format="xport"),
        pd.read_sas(dpq_path, format="xport"),
        pd.read_sas(mcq_path, format="xport"),
    )


def _prepare_demographics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = frame.loc[:, columns].copy()
    for column in X:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    # NCHS special codes mean refused/don't know, not real numerical values.
    for column in ("RIAGENDR", "RIDRETH3", "DMDEDUC2"):
        if column in X:
            X[column] = X[column].replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})
    if "INDFMPIR" in X:
        X["INDFMPIR"] = X["INDFMPIR"].replace({"." : np.nan})
    return X


def load_depression_dataset(raw_dir: Path = RAW_DATA) -> ResearchDataset:
    demo, dpq, _ = _nhanes_frames(raw_dir)
    data = demo[["SEQN", *DEPRESSION_FEATURES]].merge(dpq[["SEQN", *PHQ_ITEMS]], on="SEQN", how="inner")
    data = data[pd.to_numeric(data["RIDAGEYR"], errors="coerce") >= 18].copy()
    for column in PHQ_ITEMS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = data[PHQ_ITEMS].notna().all(axis=1) & ~data[PHQ_ITEMS].isin([7, 9]).any(axis=1)
    data = data.loc[valid].copy()
    score = data[PHQ_ITEMS].sum(axis=1)
    y = (score >= 10).astype(int).rename("elevated_phq9")
    X = _prepare_demographics(data, DEPRESSION_FEATURES)
    groups = data["SEQN"].map(lambda value: str(int(value)))
    profile = _profile(
        "depression", X, y, groups,
        raw_dpq_rows=5735, age_eligibility="18 years and older", target_definition="PHQ-9 sum >= 10; 9 questionnaire items are excluded from predictors.",
        target_threshold=10, self_report=True,
        demographic_limitation="NHANES is a US population survey sample; demographic predictors are not a clinical speech model and survey weighting is not applied.",
    )
    return ResearchDataset("depression", X, y, groups, "classification", profile)


def load_stroke_history_dataset(raw_dir: Path = RAW_DATA) -> ResearchDataset:
    demo, _, mcq = _nhanes_frames(raw_dir)
    data = demo[["SEQN", *STROKE_FEATURES]].merge(mcq[["SEQN", "MCQ160F"]], on="SEQN", how="inner")
    data = data[pd.to_numeric(data["RIDAGEYR"], errors="coerce") >= 20].copy()
    target = pd.to_numeric(data["MCQ160F"], errors="coerce")
    valid = target.isin([1, 2])
    data = data.loc[valid].copy()
    target = target.loc[valid]
    y = (target == 1).astype(int).rename("reported_stroke_history")
    X = _prepare_demographics(data, STROKE_FEATURES)
    groups = data["SEQN"].map(lambda value: str(int(value)))
    profile = _profile(
        "stroke_history", X, y, groups,
        raw_mcq_rows=9575, age_eligibility="20 years and older", target_definition="MCQ160F: participant reported ever being told by a health professional that they had a stroke.",
        target_codes={"1": "Yes", "2": "No"}, acute_stroke=False,
        limitation="Retrospective self-reported history classification from demographic features; not acute stroke detection, diagnosis, or recovery prediction.",
    )
    return ResearchDataset("stroke_history", X, y, groups, "classification", profile)


def _sisfall_features(samples: np.ndarray) -> dict[str, float]:
    feature_row: dict[str, float] = {}
    channel_names = ["adxl_x", "adxl_y", "adxl_z", "gyro_x", "gyro_y", "gyro_z", "mma_x", "mma_y", "mma_z"]
    for column, name in enumerate(channel_names):
        signal = np.asarray(samples[:, column], dtype=np.float64)
        feature_row[f"{name}_mean"] = float(np.mean(signal))
        feature_row[f"{name}_std"] = float(np.std(signal))
        feature_row[f"{name}_rms"] = float(np.sqrt(np.mean(signal * signal)))
        feature_row[f"{name}_range"] = float(np.ptp(signal))
        feature_row[f"{name}_q10"] = float(np.quantile(signal, 0.10))
        feature_row[f"{name}_q90"] = float(np.quantile(signal, 0.90))
    for sensor_name, columns in (("adxl", slice(0, 3)), ("gyro", slice(3, 6)), ("mma", slice(6, 9))):
        magnitude = np.linalg.norm(samples[:, columns], axis=1)
        feature_row[f"{sensor_name}_magnitude_mean"] = float(np.mean(magnitude))
        feature_row[f"{sensor_name}_magnitude_std"] = float(np.std(magnitude))
        feature_row[f"{sensor_name}_magnitude_max"] = float(np.max(magnitude))
    return feature_row


def load_fall_event_dataset(raw_dir: Path = RAW_DATA) -> ResearchDataset:
    source = SOURCES["fall_event"]
    archive_path = download_if_missing(source["download"], raw_dir / "sisfall.zip", 238324928)
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"/(?:SA\d+|SE\d+)/(D\d{2}|F\d{2})_(SA\d{2}|SE\d{2})_R\d+\.txt$", re.IGNORECASE)
    with zipfile.ZipFile(archive_path) as archive:
        member_names = [name for name in archive.namelist() if pattern.search(name)]
        for name in member_names:
            match = pattern.search(name)
            assert match is not None
            code, subject = match.groups()
            raw = archive.read(name).replace(b";", b"")
            values = np.fromstring(raw.decode("ascii"), sep=",")
            if values.size == 0 or values.size % 9 != 0:
                raise ValueError(f"Unexpected SisFall recording shape: {name}")
            samples = values.reshape(-1, 9)
            rows.append({
                **_sisfall_features(samples),
                "subject": subject.upper(),
                "recording": Path(name).name,
                "target": int(code.upper().startswith("F")),
            })
    frame = pd.DataFrame(rows)
    if frame.empty or frame["subject"].nunique() < 10 or frame["target"].nunique() != 2:
        raise ValueError("SisFall validation failed: insufficient participant/class coverage")
    feature_columns = [column for column in frame.columns if column not in {"subject", "recording", "target"}]
    X = frame[feature_columns]
    y = frame["target"].astype(int)
    groups = frame["subject"].astype(str)
    profile = _profile(
        "fall_event", X, y, groups,
        recording_count=len(frame), subjects=groups.nunique(), feature_engineering="Per-recording time-domain summary features for 9 accelerometer/gyroscope axes and three sensor magnitudes.",
        split_note="Grouped by participant; every event from one participant stays in one fold.",
        limitation="Controlled scripted falls and ADLs; event detection is not prospective fall-risk estimation. Participants are healthy volunteers, not a clinical elderly fall-risk cohort.",
    )
    return ResearchDataset("fall_event", X, y, groups, "classification", profile)


def _spectral_features(signal: np.ndarray, fs: float) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)[::10]
    downsampled_fs = fs / 10
    frequencies, power = welch(signal, fs=downsampled_fs, nperseg=min(2048, len(signal)))
    band = (frequencies >= 2) & (frequencies <= 20)
    if band.any():
        band_freqs = frequencies[band]
        band_power = power[band]
        peak_index = int(np.argmax(band_power))
        normalized = band_power / max(float(np.sum(band_power)), np.finfo(float).eps)
        spectral_entropy = float(-np.sum(normalized * np.log(normalized + np.finfo(float).eps)))
        peak_frequency = float(band_freqs[peak_index])
        integrated_power = float(np.trapezoid(band_power, band_freqs))
    else:
        spectral_entropy = peak_frequency = integrated_power = 0.0
    return {
        "rms": float(np.sqrt(np.mean(signal * signal))),
        "std": float(np.std(signal)),
        "peak_frequency_hz": peak_frequency,
        "band_power_2_20hz": integrated_power,
        "spectral_entropy_2_20hz": spectral_entropy,
    }


def load_essential_tremor_dataset(raw_dir: Path = RAW_DATA) -> ResearchDataset:
    source = SOURCES["essential_tremor"]
    archive_path = download_if_missing(source["download"], raw_dir / "et_accelerometry.mat", 90073568)
    clinical_path = download_if_missing(
        "https://zenodo.org/api/records/19130599/files/clinical_data.mat/content",
        raw_dir / "et_clinical.mat",
        800,
    )
    signal_records = loadmat(archive_path, squeeze_me=True, struct_as_record=False)["acc_signal_database"]
    clinical_records = loadmat(clinical_path, squeeze_me=True, struct_as_record=False)["clinical_data"]
    scores: dict[str, float] = {}
    for record in clinical_records:
        subject = str(record.subject)
        try:
            score = float(record.FTM_total)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            scores[subject] = score
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in signal_records:
        subject = str(record.subject)
        if subject not in scores:
            continue
        grouped.setdefault(subject, {})
        for condition in ("rest", "posture"):
            features = _spectral_features(getattr(record, condition), float(record.Fs))
            for name, value in features.items():
                grouped[subject].setdefault(f"{condition}_{name}", []).append(value)
    rows = []
    for subject, features in grouped.items():
        row = {name: float(np.mean(values)) for name, values in features.items()}
        row["subject"] = subject
        row["target"] = scores[subject]
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) < 20:
        raise ValueError("Essential tremor severity dataset contains fewer than 20 scored participants")
    X = frame.drop(columns=["subject", "target"])
    y = frame["target"].astype(float)
    groups = frame["subject"].astype(str)
    profile = _profile(
        "essential_tremor", X, y, groups,
        recording_count=len(signal_records), eligible_scored_subjects=len(frame), target_definition="Clinical FTM_total tremor severity score, aggregated once per participant.",
        clinical_labels="All participants have essential tremor; no healthy comparison group is provided.",
        limitation="29-patient cohort, single source, no controls, small sample; this is exploratory within-ET severity regression and cannot diagnose ET.",
    )
    return ResearchDataset("essential_tremor", X, y, groups, "regression", profile)


def load_als_dataset(raw_dir: Path = RAW_DATA) -> ResearchDataset:
    source = SOURCES["als"]
    matrix_path = download_if_missing(source["download"], raw_dir / "als_GSE233881_series_matrix.txt.gz", 2285340)
    with gzip.open(matrix_path, "rt", encoding="latin-1") as compressed:
        lines = compressed.read().splitlines()
    title_line = next((line for line in lines if line.startswith("!Sample_title\t")), None)
    disease_line = next((line for line in lines if line.startswith("!Sample_characteristics_ch1\t\"disease state:")), None)
    try:
        table_start = lines.index("!series_matrix_table_begin") + 1
        table_end = lines.index("!series_matrix_table_end")
    except ValueError as error:
        raise ValueError("GSE233881 series matrix table markers are missing") from error
    if title_line is None or disease_line is None:
        raise ValueError("GSE233881 sample titles or disease-state annotations are missing")
    sample_titles = next(csv.reader([title_line.split("\t", 1)[1]], delimiter="\t"))
    disease_states = next(csv.reader([disease_line.split("\t", 1)[1]], delimiter="\t"))
    if len(sample_titles) != 12 or len(disease_states) != 12:
        raise ValueError("Expected 12 participant-level GSE233881 labels")
    sample_names = [title.strip('"') for title in sample_titles]
    expression = pd.read_csv(io.StringIO("\n".join(lines[table_start:table_end])), sep="\t", index_col=0)
    if expression.shape[1] != len(sample_names):
        raise ValueError("GSE233881 expression columns do not match sample annotations")
    expression.columns = sample_names
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = expression.groupby(level=0, sort=False).mean()
    matrix = expression.T
    y = pd.Series(
        [int(state.strip('"').lower().startswith("disease state: sporadic amyotrophic lateral sclerosis")) for state in disease_states],
        index=sample_names,
        name="als",
    )
    groups = pd.Series(sample_names, index=sample_names, dtype=str)
    profile = _profile(
        "als", matrix, y, groups,
        primary_series="GSE233881",
        label_counts={"healthy control": int((y == 0).sum()), "sporadic ALS": int((y == 1).sum())},
        limitation="Only 12 participants (9 sporadic ALS, 3 controls); very small and class-imbalanced. Results are exploratory and not generalizable or clinically validated.",
    )
    return ResearchDataset("als", matrix, y, groups, "classification", profile)