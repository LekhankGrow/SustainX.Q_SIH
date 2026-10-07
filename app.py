from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from src.baseline import BASELINE_FEATURES, compare_to_baseline, make_entry
from src.data import DATA_URL
from src.disease_registry import get_research_registry
from src.experiment import load_predictions, load_report, run_experiment
from src.quality import assess_audio

st.set_page_config(
    page_title="SustainX.Q | Voice Research",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = "SustainX.Q is a research prototype and screening-support platform. It is not a medical diagnostic system. Parkinson's and other disease modules are based on exploratory machine-learning experiments using public research data and are not clinical validation. Additional modules are included when their data, evaluation, and artifact metadata are available and are explicitly marked as research-only."

RESEARCH_MODULES = get_research_registry()["modules"]
ACTIVE_RESEARCH_MODULES = [module for module in RESEARCH_MODULES if module["id"] != "skin_future"]
DISPLAY_NAME_BY_ID = {
    "parkinson": "Parkinson's Disease",
    "alzheimer": "Alzheimer's / MCI",
    "depression": "Depression",
    "stroke_history": "Stroke",
    "als": "ALS",
    "essential_tremor": "Essential Tremor",
    "fall_event": "Fall Risk",
}
DISPLAY_NAME_BY_ID["skin_future"] = "Skin Lesion Screening"
FUTURE_MODULES = {"Skin Lesion Screening": {"signal": "Image-based screening research only", "purpose": "Future image-feature and questionnaire research direction.", "direction": "Not currently implemented."}}

SKIN_FUTURE = {
    "title": "Skin Lesion Screening — Future Update",
    "description": "Planned future module exploring image-based skin screening using visual features and questionnaire-based risk signals.",
    "status": "Not currently implemented.",
}

st.markdown(
    """
    <style>
      :root {
        --ink: #142b27; --muted: #60716c; --paper: #f3f6f2; --line: #dbe4dc;
        --green: #155b49; --lime: #d7ef78; --coral: #e47b62; --blue: #4168bc;
      }
    html, body, [class*="css"] { font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; color: var(--ink); }
      .stApp { background: var(--paper); }
    h1, h2, h3 { font-family: 'Georgia', serif !important; letter-spacing: 0 !important; color: var(--ink); }
      h1 { font-size: 2.35rem !important; line-height: 1.12 !important; }
      .block-container { max-width: 1320px; padding-top: 2.1rem; padding-bottom: 4rem; }
      [data-testid="stSidebar"] { background: #e8eee8; border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
      .brand { display:flex; align-items:center; gap:10px; margin: 4px 0 28px; }
      .brand-mark { width:34px;height:34px;display:grid;place-items:center;border-radius:10px;background:var(--green);color:white;font-size:18px; }
    .brand-name { font-family:'Georgia',serif;font-weight:700;font-size:17px;line-height:1.1; }
      .brand-sub { color:var(--muted);font-size:11px;margin-top:3px; }
      .eyebrow { color:var(--green);text-transform:uppercase;font-weight:700;font-size:11px;letter-spacing:1.5px;margin-bottom:8px; }
      .lead { color:var(--muted);font-size:16px;line-height:1.65;max-width:760px; }
      .notice { border:1px solid #ecc6b9;background:#fff0e9;padding:12px 15px;border-radius:8px;color:#713e32;font-size:13px;font-weight:600;margin: 4px 0 18px; }
      .surface { background:#fff;border:1px solid var(--line);border-radius:8px;padding:20px 22px; }
      .metric-label { color:var(--muted);font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.6px; }
    .metric-value { color:var(--ink);font-family:'Georgia',serif;font-size:25px;font-weight:700;margin-top:6px; }
      .metric-note { color:var(--muted);font-size:12px;margin-top:4px; }
      .section-title { font-family:'Space Grotesk',sans-serif;font-size:19px;font-weight:700;margin: 18px 0 8px; }
      .small-note { color:var(--muted);font-size:12px;line-height:1.5; }
      .tag { display:inline-block;padding:4px 9px;border-radius:4px;background:#e7eedb;color:#315741;font-size:11px;font-weight:700; }
    .app-footer { border-top:1px solid var(--line);padding-top:14px;margin-top:42px;color:var(--muted);font-size:11px;text-align:center; }
      .stButton > button, .stDownloadButton > button { border-radius:6px;border:1px solid var(--green);font-weight:700; }
      .stButton > button[kind="primary"] { background:var(--green); }
      [data-testid="stMetric"] { background:white;border:1px solid var(--line);padding:14px 16px;border-radius:8px; }
      div[data-testid="stAlert"] { border-radius:7px; }
      footer { visibility:hidden; }
      @media(max-width:700px) { .block-container {padding:1.2rem 1rem 3rem;} h1 {font-size:1.85rem !important;} }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p class="lead">{description}</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="notice">{DISCLAIMER}</div>', unsafe_allow_html=True)


def show_metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="surface"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def render_home() -> None:
    page_header(
        "SIH 2026 · Problem Statement 26139",
        "SustainX.Q",
        "Hybrid Quantum-Classical Intelligence for Early Disease Screening Research. SustainX.Q is a research-oriented platform exploring hybrid classical and quantum machine learning for early health and neurological screening signals.",
    )
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown('<div class="surface" style="border-top:4px solid #155b49"><div class="eyebrow">Active research module · Parkinson\'s Disease</div><h2 style="margin-top:0">Voice screening signals</h2><p class="lead">Explore voice-based screening signals using classical and hybrid quantum-classical machine learning.</p><span class="tag">ACTIVE</span><p class="small-note" style="margin-top:12px">Research prototype only. Microphone quality and personal baseline features are not scored by the UCI-trained model.</p></div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Start Parkinson's voice screening", type="primary", use_container_width=True):
            st.session_state["active_page"] = "Voice screening"
            st.rerun()
    with right:
        report = load_report()
        state = "Evaluation not run yet" if report is None else "Measured benchmark available"
        status = "Awaiting real evaluation" if report is None else report.get("quantum_utility", {}).get("status", "Quantum evaluation " + report.get("quantum_status", "unknown"))
        st.markdown(f'<div class="surface"><div class="eyebrow">Current research state</div><h3>{state}</h3><p>{status}</p><div class="small-note">No model output is generated from the microphone. No clinical validation is claimed.</div></div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="surface"><div class="eyebrow">Research question</div><h3 style="margin-top:0">Does the hybrid model help?</h3><p>We compare it with classical baselines under the same person-level evaluation.</p><div class="small-note">An inconclusive result is valid. Quantum benefit is not assumed.</div></div>', unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Available research modules</div>", unsafe_allow_html=True)
    module_columns = st.columns(3)
    for index, module in enumerate(ACTIVE_RESEARCH_MODULES):
        with module_columns[index % len(module_columns)]:
            badge = "ACTIVE" if module["status"] == "active_research" else module["status"].replace("_", " ").upper()
            display_name = DISPLAY_NAME_BY_ID.get(module["id"], module["name"])
            st.markdown(f'<div class="surface"><div class="eyebrow">{display_name}</div><h3>{display_name}</h3><p class="small-note">{module["input_modality"]}</p><span class="tag">{badge}</span></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("<div class='section-title'>Built around careful evaluation</div>", unsafe_allow_html=True)
    cards = st.columns(3)
    for column, label, text in zip(
        cards,
        ["Person-level splits", "Compact quantum circuit", "Clear input boundary"],
        ["All recordings from one person stay together during evaluation.", "PennyLane simulator with a small, fold-fitted feature bottleneck.", "Microphone checks and the UCI model benchmark remain separate."],
        strict=True,
    ):
        with column:
            st.markdown(f'<div class="surface"><div class="eyebrow">{label}</div><div>{text}</div></div>', unsafe_allow_html=True)
    with st.expander("Future Updates"):
        st.markdown(f'<div class="surface"><div class="eyebrow">Future Update</div><h3>{SKIN_FUTURE["title"]}</h3><p>{SKIN_FUTURE["description"]}</p><span class="tag">{SKIN_FUTURE["status"]}</span></div>', unsafe_allow_html=True)


def render_modules() -> None:
    page_header("Module directory · research status", "Parkinson plus modular biomedical research experiments.", "The app includes Parkinson's voice screening and the additional research modules that have validated data and measured artifact metadata. Other modules remain clearly labeled as research-only and are not clinical predictors.")
    module_names = [
        "Parkinson's Disease",
        "Alzheimer's / MCI",
        "Depression",
        "Stroke",
        "ALS",
        "Essential Tremor",
        "Fall Risk",
    ]
    selected = st.selectbox("Select research module", module_names, key="module_selector")
    selected_module = next(module for module in ACTIVE_RESEARCH_MODULES if DISPLAY_NAME_BY_ID.get(module["id"]) == selected)
    status = selected_module["status"]
    if selected_module["id"] == "parkinson":
        st.markdown('<div class="surface" style="border-top:4px solid #155b49"><div class="eyebrow">ACTIVE · CURRENTLY IMPLEMENTED</div><h2>Parkinson\'s Disease</h2><p class="lead">Voice-based research screening signals using classical and hybrid quantum-classical machine learning.</p><p>Includes recording quality checks, an optional personal acoustic baseline, subject-disjoint model benchmarks, and explainability.</p><p class="small-note">Microphone recordings are not scored by the UCI-trained model because feature compatibility is not established.</p></div>', unsafe_allow_html=True)
        actions = st.columns(2)
        with actions[0]:
            if st.button("Open voice screening", type="primary", use_container_width=True):
                st.session_state["active_page"] = "Voice screening"
                st.rerun()
        with actions[1]:
            if st.button("Open measured benchmark", use_container_width=True):
                st.session_state["active_page"] = "Model benchmark"
                st.rerun()
    else:
        details = selected_module
        badge = "ACTIVE" if status == "active_research" else status.upper().replace("_", " ")
        st.markdown(f'<div class="surface"><div class="eyebrow">Research Extension</div><h2>{selected}</h2><span class="tag">Experimental / Future Module</span><p><b>Potential signals</b><br>{details["input_modality"]}</p><p><b>Research purpose</b><br>{details["purpose"]}</p><p><b>Future direction</b><br>{details["limitation"]}</p><p><b>Status</b><br>Coming in future research versions. Experimental research module — model not currently validated.</p><p class="small-note">This preview does not provide a screening, prediction, confidence, or risk score.</p></div>', unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Included research modules</div>", unsafe_allow_html=True)
    columns = st.columns(3)
    for index, module in enumerate(ACTIVE_RESEARCH_MODULES):
        with columns[index % len(columns)]:
            badge = "ACTIVE" if module["status"] == "active_research" else module["status"].upper().replace("_", " ")
            display_name = DISPLAY_NAME_BY_ID.get(module["id"], module["name"])
            st.markdown(f'<div class="surface"><div class="eyebrow">{display_name}</div><h3>{display_name}</h3><p class="small-note">{module["input_modality"]}</p><span class="tag">{badge}</span></div>', unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Future Updates</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="surface"><div class="eyebrow">Future Update</div><h3>{SKIN_FUTURE["title"]}</h3><p>{SKIN_FUTURE["description"]}</p><span class="tag">{SKIN_FUTURE["status"]}</span></div>', unsafe_allow_html=True)


def render_voice() -> None:
    page_header("Input quality · no diagnosis", "A better recording starts here.", "Check signal quality and optionally track your own acoustic measurements. These measurements are not fed into the UCI disease model.")
    st.info("Your recording is processed by this app for signal checks only. Raw audio is not written to persistent storage or sent to an external AI service. Derived measurements can remain in this session's app memory unless you delete them or export them.")
    audio = st.audio_input("Record a short sustained voice sample")
    if audio is None:
        st.caption("Aim for 3–15 seconds in a quiet room, with the microphone about 15–20 cm away.")
        return
    try:
        assessment = assess_audio(audio.getvalue())
    except ValueError as error:
        st.error(str(error))
        return
    st.session_state["latest_audio_assessment"] = assessment.to_dict()
    if assessment.passed:
        st.success("Recording quality passed the basic signal checks.")
    else:
        st.error("Recording quality is insufficient. Please record again before comparing measurements.")
        for reason in assessment.reasons:
            st.write(f"• {reason}")
    quality_columns = st.columns(4)
    quality_columns[0].metric("Duration", f"{assessment.duration_seconds:.1f} s")
    quality_columns[1].metric("Signal level (RMS)", f"{assessment.rms:.3f}")
    quality_columns[2].metric("Clipped samples", f"{assessment.clipping_fraction:.2%}")
    quality_columns[3].metric("Silent frames", f"{assessment.silent_frame_fraction:.1%}")
    st.caption(f"Activity proxy: {assessment.speech_activity_proxy:.1%}. This is a signal-energy proxy, not speech recognition or proof that the recording contains a voice.")
    if st.button("View recording summary"):
        st.session_state["active_page"] = "Screening result"
        st.rerun()
    st.markdown("<div class='section-title'>Model scoring status</div>", unsafe_allow_html=True)
    st.warning("Live microphone disease inference is unavailable. The benchmark model expects UCI engineered voice measurements; these microphone-derived acoustic features have not been shown compatible. No disease score was produced.")
    if assessment.passed:
        if st.button("Add quality-passing sample to my session baseline", type="primary"):
            entries = st.session_state.setdefault("baseline_entries", [])
            entries.append(make_entry(assessment.features))
            st.session_state["baseline_entries"] = entries
            st.success("Derived measurements added. Raw audio was not added to your baseline.")


def render_baseline() -> None:
    page_header("Your measurements · local to this session", "Change relative to your personal baseline.", "A personal baseline can show how your acoustic measurements vary over time. It is not a diagnostic reference and may itself contain atypical variation.")
    entries = st.session_state.setdefault("baseline_entries", [])
    st.info("The baseline is held in temporary app-session memory and is not written to persistent server storage. Export it to keep a copy between visits; it contains numeric measurements and timestamps, not raw recordings.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="surface"><div class="eyebrow">Session baseline</div></div>', unsafe_allow_html=True)
        st.metric("Quality-passing samples", len(entries))
        if entries:
            current_features = entries[-1]["features"]
            comparison = compare_to_baseline(current_features, entries[:-1] or entries)
            comparison_frame = pd.DataFrame([
                {"Feature": name.replace("_", " ").title(), "Baseline": values["baseline"], "Current": values["current"], "Change (%)": values["relative_change_percent"]}
                for name, values in comparison.items()
            ])
            st.dataframe(comparison_frame, hide_index=True, use_container_width=True)
            st.caption("Most recent sample compared with the mean of earlier samples. When there is only one sample, it is compared with itself.")
        else:
            st.caption("No baseline yet. Add a quality-passing sample from Voice screening.")
    with right:
        if entries:
            timeline = pd.DataFrame([
                {"Captured": entry["captured_at"], **entry["features"]}
                for entry in entries
            ])
            feature = st.selectbox("Acoustic measure", BASELINE_FEATURES, format_func=lambda name: name.replace("_", " ").title())
            st.plotly_chart(px.line(timeline, x="Captured", y=feature, markers=True, color_discrete_sequence=["#155b49"]), use_container_width=True)
            payload = json.dumps(entries, indent=2)
            st.download_button("Export session baseline (JSON)", payload, file_name="sustainx-q-voice-baseline.json", mime="application/json")
        imported = st.file_uploader("Import a previous baseline", type=["json"], key="baseline_import")
        if imported is not None and st.button("Load imported baseline"):
            try:
                incoming = json.loads(imported.getvalue())
                if not isinstance(incoming, list):
                    raise ValueError("Expected a list of baseline entries")
                checked = [make_entry(item["features"], item["captured_at"]) for item in incoming]
                st.session_state["baseline_entries"] = checked
                st.rerun()
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                st.error(f"Baseline file could not be loaded: {error}")
    if entries and st.button("Delete session baseline", type="secondary"):
        st.session_state["baseline_entries"] = []
        st.rerun()
    st.caption("A change in acoustic measurements is not a diagnosis and must not be interpreted as disease progression.")


def render_result() -> None:
    page_header("Session result · quality and measurement", "Your recording summary.", "This result describes recording quality and acoustic measurements only. Disease inference from microphone input is unavailable.")
    assessment = st.session_state.get("latest_audio_assessment")
    if assessment is None:
        st.info("No recording has been assessed in this session yet.")
        if st.button("Go to voice screening", type="primary"):
            st.session_state["active_page"] = "Voice screening"
            st.rerun()
        return
    if assessment["passed"]:
        st.success("Basic recording quality checks passed.")
    else:
        st.error("Recording quality is insufficient. Retake before using these measurements.")
        for reason in assessment["reasons"]:
            st.write(f"• {reason}")
    columns = st.columns(4)
    columns[0].metric("Duration", f"{assessment['duration_seconds']:.1f} s")
    columns[1].metric("RMS level", f"{assessment['rms']:.3f}")
    columns[2].metric("Clipped samples", f"{assessment['clipping_fraction']:.2%}")
    columns[3].metric("Silent frames", f"{assessment['silent_frame_fraction']:.1%}")
    st.warning("No Parkinson's disease score was produced. The UCI benchmark uses engineered features that are not known to match these microphone-derived measurements.")
    if assessment["passed"]:
        entries = st.session_state.get("baseline_entries", [])
        if len(entries) > 1:
            comparison = compare_to_baseline(assessment["features"], entries[:-1])
            table = pd.DataFrame([
                {"Acoustic measurement": name.replace("_", " ").title(), "Change vs. earlier personal baseline (%)": values["relative_change_percent"]}
                for name, values in comparison.items()
            ])
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption("Descriptive change only. A personal baseline can itself contain atypical variation and is not a diagnostic reference.")
        else:
            st.caption("Add repeated quality-passing recordings to compare measurements with your personal baseline.")
    st.info("If you are concerned about your health, contact a qualified healthcare professional. Do not use this prototype to delay care.")


def _run_benchmark(qubits: int = 4, epochs: int = 15) -> None:
    with st.status("Running grouped benchmark", expanded=True) as status:
        st.write("Downloading and validating the UCI dataset if needed…")
        st.write(f"Evaluating classical baselines and the {qubits}-qubit PennyLane simulator model…")
        report = run_experiment(folds=5, qubits=qubits, layers=2, epochs=epochs)
        status.update(label="Benchmark run finished", state="complete")
    if report.get("quantum_status") == "completed":
        st.success("Actual classical and quantum results were saved.")
    else:
        st.warning("Classical results were saved. Quantum evaluation is marked unavailable; see the run status for the actual error.")
    st.rerun()


def render_benchmark() -> None:
    page_header("Measured results · subject-disjoint", "Classical vs. quantum.", "Every number below comes from saved out-of-fold predictions. Results are exploratory: this dataset contains few independent people and is not a clinical validation study.")
    report = load_report()
    if report is None:
        st.warning("No benchmark has been run yet. No results are displayed until models are evaluated on the real dataset.")
        qubits = st.selectbox("Quantum input / qubit budget", [4, 6, 8], index=0)
        epochs = st.selectbox("Training epochs", [5, 10, 15], index=2)
        if st.button("Run real-data benchmark", type="primary"):
            _run_benchmark(qubits, epochs)
        st.caption("First run downloads the UCI Parkinsons feature dataset. This can take several minutes, especially for the simulator model.")
        return
    dataset = report.get("dataset", {})
    info = st.columns(4)
    info[0].metric("Recordings", dataset.get("rows", "Not validated"))
    info[1].metric("People", dataset.get("subjects", "Not validated"))
    info[2].metric("Engineered features", dataset.get("features", "Not validated"))
    info[3].metric("Missing cells", dataset.get("missing_cells", "Not validated"))
    st.caption(f"Data source: {DATA_URL} · UCI DOI: 10.24432/C59C74 · License: {dataset.get('license', 'CC BY 4.0')}")
    st.caption(f"Recording labels: {dataset.get('labels', {})} · Person labels: {dataset.get('subject_labels', {})}")
    if dataset.get("rows") is not None and dataset["rows"] != 197:
        st.info(f"The downloaded table contains {dataset['rows']} rows. UCI's page also displays a 197-instance headline; this app reports the validated file count.")
    classical = report.get("classical_models", {})
    quantum = report.get("quantum", {})
    predictions = load_predictions()
    if not classical:
        st.error("Benchmark metadata is incomplete; no model results are shown.")
        return
    rows = []
    for name, values in classical.items():
        rows.append({
            "Model": name,
            **{metric: values.get(metric) for metric in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")},
            "Accuracy (fold mean ± SD)": _mean_sd(values, "accuracy"),
            "ROC-AUC (fold mean ± SD)": _mean_sd(values, "roc_auc"),
            "Train time (s)": values.get("training_seconds_mean"),
            "Inference (ms/recording)": values.get("inference_ms_per_recording_mean"),
            "Complexity / setup": {
                "Logistic Regression": "22 inputs · linear",
                "RBF SVM": "22 inputs · RBF kernel",
                "Random Forest": "250 trees · min leaf 2",
            }.get(name, "Classical pipeline"),
        })
    if report.get("quantum_status") == "completed":
        q = quantum.get("summary", {})
        rows.append({
            "Model": f"PennyLane VQC ({quantum.get('configuration', {}).get('qubits')} qubits)",
            **{metric: q.get(metric) for metric in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")},
            "Accuracy (fold mean ± SD)": _mean_sd(quantum, "accuracy"),
            "ROC-AUC (fold mean ± SD)": _mean_sd(quantum, "roc_auc"),
            "Train time (s)": quantum.get("training_seconds_mean"),
            "Inference (ms/recording)": quantum.get("inference_ms_per_recording_mean"),
            "Complexity / setup": f"{quantum.get('configuration', {}).get('qubits')} qubits · {quantum.get('configuration', {}).get('layers')} layers",
        })
    st.markdown("<div class='section-title'>Out-of-fold results, aggregated per person</div>", unsafe_allow_html=True)
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table.style.format(precision=3, na_rep="n/a", subset=["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "Train time (s)", "Inference (ms/recording)"]), use_container_width=True)
    st.caption("Metrics are calculated after averaging each person's recording scores. ROC-AUC and model scores are exploratory and not medically calibrated.")
    utility = report.get("quantum_utility")
    if utility:
        st.markdown("<div class='section-title'>Quantum Utility</div>", unsafe_allow_html=True)
        st.markdown(f"**{utility.get('status', 'Insufficient evidence')}**")
        st.write(utility.get("reason", "The comparison could not be completed."))
        if utility.get("observed_delta") is not None:
            st.caption(f"Paired subject-level ROC-AUC difference: {utility['observed_delta']:.3f} · 95% bootstrap interval: {utility['bootstrap_95_percent_interval']} · {utility['subject_count']} people")
        st.caption(utility.get("caveat", ""))
    elif report.get("quantum_status") != "completed":
        st.warning(f"Quantum evaluation unavailable: {report.get('quantum_failure', {}).get('type', 'not run')}: {report.get('quantum_failure', {}).get('message', 'No quantum result was produced.')}")
    _render_feature_vector_inference(report)
    if predictions is not None:
        agreement = _model_agreement(predictions)
        if agreement is not None:
            st.markdown("<div class='section-title'>Model agreement</div>", unsafe_allow_html=True)
            st.metric("People with unanimous model threshold signal", f"{agreement:.1%}")
            if agreement < 0.8:
                st.warning("Models disagree for some people. This is not validated confidence; seek professional advice for health concerns.")
            else:
                st.info("Models show broadly consistent threshold signals in this exploratory benchmark; agreement is not validated confidence.")
            st.caption("Agreement at a 0.5 score threshold is descriptive only; scores are not calibrated clinical confidence.")
    controls = st.columns([1, 1, 2])
    with controls[0]:
        qubits = st.selectbox("Quantum input / qubit budget", [4, 6, 8], index=0, key="benchmark_qubits")
    with controls[1]:
        epochs = st.selectbox("Training epochs", [5, 10, 15], index=2, key="benchmark_epochs")
    if st.button("Rerun evaluation", type="secondary"):
        _run_benchmark(qubits, epochs)


def _render_feature_vector_inference(report: dict) -> None:
    with st.expander("Research inference from compatible engineered features"):
        st.warning("This accepts one row of UCI-compatible numerical features only. It does not accept microphone recordings. Scores are uncalibrated and not a diagnosis; the bundled models were refit on all UCI records and this input has no independent validation guarantee.")
        manifest_path = __import__("pathlib").Path("artifacts/model_manifest.json")
        if not manifest_path.exists():
            st.caption("Model artifacts are not available yet. Run the benchmark to create them.")
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        template = pd.DataFrame(columns=manifest["feature_columns"]).to_csv(index=False)
        st.download_button("Download feature CSV template", template, file_name="sustainx-q-feature-template.csv", mime="text/csv")
        uploaded = st.file_uploader("Upload a one-row CSV with exactly these 22 features", type=["csv"], key="feature_inference_csv")
        if uploaded is None or not st.button("Run research feature-vector inference", key="run_feature_inference"):
            return
        try:
            input_frame = pd.read_csv(uploaded)
            from src.inference import load_model_bundle, predict_compatible_features

            bundle, loaded_manifest = load_model_bundle()
            result = predict_compatible_features(input_frame, bundle, loaded_manifest)
            st.markdown("**Model outputs (uncalibrated; not risk probabilities)**")
            score_frame = pd.DataFrame([
                {"Model": name, "Score": score, "Threshold signal at 0.5": "Higher" if result["threshold_signals"][name] else "Lower"}
                for name, score in result["model_scores"].items()
            ])
            st.dataframe(score_frame, hide_index=True, use_container_width=True)
            st.metric("Model agreement at the experimental threshold", f"{result['model_agreement_fraction']:.0%}")
            st.caption(result["score_note"])
            st.caption(result["validation_note"])
            importance = report.get("explainability", {}).get("feature_importance_mean", [])
            if importance:
                values = input_frame.iloc[0]
                top_features = pd.DataFrame(importance).groupby("feature")["importance_mean"].mean().nlargest(5).index
                st.markdown("Most influential features in held-out classical model analysis, with values from this input:")
                st.dataframe(pd.DataFrame([{"Feature": name, "Input value": float(values[name])} for name in top_features]), hide_index=True, use_container_width=True)
                st.caption("Global feature importance is not a causal or person-specific explanation.")
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as error:
            st.error(f"Input or model artifact rejected: {error}")


def _model_agreement(predictions: pd.DataFrame) -> float | None:
    if predictions.empty:
        return None
    votes = predictions.assign(signal=predictions["score"] >= 0.5).pivot_table(index="subject", columns="model", values="signal", aggfunc="first")
    votes = votes.dropna()
    if votes.empty:
        return None
    return float(votes.nunique(axis=1).eq(1).mean())


def _mean_sd(model_report: dict, metric: str) -> str:
    means = model_report.get("fold_metrics_mean", {})
    deviations = model_report.get("fold_metrics_std", {})
    mean = means.get(metric)
    deviation = deviations.get(metric)
    if mean is None or deviation is None:
        return "n/a"
    return f"{mean:.3f} ± {deviation:.3f}"


def render_quantum_lab() -> None:
    page_header("Quantum Lab · local simulator", "A compact circuit, measured in context.", "The circuit tests a hybrid approach: classical fold-local feature reduction feeds a shallow variational quantum circuit simulated locally.")
    report = load_report()
    q = report.get("quantum", {}) if report else {}
    config = q.get("configuration", {})
    if report is None or report.get("quantum_status") != "completed":
        st.warning("No completed quantum experiment is available yet. The lab will not show simulated or placeholder performance.")
        st.code("Angle embedding (Y) → shallow strongly entangling layers → Pauli-Z expectation")
        if report and report.get("quantum_failure"):
            st.caption(f"Latest attempt: {report['quantum_failure'].get('type')}: {report['quantum_failure'].get('message')}")
        return
    cols = st.columns(4)
    cols[0].metric("Qubits", config.get("qubits", "n/a"))
    cols[1].metric("Circuit layers", config.get("layers", "n/a"))
    cols[2].metric("Encoding", "Angle · Y")
    cols[3].metric("Simulator", "default.qubit")
    st.markdown("<div class='section-title'>Adaptive feature bottleneck</div>", unsafe_allow_html=True)
    fold_info = pd.DataFrame(config.get("fold_configurations", []))
    if not fold_info.empty:
        st.dataframe(fold_info, hide_index=True, use_container_width=True)
    st.write(f"Training optimizer: {config.get('optimizer')} · Epochs: {config.get('epochs')} · Ansatz: {config.get('ansatz')}")
    st.markdown("<div class='section-title'>Circuit structure</div>", unsafe_allow_html=True)
    try:
        import pennylane as qml

        from src.quantum import build_quantum_circuit

        qubits = int(config["qubits"])
        layers = int(config["layers"])
        diagram = qml.draw(build_quantum_circuit(qubits, layers))( [0.1] * qubits, [[[0.1, 0.2, 0.3] for _ in range(qubits)] for _ in range(layers)] )
        st.code(diagram)
    except Exception as error:
        st.caption(f"Circuit diagram unavailable for this environment: {type(error).__name__}: {error}")
    st.markdown("<div class='section-title'>Why this is not quantum advantage</div>", unsafe_allow_html=True)
    st.write("The experiment uses a classical computer to simulate a small quantum circuit and compares it with classical baselines on a small public dataset. It does not show a computational speedup, generalization to new populations, or clinical benefit. Results can indicate benefit, no measurable benefit, or insufficient evidence under these tested conditions only.")
    st.caption(config.get("score_note", "Quantum output is not medically calibrated."))


def render_explainability() -> None:
    page_header("Model behavior · not causation", "What the fitted models use.", "Held-out permutation importance measures how model performance changes when one feature is disrupted. It is descriptive, dataset-dependent, and not evidence of biological cause.")
    report = load_report()
    importance = report.get("explainability", {}).get("feature_importance_mean", []) if report else []
    if not importance:
        st.warning("Run the real-data benchmark to calculate held-out feature importance.")
        return
    frame = pd.DataFrame(importance)
    model = st.selectbox("Classical model", frame["model"].unique().tolist())
    selected = frame[frame["model"] == model].head(12).sort_values("importance_mean")
    figure = px.bar(selected, x="importance_mean", y="feature", orientation="h", labels={"importance_mean": "Mean held-out ROC-AUC change", "feature": "UCI engineered feature"}, color_discrete_sequence=["#4168bc"])
    figure.update_layout(height=460, margin=dict(l=10, r=15, t=15, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True)
    st.markdown("<div class='section-title'>Quantum model explanation</div>", unsafe_allow_html=True)
    st.write("The quantum model uses fold-fitted PCA components, angle encoding, entangling variational layers, and a Pauli-Z expectation readout. Components are combinations of original measurements; the circuit configuration and measured comparison are the available explanation, not a causal per-person attribution.")


def render_methodology() -> None:
    page_header("Research protocol · modular architecture", "Method before headline metrics.", "SustainX.Q is designed as a modular platform for studying multiple biomedical signals. The current implementation focuses on Parkinson's voice screening; every additional disease module remains future research until suitable data and validation exist.")
    st.markdown("""
    **Dataset.** UCI Parkinsons engineered biomedical voice measurements, CC BY 4.0, DOI [10.24432/C59C74](https://doi.org/10.24432/C59C74). The UCI description reports 195 recordings from 31 people; the repository validates the actual downloaded file and records its checksum.

    **Splitting.** Stratified group cross-validation uses the parsed person identifier. Every recording from one person remains in one fold. Imputation, standardization, and PCA for the quantum bottleneck are fit only on each training fold. Metrics are calculated after averaging out-of-fold recording scores per person.

    **Classical models.** Logistic Regression, RBF SVM, and Random Forest use the same folds. The report includes accuracy, sensitivity, specificity, precision, F1, ROC-AUC, training time, inference time, and fold variation.

    **Quantum model.** A PennyLane variational circuit runs on the local `default.qubit` simulator. Classical fold-local preprocessing reduces inputs to a compact PCA representation; AngleEmbedding and shallow entangling layers produce a Pauli-Z expectation. Rescaled output is a score, not a calibrated probability.

    **Quantum utility.** The comparison uses paired subject-level out-of-fold ROC-AUC differences against the RBF SVM, a pre-set 0.05 practical margin, and a class-stratified subject bootstrap. This is exploratory, and the interval does not capture all training or external-validation uncertainty.

    **Input boundary.** UCI provides engineered features, not the raw recordings used to create them. Microphone recordings therefore receive only signal quality checks and personal-baseline measurements; no disease prediction is generated from them.
    """)
    st.markdown("<div class='section-title'>Research roadmap</div>", unsafe_allow_html=True)
    phase_columns = st.columns(3)
    with phase_columns[0]:
        st.markdown("""
        <div class="surface"><div class="eyebrow">Phase 1 · Current MVP</div>
        <p>✓ Parkinson's voice screening research pipeline</p>
        <p>✓ Classical ML benchmarks</p><p>✓ Hybrid quantum-classical benchmark</p>
        <p>✓ Explainability</p><p>✓ Personal baseline</p><p>✓ Recording quality assessment</p></div>
        """, unsafe_allow_html=True)
    with phase_columns[1]:
        extensions = "".join(f"<p>○ {name}</p>" for name in FUTURE_MODULES)
        st.markdown(f'<div class="surface"><div class="eyebrow">Phase 2 · Neurological extensions</div>{extensions}<p class="small-note">Research extensions only; no validated predictions.</p></div>', unsafe_allow_html=True)
    with phase_columns[2]:
        st.markdown("""
        <div class="surface"><div class="eyebrow">Phase 3 · Multimodal research</div>
        <p>○ Voice</p><p>○ Handwriting</p><p>○ Typing rhythm</p>
        <p>○ Gait</p><p>○ Multimodal feature fusion</p></div>
        """, unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Future Updates</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="surface"><div class="eyebrow">Future Update</div><h3>{SKIN_FUTURE["title"]}</h3><p>{SKIN_FUTURE["description"]}</p><span class="tag">{SKIN_FUTURE["status"]}</span></div>', unsafe_allow_html=True)
    report = load_report()
    if report:
        with st.expander("Validated dataset metadata"):
            st.json(report.get("dataset", {}))
        with st.expander("Evaluation settings"):
            st.json(report.get("evaluation", {}))


def render_about() -> None:
    page_header("Project scope · SIH 2026 PS 26139", "SustainX.Q is built for modular research.", "A modular platform exploring hybrid classical and quantum machine learning for early health and neurological screening signals.")
    st.markdown("""
    The current implementation focuses on Parkinson's voice screening with a real UCI-based research benchmark. The architecture is intended to support additional biomedical signals and diseases in future research iterations.

    Alzheimer's/MCI, depression, stroke aftermath and recovery-related signals, ALS, essential tremor, and fall risk are **research extensions only**. These modules do not currently provide validated predictions, risk scores, or clinical screening.

    Microphone input is used for basic signal-quality checks and an optional personal acoustic baseline. It is not sent to an external AI service or written to disk by this app. The UCI-trained model cannot score microphone input because compatible feature extraction has not been demonstrated.

    **SustainX.Q is a research prototype and screening-support platform, not a medical diagnostic system.** Parkinson's results are exploratory experiments using public research data and are not clinical validation.
    """)
    st.markdown("<div class='section-title'>Future Updates</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="surface"><div class="eyebrow">Future Update</div><h3>{SKIN_FUTURE["title"]}</h3><p>{SKIN_FUTURE["description"]}</p><span class="tag">{SKIN_FUTURE["status"]}</span></div>', unsafe_allow_html=True)
    st.info("If you have health concerns, consult a qualified healthcare professional. Do not use this prototype to delay care.")


def main() -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">◉</div><div><div class="brand-name">SustainX.Q</div><div class="brand-sub">VOICE RESEARCH PLATFORM</div></div></div>', unsafe_allow_html=True)
        pages = ["Home", "Research modules", "Voice screening", "Screening result", "Personal baseline", "Model benchmark", "Quantum Lab", "Explainability", "Research / methodology", "About / disclaimer"]
        current = st.session_state.get("active_page", "Home")
        if current not in pages:
            current = "Home"
        selected = st.radio("Navigate", pages, index=pages.index(current), label_visibility="collapsed")
        st.session_state["active_page"] = selected
        st.markdown("---")
        st.caption("PARKINSON VOICE · RESEARCH BUILD")
        st.caption("Not clinically validated")

    page = st.session_state["active_page"]
    renderers = {
        "Home": render_home,
        "Research modules": render_modules,
        "Voice screening": render_voice,
        "Screening result": render_result,
        "Personal baseline": render_baseline,
        "Model benchmark": render_benchmark,
        "Quantum Lab": render_quantum_lab,
        "Explainability": render_explainability,
        "Research / methodology": render_methodology,
        "About / disclaimer": render_about,
    }
    renderers[page]()
    st.markdown('<div class="app-footer">SustainX.Q · Research prototype · Not clinically validated</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()