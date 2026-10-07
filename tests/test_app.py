from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_all_streamlit_pages_render_without_exceptions() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert len(app.exception) == 0
    for page in [
        "Voice screening",
        "Screening result",
        "Personal baseline",
        "Model benchmark",
        "Quantum Lab",
        "Explainability",
        "Research / methodology",
        "About / disclaimer",
    ]:
        app.radio[0].set_value(page).run()
        assert len(app.exception) == 0, f"Page failed to render: {page}"


def test_branding_and_home_roadmap_are_visible() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()
    rendered = "\n".join(item.value for item in app.markdown)
    source = Path("app.py").read_text(encoding="utf-8")
    assert 'page_title="SustainX.Q | Voice Research"' in source
    assert "SustainX.Q" in rendered
    assert "SustainX</div>" not in rendered
    assert "Parkinson's Disease" in rendered
    for module in ["Alzheimer's / MCI", "Depression", "Stroke", "ALS", "Essential Tremor", "Fall Risk"]:
        assert module in rendered


def test_future_module_selector_never_exposes_prediction_widgets() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.radio[0].set_value("Research modules").run()
    options = app.selectbox[0].options
    assert options == [
        "Parkinson's Disease",
        "Alzheimer's / MCI",
        "Depression",
        "Stroke",
        "ALS",
        "Essential Tremor",
        "Fall Risk",
    ]
    for module in options[1:]:
        app.selectbox[0].set_value(module).run()
        assert len(app.exception) == 0
        assert len(app.metric) == 0, f"Unsupported module exposed metrics: {module}"
        assert len(app.number_input) == 0, f"Unsupported module exposed numeric inputs: {module}"
        rendered = "\n".join(item.value for item in app.markdown)
        assert "Research Extension" in rendered
        assert "Experimental / Future Module" in rendered
        assert "model not currently validated" in rendered
        assert "does not provide a screening, prediction, confidence, or risk score" in rendered


def test_skin_module_is_future_update_only() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.radio[0].set_value("Research modules").run()
    options = app.selectbox[0].options
    assert all("Skin" not in option for option in options)
    rendered = "\n".join(item.value for item in app.markdown)
    future_heading = rendered.rfind("Future Updates")
    skin_card = rendered.find("Skin Lesion Screening — Future Update")
    assert future_heading >= 0
    assert skin_card > future_heading
    assert "Not currently implemented." in rendered
    assert len(app.metric) == 0