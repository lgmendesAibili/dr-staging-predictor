"""
Diabetic Retinopathy Baseline Staging Predictor — Streamlit Web Application

Interactive demonstration of the machine learning model described in:

    ARVO 2026 submission — Mendes L et al.
    "Baseline Staging of Diabetic Retinopathy (ETDRS 35 vs 43-47) from
    Static, Dynamic, and Systemic Features"

Users enter 5 features (static OCT retinal thickness, OCTA vascular ratios,
longitudinal microaneurysm turnover, and diabetes duration) and receive a
binary classification between:

    - ETDRS 35              — mild/moderate NPDR
    - ETDRS 43-47           — moderate-to-severe NPDR

Each prediction is explained with SHAP waterfall and decision plots.

SHAP explanations use a synthetic background derived from the fitted
StandardScaler, requiring no patient data at runtime.

Dependencies:
    streamlit, numpy, pandas, joblib, scikit-learn, shap, matplotlib

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# High DPI for crisp rendering on retina/HiDPI displays
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

# Resolve paths relative to this file
APP_DIR = Path(__file__).parent

# Page configuration
st.set_page_config(
    page_title="DR Baseline Staging Predictor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clinical-grade CSS
st.markdown('''
<style>
    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Header banner */
    .clinical-header {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .clinical-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .clinical-header p {
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* Research notice */
    .research-notice {
        background: #eaf2f8;
        border: 1px solid #aed6f1;
        border-radius: 0.5rem;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        font-size: 0.85rem;
        color: #1a5276;
        text-align: center;
    }
    .research-notice b {
        color: #c0392b;
    }

    /* Input card */
    .input-card {
        background: #ffffff;
        border: 1px solid #d5dbdb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .input-card h3 {
        color: #1a5276;
        margin-top: 0;
        font-size: 1.0rem;
        border-bottom: 2px solid #2980b9;
        padding-bottom: 0.5rem;
        min-height: 3.2rem;
    }

    /* Result cards */
    .result-card {
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .result-progression {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }
    .result-stable {
        background: linear-gradient(135deg, #27ae60 0%, #1e8449 100%);
        color: white;
    }
    .result-card h2 {
        margin: 0 0 0.3rem 0;
        font-size: 1.5rem;
    }
    .result-card .label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Feature reference table */
    .ref-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .ref-table th {
        background: #1a5276;
        color: white;
        padding: 0.5rem 0.75rem;
        text-align: left;
        font-weight: 500;
    }
    .ref-table td {
        padding: 0.4rem 0.75rem;
        border-bottom: 1px solid #ecf0f1;
    }
    .ref-table tr:nth-child(even) {
        background: #f8f9fa;
    }

    /* Validation badge */
    .validation-ok {
        color: #27ae60;
        font-weight: 500;
    }
    .validation-warn {
        color: #e74c3c;
        font-weight: 500;
    }

    /* Section divider */
    .section-label {
        color: #1a5276;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2980b9;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fef9e7;
        border-left: 4px solid #f39c12;
        padding: 0.75rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        font-size: 0.8rem;
        color: #7d6608;
        margin-top: 1rem;
    }

    /* Citation box */
    .citation-box {
        background: #f4f6f9;
        border: 1px solid #d5dbdb;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 0.78rem;
        color: #2c3e50;
        margin-top: 1rem;
        line-height: 1.5;
    }

    /* SHAP explanation text */
    .shap-guide {
        background: #f0f4f8;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #34495e;
        margin-bottom: 1rem;
        border-left: 3px solid #2980b9;
    }

    /* Model section containers */
    .model-section {
        border-radius: 0.75rem;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
    }
    .model-section-primary {
        background: #f4f6f8;
        border: none;
        border-radius: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .model-section-header {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin: 0;
        padding-bottom: 0;
        color: #1a5276;
    }

    /* Glossary tooltip styling */
    .glossary-note {
        background: #fbfbfb;
        border: 1px dashed #d5dbdb;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #566573;
        margin: 1rem 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f8f9fa;
    }
</style>
''', unsafe_allow_html=True)


# --- Model configurations ---
#
# A dict-of-dicts makes it trivial to add additional scenarios later
# (e.g. 12mo StatDynSys, London progression). Each entry is self-contained:
# artifacts, feature list, training-set size, and the class labels used in
# the UI.

MODEL_CONFIGS = {
    "arvo_6mo": {
        "label": "ARVO 2026 — 6-month Static+Dynamic+Systemic",
        "short_label": "Baseline DR Staging",
        "window": "baseline severity",
        "model_file": "logistic_model_ARVO2026_6mo_StatDynSys.pkl",
        "scaler_file": "scaler_ARVO2026_6mo_StatDynSys.pkl",
        "boundaries_file": "boundaries_ARVO2026_6mo_StatDynSys.pkl",
        # Feature names must match the columns used when the model was exported
        "feature_names": [
            "MA_turnover",
            "RT_OutR_Temp",
            "RT_OutR_Inf",
            "[VD/PD]_InR_SCP_SS6",
            "DiabetesYears",
        ],
        "patients_total": 161,
        "patients_pos": 85,   # ETDRS 43-47
        "patients_neg": 76,   # ETDRS 35
        "patients_smote": 170,
        "auc_cv": "0.84 ± 0.06",
        "bacc_cv": "0.76 ± 0.06",
        "css_class": "model-section-primary",
        # Class name mapping: 1 = severe (positive class), 0 = mild/moderate
        "class_positive": "ETDRS 43-47 (moderate-to-severe NPDR)",
        "class_negative": "ETDRS 35 (mild/moderate NPDR)",
    },
}


@st.cache_resource
def load_model_and_data(model_key):
    """Load a trained model, scaler, and feature boundaries by config key."""
    config = MODEL_CONFIGS[model_key]
    try:
        model = joblib.load(APP_DIR / config["model_file"])
        scaler = joblib.load(APP_DIR / config["scaler_file"])
        boundaries = joblib.load(APP_DIR / config["boundaries_file"])
        return model, scaler, boundaries
    except FileNotFoundError as e:
        st.error(f"Error loading {config['label']} files: {e}")
        st.stop()


def validate_input(value, feature_name, boundaries):
    """Check whether a feature value falls within training data boundaries.

    Returns (is_valid, error_message). Message is empty when valid.
    """
    min_val = boundaries[feature_name]['min']
    max_val = boundaries[feature_name]['max']

    if value < min_val or value > max_val:
        return False, (
            f"{feature_name} is outside training range "
            f"[{min_val:.3f}, {max_val:.3f}]"
        )
    return True, ""


@st.cache_resource
def get_shap_explainer(_model, _scaler):
    """Create and cache a SHAP LinearExplainer using a synthetic background.

    Using a zero-vector background in scaled space is equivalent to using the
    training set mean: the StandardScaler has already centred the data, so
    the zero vector *is* the mean. This avoids shipping patient data.
    """
    background = np.zeros((1, len(_scaler.mean_)))
    return shap.LinearExplainer(_model, background)


def display_shap_guide():
    """Render the SHAP explanation guide text (call once before plots)."""
    st.markdown('''
    <div class="shap-guide">
        <b>How to read these plots:</b><br>
        <b>Waterfall plot:</b> Shows each feature's individual contribution to the
        prediction. Bars extending to the <b style="color:#c0392b">right (red)</b>
        push the output toward the more severe stage (ETDRS 43-47); bars to the
        <b style="color:#2980b9">left (blue)</b> push toward the milder stage
        (ETDRS 35). The bar length indicates how strongly that feature influences
        this individual case. E[f(x)] is the base value (average model output
        across the training population).<br>
        <b>Decision plot:</b> Traces the cumulative path from the base value
        (vertical grey line) to the final model output. Each line segment shows
        the shift caused by one feature, applied sequentially from bottom to top.
        A final value to the right of the base value indicates a prediction
        toward ETDRS 43-47; to the left, toward ETDRS 35.<br>
        <b>Note:</b> All values in these plots are expressed in <b>log-odds</b>
        (the model's internal scoring scale), not probabilities. Positive
        log-odds favour ETDRS 43-47; negative log-odds favour ETDRS 35. The
        magnitude indicates how strongly each feature influences the prediction.
    </div>
    ''', unsafe_allow_html=True)


def display_shap_plots(model, scaler, input_data, input_raw, feature_names,
                       display_names):
    """Render SHAP waterfall plot and decision plot for a prediction.

    *display_names* are human-readable labels shown on the plots; *feature_names*
    are the raw column names used internally by the model/scaler.
    """
    explainer = get_shap_explainer(model, scaler)
    shap_values = explainer.shap_values(input_data)

    col_waterfall, col_decision = st.columns(2)

    with col_waterfall:
        st.markdown('<p class="section-label">Waterfall Plot</p>',
                    unsafe_allow_html=True)
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_raw[0],
            feature_names=display_names,
        )
        fig_wf, ax_wf = plt.subplots(figsize=(10, 4), dpi=200)
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig_wf, bbox_inches='tight', dpi=200)
        plt.close(fig_wf)

    with col_decision:
        st.markdown('<p class="section-label">Decision Plot</p>',
                    unsafe_allow_html=True)
        fig_dp, ax_dp = plt.subplots(figsize=(10, 4), dpi=200)
        shap.decision_plot(
            explainer.expected_value,
            shap_values,
            feature_names=display_names,
            show=False,
        )
        st.pyplot(fig_dp, bbox_inches='tight', dpi=200)
        plt.close(fig_dp)


# Human-readable feature descriptions (shown in inputs, plots, and sidebar)
FEATURE_LABELS = {
    "MA_turnover": "MA Turnover (6mo)",
    "RT_OutR_Temp": "RT Outer Ring Temporal (µm)",
    "RT_OutR_Inf": "RT Outer Ring Inferior (µm)",
    "[VD/PD]_InR_SCP_SS6": "VD/PD Inner Ring SCP (ratio)",
    "DiabetesYears": "Diabetes Duration (years)",
}

# Per-feature help text (tooltip beside the number_input)
FEATURE_DESCRIPTIONS = {
    "MA_turnover":
        "Microaneurysm turnover: longitudinal rate of appearance and "
        "disappearance of microaneurysms over a 6-month window. Higher "
        "values indicate greater retinal vascular instability.",
    "RT_OutR_Temp":
        "Retinal thickness, outer ring, temporal quadrant (ETDRS grid). "
        "Measured from OCT at baseline.",
    "RT_OutR_Inf":
        "Retinal thickness, outer ring, inferior quadrant (ETDRS grid). "
        "Measured from OCT at baseline.",
    "[VD/PD]_InR_SCP_SS6":
        "Ratio of Vessel Density to Perfusion Density in the inner ring "
        "of the superficial capillary plexus (SS 6×6 mm OCTA scan).",
    "DiabetesYears":
        "Duration of diabetes mellitus in years since diagnosis.",
}

# Which features are constrained to integer inputs.
# Integer features in the training data: RT_OutR_Temp, RT_OutR_Inf, DiabetesYears.
INTEGER_FEATURES = {"RT_OutR_Temp", "RT_OutR_Inf", "DiabetesYears"}


def main():
    """Main Streamlit application entry point."""

    # Header
    st.markdown('''
    <div class="clinical-header">
        <h1>👁️ Diabetic Retinopathy Baseline Staging</h1>
        <p>ETDRS 35 (mild/moderate NPDR) vs. ETDRS 43-47 (moderate-to-severe NPDR)
        &mdash; Logistic Regression with Static, Dynamic, and Systemic Features</p>
    </div>
    ''', unsafe_allow_html=True)

    # Research notice
    st.markdown('''
    <div class="research-notice">
        <b>Not for clinical use.</b> This application is a research demonstration
        accompanying the ARVO 2026 submission by Mendes L et al.
        (<i>AIBILI / University of Coimbra</i>), illustrating the behaviour of
        a 5-feature explainable logistic regression model trained on the
        CHART clinical dataset. It is not a validated medical device.
    </div>
    ''', unsafe_allow_html=True)

    # Load the (currently single) model
    model_key = "arvo_6mo"
    config = MODEL_CONFIGS[model_key]
    model, scaler, bounds = load_model_and_data(model_key)
    feature_names = config["feature_names"]
    display_names = [FEATURE_LABELS[f] for f in feature_names]

    # --- Input Section ---
    st.markdown('<p class="section-label">Patient Features</p>',
                unsafe_allow_html=True)

    st.markdown('''
    <div class="glossary-note">
        <b>Abbreviations.</b>
        <b>DR</b> Diabetic Retinopathy.
        <b>ETDRS</b> Early Treatment Diabetic Retinopathy Study severity scale
        (35 = mild/moderate NPDR; 43 / 47 = moderate-to-severe NPDR).
        <b>NPDR</b> Non-Proliferative DR.
        <b>OCT</b> Optical Coherence Tomography.
        <b>OCTA</b> OCT Angiography.
        <b>RT</b> Retinal Thickness (µm).
        <b>MA</b> Microaneurysm.
        <b>SCP</b> Superficial Capillary Plexus.
        <b>VD/PD</b> Vessel Density / Perfusion Density.
        <b>SS 6×6</b> 6×6 mm single-scan OCTA acquisition.
    </div>
    ''', unsafe_allow_html=True)

    input_cols = st.columns(len(feature_names))
    inputs = {}

    for i, feature in enumerate(feature_names):
        b = bounds[feature]
        label = FEATURE_LABELS[feature]
        help_text = (
            f"{FEATURE_DESCRIPTIONS[feature]} "
            f"Range: [{b['min']:.3f} — {b['max']:.3f}] | "
            f"Mean: {b['mean']:.3f} | SD: {b['std']:.3f}"
        )

        with input_cols[i]:
            st.markdown(
                f'<div class="input-card"><h3>{label}</h3></div>',
                unsafe_allow_html=True,
            )
            if feature in INTEGER_FEATURES:
                value = st.number_input(
                    f"{feature}",
                    min_value=int(b['min']),
                    max_value=int(b['max']),
                    value=round(b['mean']),
                    step=1,
                    key=feature,
                    label_visibility="collapsed",
                    help=help_text,
                )
            else:
                # Step based on 1/10 of SD, floored at something sensible
                step = max(float(b['std'] / 10), 0.01)
                value = st.number_input(
                    f"{feature}",
                    min_value=float(max(0, b['min'])),
                    max_value=float(b['max']),
                    value=float(b['mean']),
                    step=step,
                    key=feature,
                    label_visibility="collapsed",
                    help=help_text,
                )
            inputs[feature] = value

            is_valid, _ = validate_input(value, feature, bounds)
            if not is_valid:
                st.markdown(
                    '<span class="validation-warn">Outside training range</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="validation-ok">Within range</span>',
                    unsafe_allow_html=True,
                )

    # --- Predict ---
    st.markdown("")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_clicked = st.button(
            "Estimate Baseline DR Severity",
            type="primary",
            use_container_width=True,
        )

    if predict_clicked:
        # Guard against missing or NaN inputs
        invalid_inputs = [
            f for f in feature_names
            if inputs[f] is None or np.isnan(float(inputs[f]))
        ]
        if invalid_inputs:
            st.error(
                "**Missing or invalid values for: "
                f"{', '.join(invalid_inputs)}**"
            )
            st.stop()

        # Validate against training range
        validation_errors = []
        for feat in feature_names:
            is_valid, msg = validate_input(inputs[feat], feat, bounds)
            if not is_valid:
                validation_errors.append(msg)

        if validation_errors:
            st.error("**Input Validation Errors**")
            for error in validation_errors:
                st.warning(error)
        else:
            try:
                input_raw = np.array([[inputs[f] for f in feature_names]])
                input_scaled = scaler.transform(input_raw)
                prediction = model.predict(input_scaled)[0]
                score = model.decision_function(input_scaled)[0]

                # Styled model section container
                st.markdown(
                    f'<div class="model-section {config["css_class"]}">'
                    f'<p class="model-section-header">{config["short_label"]}'
                    f' &mdash; Features: {", ".join(display_names)}</p></div>',
                    unsafe_allow_html=True,
                )

                # Result card with decision score
                res_col1, res_col2, res_col3 = st.columns([1, 3, 1])
                with res_col2:
                    score_text = (
                        f'<div style="font-size:0.82rem; margin-top:0.4rem; '
                        f'opacity:0.85;">Decision score: {score:+.2f} '
                        f'(log-odds)</div>'
                    )
                    if prediction == 1:
                        st.markdown(f'''
                        <div class="result-card result-progression">
                            <h2>MODERATE-TO-SEVERE NPDR</h2>
                            <div class="label">Model classifies this eye as
                            <b>{config["class_positive"]}</b></div>
                            {score_text}
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="result-card result-stable">
                            <h2>MILD / MODERATE NPDR</h2>
                            <div class="label">Model classifies this eye as
                            <b>{config["class_negative"]}</b></div>
                            {score_text}
                        </div>
                        ''', unsafe_allow_html=True)

                # SHAP analysis
                st.markdown("")
                st.markdown(
                    '<p class="section-label">Model Explainability '
                    '(SHAP Analysis)</p>',
                    unsafe_allow_html=True,
                )
                display_shap_guide()
                st.caption(
                    "The **decision score** shown in the result card is the "
                    "model's raw log-odds output. It is not a probability. "
                    "Higher absolute values indicate stronger model confidence; "
                    "values near zero indicate borderline cases. Use it only as "
                    "a relative indication of prediction certainty."
                )

                display_shap_plots(
                    model, scaler, input_scaled, input_raw,
                    feature_names, display_names,
                )

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"### {config['short_label']} Model")
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | Algorithm | Logistic Regression |
        | Features | {len(feature_names)} (FFS-selected) |
        | Training set | {config['patients_total']} eyes |
        | Class balance | {config['patients_pos']} (43-47) / {config['patients_neg']} (35) |
        | Balancing | SMOTE (k=5) → {config['patients_smote']} eyes |
        | Preprocessing | StandardScaler |
        | CV AUC (50-fold) | {config['auc_cv']} |
        | CV Balanced Acc. | {config['bacc_cv']} |
        | Task | Baseline ETDRS staging |
        """)

        st.markdown(f"**Feature Reference Ranges**")
        table_rows = ""
        for feature in feature_names:
            b = bounds[feature]
            table_rows += (
                f"<tr><td><b>{FEATURE_LABELS[feature]}</b></td>"
                f"<td>{b['min']:.3f}</td><td>{b['max']:.3f}</td>"
                f"<td>{b['mean']:.3f}</td><td>{b['std']:.3f}</td></tr>"
            )

        st.markdown(f'''
        <table class="ref-table">
            <thead><tr><th>Feature</th><th>Min</th><th>Max</th>
            <th>Mean</th><th>SD</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
        ''', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div class="citation-box">
            <b>Citation</b><br>
            Mendes L et al. <i>"Baseline Staging of Diabetic Retinopathy
            (ETDRS 35 vs 43-47) from Static, Dynamic, and Systemic
            Features."</i> ARVO 2026 submission (AIBILI / University of
            Coimbra).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            <b>Research Demonstration Only</b><br>
            This application is provided for research and educational
            purposes to demonstrate the algorithm described in the
            accompanying submission. It is <b>not</b> a certified or
            validated medical device and must <b>not</b> be used for
            clinical decision-making. Predictions are illustrative and
            should not replace professional ophthalmological assessment.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
