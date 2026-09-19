from __future__ import annotations

import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    px = None
    PLOTLY_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "final_pipeline.pkl"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"
METADATA_PATH = ARTIFACTS_DIR / "feature_metadata.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"

DATASET_PATH_1 = ARTIFACTS_DIR / "heart_disease_clean.csv"
DATASET_PATH_2 = BASE_DIR / "heart_disease_clean.csv"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heartcare-ai")


# ============================================================
# IMPORTANT:
# THE SAVED MODEL EXPECTS THIS FUNCTION UNDER __main__
# ============================================================

def add_engineered_features(data):
    out = data.copy()

    out["age_group"] = pd.cut(
        out["age"],
        bins=[0, 40, 50, 60, 70, 100],
        labels=False,
        include_lowest=True
    )

    out["thalach_age_ratio"] = (
        out["thalach"] /
        out["age"].replace(0, np.nan)
    )

    out["chol_age_ratio"] = (
        out["chol"] /
        out["age"].replace(0, np.nan)
    )

    out["bp_age_ratio"] = (
        out["trestbps"] /
        out["age"].replace(0, np.nan)
    )

    out["oldpeak_thalach_ratio"] = (
        out["oldpeak"] /
        out["thalach"].replace(0, np.nan)
    )

    out["exercise_stress_index"] = (
        out["oldpeak"] * (1 + out["exang"])
    )

    out["cardio_burden"] = (
        out["ca"] +
        out["exang"] +
        out["oldpeak"]
    )

    return out.replace([np.inf, -np.inf], np.nan)


# ============================================================
# DATA
# ============================================================

RAW_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

ENGINEERED_FEATURES = [
    "age_group",
    "thalach_age_ratio",
    "chol_age_ratio",
    "bp_age_ratio",
    "oldpeak_thalach_ratio",
    "exercise_stress_index",
    "cardio_burden",
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    try:
        model = joblib.load(MODEL_PATH)
        return model

    except Exception as error:
        logger.exception("Model loading failed")
        raise RuntimeError(
            "The trained model could not be loaded."
        ) from error


# ============================================================
# LOAD JSON
# ============================================================

@st.cache_data
def load_json(path):

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as error:
        logger.exception("JSON loading failed")
        return {}


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if DATASET_PATH_1.exists():
        try:
            return pd.read_csv(DATASET_PATH_1)
        except Exception:
            pass

    if DATASET_PATH_2.exists():
        try:
            return pd.read_csv(DATASET_PATH_2)
        except Exception:
            pass

    return None


# ============================================================
# LOAD ARTIFACTS
# ============================================================

try:
    MODEL = load_model()
    MODEL_AVAILABLE = True
    MODEL_ERROR = None

except Exception as error:
    MODEL = None
    MODEL_AVAILABLE = False
    MODEL_ERROR = str(error)


METRICS = load_json(METRICS_PATH)
METADATA = load_json(METADATA_PATH)
DATASET = load_dataset()


# ============================================================
# EVALUATION / MONITORING HELPERS
# ============================================================

@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return None
    try:
        return pd.read_csv(PREDICTIONS_PATH)
    except Exception as error:
        logger.exception("Prediction artifact loading failed")
        return None


PREDICTIONS = load_predictions()


def evaluation_arrays():
    if PREDICTIONS is None:
        return None, None
    required = {"actual", "probability"}
    if not required.issubset(PREDICTIONS.columns):
        return None, None
    y_true = pd.to_numeric(PREDICTIONS["actual"], errors="coerce")
    y_prob = pd.to_numeric(PREDICTIONS["probability"], errors="coerce")
    mask = y_true.notna() & y_prob.notna()
    return y_true[mask].astype(int).to_numpy(), y_prob[mask].astype(float).to_numpy()


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "positive": int(y_pred.sum()),
        "negative": int((y_pred == 0).sum()),
    }


def render_plotly(fig):
    if PLOTLY_AVAILABLE:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8f9fa"),
            margin=dict(l=35, r=25, t=55, b=35),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Interactive charts require Plotly. The numerical analysis is still available below.")


def psi_score(reference, current, bins=10):
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy()
    if len(ref) < 5 or len(cur) < 5:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    c = np.histogram(cur, bins=edges)[0] / len(cur)
    r = np.clip(r, 1e-6, None)
    c = np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def local_sensitivity(patient_df):
    if MODEL is None:
        return pd.DataFrame()
    try:
        base = float(MODEL.predict_proba(patient_df)[0, 1])
        rows = []
        reference = DATASET if DATASET is not None else PREDICTIONS
        for feature in RAW_FEATURES:
            if feature not in patient_df.columns or reference is None or feature not in reference.columns:
                continue
            value = patient_df.iloc[0][feature]
            replacement = reference[feature].median()
            changed = patient_df.copy()
            changed.loc[changed.index[0], feature] = replacement
            altered = float(MODEL.predict_proba(changed)[0, 1])
            rows.append({
                "Feature": feature,
                "Patient Value": value,
                "Reference Value": replacement,
                "Probability Change": altered - base,
                "Absolute Impact": abs(altered - base),
            })
        return pd.DataFrame(rows).sort_values("Absolute Impact", ascending=False)
    except Exception as error:
        logger.exception("Local sensitivity failed")
        return pd.DataFrame()


def global_permutation_importance():
    if MODEL is None or PREDICTIONS is None:
        return pd.DataFrame()
    if not set(RAW_FEATURES + ["actual"]).issubset(PREDICTIONS.columns):
        return pd.DataFrame()
    try:
        X = PREDICTIONS[RAW_FEATURES].copy()
        y = PREDICTIONS["actual"].astype(int)
        result = permutation_importance(
            MODEL, X, y, scoring="roc_auc", n_repeats=8,
            random_state=RANDOM_STATE if isinstance(RANDOM_STATE, int) else 42
        )
        return pd.DataFrame({
            "Feature": RAW_FEATURES,
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }).sort_values("Importance", ascending=False)
    except Exception as error:
        logger.exception("Permutation importance failed")
        return pd.DataFrame()


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_NAME = (
    METADATA.get("model_name")
    or METRICS.get("model_name")
    or "Heart Disease ML Model"
)

THRESHOLD = float(
    METADATA.get(
        "threshold",
        METRICS.get("threshold", 0.39)
    )
)

RANDOM_STATE = METADATA.get(
    "random_state",
    METRICS.get("random_state", 42)
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ============================================================
# CSS
# ============================================================

st.html(
    """
<style>

:root {
    --bg: #050507;
    --bg2: #0d0709;
    --card: #16090e;
    --card2: #210c13;
    --red: #ef3340;
    --red2: #ff5966;
    --light-red: #ff9da5;
    --muted: #a99ca1;
    --border: rgba(239,51,64,.20);
    --green: #22c55e;
    --yellow: #eab308;
}

/* APP */

.stApp {
    background:
        radial-gradient(
            circle at 5% 0%,
            rgba(239,51,64,.16),
            transparent 28%
        ),
        radial-gradient(
            circle at 95% 5%,
            rgba(255,83,96,.10),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #050507,
            #0b0709,
            #16080e
        );
    color: white;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 30px;
    padding-bottom: 70px;
}


/* ANIMATIONS */

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(25px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes heartbeat {

    0%, 100% {
        transform: scale(1);
    }

    15% {
        transform: scale(1.18);
    }

    30% {
        transform: scale(1);
    }

    45% {
        transform: scale(1.10);
    }

    60% {
        transform: scale(1);
    }
}

@keyframes pulseGlow {

    0%, 100% {
        box-shadow:
            0 0 10px rgba(239,51,64,.15);
    }

    50% {
        box-shadow:
            0 0 35px rgba(239,51,64,.30);
    }
}


/* HERO */

.hero {
    position: relative;
    overflow: hidden;

    min-height: 270px;

    padding: 45px;

    border-radius: 28px;

    background:
        radial-gradient(
            circle at 85% 45%,
            rgba(239,51,64,.28),
            transparent 25%
        ),
        linear-gradient(
            115deg,
            #18070c,
            #3b0d17,
            #100609
        );

    border: 1px solid rgba(239,51,64,.35);

    box-shadow:
        0 25px 80px rgba(0,0,0,.50);

    animation: fadeUp .7s ease;
}

.hero-heart {
    position: absolute;

    right: 70px;
    top: 40px;

    font-size: 115px;

    animation: heartbeat 2s infinite;

    filter:
        drop-shadow(
            0 0 35px rgba(239,51,64,.65)
        );
}

.hero-title {
    position: relative;

    font-size: 50px;

    font-weight: 900;

    letter-spacing: -2px;

    margin-bottom: 10px;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #ffb7bd,
            #ef3340
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    position: relative;

    max-width: 800px;

    color: #d9ced2;

    font-size: 16px;

    line-height: 1.7;
}

.badge {
    display: inline-block;

    padding: 8px 15px;

    margin:
        15px 6px 0 0;

    border-radius: 50px;

    background:
        rgba(239,51,64,.08);

    border:
        1px solid rgba(239,51,64,.30);

    color:
        #ffb3ba;

    font-size: 11px;

    font-weight: 800;
}


/* CARDS */

.card {
    padding: 23px;

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(35,11,17,.97),
            rgba(17,7,11,.97)
        );

    border:
        1px solid rgba(239,51,64,.18);

    transition:
        .3s ease;

    animation:
        fadeUp .5s ease;
}

.card:hover {
    transform:
        translateY(-5px);

    border-color:
        rgba(239,51,64,.55);

    box-shadow:
        0 18px 55px rgba(239,51,64,.12);
}


/* METRICS */

.metric-card {
    padding: 20px;

    min-height: 105px;

    border-radius: 17px;

    background:
        rgba(27,10,15,.92);

    border:
        1px solid rgba(239,51,64,.18);

    transition:
        .25s ease;

    animation:
        fadeUp .45s ease;
}

.metric-card:hover {
    transform:
        translateY(-4px);

    border-color:
        rgba(239,51,64,.55);
}

.metric-label {
    color:
        #9d9296;

    font-size:
        11px;

    text-transform:
        uppercase;

    letter-spacing:
        1.2px;
}

.metric-value {
    color:
        #ff5966;

    font-size:
        28px;

    font-weight:
        900;

    margin-top:
        7px;
}


/* INPUTS */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {

    background:
        #170b10 !important;

    border:
        1px solid rgba(239,51,64,.20) !important;

    border-radius:
        11px !important;

    color:
        white !important;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {

    border-color:
        #ef3340 !important;

    box-shadow:
        0 0 0 1px
        rgba(239,51,64,.25)
        !important;
}


/* BUTTONS */

.stButton > button {

    min-height:
        50px;

    border-radius:
        12px;

    border:
        1px solid
        rgba(255,255,255,.08);

    color:
        white;

    font-weight:
        800;

    background:
        linear-gradient(
            90deg,
            #a91627,
            #ef3340,
            #ff5360
        );

    box-shadow:
        0 10px 30px
        rgba(239,51,64,.20);

    transition:
        .25s ease;
}

.stButton > button:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 15px 45px
        rgba(239,51,64,.40);
}


/* RESULT */

.result {

    padding:
        35px;

    border-radius:
        24px;

    text-align:
        center;

    margin:
        25px 0;

    animation:
        fadeUp .5s ease;
}

.result-positive {

    background:
        linear-gradient(
            135deg,
            rgba(100,10,22,.90),
            rgba(35,7,12,.90)
        );

    border:
        1px solid
        rgba(239,51,64,.45);
}

.result-negative {

    background:
        linear-gradient(
            135deg,
            rgba(7,70,43,.60),
            rgba(5,30,20,.70)
        );

    border:
        1px solid
        rgba(34,197,94,.30);
}

.result-uncertain {

    background:
        linear-gradient(
            135deg,
            rgba(100,60,5,.55),
            rgba(40,25,5,.65)
        );

    border:
        1px solid
        rgba(234,179,8,.35);
}

.result-icon {

    font-size:
        55px;

    animation:
        heartbeat 2s infinite;
}

.result-title {

    font-size:
        29px;

    font-weight:
        900;

    margin:
        8px 0;
}

.probability {

    color:
        #ff5966;

    font-size:
        58px;

    font-weight:
        900;

    margin:
        10px 0;
}

.progress {

    height:
        9px;

    margin-top:
        20px;

    border-radius:
        20px;

    background:
        rgba(255,255,255,.08);

    overflow:
        hidden;
}

.progress-inner {

    height:
        100%;

    border-radius:
        20px;

    background:
        linear-gradient(
            90deg,
            #ff5360,
            #ef3340,
            #ff9aa3
        );
}


/* SIDEBAR */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #10070b,
            #080508
        );

    border-right:
        1px solid
        rgba(239,51,64,.18);
}

.brand {

    text-align:
        center;

    padding:
        15px 10px 25px;
}

.brand-heart {

    font-size:
        52px;

    animation:
        heartbeat 2s infinite;

    filter:
        drop-shadow(
            0 0 18px
            rgba(239,51,64,.65)
        );
}

.brand-name {

    font-size:
        24px;

    font-weight:
        900;
}

.brand-name span {

    color:
        #ef3340;
}

.brand-sub {

    color:
        #8e8287;

    font-size:
        11px;
}


/* FOOTER */

.footer {

    margin-top:
        55px;

    padding:
        30px;

    text-align:
        center;

    color:
        #766b70;

    border-top:
        1px solid
        rgba(239,51,64,.12);
}

.footer-name {

    color:
        #ef3340;

    font-size:
        20px;

    font-weight:
        900;
}

.footer a {

    color:
        #ff5966;

    text-decoration:
        none;

    margin:
        0 8px;
}

.footer a:hover {

    color:
        #ffffff;
}


/* MOBILE */

@media (max-width: 900px) {

    .hero {
        padding:
            28px;
    }

    .hero-title {
        font-size:
            35px;
    }

    .hero-heart {
        opacity:
            .12;

        right:
            5px;
    }
}

</style>
"""
)


# ============================================================
# HELPERS
# ============================================================

def metric_card(label, value):
    st.html(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """
    )


def metric_grid(items, columns=4):

    cols = st.columns(columns)

    for col, (label, value) in zip(cols, items):

        with col:
            metric_card(label, value)


def section_title(title, subtitle=None):

    st.markdown(f"## {title}")

    if subtitle:
        st.caption(subtitle)


def probability_band(probability):

    if probability < 0.25:
        return "Low Model Probability"

    if probability < 0.50:
        return "Moderate Model Probability"

    if probability < 0.75:
        return "Elevated Model Probability"

    return "High Model Probability"


def prediction_result(probability):

    prediction = int(
        probability >= THRESHOLD
    )

    distance = abs(
        probability - THRESHOLD
    )

    if distance <= 0.10:

        return (
            prediction,
            "Prediction Near Threshold",
            "uncertain",
            "🟡",
        )

    if prediction == 1:

        return (
            prediction,
            "Higher Likelihood",
            "positive",
            "⚠️",
        )

    return (
        prediction,
        "Lower Likelihood",
        "negative",
        "💚",
    )


def validate_patient(data):

    errors = []

    ranges = {
        "age": (1, 120),
        "sex": (0, 1),
        "cp": (0, 4),
        "trestbps": (50, 250),
        "chol": (50, 700),
        "fbs": (0, 1),
        "restecg": (0, 2),
        "thalach": (50, 250),
        "exang": (0, 1),
        "oldpeak": (0, 10),
        "slope": (0, 3),
        "ca": (0, 4),
        "thal": (0, 7),
    }

    for feature in RAW_FEATURES:

        if feature not in data:
            errors.append(
                f"Missing feature: {feature}"
            )
            continue

        value = data[feature]

        if pd.isna(value):
            errors.append(
                f"{feature} cannot be empty."
            )
            continue

        if feature in ranges:

            low, high = ranges[feature]

            if value < low or value > high:

                errors.append(
                    f"{feature} must be between {low} and {high}."
                )

    return errors


def predict_patient(patient_df):

    if MODEL is None:
        raise RuntimeError(
            "Model is not available."
        )

    probabilities = MODEL.predict_proba(
        patient_df
    )

    probability = float(
        probabilities[0][1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    return probability, prediction


def create_csv_report(
    patient,
    probability,
    prediction_label,
    timestamp
):

    report = patient.copy()

    report["model"] = MODEL_NAME
    report["model_probability"] = probability
    report["decision_threshold"] = THRESHOLD
    report["prediction"] = prediction_label
    report["generated_at"] = timestamp

    return report.to_csv(
        index=False
    ).encode("utf-8")


def create_text_report(
    patient,
    probability,
    prediction_label,
    timestamp
):

    lines = []

    lines.append(
        "HEARTCARE AI"
    )

    lines.append(
        "AI Prediction Report"
    )

    lines.append(
        "=" * 60
    )

    lines.append(
        f"Generated: {timestamp}"
    )

    lines.append(
        f"Model: {MODEL_NAME}"
    )

    lines.append(
        f"Model Probability: {probability:.2%}"
    )

    lines.append(
        f"Decision Threshold: {THRESHOLD:.2%}"
    )

    lines.append(
        f"Model Prediction: {prediction_label}"
    )

    lines.append("")
    lines.append(
        "PATIENT INPUTS"
    )

    lines.append(
        "-" * 60
    )

    for column in patient.columns:

        value = patient.iloc[0][column]

        lines.append(
            f"{column}: {value}"
        )

    lines.append("")
    lines.append(
        "DISCLAIMER"
    )

    lines.append(
        "This application is for educational and research purposes."
    )

    lines.append(
        "It is not a medical diagnostic system."
    )

    lines.append(
        "It must not be used to make clinical decisions."
    )

    return "\n".join(lines).encode(
        "utf-8"
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-heart">
            ❤️
        </div>

        <div class="hero-title">
            HeartCare AI
        </div>

        <div class="hero-subtitle">

            Intelligent Heart Risk Analysis
            • Explainable Machine Learning
            • Advanced Analytics

            <br><br>

            A professional machine learning interface
            for exploring model-generated heart disease
            probabilities from clinical input features.

        </div>

        <span class="badge">
            🤖 Machine Learning
        </span>

        <span class="badge">
            📊 Predictive Analytics
        </span>

        <span class="badge">
            🧠 Explainable AI
        </span>

        <span class="badge">
            🛡 Model Monitoring
        </span>

    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="brand">

            <div class="brand-heart">
                ❤️
            </div>

            <div class="brand-name">
                HeartCare <span>AI</span>
            </div>

            <div class="brand-sub">
                Predict • Explain • Explore
            </div>

        </div>
        """
    )

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Prediction",
            "Batch Prediction",
            "Data Explorer",
            "Threshold Lab",
            "Evaluation Studio",
            "Explainability",
            "Patient Comparison",
            "Model Card",
            "Model Monitor",
            "Model Insights",
            "Model Validation",
            "Error Analysis",
            "Prediction History",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.write(
        f"**{MODEL_NAME}**"
    )

    st.divider()

    if METRICS:

        st.markdown(
            "### 📊 Performance"
        )

        if "roc_auc" in METRICS:

            st.metric(
                "ROC-AUC",
                f"{METRICS['roc_auc']:.2%}"
            )

        if "recall" in METRICS:

            st.metric(
                "Recall",
                f"{METRICS['recall']:.2%}"
            )

        if "f1" in METRICS:

            st.metric(
                "F1 Score",
                f"{METRICS['f1']:.2%}"
            )

    st.divider()

    st.caption(
        f"Threshold: {THRESHOLD:.2f}"
    )

    st.caption(
        f"Random State: {RANDOM_STATE}"
    )

    if MODEL_AVAILABLE:

        st.success(
            "● Model Ready"
        )

    else:

        st.error(
            "● Model Unavailable"
        )


# ============================================================
# HOME
# ============================================================

if page == "Home":

    section_title(
        "📊 Model Performance",
        "Evaluation metrics loaded from the existing model artifact."
    )

    if METRICS:

        metric_grid(
            [
                (
                    "Accuracy",
                    f"{METRICS.get('accuracy', 0):.2%}"
                ),
                (
                    "Precision",
                    f"{METRICS.get('precision', 0):.2%}"
                ),
                (
                    "Recall",
                    f"{METRICS.get('recall', 0):.2%}"
                ),
                (
                    "F1 Score",
                    f"{METRICS.get('f1', 0):.2%}"
                ),
                (
                    "ROC-AUC",
                    f"{METRICS.get('roc_auc', 0):.2%}"
                ),
                (
                    "PR-AUC",
                    f"{METRICS.get('pr_auc', 0):.2%}"
                ),
            ],
            columns=6
        )

    else:

        st.info(
            "Model metrics are not available in the current artifact set."
        )

    st.markdown(
        "## ⚡ Live Session"
    )

    history = (
        st.session_state.prediction_history
    )

    total_predictions = len(
        history
    )

    if total_predictions:

        probabilities = [
            item["Probability"]
            for item in history
        ]

        higher = sum(
            item["Prediction"] == "Higher Likelihood"
            for item in history
        )

        lower = (
            total_predictions -
            higher
        )

        average_probability = np.mean(
            probabilities
        )

        maximum_probability = np.max(
            probabilities
        )

        minimum_probability = np.min(
            probabilities
        )

    else:

        higher = 0
        lower = 0
        average_probability = None
        maximum_probability = None
        minimum_probability = None

    metric_grid(
        [
            (
                "Predictions",
                str(total_predictions)
            ),
            (
                "Higher Likelihood",
                str(higher)
            ),
            (
                "Lower Likelihood",
                str(lower)
            ),
            (
                "Average Probability",
                f"{average_probability:.1%}"
                if average_probability is not None
                else "—"
            ),
            (
                "Maximum Probability",
                f"{maximum_probability:.1%}"
                if maximum_probability is not None
                else "—"
            ),
            (
                "Minimum Probability",
                f"{minimum_probability:.1%}"
                if minimum_probability is not None
                else "—"
            ),
        ],
        columns=6
    )

    st.markdown(
        "## 🧠 What makes HeartCare AI different?"
    )

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (
            c1,
            "🤖",
            "Machine Learning",
            "Uses the trained pipeline already created for this project."
        ),
        (
            c2,
            "📈",
            "Model Probability",
            "Displays the model-generated probability rather than medical certainty."
        ),
        (
            c3,
            "⚙️",
            "Feature Engineering",
            "Uses the engineered features required by the trained pipeline."
        ),
        (
            c4,
            "📊",
            "Analytics",
            "Explore model metrics, dataset statistics and prediction history."
        ),
    ]

    for column, icon, title, description in cards:

        with column:

            st.html(
                f"""
                <div class="card">

                    <div style="font-size:35px;">
                        {icon}
                    </div>

                    <h3 style="color:#ff5966;">
                        {title}
                    </h3>

                    <p style="color:#aaa0a4;line-height:1.7;">
                        {description}
                    </p>

                </div>
                """
            )

    st.markdown(
        "## ⚙️ System Status"
    )

    dataset_status = (
        "Available"
        if DATASET is not None
        else "Unavailable"
    )

    metric_grid(
        [
            (
                "Model",
                "READY"
                if MODEL_AVAILABLE
                else "ERROR"
            ),
            (
                "Metadata",
                "AVAILABLE"
                if METADATA
                else "MISSING"
            ),
            (
                "Metrics",
                "AVAILABLE"
                if METRICS
                else "MISSING"
            ),
            (
                "Dataset",
                dataset_status
            ),
            (
                "Features",
                str(len(RAW_FEATURES))
            ),
            (
                "Threshold",
                f"{THRESHOLD:.2f}"
            ),
        ],
        columns=6
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    section_title(
        "❤️ Patient Prediction",
        "Enter patient features and run the trained model."
    )

    if not MODEL_AVAILABLE:

        st.error(
            "The trained model could not be loaded."
        )

        st.stop()

    demo = st.checkbox(
        "✨ Load Demo Patient"
    )

    demo_values = {

        "age": 63,
        "sex": 1,
        "cp": 2,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,

    } if demo else {}

    st.markdown(
        "### 👤 Patient Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.html(
            """
            <div class="card">
                <h3 style="color:#ff5966;">
                    👤 Demographics
                </h3>
            </div>
            """
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=demo_values.get(
                "age",
                55
            )
        )

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x:
                "Female"
                if x == 0
                else "Male",
            index=(
                1
                if demo_values.get("sex") == 1
                else 0
            )
        )

        cp_labels = {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic",
            4: "Other / Dataset Code"
        }

        cp = st.selectbox(
            "Chest Pain Type",
            list(cp_labels.keys()),
            format_func=lambda x:
                f"{x} — {cp_labels[x]}",
            index=demo_values.get(
                "cp",
                0
            )
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=demo_values.get(
                "trestbps",
                130
            )
        )

    with col2:

        st.html(
            """
            <div class="card">
                <h3 style="color:#ff5966;">
                    🫀 Cardiovascular
                </h3>
            </div>
            """
        )

        chol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=700,
            value=demo_values.get(
                "chol",
                240
            )
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0, 1],
            format_func=lambda x:
                "No"
                if x == 0
                else "Yes",
            index=demo_values.get(
                "fbs",
                0
            )
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            index=demo_values.get(
                "restecg",
                0
            )
        )

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=demo_values.get(
                "thalach",
                150
            )
        )

    with col3:

        st.html(
            """
            <div class="card">
                <h3 style="color:#ff5966;">
                    🏃 Exercise Indicators
                </h3>
            </div>
            """
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1],
            format_func=lambda x:
                "No"
                if x == 0
                else "Yes",
            index=demo_values.get(
                "exang",
                0
            )
        )

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=float(
                demo_values.get(
                    "oldpeak",
                    1.0
                )
            ),
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            [0, 1, 2, 3],
            index=demo_values.get(
                "slope",
                0
            )
        )

        ca = st.selectbox(
            "Major Vessels (CA)",
            [0, 1, 2, 3, 4],
            index=demo_values.get(
                "ca",
                0
            )
        )

        thal = st.selectbox(
            "Thal",
            [0, 1, 2, 3, 4, 5, 6, 7],
            index=demo_values.get(
                "thal",
                1
            )
        )

    patient = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }]
    )

    errors = validate_patient(
        patient.iloc[0].to_dict()
    )

    if errors:

        for error in errors:

            st.error(
                f"⚠️ {error}"
            )

    st.markdown("")

    _, center, _ = st.columns(
        [1, 2, 1]
    )

    with center:

        analyze = st.button(
            "❤️  ANALYZE PATIENT",
            use_container_width=True,
            disabled=bool(errors)
        )

    if analyze:

        try:

            with st.spinner(
                "🧠 Running machine learning inference..."
            ):

                probability, prediction = (
                    predict_patient(patient)
                )

            (
                prediction_value,
                result_title,
                result_class,
                result_icon,
            ) = prediction_result(
                probability
            )

            percentage = (
                probability * 100
            )

            distance = abs(
                probability -
                THRESHOLD
            )

            timestamp = (
                datetime.now()
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            # HISTORY

            history_record = (
                patient.copy()
            )

            history_record[
                "Probability"
            ] = probability

            history_record[
                "Prediction"
            ] = result_title

            history_record[
                "Threshold"
            ] = THRESHOLD

            history_record[
                "Timestamp"
            ] = timestamp

            st.session_state.prediction_history.append(
                {
                    "Timestamp": timestamp,
                    "Probability": probability,
                    "Prediction": result_title,
                    "Threshold": THRESHOLD,
                }
            )

            st.session_state.last_prediction = {
                "patient": patient.copy(),
                "probability": probability,
                "prediction": result_title,
                "timestamp": timestamp,
            }

            # RESULT

            st.html(
                f"""
                <div class="result result-{result_class}">

                    <div class="result-icon">
                        {result_icon}
                    </div>

                    <div class="result-title">
                        {result_title}
                    </div>

                    <div class="probability">
                        {percentage:.1f}%
                    </div>

                    <div style="
                        color:#d0c5c9;
                        line-height:1.7;
                    ">

                        Model-generated probability

                        <br>

                        The model probability is
                        {"above" if probability >= THRESHOLD else "below"}
                        the configured decision threshold.

                    </div>

                    <div class="progress">

                        <div
                            class="progress-inner"
                            style="width:{percentage}%"
                        ></div>

                    </div>

                </div>
                """
            )

            st.markdown(
                "### 📐 Prediction Analysis"
            )

            metric_grid(
                [
                    (
                        "Model Probability",
                        f"{percentage:.2f}%"
                    ),
                    (
                        "Decision Threshold",
                        f"{THRESHOLD:.2f}"
                    ),
                    (
                        "Distance",
                        f"{distance:.3f}"
                    ),
                    (
                        "Probability Zone",
                        probability_band(
                            probability
                        )
                    ),
                ],
                columns=4
            )

            left, right = st.columns(2)

            with left:

                st.html(
                    f"""
                    <div class="card"
                         style="text-align:center;">

                        <div class="metric-label">
                            MODEL PROBABILITY
                        </div>

                        <div style="
                            width:190px;
                            height:190px;
                            border-radius:50%;
                            margin:25px auto;
                            display:flex;
                            align-items:center;
                            justify-content:center;

                            background:
                            conic-gradient(
                                #ef3340
                                {percentage}%,
                                rgba(255,255,255,.08)
                                0
                            );

                            box-shadow:
                            0 0 45px
                            rgba(239,51,64,.20);
                        ">

                            <div style="
                                width:145px;
                                height:145px;
                                border-radius:50%;
                                background:#10070b;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                font-size:30px;
                                font-weight:900;
                            ">

                                {percentage:.1f}%

                            </div>

                        </div>

                        <div style="
                            color:#ff5966;
                            font-size:21px;
                            font-weight:900;
                        ">

                            {probability_band(
                                probability
                            )}

                        </div>

                        <div style="
                            margin-top:10px;
                            color:#aaa0a4;
                        ">

                            Threshold:
                            {THRESHOLD:.2f}

                        </div>

                    </div>
                    """
                )

            with right:

                st.markdown(
                    "#### ⚙️ Engineered Features"
                )

                engineered = (
                    add_engineered_features(
                        patient
                    )
                )

                engineered_display = (
                    engineered
                    .T
                    .reset_index()
                )

                engineered_display.columns = [
                    "Feature",
                    "Value"
                ]

                st.dataframe(
                    engineered_display,
                    use_container_width=True,
                    hide_index=True
                )

            # MODEL EXPLANATION

            st.markdown(
                "### 🧠 Model Explanation"
            )

            explanation_available = False

            try:

                if hasattr(
                    MODEL,
                    "named_steps"
                ):

                    steps = MODEL.named_steps

                    estimator = (
                        list(
                            steps.values()
                        )[-1]
                    )

                    if hasattr(
                        estimator,
                        "coef_"
                    ):

                        preprocessor = None

                        for name, step in steps.items():

                            if hasattr(
                                step,
                                "get_feature_names_out"
                            ):

                                preprocessor = step

                        if preprocessor is not None:

                            transformed = (
                                MODEL[:-1]
                                .transform(
                                    patient
                                )
                            )

                            feature_names = (
                                preprocessor
                                .get_feature_names_out()
                            )

                            coefficients = (
                                estimator
                                .coef_[0]
                            )

                            contributions = (
                                transformed[0]
                                * coefficients
                            )

                            explanation_df = pd.DataFrame(
                                {
                                    "Feature":
                                        feature_names,
                                    "Contribution":
                                        contributions,
                                }
                            )

                            explanation_df[
                                "Absolute Contribution"
                            ] = (
                                explanation_df[
                                    "Contribution"
                                ].abs()
                            )

                            explanation_df = (
                                explanation_df
                                .sort_values(
                                    "Absolute Contribution",
                                    ascending=False
                                )
                                .head(10)
                            )

                            explanation_df[
                                "Direction"
                            ] = np.where(
                                explanation_df[
                                    "Contribution"
                                ] >= 0,
                                "↑ Pushes probability up",
                                "↓ Pushes probability down"
                            )

                            st.dataframe(
                                explanation_df[
                                    [
                                        "Feature",
                                        "Contribution",
                                        "Direction"
                                    ]
                                ].round(4),
                                use_container_width=True,
                                hide_index=True
                            )

                            explanation_available = True

            except Exception as error:

                logger.exception(
                    "Explanation failed"
                )

            if not explanation_available:

                st.info(
                    "Detailed local explanation is not available "
                    "from the current model artifact."
                )

            # DOWNLOADS

            st.markdown(
                "### 📥 Download"
            )

            csv_report = create_csv_report(
                patient,
                probability,
                result_title,
                timestamp
            )

            text_report = create_text_report(
                patient,
                probability,
                result_title,
                timestamp
            )

            d1, d2 = st.columns(2)

            with d1:

                st.download_button(
                    "📊 Download CSV Report",
                    data=csv_report,
                    file_name=(
                        "heartcare_ai_prediction.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

            with d2:

                st.download_button(
                    "📄 Download AI Report",
                    data=text_report,
                    file_name=(
                        "heartcare_ai_report.txt"
                    ),
                    mime="text/plain",
                    use_container_width=True
                )

            st.warning(
                "This application is for educational and research "
                "purposes. It is not a medical diagnostic system "
                "and should not be used to make clinical decisions."
            )

        except Exception as error:

            logger.exception(
                "Prediction failed"
            )

            st.error(
                "❌ Prediction could not be completed. "
                "Please verify the model artifacts and input values."
            )


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    section_title(
        "📁 Batch Prediction",
        "Upload a CSV containing the 13 model input features."
    )

    template = pd.DataFrame(
        columns=RAW_FEATURES
    )

    st.download_button(
        "📥 Download CSV Template",
        data=template.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "heartcare_batch_template.csv"
        ),
        mime="text/csv"
    )

    uploaded = st.file_uploader(
        "Upload Patient CSV",
        type=["csv"]
    )

    if uploaded:

        try:

            batch = pd.read_csv(
                uploaded
            )

            missing = [
                feature
                for feature in RAW_FEATURES
                if feature not in batch.columns
            ]

            if missing:

                st.error(
                    "Missing required columns:"
                )

                st.code(
                    ", ".join(missing)
                )

            else:

                batch = batch[
                    RAW_FEATURES
                ].copy()

                st.success(
                    f"Loaded {len(batch):,} rows."
                )

                st.dataframe(
                    batch.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                if st.button(
                    "🚀 RUN BATCH PREDICTION",
                    use_container_width=True
                ):

                    try:

                        probabilities = (
                            MODEL
                            .predict_proba(
                                batch
                            )[:, 1]
                        )

                        predictions = (
                            probabilities >= THRESHOLD
                        ).astype(int)

                        results = batch.copy()

                        results[
                            "model_probability"
                        ] = probabilities

                        results[
                            "threshold"
                        ] = THRESHOLD

                        results[
                            "prediction"
                        ] = np.where(
                            predictions == 1,
                            "Higher Likelihood",
                            "Lower Likelihood"
                        )

                        metric_grid(
                            [
                                (
                                    "Total Rows",
                                    f"{len(results):,}"
                                ),
                                (
                                    "Higher Likelihood",
                                    f"{sum(predictions):,}"
                                ),
                                (
                                    "Lower Likelihood",
                                    f"{len(results) - sum(predictions):,}"
                                ),
                                (
                                    "Average Probability",
                                    f"{probabilities.mean():.1%}"
                                ),
                            ],
                            columns=4
                        )

                        st.dataframe(
                            results,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "📥 Download Batch Results",
                            data=results.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name=(
                                "heartcare_batch_predictions.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception:

                        logger.exception(
                            "Batch prediction failed"
                        )

                        st.error(
                            "Batch prediction failed."
                        )

        except Exception:

            logger.exception(
                "CSV loading failed"
            )

            st.error(
                "The uploaded CSV could not be read."
            )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    section_title(
        "🔬 Data Explorer",
        "Explore the available heart disease dataset."
    )

    if DATASET is None:

        st.warning(
            "heart_disease_clean.csv is not available."
        )

    else:

        metric_grid(
            [
                (
                    "Rows",
                    f"{len(DATASET):,}"
                ),
                (
                    "Columns",
                    str(DATASET.shape[1])
                ),
                (
                    "Missing Values",
                    f"{DATASET.isna().sum().sum():,}"
                ),
                (
                    "Duplicates",
                    f"{DATASET.duplicated().sum():,}"
                ),
            ],
            columns=4
        )

        st.markdown(
            "### 📋 Dataset Preview"
        )

        st.dataframe(
            DATASET.head(20),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "### 📊 Descriptive Statistics"
        )

        st.dataframe(
            DATASET.describe()
            .T
            .round(3),
            use_container_width=True
        )

        st.markdown(
            "### 🎯 Target Distribution"
        )

        if "target" in DATASET.columns:

            counts = (
                DATASET["target"]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(
                counts
            )

        st.markdown(
            "### 📈 Feature Distribution"
        )

        numeric_columns = (
            DATASET
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        if numeric_columns:

            selected_feature = st.selectbox(
                "Select Feature",
                numeric_columns
            )

            st.bar_chart(
                DATASET[
                    selected_feature
                ].value_counts(
                    bins=20
                ).sort_index()
            )

        st.markdown(
            "### 🔗 Correlation Matrix"
        )

        correlation = (
            DATASET
            .select_dtypes(
                include=np.number
            )
            .corr()
            .round(2)
        )

        st.dataframe(
            correlation,
            use_container_width=True
        )


# ============================================================
# THRESHOLD LAB
# ============================================================

elif page == "Threshold Lab":

    section_title(
        "🎚 Threshold Lab",
        "Explore how the configured decision threshold changes classification behavior."
    )

    y_true, y_prob = evaluation_arrays()
    if y_true is None:
        st.warning("Threshold analysis is not available in the current artifact set.")
    else:
        selected = st.slider("Explorer Threshold", 0.10, 0.90, float(THRESHOLD), 0.01)
        st.info(f"Deployed threshold: **{THRESHOLD:.2f}**  •  Explorer threshold: **{selected:.2f}**")
        tm = metrics_at_threshold(y_true, y_prob, selected)
        metric_grid([
            ("Accuracy", f"{tm['accuracy']:.2%}"),
            ("Precision", f"{tm['precision']:.2%}"),
            ("Recall", f"{tm['recall']:.2%}"),
            ("Specificity", f"{tm['specificity']:.2%}"),
            ("F1 Score", f"{tm['f1']:.2%}"),
        ], columns=5)
        metric_grid([
            ("True Positives", str(tm["tp"])),
            ("True Negatives", str(tm["tn"])),
            ("False Positives", str(tm["fp"])),
            ("False Negatives", str(tm["fn"])),
        ], columns=4)

        thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)
        sweep_rows = []
        for t in thresholds:
            m = metrics_at_threshold(y_true, y_prob, float(t))
            sweep_rows.append({"Threshold": t, "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"], "Specificity": m["specificity"], "Accuracy": m["accuracy"]})
        sweep = pd.DataFrame(sweep_rows)
        st.markdown("### 📉 Metric Behavior Across Thresholds")
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            for col in ["Precision", "Recall", "F1", "Specificity", "Accuracy"]:
                fig.add_trace(go.Scatter(x=sweep["Threshold"], y=sweep[col], mode="lines", name=col))
            fig.add_vline(x=selected, line_dash="dash", annotation_text="Explorer threshold")
            render_plotly(fig)
        else:
            st.line_chart(sweep.set_index("Threshold")[['Precision','Recall','F1','Specificity','Accuracy']])
        st.dataframe(sweep.round(4), use_container_width=True, hide_index=True)


# ============================================================
# EVALUATION STUDIO
# ============================================================

elif page == "Evaluation Studio":

    section_title(
        "📈 Evaluation Studio",
        "ROC, Precision-Recall, Confusion Matrix, and Calibration analysis from saved out-of-sample predictions."
    )

    y_true, y_prob = evaluation_arrays()
    if y_true is None:
        st.warning("Evaluation plots are not available in the current artifact set.")
    else:
        metric_grid([
            ("ROC-AUC", f"{roc_auc_score(y_true, y_prob):.2%}"),
            ("PR-AUC", f"{average_precision_score(y_true, y_prob):.2%}"),
            ("Brier Score", f"{brier_score_loss(y_true, y_prob):.4f}"),
            ("Samples", str(len(y_true))),
        ], columns=4)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ROC Curve")
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC-AUC {roc_auc_score(y_true,y_prob):.3f}"))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
                render_plotly(fig)
            else:
                st.line_chart(pd.DataFrame({"TPR": tpr}, index=fpr))
        with c2:
            st.markdown("### Precision-Recall Curve")
            if PLOTLY_AVAILABLE:
                fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", name="Precision-Recall"))
                render_plotly(fig)
            else:
                st.line_chart(pd.DataFrame({"Precision": precision}, index=recall))

        st.markdown("### 🎯 Confusion Matrix at Deployed Threshold")
        cm = confusion_matrix(y_true, (y_prob >= THRESHOLD).astype(int), labels=[0,1])
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        st.dataframe(cm_df, use_container_width=True)
        st.caption(f"Configured decision threshold = {THRESHOLD:.2f}")

        st.markdown("### 🧭 Calibration")
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=6, strategy="quantile")
        cal_df = pd.DataFrame({"Mean Model Probability": mean_pred, "Observed Positive Rate": frac_pos})
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Model"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect calibration", line=dict(dash="dash")))
            render_plotly(fig)
        else:
            st.dataframe(cal_df, use_container_width=True)
        st.dataframe(cal_df.round(4), use_container_width=True, hide_index=True)


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    section_title(
        "🧠 Explainability",
        "Model-based feature analysis plus local probability sensitivity."
    )

    st.markdown("### 🌍 Global Feature Importance")
    importance = global_permutation_importance()
    if importance.empty:
        st.info("Global permutation importance is not available in the current artifact set.")
    else:
        if PLOTLY_AVAILABLE:
            plot_df = importance.sort_values("Importance", ascending=True)
            fig = px.bar(plot_df, x="Importance", y="Feature", orientation="h", error_x="Std", title="Permutation Importance (ROC-AUC decrease)")
            render_plotly(fig)
        st.dataframe(importance.round(5), use_container_width=True, hide_index=True)
        st.caption("Higher permutation importance means shuffling that feature changed ROC-AUC more on the saved prediction set. This is model behavior, not a medical causal effect.")

    st.markdown("### 🔍 Local Sensitivity")
    if st.session_state.last_prediction is None:
        st.info("Make a prediction first, then return here to inspect the current patient's local probability sensitivity.")
    else:
        patient = st.session_state.last_prediction.get("input")
        if patient is None:
            st.info("Local sensitivity is unavailable for the current session prediction.")
        else:
            local = local_sensitivity(pd.DataFrame([patient]))
            if local.empty:
                st.info("Local sensitivity could not be calculated safely for this prediction.")
            else:
                st.dataframe(local.round(4), use_container_width=True, hide_index=True)
                st.caption("Probability Change shows how the model probability changes when one feature is replaced by its dataset median while other inputs remain fixed. It is a local sensitivity analysis, not a clinical explanation.")


# ============================================================
# MODEL MONITOR
# ============================================================

elif page == "Model Monitor":

    section_title(
        "🖥️ Model Monitor",
        "Artifact health, schema readiness, and distribution-shift signals."
    )

    health = [
        ("Model Loaded", "YES" if MODEL_AVAILABLE else "NO"),
        ("Model Artifact", "YES" if MODEL_PATH.exists() else "NO"),
        ("Metrics Artifact", "YES" if METRICS_PATH.exists() else "NO"),
        ("Metadata Artifact", "YES" if METADATA_PATH.exists() else "NO"),
        ("Predictions Artifact", "YES" if PREDICTIONS_PATH.exists() else "NO"),
        ("Dataset Available", "YES" if DATASET is not None else "NO"),
    ]
    metric_grid(health, columns=3)

    st.markdown("### ⚙️ Deployment State")
    metric_grid([
        ("Model", MODEL_NAME),
        ("Threshold", f"{THRESHOLD:.2f}"),
        ("Raw Features", str(len(RAW_FEATURES))),
        ("Engineered Features", str(len(ENGINEERED_FEATURES))),
    ], columns=4)

    if MODEL_ERROR:
        st.error("The trained model could not be loaded. Check the artifact and Python environment.")

    st.markdown("### 🌊 Distribution Shift")
    if DATASET is None or PREDICTIONS is None:
        st.info("Drift analysis requires both the reference dataset and predictions artifact.")
    else:
        rows = []
        for feature in RAW_FEATURES:
            if feature not in DATASET.columns or feature not in PREDICTIONS.columns:
                continue
            score = psi_score(DATASET[feature], PREDICTIONS[feature])
            rows.append({"Feature": feature, "PSI-style Shift": score})
        drift = pd.DataFrame(rows).sort_values("PSI-style Shift", ascending=False)
        st.dataframe(drift.round(5), use_container_width=True, hide_index=True)
        st.caption("This compares the saved prediction-input sample with the reference dataset. It is a monitoring signal, not a proof of model failure. Small samples should be interpreted cautiously.")

    st.markdown("### 📦 Artifact Summary")
    artifact_rows = []
    for name, path in [("Model", MODEL_PATH), ("Metrics", METRICS_PATH), ("Metadata", METADATA_PATH), ("Predictions", PREDICTIONS_PATH)]:
        artifact_rows.append({"Artifact": name, "Available": path.exists(), "Size (KB)": round(path.stat().st_size / 1024, 1) if path.exists() else None})
    st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Patient Comparison":

    section_title(
        "👥 Patient Comparison",
        "Compare two patient profiles using the same trained model."
    )

    if MODEL is None:
        st.error("The trained model is not available.")
    else:
        defaults = {
            "age": 55, "sex": 1, "cp": 0, "trestbps": 130,
            "chol": 240, "fbs": 0, "restecg": 0, "thalach": 150,
            "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
        }

        def comparison_form(prefix, title):
            st.markdown(f"### {title}")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                age = st.number_input("Age", 1, 120, defaults["age"], key=f"{prefix}_age")
                sex = st.selectbox("Sex", [0, 1], index=1, key=f"{prefix}_sex", format_func=lambda x: "Male" if x == 1 else "Female")
                cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], key=f"{prefix}_cp")
            with c2:
                trestbps = st.number_input("Resting BP", 50, 250, defaults["trestbps"], key=f"{prefix}_bp")
                chol = st.number_input("Cholesterol", 50, 700, defaults["chol"], key=f"{prefix}_chol")
                fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], key=f"{prefix}_fbs")
            with c3:
                restecg = st.selectbox("Resting ECG", [0, 1, 2], key=f"{prefix}_ecg")
                thalach = st.number_input("Max Heart Rate", 50, 250, defaults["thalach"], key=f"{prefix}_thalach")
                exang = st.selectbox("Exercise Angina", [0, 1], key=f"{prefix}_exang", format_func=lambda x: "Yes" if x else "No")
            with c4:
                oldpeak = st.number_input("Oldpeak", 0.0, 10.0, defaults["oldpeak"], 0.1, key=f"{prefix}_oldpeak")
                slope = st.selectbox("Slope", [0, 1, 2], key=f"{prefix}_slope")
                ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3, 4], key=f"{prefix}_ca")
                thal = st.selectbox("Thal", [0, 1, 2, 3, 6, 7], key=f"{prefix}_thal")
            return pd.DataFrame([{
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
                "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }])

        left, right = st.columns(2)
        with left:
            patient_a = comparison_form("compare_a", "Patient A")
        with right:
            patient_b = comparison_form("compare_b", "Patient B")

        if st.button("Compare Patients", type="primary", use_container_width=True):
            try:
                pa, _ = predict_patient(patient_a)
                pb, _ = predict_patient(patient_b)
                delta = pa - pb
                metric_grid([
                    ("Patient A Probability", f"{pa:.1%}"),
                    ("Patient B Probability", f"{pb:.1%}"),
                    ("Absolute Difference", f"{abs(delta):.1%}"),
                    ("Decision Threshold", f"{THRESHOLD:.1%}"),
                ], columns=4)
                comparison = pd.DataFrame({
                    "Feature": RAW_FEATURES,
                    "Patient A": patient_a.iloc[0].values,
                    "Patient B": patient_b.iloc[0].values,
                })
                st.markdown("### Feature-by-Feature Comparison")
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                chart = pd.DataFrame({"Patient": ["Patient A", "Patient B"], "Model Probability": [pa, pb]})
                if PLOTLY_AVAILABLE:
                    fig = px.bar(chart, x="Patient", y="Model Probability", range_y=[0, 1], text_auto=".1%")
                    fig.update_layout(template="plotly_dark", height=360, yaxis_title="Probability")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(chart.set_index("Patient"))
                st.caption("This comparison reflects model output only; it is not a clinical assessment.")
            except Exception as error:
                st.error(f"Comparison could not be completed: {error}")


# ============================================================
# MODEL CARD
# ============================================================

elif page == "Model Card":

    section_title(
        "📋 Model Card",
        "A concise technical record of the deployed HeartCare AI model."
    )

    metric_grid([
        ("Model", MODEL_NAME),
        ("Threshold", f"{THRESHOLD:.2f}"),
        ("Dataset Rows", str(len(DATASET)) if DATASET is not None else "N/A"),
        ("Raw Features", str(len(RAW_FEATURES))),
    ], columns=4)

    st.markdown("### 🎯 Evaluation")
    metric_grid([
        ("Accuracy", f"{METRICS.get('accuracy', 0):.2%}"),
        ("Precision", f"{METRICS.get('precision', 0):.2%}"),
        ("Recall", f"{METRICS.get('recall', 0):.2%}"),
        ("F1", f"{METRICS.get('f1', 0):.2%}"),
        ("ROC-AUC", f"{METRICS.get('roc_auc', 0):.2%}"),
        ("PR-AUC", f"{METRICS.get('pr_auc', 0):.2%}"),
        ("Brier Score", f"{METRICS.get('brier_score', 0):.4f}"),
    ], columns=4)

    st.markdown("### ⚙️ Model Configuration")
    st.dataframe(pd.DataFrame([
        {"Property": "Target", "Value": METADATA.get("target", "N/A")},
        {"Property": "Random State", "Value": METADATA.get("random_state", RANDOM_STATE)},
        {"Property": "Test Size", "Value": METADATA.get("test_size", "N/A")},
        {"Property": "Threshold Tuning", "Value": METADATA.get("threshold_tuning", "N/A")},
        {"Property": "Deployment Training", "Value": METADATA.get("deployment_training", "N/A")},
        {"Property": "Selected Features", "Value": ", ".join(METADATA.get("project_final_features", []))},
    ]), use_container_width=True, hide_index=True)

    st.markdown("### 🧩 Feature Engineering")
    engineered = METADATA.get("feature_engineering", ENGINEERED_FEATURES)
    if isinstance(engineered, dict):
        engineered = list(engineered.keys())
    st.write(", ".join(engineered) if engineered else "Not available in the current artifact set.")

    st.markdown("### ⚠️ Limitations")
    st.warning("The model is based on a limited UCI-style heart disease dataset and its predictions should be interpreted as machine-learning outputs for educational/research use, not as medical diagnoses.")

    st.markdown("### 📦 Artifact Status")
    status = pd.DataFrame([
        {"Artifact": "Trained Pipeline", "Available": MODEL is not None},
        {"Artifact": "Feature Metadata", "Available": bool(METADATA)},
        {"Artifact": "Model Metrics", "Available": bool(METRICS)},
        {"Artifact": "Evaluation Predictions", "Available": PREDICTIONS_PATH.exists()},
        {"Artifact": "Reference Dataset", "Available": DATASET is not None},
    ])
    st.dataframe(status, use_container_width=True, hide_index=True)


elif page == "Model Insights":

    section_title(
        "🧠 Model Insights",
        "Technical information loaded from the existing artifacts."
    )

    metric_grid(
        [
            (
                "Model",
                MODEL_NAME
            ),
            (
                "Threshold",
                f"{THRESHOLD:.2f}"
            ),
            (
                "Random State",
                str(RANDOM_STATE)
            ),
            (
                "Raw Features",
                str(len(RAW_FEATURES))
            ),
        ],
        columns=4
    )

    st.markdown(
        "### 📊 Evaluation Metrics"
    )

    if METRICS:

        metric_grid(
            [
                (
                    "Accuracy",
                    f"{METRICS.get('accuracy', 0):.2%}"
                ),
                (
                    "Precision",
                    f"{METRICS.get('precision', 0):.2%}"
                ),
                (
                    "Recall",
                    f"{METRICS.get('recall', 0):.2%}"
                ),
                (
                    "F1",
                    f"{METRICS.get('f1', 0):.2%}"
                ),
                (
                    "ROC-AUC",
                    f"{METRICS.get('roc_auc', 0):.2%}"
                ),
                (
                    "PR-AUC",
                    f"{METRICS.get('pr_auc', 0):.2%}"
                ),
                (
                    "Brier Score",
                    f"{METRICS.get('brier_score', 0):.4f}"
                ),
            ],
            columns=4
        )

    else:

        st.info(
            "Evaluation metrics are not available "
            "in the current artifact set."
        )

    st.markdown(
        "### ⚙️ Feature Engineering"
    )

    for feature in ENGINEERED_FEATURES:

        st.html(
            f"""
            <div class="card"
                 style="
                    padding:12px 18px;
                    margin:7px 0;
                 ">

                <span style="
                    color:#ff5966;
                    font-weight:800;
                ">
                    {feature}
                </span>

            </div>
            """
        )

    st.markdown(
        "### 🧩 Model Metadata"
    )

    if METADATA:

        metadata_display = {
            key: value
            for key, value
            in METADATA.items()
            if not isinstance(
                value,
                (dict, list)
            )
        }

        if metadata_display:

            st.json(
                metadata_display
            )

    else:

        st.info(
            "Metadata is not available."
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    section_title(
        "🕘 Prediction History",
        "Predictions made during the current Streamlit session."
    )

    history = (
        st.session_state.prediction_history
    )

    if not history:

        st.info(
            "No predictions have been made yet."
        )

    else:

        history_df = pd.DataFrame(
            history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        metric_grid(
            [
                (
                    "Predictions",
                    str(len(history_df))
                ),
                (
                    "Average Probability",
                    f"{history_df['Probability'].mean():.1%}"
                ),
                (
                    "Maximum Probability",
                    f"{history_df['Probability'].max():.1%}"
                ),
                (
                    "Minimum Probability",
                    f"{history_df['Probability'].min():.1%}"
                ),
            ],
            columns=4
        )

        st.download_button(
            "📥 Download History CSV",
            data=history_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                "heartcare_prediction_history.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

        if st.button(
            "🗑 Clear History",
            use_container_width=True
        ):

            st.session_state.prediction_history = []

            st.rerun()



# ============================================================
# MODEL VALIDATION
# ============================================================

elif page == "Model Validation":

    section_title(
        "🧪 Model Validation",
        "Evaluate the saved model using the recorded evaluation predictions."
    )

    y_true, y_prob = evaluation_arrays()

    if y_true is None or len(y_true) == 0:
        st.warning(
            "Validation data is unavailable. Make sure artifacts/predictions.csv "
            "contains the actual and probability columns."
        )
    else:
        validation_threshold = float(THRESHOLD)
        y_pred = (y_prob >= validation_threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)

        metric_grid(
            [
                ("Accuracy", f"{accuracy:.2%}"),
                ("Precision", f"{precision:.2%}"),
                ("Recall", f"{recall:.2%}"),
                ("F1 Score", f"{f1:.2%}"),
                ("ROC-AUC", f"{roc_auc:.2%}"),
                ("PR-AUC", f"{pr_auc:.2%}"),
            ],
            columns=6
        )

        st.markdown("### Evaluation Details")
        c1, c2, c3 = st.columns(3)
        c1.metric("Decision Threshold", f"{validation_threshold:.2f}")
        c2.metric("Brier Score", f"{brier:.4f}")
        c3.metric("Evaluation Samples", f"{len(y_true):,}")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        st.markdown("### Confusion Matrix")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Negative", int(tn))
        c2.metric("False Positive", int(fp))
        c3.metric("False Negative", int(fn))
        c4.metric("True Positive", int(tp))

        if PLOTLY_AVAILABLE:
            left, right = st.columns(2)

            with left:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"ROC-AUC {roc_auc:.3f}",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random baseline",
                        line=dict(dash="dash"),
                    )
                )
                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=390,
                )
                st.plotly_chart(fig, use_container_width=True)

            with right:
                precision_curve, recall_curve, _ = precision_recall_curve(
                    y_true, y_prob
                )
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=recall_curve,
                        y=precision_curve,
                        mode="lines",
                        name=f"PR-AUC {pr_auc:.3f}",
                    )
                )
                fig.update_layout(
                    title="Precision-Recall Curve",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=390,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Recorded Artifact Metrics")
        if METRICS:
            recorded = pd.DataFrame(
                [
                    {
                        "Metric": key.replace("_", " ").title(),
                        "Value": value,
                    }
                    for key, value in METRICS.items()
                    if isinstance(value, (int, float))
                ]
            )
            st.dataframe(recorded, use_container_width=True, hide_index=True)

# ============================================================
# ERROR ANALYSIS
# ============================================================

elif page == "Error Analysis":

    section_title(
        "🔍 Error Analysis",
        "Inspect false positives, false negatives, and correct predictions."
    )

    if PREDICTIONS is None or PREDICTIONS.empty:
        st.warning("artifacts/predictions.csv is unavailable.")
    elif not {"actual", "probability"}.issubset(PREDICTIONS.columns):
        st.error(
            "The predictions artifact must contain at least: actual and probability."
        )
    else:
        errors = PREDICTIONS.copy()
        errors["actual"] = pd.to_numeric(errors["actual"], errors="coerce")
        errors["probability"] = pd.to_numeric(
            errors["probability"], errors="coerce"
        )
        errors = errors.dropna(subset=["actual", "probability"])
        errors["actual"] = errors["actual"].astype(int)
        errors["prediction"] = (
            errors["probability"] >= float(THRESHOLD)
        ).astype(int)

        errors["error_type"] = np.select(
            [
                (errors["actual"] == 1) & (errors["prediction"] == 0),
                (errors["actual"] == 0) & (errors["prediction"] == 1),
                errors["actual"] == errors["prediction"],
            ],
            [
                "False Negative",
                "False Positive",
                "Correct",
            ],
            default="Unknown",
        )

        counts = errors["error_type"].value_counts()
        metric_grid(
            [
                ("Correct", str(int(counts.get("Correct", 0)))),
                ("False Positives", str(int(counts.get("False Positive", 0)))),
                ("False Negatives", str(int(counts.get("False Negative", 0)))),
                ("Total Samples", str(len(errors))),
            ],
            columns=4
        )

        st.markdown("### Error Distribution")

        if PLOTLY_AVAILABLE:
            chart_data = (
                errors["error_type"]
                .value_counts()
                .rename_axis("Type")
                .reset_index(name="Count")
            )
            fig = px.bar(
                chart_data,
                x="Type",
                y="Count",
                title="Prediction Outcome Distribution",
                text="Count",
            )
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Inspect Cases")

        selected_type = st.selectbox(
            "Show",
            ["All", "False Negative", "False Positive", "Correct"],
        )

        if selected_type == "All":
            view = errors
        else:
            view = errors[errors["error_type"] == selected_type]

        preferred_columns = [
            "actual",
            "prediction",
            "probability",
            "threshold",
            "error_type",
        ]
        columns = [c for c in preferred_columns if c in view.columns]
        extra_columns = [
            c for c in view.columns
            if c not in columns and c != "prediction"
        ]
        view = view[columns + extra_columns]

        st.dataframe(
            view.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Error Analysis CSV",
            view.to_csv(index=False).encode("utf-8"),
            "heartcare_error_analysis.csv",
            "text/csv",
            use_container_width=True,
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    section_title(
        "❤️ About HeartCare AI"
    )

    st.html(
        """
        <div class="card">

            <h2 style="color:#ff5966;">
                HeartCare AI
            </h2>

            <p style="
                color:#cfc4c8;
                line-height:1.8;
            ">

                HeartCare AI is an educational machine learning
                application built around a trained heart disease
                prediction pipeline.

            </p>

            <h3 style="color:#ff5966;">
                🚀 Project Capabilities
            </h3>

            <ul style="
                color:#aaa0a4;
                line-height:2;
            ">

                <li>Interactive patient prediction</li>
                <li>Model-generated probability</li>
                <li>Decision threshold analysis</li>
                <li>Batch CSV prediction</li>
                <li>Dataset exploration</li>
                <li>Model performance metrics</li>
                <li>Feature engineering inspection</li>
                <li>Prediction history</li>
                <li>Downloadable reports</li>

            </ul>

        </div>
        """
    )

    st.markdown(
        "### 🛡 Medical Disclaimer"
    )

    st.warning(
        "This application is for educational and research "
        "purposes only. It is not a medical diagnostic system "
        "and should not replace professional medical advice."
    )


# ============================================================
# FOOTER
# ============================================================

CURRENT_YEAR = datetime.now().year

st.html(
    f"""
    <div class="footer">

        <div class="footer-name">
            ❤️ HeartCare AI
        </div>

        <div style="
            margin-top:10px;
            margin-bottom:14px;
        ">

            Machine Learning •
            Predictive Analytics •
            Explainable AI

        </div>

        <div>

            <a
                href="https://www.linkedin.com/in/nouramaherelamin/"
                target="_blank"
            >
                🔗 LinkedIn
            </a>

            |

            <a
                href="https://github.com/nouramaherelamin"
                target="_blank"
            >
                💻 GitHub
            </a>

        </div>

        <div style="
            margin-top:15px;
            font-size:12px;
        ">

            © {CURRENT_YEAR}
            Noura Maher Elamin.
            All Rights Reserved.

        </div>

    </div>
    """
)