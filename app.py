import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from pycaret.classification import load_model as load_cls, predict_model as predict_cls
from pycaret.clustering import load_model as load_clu, predict_model as predict_clu

st.set_page_config(page_title="Cognitive SOAR — Threat Attribution", layout="wide")

st.title("🧠 Cognitive SOAR — From Prediction to Attribution")
st.write(
    "Submit URL features. The classifier predicts **MALICIOUS/BENIGN**. "
    "If MALICIOUS, we also infer a likely **Threat Actor Profile** via clustering."
)

# Paths and artifacts
CLASSIFIER_PATH = "models/phishing_url_detector"
CLUSTER_PATH = "models/threat_actor_profiler"
MAP_PATH = Path("models/cluster_profile_map.json")

# Load models (classifier required, clusterer optional until needed)
cls_model = load_cls(CLASSIFIER_PATH)
clu_model = load_clu(CLUSTER_PATH) if Path(f"{CLUSTER_PATH}.pkl").exists() else None
cluster_map = json.loads(MAP_PATH.read_text()) if MAP_PATH.exists() else {}

FEATURES = [
    "SSLfinal_State",
    "Prefix_Suffix",
    "Shortining_Service",
    "having_IP_Address",
    "abnormal_url",
    "has_political_keyword",
    "suspicious_tld",
    "url_length",
    "num_subdomains",
    "redirect_count",
    "domain_age_days",
    "https_token_in_url",
]

# Presets for quick testing
PRESETS = {
    "Benign": {
        "SSLfinal_State": 1, "Prefix_Suffix": 0, "Shortining_Service": 0, "having_IP_Address": 0,
        "abnormal_url": 0, "has_political_keyword": 0, "suspicious_tld": 0,
        "url_length": 25, "num_subdomains": 1, "redirect_count": 0, "domain_age_days": 2000,
        "https_token_in_url": 0,
    },
    "Organized Cybercrime": {
        "SSLfinal_State": 0, "Prefix_Suffix": 1, "Shortining_Service": 1, "having_IP_Address": 1,
        "abnormal_url": 1, "has_political_keyword": 0, "suspicious_tld": 1,
        "url_length": 90, "num_subdomains": 3, "redirect_count": 3, "domain_age_days": 20,
        "https_token_in_url": 1,
    },
    "State-Sponsored": {
        "SSLfinal_State": 1, "Prefix_Suffix": 1, "Shortining_Service": 0, "having_IP_Address": 0,
        "abnormal_url": 0, "has_political_keyword": 0, "suspicious_tld": 0,
        "url_length": 55, "num_subdomains": 1, "redirect_count": 0, "domain_age_days": 2500,
        "https_token_in_url": 0,
    },
    "Hacktivist": {
        "SSLfinal_State": 1, "Prefix_Suffix": 0, "Shortining_Service": 0, "having_IP_Address": 0,
        "abnormal_url": 1, "has_political_keyword": 1, "suspicious_tld": 0,
        "url_length": 45, "num_subdomains": 2, "redirect_count": 1, "domain_age_days": 800,
        "https_token_in_url": 0,
    },
}

with st.sidebar:
    st.header("Quick Presets")
    preset = st.selectbox("Load example values", list(PRESETS.keys()))
    if st.button("Load Preset"):
        for k, v in PRESETS[preset].items():
            st.session_state[k] = v
    st.caption("Use presets to populate the form quickly for screenshots/testing.")

tabs = st.tabs(["🔎 Classifier", "🧩 Threat Attribution", "ℹ️ About"])

# -----------------------
# Classifier tab
# -----------------------
with tabs[0]:
    st.subheader("Input Features")
    cols = st.columns(3)
    vals = {}

    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            if feat in ["url_length", "num_subdomains", "redirect_count", "domain_age_days"]:
                if feat == "url_length":
                    minv, maxv, step = 1, 200, 1
                elif feat == "num_subdomains":
                    minv, maxv, step = 0, 5, 1
                elif feat == "redirect_count":
                    minv, maxv, step = 0, 10, 1
                else:  # domain_age_days
                    minv, maxv, step = 1, 4000, 1
                default = st.session_state.get(feat, PRESETS["Benign"].get(feat, 0))
                vals[feat] = st.number_input(feat, min_value=minv, max_value=maxv, value=int(default), step=step)
            else:
                default = st.session_state.get(feat, PRESETS["Benign"].get(feat, 0))
                vals[feat] = st.selectbox(feat, [0, 1], index=int(default))

    df_input = pd.DataFrame([vals])

    # Back-compat: if an older classifier was trained with 'actor_profile' as a feature,
    # provide a placeholder so predict_model doesn't error.
    if "actor_profile" not in df_input.columns:
        df_input["actor_profile"] = "UNKNOWN"

    if st.button("Predict"):
        pred = predict_cls(cls_model, data=df_input)
        label = pred["prediction_label"].iloc[0]
        score = pred["prediction_score"].iloc[0]
        st.metric("Verdict", label, help="Classifier output")
        st.progress(min(max(score, 0.0), 1.0), text=f"Confidence: {score:.2f}")
        st.session_state["last_input"] = df_input
        st.session_state["last_label"] = label

# -----------------------
# Threat Attribution tab
# -----------------------
with tabs[1]:
    st.subheader("Threat Attribution")
    if st.session_state.get("last_label") != "MALICIOUS":
        st.info("Run a prediction in the **Classifier** tab, and ensure the verdict is MALICIOUS to see attribution.")
    else:
        if (clu_model is None) or (not cluster_map):
            st.warning("Clustering model or mapping not found. Train models first (run `python train_model.py`) and ensure `models/cluster_profile_map.json` exists.")
        else:
            df_in = st.session_state["last_input"][FEATURES]
            clu_pred = predict_clu(clu_model, data=df_in)

            # Handle labels like "Cluster 1" or just 1
            cluster_raw = str(clu_pred["Cluster"].iloc[0])
            match = re.search(r"(\d+)", cluster_raw)
            cluster_key = match.group(1) if match else cluster_raw

            profile = cluster_map.get(cluster_key, f"Cluster {cluster_raw}")
            st.metric("Predicted Actor Profile", profile)

            DESCR = {
                "State-Sponsored": "High sophistication, subtle tactics, often well-resourced campaigns. Likely to use valid SSL and long-lived domains; objectives usually espionage or strategic influence.",
                "Organized Cybercrime": "Noisy, scalable monetization-driven operations. Frequent use of URL shorteners, IP-based hosts, and abnormal structures; objectives are credential theft and financial gain.",
                "Hacktivist": "Opportunistic campaigns with ideological or political motives. May reference topical or political keywords and mix hygiene levels.",
            }
            st.write(DESCR.get(profile, "No description available."))

# -----------------------
# About tab
# -----------------------
with tabs[2]:
    st.markdown(
        """
        ### About
        - Classifier: PyCaret classification.
        - Clusterer: PyCaret clustering (`kmeans`, k=3) trained on malicious-only samples.
        - Mapping: Cluster ID → Actor label derived from centroid heuristics saved in `models/cluster_profile_map.json`.
        """
    )
