import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.pkl")
pipe = joblib.load(MODEL_PATH)

disposition_map = {0: "CANDIDATE", 1: "CONFIRMED", 2: "FALSE POSITIVE"}

FEATURES = [
    "koi_period", "koi_time0bk", "koi_impact", "koi_duration",
    "koi_depth", "koi_prad", "koi_teq", "koi_srad", "koi_smass",
    "koi_steff", "koi_model_snr", "koi_dor", "koi_insol",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"
]

st.set_page_config(layout="wide", page_title="Kepler Disposition Classifier")
st.title("Kepler KOI Disposition Classifier")

st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload Kepler CSV (optional)", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, comment='#', on_bad_lines='skip')
        st.write("Preview of uploaded data")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

st.header("Single classifier")
with st.form("single"):
    col1, col2, col3 = st.columns(3)
    koi_period = col1.number_input("Orbital period (days)", value=10.0)
    koi_time0bk = col2.number_input("Time of first transit (BKJD)", value=100.0)
    koi_impact = col3.number_input("Impact parameter", value=0.5)

    col4, col5, col6 = st.columns(3)
    koi_duration = col4.number_input("Transit duration (hours)", value=5.0)
    koi_depth = col5.number_input("Transit depth (ppm)", value=500.0)
    koi_prad = col6.number_input("Planet radius (Earth radii)", value=1.0)

    col7, col8, col9 = st.columns(3)
    koi_teq = col7.number_input("Equilibrium temp (K)", value=300.0)
    koi_srad = col8.number_input("Stellar radius (Solar radii)", value=1.0)
    koi_smass = col9.number_input("Stellar mass (Solar masses)", value=1.0)

    col10, col11, col12 = st.columns(3)
    koi_steff = col10.number_input("Stellar effective temp (K)", value=5778.0)
    koi_model_snr = col11.number_input("Model SNR", value=50.0)
    koi_dor = col12.number_input("Depth of transit (dor)", value=0.01)

    col13, col14, col15, col16 = st.columns(4)
    koi_insol = col13.number_input("Insolation flux", value=100.0)
    koi_fpflag_nt = col14.number_input("FP flag NT", value=0)
    koi_fpflag_ss = col15.number_input("FP flag SS", value=0)
    koi_fpflag_co = col16.number_input("FP flag CO", value=0)
    koi_fpflag_ec = st.number_input("FP flag EC", value=0)

    submitted = st.form_submit_button("Classifier")
    if submitted:
        input_df = pd.DataFrame([{
            "koi_period": koi_period,
            "koi_time0bk": koi_time0bk,
            "koi_impact": koi_impact,
            "koi_duration": koi_duration,
            "koi_depth": koi_depth,
            "koi_prad": koi_prad,
            "koi_teq": koi_teq,
            "koi_srad": koi_srad,
            "koi_smass": koi_smass,
            "koi_steff": koi_steff,
            "koi_model_snr": koi_model_snr,
            "koi_dor": koi_dor,
            "koi_insol": koi_insol,
            "koi_fpflag_nt": koi_fpflag_nt,
            "koi_fpflag_ss": koi_fpflag_ss,
            "koi_fpflag_co": koi_fpflag_co,
            "koi_fpflag_ec": koi_fpflag_ec
        }])
        pred = pipe.predict(input_df)[0]
        proba = pipe.predict_proba(input_df)[0]
        st.success(f"Classifier: {disposition_map.get(pred, pred)}")
        st.write(dict(zip(pipe.classes_, proba)))

if uploaded is not None:
    st.header("Batch predictions")
    if all(f in df.columns for f in FEATURES):
        preds = pipe.predict(df[FEATURES])
        probs = pipe.predict_proba(df[FEATURES])

    if all(isinstance(c, int) for c in pipe.classes_):
        out = pd.DataFrame(probs, columns=[disposition_map[c] for c in pipe.classes_])
        out["prediction"] = [disposition_map[p] for p in preds]
    else:
        out = pd.DataFrame(probs, columns=pipe.classes_)
        out["prediction"] = preds

    st.dataframe(out.head())

st.header("Exploratory plots (from uploaded data)")

if uploaded is not None:
    available_cols = df.columns.tolist()
    default_cols = [c for c in ["koi_period", "koi_prad"] if c in available_cols]
    
    cols = st.multiselect(
        "Choose x and y", 
        options=available_cols, 
        default=default_cols
    )

    if len(cols) >= 2:
        fig, ax = plt.subplots(figsize=(8,6))
        
        if "koi_disposition" in df.columns:
            df['Disposition'] = df['koi_disposition'].map(disposition_map).fillna(df['koi_disposition'])
            sns.scatterplot(data=df, x=cols[0], y=cols[1], hue="Disposition", palette="Set1", alpha=0.7, ax=ax)
        else:
            sns.scatterplot(data=df, x=cols[0], y=cols[1], alpha=0.7, ax=ax)
    
        if "koi_period" in cols and "koi_prad" in cols:
            ax.set_xscale("log")
            ax.set_yscale("log")
        
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_title(f"{cols[1]} vs {cols[0]} by Disposition")
        ax.legend(title="Disposition")
        
        st.pyplot(fig)



