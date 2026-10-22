"""Streamlit app: AI-based Ra prediction for end milling.  Run: streamlit run app.py"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import generate_dataset
import train as trainer
from common import MATERIALS, RAW_COLS, add_features, quality_grade
from optimizer import optimize_parameters

ART = Path("artifacts")
DATA = Path("data/milling_data.csv")
st.set_page_config(page_title="End Milling Ra Predictor", page_icon="🛠️", layout="wide")


@st.cache_resource
def load_model():
    if not (ART / "model.joblib").exists():
        with st.spinner("First run: generating data and training models..."):
            generate_dataset.generate()
            trainer.run()
    return joblib.load(ART / "model.joblib")


def predict(model, df):
    return model.predict(add_features(df[RAW_COLS]))


model = load_model()
meta = json.loads((ART / "meta.json").read_text())

st.title("🛠️ AI-Based Surface Roughness (Ra) Prediction – End Milling")
st.caption(f"Active model: **{meta['best_model']}** · trained on {meta['n_samples']} samples")

t_pred, t_opt, t_batch, t_report, t_train = st.tabs(
    ["🔮 Predict", "🎯 Optimize", "📂 Batch", "📊 Model report", "♻️ Data & retrain"])

# ---------------------------------------------------------------- Predict
with t_pred:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        material = st.selectbox("Workpiece material", MATERIALS)
        a, b = st.columns(2)
        tool_dia = a.selectbox("Tool diameter (mm)", [6, 8, 10, 12], index=2)
        flutes = b.selectbox("Flutes", [2, 3, 4], index=1)
        speed = st.slider("Spindle speed (rpm)", 2000, 10000, 6000, 100)
        fz = st.slider("Feed per tooth (mm/tooth)", 0.02, 0.15, 0.06, 0.005)
        doc = st.slider("Axial depth of cut (mm)", 0.5, 4.0, 2.0, 0.1)
        woc = st.slider("Radial depth of cut (mm)", 0.5, 6.0, 3.0, 0.1)
        wear = st.slider("Tool wear VB (mm)", 0.0, 0.30, 0.05, 0.01)
    row = pd.DataFrame([dict(speed=speed, feed_per_tooth=fz, doc=doc, woc=woc, tool_wear=wear,
                             tool_dia=tool_dia, flutes=flutes, material=material)])
    ra = float(predict(model, row)[0])
    feat = add_features(row).iloc[0]
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Ra", f"{ra:.3f} µm")
        m2.metric("Cutting speed", f"{feat.cutting_speed:.0f} m/min")
        m3.metric("MRR", f"{feat.mrr:.1f} cm³/min")
        st.info(f"Surface quality: **{quality_grade(ra)}**")
        st.markdown("**What-if: effect of feed per tooth**")
        sweep = pd.concat([row] * 40, ignore_index=True)
        sweep["feed_per_tooth"] = np.linspace(0.02, 0.15, 40)
        st.line_chart(pd.DataFrame({"Ra (µm)": predict(model, sweep)},
                                   index=sweep["feed_per_tooth"].round(3)))
        st.markdown("**What-if: effect of tool wear**")
        sw = pd.concat([row] * 30, ignore_index=True)
        sw["tool_wear"] = np.linspace(0, 0.3, 30)
        st.line_chart(pd.DataFrame({"Ra (µm)": predict(model, sw)}, index=sw["tool_wear"].round(3)))

# ---------------------------------------------------------------- Optimize
with t_opt:
    st.write("Finds cutting parameters using the trained model + differential evolution.")
    a, b, c, d = st.columns(4)
    o_mat = a.selectbox("Material", MATERIALS, key="om")
    o_dia = b.selectbox("Tool dia (mm)", [6, 8, 10, 12], 2, key="od")
    o_fl = c.selectbox("Flutes", [2, 3, 4], 1, key="of")
    o_wear = d.slider("Tool wear VB (mm)", 0.0, 0.3, 0.05, 0.01, key="ow")
    e, f = st.columns(2)
    ra_lim = e.number_input("Max allowed Ra (µm)", 0.1, 10.0, 1.6, 0.1)
    w = f.slider("Objective: productivity  ←→  surface finish", 0.0, 1.0, 0.3, 0.05)
    s1, s2, s3, s4 = st.columns(4)
    sp = s1.slider("Speed range (rpm)", 2000, 10000, (3000, 9000), 100)
    fr = s2.slider("Feed/tooth range", 0.02, 0.15, (0.03, 0.12), 0.005)
    dr = s3.slider("DOC range (mm)", 0.5, 4.0, (0.5, 3.0), 0.1)
    wr = s4.slider("WOC range (mm)", 0.5, 6.0, (0.5, 4.0), 0.1)
    if st.button("Optimize", type="primary"):
        with st.spinner("Searching..."):
            best = optimize_parameters(model, o_mat, o_dia, o_fl, o_wear,
                                       [sp, fr, dr, wr], ra_lim, w)
        st.success("Optimization complete" if best["feasible"]
                   else "No fully feasible solution in these ranges – showing closest.")
        r = st.columns(6)
        r[0].metric("Speed", f"{best.speed:.0f} rpm")
        r[1].metric("Feed/tooth", f"{best.feed_per_tooth:.3f} mm")
        r[2].metric("DOC", f"{best.doc:.2f} mm")
        r[3].metric("WOC", f"{best.woc:.2f} mm")
        r[4].metric("Predicted Ra", f"{best.Ra_pred:.3f} µm")
        r[5].metric("MRR", f"{best.mrr:.1f} cm³/min")

# ---------------------------------------------------------------- Batch
with t_batch:
    st.write("Upload a CSV with columns: `" + "`, `".join(RAW_COLS) + "`")
    st.download_button("Download template", pd.DataFrame(columns=RAW_COLS).to_csv(index=False),
                       "template.csv")
    up = st.file_uploader("CSV file", type="csv", key="batch")
    if up:
        bdf = pd.read_csv(up)
        miss = [c for c in RAW_COLS if c not in bdf.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            bdf["Ra_pred"] = predict(model, bdf)
            st.dataframe(bdf, use_container_width=True)
            st.download_button("Download predictions", bdf.to_csv(index=False), "ra_predictions.csv")

# ---------------------------------------------------------------- Report
with t_report:
    st.dataframe(pd.read_csv(ART / "model_comparison.csv").round(4), use_container_width=True)
    a, b = st.columns(2)
    a.image(str(ART / "pred_vs_actual.png"))
    b.image(str(ART / "feature_importance.png"))

# ---------------------------------------------------------------- Retrain
with t_train:
    st.write("Train on your own experimental data (CSV with the input columns plus `Ra`).")
    up = st.file_uploader("Training CSV", type="csv", key="train")
    if st.button("Retrain models") and up is not None:
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_bytes(up.getvalue())
        with st.spinner("Training..."):
            res = trainer.run(str(DATA))
        load_model.clear()
        st.success("Done – reload the page to use the new model.")
        st.dataframe(res.round(4))
    elif st.button("Regenerate synthetic data & retrain"):
        with st.spinner("Training..."):
            generate_dataset.generate()
            res = trainer.run()
        load_model.clear()
        st.success("Done – reload the page.")
        st.dataframe(res.round(4))
