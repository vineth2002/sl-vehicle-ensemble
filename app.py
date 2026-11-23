"""Ensemble Learning Dashboard - Sri Lankan used-car prices.

Run locally:  streamlit run app.py
Loads the models trained in ensemble_vehicle_prices.ipynb from models/*.pkl.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ml_utils  # noqa: F401  (needed so joblib can unpickle RFANNHybrid / CatBoostModel)
from ml_utils import FEATURES, add_features

ROOT = Path(__file__).parent
FAMILY_COLORS = {"Baseline": "#888780", "Bagging": "#1D9E75", "Boosting": "#534AB7",
                 "Stacking": "#D85A30", "Hybrid": "#D4537E"}
MAIN_MODELS = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Stacking", "Hybrid RF + ANN"]

st.set_page_config(page_title="Sri Lankan Car Price - Ensemble Lab", page_icon="🚗", layout="wide")


@st.cache_data
def load_results():
    meta = json.loads((ROOT / "results/meta.json").read_text())
    metrics = pd.read_csv(ROOT / "results/metrics.csv").set_index("model")
    perm = pd.read_csv(ROOT / "results/permutation_importance.csv")
    preds = pd.read_csv(ROOT / "results/test_predictions.csv")
    return meta, metrics, perm, preds


@st.cache_resource
def load_model(path: str):
    return joblib.load(ROOT / path)


meta, metrics, perm, preds = load_results()
best = meta["best_model"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🚗 Ensemble Lab")
    st.caption("Machine Learning and Pattern Recognition - Hands-On Ensemble Learning Project")
    st.markdown(
        f"""
**Data:** {meta['rows_after_cleaning']:,} used-car listings from **ikman.lk** and **Riyasewana.com**
(Dec 2024 - Feb 2025)

**Train / test:** {meta['train_rows']:,} / {meta['test_rows']:,}

**Target:** asking price (lakhs), modelled as log(price)

**Best model:** {best}
"""
    )
    st.divider()
    st.caption("Built with Streamlit. AI assistance was used in building this project; "
               "see the *About & AI usage* tab.")

st.title("Sri Lankan Used-Car Prices: Ensemble Learning Dashboard")
st.caption("Bagging vs Boosting vs Stacking vs Hybrid (RF + ANN), trained on local Sri Lankan marketplace data")

tab_overview, tab_compare, tab_importance, tab_predict, tab_about = st.tabs(
    ["Overview", "Model comparison", "Feature importance", "Live prediction", "About & AI usage"]
)

# ---------------------------------------------------------------- overview
with tab_overview:
    b = metrics.loc[best]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", best)
    c2.metric("Typical error (MAPE)", f"{b['MAPE_pct']:.1f}%")
    c3.metric("R² on log(price)", f"{b['R2_log']:.3f}")
    c4.metric("Mean absolute error", f"Rs {b['MAE_lakhs']:.1f} lakhs")

    st.markdown("#### What this project compares")
    cols = st.columns(4)
    blurbs = [
        ("Bagging", "Random Forest", "Hundreds of deep trees on random samples of the data, averaged. Cuts variance."),
        ("Boosting", "XGBoost, LightGBM, CatBoost", "Shallow trees added one after another, each fixing the last one's errors. Cuts bias."),
        ("Stacking", "RF + LightGBM + CatBoost + SVR", "Different models' predictions combined by a Ridge meta-learner."),
        ("Hybrid", "Random Forest + ANN", "The forest predicts, then a small neural network learns to correct its errors."),
    ]
    for col, (fam, models, text) in zip(cols, blurbs):
        with col:
            st.markdown(f"<span style='color:{FAMILY_COLORS[fam]};font-weight:600'>{fam}</span>", unsafe_allow_html=True)
            st.markdown(f"**{models}**")
            st.caption(text)

    st.markdown("#### Sri Lankan context: the vehicle import ban")
    st.info(
        "Sri Lanka suspended vehicle imports from March 2020 until 1 February 2025. Very few cars made in "
        "2020 or later exist in this dataset, so every model is least reliable for the newest cars. The "
        "listings were scraped right around the date the ban was lifted, and a 'listed after ban lifted' "
        "flag is one of the model features."
    )
    fig = px.scatter(
        preds.sample(min(1500, len(preds)), random_state=0), x="age", y="actual_price", log_y=True,
        color="fuel_type", opacity=0.6, labels={"age": "Car age (years)", "actual_price": "Price (lakhs, log scale)"},
        title="Test-set cars: price falls with age (log scale)",
    )
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- comparison
with tab_compare:
    st.markdown("All metrics are on the **held-out test set** (20% of listings the models never saw).")
    families = st.multiselect("Show model families", list(FAMILY_COLORS), default=list(FAMILY_COLORS))
    view = metrics[metrics["family"].isin(families)].copy()

    show = view[["family", "RMSE_log", "R2_log", "MAPE_pct", "MAE_lakhs", "RMSE_lakhs", "R2_lakhs",
                 "train_R2_log", "seconds"]].rename(columns={
        "RMSE_log": "RMSE (log)", "R2_log": "R² (log)", "MAPE_pct": "MAPE %", "MAE_lakhs": "MAE (lakhs)",
        "RMSE_lakhs": "RMSE (lakhs)", "R2_lakhs": "R² (lakhs)", "train_R2_log": "Train R² (log)",
        "seconds": "Tune + fit (s)"})
    st.dataframe(show.style.format(precision=3).format({"MAPE %": "{:.1f}", "Tune + fit (s)": "{:.0f}"})
                 .highlight_min(subset=["RMSE (log)", "MAPE %", "MAE (lakhs)"], color="#C0DD97")
                 .highlight_max(subset=["R² (log)", "R² (lakhs)"], color="#C0DD97"), width="stretch")

    metric_label = st.radio("Metric to chart", ["RMSE (log)", "MAPE %", "R² (log)", "MAE (lakhs)"], horizontal=True)
    col_map = {"RMSE (log)": "RMSE_log", "MAPE %": "MAPE_pct", "R² (log)": "R2_log", "MAE (lakhs)": "MAE_lakhs"}
    m = col_map[metric_label]
    asc = m != "R2_log"
    chart_df = view.reset_index().sort_values(m, ascending=not asc)
    fig = px.bar(chart_df, x=m, y="model", color="family", orientation="h", color_discrete_map=FAMILY_COLORS,
                 labels={m: metric_label, "model": ""}, title=f"{metric_label} by model")
    if m == "R2_log":
        fig.update_xaxes(range=[chart_df[m].min() - 0.02, 1])
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        gap = view.reset_index()
        fig = go.Figure([
            go.Bar(name="Train R²", x=gap["model"], y=gap["train_R2_log"], marker_color="#AFA9EC"),
            go.Bar(name="Test R²", x=gap["model"], y=gap["R2_log"], marker_color="#534AB7"),
        ])
        fig.update_layout(barmode="group", title="Overfitting check: train vs test R² (log)",
                          yaxis_range=[0.8, 1.0], height=420)
        st.plotly_chart(fig, width="stretch")
    with right:
        w = pd.Series(meta["stacking_weights"]).rename({"rf": "Random Forest", "lgbm": "LightGBM",
                                                       "catboost": "CatBoost", "svr": "SVR"})
        fig = px.bar(x=w.index, y=w.values, labels={"x": "Base model", "y": "Ridge weight"},
                     title="Stacking: how much the meta-learner trusts each base model",
                     color_discrete_sequence=["#D85A30"])
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Predicted vs actual")
    chosen = st.selectbox("Model", metrics.index.tolist(), index=metrics.index.tolist().index(best))
    p = preds[["actual_price", f"pred::{chosen}", "brand", "model", "age"]].rename(columns={f"pred::{chosen}": "predicted"})
    p["error_%"] = 100 * (p["predicted"] - p["actual_price"]) / p["actual_price"]
    fig = px.scatter(p, x="actual_price", y="predicted", log_x=True, log_y=True, opacity=0.5,
                     hover_data=["brand", "model", "age", "error_%"],
                     labels={"actual_price": "Actual (lakhs)", "predicted": "Predicted (lakhs)"},
                     color_discrete_sequence=["#534AB7"])
    lo, hi = p["actual_price"].min() * 0.8, p["actual_price"].max() * 1.2
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="#D85A30"),
                             name="Perfect prediction"))
    st.plotly_chart(fig, width="stretch")
    within10 = (p["error_%"].abs() <= 10).mean()
    within20 = (p["error_%"].abs() <= 20).mean()
    st.caption(f"{chosen}: {within10:.0%} of test cars within ±10% of the asking price, {within20:.0%} within ±20%.")

    st.markdown("#### Where do models struggle? Median % error by car age")
    groups = pd.cut(preds["age"], [0, 5, 10, 20, 100], labels=["1-5 yrs (2020+)", "6-10 yrs", "11-20 yrs", "20+ yrs"])
    rows = {}
    for name in metrics.index:
        ape = 100 * (preds[f"pred::{name}"] - preds["actual_price"]).abs() / preds["actual_price"]
        rows[name] = ape.groupby(groups, observed=False).median()
    age_tbl = pd.DataFrame(rows).T
    st.dataframe(age_tbl.style.format("{:.1f}%").background_gradient(cmap="Reds", axis=None), width="stretch")
    st.caption("Test cars per group: " + ", ".join(f"{k}: {v}" for k, v in groups.value_counts().sort_index().items()))

    with st.expander("Best hyperparameters found by GridSearchCV"):
        st.json(meta["tuning"])

# ---------------------------------------------------------------- importance
with tab_importance:
    st.markdown(
        "**Permutation importance:** each feature is shuffled on the test set and we measure how much the "
        "error (RMSE on log price) increases. A bigger increase means the model relies on that feature more. "
        "It works for every model, including Stacking and the Hybrid."
    )
    pretty = {"model": "Car model", "brand": "Brand", "age": "Age", "engine_cc": "Engine (cc)",
              "mileage_km": "Mileage", "km_per_year": "Km per year", "fuel_type": "Fuel type", "gear": "Gear",
              "town": "Town", "leasing": "Leasing", "condition": "Condition (new/used)",
              "listed_after_ban": "Listed after ban lifted", "air_condition": "Air conditioning",
              "power_steering": "Power steering", "power_mirror": "Power mirrors", "power_window": "Power windows"}
    perm["Feature"] = perm["feature"].map(pretty)
    available = [m for m in MAIN_MODELS if m in perm["model"].unique()]
    sel = st.selectbox("Model", available, index=available.index(best) if best in available else 0)
    d = perm[perm["model"] == sel].sort_values("importance")
    fam = metrics.loc[sel, "family"]
    fig = px.bar(d, x="importance", y="Feature", orientation="h", error_x="std",
                 color_discrete_sequence=[FAMILY_COLORS[fam]],
                 labels={"importance": "Increase in RMSE (log) when shuffled", "Feature": ""},
                 title=f"{sel}: permutation importance")
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### All models side by side")
    pivot = perm.pivot_table(index="Feature", columns="model", values="importance")[available]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(pivot, text_auto=".3f", aspect="auto", color_continuous_scale="Purples",
                    labels={"color": "Importance"})
    fig.update_layout(height=560)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- prediction
with tab_predict:
    st.markdown("Describe a car and every trained model will estimate its asking price.")
    st.caption("Mileage is not an input: in this dataset mileage was always recorded as age × 11,000 km, "
               "so it carries no extra information and the models do not use it (see notebook section 4).")
    brands = meta["brands_by_count"]
    c1, c2, c3 = st.columns(3)
    with c1:
        brand = st.selectbox("Brand", brands, index=brands.index("TOYOTA") if "TOYOTA" in brands else 0)
        models_for_brand = meta["brand_models"].get(brand, [])
        car_model = st.selectbox("Model", models_for_brand)
        yom = st.slider("Year of manufacture", 1990, meta["ranges"]["yom"][1], 2015)
        engine = st.number_input("Engine capacity (cc)", 600, 5000, 1300, step=50)
    with c2:
        fuel = st.selectbox("Fuel type", meta["options"]["fuel_type"],
                            index=meta["options"]["fuel_type"].index("Petrol") if "Petrol" in meta["options"]["fuel_type"] else 0)
        gear = st.selectbox("Transmission", meta["options"]["gear"],
                            index=meta["options"]["gear"].index("Automatic") if "Automatic" in meta["options"]["gear"] else 0)
        town = st.selectbox("Town", meta["towns"])
    with c3:
        condition = st.selectbox("Condition", meta["options"]["condition"],
                                 index=meta["options"]["condition"].index("USED") if "USED" in meta["options"]["condition"] else 0)
        leasing = st.selectbox("Leasing", meta["options"]["leasing"])
        st.markdown("Equipment")
        ac = st.checkbox("Air conditioning", True)
        ps = st.checkbox("Power steering", True)
        pm = st.checkbox("Power mirrors", True)
        pw = st.checkbox("Power windows", True)
        after_ban = st.toggle("Listed after import ban lifted (Feb 2025)", True)

    avail = lambda flag: "Available" if flag else "Not_Available"
    row = pd.DataFrame([{
        "brand": brand, "model": car_model, "yom": yom, "engine_cc": float(engine), "gear": gear,
        "fuel_type": fuel, "town": town, "leasing": leasing,
        "condition": condition, "air_condition": avail(ac), "power_steering": avail(ps),
        "power_mirror": avail(pm), "power_window": avail(pw), "listed_after_ban": int(after_ban),
    }])
    X_new = add_features(row)[FEATURES]

    if yom >= 2020:
        st.warning("Cars made in 2020 or later are rare in the training data because of the import ban, "
                   "so these estimates are less reliable.")

    estimates = {}
    with st.spinner("Asking the models..."):
        for name, path in meta["model_files"].items():
            try:
                estimates[name] = float(np.exp(load_model(path).predict(X_new)[0]))
            except Exception as exc:  # keep the app usable if one model fails to load
                st.error(f"{name} could not predict: {exc}")
    if estimates:
        est = pd.DataFrame({"model": list(estimates), "price": list(estimates.values())})
        est["family"] = est["model"].map(metrics["family"])
        top = estimates.get(best, est["price"].median())
        mape = metrics.loc[best, "MAPE_pct"] / 100
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Estimate ({best})", f"Rs {top:,.1f} lakhs", help="1 lakh = Rs 100,000")
        k2.metric("In rupees", f"Rs {top * 100_000:,.0f}")
        k3.metric("Typical range (± test MAPE)", f"{top * (1 - mape):,.1f} - {top * (1 + mape):,.1f} lakhs")
        est = est.sort_values("price")
        fig = px.bar(est, x="price", y="model", color="family", orientation="h", color_discrete_map=FAMILY_COLORS,
                     labels={"price": "Estimated price (lakhs)", "model": ""}, title="What each model thinks")
        fig.add_vline(x=top, line_dash="dash", line_color="#D85A30")
        st.plotly_chart(fig, width="stretch")
        spread = est["price"].max() / est["price"].min() - 1
        st.caption(f"Models disagree by {spread:.0%} between the lowest and highest estimate. A large spread "
                   "usually means the car is unusual compared with the training data.")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
#### Method in brief
1. **Data:** SriLankan Vehicle (Car) Price Dataset (Prasad Nirmal, Kaggle, MIT licence): listings scraped from
   ikman.lk and Riyasewana.com between December 2024 and February 2025. 18 duplicates and one clearly
   erroneous row removed.
2. **Features:** brand, model, age, engine size, fuel, transmission, town, leasing, condition, four
   equipment flags and a "listed after import ban lifted" flag. Mileage was dropped because the dataset
   records it as exactly age × 11,000 km for every car, so it adds no information beyond age.
3. **Target:** log(price), because raw prices are heavily right-skewed. Predictions are converted back with exp().
4. **Preprocessing (inside sklearn Pipelines, fitted on training data only):** target encoding for car model,
   one-hot encoding for other categories, scaling for SVR / ANN / Ridge. CatBoost uses native categorical handling.
5. **Tuning:** GridSearchCV with 3-fold cross-validation, scored by RMSE on log(price).
6. **Evaluation:** 20% held-out test set; RMSE (log) as the primary metric, plus MAE, R² and MAPE.

## 13. AI Usage Acknowledgment

I used **Claude (Anthropic)** in this project for:

- **Learning:** explanations of the ML concepts and ensemble methods.
- **Dataset choice:** comparing possible Sri Lankan datasets.
- **Code:** for the code of `ml_utils.py`, this dashboard and notebook.

"""
    )
