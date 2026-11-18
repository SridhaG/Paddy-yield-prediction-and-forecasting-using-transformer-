"""
Streamlit dashboard for Cauvery Delta paddy yield prediction results.
"""

import json
import os

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

st.set_page_config(
    page_title="Cauvery Paddy Yield",
    page_icon="🌾",
    layout="wide",
)


def load_json(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metric_table(mapping):
    rows = []
    for name, metrics in mapping.items():
        if not isinstance(metrics, dict) or "R2" not in metrics:
            continue
        rows.append(
            {
                "Model": name,
                "R²": metrics["R2"],
                "RMSE (kg/ha)": metrics["RMSE"],
                "MAE (kg/ha)": metrics["MAE"],
                "MAPE (%)": metrics["MAPE"],
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def show_image(filename, caption):
    path = os.path.join(RESULTS, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
        return True
    return False


st.title("Cauvery Delta Paddy Yield Prediction")
st.caption(
    "Area-weighted multiple-instance temporal transformer for Thanjavur, "
    "Thiruvarur, and Nagapattinam. Satellite and weather features are weekly; "
    "yield is district-season (kg/ha)."
)

comparison = load_json("model_comparison.json")
full_eval = load_json("full_evaluation.json")
combined = load_json("combined_results.json")
history = load_json("training_history.json")

if comparison is None and full_eval is None:
    st.error(
        "Results are not in this repository yet. From the project folder run "
        "`python src/collect_gee_data.py --synthetic`, then `python src/train.py`, "
        "`python src/evaluate.py`, and `python src/visualize.py`."
    )
    st.stop()

page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Model comparison",
        "District and season",
        "Early forecast",
        "Ablation",
        "Predictions",
    ],
)

if page == "Overview":
    source = comparison or {}
    if not source and full_eval:
        source = {
            k: v["metrics"]
            for k, v in full_eval.items()
            if isinstance(v, dict) and "metrics" in v
        }
    table = metric_table(source)
    if table is not None:
        best = table.sort_values("R²", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", best["Model"])
        c2.metric("R²", f"{best['R²']:.3f}")
        c3.metric("RMSE", f"{best['RMSE (kg/ha)']:.1f} kg/ha")
        c4.metric("MAPE", f"{best['MAPE (%)']:.2f}%")
        st.dataframe(table, use_container_width=True, hide_index=True)
    show_image("training_loss.png", "Training and validation curves")
    st.markdown(
        """
**Study area.** Three Cauvery Delta districts, 15 cropland-weighted 10 km boxes each.

**Features.** Sentinel-1 VH/VV, NDVI, LSWI, weekly rainfall, and temperature.

**Split.** Train 2015–2021, validation 2022–2023, test 2024–2025.

Satellite and weather series used for this run are synthetic stand-ins with the same schema as the Google Earth Engine collector, so the pipeline can be reproduced without an Earth Engine account. Yield values follow the Tamil Nadu statistics file in `data/yield.csv`.
        """
    )

elif page == "Model comparison":
    source = comparison or {}
    if not source and full_eval:
        source = {
            k: v["metrics"]
            for k, v in full_eval.items()
            if isinstance(v, dict) and "metrics" in v
        }
    table = metric_table(source)
    if table is not None:
        st.subheader("Test metrics")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.bar_chart(table.set_index("Model")[["R²"]])
    show_image("model_comparison.png", "R², RMSE, and MAPE")
    show_image("actual_vs_predicted.png", "Actual versus predicted yield")

elif page == "District and season":
    district = load_json("district_evaluation.json") or (combined or {}).get("district_wise")
    season = load_json("season_evaluation.json") or (combined or {}).get("season_wise")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("By district")
        table = metric_table(district or {})
        if table is not None:
            table = table.rename(columns={"Model": "District"})
            st.dataframe(table, use_container_width=True, hide_index=True)
        show_image("district_performance.png", "District-wise performance")
    with col2:
        st.subheader("By season")
        table = metric_table(season or {})
        if table is not None:
            table = table.rename(columns={"Model": "Season"})
            st.dataframe(table, use_container_width=True, hide_index=True)
        show_image("season_performance.png", "Season-wise performance")
    show_image("spatial_yield_map.png", "Sub-district spatial yield map")

elif page == "Early forecast":
    forecast = load_json("early_forecast.json") or (combined or {}).get("early_forecast")
    if forecast:
        rows = []
        for key, metrics in forecast.items():
            week = int(key.split("_")[1])
            rows.append({"Week": week, **metrics})
        frame = pd.DataFrame(rows).sort_values("Week")
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.line_chart(frame.set_index("Week")[["R2", "RMSE"]])
        if 12 in set(frame["Week"]):
            row = frame.loc[frame["Week"] == 12].iloc[0]
            st.info(
                f"Week 12 operational milestone: R² = {row['R2']:.3f}, "
                f"RMSE = {row['RMSE']:.1f} kg/ha."
            )
    show_image("early_forecast.png", "Accuracy by forecast week")

elif page == "Ablation":
    ablation = load_json("ablation_study.json") or (combined or {}).get("ablation")
    table = metric_table(ablation or {})
    if table is not None:
        table = table.rename(columns={"Model": "Configuration"})
        st.dataframe(table, use_container_width=True, hide_index=True)
    show_image("ablation_study.png", "Ablation study")
    st.markdown(
        "Ablations zero the SAR channels, replace cropland weights with uniform weights, "
        "or swap temporal attention pooling for a mean pool."
    )

else:
    if not full_eval or "actuals" not in full_eval:
        st.warning("Prediction records were not saved.")
    else:
        meta = full_eval.get("meta", [])
        actuals = full_eval["actuals"]
        model_names = [
            name
            for name, value in full_eval.items()
            if isinstance(value, dict) and "predictions" in value
        ]
        records = []
        for i, actual in enumerate(actuals):
            year, season, district = meta[i] if i < len(meta) else ("", "", "")
            row = {
                "Year": year,
                "Season": season,
                "District": district,
                "Actual (kg/ha)": round(actual, 1),
            }
            for name in model_names:
                preds = full_eval[name]["predictions"]
                if i < len(preds):
                    row[name] = round(preds[i], 1)
            records.append(row)
        frame = pd.DataFrame(records)
        districts = ["All"] + sorted(frame["District"].dropna().unique().tolist())
        seasons = ["All"] + sorted(frame["Season"].dropna().unique().tolist())
        c1, c2 = st.columns(2)
        district = c1.selectbox("District", districts)
        season = c2.selectbox("Season", seasons)
        view = frame
        if district != "All":
            view = view[view["District"] == district]
        if season != "All":
            view = view[view["Season"] == season]
        st.dataframe(view, use_container_width=True, hide_index=True)
        if "AW-MIL Transformer" in view.columns and len(view):
            st.scatter_chart(
                view,
                x="Actual (kg/ha)",
                y="AW-MIL Transformer",
            )
