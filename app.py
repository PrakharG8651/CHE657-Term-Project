"""Small Streamlit front end for uploaded single-reactant time-course data."""
from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st
from scipy.interpolate import make_smoothing_spline

from rate_law_discovery import _preprocess_group, discover_rate_law, integrate_rate_law, predict_rate


st.set_page_config(page_title="Rate Law Discovery", page_icon="⚗️", layout="wide")
st.title("Single-Reactant Rate Law Discovery")
st.write("Upload concentration-versus-time measurements. The analysis assumes a batch system and reports a law of the form **r = −dCₐ/dt = f(Cₐ)**.")
st.info("PySR fits models with raw, Savitzky–Golay, and GCV preprocessing. The site simplifies and refits algebraic candidates, then compares them by held-out concentration-trajectory reconstruction with a one-standard-error preference for simpler equations. First use can take several minutes while the Julia backend initializes.")
st.markdown("**Recommended CSV column order:** `time_min, CA, trajectory_id` — one observation per row. `trajectory_id` is optional for one run; with multiple runs, repeat the ID on every row in that run. Column names and order can vary because you select them below.")
st.code("time_min,CA,trajectory_id\n0.0,0.0500,run_1\n0.25,0.0488,run_1\n0.0,0.0750,run_2\n0.25,0.0724,run_2", language="csv")

file = st.file_uploader("Trajectory data (CSV)", type=["csv"])
if file is None:
    st.caption("Required columns: time and concentration. Optional trajectory/group column supports multiple runs or replicates.")
    st.stop()

try:
    data = pd.read_csv(file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

columns = list(data.columns)
if len(columns) < 2:
    st.error("CSV needs at least a time column and a concentration column.")
    st.stop()

left, right = st.columns(2)
time_col = left.selectbox("Time column", columns, index=0)
ca_options = [c for c in columns if c != time_col]
ca_col = right.selectbox("Concentration column", ca_options, index=0)
group_options = ["(single trajectory)"] + [c for c in columns if c not in (time_col, ca_col)]
group_choice = st.selectbox("Trajectory / replicate column (optional)", group_options)
group_col = None if group_choice == "(single trajectory)" else group_choice

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20), use_container_width=True)

if st.button("Discover rate law", type="primary"):
    try:
        with st.spinner("Smoothing trajectories and searching equations with PySR…"):
            result = discover_rate_law(
                data,
                time_col=time_col,
                ca_col=ca_col,
                trajectory_col=group_col,
            )
        st.session_state["discovery_result"] = result.to_dict()
    except Exception as exc:
        st.error(str(exc))

if "discovery_result" in st.session_state:
    result = st.session_state["discovery_result"]
    st.subheader("Discovered equation")
    equation_latex = sp.latex(sp.sympify(result["equation"])).replace("x_{0}", "C_A")
    st.latex(r"r=-\frac{dC_A}{dt}=" + equation_latex)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rate RMSE", f"{result['rmse_rate']:.4g}")
    m2.metric("Rate MAE", f"{result['mae_rate']:.4g}")
    m3.metric("Rate R²", f"{result['r_squared_rate']:.4f}")
    m4.metric("Validation trajectory RMSE", f"{result['validation_trajectory_rmse']:.4g}")
    st.caption(f"Selected preprocessing: {result['method']}. Validation: {result['validation_scope']}. Fit used {result['observations']} observations across {result['trajectories']} trajectories over CA = {result['concentration_range'][0]:.4g}–{result['concentration_range'][1]:.4g}.")
    for warning in result["warnings"]:
        st.warning(warning)
    st.subheader("Candidate equations")
    st.dataframe(pd.DataFrame(result["candidates"]), use_container_width=True)

    st.subheader("Data and model plots")
    plot_data = data[[time_col, ca_col] + ([group_col] if group_col else [])].dropna().copy()
    plot_data = plot_data.rename(columns={time_col: "_time", ca_col: "_ca"})
    if group_col:
        plot_data = plot_data.rename(columns={group_col: "_trajectory"})
    else:
        plot_data["_trajectory"] = "trajectory_1"
    fitted_groups = []
    for trajectory, group in plot_data.groupby("_trajectory", sort=False):
        group = group.sort_values("_time").copy()
        t = group["_time"].to_numpy(float)
        observed_ca = group["_ca"].to_numpy(float)
        if len(t) < 5 or np.any(np.diff(t) <= 0):
            continue
        spline = make_smoothing_spline(t, observed_ca, lam=None)
        group["_CA_fit"] = spline(t)
        group["_rate_fit"] = -spline.derivative()(t)
        group["_label"] = str(trajectory)
        fitted_groups.append(group)
    if fitted_groups:
        plotted = pd.concat(fitted_groups, ignore_index=True)
        selected_groups = pd.concat(
            [_preprocess_group(group, result["method"]) for _, group in plot_data.groupby("_trajectory", sort=False)],
            ignore_index=True,
        )
        chart1, chart2 = st.tabs(["Concentration vs time", "Rate vs concentration"])
        with chart1:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for label, group in plotted.groupby("_label", sort=False):
                group = group.sort_values("_time")
                ax.scatter(group["_time"], group["_ca"], s=14, alpha=0.45, label=f"{label}: measured")
                ax.plot(group["_time"], group["_CA_fit"], linewidth=2, label=f"{label}: smoothed")
            ax.set(xlabel=f"Time ({time_col} units)", ylabel=f"Concentration ({ca_col} units)", title="Measured and smoothed trajectories")
            ax.grid(alpha=0.25)
            ax.legend(fontsize="small", ncol=2)
            st.pyplot(fig, use_container_width=True)
        with chart2:
            ca_values = selected_groups["CA_fit"].to_numpy(float)
            rate_values = selected_groups["rate_fit"].to_numpy(float)
            valid = selected_groups["_valid"].to_numpy(bool) & (ca_values > 0) & np.isfinite(ca_values) & np.isfinite(rate_values)
            if not np.any(valid):
                st.warning("No positive finite smoothed concentrations were available for the rate plot.")
            else:
                x_grid = np.linspace(max(0.0, float(np.min(ca_values[valid]))), float(np.max(ca_values[valid])), 300)
                try:
                    y_grid = predict_rate(result["equation"], x_grid)
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    ax.scatter(ca_values[valid], rate_values[valid], s=16, alpha=0.45, label=f"{result['method']} rate estimates")
                    ax.plot(x_grid, y_grid, color="crimson", linewidth=2.5, label="Simplified PySR law")
                    ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
                    ax.set(xlabel=f"Concentration ({ca_col} units)", ylabel="Rate (concentration unit / time unit)", title="Observed rate estimates and discovered law")
                    ax.grid(alpha=0.25)
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not plot the fitted rate law: {exc}")

        st.subheader("Analysis summary")
        median_rate = float(np.nanmedian(selected_groups.loc[selected_groups["_valid"], "rate_fit"]))
        low_rate = float(predict_rate(result["equation"], result["concentration_range"][0]))
        high_rate = float(predict_rate(result["equation"], result["concentration_range"][1]))
        direction = "increases" if high_rate > low_rate else "decreases" if high_rate < low_rate else "stays approximately constant"
        summary_cols = st.columns(3)
        summary_cols[0].metric("Trajectories analyzed", result["trajectories"])
        summary_cols[1].metric("Median estimated rate", f"{median_rate:.4g}")
        summary_cols[2].metric("Trend across observed CA range", direction)
        st.markdown(
            f"The fitted equation describes rate as a function of **CA alone** over the observed concentration interval "
            f"**{result['concentration_range'][0]:.4g} to {result['concentration_range'][1]:.4g}**. "
            f"Across that interval, the equation's rate {direction} ({low_rate:.4g} to {high_rate:.4g}). "
            f"The rate-fit R² is **{result['r_squared_rate']:.3f}** on training data. Held-out trajectory RMSE is **{result['validation_trajectory_rmse']:.4g}** ({result['validation_scope']})."
        )
    else:
        st.warning("Could not draw trajectory plots: each selected trajectory needs at least five rows with increasing time values.")

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", file.name.rsplit(".", 1)[0]) or "dataset"
    st.download_button(
        "Download equation and metrics (JSON)",
        data=json.dumps(result, indent=2),
        file_name=f"{safe_name}_rate_law.json",
        mime="application/json",
    )

    st.subheader("Simulate concentration from the discovered law")
    c1, c2 = st.columns(2)
    initial_ca = c1.number_input("Initial CA", min_value=0.0, value=float(data[ca_col].dropna().iloc[0]))
    duration = c2.number_input("Prediction duration", min_value=0.001, value=float(data[time_col].max() - data[time_col].min()))
    if st.button("Plot predicted trajectory"):
        time_grid = pd.Series(data[time_col]).dropna().sort_values().unique()
        time_grid = time_grid.astype(float)
        # Start the integration at time zero for a conventional batch-trajectory display.
        if len(time_grid) < 2 or time_grid[0] < 0:
            st.error("Trajectory plot needs at least two nonnegative time values.")
        else:
            pred_time = __import__("numpy").linspace(float(time_grid[0]), float(time_grid[0] + duration), 200)
            try:
                pred_ca = integrate_rate_law(result["equation"], pred_time, initial_ca)
                fig, ax = plt.subplots()
                ax.plot(pred_time, pred_ca, label="Predicted CA")
                ax.set(xlabel=f"Time ({time_col} units)", ylabel=f"Concentration ({ca_col} units)")
                ax.grid(alpha=0.25)
                ax.legend()
                st.pyplot(fig)
            except Exception as exc:
                st.error(f"Could not integrate this equation: {exc}")
