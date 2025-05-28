"""Logic for the Food & Beverage forecast line plot tab."""

from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from helpers import resample_df_column


def fnb_forecast_lineplot_logic(
    food_beverage_historic_data: pd.DataFrame, food_beverage_forecast_data: pd.DataFrame
):
    """
    Lógica para el tab de pronóstico de platillos (líneas de tiempo).

    Args:
        food_beverage_historic_data: Datos históricos de F&B.
        food_beverage_forecast_data: Datos pronosticados de F&B.
    """
    st.subheader("Histórico vs. pronóstico — Dish Lines")

    # Filtros y controles
    min_all, max_all = (
        food_beverage_historic_data["date"].min(),
        food_beverage_forecast_data["date"].max(),
    )

    ctrl_row1 = st.columns((3, 1, 1), gap="medium")
    with ctrl_row1[0]:
        line_range = st.slider(
            label="Rango de fechas",
            min_value=min_all.date(),
            max_value=max_all.date(),
            value=(min_all.date(), max_all.date()),
            step=timedelta(days=1),
            format="YYYY-MM-DD",
            key="line_date",
        )
    with ctrl_row1[1]:
        line_metric = st.radio(
            "Métrica",
            options=("Unidades", "Ganancias"),
            horizontal=True,
            key="line_metric",
        )
    with ctrl_row1[2]:
        granularity = st.radio(
            "Granularidad",
            options=("Diaria", "Semanal"),
            horizontal=True,
            key="line_gran",
        )

    # Seleccionador de platillos
    with st.expander("✚ Filtrar por ID de platillo", expanded=False):
        dishes_all = sorted(
            set(food_beverage_historic_data["dish_id"]).union(
                food_beverage_forecast_data["dish_id"]
            )
        )
        dishes_sel = st.multiselect(
            "Elige platillos de tu interés",
            options=dishes_all,
            default=dishes_all[:6],  # primeros 6
            placeholder="Type to search…",
            key="line_dish",
        )

    # Filtrado de datos según rango de fechas
    start_ts, end_ts = map(pd.Timestamp, line_range)

    hist_filt = food_beverage_historic_data[
        food_beverage_historic_data["dish_id"].isin(dishes_sel)
    ].loc[lambda d: d["date"].between(start_ts, end_ts)]
    fcst_filt = food_beverage_forecast_data[
        food_beverage_forecast_data["dish_id"].isin(dishes_sel)
    ].loc[lambda d: d["date"].between(start_ts, end_ts)]

    # Hacemos resampling según métrica y granularidad
    rule = "D" if granularity == "Diaria" else "W-MON"
    val_col = "units_sold" if line_metric == "Unidades" else "profit"
    y_title = "Units" if line_metric == "Unidades" else "Profit ($)"
    hover_fmt = ",d" if line_metric == "Unidades" else ",.0f"

    hist_rs = resample_df_column(hist_filt, rule, val_col)
    fcst_rs = resample_df_column(fcst_filt, rule, val_col)

    # ── build line traces ------------------------------------------
    fig = go.Figure()

    for dish in dishes_sel:
        # historic trace
        hd = (
            hist_rs[hist_rs["dish_id"] == dish]
            if "dish_id" in hist_rs
            else hist_rs.assign(dish_id=hist_filt["dish_id"].iloc[0])
        )
        fd = (
            fcst_rs[fcst_rs["dish_id"] == dish]
            if "dish_id" in fcst_rs
            else fcst_rs.assign(dish_id=fcst_filt["dish_id"].iloc[0])
        )

        fig.add_trace(
            go.Scatter(
                x=hd["date"],
                y=hd[val_col],
                mode="lines",
                name=f"{dish} (hist)",
                hovertemplate=f"{dish}<br>%{{x|%Y-%m-%d}}<br>%{{y:{hover_fmt}}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fd["date"],
                y=fd[val_col],
                mode="lines",
                name=f"{dish} (fcst)",
                line=dict(dash="dot"),
                hovertemplate=f"{dish}<br>%{{x|%Y-%m-%d}}<br>%{{y:{hover_fmt}}}<extra></extra>",
            )
        )

    # Gráfico de líneas!
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Date",
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
