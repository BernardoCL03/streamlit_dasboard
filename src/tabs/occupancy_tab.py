"""Logic for the occupancy tab (historic vs. forecasted occupancy)."""

from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from helpers import resample_df


def occupancy_tab_logic(historic_data: pd.DataFrame, forecast_data: pd.DataFrame):
    # st.subheader("Ocupación real + predicción superpuesta")

    # Rango máximo y mínimo de las fechas (historic + forecast)
    full_min, full_max = historic_data["date"].min(), forecast_data["date"].max()
    fcst_min, fcst_max = forecast_data["date"].min(), forecast_data["date"].max()

    default_fcst_end = min(fcst_min + timedelta(days=29), fcst_max)
    slider_row = st.columns((1, 1), gap="medium")

    with slider_row[0]:
        st.subheader("Histórico + predicciones")
        range_all = st.slider(
            label="Esto no se ve",
            label_visibility="collapsed",  # sin visibilidad
            min_value=full_min.date(),
            max_value=full_max.date(),
            value=(full_min.date(), full_max.date()),
            step=timedelta(days=1),
            format="YYYY-MM-DD",
            key="range_all",
        )

    with slider_row[1]:
        st.subheader("Ventada de predicciones")
        range_fcst = st.slider(
            label="Esto tampoco",
            label_visibility="collapsed",
            min_value=fcst_min.date(),
            max_value=fcst_max.date(),
            value=(fcst_min.date(), default_fcst_end.date()),
            step=timedelta(days=1),
            format="YYYY-MM-DD",
            key="range_fcst",
        )

    # === SUBSET de DATOS ===
    granularity_map = {
        "Diaria": "D",  # Datos crudos
        "Semanal": "W-MON",  # Semana que comienza el lunes
        "Mensual": "M",  # Fin de mes
    }

    gran_choice = st.radio(
        "Granularidad",
        options=("Diaria", "Semanal", "Mensual"),  # Debe matchear con granularity_map
        horizontal=True,
        index=0,  # default = Daily
        key="gran_all",
    )

    # Convert to Timestamp just once
    start_all, end_all = map(pd.Timestamp, range_all)
    start_fcst, end_fcst = map(pd.Timestamp, range_fcst)

    # Filtros base (todavía diario)
    hist_filt = historic_data[(historic_data["date"].between(start_all, end_all))]
    fcst_filt = historic_data[(historic_data["date"].between(start_all, end_all))]
    fcst_zoom = historic_data[(historic_data["date"].between(start_fcst, end_fcst))]

    # Aplicamos la misma regla a los 3
    rule = granularity_map[gran_choice]
    hist_sel = resample_df(hist_filt, rule)
    fcst_sel = resample_df(fcst_filt, rule)
    fcst_only = resample_df(fcst_zoom, rule)

    chart_col = st.columns((3, 2), gap="medium")

    # Métricas atractivas para los TCA
    metric_col = st.columns(2)
    with metric_col[0]:
        st.metric(
            "Huéspedes totales",
            f"{hist_sel['guests'].sum():,}",
            help="Número total de huéspedes en el periodo seleccionado.",
        )

    with metric_col[1]:
        st.metric(
            "Huéspedes pronosticados",
            f"{fcst_only['guests'].sum():,}",
            help="Número total de huéspedes pronosticados en el periodo seleccionado.",
        )

    # === GRÁFICAS ===
    chart_col = st.columns((3, 2), gap="medium")

    # 1) Hisórico + Predicciones
    with chart_col[0]:
        st.subheader("Ocupación histórica con pronóstico de 30 días")

        fig = go.Figure()

        # Línea historica
        fig.add_trace(
            go.Scatter(
                x=hist_sel["date"],
                y=hist_sel["guests"],
                mode="lines",
                name="Real",
                hovertemplate="%{x|%b %d, %Y}<br>Guests: %{y:,}<extra></extra>",
            )
        )

        # Línea de pronóstico
        fig.add_trace(
            go.Scatter(
                x=fcst_sel["date"],
                y=fcst_sel["guests"],
                mode="lines",
                name="Pronosticados",
                line=dict(dash="dot"),
                hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:,}<extra></extra>",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=25, b=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            xaxis_title="Fecha",
            yaxis_title="Número de huéspedes",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2) Solamente predicciones
    with chart_col[1]:
        st.subheader("Pronóstico de ocupación (zoom)")

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=fcst_only["date"],
                y=fcst_only["guests"],
                mode="lines+markers",
                name="Pronosticados",
                hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:,}<extra></extra>",
            )
        )
        fig2.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Fecha",
            yaxis_title="Número de huéspedes",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
