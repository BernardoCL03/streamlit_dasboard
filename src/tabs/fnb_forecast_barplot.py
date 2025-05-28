"""Logic for the Food & Beverage forecast bar plot tab"""

from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def fnb_forecast_barplot_logic(food_beverage_forecast_data: pd.DataFrame):
    """
    Lógica para el tab de pronóstico de platillos (gráfico de barras).

    Args:
        food_beverage_forecast_data: Datos pronosticados de F&B.
    """
    st.subheader("🍽️ Pronóstico de consumo de platillos de los siguientes 30 días")

    # Filtros y controles
    top_row = st.columns((3, 1), gap="medium")
    fc_min, fc_max = (
        food_beverage_forecast_data["date"].min(),
        food_beverage_forecast_data["date"].max(),
    )

    top_row = st.columns((3, 1), gap="medium")
    with top_row[0]:
        fc_range = st.slider(
            "Forecast window",
            min_value=fc_min.date(),
            max_value=fc_max.date(),
            value=(fc_min.date(), fc_max.date()),
            step=timedelta(days=1),
            format="YYYY-MM-DD",
            key="fc_date",
        )
    with top_row[1]:
        fc_metric = st.radio(
            "Metric",
            options=("Unidades", "Ganancias"),
            horizontal=True,
            key="fc_metric",
        )

    # Seleccionador de platillos
    with st.expander("✚ Filtrar por ID de platillo", expanded=False):
        all_fc_dishes = sorted(food_beverage_forecast_data["dish_id"].unique())
        sel_fc_dishes = st.multiselect(
            "Tick one or more dishes",
            options=all_fc_dishes,
            default=all_fc_dishes,
            placeholder="Type to search…",
            key="fc_dish",
        )

    # Filtrado de datos segun rango de fechas y platillos seleccionados
    start_fc, end_fc = map(pd.Timestamp, fc_range)
    fc_filt = food_beverage_forecast_data[
        (food_beverage_forecast_data["date"].between(start_fc, end_fc))
        & (food_beverage_forecast_data["dish_id"].isin(sel_fc_dishes))
    ]

    # Métricas atractivas para TCA
    m = st.columns(2)
    m[0].metric(
        "Unidades pronosticadas a vender", f"{int(fc_filt['units_sold'].sum()):,}"
    )
    m[1].metric("Ganancias pronosticadas", f"${fc_filt['profit'].sum():,.0f} MXN")

    # Agregaciones por platillo
    if fc_metric == "Unidades":
        fc_agg = (
            fc_filt.groupby("dish_id")["units_sold"]
            .sum()
            .sort_values(ascending=False)  # ← fixed
        )
        y_title = "Unidades"
    else:
        fc_agg = (
            fc_filt.groupby("dish_id")["profit"]
            .sum()
            .sort_values(ascending=False)  # ← fixed
        )
        y_title = "Ganancias (MXN)"

    # Top N platillos pronosticados
    raw_sizes = [5, 10, 20, 50, len(fc_agg)]
    sizes = sorted(set(raw_sizes))
    labels = [("Todos" if n == len(fc_agg) else str(n)) for n in sizes]
    label_to_val = dict(zip(labels, sizes))
    chosen_label = st.selectbox(
        "Mostrar top …",
        labels,
        index=labels.index("All") if "All" in labels else len(labels) - 1,
        key="fc_top_n",
    )
    top_n = label_to_val[chosen_label]
    fc_slice = fc_agg.head(top_n)

    # --- layout (chart + dataframe) -------------------------------
    chart_col, table_col = st.columns((3, 2), gap="large")

    # horizontal bar
    with chart_col:
        fig_fc = go.Figure(
            go.Bar(
                x=fc_slice.values,
                y=fc_slice.index.astype(str),
                orientation="h",
                text=[
                    f"${v:,.0f}" if fc_metric == "Ganancias" else f"{v:,d}"
                    for v in fc_slice.values
                ],
                textposition="auto",
            )
        )
        fig_fc.update_yaxes(autorange="reversed")
        fig_fc.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=0, t=25, b=10),
            xaxis_title=y_title,
            yaxis_title="ID de platillo",
            title=f"Top {top_n if top_n!=len(fc_agg) else 'todos'} platillos pronosticados por: {fc_metric}",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    # Dataframe con predicciones de ventas
    # TODO: posiblemente agregar un botón para agrupar los platillos
    # por 'fc_metric', para que sepan de los pronósicos de ventas y ganancias.
    with table_col:
        disp_cols = [
            "date",
            "dish_id",
            "units_sold" if fc_metric == "Unidades" else "profit",
        ]
        df_disp = fc_filt[disp_cols].sort_values(["date", "dish_id"])
        st.dataframe(
            df_disp,
            hide_index=True,
            height=450,
            column_config={
                "units_sold": st.column_config.NumberColumn(format="%d"),
                "profit": st.column_config.NumberColumn(format="$%.0f"),
            },
        )
