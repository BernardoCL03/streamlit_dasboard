"""Main logic for streamlit dashboard application."""

from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from PIL import Image
from load_data import load_occupancy_csv, load_food_beverage_csv
from helpers import resample_df_column

# Tabs
from tabs.occupancy_tab import occupancy_tab_logic

repo_path = Path(__file__).resolve().parent.parent
web_icon = Image.open(f"{repo_path}/data/page_icon.webp")
st.set_page_config(page_title="Dashboard TCA", page_icon=web_icon, layout="wide")
st.markdown(
    """
<style>
/* tighten the padding around metrics (flash-cards) */
[data-testid="metric-container"] {
    padding: 10px 15px;
}
</style>
""",
    unsafe_allow_html=True,
)

# === CARGA DE DATOS ===
try:
    # Ocuppancy
    hist = load_occupancy_csv(
        f"{repo_path}/data/hotel_occupancy_actual.csv", parse_dates=["date"]
    )
    fcst = load_occupancy_csv(
        f"{repo_path}/data/hotel_occupancy_forecast.csv", parse_dates=["date"]
    )
    # Food % Beverage
    fnb_hist = load_food_beverage_csv(
        f"{repo_path}/data/fnb_sales_actual.csv", parse_dates=["date"]
    )
    fnb_fcst = load_food_beverage_csv(
        f"{repo_path}/data/fnb_sales_forecast.csv", parse_dates=["date"]
    )
except FileNotFoundError:
    st.error(
        "No se encontraron los archivos. Coloca los archivos en la carpeta 'data/'."
    )
    st.stop()


# === STREAMLIT UI ===
st.title("OUMAJI MVP Dashboard")

occupancy_tab, fnb_tab_bi, fnb_tab_forecast, fnb_tab_forecast2 = st.tabs(
    [
        "📈  Ocupación en el tiempo",
        "📊  F&B BI",
        "🍔  F&B Forecast I",
        "🍕  F&B Forecast II",
    ]
)

# === OCCUPANCY ===™
with occupancy_tab:
    occupancy_tab_logic(historic_data=hist, forecast_data=fcst)

# === FOOD & BEVERAGE BI ===
with fnb_tab_bi:

    fnb_min, fnb_max = fnb_hist["date"].min(), fnb_hist["date"].max()
    ctrl1, ctrl2 = st.columns((3, 1), gap="medium")

    # Filtros
    with ctrl1:
        date_range = st.slider(
            "Rango de fechas",
            min_value=fnb_min.date(),
            max_value=fnb_max.date(),
            value=(fnb_min.date(), fnb_max.date()),
            step=timedelta(days=1),
            format="YYYY-MM-DD",
            key="fnb_date",
        )

    with ctrl2:
        metric_type = st.radio(
            "Métrica",
            options=("Unidades vendidas", "Ganancias"),
            horizontal=True,
            key="fnb_metric",
        )

    # Seleccionador de platillos
    all_dishes = sorted(fnb_hist["dish_id"].unique())
    with st.expander("✚ Filtrar por ID de platillo", expanded=False):
        sel_dishes = st.multiselect(
            "Elige platillos de tu interés",
            options=all_dishes,
            default=all_dishes,
            placeholder="Type to search…",
            key="fnb_dish",
        )

    # Filtrado de datos segun rango de fechas y platillos seleccionados
    start_fnb, end_fnb = map(pd.Timestamp, date_range)
    filt = fnb_hist[
        (fnb_hist["date"].between(start_fnb, end_fnb))
        & (fnb_hist["dish_id"].isin(sel_dishes))
    ]

    # Métricas atractivas para TCA
    mcol = st.columns(2)
    mcol[0].metric("Total de unidades vendidas", f"{int(filt['units_sold'].sum()):,}")
    mcol[1].metric("Ganancias totales", f"${filt['profit'].sum():,.0f} MXN")

    if metric_type == "Unidades vendidas":
        dish_agg = (
            filt.groupby("dish_id")["units_sold"].sum().sort_values(ascending=False)
        )

    else:
        dish_agg = filt.groupby("dish_id")["profit"].sum().sort_values(ascending=False)

    # Enumeración de opciones para el top N
    raw_sizes = [5, 10, 20, 50, len(dish_agg)]
    sizes = sorted(set(raw_sizes))

    labels = [("Todos" if n == len(dish_agg) else str(n)) for n in sizes]
    label_to_value = dict(zip(labels, sizes))

    chosen_label = st.selectbox(
        "Mostrar top …",
        options=labels,
        index=labels.index("Todos") if "Todos" in labels else len(labels) - 1,
        key="fnb_top_n",
    )
    top_n = label_to_value[chosen_label]
    dish_slice = dish_agg.head(
        top_n
    )  # Top N platillos más vendidos o con más ganancias
    top_dishes = dish_slice.index.tolist()  # lista de IDs a usar en ambos charts

    # Paleta de colores que se repite (24)
    palette = px.colors.qualitative.Light24
    color_map = {
        dish_id: palette[i % len(palette)] for i, dish_id in enumerate(top_dishes)
    }

    # === GRÁFICAS ===
    ch_col = st.columns((3, 2), gap="large")

    # Gráfico de barras por platillo venta o ganancias
    with ch_col[0]:
        st.subheader(f"Top {top_n} platillos por: {metric_type}")
        fig_dish = go.Figure()
        for platillo in top_dishes:
            fig_dish.add_trace(
                go.Bar(
                    x=[dish_slice[platillo]],
                    y=[str(platillo)],
                    orientation="h",
                    marker_color=color_map[platillo],
                    name=str(platillo),
                    text=[
                        (
                            f"${dish_slice[platillo]:,.0f}"
                            if metric_type == "Ganancias"
                            else f"{dish_slice[platillo]:,d}"
                        )
                    ],
                    textposition="auto",
                    showlegend=False,  # evita leyenda redundante
                )
            )

        fig_dish.update_yaxes(autorange="reversed")
        fig_dish.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=0, t=25, b=10),
            xaxis_title=(
                "Ganancias (MXN)" if metric_type == "Ganancias" else "Unidades vendidas"
            ),
            yaxis_title="ID de platillo",
        )
        st.plotly_chart(fig_dish, use_container_width=True)

    # Gráfico de barras apilado por día de la semana
    with ch_col[1]:
        st.subheader("Distribución por día de la semana")

        # Checkbox para normalizar
        pct_mode = st.checkbox(
            "Mostrar cada día como porcentaje (100 %)", value=False, key="fnb_pct"
        )

        # Preparar dataframe solo con Top N platillos
        filt_top = filt[filt["dish_id"].isin(top_dishes)].copy()
        filt_top["weekday"] = filt_top["date"].dt.day_name()

        # Agregado por día y platillo
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        if metric_type == "Unidades vendidas":
            grouped = (
                filt_top.groupby(["weekday", "dish_id"])["units_sold"]
                .sum()
                .unstack(fill_value=0)
            )
        else:
            grouped = (
                filt_top.groupby(["weekday", "dish_id"])["profit"]
                .sum()
                .unstack(fill_value=0)
            )
        grouped = grouped.reindex(weekday_order).fillna(0)  # aplica orden de días
        # Ordena columnas en el mismo orden de color_map
        grouped = grouped[top_dishes]

        # Normaliza a porcentaje si se pidió
        if pct_mode:
            grouped_pct = grouped.div(grouped.sum(axis=1), axis=0).fillna(0)
            data_to_plot = grouped_pct
            yaxis_title = "Proporción"
            hover_fmt = ".1%"
        else:
            data_to_plot = grouped
            yaxis_title = (
                "Ganancias (MXN)" if metric_type == "Ganancias" else "Unidades vendidas"
            )
            hover_fmt = ",.0f" if metric_type == "Ganancias" else ",d"

        # Aquí se construye el gráfico de barras apilado!
        fig_week = go.Figure()
        for platillo in top_dishes:
            fig_week.add_trace(
                go.Bar(
                    x=data_to_plot.index,  # Monday … Sunday
                    y=data_to_plot[platillo],
                    name=str(platillo),
                    marker_color=color_map[platillo],
                    hovertemplate=f"%{{x}}<br>ID {platillo}: %{{y:{hover_fmt}}}<extra></extra>",
                )
            )

        fig_week.update_layout(
            barmode="stack",
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Día de la semana",
            yaxis_title=yaxis_title,
            legend_title="Platillo",
        )

        st.plotly_chart(fig_week, use_container_width=True)


# === FOOD & BEVERAGE FORECAST ===
with fnb_tab_forecast:
    st.subheader("🍽️ Pronóstico de consumo de platillos de los siguientes 30 días")

    # Filtros y controles
    top_row = st.columns((3, 1), gap="medium")
    fc_min, fc_max = fnb_fcst["date"].min(), fnb_fcst["date"].max()

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
        all_fc_dishes = sorted(fnb_fcst["dish_id"].unique())
        sel_fc_dishes = st.multiselect(
            "Tick one or more dishes",
            options=all_fc_dishes,
            default=all_fc_dishes,
            placeholder="Type to search…",
            key="fc_dish",
        )

    # Filtrado de datos segun rango de fechas y platillos seleccionados
    start_fc, end_fc = map(pd.Timestamp, fc_range)
    fc_filt = fnb_fcst[
        (fnb_fcst["date"].between(start_fc, end_fc))
        & (fnb_fcst["dish_id"].isin(sel_fc_dishes))
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
        y_title, fmt = "Unidades", ",d"
    else:
        fc_agg = (
            fc_filt.groupby("dish_id")["profit"]
            .sum()
            .sort_values(ascending=False)  # ← fixed
        )
        y_title, fmt = "Ganancias (MXN)", ",.0f"

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

# === FOOD & BEVERAGE FORECAST 2 ===
with fnb_tab_forecast2:
    st.subheader("Histórico vs. pronóstico — Dish Lines")

    # Filtros y controles
    min_all, max_all = fnb_hist["date"].min(), fnb_fcst["date"].max()

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
        dishes_all = sorted(set(fnb_hist["dish_id"]).union(fnb_fcst["dish_id"]))
        dishes_sel = st.multiselect(
            "Elige platillos de tu interés",
            options=dishes_all,
            default=dishes_all[:6],  # primeros 6
            placeholder="Type to search…",
            key="line_dish",
        )

    # Filtrado de datos según rango de fechas
    start_ts, end_ts = map(pd.Timestamp, line_range)

    hist_filt = fnb_hist[fnb_hist["dish_id"].isin(dishes_sel)].loc[
        lambda d: d["date"].between(start_ts, end_ts)
    ]
    fcst_filt = fnb_fcst[fnb_fcst["dish_id"].isin(dishes_sel)].loc[
        lambda d: d["date"].between(start_ts, end_ts)
    ]

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


########################
# Pie de página
# with st.expander("About this dashboard", expanded=False):
#     st.markdown(
#         """
# * **Data source**: internal PMS exports for actuals, plus your preferred time-series model for forecasts.
# * **Flash-cards**: summed guests over the selected ranges (you can switch to *average occupancy rate* if that’s more meaningful).
# * **Filters** act instantly; no “apply” button needed thanks to Pandas boolean indexing.
# * **Visuals**: Plotly Dark template for quick PowerBI-like polish; dotted line = future.
# """
#     )
