"""Logic for the Food&Beverage BI tab"""

from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def food_beverage_bi_logic(food_beverage_historic_data: pd.DataFrame):
    fnb_min, fnb_max = (
        food_beverage_historic_data["date"].min(),
        food_beverage_historic_data["date"].max(),
    )
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
    all_dishes = sorted(food_beverage_historic_data["dish_id"].unique())
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
    filt = food_beverage_historic_data[
        (food_beverage_historic_data["date"].between(start_fnb, end_fnb))
        & (food_beverage_historic_data["dish_id"].isin(sel_dishes))
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
