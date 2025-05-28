"""Main logic for streamlit dashboard application."""

from pathlib import Path
import streamlit as st
from PIL import Image
from load_data import load_occupancy_csv, load_food_beverage_csv

# Tabs
from tabs.occupancy_tab import occupancy_tab_logic
from tabs.fnb_bi import food_beverage_bi_logic
from tabs.fnb_forecast_lineplot import fnb_forecast_lineplot_logic
from tabs.fnb_forecast_barplot import fnb_forecast_barplot_logic

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
    food_beverage_bi_logic(food_beverage_historic_data=fnb_hist)

# === FOOD & BEVERAGE FORECAST ===
with fnb_tab_forecast:
    fnb_forecast_barplot_logic(food_beverage_forecast_data=fnb_fcst)

# === FOOD & BEVERAGE FORECAST 2 ===
with fnb_tab_forecast2:
    fnb_forecast_lineplot_logic(
        food_beverage_historic_data=fnb_hist, food_beverage_forecast_data=fnb_fcst
    )


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
