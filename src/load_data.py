"""Módulo para cargar y preparar datos de CSVs para Streamlit.
Se utiliza cache para que no se tenga que cargar cada vez que se actualiza la app."""

import streamlit as st
import pandas as pd
from pathlib import Path


@st.cache_data(show_spinner="Cargando datos…")
def load_occupancy_csv(csv_path: str | Path, parse_dates: list[str]) -> pd.DataFrame:
    """Lee y cachea el CSV especificado."""
    return pd.read_csv(csv_path, parse_dates=parse_dates)


@st.cache_data(show_spinner="Cargando datos de F&B")
def load_food_beverage_csv(
    csv_path: str | Path, parse_dates: list[str]
) -> pd.DataFrame:
    """Lee y cachea el CSV de F&B especificado."""
    return pd.read_csv(csv_path, parse_dates=parse_dates)
