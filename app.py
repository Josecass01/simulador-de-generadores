import streamlit as st
import pandas as pd

from config_panel import render_sidebar
from view import process_data, render_results

st.set_page_config(page_title="Generador de Números Aleatorios", layout="wide")

# ========= Session State =========
if "numbers" not in st.session_state:
    st.session_state["numbers"] = []
if "generation_df" not in st.session_state:
    st.session_state["generation_df"] = None

# ===== UI principal =====
st.title("Resultados")

# ===== Sidebar (config + botones) =====
params, run_clicked, clear_clicked = render_sidebar()

# ===== Acciones =====
if clear_clicked:
    st.session_state["numbers"] = []
    st.session_state["generation_df"] = None
    st.success("Se limpiaron los resultados.")
    st.stop()

if run_clicked:
    numbers, generation_df = process_data(params)
    st.session_state["numbers"] = numbers
    st.session_state["generation_df"] = generation_df

# ===== Mostrar resultados persistidos =====
numbers = st.session_state["numbers"]
generation_df = st.session_state["generation_df"]

render_results(numbers=numbers,
               generation_df=generation_df,
               alpha=params.get("alpha", 0.05),
               expected_mean=params.get("expected_mean", 0.5))
