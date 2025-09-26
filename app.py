import streamlit as st
from config_panel import render_config_form
from view import process_data, render_results

st.set_page_config(
    page_title="Taller de Simulación Digital",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========= Inicialización del Estado de la Sesión =========
if "numbers" not in st.session_state:
    st.session_state.numbers = []
if "generation_df" not in st.session_state:
    st.session_state.generation_df = None
if "params" not in st.session_state:
    st.session_state.params = {}

# ===== Títulos y Autores =====
st.title("Taller de Simulación Digital")
st.header("Simulador de Generadores de Números Aleatorios")
st.markdown("---")
st.markdown(
    '**Autores:** <span style="color: #FF4B4B; font-weight: bold;">Jose Luis Castellanos Guardia y Hector Fabian Diaz Cañas</span>',
    unsafe_allow_html=True
)

# ===== Apartado de explicaciones =====
with st.expander("📖 Ver explicación de los métodos"):
    st.subheader("Xorshift32")
    st.markdown("""
    Es un generador moderno, rápido y de alta calidad. No se basa en aritmética tradicional, sino en operaciones de bits (XOR y desplazamientos).
    - **¿Cómo funciona?:** Toma un número (la semilla) y "mezcla" sus bits de una manera compleja pero muy eficiente para una computadora. El resultado de esta mezcla es el siguiente número en la secuencia.
    - **Ventaja:** Produce secuencias muy largas (más de 4 mil millones de números antes de repetirse) y de excelente calidad estadística.
    """)
    st.subheader("Cuadrados Medios")
    st.markdown("""
    Es uno de los primeros algoritmos propuestos (por John von Neumann). Es muy simple y se utiliza principalmente con fines educativos.
    - **¿Cómo funciona?:** Se toma una semilla inicial, se la eleva al cuadrado y se extraen los dígitos del centro del resultado. Este nuevo número es tanto el siguiente número aleatorio de la secuencia como la semilla para la próxima iteración.
    - **Desventaja:** Tiende a degenerar rápidamente, cayendo en ciclos cortos o en el número cero.
    """)
    st.subheader("Bernoulli")
    st.markdown("""
    No es un generador de números aleatorios uniformes, sino un **simulador de una distribución de probabilidad**. Genera el resultado de un único experimento con dos posibles salidas: éxito (1) o fracaso (0).
    - **¿Cómo funciona?:** Primero, genera un número aleatorio `u` entre 0 y 1 (usando un generador interno como LCG). Luego, compara este número con una probabilidad de éxito `p` definida por el usuario. Si `u <= p`, el resultado es 1; de lo contrario, es 0.
    - **Ejemplo:** Simular el lanzamiento de una moneda (donde p=0.5) o la probabilidad de que una pieza sea defectuosa.
    """)

# ===== Panel de Configuración y Botones de Acción =====
params, run_clicked, clear_clicked = render_config_form()

if clear_clicked:
    st.session_state.numbers = []
    st.session_state.generation_df = None
    st.session_state.params = {}
    st.success("Resultados y configuración limpiados.")

if run_clicked:
    st.session_state.params = params
    numbers, generation_df = process_data(params)
    st.session_state.numbers = numbers
    st.session_state.generation_df = generation_df

# ===== Área de Resultados =====
if st.session_state.numbers:
    st.divider()
    st.header("Análisis de Resultados")
    # CAMBIO: Pasamos el nuevo parámetro 'poker_digits'
    render_results(
        numbers=st.session_state.numbers,
        generation_df=st.session_state.generation_df,
        alpha=st.session_state.params.get("alpha", 0.05),
        bins=st.session_state.params.get("bins", 10),
        poker_digits=st.session_state.params.get("poker_digits", 3)
    )
else:
    st.info("Configura los parámetros y haz clic en 'Generar / Procesar' para ver los resultados.")