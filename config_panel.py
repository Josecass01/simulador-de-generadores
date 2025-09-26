import streamlit as st

def render_sidebar():
    st.sidebar.title("Configuración")
    st.sidebar.caption("Generador de Números Aleatorios")

    # Modo
    mode = st.sidebar.radio("Modo de entrada", ["Generar", "Cargar"], index=0, horizontal=True)

    # Parámetros por modo
    params = {"mode": mode}

    if mode == "Generar":
        st.sidebar.subheader("Parámetros de Generación")
        method = st.sidebar.selectbox("Método", ["Cuadrados Medios", "Productos Medios", "Bernoulli"], index=0)
        cantidad = st.sidebar.number_input("Cantidad", min_value=1, max_value=1000, value=10, step=1)

        params["method"] = method
        params["cantidad"] = int(cantidad)

        if method == "Cuadrados Medios":
            params["semilla"] = int(st.sidebar.number_input("Semilla (X₀)", min_value=0, value=5735, step=1))
            params["digitos"] = int(st.sidebar.number_input("Dígitos", min_value=4, max_value=8, value=4, step=1))
        elif method == "Productos Medios":
            params["semilla1"] = int(st.sidebar.number_input("Semilla 1 (X₀)", min_value=0, value=5015, step=1))
            params["semilla2"] = int(st.sidebar.number_input("Semilla 2 (X₁)", min_value=0, value=5734, step=1))
            params["digitos"] = 4   # fijo como en tu código
        else:
            params["p"] = float(st.sidebar.number_input("Probabilidad (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01))
            params["seed_b"] = int(st.sidebar.number_input("Semilla", min_value=0, value=12345, step=1))

        st.sidebar.subheader("Configuración de Pruebas")
        params["alpha"] = float(st.sidebar.number_input("Grado Aceptación (α)", min_value=0.0, max_value=1.0, step=0.01, value=0.05))
        params["expected_mean"] = float(st.sidebar.number_input("Media Esperada", step=0.01, value=0.5))

    else:
        st.sidebar.subheader("Cargar Números")
        params["text"] = st.sidebar.text_area(
            "Números (0..1) separados por comas o líneas", height=200,
            placeholder="0.8902, 0.2456, 0.0319\n0.8902\n0.2456\n0.0319"
        )
        st.sidebar.subheader("Configuración de Pruebas")
        params["alpha"] = float(st.sidebar.number_input("Grado Aceptación (α)", min_value=0.0, max_value=1.0, step=0.01, value=0.05))
        params["expected_mean"] = float(st.sidebar.number_input("Media Esperada", step=0.01, value=0.5))

    # Botones
    col1, col2 = st.sidebar.columns(2)
    run_clicked = col1.button("▶ Procesar / Ejecutar", use_container_width=True)
    clear_clicked = col2.button("✕ Limpiar", use_container_width=True)

    return params, run_clicked, clear_clicked
