import streamlit as st

def render_config_form():
    """
    Renderiza el formulario de configuración dentro de un expander
    en la página principal.
    """
    params = {}
    
    with st.expander("⚙️ **Configuración y Parámetros**", expanded=True):
        with st.form(key="config_form"):
            
            mode = st.radio(
                "Modo de entrada",
                ["Generar", "Cargar"],
                index=0,
                horizontal=True,
                help="Elige si quieres generar números con un método o cargarlos manualmente."
            )
            params["mode"] = mode
            
            st.divider()

            if mode == "Generar":
                c1, c2 = st.columns(2)
                with c1:
                    method = st.selectbox(
                        "Método de Generación",
                        ["Xorshift32", "Cuadrados Medios", "Bernoulli"]
                    )
                    params["method"] = method
                    
                    if method == "Cuadrados Medios":
                        params["semilla"] = st.number_input("Semilla (X₀)", min_value=1000, value=5735, step=1)
                        params["digitos"] = st.number_input("Dígitos", min_value=4, max_value=8, value=4, step=1)
                    
                    elif method == "Xorshift32":
                        params["semilla"] = st.number_input("Semilla (Estado Inicial)", min_value=1, value=123456789, step=1)

                    else: # Bernoulli
                        params["p"] = st.slider("Probabilidad (p)", 0.0, 1.0, 0.5, 0.01)
                        params["seed_b"] = st.number_input("Semilla (LCG)", value=12345)

                with c2:
                    cantidad = st.number_input("Cantidad de Números", min_value=1, max_value=5000, value=50, step=1)
                    params["cantidad"] = int(cantidad)
                    
                    st.subheader("Configuración de Pruebas")
                    params["alpha"] = st.slider("Nivel de Significancia (α)", 0.01, 0.20, 0.05, 0.01)
                    params["bins"] = st.number_input("Intervalos para Chi²", min_value=2, max_value=100, value=10, step=1)
                    # NUEVO PARÁMETRO PARA PÓKER
                    params["poker_digits"] = st.selectbox("Dígitos para Póker", [3, 5], index=0)

            else: # Modo Cargar
                params["text"] = st.text_area(
                    "Pega los números (0 a 1) separados por comas, espacios o saltos de línea",
                    height=200,
                    placeholder="0.8902, 0.2456, 0.0319..."
                )
                st.subheader("Configuración de Pruebas")
                params["alpha"] = st.slider("Nivel de Significancia (α)", 0.01, 0.20, 0.05, 0.01)
                params["bins"] = st.number_input("Intervalos para Chi²", min_value=2, max_value=100, value=10, step=1)
                # NUEVO PARÁMETRO PARA PÓKER
                params["poker_digits"] = st.selectbox("Dígitos para Póker", [3, 5], index=0)


            run_clicked = st.form_submit_button(
                label="▶ Generar / Procesar",
                type="primary",
                use_container_width=True
            )

    clear_clicked = st.button("✕ Limpiar Resultados", use_container_width=True)

    return params, run_clicked, clear_clicked