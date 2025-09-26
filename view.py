import math
import numpy as np
import pandas as pd
import streamlit as st
import re

# ====== Helpers de UI (desplegables + badge) ======
def _inject_badge_css():
    st.markdown("""
    <style>
      .badge{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;
             font-weight:600;font-size:.85rem;border:1px solid;}
      .badge-success{background:#ecfdf5;color:#065f46;border-color:#a7f3d0;}
      .badge-error{background:#fef2f2;color:#991b1b;border-color:#fecaca;}
      .badge-icon{width:14px;height:14px;display:inline-block;}
    </style>
    """, unsafe_allow_html=True)

def _badge_html(passes: bool) -> str:
    if passes:
        return """
        <span class="badge badge-success">
          <svg class="badge-icon" viewBox="0 0 20 20" aria-hidden="true">
            <path fill="currentColor" d="M7.5 13.5l-3-3 1.4-1.4 1.6 1.6 5-5 1.4 1.4-6.4 6.4z"/>
          </svg> APROBADO
        </span>
        """
    return """
    <span class="badge badge-error">
      <svg class="badge-icon" viewBox="0 0 20 20" aria-hidden="true">
        <path d="M6 6l8 8M14 6l-8 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
      </svg> NO APROBADO
    </span>
    """

def _init_dash_state():
    st.session_state.setdefault("exp_stats", True)
    st.session_state.setdefault("exp_ks", True)
    st.session_state.setdefault("exp_seq", True)
    st.session_state.setdefault("exp_steps", True)
    st.session_state.setdefault("exp_dl", True)   # descargas opcional
# ====== Helpers de UI (desplegables + badge) ======
def _inject_badge_css():
    st.markdown("""
    <style>
      .badge{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;
             font-weight:600;font-size:.85rem;border:1px solid;}
      .badge-success{background:#ecfdf5;color:#065f46;border-color:#a7f3d0;}
      .badge-error{background:#fef2f2;color:#991b1b;border-color:#fecaca;}
      .badge-icon{width:14px;height:14px;display:inline-block;}
    </style>
    """, unsafe_allow_html=True)

def _badge_html(passes: bool) -> str:
    if passes:
        return """
        <span class="badge badge-success">
          <svg class="badge-icon" viewBox="0 0 20 20" aria-hidden="true">
            <path fill="currentColor" d="M7.5 13.5l-3-3 1.4-1.4 1.6 1.6 5-5 1.4 1.4-6.4 6.4z"/>
          </svg> APROBADO
        </span>
        """
    return """
    <span class="badge badge-error">
      <svg class="badge-icon" viewBox="0 0 20 20" aria-hidden="true">
        <path d="M6 6l8 8M14 6l-8 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
      </svg> NO APROBADO
    </span>
    """

def _init_dash_state():
    st.session_state.setdefault("exp_stats", True)
    st.session_state.setdefault("exp_ks", True)
    st.session_state.setdefault("exp_seq", True)
    st.session_state.setdefault("exp_steps", True)
    st.session_state.setdefault("exp_dl", True)   # descargas opcional


# ===================== LÓGICA / GENERADORES =====================
def extract_middle_digits(number: int, digits: int) -> int:
    s = str(number).rjust(digits * 2, "0")
    start = (len(s) - digits) // 2
    return int(s[start:start+digits])

class LCG:
    def __init__(self, seed: int = 12345):
        self.current = seed
        self.a = 1664525
        self.c = 1013904223
        self.m = 2 ** 32
    def next(self) -> float:
        self.current = (self.a * self.current + self.c) % self.m
        return self.current / self.m

def generate_cuadrados_medios(seed: int, cantidad: int, digits: int):
    results = []
    x = seed
    for i in range(cantidad):
        y = x * x
        y_str = str(y)
        pos = (len(y_str) - digits) // 2 + 1
        next_x = extract_middle_digits(y, digits)
        r = float("0." + str(next_x).rjust(digits, "0"))
        results.append({
            "i": i+1, "x": x, "y": y, "longitud": len(y_str), "posicion": pos, "nextX": next_x, "r": r
        })
        x = next_x
        if x == 0:
            break
    return results

def generate_productos_medios(seed1: int, seed2: int, cantidad: int, digits: int):
    results = []
    x0, x1 = seed1, seed2
    for i in range(cantidad):
        y = x0 * x1
        y_str = str(y)
        pos = (len(y_str) - digits) // 2 + 1
        next_x = extract_middle_digits(y, digits)
        r = float("0." + str(next_x).rjust(digits, "0"))
        results.append({
            "i": i+1, "x0": x0, "x1": x1, "y": y, "longitud": len(y_str), "posicion": pos, "nextX": next_x, "r": r
        })
        x0, x1 = x1, next_x
        if x1 == 0:
            break
    return results

def generate_bernoulli(p: float, cantidad: int, seed: int):
    results = []
    lcg = LCG(seed)
    for i in range(cantidad):
        u = lcg.next()
        b = 1 if u <= p else 0
        r = round(b + u/10, 4)
        results.append({"i": i+1, "u": u, "p": p, "bernoulli": b, "r": r})
    return results

def parse_uploaded_numbers(text: str):
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in re.split(r"[\,\n\r\s]+", text) if p.strip()]
    nums = []
    for p in parts:
        try:
            v = float(p)
            if 0.0 <= v <= 1.0:
                nums.append(v)
        except ValueError:
            pass
    return nums

def calculate_stats(numbers):
    n = len(numbers)
    if n == 0:
        return None
    mean = float(np.mean(numbers))
    std = float(np.std(numbers, ddof=1)) if n > 1 else 0.0
    variance = std**2
    vmin = float(np.min(numbers))
    vmax = float(np.max(numbers))
    return {"mean": mean, "variance": variance, "std": std, "min": vmin, "max": vmax, "count": n}

def ks_critical_value(n, alpha):
    table = {0.2: 1.07, 0.1: 1.22, 0.05: 1.36, 0.02: 1.52, 0.01: 1.63}
    closest = min(table.keys(), key=lambda a: abs(a - alpha))
    return table[closest] / math.sqrt(max(n, 1))

def ks_test_uniform(numbers, alpha=0.05):
    n = len(numbers)
    if n == 0:
        return {"d": 0.0, "critical": 0.0, "passes": False}
    sorted_nums = np.sort(np.array(numbers))
    empirical = np.arange(1, n+1) / n
    d = float(np.max(np.abs(empirical - sorted_nums)))
    critical = ks_critical_value(n, alpha)
    return {"d": d, "critical": critical, "passes": d < critical}

def process_data(params):
    """Devuelve (numbers, generation_df) según parámetros."""
    mode = params.get("mode", "Generar")
    numbers = []
    generation_df = None

    if mode == "Generar":
        method = params.get("method", "Cuadrados Medios")
        cantidad = int(params.get("cantidad", 10))
        if method == "Cuadrados Medios":
            data = generate_cuadrados_medios(int(params["semilla"]), cantidad, int(params["digitos"]))
            generation_df = pd.DataFrame(data)
            numbers = generation_df["r"].tolist() if not generation_df.empty else []
        elif method == "Productos Medios":
            data = generate_productos_medios(int(params["semilla1"]), int(params["semilla2"]), cantidad, int(params["digitos"]))
            generation_df = pd.DataFrame(data)
            numbers = generation_df["r"].tolist() if not generation_df.empty else []
        else:
            data = generate_bernoulli(float(params["p"]), cantidad, int(params["seed_b"]))
            generation_df = pd.DataFrame(data)
            numbers = generation_df["r"].tolist() if not generation_df.empty else []
    else:
        numbers = parse_uploaded_numbers(params.get("text") or "")
        generation_df = pd.DataFrame({"i": list(range(1, len(numbers)+1)), "r": numbers}) if numbers else None

    return numbers, generation_df

def render_results(numbers, generation_df, alpha=0.05, expected_mean=0.5):
    _init_dash_state()
    _inject_badge_css()

    # ===== Cálculo de métricas y KS =====
    stats = calculate_stats(numbers) if numbers else None
    ks = ks_test_uniform(numbers, alpha=alpha) if numbers else {"d": 0, "critical": 0, "passes": False}

    # ===== Controles globales (expandir/colapsar todo) =====
    cA, cB, cC = st.columns([1,1,6])
    if cA.button("Expandir todo", use_container_width=True):
        st.session_state.update(exp_stats=True, exp_ks=True, exp_seq=True, exp_steps=True, exp_dl=True)
        st.rerun()
    if cB.button("Colapsar todo", use_container_width=True):
        st.session_state.update(exp_stats=False, exp_ks=False, exp_seq=False, exp_steps=False, exp_dl=False)
        st.rerun()
    with cC:
        st.caption("Dashboard desplegable · usa los botones o los headers de cada sección")

    # ===== Sección: Estadísticas =====
    with st.expander("Estadísticas", expanded=st.session_state.exp_stats):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Media", f"{stats['mean']:.6f}" if stats else "-")
        c2.metric("Desv. Estándar", f"{stats['std']:.6f}" if stats else "-")
        c3.metric("Mín", f"{stats['min']:.6f}" if stats else "-")
        c4.metric("Máx", f"{stats['max']:.6f}" if stats else "-")
        c5.metric("Cantidad", f"{stats['count']}" if stats else "0")
        if not numbers:
            st.caption("Pulsa **Procesar / Ejecutar** para calcular.")

    # # ===== Sección: Prueba KS =====
    # with st.expander(" Prueba de Kolmogórov–Smirnov (U[0,1])", expanded=st.session_state.exp_ks):
    #     if numbers:
    #         st.markdown(_badge_html(ks["passes"]), unsafe_allow_html=True)
    #         st.write(
    #             f"**KS (α≈{alpha:.2f})** · D = `{ks['d']:.6f}` · Crítico = `{ks['critical']:.6f}`"
    #         )
    #         if stats:
    #             st.write(
    #                 f"**Media esperada:** `{expected_mean:.6f}` · **Media observada:** `{stats['mean']:.6f}`"
    #             )
    #     else:
    #         st.info("Sin datos aún para evaluar KS.")

    # ===== Sección: Secuencia rᵢ =====
    with st.expander(" Secuencia rᵢ", expanded=st.session_state.exp_seq):
        if numbers:
            per_row = 6
            rows = [numbers[i:i+per_row] for i in range(0, len(numbers), per_row)]
            idx = 0
            for row in rows:
                cols = st.columns(len(row))
                for j, v in enumerate(row):
                    with cols[j]:
                        idx += 1
                        st.code(f"r[{idx}] = {v:.6f}")
        else:
            st.caption("Sin datos todavía.")

    # ===== Sección: Pasos del método =====
    with st.expander("Pasos del método", expanded=st.session_state.exp_steps):
        if generation_df is not None and not getattr(generation_df, "empty", True):
            st.dataframe(generation_df, use_container_width=True)
        else:
            st.caption("No hay pasos que mostrar.")

    # ===== Sección: Descargas =====
    with st.expander(" Descargas", expanded=st.session_state.exp_dl):
        if numbers:
            df_out = pd.DataFrame({"r": numbers})
            csv_numbers = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar rᵢ (CSV)", data=csv_numbers, file_name="ri.csv", mime="text/csv")
            if generation_df is not None and not getattr(generation_df, "empty", True):
                csv_steps = generation_df.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar pasos (CSV)", data=csv_steps, file_name="pasos.csv", mime="text/csv")
        else:
            st.caption("No hay datos para descargar.")
