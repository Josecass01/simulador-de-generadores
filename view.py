import math
import numpy as np
import pandas as pd
import streamlit as st
import re
import altair as alt
from scipy.stats import chi2, norm # Importamos la distribución normal para la prueba de medias
from collections import Counter

# ... (todas las funciones de generación y las pruebas KS, Chi-cuadrado y Póker se quedan igual) ...
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
        next_x = extract_middle_digits(y, digits)
        r = float("0." + str(next_x).rjust(digits, "0"))
        results.append({"i": i+1, "Xᵢ": x, "Y = Xᵢ²": y, "Xᵢ₊₁": next_x, "rᵢ": r})
        x = next_x
        if x == 0: break
    return results

def generate_xorshift32(seed: int, cantidad: int):
    results = []
    x = seed
    if x == 0:
        x = 1
    for i in range(cantidad):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        r = x / 0xFFFFFFFF
        results.append({"i": i + 1, "estado_x": x, "rᵢ": r})
    return results

def generate_bernoulli(p: float, cantidad: int, seed: int):
    results = []
    lcg = LCG(seed)
    for i in range(cantidad):
        u = lcg.next()
        b = 1 if u <= p else 0
        results.append({"i": i+1, "U(0,1)": f"{u:.4f}", "p": p, "Resultado Bernoulli": b})
    return results

def parse_uploaded_numbers(text: str):
    if not text or not text.strip(): return []
    parts = [p.strip() for p in re.split(r"[,\n\r\s]+", text) if p.strip()]
    nums = []
    for p in parts:
        try:
            v = float(p)
            if 0.0 <= v <= 1.0: nums.append(v)
        except ValueError: pass
    return nums

def calculate_stats(numbers):
    n = len(numbers)
    if n == 0: return None
    variance = float(np.var(numbers, ddof=1)) if n > 1 else 0.0
    mean = float(np.mean(numbers))
    std = math.sqrt(variance) if variance > 0 else 0.0
    return {"Media": mean, "Varianza": variance, "Desv. Estándar": std, "Mínimo": float(np.min(numbers)), "Máximo": float(np.max(numbers)), "Cantidad": n}

def ks_critical_value(n, alpha):
    table = {0.2: 1.07, 0.1: 1.22, 0.05: 1.36, 0.02: 1.52, 0.01: 1.63}
    closest = min(table.keys(), key=lambda a: abs(a - alpha))
    return table[closest] / math.sqrt(max(n, 1))

def ks_test_uniform(numbers, alpha=0.05):
    n = len(numbers)
    if n == 0: return {"d": 0.0, "critical": 0.0, "passes": False}
    sorted_nums = np.sort(np.array(numbers))
    empirical = np.arange(1, n+1) / n
    d = float(np.max(np.abs(empirical - sorted_nums)))
    critical = ks_critical_value(n, alpha)
    return {"d": d, "critical": critical, "passes": d < critical}

def chi_squared_test(numbers, bins, alpha=0.05):
    n = len(numbers)
    if n == 0: return None
    observed_freq, _ = np.histogram(numbers, bins=bins, range=(0,1))
    expected_freq = n / bins
    if expected_freq < 5: return None
    dof = bins - 1
    if dof <= 0: return None
    chi2_stat = float(np.sum((observed_freq - expected_freq)**2 / expected_freq))
    critical_value = chi2.ppf(1 - alpha, dof)
    passes = chi2_stat < critical_value
    return {"chi2_stat": chi2_stat, "critical": critical_value, "passes": passes, "dof": dof, "observed": observed_freq, "expected": expected_freq}

def poker_test(numbers, digits, alpha=0.05):
    n = len(numbers)
    if n == 0: return None
    if digits == 3:
        categories = {"Todos diferentes": 0.72, "Un par": 0.27, "Tercia": 0.01}
        dof = 2
    elif digits == 5:
        categories = {"Todos diferentes": 0.3024, "Un par": 0.5040, "Dos pares": 0.1080, "Tercia": 0.0720, "Full": 0.0090, "Póker": 0.0045, "Quintilla": 0.0001}
        dof = 6
    else: return None
    hands = []
    for num in numbers:
        s_num = str(int(num * (10**digits))).zfill(digits)
        counts = Counter(s_num)
        if digits == 3:
            if 3 in counts.values(): hands.append("Tercia")
            elif 2 in counts.values(): hands.append("Un par")
            else: hands.append("Todos diferentes")
        elif digits == 5:
            if 5 in counts.values(): hands.append("Quintilla")
            elif 4 in counts.values(): hands.append("Póker")
            elif 3 in counts.values() and 2 in counts.values(): hands.append("Full")
            elif 3 in counts.values(): hands.append("Tercia")
            elif list(counts.values()).count(2) == 2: hands.append("Dos pares")
            elif 2 in counts.values(): hands.append("Un par")
            else: hands.append("Todos diferentes")
    observed_freq = Counter(hands)
    chi2_stat = 0
    results_data = []
    for cat, prob in categories.items():
        expected = n * prob
        observed = observed_freq[cat]
        if expected > 0: chi2_stat += ((observed - expected)**2) / expected
        results_data.append({"category": cat, "observed": observed, "expected": expected, "prob": prob})
    critical_value = chi2.ppf(1 - alpha, dof)
    passes = chi2_stat < critical_value
    return {"chi2_stat": chi2_stat, "critical": critical_value, "passes": passes, "results_df": pd.DataFrame(results_data)}

def variance_test(numbers, alpha=0.05):
    n = len(numbers)
    if n <= 1: return None
    sample_variance = float(np.var(numbers, ddof=1))
    dof = n - 1
    chi2_lower = chi2.ppf(alpha / 2, dof)
    chi2_upper = chi2.ppf(1 - (alpha / 2), dof)
    lower_limit = chi2_lower / (12 * dof)
    upper_limit = chi2_upper / (12 * dof)
    passes = lower_limit < sample_variance < upper_limit
    return {"sample_variance": sample_variance, "lower_limit": lower_limit, "upper_limit": upper_limit, "passes": passes}

# NUEVA FUNCIÓN PARA LA PRUEBA DE MEDIAS
def means_test(numbers, alpha=0.05):
    n = len(numbers)
    if n == 0:
        return None
    
    sample_mean = float(np.mean(numbers))
    
    # Calcular el estadístico Z
    z_stat = (sample_mean - 0.5) * math.sqrt(n) / math.sqrt(1/12)
    
    # Valor crítico de la distribución normal estándar
    z_critical = norm.ppf(1 - alpha / 2)
    
    # La prueba pasa si el estadístico Z está dentro del intervalo de aceptación
    passes = abs(z_stat) < z_critical
    
    return {
        "sample_mean": sample_mean,
        "z_stat": z_stat,
        "z_critical": z_critical,
        "passes": passes
    }

def process_data(params):
    # ... (código sin cambios) ...
    mode = params.get("mode", "Generar")
    if mode == "Generar":
        method = params.get("method", "Cuadrados Medios")
        cantidad = int(params.get("cantidad", 10))
        if method == "Cuadrados Medios":
            data = generate_cuadrados_medios(int(params["semilla"]), cantidad, int(params["digitos"]))
            df = pd.DataFrame(data)
            return (df["rᵢ"].tolist(), df) if not df.empty else ([], None)
        elif method == "Xorshift32":
            data = generate_xorshift32(int(params["semilla"]), cantidad)
            df = pd.DataFrame(data)
            return (df["rᵢ"].tolist(), df) if not df.empty else ([], None)
        else:
            data = generate_bernoulli(float(params["p"]), cantidad, int(params["seed_b"]))
            df = pd.DataFrame(data)
            return (df["Resultado Bernoulli"].tolist(), df) if not df.empty else ([], None)
    else:
        numbers = parse_uploaded_numbers(params.get("text") or "")
        df = pd.DataFrame({"i": list(range(1, len(numbers)+1)), "rᵢ": numbers}) if numbers else None
        return numbers, df

def render_distribution_chart(numbers):
    # ... (código sin cambios) ...
    df = pd.DataFrame({'value': numbers})
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('value:Q', bin=alt.Bin(maxbins=30), title='Valor'),
        alt.Y('count()', title='Frecuencia')
    ).properties(
        title='Distribución de los Números Generados'
    )
    st.altair_chart(chart, use_container_width=True)

# ===================== LÓGICA DE RENDERIZADO (MODIFICADA) =====================
def render_results(numbers, generation_df, alpha, bins, poker_digits):
    stats = calculate_stats(numbers)
    ks_results = ks_test_uniform(numbers, alpha)
    chi2_results = chi_squared_test(numbers, bins, alpha)
    poker_results = poker_test(numbers, poker_digits, alpha)
    var_results = variance_test(numbers, alpha)
    mean_results = means_test(numbers, alpha) # Llamamos a la nueva prueba

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resumen y Gráficos",
        "🔬 Pruebas Estadísticas",
        "🔢 Secuencia y Descarga",
        "👣 Pasos del Método"
    ])

    with tab1:
        st.subheader("Estadísticas Descriptivas")
        if stats:
            cols = st.columns(len(stats))
            for i, (key, value) in enumerate(stats.items()):
                cols[i].metric(label=key, value=f"{value:.4f}" if isinstance(value, float) else value)
        st.divider()
        render_distribution_chart(numbers)

    with tab2:
        st.subheader("Prueba de Medias")
        if mean_results:
            if mean_results['passes']:
                st.success(f"PASA la prueba de Medias. El estadístico Z ({mean_results['z_stat']:.4f}) está dentro del rango aceptable (±{mean_results['z_critical']:.4f}).")
            else:
                st.error(f"NO PASA la prueba de Medias. El estadístico Z ({mean_results['z_stat']:.4f}) está fuera del rango aceptable (±{mean_results['z_critical']:.4f}).")
            st.metric(label="Media Observada", value=f"{mean_results['sample_mean']:.4f}", delta=f"{mean_results['sample_mean'] - 0.5:.4f} vs. la media esperada de 0.5")
        else:
            st.warning("No se pudo realizar la prueba de Medias (se requiere n > 0).")
        
        st.divider()

        st.subheader("Prueba de Varianza")
        if var_results:
            if var_results['passes']:
                st.success(f"PASA la prueba de Varianza. La varianza observada ({var_results['sample_variance']:.4f}) está dentro del intervalo de aceptación [{var_results['lower_limit']:.4f}, {var_results['upper_limit']:.4f}].")
            else:
                st.error(f"NO PASA la prueba de Varianza. La varianza observada ({var_results['sample_variance']:.4f}) está fuera del intervalo de aceptación [{var_results['lower_limit']:.4f}, {var_results['upper_limit']:.4f}].")
        else:
            st.warning("No se pudo realizar la prueba de Varianza (se requiere n > 1).")

        st.divider()

        st.subheader("Prueba de Bondad de Ajuste: Kolmogórov-Smirnov")
        if ks_results['passes']:
            st.success(f"PASA la prueba KS. El estadístico D ({ks_results['d']:.4f}) es menor que el valor crítico ({ks_results['critical']:.4f}).")
        else:
            st.error(f"NO PASA la prueba KS. El estadístico D ({ks_results['d']:.4f}) es mayor o igual que el valor crítico ({ks_results['critical']:.4f}).")

        st.divider()

        st.subheader(f"Prueba de Bondad de Ajuste: Chi-Cuadrado")
        if chi2_results:
            if chi2_results['passes']:
                st.success(f"PASA la prueba χ². El estadístico ({chi2_results['chi2_stat']:.4f}) es menor que el valor crítico ({chi2_results['critical']:.4f}).")
            else:
                st.error(f"NO PASA la prueba χ². El estadístico ({chi2_results['chi2_stat']:.4f}) es mayor o igual que el valor crítico ({chi2_results['critical']:.4f}).")
            freq_df = pd.DataFrame({'Intervalo': [f'Int. {i+1}' for i in range(bins)], 'Frec. Observada (Oᵢ)': chi2_results['observed'], 'Frec. Esperada (Eᵢ)': [f"{chi2_results['expected']:.2f}"] * bins})
            st.dataframe(freq_df, use_container_width=True)
        else:
            st.warning("No se pudo realizar la prueba Chi-Cuadrado. Asegúrate de que la frecuencia esperada por intervalo sea >= 5.")

        st.divider()

        st.subheader(f"Prueba de Póker")
        if poker_results:
            if poker_results['passes']:
                st.success(f"PASA la prueba de Póker. El estadístico ({poker_results['chi2_stat']:.4f}) es menor que el valor crítico ({poker_results['critical']:.4f}).")
            else:
                st.error(f"NO PASA la prueba de Póker. El estadístico ({poker_results['chi2_stat']:.4f}) es mayor o igual que el valor crítico ({poker_results['critical']:.4f}).")
            df = poker_results['results_df'].rename(columns={'category': 'Categoría', 'observed': 'Frec. Observada', 'expected': 'Frec. Esperada', 'prob': 'Prob. Teórica'})
            df['Frec. Esperada'] = df['Frec. Esperada'].map('{:,.2f}'.format)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No se pudo realizar la prueba de Póker. Solo está implementada para 3 y 5 dígitos.")

    with tab3:
        # ... (código sin cambios)
        st.subheader("Secuencia de Números Generados (`rᵢ`)")
        numbers_str = ", ".join([f"{n:.4f}" for n in numbers])
        st.code(numbers_str, language="")
        st.subheader("Descargas")
        col1, col2 = st.columns(2)
        with col1:
            csv_numbers = pd.DataFrame({"numeros_generados": numbers}).to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Descargar Números (CSV)", data=csv_numbers, file_name="numeros_generados.csv", mime="text/csv", use_container_width=True)
        if generation_df is not None:
             with col2:
                csv_steps = generation_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Descargar Pasos (CSV)", data=csv_steps, file_name="pasos_generacion.csv", mime="text/csv", use_container_width=True)
    with tab4:
        # ... (código sin cambios)
        st.subheader("Tabla de Generación Paso a Paso")
        if generation_df is not None and not generation_df.empty:
            st.dataframe(generation_df, use_container_width=True, height=500)
        else:
            st.info("No hay pasos de generación disponibles para el modo 'Cargar'.")