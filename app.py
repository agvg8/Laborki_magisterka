import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from benchmarks import Benchmarks
from algorithms import PSO, GWO, Firefly, BAT, ABC, WOA, ACO_TSP, DiscretePSO
from fpdf import FPDF
import io

# Inicjalizacja stanu sesji
if 'results' not in st.session_state:
    st.session_state.results = None


def create_pdf(algo, func, params, stats_df, convergence_plot, tsp_plot=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Raport z Badania Algorytmow Rojowych", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. Konfiguracja Eksperymentu:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt=f"Algorytm: {algo}", ln=True)
    pdf.cell(200, 10, txt=f"Funkcja celu: {func}", ln=True)
    for k, v in params.items():
        pdf.cell(200, 10, txt=f"Parametr {k}: {v}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Wyniki Statystyczne:", ln=True)
    pdf.set_font("Arial", size=10)
    for index, row in stats_df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Metryka']}: {row['Wartość']}", ln=True)
    pdf.ln(5)
    convergence_plot.savefig("temp_conv.png", bbox_inches='tight')
    pdf.image("temp_conv.png", x=10, y=pdf.get_y(), w=100)
    if tsp_plot:
        pdf.ln(70)  # Odstęp dla obrazka
        tsp_plot.savefig("temp_tsp.png", bbox_inches='tight')
        pdf.image("temp_tsp.png", x=10, y=pdf.get_y(), w=100)
    return pdf.output(dest='S').encode('latin-1')


st.set_page_config(page_title="Badania Algorytmów Rojowych", layout="wide")

st.sidebar.header("⚙️ Konfiguracja Eksperymentu")

# 1. Wybór Problemu i Algorytmu
func_choice = st.sidebar.selectbox("Funkcja celu (Problem)", ["Rastrigin", "Schwefel", "Eggholder", "TSP"])

continuous_algos = ["PSO", "GWO", "Firefly", "BAT", "ABC", "WOA"]
discrete_algos = ["ACO", "DiscretePSO"]

available_algos = discrete_algos if func_choice == "TSP" else continuous_algos
algo_choice = st.sidebar.selectbox("Algorytm", available_algos)

# 2. Parametry z możliwością ręcznego wpisania (number_input działa jak pole tekstowe + strzałki)
# Używamy min_value i max_value, aby trzymać się zakresów z instrukcji
pop_size = st.sidebar.number_input(
    "Rozmiar populacji",
    min_value=1,
    max_value=1000,
    value=50,
    step=1,
    help="Możesz wpisać wartość ręcznie lub użyć przycisków."
)

max_iters = st.sidebar.number_input(
    "Liczba generacji (Iteracje)",
    min_value=1,
    max_value=2000,
    value=100,
    step=1
)

n_runs = st.sidebar.number_input(
    "Liczba powtórzeń (Statystyka)",
    min_value=1,
    max_value=100,
    value=20,
    step=1
)

# 3. Parametry specyficzne dla problemu
if func_choice != "TSP":
    dimensions = st.sidebar.number_input(
        "Wymiarowość (D)",
        min_value=2,
        max_value=100,
        value=2,
        step=1
    )
    # Mapowanie granic dla PDF i algorytmów
    bounds_map = {"Rastrigin": (-5.12, 5.12), "Schwefel": (-500, 500), "Eggholder": (-512, 512)}
    selected_bounds = bounds_map[func_choice]
else:
    n_cities = st.sidebar.number_input(
        "Liczba miast (TSP)",
        min_value=5,
        max_value=200,
        value=20,
        step=1
    )

## --- GŁÓWNY PANEL ---
st.title("🔬 Laboratorium Algorytmów Rojowych")

# 1. SEKCJA: USTAWIENIA (Szeroka sekcja pod tytułem)
with st.container():
    st.subheader("📋 Parametry specyficzne algorytmu")
    params = {}
    p_spec1, p_spec2, p_spec3 = st.columns(3)

    if algo_choice == "PSO":
        with p_spec1:
            params['w'] = st.slider("Inertia (w)", 0.0, 1.0, 0.5)
        with p_spec2:
            params['c1'] = st.slider("C1 (Cognitive)", 0.0, 3.0, 1.5)
        with p_spec3:
            params['c2'] = st.slider("C2 (Social)", 0.0, 3.0, 1.5)
    elif algo_choice == "Firefly":
        with p_spec1:
            params['alpha'] = st.slider("Alpha (Randomness)", 0.0, 1.0, 0.2)
        with p_spec2:
            params['gamma'] = st.slider("Gamma (Absorption)", 0.0, 5.0, 1.0)
    elif algo_choice == "ACO":
        with p_spec1:
            params['rho'] = st.slider("Evaporation (rho)", 0.0, 1.0, 0.1)
    else:
        st.info("Ten algorytm korzysta z domyślnych ustawień parametrów specyficznych.")

# 2. PRZYCISK (Na pełną szerokość)
if st.button("🚀 URUCHOM PEŁNĄ ANALIZĘ", use_container_width=True):
    all_best_scores = []
    last_history = []
    last_best_pos = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    if func_choice == "TSP":
        coords, dist_matrix = Benchmarks.generate_tsp_data(n_cities)
    else:
        obj_func = getattr(Benchmarks, func_choice.lower())

    start_time = time.time()
    for run in range(n_runs):
        if func_choice == "TSP":
            if algo_choice == "ACO":
                model = ACO_TSP(dist_matrix, n_ants=pop_size, n_iter=max_iters, rho=params.get('rho', 0.1))
            else:
                model = DiscretePSO(dist_matrix, n_particles=pop_size, max_iter=max_iters)
        else:
            algos = {"PSO": PSO(obj_func, dimensions, pop_size, max_iters, selected_bounds),
                     "GWO": GWO(obj_func, dimensions, pop_size, max_iters, selected_bounds),
                     "Firefly": Firefly(obj_func, dimensions, pop_size, max_iters, selected_bounds),
                     "BAT": BAT(obj_func, dimensions, pop_size, max_iters, selected_bounds),
                     "ABC": ABC(obj_func, dimensions, pop_size, max_iters, selected_bounds),
                     "WOA": WOA(obj_func, dimensions, pop_size, max_iters, selected_bounds)}
            model = algos[algo_choice]

        if algo_choice in ["PSO", "Firefly"]:
            best_pos, best_score, history = model.solve(**params)
        else:
            best_pos, best_score, history = model.solve()

        all_best_scores.append(best_score)
        last_history = history
        last_best_pos = best_pos
        progress_bar.progress((run + 1) / n_runs)
        status_text.text(f"Bieg {run + 1}/{n_runs}...")

    total_time = time.time() - start_time

    # Przygotowanie wykresów
    fig_conv, ax_conv = plt.subplots()
    ax_conv.plot(last_history)
    ax_conv.set_title(f"Zbieznosc: {algo_choice}")

    fig_map = None
    if func_choice == "TSP":
        fig_map, ax_map = plt.subplots()
        ax_map.scatter(coords[:, 0], coords[:, 1], c='red')
        for i in range(len(last_best_pos) - 1):
            p1, p2 = last_best_pos[i], last_best_pos[i + 1]
            ax_map.plot([coords[p1, 0], coords[p2, 0]], [coords[p1, 1], coords[p2, 1]], 'b-')
        p1, p2 = last_best_pos[-1], last_best_pos[0]
        ax_map.plot([coords[p1, 0], coords[p2, 0]], [coords[p1, 1], coords[p2, 1]], 'b-')

    st.session_state.results = {
        'df_stats': pd.DataFrame({
            "Metryka": ["Srednia", "Odchylenie Std", "Najlepszy", "Czas (s)"],
            "Wartość": [np.mean(all_best_scores), np.std(all_best_scores), np.min(all_best_scores), f"{total_time:.2f}"]
        }),
        'fig': fig_conv,
        'fig_map': fig_map,
        'algo': algo_choice,
        'func': func_choice,
        'params': params,
        'pop_size': pop_size, 'max_iters': max_iters, 'n_runs': n_runs
    }
    st.rerun()

# 3. SEKCJA: BIEŻĄCE USTAWIENIA
st.divider()
st.subheader("⚙️ Bieżące ustawienia eksperymentu")
b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
b_col1.metric("Algorytm", algo_choice)
b_col2.metric("Problem", func_choice)
b_col3.metric("Populacja", pop_size)
b_col4.metric("Generacje", max_iters)
b_col5.metric("Powtórzenia", n_runs)

# 4. SEKCJA: WYNIKI I ANALIZA
st.divider()
st.subheader("📊 Wyniki i analiza")

if st.session_state.results is not None:
    res = st.session_state.results

    # Mniejsze wykresy i statystyki obok siebie
    r_col1, r_col2 = st.columns([2, 1])
    with r_col1:
        st.markdown("##### Wykres zbieżności")
        st.pyplot(res['fig'])
    with r_col2:
        st.markdown("##### Statystyki zbiorcze")
        st.table(res['df_stats'])

    if res['fig_map'] is not None:
        st.markdown("##### Wizualizacja trasy TSP")
        st.pyplot(res['fig_map'])

    st.divider()
    st.subheader("📥 Pobierz raporty")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("💾 CSV", res['df_stats'].to_csv(index=False).encode('utf-8'), "wyniki.csv")
    with dl2:
        img_buf = io.BytesIO()
        res['fig'].savefig(img_buf, format="png")
        st.download_button("🖼️ PNG", img_buf.getvalue(), "wykres.png")
    with dl3:
        try:
            report_p = {"Pop": res['pop_size'], "Gen": res['max_iters'], "Runs": res['n_runs'], **res['params']}
            pdf_b = create_pdf(res['algo'], res['func'], report_p, res['df_stats'], res['fig'], res['fig_map'])
            st.download_button("📄 PDF", pdf_b, "raport.pdf")
        except:
            st.error("Blad PDF")
else:
    st.info("Kliknij przycisk powyżej, aby wygenerować wyniki.")