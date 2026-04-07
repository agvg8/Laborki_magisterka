import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from benchmarks import Benchmarks
from algorithms import PSO, GWO, Firefly, BAT, ABC, WOA, ACO_TSP, DiscretePSO
from fpdf import FPDF
import io

from fpdf import FPDF
import io

if 'results' not in st.session_state:
    st.session_state.results = None

def create_pdf(algo, func, params, stats_df, convergence_plot, tsp_plot=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)

    # Nagłówek
    pdf.cell(200, 10, txt="Raport z Badania Algorytmów Rojowych", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)

    # Konfiguracja
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. Konfiguracja Eksperymentu:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt=f"Algorytm: {algo}", ln=True)
    pdf.cell(200, 10, txt=f"Funkcja celu: {func}", ln=True)
    for k, v in params.items():
        pdf.cell(200, 10, txt=f"Parametr {k}: {v}", ln=True)
    pdf.ln(5)

    # Statystyki
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Wyniki Statystyczne:", ln=True)
    pdf.set_font("Arial", size=10)
    for index, row in stats_df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Metryka']}: {row['Wartość']}", ln=True)
    pdf.ln(5)

    # Wykresy (Zapisujemy plot do pamięci i wrzucamy do PDF)
    convergence_plot.savefig("temp_conv.png", bbox_inches='tight')
    pdf.image("temp_conv.png", x=10, y=pdf.get_y(), w=100)

    if tsp_plot:
        tsp_plot.savefig("temp_tsp.png", bbox_inches='tight')
        pdf.image("temp_tsp.png", x=110, y=pdf.get_y(), w=90)

    return pdf.output(dest='S').encode('latin-1')

st.set_page_config(page_title="Badania Algorytmów Rojowych", layout="wide")

## --- SIDEBAR ---
st.sidebar.header("⚙️ Konfiguracja Eksperymentu")

# 1. NAJPIERW WYBIERAMY PROBLEM (FUNKCJĘ)
func_choice = st.sidebar.selectbox("Funkcja celu (Problem)", ["Rastrigin", "Schwefel", "Eggholder", "TSP"])

# 2. DEFINIUJEMY KOMPATYBILNOŚĆ
continuous_algos = ["PSO", "GWO", "Firefly", "BAT", "ABC", "WOA"]
discrete_algos = ["ACO", "DiscretePSO"]

# 3. FILTRUJEMY ALGORYTMY NA PODSTAWIE PROBLEMU
if func_choice == "TSP":
    available_algos = discrete_algos
    st.sidebar.info("💡 Wybrano problem dyskretny. Dostępne algorytmy operują na permutacjach.")
else:
    available_algos = continuous_algos
    st.sidebar.info("💡 Wybrano funkcję ciągłą. Dostępne algorytmy operują na liczbach rzeczywistych.")

algo_choice = st.sidebar.selectbox("Algorytm", available_algos)

# Reszta ustawień sidebaru
pop_size = st.sidebar.select_slider("Rozmiar populacji", options=[10, 20, 50, 100])
max_iters = st.sidebar.select_slider("Liczba generacji", options=[50, 100, 200, 500])
n_runs = st.sidebar.number_input("Liczba powtórzeń (dla statystyki)", min_value=1, max_value=40, value=20)

if func_choice != "TSP":
    dimensions = st.sidebar.number_input("Wymiarowość (D)", min_value=2, max_value=10, value=2)
    bounds_map = {"Rastrigin": (-5.12, 5.12), "Schwefel": (-500, 500), "Eggholder": (-512, 512)}
    selected_bounds = bounds_map[func_choice]
else:
    n_cities = st.sidebar.select_slider("Liczba miast", options=[10, 20, 50])

## --- GŁÓWNY PANEL ---
## --- GŁÓWNY PANEL ---
## --- GŁÓWNY PANEL ---
st.title("🔬 Laboratorium Algorytmów Rojowych")

# 1. SEKCJA: USTAWIENIA (Szeroko, od sidebaru do prawej)
with st.container():
    st.subheader("📋 Parametry specyficzne algorytmu")
    # Ustawiamy parametry w kolumnach, żeby nie były jedna pod drugą (oszczędność miejsca)
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
        st.info("Ten algorytm nie wymaga dodatkowych parametrów lub korzysta z domyślnych.")


# 3. SEKCJA: BIEŻĄCE USTAWIENIA (Pod przyciskiem)
st.divider()
st.subheader("⚙️ Bieżące ustawienia eksperymentu")
b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
b_col1.metric("Algorytm", algo_choice)
b_col2.metric("Problem", func_choice)
b_col3.metric("Populacja", pop_size)
b_col4.metric("Generacje", max_iters)
b_col5.metric("Powtórzenia", n_runs)

# 4. SEKCJA: WYNIKI I ANALIZA (Na samym dole)
st.divider()
st.subheader("📊 Wyniki i analiza")

if st.session_state.results is not None:
    res = st.session_state.results

    # Układ wyników: Wykres i Tabela obok siebie
    res_col_left, res_col_right = st.columns([2, 1])

    with res_col_left:
        st.markdown("##### Zbieżność algorytmu")
        st.pyplot(res['fig'])

    with res_col_right:
        st.markdown("##### Statystyki zbiorcze")
        st.table(res['df_stats'])

    # Jeśli TSP - mapa pod spodem
    if res['fig_map'] is not None:
        st.markdown("##### Wizualizacja najlepszej trasy (TSP)")
        st.pyplot(res['fig_map'])

    # Raporty na samym dole
    st.divider()
    st.subheader("📥 Pobieranie raportów")
    dl1, dl2, dl3 = st.columns(3)
    # ... Twoje download_buttons ...

else:
    st.warning("Brak danych do wyświetlenia. Skonfiguruj parametry i kliknij przycisk powyżej.")

# --- LOGIKA URUCHAMIANIA (OBLICZENIA) ---
if st.button("🚀 URUCHOM PEŁNĄ ANALIZĘ"):
    all_best_scores = []
    last_history = []
    last_best_pos = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Przygotowanie danych TSP raz przed pętlą
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
            algos = {
                "PSO": PSO(obj_func, dimensions, pop_size, max_iters, selected_bounds),
                "GWO": GWO(obj_func, dimensions, pop_size, max_iters, selected_bounds),
                "Firefly": Firefly(obj_func, dimensions, pop_size, max_iters, selected_bounds),
                "BAT": BAT(obj_func, dimensions, pop_size, max_iters, selected_bounds),
                "ABC": ABC(obj_func, dimensions, pop_size, max_iters, selected_bounds),
                "WOA": WOA(obj_func, dimensions, pop_size, max_iters, selected_bounds)
            }
            model = algos[algo_choice]

        if algo_choice in ["PSO", "Firefly"]:
            best_pos, best_score, history = model.solve(**params)
        else:
            best_pos, best_score, history = model.solve()

        all_best_scores.append(best_score)
        last_history = history
        last_best_pos = best_pos

        progress_bar.progress((run + 1) / n_runs)
        status_text.text(f"Bieg {run + 1}/{n_runs} - Wynik: {best_score:.4f}")

    total_time = time.time() - start_time

    # --- PRZYGOTOWANIE WYNIKÓW DO ZAPISU ---
    # 1. Tabela statystyk (bez polskich znaków dla PDF)
    df_stats = pd.DataFrame({
        "Metryka": ["Srednia", "Odchylenie Std", "Najlepszy", "Czas (s)"],
        "Wartość": [np.mean(all_best_scores), np.std(all_best_scores), np.min(all_best_scores), f"{total_time:.2f}"]
    })

    # 2. Wykres zbieżności
    fig_conv, ax_conv = plt.subplots()
    ax_conv.plot(last_history)
    ax_conv.set_title(f"Zbieznosc: {algo_choice} na {func_choice}")
    ax_conv.set_xlabel("Generacja")
    ax_conv.set_ylabel("Fitness")

    # 3. Mapa TSP
    fig_map = None
    if func_choice == "TSP":
        fig_map, ax_map = plt.subplots()
        ax_map.scatter(coords[:, 0], coords[:, 1], c='red')
        for i in range(len(last_best_pos) - 1):
            p1, p2 = last_best_pos[i], last_best_pos[i + 1]
            ax_map.plot([coords[p1, 0], coords[p2, 0]], [coords[p1, 1], coords[p2, 1]], 'b-')
        p1, p2 = last_best_pos[-1], last_best_pos[0]
        ax_map.plot([coords[p1, 0], coords[p2, 0]], [coords[p1, 1], coords[p2, 1]], 'b-')
        ax_map.set_title("Najlepsza wyznaczona trasa")

    # --- ZAPIS DO SESJI ---
    st.session_state.results = {
        'df_stats': df_stats,
        'fig': fig_conv,
        'fig_map': fig_map,
        'algo': algo_choice,
        'func': func_choice,
        'params': params,
        'pop_size': pop_size,
        'max_iters': max_iters,
        'n_runs': n_runs
    }
    st.rerun()  # Odśwież, aby pokazać wyniki z sesji

# --- SEKCJA WYŚWIETLANIA WYNIKÓW (POZA BLOKIEM BUTTON) ---
if st.session_state.results is not None:
    res = st.session_state.results

    with col2:
        st.subheader(f"📊 Wyniki analizy: {res['algo']}")
        st.pyplot(res['fig'])
        st.table(res['df_stats'])

        if res['fig_map'] is not None:
            st.subheader("🗺️ Wizualizacja trasy TSP")
            st.pyplot(res['fig_map'])

        # --- SEKCJA POBIERANIA ---
        st.divider()
        st.subheader("📥 Pobierz dane i raport")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            csv_data = res['df_stats'].to_csv(index=False).encode('utf-8')
            st.download_button("💾 Pobierz CSV", csv_data, f"wyniki_{res['algo']}.csv", "text/csv")

        with dl_col2:
            img_buffer = io.BytesIO()
            res['fig'].savefig(img_buffer, format="png", bbox_inches='tight')
            st.download_button("🖼️ Pobierz Wykres (PNG)", img_buffer.getvalue(), "wykres_zbieznosci.png", "image/png")

        with dl_col3:
            try:
                # Przygotowanie parametrów do PDF
                report_params = {
                    "Rozmiar populacji": res['pop_size'],
                    "Liczba generacji": res['max_iters'],
                    "Liczba powtorzen": res['n_runs'],
                    **res['params']
                }

                pdf_bytes = create_pdf(
                    res['algo'],
                    res['func'],
                    report_params,
                    res['df_stats'],
                    res['fig'],
                    res['fig_map']
                )

                st.download_button("📄 Pobierz Raport PDF", pdf_bytes, f"raport_{res['algo']}.pdf", "application/pdf")
            except Exception as e:
                st.error(f"Błąd PDF: {e}")