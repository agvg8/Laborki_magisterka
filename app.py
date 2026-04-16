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
    # używamy pozycyjnych argumentów: (w, h, txt, border, ln, align)
    pdf.cell(200, 10, "Raport z Badania Algorytmow Rojowych", 0, 1, 'C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "1. Konfiguracja Eksperymentu:", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, f"Algorytm: {algo}", 0, 1)
    pdf.cell(200, 10, f"Funkcja celu: {func}", 0, 1)
    for k, v in params.items():
        pdf.cell(200, 10, f"Parametr {k}: {v}", 0, 1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "2. Wyniki Statystyczne:", 0, 1)
    pdf.set_font("Arial", size=10)
    for index, row in stats_df.iterrows():
        pdf.cell(200, 10, f"{row['Metryka']}: {row['Wartość']}", 0, 1)
    pdf.ln(5)
    # Dodaj wykres zbieżności jeśli jest dostępny
    if convergence_plot is not None:
        try:
            convergence_plot.savefig("temp_conv.png", bbox_inches='tight')
            pdf.image("temp_conv.png", x=10, y=pdf.get_y(), w=100)
        except Exception:
            pdf.cell(200, 10, "Nie można dodać wykresu zbieżności.", 0, 1)
    else:
        pdf.cell(200, 10, "Brak wykresu zbieżności.", 0, 1)

    if tsp_plot:
        try:
            pdf.ln(5)
            tsp_plot.savefig("temp_tsp.png", bbox_inches='tight')
            pdf.image("temp_tsp.png", x=10, y=pdf.get_y(), w=100)
        except Exception:
            pdf.cell(200, 10, "Nie można dodać wykresu TSP.", 0, 1)

    out = pdf.output(dest='S')
    # pdf.output może zwracać bytes, bytearray lub str; ujednolicone zwrócenie bytes
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    elif isinstance(out, str):
        return out.encode('latin-1')
    else:
        # fallback
        return str(out).encode('latin-1')

def plot_3d_function(func, bounds, resolution=100):
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.array([func(np.array([xi, yi])) for xi, yi in zip(X.flatten(), Y.flatten())])
    Z = Z.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    ax.set_title("Powierzchnia funkcji celu")
    return fig

def plot_contour(func, bounds, best_pos=None, resolution=100):
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.array([func(np.array([xi, yi])) for xi, yi in zip(X.flatten(), Y.flatten())])
    Z = Z.reshape(X.shape)

    fig, ax = plt.subplots()
    cp = ax.contourf(X, Y, Z, levels=50)
    plt.colorbar(cp)

    if best_pos is not None and len(best_pos) >= 2:
        ax.scatter(best_pos[0], best_pos[1], color='red', s=100, label='Best')
        ax.legend()

    ax.set_title("Mapa konturowa funkcji")
    return fig

def plot_trajectory(func, bounds, history_pos):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)

    Z = np.array([func(np.array([xi, yi])) for xi, yi in zip(X.flatten(), Y.flatten())])
    Z = Z.reshape(X.shape)

    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, levels=30)

    traj = np.array(history_pos)
    ax.plot(traj[:, 0], traj[:, 1], 'r.-', label="Trajektoria")

    ax.legend()
    ax.set_title("Trajektoria optymalizacji")
    return fig

def plot_histogram(scores):
    fig, ax = plt.subplots()
    ax.hist(scores, bins=10)
    ax.set_title("Histogram wyników")
    return fig

def plot_boxplot(scores):
    fig, ax = plt.subplots()
    ax.boxplot(scores)
    ax.set_title("Boxplot wyników")
    return fig


# Helper do wyświetlania figur matplotlib jako obrazków o ustalonej szerokości
def display_figure(fig, width=750):
    if fig is None:
        return
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        st.image(buf.getvalue(), width=width)
    except Exception:
        # Fallback: spróbuj użyć st.pyplot jeśli st.image zawiedzie
        try:
            st.pyplot(fig)
        except Exception:
            st.write("Nie można wyświetlić wykresu.")

# zamiast tworzyć wykresy przed definicją zmiennych inicjalizujemy je jako None
fig_conv = None
fig_3d = None
fig_contour = None
fig_hist = None
fig_box = None

# Inicjalizacja zmiennych używanych warunkowo
n_cities = None
coords = None
dist_matrix = None
obj_func = None
dimensions = None
selected_bounds = None

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
        st.info("**Inertia** ($w$ - bezwładność): Decyduje o tym, jak bardzo cząstka ufa swojej poprzedniej prędkości. Wysoka wartość sprzyja eksploracji (szukaniu w nowych obszarach), niska – eksploatacji (dokładnemu przeszukiwaniu okolicy).")
        st.info(
            "**Parametr $c_1$** (parametr poznawczy): Określa skłonność cząstki do powrotu do jej własnego najlepszego dotychczasowego położenia ($pbest$).")
        st.info(
        "**Parametr $c_2$** (parametr społeczny): Określa skłonność cząstki do podążania za najlepszym rozwiązaniem znalezionym przez cały rój ($gbest$).")

    elif algo_choice == "Firefly":
        with p_spec1:
            params['alpha'] = st.slider("Alpha (Randomness)", 0.0, 1.0, 0.2)
        with p_spec2:
            params['gamma'] = st.slider("Gamma (Absorption)", 0.0, 5.0, 1.0)
        st.info(
            "**Alfa ($\\alpha$)**: To parametr losowości. Odpowiada za \"szum\" w ruchu świetlika. Jeśli jest wysoki, algorytm silniej przeszukuje przestrzeń (eksploracja).")
        st.info(
            "**Gamma ($\\gamma$)**: Współczynnik absorpcji światła. Decyduje o tym, jak szybko atrakcyjność innego świetlika maleje wraz z odległością. Przy bardzo dużej gammie świetliki stają się \"krótkowzroczne\" i słabiej na siebie oddziałują.")


    elif algo_choice == "ACO":
        with p_spec1:
            params['rho'] = st.slider("Evaporation (rho)", 0.0, 1.0, 0.1)
        st.info(
            "**Parametr $\\rho$** (Współczynnik parowania feromonu): Określa, jak szybko ślad feromonowy zanika w czasie. Jeśli $\\rho$ jest wysokie, feromony szybko znikają, co zmusza \"mrówki\" do ciągłego szukania nowych ścieżek (eksploracja).")


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
        func_name = func_choice or ''
        if func_name:
            obj_func = getattr(Benchmarks, func_name.lower())
        else:
            obj_func = None

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
            result = model.solve(**params)
        else:
            result = model.solve()

        # Bezpieczne rozpakowanie wyniku: algorytmy mogą zwracać 3 lub 4 elementy
        history_pos = None
        if isinstance(result, (list, tuple)):
            if len(result) == 4:
                best_pos, best_score, history, history_pos = result
            elif len(result) == 3:
                best_pos, best_score, history = result
            else:
                # Nieoczekiwany format; próbujemy dopasować najważniejsze elementy
                try:
                    best_pos = result[0]
                    best_score = result[1]
                    history = result[2] if len(result) > 2 else []
                except Exception:
                    raise ValueError(f"Nieoczekiwany format zwracany przez solve(): {result}")
        else:
            # Jeżeli model zwraca pojedynczą wartość (np. score)
            best_pos = None
            best_score = result
            history = []

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
        'pop_size': pop_size,
        'max_iters': max_iters,
        'n_runs': n_runs,
        'all_scores': all_best_scores,
        'dimensions': (dimensions if func_choice != 'TSP' else None),
        'selected_bounds': (selected_bounds if func_choice != 'TSP' else None),
        'coords': (coords if func_choice == 'TSP' else None),
        'last_best_pos': last_best_pos
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
        display_figure(res['fig'], width=750)
    with r_col2:
        st.markdown("##### Statystyki zbiorcze")
        st.table(res['df_stats'])

    if res.get('fig_map') is not None:
        st.markdown("##### Wizualizacja trasy TSP")
        display_figure(res['fig_map'], width=750)

    st.markdown("##### Dodatkowe wizualizacje")

    # Tworzymy histogram i boxplot na podstawie wyników
    try:
        fig_hist = plot_histogram(res['all_scores'])
        fig_box = plot_boxplot(res['all_scores'])
    except Exception:
        fig_hist = None
        fig_box = None

    # Jeżeli problem jest ciągły i 2D - rysujemy 3D i kontur
    if res.get('func') != 'TSP' and res.get('dimensions') == 2:
        try:
            func_name = res.get('func') or ''
            if func_name:
                obj_func = getattr(Benchmarks, func_name.lower())
                fig_3d = plot_3d_function(obj_func, res.get('selected_bounds'))
                fig_contour = plot_contour(obj_func, res.get('selected_bounds'), res.get('last_best_pos'))
            else:
                fig_3d = None
                fig_contour = None
        except Exception:
            fig_3d = None
            fig_contour = None

    if fig_contour:
        display_figure(fig_contour, width=750)

    if fig_3d:
        display_figure(fig_3d, width=750)

    if fig_hist:
        display_figure(fig_hist, width=750)
    if fig_box:
        display_figure(fig_box, width=750)

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
            pdf_b = create_pdf(res['algo'], res['func'], report_p, res['df_stats'], res['fig'], res.get('fig_map'))
            st.download_button("📄 PDF", pdf_b, "raport.pdf")
        except Exception as e:
            st.error(f"Blad PDF: {e}")


else:
    st.info("Kliknij przycisk powyżej, aby wygenerować wyniki.")
