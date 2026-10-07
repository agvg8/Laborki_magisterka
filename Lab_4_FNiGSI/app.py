import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="FuzzyEco Analyzer 2026", layout="wide")

# Inicjalizacja stanu sesji (zapobiega błędom przy odświeżaniu)
if 'simulation_done' not in st.session_state:
    st.session_state['simulation_done'] = False
if 'res_val' not in st.session_state:
    st.session_state['res_val'] = 0.0

st.title("🧠 System Wnioskowania Rozmytego w Ekonomii")
st.markdown("Projekt zrealizowany zgodnie z wytycznymi środowiska **jFuzzy Logic**.")

# --- DEFINICJA WSZYSTKICH 4 SCENARIUSZY ---
DATA_CONFIG = {
    "Ocena ryzyka kredytowego": {
        "inputs": ["Dochód", "Zadłużenie", "Historia spłat"],
        "output": "Ryzyko kredytowe",
        "rules": [
            ["niski", "wysoki", "niski", "duze"],
            ["niski", "wysoki", "sredni", "duze"],
            ["niski", "sredni", "niski", "duze"],
            ["sredni", "wysoki", "niski", "duze"],
            ["sredni", "sredni", "sredni", "umiarkowane"],
            ["sredni", "niski", "wysoki", "male"],
            ["wysoki", "wysoki", "sredni", "umiarkowane"],
            ["wysoki", "niski", "wysoki", "male"],
            ["wysoki", "sredni", "wysoki", "male"],
            ["niski", "niski", "wysoki", "umiarkowane"]
        ]
    },
    "Atrakcyjność inwestycji": {
        "inputs": ["Stopa zwrotu", "Poziom ryzyka", "Płynność"],
        "output": "Atrakcyjność",
        "rules": [
            ["wysoki", "niski", "wysoki", "duze"],
            ["niski", "wysoki", "niski", "male"],
            ["sredni", "sredni", "sredni", "umiarkowane"],
            ["wysoki", "wysoki", "wysoki", "umiarkowane"],
            ["niski", "niski", "wysoki", "umiarkowane"],
            ["wysoki", "niski", "niski", "duze"],
            ["sredni", "wysoki", "wysoki", "male"],
            ["sredni", "niski", "sredni", "duze"]
        ]
    },
    "Kondycja ekonomiczna przedsiębiorstwa": {
        "inputs": ["Rentowność", "Zadłużenie", "Płynność finansowa"],
        "output": "Kondycja firmy",
        "rules": [
            ["wysoki", "niski", "wysoki", "duze"],
            ["niski", "wysoki", "niski", "male"],
            ["sredni", "sredni", "sredni", "umiarkowane"],
            ["wysoki", "wysoki", "wysoki", "umiarkowane"],
            ["niski", "niski", "niski", "male"],
            ["sredni", "wysoki", "niski", "male"],
            ["wysoki", "sredni", "sredni", "duze"],
            ["niski", "sredni", "wysoki", "umiarkowane"]
        ]
    },
    "Decyzja dotycząca ceny produktu": {
        "inputs": ["Poziom popytu", "Poziom kosztów", "Konkurencja"],
        "output": "Rekomendacja cenowa",
        "rules": [
            ["wysoki", "niski", "niski", "duze"],
            ["niski", "wysoki", "wysoki", "male"],
            ["sredni", "sredni", "sredni", "umiarkowane"],
            ["wysoki", "wysoki", "wysoki", "male"],
            ["niski", "niski", "niski", "duze"],
            ["wysoki", "sredni", "niski", "duze"],
            ["sredni", "wysoki", "wysoki", "male"],
            ["niski", "sredni", "sredni", "umiarkowane"]
        ]
    }
}

# --- SIDEBAR ---
choice = st.sidebar.selectbox("Wybierz scenariusz:", list(DATA_CONFIG.keys()))
config = DATA_CONFIG[choice]
names = config["inputs"]

# --- SILNIK LOGIKI ROZMYTEJ ---
u_input = np.arange(0, 100001, 200)  # Zwiększony krok dla wydajności
u_output = np.arange(0, 101, 1)

in1 = ctrl.Antecedent(u_input, 'in1')
in2 = ctrl.Antecedent(u_input, 'in2')
in3 = ctrl.Antecedent(u_input, 'in3')
out = ctrl.Consequent(u_output, 'out')

# Funkcje przynależności
for var in [in1, in2, in3]:
    var['niski'] = fuzz.trapmf(var.universe, [0, 0, 20000, 50000])
    var['sredni'] = fuzz.trimf(var.universe, [30000, 50000, 70000])
    var['wysoki'] = fuzz.trapmf(var.universe, [50000, 80000, 100000, 100000])

out['male'] = fuzz.trimf(out.universe, [0, 25, 50])
out['umiarkowane'] = fuzz.trimf(out.universe, [25, 50, 75])
out['duze'] = fuzz.trimf(out.universe, [50, 75, 100])

# Budowa bazy reguł
fuzzy_rules = []
for r in config["rules"]:
    rule = ctrl.Rule(in1[r[0]] & in2[r[1]] & in3[r[2]], out[r[3]])
    fuzzy_rules.append(rule)

system_ctrl = ctrl.ControlSystem(fuzzy_rules)
simulation = ctrl.ControlSystemSimulation(system_ctrl)

# --- INTERFEJS UŻYTKOWNIKA ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("⌨️ Parametry wejściowe")
    v1 = st.number_input(f"{names[0]} (0 - 100k)", 0, 100000, 50000)
    v2 = st.number_input(f"{names[1]} (0 - 100k)", 0, 100000, 20000)
    v3 = st.number_input(f"{names[2]} (0 - 100k)", 0, 100000, 80000)

    if st.button("🚀 Przeprowadź wnioskowanie"):
        try:
            simulation.input['in1'] = v1
            simulation.input['in2'] = v2
            simulation.input['in3'] = v3
            simulation.compute()

            st.session_state['res_val'] = simulation.output['out']
            st.session_state['simulation_done'] = True
        except KeyError:
            st.error("⚠️ Nie aktywowano żadnej reguły! Zmień parametry (np. Dochód 90k, Zadłużenie 10k, Historia 90k).")
            st.session_state['simulation_done'] = False

    if st.session_state['simulation_done']:
        st.metric(f"Wynik: {config['output']}", f"{st.session_state['res_val']:.2f}%")
        st.write(f"Wykorzystano metodę wyostrzania: **Środek ciężkości (COG)**.")

with col2:
    st.subheader("📊 Wykres funkcji przynależności wyjścia")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Ręczne rysowanie funkcji dla lepszej stabilności w Streamlit
    for label in ['male', 'umiarkowane', 'duze']:
        ax.plot(out.universe, out[label].mf, label=label, linewidth=2)

    if st.session_state['simulation_done']:
        # Dodanie linii wyniku
        ax.axvline(x=st.session_state['res_val'], color='red', linestyle='--', label='Wynik (COG)')
        ax.fill_between(out.universe, 0, 0.1, alpha=0.2, color='red')

    ax.set_ylim(0, 1.1)
    ax.legend()
    st.pyplot(fig)

st.divider()

# --- SEKCJA RAPORTU ---
st.subheader("📄 Dane techniczne do raportu projektowego")
t1, t2 = st.tabs(["Tabela Reguł", "Opis Modelu"])

with t1:
    st.markdown("**Baza reguł rozmytych (Model Uproszczony)**:")
    df_rules = pd.DataFrame(config["rules"], columns=names + [config["output"]])
    st.table(df_rules)

with t2:
    st.write(f"**Problem**: {choice}")
    st.write(f"**Uniwersum (U)**: 0 - 100,000")
    st.write(f"**Metoda Defuzyfikacji**: Center of Gravity (COG)")
    st.latex(r"y^{*}=\frac{\int y\cdot\mu(y)dy}{\int\mu(y)dy}")