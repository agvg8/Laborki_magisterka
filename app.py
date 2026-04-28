import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

# Konfiguracja strony
st.set_page_config(page_title="Analiza Regresji Wielorakiej", layout="wide")

# --- BOCZNY PANEL ---
st.sidebar.header("Ustawienia Modelu")
dataset_choice = st.sidebar.radio(
    "Wybierz zbiór danych:",
    ("Energy Efficiency", "California Housing")
)


# Ładowanie danych
@st.cache_data
def load_data(choice):
    if choice == "California Housing":
        data = fetch_california_housing(as_frame=True)
        df = data.frame.rename(columns={'MedHouseVal': 'y'})
        # Opis zmiennych na podstawie [cite: 371-379]
        return df, "Mediana wartości domu (y)"
    else:
        # Dane Energy Efficiency [cite: 141, 149-151]
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        df = pd.read_excel(url)
        df.columns = [
            'RelativeCompactness', 'SurfaceArea', 'WallArea', 'RoofArea',
            'OverallHeight', 'Orientation', 'GlazingArea',
            'GlazingAreaDistribution', 'HeatingLoad', 'CoolingLoad'
        ]
        return df, "HeatingLoad (y)"


df, target_label = load_data(dataset_choice)

# Wybór zmiennych w bocznym panelu
all_features = [col for col in df.columns if col not in ['y', 'HeatingLoad', 'CoolingLoad']]
selected_features = st.sidebar.multiselect(
    "Wybierz zmienne objaśniające (X):",
    all_features,
    default=all_features[:3]
)

# --- GŁÓWNY PANEL ---
st.title(f"📊 Analiza Regresji: {dataset_choice}")
st.write(f"Model bada wpływ wybranych czynników na: **{target_label}**.")

# Sekcja suwaków do predykcji
st.subheader("1. Parametry do predykcji")
col1, col2 = st.columns(2)
user_input = {}

for i, feature in enumerate(selected_features):
    with col1 if i % 2 == 0 else col2:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        user_input[feature] = st.slider(f"{feature}", min_val, max_val, float(df[feature].mean()))

generate_button = st.button("🚀 GENERUJ ANALIZĘ I PREDYKCJĘ")

if generate_button:
    if not selected_features:
        st.error("Proszę wybrać przynajmniej jedną zmienną objaśniającą!")
    else:
        # Przygotowanie danych do modelu [cite: 69-71, 275-276]
        X = df[selected_features]
        y = df['y'] if dataset_choice == "California Housing" else df['HeatingLoad']
        X_with_const = sm.add_constant(X)

        model = sm.OLS(y, X_with_const).fit()

        # --- WYNIKI I PREDYKCJA ---
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Prognozowana wartość (Y)", f"{model.predict([1] + list(user_input.values()))[0]:.2f}")
            st.write(f"**R² (Współczynnik determinacji):** {model.rsquared:.4f}")
            # Interpretacja R2
            st.caption("Określa, jaki procent zmienności zmiennej zależnej jest wyjaśniany przez model.")

        with res_col2:
            st.write("**Współczynniki regresji:**")
            st.dataframe(model.params.to_frame(name="Beta"))

        # --- DIAGNOSTYKA [cite: 41-48, 116-133] ---
        st.subheader("2. Diagnostyka modelu (Weryfikacja założeń)")

        diag_tab1, diag_tab2, diag_tab3 = st.tabs(["Wykresy Reszt", "Testy Statystyczne", "Współliniowość (VIF)"])

        with diag_tab1:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            # Reszty vs Predykcja (Liniowość/Homoscedastyczność) [cite: 43, 284-288]
            ax[0].scatter(model.fittedvalues, model.resid, alpha=0.5)
            ax[0].axhline(0, color='red', linestyle='--')
            ax[0].set_title("Reszty vs Wartości dopasowane")
            ax[0].set_xlabel("Fitted")
            ax[0].set_ylabel("Residuals")

            # QQ-plot (Normalność) [cite: 78, 117-119]
            sm.qqplot(model.resid, line='45', fit=True, ax=ax[1])
            ax[1].set_title("Wykres QQ-plot reszt")
            st.pyplot(fig)

        with diag_tab2:
            # Test Jarque-Bera [cite: 82, 120-122]
            jb_stat, jb_p, _, _ = jarque_bera(model.resid)
            # Test Breusch-Pagana [cite: 85, 124]
            bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X_with_const)

            st.write(f"**Test Jarque-Bera (Normalność):** p-value = `{jb_p:.4g}`")
            st.write(f"**Test Breusch-Pagana (Homoscedastyczność):** p-value = `{bp_p:.4g}`")
            st.info("Jeśli p-value < 0.05, założenie może być naruszone.")

        with diag_tab3:
            # VIF [cite: 95-102, 130]
            if len(selected_features) > 1:
                vif_data = pd.DataFrame()
                vif_data["Cecha"] = X.columns
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                st.table(vif_data)
                st.caption("VIF > 5-10 wskazuje na silną współliniowość zmiennych.")
            else:
                st.write("Wybierz więcej zmiennych, aby obliczyć VIF.")

        # --- PODSUMOWANIE ---
        st.subheader("3. Pełny raport tekstowy (Statsmodels)")
        st.text(model.summary().as_text())