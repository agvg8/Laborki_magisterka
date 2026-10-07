import os
import glob
import streamlit as st
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import plotly.express as px

# 1. KONFIGURACJA STRONY
st.set_page_config(page_title="Laboratorium Regresji", layout="wide")

# 2. NAWIGACJA W SIDEBARZE
st.sidebar.title("Nawigacja")
page = st.sidebar.radio("Przejdź do:", ["💸 Napiwki", "💎 Diamenty", "👨‍💻 Społeczność"])

# --- ZADANIE 1 ---
if page == "💸 Napiwki":
    st.title("Zadanie 1: Analiza Napiwków")

    tips = sns.load_dataset("tips")

    # Parametry w sidebarze
    st.sidebar.header("Ustawienia modelu")
    predictor = st.sidebar.radio("Wybierz predyktor (X):", ["total_bill", "size"])

    # Obliczenia
    Y = tips["tip"]
    X = tips[predictor]
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const).fit()

    # Wykres i wyniki
    fig = px.scatter(tips, x=predictor, y="tip", trendline="ols", trendline_color_override="red")
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"**Wynik R²:** {model.rsquared:.3f}")
    st.text(model.summary())

    # Dodaj to pod wynikami regresji w Zadaniu 1
    st.subheader("Diagnostyka modelu")

    # Obliczanie reszt
    residuals = model.resid

    col_diag1, col_diag2 = st.columns(2)

    with col_diag1:
        st.write("Wykres reszt (sprawdzenie lejka)")
        fig_resid = px.scatter(x=model.fittedvalues, y=residuals,
                               labels={'x': 'Wartości dopasowane', 'y': 'Reszty'})
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid)

    with col_diag2:
        st.write("Rozkład reszt (sprawdzenie normalności)")
        fig_hist = px.histogram(residuals, nbins=20, labels={'value': 'Błąd (Reszta)'})
        st.plotly_chart(fig_hist)

    st.divider()
    st.header("🧠 Odpowiedzi na pytania do Zadania 1")

    with st.expander("Kliknij, aby rozwinąć odpowiedzi na pytania teoretyczne"):
        st.subheader("1. Interpretacja modelu")
        st.write("**1.1 Co oznacza wartość współczynnika $\\beta_1$ w tym modelu?**")
        st.write(
            "Współczynnik $\\beta_1$ (przy zmiennej `total_bill`) informuje o tym, o ile jednostek wzrośnie zmienna zależna (napiwek), gdy zmienna niezależna (rachunek) wzrośnie o 1 jednostkę. Jeśli np. $\\beta_1 = 0.10$, oznacza to, że z każdym dodatkowym dolarem na rachunku, napiwek rośnie średnio o 10 centów.")

        st.write("**1.2 Jak interpretować wyraz wolny $\\beta_0$ w kontekście danych o rachunkach i napiwkach?**")
        st.write(
            "$\\beta_0$ to teoretyczna wartość napiwku w sytuacji, gdy rachunek wynosi 0. Jest to punkt przecięcia linii regresji z osią OY.")

        st.write("**1.3 Czy wyraz wolny $\\beta_0$ ma w tym przypadku sens praktyczny? Dlaczego tak/nie?**")
        st.write(
            "Zazwyczaj nie ma sensu praktycznego. Nikt nie zostawia napiwku przy zerowym rachunku. Wartość ta służy jedynie do poprawnego matematycznego umiejscowienia linii na wykresie; często bywa ona dodatnia (sugerując napiwek bazowy) lub ujemna (co jest fizycznie niemożliwe).")

        st.divider()
        st.subheader("2. Statystyki dopasowania")
        st.write("**2.1 Jak interpretować wartość $R^2$ otrzymaną w podsumowaniu modelu?**")
        st.write(
            "$R^2$ (współczynnik determinacji) mówi nam, jaka część zmienności napiwków jest wyjaśniana przez wysokość rachunku. Jeśli $R^2 = 0.45$, oznacza to, że rachunek w 45% decyduje o wysokości napiwku, a pozostałe 55% zależy od innych czynników.")

        st.write("**2.2 Co oznacza wysoka wartość p-value przy danym współczynniku?**")
        st.write(
            "Wysokie p-value (powyżej 0.05) oznacza, że dany predyktor jest nieistotny statystycznie. Sugeruje to, że zmiany tej zmiennej nie mają realnego wpływu na napiwek, a zaobserwowana relacja może być dziełem przypadku.")

        st.write("**2.3 Jakie wnioski można wyciągnąć, jeśli test F wskazuje na istotność całego modelu?**")
        st.write(
            "Oznacza to, że model jako całość dostarcza istotnie lepszych przewidywań niż proste zgadywanie średniej wartości napiwku. Przynajmniej jeden z predyktorów ma realny wpływ na zmienną zależną.")

        st.divider()
        st.subheader("3. Reszty i diagnostyka")
        st.write("**3.1 Jak sprawdzić, czy reszty mają rozkład normalny w tym przykładzie?**")
        st.write(
            "Można to zrobić wizualnie (patrząc na histogram reszt lub wykres QQ-plot) oraz testem statystycznym (np. test Jarque-Bera dostępny w raporcie OLS).")

        st.write("**3.2 Co by oznaczało, gdyby reszty układały się w „kształt lejka” na wykresie reszt?**")
        st.write(
            "Oznacza to heteroskedastyczność. W praktyce: błąd przewidywania rośnie wraz z rachunkiem. Przy małych rachunkach napiwki są przewidywalne, ale przy dużych rozbieżności w hojności klientów stają się znacznie większe.")

        st.write("**3.3 Jakie wnioski można wyciągnąć, jeśli QQ-plot reszt znacząco odbiega od linii 45°?**")
        st.write(
            "Oznacza to, że błędy modelu nie mają rozkładu normalnego. Sugeruje to obecność wartości odstających lub to, że relacja między danymi nie jest liniowa.")

        st.divider()
        st.subheader("4. Rozszerzenie modelu")
        st.write("**4.1 Jak zmieniłby się model, gdybyśmy uwzględnili także zmienną size?**")
        st.write(
            "Model stałby się regresją wieloraką. Współczynnik $R^2$ prawdopodobnie by wzrósł, ponieważ liczba osób przy stole to dodatkowa istotna informacja pomagająca oszacować napiwek.")

        st.write("**4.2 Jak można uwzględnić zmienne kategoryczne, takie jak day lub time?**")
        st.write(
            "Należy zastosować kodowanie (encoding), tworząc tzw. zmienne binarne (0 i 1), które reprezentują obecność danej cechy (np. 1 dla kolacji, 0 dla lunchu).")

        st.write("**4.3 Czy włączenie dodatkowych zmiennych zawsze poprawia dopasowanie modelu?**")
        st.write(
            "Nie zawsze. Choć $R^2$ prawie zawsze wzrośnie, skorygowane $R^2$ (Adjusted R-squared) może spaść, jeśli nowa zmienna nie wnosi istotnej informacji. Nadmierna liczba zmiennych prowadzi też do przeuczenia modelu.")

        st.divider()
        st.subheader("5. Praktyczne aspekty")
        st.write("**5.1 Jak można wykorzystać taki model w praktyce np. w restauracji?**")
        st.write(
            "Manager może prognozować przychody personelu, optymalizować grafik pracy lub ustawiać sugerowane kwoty napiwków na terminalach płatniczych.")

        st.write("**5.2 Jakie ograniczenia ma ten model (tylko tip ~ total_bill)?**")
        st.write(
            "Model ignoruje czynniki ludzkie, jakość jedzenia, czas oczekiwania oraz różnice kulturowe, zakładając, że tylko kwota rachunku ma znaczenie.")

        st.write("**5.3 Jakie inne czynniki mogłyby wpływać na wysokość napiwku?**")
        st.write(
            "Uśmiech personelu, jakość serwisu, pogoda, rodzaj płatności (karta/gotówka) oraz to, czy w posiłku spożywano alkohol.")


# --- Diamenty ---
elif page == "💎 Diamenty":
    st.title("💎 Analiza diamentów")

    # 1. Lokalizacja plików
    dataset_root = "datasets"
    # Szukamy wszystkich CSV w podfolderach
    all_csv = glob.glob(os.path.join(dataset_root, "**", "*.csv"), recursive=True)

    if not all_csv:
        st.error(f"Nie znaleziono plików CSV w folderze '{dataset_root}'")
    else:
        # Wybór pliku przez użytkownika
        file_map = {os.path.basename(f): f for f in all_csv}
        selected_name = st.selectbox("Wybierz plik do analizy:", options=list(file_map.keys()))

        # Wczytanie danych
        df = pd.read_csv(file_map[selected_name])

        st.info("❔ **Longley**: Klasyczna regresja wieloraka na danych ekonomicznych USA.")


        # --- PODGLĄD DANYCH (Zawsze na górze) ---
        st.subheader("📋 Podgląd zestawu danych")
        st.dataframe(df.head(10), width="stretch") # Pokazujemy pierwsze 10 wierszy

        # --- NARZĘDZIE KODOWANIA (Dla kolumn tekstowych jak 'clarity' czy 'type') ---
        with st.expander("🛠️ Narzędzia: Przetwórz kolumny tekstowe na liczby"):
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            if text_cols:
                to_encode = st.selectbox("Wybierz kolumnę do zakodowania:", text_cols)
                if st.button("Koduj kolumnę"):
                    new_name = f"{to_encode}_encoded"
                    df[new_name] = df[to_encode].astype('category').cat.codes
                    st.session_state[f"df_{selected_name}"] = df
                    st.success(f"Dodano kolumnę: {new_name}")
                    st.rerun()
            else:
                st.write("Brak kolumn tekstowych do zakodowania.")

        # Odświeżenie danych z pamięci sesji
        if f"df_{selected_name}" in st.session_state:
            df = st.session_state[f"df_{selected_name}"]

        st.divider()

        # --- KONFIGURACJA REGRESJI ---
        st.subheader("⚙️ Ustawienia Modelu")
        # Tylko kolumny numeryczne mogą być użyte w regresji [cite: 12]
        num_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(num_cols) < 2:
            st.warning("Za mało kolumn numerycznych do analizy.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                target_y = st.selectbox("Zmienna zależna (Y - co przewidujemy):", options=num_cols)
            with c2:
                # Możliwość wyboru wielu zmiennych X [cite: 10]
                features_x = st.multiselect("Zmienne niezależne (X - predyktory):",
                                           options=[c for c in num_cols if c != target_y])

            if features_x:
                # --- OBLICZENIA (Zgodnie z zasadami MNK) [cite: 25, 26, 30] ---
                Y = df[target_y]
                X = df[features_x]
                X_with_const = sm.add_constant(X) # Dodanie wyrazu wolnego beta_0

                model = sm.OLS(Y, X_with_const).fit() # Dopasowanie modelu [cite: 55]

                # --- LEGENDA DLA ZAKODOWANYCH KOLUMN ---
                for feat in features_x:
                    if feat.endswith('_encoded'):
                        original = feat.replace('_encoded', '')
                        if original in df.columns:
                            st.sidebar.markdown(f"**Legenda dla {feat}:**")
                            # Mapowanie wartości
                            mapping = df[[original, feat]].drop_duplicates().sort_values(by=feat)
                            st.sidebar.dataframe(mapping, hide_index=True)

                # --- WYNIKI STATYSTYCZNE ---
                st.subheader("📈 Wyniki modelu")
                m1, m2, m3 = st.columns(3)
                m1.metric("R² (Dopasowanie)", f"{model.rsquared:.4f}") # [cite: 70, 77]
                m2.metric("Istotność (p-value)", f"{model.f_pvalue:.2e}") # [cite: 78, 79]
                m3.metric("Liczba obserwacji", f"{int(model.nobs)}")

                st.write("**Współczynniki Beta (Wpływ na cenę):**")
                st.table(model.params.to_frame(name='Wartość'))

                # --- WYKRES (NAPRAWIONA SKALA) ---
                st.subheader("🔍 Porównanie: Dane Rzeczywiste vs Model")
                df['predictions'] = model.predict(X_with_const)

                # KLUCZ: Ustalamy zakres osi TYLKO na podstawie prawdziwych danych (np. setki)
                y_min = df[target_y].min()
                y_max = df[target_y].max()
                # Margines 5%
                margin = (y_max - y_min) * 0.05
                safe_range = [y_min - margin, y_max + margin]

                fig = px.scatter(df, x=target_y, y='predictions',
                                 title=f"Skala rzeczywista dla: {target_y}",
                                 labels={target_y: 'Cena Rzeczywista', 'predictions': 'Cena Przewidziana'},
                                 opacity=0.4,
                                 template="plotly_white")

                # WYMUSZAMY SKALĘ (To zapobiega "milionom" na osiach)
                fig.update_xaxes(range=safe_range)
                fig.update_yaxes(range=safe_range)

                # Linia idealnego dopasowania (Y=X)
                fig.add_shape(type="line", x0=y_min, y0=y_min, x1=y_max, y1=y_max,
                             line=dict(color="Red", dash="dash"))

                st.plotly_chart(fig, width="stretch")

                with st.expander("Pełny raport OLS Summary"):
                    st.text(model.summary())
            else:
                st.info("Wybierz przynajmniej jedną zmienną X, aby zbudować model.")

elif page == "👨‍💻 Społeczność":
    st.title("‍👨‍💻 Społeczności: Analiza Regresji na Różnych Zbiorach Danych")


    dataset_choice = st.selectbox(
        "Wybierz zbiór danych do analizy:",
        options=[
            "Kwartet Anscombe'a",
            "Zatrudnienie Longley'a",
            "Rozwój Świata Gapminder"
        ]
    )

    df = None
    target_y = ""
    features_x = []

    # --- 1. ŁADOWANIE DANYCH ---
    if dataset_choice == "Kwartet Anscombe'a":
        df_all = sns.load_dataset("anscombe")
        st.info("💡 **Anscombe**: Serie I-IV mają identyczne statystyki, ale inne wykresy.")

        serie = st.selectbox("Wybierz serię danych:", options=['I', 'II', 'III', 'IV'])
        df = df_all[df_all['dataset'] == serie].copy()

        target_y = 'y'
        features_x = ['x']

    elif dataset_choice == "Zatrudnienie Longley'a":
        data_longley = sm.datasets.longley.load_pandas()
        df = data_longley.data
        st.info("❔ **Longley**: Klasyczna regresja wieloraka na danych ekonomicznych USA.")
        st.info("💡 TOTEMP: (Total Employment) Całkowita liczba osób zatrudnionych.\n")
        st.info("💡 GNPDEFL: (GNP Deflator) Wskaźnik inflacji (deflator PKB).\n")
        st.info("💡 GNP: (Gross National Product) Produkt Narodowy Brutto.\n")
        st.info("💡 UNEMP: (Unemployed) Liczba osób bezrobotnych.\n")
        st.info("💡 ARMED: Liczba osób w siłach zbrojnych.\n")
        st.info("💡 POP: (Population) Całkowita populacja nieinstytucjonalna.\n")
        st.info("💡 YEAR: Rok badania.")

        num_cols = df.columns.tolist()
        col_y, col_x = st.columns(2)

        with col_y:
            default_y_idx = num_cols.index('TOTEMP') if 'TOTEMP' in num_cols else 0
            target_y = st.selectbox("Cel analizy (Y):", options=num_cols, index=default_y_idx)

        with col_x:
            available_options = [c for c in num_cols if c != target_y]
            # Bezpieczny wybór domyślny
            default_features = [available_options[0]] if available_options else []
            features_x = st.multiselect("Czynniki (X):", options=available_options, default=default_features)

    elif dataset_choice == "Rozwój Świata Gapminder":
        df = px.data.gapminder().query("year == 2007").copy()
        st.info("❔ **Gapminder**: Czy bogactwo kraju (GDP) przekłada się na długość życia?")
        st.info("💡 lifeExp: (Life Expectancy) Średnia długość życia w latach (Twój cel Y).\n")
        st.info("💡 gdpPercap: (GDP per Capita) PKB na mieszkańca (miara bogactwa).\n")
        st.info("💡 pop: Populacja kraju.")

        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        col_y, col_x = st.columns(2)

        with col_y:
            target_y = st.selectbox("Cel (Y):", options=num_cols, index=num_cols.index('lifeExp'))
        with col_x:
            available_x = [c for c in num_cols if c != target_y]
            features_x = st.multiselect("Predyktory (X):", options=available_x, default=['gdpPercap'] if 'gdpPercap' in available_x else [])

    # --- 2. PODGLĄD I OBLICZENIA ---
    if df is not None:
        st.subheader("📋 Podgląd danych")
        st.dataframe(df.head(10), width="stretch")

        if features_x:
            # Przygotowanie danych do OLS
            Y = df[target_y]
            X = df[features_x]
            X_const = sm.add_constant(X)

            model = sm.OLS(Y, X_const).fit()

            # --- 3. WYNIKI ---
            st.divider()
            st.subheader("📊 Wyniki Statystyczne")

            res1, res2, res3 = st.columns(3)
            res1.metric("R² (Dopasowanie)", f"{model.rsquared:.3f}")
            res2.metric("Istotność (p-value)", f"{model.f_pvalue:.2e}")
            res3.metric("Liczba danych", int(model.nobs))

            # --- 4. WYKRES DIAGNOSTYCZNY ---
            st.subheader("🔍 Wykres: Rzeczywistość vs Model")
            df['pred'] = model.predict(X_const)

            actual_min, actual_max = df[target_y].min(), df[target_y].max()
            diff = actual_max - actual_min
            # Bezpieczny margines, by punkty nie dotykały krawędzi
            safe_range = [actual_min - (diff * 0.1), actual_max + (diff * 0.1)]

            fig = px.scatter(df, x=target_y, y='pred',
                             title=f"Analiza: {dataset_choice}",
                             labels={target_y: 'Dane Rzeczywiste', 'pred': 'Przewidywania Modelu'},
                             opacity=0.7, template="plotly_white",
                             hover_name='country' if 'country' in df.columns else None,
                             color_discrete_sequence=['#00CC96'])

            # Linia 45 stopni (idealne dopasowanie)
            fig.add_shape(type="line", x0=actual_min, y0=actual_min, x1=actual_max, y1=actual_max,
                          line=dict(color="Red", dash="dash"))

            fig.update_xaxes(range=safe_range)
            fig.update_yaxes(range=safe_range)

            st.plotly_chart(fig, width="stretch")

            with st.expander("Pełne podsumowanie OLS (Szczegóły)"):
                st.text(model.summary())
        else:
            st.warning("Wybierz co najmniej jedną zmienną objaśniającą (X).")