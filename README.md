<h1 align="center">🎓 Laboratoria – studia magisterskie</h1>

<p align="center">
  <b>Informatyka Stosowana · specjalność Sztuczna Inteligencja</b><br>
  Politechnika Bydgoska
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/statsmodels-4051B5?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
</p>

<p align="center">
  Zbiór interaktywnych aplikacji laboratoryjnych z uczenia maszynowego, statystyki,<br>
  logiki rozmytej i optymalizacji rojowej. Każde laboratorium to osobna aplikacja Streamlit.
</p>

---

## 📚 Spis laboratoriów

| | Laboratorium | Temat | Kluczowe technologie |
|:-:|---|---|---|
| 🧠 | [**Perceptron**](./Perceptron) | Perceptron od zera uczący się bramek logicznych AND / OR / XOR | NumPy, Matplotlib |
| 📈 | [**Regresja liniowa**](./Regresja_liniowa) | Regresja OLS na kilku zbiorach danych z diagnostyką reszt | statsmodels, Plotly, seaborn |
| 📊 | [**Regresja wieloraka**](./Lab4_Regresja_wieloraka) | Model wielu zmiennych z weryfikacją założeń i predykcją | statsmodels, scikit-learn |
| 🎯 | [**Regresja logistyczna**](./Regresja_Logistyczna) | Klasyfikacja binarna, kalibracja prawdopodobieństw, odds ratio | scikit-learn |
| 🌫️ | [**Wnioskowanie rozmyte**](./Lab_4_FNiGSI) | System rozmyty do decyzji ekonomicznych | scikit-fuzzy |
| 🐝 | [**Algorytmy rojowe**](./Laboratorium_Algorytmow_Rojowych) | 8 algorytmów metaheurystycznych na funkcjach testowych i TSP | NumPy, FPDF |

---

## 🔍 Szczegóły

### 🧠 Perceptron
Implementacja perceptronu bez bibliotek ML: ręczna pętla uczenia, aktualizacja wag i biasu na podstawie błędu.
- Wybór bramki logicznej: **AND, OR, XOR** (z pokazaniem, dlaczego XOR jest nieliniowo separowalny)
- Funkcje aktywacji: **skokowa, sigmoidalna, ReLU**
- Regulowany współczynnik uczenia i liczba epok
- Wykresy analityczne i porównanie wyników oczekiwanych z predykcją

### 📈 Regresja liniowa
Aplikacja z trzema zadaniami w nawigacji bocznej:
- **Napiwki** – regresja prosta na zbiorze `tips`, wykres reszt i rozkład błędów
- **Diamenty** – model ceny diamentów na danych z Kaggle, porównanie wartości rzeczywistych z modelem
- **Społeczność** – analiza regresji na różnych zbiorach danych
- Interpretacja współczynników, R², diagnostyka założeń i odpowiedzi na pytania teoretyczne

### 📊 Regresja wieloraka
- Zbiory danych: **Energy Efficiency** (UCI) i **California Housing**
- Dowolny wybór zmiennych objaśniających i suwaki do predykcji
- Diagnostyka założeń modelu: **VIF** (współliniowość), **test Breuscha-Pagana** (heteroskedastyczność), **test Jarque-Bera** (normalność reszt)
- Pełny raport z `statsmodels`

### 🎯 Regresja logistyczna
- Interaktywne wprowadzenie do **prawdopodobieństwa, szansy (odds) i logitu** z symulatorem funkcji sigmoidalnej
- Zbiory: dane własne (wiek vs zgon), **Breast Cancer**, syntetyczne zbalansowane i niezbalansowane (90/10), możliwość wgrania własnego CSV/XLSX
- **Kalibracja prawdopodobieństw**: metoda Platta i regresja izotoniczna
- Metryki: **ROC AUC, Average Precision, Brier score**, krzywa kalibracji
- Optymalny próg decyzyjny, macierz pomyłek, ilorazy szans dla cech

### 🌫️ Wnioskowanie rozmyte (FNiGSI)
System wnioskowania rozmytego dla czterech scenariuszy ekonomicznych:
- ocena **ryzyka kredytowego**
- **atrakcyjność inwestycji**
- **kondycja ekonomiczna przedsiębiorstwa**
- oraz czwarty scenariusz decyzyjny

Każdy z nich ma 3 zmienne wejściowe, bazę reguł i wizualizację funkcji przynależności wyjścia.

### 🐝 Algorytmy rojowe
Własne implementacje algorytmów inteligencji roju na wspólnej klasie bazowej:

| Optymalizacja ciągła | Problem komiwojażera (TSP) |
|---|---|
| PSO · GWO · Firefly · BAT · ABC · WOA | ACO · Discrete PSO |

- Funkcje testowe z wizualizacją 3D powierzchni
- Wykresy zbieżności i statystyki z wielu uruchomień
- **Generowanie raportu PDF** z wynikami eksperymentu

---

## 🚀 Uruchomienie

Każde laboratorium jest niezależne i ma własny `requirements.txt`.

```bash
git clone https://github.com/agvg8/Laborki_magisterka.git
cd Laborki_magisterka/Regresja_Logistyczna   # wybierz folder

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Aplikacja otworzy się w przeglądarce pod adresem `http://localhost:8501`.

---
