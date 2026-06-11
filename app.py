'''import streamlit as st
import numpy as np
import math

# Definiujemy funkcje aktywacji
def step_function(z):
    return 1 if z > 0 else 0

def sigmoid_function(z):
    # Ograniczamy z, żeby uniknąć OverflowError przy dużych ujemnych wartościach
    z = max(-500, min(500, z))
    return 1 / (1 + math.exp(-z))

def relu_function(z):
    return max(0, z)


class Perceptron:
    def __init__(self, learning_rate, epochs, activation_function):
        self.lr = learning_rate
        self.epochs = epochs
        self.activation_fn = activation_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Inicjalizacja wag małymi losowymi wartościami
        np.random.seed(42)  # stały seed dla powtarzalności wyników
        self.weights = np.random.uniform(-0.5, 0.5, len(X[0]))
        self.bias = np.random.uniform(-0.5, 0.5)

        # Główna pętla uczenia
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                # Oblicz sumę ważoną z = w1*x1 + w2*x2 + b
                z = np.dot(xi, self.weights) + self.bias

                # Wyznacz predykcję
                y_pred = self.activation_fn(z)

                # Jeśli funkcja aktywacji zwraca wartości ciągłe (np. Sigmoid),
                # binaryzujemy ją do 0 lub 1 na potrzeby bramek logicznych
                if self.activation_fn in [sigmoid_function, relu_function]:
                    y_pred = 1 if y_pred >= 0.5 else 0

                # Oblicz błąd e = y_true - y_pred
                error = target - y_pred

                # Aktualizacja parametrów
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        y_pred = self.activation_fn(z)
        return 1 if y_pred >= 0.5 else 0

    def evaluate(self, X, y):
        predictions = [self.predict(xi) for xi in X]
        accuracy = np.mean(predictions == y)
        return accuracy


st.title("🧠 Interaktywny Perceptron - Bramki Logiczne")
st.write("Projekt laboratoryjny - Wojciech Dobrosielski")

# --- PASEK BOCZNY (Sidebar) - Parametry modelu ---
st.sidebar.header("Ustawienia Perceptronu")

# Wybór bramki logicznej
bramka = st.sidebar.selectbox("Wybierz bramkę logiczną", ["AND", "OR", "XOR"])

# Wybór funkcji aktywacji
funkcja_aktywacji_nazwa = st.sidebar.selectbox(
    "Funkcja aktywacji", ["Step (Progowa)", "Sigmoid", "ReLU"]
)

# Slidery do parametrów hipertechnicznych
lr = st.sidebar.slider("Współczynnik uczenia (Learning Rate)", 0.01, 1.0, 0.1, step=0.01)
epochs = st.sidebar.slider("Liczba epok (Epochs)", 1, 100, 10)

# Mapowanie wybranej funkcji aktywacji
slownik_funkcji = {
    "Step (Progowa)": step_function,
    "Sigmoid": sigmoid_function,
    "ReLU": relu_function
}
wybrana_funkcja = slownik_funkcji[funkcja_aktywacji_nazwa]

# --- PRZYGOTOWANIE DANYCH WEJŚCIOWYCH ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

if bramka == "AND":
    y = np.array([0, 0, 0, 1])
elif bramka == "OR":
    y = np.array([0, 1, 1, 1])
else:  # XOR
    y = np.array([0, 1, 1, 0])

    if st.sidebar.button("Uruchom proces uczenia"):
        st.subheader(f"Wyniki uczenia dla bramki: {bramka}")

        # Tworzenie i trenowanie instancji perceptronu
        p = Perceptron(learning_rate=lr, epochs=epochs, activation_function=wybrana_funkcja)
        p.fit(X, y)

        # Ewaluacja
        accuracy = p.evaluate(X, y)

        # Kolorowe podsumowanie celności
        if accuracy == 1.0:
            st.success(f"🎉 Sukces! Model osiągnął 100% skuteczności.")
        else:
            st.error(f"❌ Skuteczność: {accuracy * 100:.0f}%. Model nie potrafi poprawnie sklasyfikować danych.")
            if bramka == "XOR":
                st.warning(
                    "💡 **Wniosek:** Operacja XOR nie jest liniowo separowalna. "
                    "Pojedynczy perceptron potrafi wyznaczyć tylko liniową granicę decyzyjną (prostą linie). "
                    "Do rozwiązania problemu XOR potrzebna jest sieć wielowarstwowa (MLP)"
                )

        # Wyświetlanie końcowych parametrów sieci
        st.markdown("### 🔢 Końcowe parametry matematyczne")
        col1, col2, col3 = st.columns(3)
        col1.metric("Waga w1", f"{p.weights[0]:.4f}")
        col2.metric("Waga w2", f"{p.weights[1]:.4f}")
        col3.metric("Bias (b)", f"{p.bias:.4f}")

        # Wyświetlanie tabeli z predykcjami modelu
        st.markdown("### 📊 Wyniki predykcji krok po kroku")
        tabela_wynikow = []
        for xi, target in zip(X, y):
            pred = p.predict(xi)
            status = "✅ Poprawnie" if pred == target else "❌ Błąd"
            tabela_wynikow.append({
                "Wejście X1": xi[0],
                "Wejście X2": xi[1],
                "Oczekiwany wynik": target,
                "Predykcja modelu": pred,
                "Status": status
            })
        st.table(tabela_wynikow)'''

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math


# --- 1. FUNKCJE AKTYWACJI ---
def step_function(z):
    return 1 if z > 0 else 0


def sigmoid_function(z):
    z = max(-500, min(500, z))
    return 1 / (1 + math.exp(-z))


def relu_function(z):
    return max(0, z)


# --- 2. KLASA PERCEPTRON (Z HISTORIĄ BŁĘDÓW) ---
class Perceptron:
    def __init__(self, learning_rate, epochs, activation_function):
        self.lr = learning_rate
        self.epochs = epochs
        self.activation_fn = activation_function
        self.weights = None
        self.bias = None
        self.errors_history = []  # Zbiera sumaryczny błąd w każdej epoce

    def fit(self, X, y):
        np.random.seed(42)
        self.weights = np.random.uniform(-0.5, 0.5, len(X[0]))
        self.bias = np.random.uniform(-0.5, 0.5)
        self.errors_history = []

        for epoch in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                y_pred = self.activation_fn(z)

                # Binaryzacja dla bramek logicznych
                if self.activation_fn in [sigmoid_function, relu_function]:
                    y_pred = 1 if y_pred >= 0.5 else 0

                error = target - y_pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    total_error += abs(error)

            self.errors_history.append(total_error)

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        y_pred = self.activation_fn(z)
        return 1 if y_pred >= 0.5 else 0

    def evaluate(self, X, y):
        predictions = [self.predict(xi) for xi in X]
        return np.mean(predictions == y)


# --- 3. KONFIGURACJA STREAMLIT ---
st.set_page_config(layout="wide")  # Szeroki układ, żeby wykresy były obok siebie
st.title("🧠 Zaawansowana Wizualizacja Perceptronu")
st.caption("Projekt laboratoryjny: Obiektowa implementacja perceptronu — Prowadzący: Wojciech Dobrosielski")

# Pasek boczny
st.sidebar.header("⚙️ Parametry Modelu")
bramka = st.sidebar.selectbox("Wybierz bramkę logiczną", ["AND", "OR", "XOR"])
funkcja_aktywacji_nazwa = st.sidebar.selectbox("Funkcja aktywacji", ["Step (Progowa)", "Sigmoid", "ReLU"])
lr = st.sidebar.slider("Współczynnik uczenia (Learning Rate)", 0.01, 1.0, 0.2, step=0.01)
epochs = st.sidebar.slider("Liczba epok (Epochs)", 1, 100, 20)

# Mapowanie funkcji
slownik_funkcji = {"Step (Progowa)": step_function, "Sigmoid": sigmoid_function, "ReLU": relu_function}
wybrana_funkcja = slownik_funkcji[funkcja_aktywacji_nazwa]

# Przygotowanie danych
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
if bramka == "AND":
    y = np.array([0, 0, 0, 1])
elif bramka == "OR":
    y = np.array([0, 1, 1, 1])
else:
    y = np.array([0, 1, 1, 0])

# Główny przycisk w sidebarze
uruchom = st.sidebar.button("🚀 Wytrenuj model", use_container_width=True)

# Instrukcja startowa w głównym polu, jeśli nic jeszcze nie kliknięto
if not uruchom:
    st.info(
        "👈 Skonfiguruj parametry w panelu bocznym i kliknij **Wytrenuj model**, aby zobaczyć pełną analizę i wykresy.")
else:
    # Uruchomienie modelu
    p = Perceptron(learning_rate=lr, epochs=epochs, activation_function=wybrana_funkcja)
    p.fit(X, y)
    accuracy = p.evaluate(X, y)

    # --- PANEL 1: WYNIK GŁÓWNY (STATUS) ---
    if accuracy == 1.0:
        st.success(f"### 🎉 Sukces! Model osiągnął 100% skuteczności dla bramki {bramka}")
    else:
        st.error(f"### ❌ Brak zbieżności. Skuteczność modelu: {accuracy * 100:.0f}%")
        if bramka == "XOR":
            st.warning(
                "💡 **Dlaczego XOR nie działa?** Jak widzisz na wykresie granicy decyzyjnej poniżej, "
                "punkty tej samej klasy leżą po przekątnych. Pojedyncza prosta linia (liniowa granica decyzyjna) "
                "nie jest w stanie oddzielić zer od jedynek. Problem ten rozwiązać może dopiero sieć wielowarstwowa (MLP)."
            )

    st.write("---")

    # --- PANEL 2: METRYKI (PARAMETRY MATEMATYCZNE) ---
    st.subheader("🔢 Parametry po zakończeniu procesu uczenia")
    c1, c2, c3 = st.columns(3)
    c1.metric("Waga W1 ($w_1$)", f"{p.weights[0]:.4f}")
    c2.metric("Waga W2 ($w_2$)", f"{p.weights[1]:.4f}")
    c3.metric("Obciążenie (Bias $b$)", f"{p.bias:.4f}")

    st.write("---")

    # --- PANEL 3: WYKRESY (DWA OBOK SIEBIE) ---
    st.subheader("📊 Wykresy analityczne")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.write("**Granica decyzyjna perceptronu w przestrzeni 2D**")
        fig, ax = plt.subplots(figsize=(5, 4))

        # Rysowanie siatki tła (obszaru decyzyjnego)
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([p.predict(pt) for pt in grid_points]).reshape(xx.shape)

        # Kolorowanie tła dla klas 0 i 1
        ax.contourf(xx, yy, Z, alpha=0.15, colors=['#ff4b4b', '#00f4a2'])

        # Rysowanie prostej rozdzielającej (jeśli wagi nie są zerowe)
        if p.weights[1] != 0:
            x_line = np.linspace(-0.5, 1.5, 100)
            y_line = -(p.weights[0] * x_line + p.bias) / p.weights[1]
            ax.plot(x_line, y_line, '--', color='gray', label='Granica decyzyjna')

        # Rysowanie punktów wejściowych (bramki logicznej)
        for xi, target in zip(X, y):
            color = '#00bc71' if target == 1 else '#de1a1a'
            marker = 'o' if target == 1 else 's'
            ax.scatter(xi[0], xi[1], color=color, s=150, edgecolors='black', marker=marker, zorder=5)
            ax.text(xi[0] + 0.04, xi[1] + 0.04, f"({xi[0]},{xi[1]})={target}", fontsize=10, weight='bold')

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("Wejście X1")
        ax.set_ylabel("Wejście X2")
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)

    with col_chart2:
        st.write("**Historia błędu w kolejnych epokach (Funkcja kosztu)**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(range(1, epochs + 1), p.errors_history, marker='o', color='#1f77b4', linewidth=2)
        ax2.set_xlabel("Epoka")
        ax2.set_ylabel("Suma błędów w epoce")
        ax2.set_title("Zbieżność algorytmu uczenia")
        ax2.grid(True, linestyle=':', alpha=0.6)
        # Wymuszenie wyświetlania liczb całkowitych na osi X
        ax2.xaxis.get_major_locator().set_params(integer=True)
        st.pyplot(fig2)

    st.write("---")

    # --- PANEL 4: TABELA WYNIKÓW ---
    st.subheader("📋 Porównanie wyników: Oczekiwane vs Predykcja")
    tabela_wynikow = []
    for xi, target in zip(X, y):
        pred = p.predict(xi)
        status = "✅ Sukces" if pred == target else "❌ Błąd"
        tabela_wynikow.append({
            "Wejście X1": xi[0],
            "Wejście X2": xi[1],
            "Wartość oczekiwana (y)": target,
            "Wynik perceptronu (y_pred)": pred,
            "Status dopasowania": status
        })
    st.table(tabela_wynikow)