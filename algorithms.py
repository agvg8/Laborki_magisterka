import numpy as np
from base_algorithm import SwarmAlgorithm


class PSO(SwarmAlgorithm):
    def solve(self, w=0.5, c1=1.5, c2=1.5):
        # Inicjalizacja
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dim))
        vel = np.zeros((self.n_particles, self.n_dim))
        pbest_pos = pos.copy()
        pbest_score = np.array([self.obj_func(p) for p in pos])

        gbest_idx = np.argmin(pbest_score)
        self.best_pos = pbest_pos[gbest_idx].copy()
        self.best_score = pbest_score[gbest_idx]

        for _ in range(self.max_iter):
            r1, r2 = np.random.rand(2)
            vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (self.best_pos - pos)
            pos += vel
            pos = np.clip(pos, self.bounds[0], self.bounds[1])

            curr_scores = np.array([self.obj_func(p) for p in pos])

            # Aktualizacja pbest
            improved_indices = curr_scores < pbest_score
            pbest_score[improved_indices] = curr_scores[improved_indices]
            pbest_pos[improved_indices] = pos[improved_indices]

            # Aktualizacja gbest
            if np.min(pbest_score) < self.best_score:
                self.best_score = np.min(pbest_score)
                self.best_pos = pbest_pos[np.argmin(pbest_score)].copy()

            self.history.append(self.best_score)
        return self.best_pos, self.best_score, self.history


class DiscretePSO:
    def __init__(self, dist_matrix, n_particles, max_iter):
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.history = []

    def _get_distance(self, path):
        d = 0
        for i in range(self.n_cities - 1):
            d += self.dist_matrix[path[i], path[i + 1]]
        return d + self.dist_matrix[path[-1], path[0]]

    def _get_swap_sequence(self, source, target):
        """Tworzy listę zamian (swapów), które przekształcą source w target"""
        swaps = []
        temp_source = list(source)
        for i in range(len(source)):
            if temp_source[i] != target[i]:
                target_idx = temp_source.index(target[i])
                swaps.append((i, target_idx))
                temp_source[i], temp_source[target_idx] = temp_source[target_idx], temp_source[i]
        return swaps

    def _apply_swaps(self, path, swaps, probability):
        """Nakłada zamiany na trasę z określonym prawdopodobieństwem"""
        new_path = list(path)
        for s1, s2 in swaps:
            if np.random.rand() < probability:
                new_path[s1], new_path[s2] = new_path[s2], new_path[s1]
        return new_path

    def solve(self, alpha=0.9, beta=0.9):
        # 1. Inicjalizacja: każda cząsteczka to losowa permutacja miast
        particles = [np.random.permutation(self.n_cities).tolist() for _ in range(self.n_particles)]
        pbest_pos = list(particles)
        pbest_scores = [self._get_distance(p) for p in particles]

        gbest_idx = np.argmin(pbest_scores)
        gbest_pos = list(pbest_pos[gbest_idx])
        gbest_score = pbest_scores[gbest_idx]

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                # Wyznacz sekwencje zamian w stronę pbest i gbest
                swap_pbest = self._get_swap_sequence(particles[i], pbest_pos[i])
                swap_gbest = self._get_swap_sequence(particles[i], gbest_pos)

                # Zastosuj zamiany (odpowiednik aktualizacji prędkości w PSO)
                particles[i] = self._apply_swaps(particles[i], swap_pbest, alpha)
                particles[i] = self._apply_swaps(particles[i], swap_gbest, beta)

                # Oblicz nowy koszt
                current_score = self._get_distance(particles[i])

                # Aktualizacja pbest
                if current_score < pbest_scores[i]:
                    pbest_scores[i] = current_score
                    pbest_pos[i] = list(particles[i])

            # Aktualizacja gbest
            if min(pbest_scores) < gbest_score:
                gbest_score = min(pbest_scores)
                gbest_pos = list(pbest_pos[np.argmin(pbest_scores)])

            self.history.append(gbest_score)

        return gbest_pos, gbest_score, self.history


class GWO(SwarmAlgorithm):
    def solve(self):
        # Inicjalizacja populacji
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dim))

        # Inicjalizacja liderów
        alpha_pos = np.zeros(self.n_dim)
        beta_pos = np.zeros(self.n_dim)
        delta_pos = np.zeros(self.n_dim)

        alpha_score = float("inf")
        beta_score = float("inf")
        delta_score = float("inf")

        for t in range(self.max_iter):
            # 1. Aktualizacja liderów
            for i in range(self.n_particles):
                # Obliczamy dopasowanie (fitness)
                score = self.obj_func(pos[i])

                if score < alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = score, pos[i].copy()
                elif score < beta_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = score, pos[i].copy()
                elif score < delta_score:
                    delta_score, delta_pos = score, pos[i].copy()

            # Parametr 'a' maleje liniowo od 2 do 0
            a = 2 - 2 * (t / self.max_iter)

            # 2. Aktualizacja pozycji wszystkich wilków
            for i in range(self.n_particles):
                # Parametry dla Alfy
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - pos[i])
                X1 = alpha_pos - A1 * D_alpha

                # Parametry dla Bety
                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * beta_pos - pos[i])
                X2 = beta_pos - A2 * D_beta

                # Parametry dla Delty
                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * delta_pos - pos[i])
                X3 = delta_pos - A3 * D_delta

                # Nowa pozycja to średnia z sugestii trzech liderów
                pos[i] = (X1 + X2 + X3) / 3

            # Ograniczenie pozycji do dopuszczalnych granic
            pos = np.clip(pos, self.bounds[0], self.bounds[1])

            # Zapisywanie historii (najlepszy wynik w tej iteracji to alpha_score)
            self.history.append(alpha_score)

        self.best_score = alpha_score
        self.best_pos = alpha_pos
        return self.best_pos, self.best_score, self.history


class Firefly(SwarmAlgorithm):
    def solve(self, alpha=0.5, beta0=1.0, gamma=1.0):
        # 1. Inicjalizacja pozycji i intensywności (jasności)
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dim))
        intensity = np.array([self.obj_func(p) for p in pos])

        # Inicjalizacja najlepszych wartości
        best_idx = np.argmin(intensity)
        self.best_score = intensity[best_idx]
        self.best_pos = pos[best_idx].copy()

        for t in range(self.max_iter):
            # Opcjonalnie: zmniejszanie parametru alpha (losowości) z czasem
            # alpha *= 0.98

            for i in range(self.n_particles):
                for j in range(self.n_particles):
                    # Jeśli świetlik j jest jaśniejszy (mniejsza wartość funkcji celu), to i leci do j
                    if intensity[j] < intensity[i]:
                        # Oblicz dystans euklidesowy
                        r = np.linalg.norm(pos[i] - pos[j])

                        # Oblicz atrakcyjność (beta)
                        beta = beta0 * np.exp(-gamma * r ** 2)

                        # Ruch świetlika i w stronę j + krok losowy
                        rand_step = alpha * (np.random.rand(self.n_dim) - 0.5)
                        pos[i] += beta * (pos[j] - pos[i]) + rand_step

                        # Pilnowanie granic
                        pos[i] = np.clip(pos[i], self.bounds[0], self.bounds[1])

            # 2. AKTUALIZACJA: Obliczamy intensywność raz po zakończeniu ruchu wszystkich świetlików
            intensity = np.array([self.obj_func(p) for p in pos])

            # Znalezienie najlepszego w tej generacji
            current_min = np.min(intensity)
            if current_min < self.best_score:
                self.best_score = current_min
                self.best_pos = pos[np.argmin(intensity)].copy()

            self.history.append(self.best_score)

        return self.best_pos, self.best_score, self.history


class ACO_TSP:
    def __init__(self, dist_matrix, n_ants, n_iter, alpha=1, beta=2, rho=0.1, q=100):
        self.dist_matrix = dist_matrix
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha  # Wpływ feromonu (pamięć zbiorowa)
        self.beta = beta  # Wpływ odległości (heurystyka lokalna)
        self.rho = rho  # Współczynnik parowania feromonu
        self.q = q  # Stała intensywności feromonu
        self.n_cities = len(dist_matrix)

        # Inicjalizacja macierzy feromonów (małe wartości dodatnie)
        self.pheromone = np.ones((self.n_cities, self.n_cities)) / self.n_cities
        self.history = []

    def solve(self):
        best_dist = float('inf')
        best_path = None

        for _ in range(self.n_iter):
            # 1. Wygeneruj ścieżki dla wszystkich mrówek
            all_paths = self._gen_all_paths()

            # 2. Oblicz długości wszystkich ścieżek
            all_distances = [self._calculate_dist(p) for p in all_paths]

            # 3. Znajdź najlepszą ścieżkę w tej iteracji
            min_dist = min(all_distances)
            if min_dist < best_dist:
                best_dist = min_dist
                best_path = all_paths[np.argmin(all_distances)]

            # 4. Aktualizacja feromonów (parowanie + nowe ślady)
            self._update_pheromones(all_paths, all_distances)

            self.history.append(best_dist)

        return best_path, best_dist, self.history

    def _gen_all_paths(self):
        all_paths = []
        for ant in range(self.n_ants):
            path = self._gen_single_ant_path()
            all_paths.append(path)
        return all_paths

    def _gen_single_ant_path(self):
        # Każda mrówka zaczyna w losowym mieście
        start_city = np.random.randint(self.n_cities)
        path = [start_city]
        visited = {start_city}

        while len(path) < self.n_cities:
            current_city = path[-1]

            # Oblicz prawdopodobieństwo przejścia do nieodwiedzonych miast
            probs = self._calculate_probabilities(current_city, visited)

            # Wybierz następne miasto na podstawie rozkładu prawdopodobieństwa
            next_city = np.random.choice(range(self.n_cities), p=probs)
            path.append(next_city)
            visited.add(next_city)

        return path

    def _calculate_probabilities(self, current_city, visited):
        # Pobierz feromony i odległości dla krawędzi z obecnego miasta
        phi = self.pheromone[current_city].copy()

        # Heurystyka: 1/odległość (im bliżej, tym lepiej)
        # Dodajemy małe epsilon, aby uniknąć dzielenia przez zero na przekątnej
        dist = self.dist_matrix[current_city].copy()
        eta = 1.0 / (dist + 1e-10)

        # Wyzeruj prawdopodobieństwo dla odwiedzonych miast
        for city in visited:
            phi[city] = 0
            eta[city] = 0

        # Wzór na wagę krawędzi: (feromon^alpha) * (heurystyka^beta)
        weights = (phi ** self.alpha) * (eta ** self.beta)

        # Normalizacja do sumy = 1
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            # Jeśli wszystkie wagi są 0 (np. przez parowanie), wybierz losowo z nieodwiedzonych
            probs = np.zeros(self.n_cities)
            unvisited = [c for c in range(self.n_cities) if c not in visited]
            probs[unvisited] = 1.0 / len(unvisited)
            return probs

        return weights / sum_weights

    def _update_pheromones(self, paths, distances):
        # Parowanie feromonów
        self.pheromone *= (1 - self.rho)

        # Dodawanie nowych feromonów przez każdą mrówkę
        for path, dist in zip(paths, distances):
            deposit = self.q / dist
            for i in range(self.n_cities - 1):
                u, v = path[i], path[i + 1]
                self.pheromone[u, v] += deposit
                self.pheromone[v, u] += deposit  # Dla grafu nieskierowanego

            # Powrót do miasta startowego
            u, v = path[-1], path[0]
            self.pheromone[u, v] += deposit
            self.pheromone[v, u] += deposit

    def _calculate_dist(self, path):
        d = 0
        for i in range(len(path) - 1):
            d += self.dist_matrix[path[i], path[i + 1]]
        d += self.dist_matrix[path[-1], path[0]]
        return d


class BAT(SwarmAlgorithm):
    def solve(self, fmin=0, fmax=2, loudness=0.5, pulse_rate=0.5):
        # Inicjalizacja
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dim))
        vel = np.zeros((self.n_particles, self.n_dim))
        freq = np.zeros(self.n_particles)

        scores = np.array([self.obj_func(p) for p in pos])
        best_idx = np.argmin(scores)
        self.best_score = scores[best_idx]
        self.best_pos = pos[best_idx].copy()

        for t in range(self.max_iter):
            for i in range(self.n_particles):
                # Echolokacja - zmiana częstotliwości i prędkości
                freq[i] = fmin + (fmax - fmin) * np.random.rand()
                vel[i] += (pos[i] - self.best_pos) * freq[i]
                new_pos = pos[i] + vel[i]

                # Lokalna modyfikacja (Random Walk)
                if np.random.rand() > pulse_rate:
                    new_pos = self.best_pos + 0.01 * np.random.randn(self.n_dim)

                new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
                new_score = self.obj_func(new_pos)

                # Akceptacja nowego rozwiązania
                if (new_score <= scores[i]) and (np.random.rand() < loudness):
                    pos[i] = new_pos
                    scores[i] = new_score

                # Aktualizacja globalnego lidera
                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_pos = new_pos.copy()

            self.history.append(self.best_score)
        return self.best_pos, self.best_score, self.history


class ABC(SwarmAlgorithm):
    def solve(self, limit=20):
        # n_particles w ABC to liczba źródeł pokarmu (połowa populacji pszczół)
        n_sources = self.n_particles // 2
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (n_sources, self.n_dim))
        scores = np.array([self.obj_func(p) for p in pos])
        trials = np.zeros(n_sources)  # licznik niepowodzeń dla źródła

        for t in range(self.max_iter):
            # FAZA ROBOTNIC (Employed Bees)
            for i in range(n_sources):
                phi = np.random.uniform(-1, 1, self.n_dim)
                k = np.random.randint(n_sources)
                while k == i: k = np.random.randint(n_sources)

                v = pos[i] + phi * (pos[i] - pos[k])
                v = np.clip(v, self.bounds[0], self.bounds[1])
                v_score = self.obj_func(v)

                if v_score < scores[i]:
                    pos[i], scores[i], trials[i] = v, v_score, 0
                else:
                    trials[i] += 1

            # FAZA OBSERWATOREK (Onlooker Bees)
            # Prawdopodobieństwo wyboru źródła (ruletka)
            probs = (1 / scores) / np.sum(1 / scores) if np.all(scores > 0) else np.ones(n_sources) / n_sources
            for _ in range(n_sources):
                i = np.random.choice(range(n_sources), p=probs)
                # ... identyczna logika jak u robotnic powyżej ...

            # FAZA ZWIADOWCÓW (Scout Bees)
            for i in range(n_sources):
                if trials[i] > limit:
                    pos[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.n_dim)
                    scores[i] = self.obj_func(pos[i])
                    trials[i] = 0

            self.best_score = np.min(scores)
            self.history.append(self.best_score)
        return pos[np.argmin(scores)], self.best_score, self.history


class WOA(SwarmAlgorithm):
    def solve(self):
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dim))
        self.best_score = float("inf")
        self.best_pos = None

        for t in range(self.max_iter):
            # Aktualizacja lidera
            for i in range(self.n_particles):
                score = self.obj_func(pos[i])
                if score < self.best_score:
                    self.best_score = score
                    self.best_pos = pos[i].copy()

            a = 2 - 2 * (t / self.max_iter)  # tak samo jak w GWO

            for i in range(self.n_particles):
                p = np.random.rand()
                if p < 0.5:
                    r = np.random.rand()
                    A, C = 2 * a * r - a, 2 * np.random.rand()
                    if np.abs(A) < 1:
                        D = np.abs(C * self.best_pos - pos[i])
                        pos[i] = self.best_pos - A * D
                    else:
                        rand_whale = pos[np.random.randint(self.n_particles)]
                        D = np.abs(C * rand_whale - pos[i])
                        pos[i] = rand_whale - A * D
                else:
                    # Ruch spiralny
                    D_best = np.abs(self.best_pos - pos[i])
                    L = np.random.uniform(-1, 1)
                    pos[i] = D_best * np.exp(0.5 * L) * np.cos(2 * np.pi * L) + self.best_pos

            pos = np.clip(pos, self.bounds[0], self.bounds[1])
            self.history.append(self.best_score)

        return self.best_pos, self.best_score, self.history


