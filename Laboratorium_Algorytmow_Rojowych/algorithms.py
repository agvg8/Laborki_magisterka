from abc import ABC, abstractmethod
import numpy as np


class SwarmAlgorithm(ABC):
    def __init__(self, obj_func, n_dim, n_particles, max_iter, bounds):
        self.obj_func = obj_func
        self.n_dim = n_dim
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.bounds = bounds

        self.history = []
        self.history_pos = []
        self.swarm_history = []

        self.best_score = float('inf')
        self.best_pos = None

    @abstractmethod
    def solve(self):
        pass


# ===================== PSO =====================
class PSO(SwarmAlgorithm):
    def solve(self, w=0.5, c1=1.5, c2=1.5):
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dim))
        vel = np.zeros((self.n_particles, self.n_dim))

        pbest_pos = pos.copy()
        pbest_score = np.array([self.obj_func(p) for p in pos])

        gbest_idx = np.argmin(pbest_score)
        self.best_pos = pbest_pos[gbest_idx].copy()
        self.best_score = pbest_score[gbest_idx]

        for _ in range(self.max_iter):
            r1 = np.random.rand(self.n_particles, self.n_dim)
            r2 = np.random.rand(self.n_particles, self.n_dim)

            vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (self.best_pos - pos)
            pos += vel
            pos = np.clip(pos, self.bounds[0], self.bounds[1])

            scores = np.array([self.obj_func(p) for p in pos])

            improved = scores < pbest_score
            pbest_score[improved] = scores[improved]
            pbest_pos[improved] = pos[improved]

            if np.min(pbest_score) < self.best_score:
                self.best_score = np.min(pbest_score)
                self.best_pos = pbest_pos[np.argmin(pbest_score)].copy()

            self.history.append(self.best_score)
            self.history_pos.append(self.best_pos.copy())
            self.swarm_history.append(pos.copy())

        return self.best_pos, self.best_score, self.history, self.history_pos


# ===================== GWO =====================
class GWO(SwarmAlgorithm):
    def solve(self):
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dim))

        alpha_pos = np.zeros(self.n_dim)
        beta_pos = np.zeros(self.n_dim)
        delta_pos = np.zeros(self.n_dim)

        alpha_score = beta_score = delta_score = float("inf")

        for t in range(self.max_iter):
            for i in range(self.n_particles):
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

            a = 2 - 2 * (t / self.max_iter)

            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - pos[i])
                X1 = alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * beta_pos - pos[i])
                X2 = beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * delta_pos - pos[i])
                X3 = delta_pos - A3 * D_delta

                pos[i] = (X1 + X2 + X3) / 3

            pos = np.clip(pos, self.bounds[0], self.bounds[1])

            self.history.append(alpha_score)
            self.history_pos.append(alpha_pos.copy())
            self.swarm_history.append(pos.copy())

        self.best_score = alpha_score
        self.best_pos = alpha_pos

        return self.best_pos, self.best_score, self.history, self.history_pos


# ===================== Firefly =====================
class Firefly(SwarmAlgorithm):
    def solve(self, alpha=0.5, beta0=1.0, gamma=1.0):
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dim))
        intensity = np.array([self.obj_func(p) for p in pos])

        best_idx = np.argmin(intensity)
        self.best_score = intensity[best_idx]
        self.best_pos = pos[best_idx].copy()

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                for j in range(self.n_particles):
                    if intensity[j] < intensity[i]:
                        r = np.linalg.norm(pos[i] - pos[j])
                        beta = beta0 * np.exp(-gamma * r ** 2)
                        rand_step = alpha * (np.random.rand(self.n_dim) - 0.5)

                        pos[i] += beta * (pos[j] - pos[i]) + rand_step
                        pos[i] = np.clip(pos[i], self.bounds[0], self.bounds[1])

            intensity = np.array([self.obj_func(p) for p in pos])

            best_idx = np.argmin(intensity)
            if intensity[best_idx] < self.best_score:
                self.best_score = intensity[best_idx]
                self.best_pos = pos[best_idx].copy()

            self.history.append(self.best_score)
            self.history_pos.append(self.best_pos.copy())
            self.swarm_history.append(pos.copy())

        return self.best_pos, self.best_score, self.history, self.history_pos


# ===================== BAT =====================
class BAT(SwarmAlgorithm):
    def solve(self, fmin=0, fmax=2, loudness=0.5, pulse_rate=0.5):
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dim))
        vel = np.zeros((self.n_particles, self.n_dim))
        freq = np.zeros(self.n_particles)

        scores = np.array([self.obj_func(p) for p in pos])
        best_idx = np.argmin(scores)

        self.best_score = scores[best_idx]
        self.best_pos = pos[best_idx].copy()

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                freq[i] = fmin + (fmax - fmin) * np.random.rand()
                vel[i] += (pos[i] - self.best_pos) * freq[i]
                new_pos = pos[i] + vel[i]

                if np.random.rand() > pulse_rate:
                    new_pos = self.best_pos + 0.01 * np.random.randn(self.n_dim)

                new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
                new_score = self.obj_func(new_pos)

                if (new_score <= scores[i]) and (np.random.rand() < loudness):
                    pos[i] = new_pos
                    scores[i] = new_score

                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_pos = new_pos.copy()

            self.history.append(self.best_score)
            self.history_pos.append(self.best_pos.copy())
            self.swarm_history.append(pos.copy())

        return self.best_pos, self.best_score, self.history, self.history_pos


# ===================== ABC =====================
class ABC(SwarmAlgorithm):
    def solve(self, limit=20):
        n_sources = self.n_particles // 2
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (n_sources, self.n_dim))
        scores = np.array([self.obj_func(p) for p in pos])
        trials = np.zeros(n_sources)

        for _ in range(self.max_iter):
            for i in range(n_sources):
                phi = np.random.uniform(-1, 1, self.n_dim)
                k = np.random.randint(n_sources)
                while k == i:
                    k = np.random.randint(n_sources)

                v = pos[i] + phi * (pos[i] - pos[k])
                v = np.clip(v, self.bounds[0], self.bounds[1])
                v_score = self.obj_func(v)

                if v_score < scores[i]:
                    pos[i], scores[i], trials[i] = v, v_score, 0
                else:
                    trials[i] += 1

            for i in range(n_sources):
                if trials[i] > limit:
                    pos[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.n_dim)
                    scores[i] = self.obj_func(pos[i])
                    trials[i] = 0

            self.best_score = np.min(scores)
            self.best_pos = pos[np.argmin(scores)].copy()

            self.history.append(self.best_score)
            self.history_pos.append(self.best_pos.copy())
            self.swarm_history.append(pos.copy())

        return self.best_pos, self.best_score, self.history, self.history_pos


# ===================== WOA =====================
class WOA(SwarmAlgorithm):
    def solve(self):
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dim))

        for t in range(self.max_iter):
            for i in range(self.n_particles):
                score = self.obj_func(pos[i])
                if score < self.best_score:
                    self.best_score = score
                    self.best_pos = pos[i].copy()

            a = 2 - 2 * (t / self.max_iter)

            for i in range(self.n_particles):
                p = np.random.rand()
                if p < 0.5:
                    r = np.random.rand()
                    A, C = 2 * a * r - a, 2 * np.random.rand()
                    if abs(A) < 1:
                        D = np.abs(C * self.best_pos - pos[i])
                        pos[i] = self.best_pos - A * D
                    else:
                        rand = pos[np.random.randint(self.n_particles)]
                        D = np.abs(C * rand - pos[i])
                        pos[i] = rand - A * D
                else:
                    D = np.abs(self.best_pos - pos[i])
                    L = np.random.uniform(-1, 1)
                    pos[i] = D * np.exp(0.5 * L) * np.cos(2 * np.pi * L) + self.best_pos

            pos = np.clip(pos, self.bounds[0], self.bounds[1])

            self.history.append(self.best_score)
            self.history_pos.append(self.best_pos.copy())
            self.swarm_history.append(pos.copy())

        return self.best_pos, self.best_score, self.history, self.history_pos

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
        swaps = []
        temp = list(source)
        for i in range(len(source)):
            if temp[i] != target[i]:
                j = temp.index(target[i])
                swaps.append((i, j))
                temp[i], temp[j] = temp[j], temp[i]
        return swaps

    def _apply_swaps(self, path, swaps, prob):
        new_path = list(path)
        for i, j in swaps:
            if np.random.rand() < prob:
                new_path[i], new_path[j] = new_path[j], new_path[i]
        return new_path

    def solve(self, alpha=0.9, beta=0.9):
        particles = [np.random.permutation(self.n_cities).tolist() for _ in range(self.n_particles)]
        pbest = list(particles)
        pbest_scores = [self._get_distance(p) for p in particles]

        gbest_idx = np.argmin(pbest_scores)
        gbest = list(pbest[gbest_idx])
        gbest_score = pbest_scores[gbest_idx]

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                sp = self._get_swap_sequence(particles[i], pbest[i])
                sg = self._get_swap_sequence(particles[i], gbest)

                particles[i] = self._apply_swaps(particles[i], sp, alpha)
                particles[i] = self._apply_swaps(particles[i], sg, beta)

                score = self._get_distance(particles[i])

                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = list(particles[i])

            if min(pbest_scores) < gbest_score:
                gbest_score = min(pbest_scores)
                gbest = list(pbest[np.argmin(pbest_scores)])

            self.history.append(gbest_score)

        return gbest, gbest_score, self.history


class ACO_TSP:
    def __init__(self, dist_matrix, n_ants, n_iter, alpha=1, beta=2, rho=0.1, q=100):
        self.dist_matrix = dist_matrix
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.n_cities = len(dist_matrix)

        self.pheromone = np.ones((self.n_cities, self.n_cities)) / self.n_cities
        self.history = []

    def solve(self):
        best_dist = float('inf')
        best_path = None

        for _ in range(self.n_iter):
            paths = [self._gen_path() for _ in range(self.n_ants)]
            dists = [self._dist(p) for p in paths]

            if min(dists) < best_dist:
                best_dist = min(dists)
                best_path = paths[np.argmin(dists)]

            self._update(paths, dists)
            self.history.append(best_dist)

        return best_path, best_dist, self.history

    def _gen_path(self):
        start = np.random.randint(self.n_cities)
        path = [start]
        visited = {start}

        while len(path) < self.n_cities:
            probs = self._probs(path[-1], visited)
            nxt = np.random.choice(range(self.n_cities), p=probs)
            path.append(nxt)
            visited.add(nxt)

        return path

    def _probs(self, city, visited):
        phi = self.pheromone[city].copy()
        eta = 1 / (self.dist_matrix[city] + 1e-10)

        for v in visited:
            phi[v] = 0
            eta[v] = 0

        w = (phi ** self.alpha) * (eta ** self.beta)
        if np.sum(w) == 0:
            w = np.ones_like(w)

        return w / np.sum(w)

    def _update(self, paths, dists):
        self.pheromone *= (1 - self.rho)

        for path, d in zip(paths, dists):
            val = self.q / d
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                self.pheromone[a, b] += val
                self.pheromone[b, a] += val

    def _dist(self, path):
        d = 0
        for i in range(len(path) - 1):
            d += self.dist_matrix[path[i], path[i + 1]]
        return d + self.dist_matrix[path[-1], path[0]]