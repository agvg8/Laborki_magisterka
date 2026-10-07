import numpy as np

class Benchmarks:
    @staticmethod
    def rastrigin(x):
        # x: np.array (osobnik lub populacja)
        A = 10
        n = x.shape[-1]
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=-1)

    @staticmethod
    def schwefel(x):
        n = x.shape[-1]
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)

    @staticmethod
    def eggholder(x):
        # Zazwyczaj 2D
        x1 = x[..., 0]
        x2 = x[..., 1]
        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47)))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
        return term1 + term2

    @staticmethod
    def get_tsp_distance(path, distance_matrix):
        # path: permutacja miast [0, 3, 1, 2]
        dist = 0
        for i in range(len(path) - 1):
            dist += distance_matrix[path[i], path[i+1]]
        dist += distance_matrix[path[-1], path[0]] # powrót
        return dist

    @staticmethod
    def generate_tsp_data(n_cities, seed=42):
        np.random.seed(seed)
        coords = np.random.rand(n_cities, 2) * 100
        # Macierz odległości euklidesowych
        dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :])**2, axis=-1))
        return coords, dist_matrix