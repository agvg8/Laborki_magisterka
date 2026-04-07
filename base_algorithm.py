from abc import ABC, abstractmethod
import numpy as np


class SwarmAlgorithm(ABC):
    def __init__(self, obj_func, n_dim, n_particles, max_iter, bounds):
        self.obj_func = obj_func
        self.n_dim = n_dim
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.bounds = bounds  # (min, max)

        # Statystyki zbieżności
        self.history = []
        self.best_score = float('inf')
        self.best_pos = None

    @abstractmethod
    def solve(self):
        """Główna pętla algorytmu"""
        pass