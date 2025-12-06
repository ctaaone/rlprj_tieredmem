import numpy as np
from dataclasses import dataclass

@dataclass
class LinUCBConfig:
    n_actions: int
    d: int              # Dimension
    alpha: float = 0.5  # Exploration coefficient
    l2: float = 1.0     # Ridge regularization

class LinUCBBandit:
    def __init__(self, cfg: LinUCBConfig):
        self.cfg = cfg
        self.As = np.stack([np.eye(cfg.d) * cfg.l2 for _ in range(cfg.n_actions)], axis=0)
        self.bs = np.zeros((cfg.n_actions, cfg.d), dtype=np.float64)
        self.counts = np.zeros(cfg.n_actions, dtype=np.int64)

    def _solve_A_inv_x(self, A, x): # A^{-1}x helper
        return np.linalg.solve(A, x)

    def _theta(self, a):
        return np.linalg.solve(self.As[a], self.bs[a])

    def act(self, obs): 
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        scores = np.empty(self.cfg.n_actions, dtype=np.float64)
        for a in range(self.cfg.n_actions):
            theta = self._theta(a)
            z = self._solve_A_inv_x(self.As[a], x)
            mean = float(x @ theta)
            bonus = self.cfg.alpha * np.sqrt(max(1e-12, float(x @ z)))
            scores[a] = mean + bonus
            # score = x^T θ + α sqrt(x^T A^{-1} x)
        return int(np.argmax(scores))

    def update(self, obs, action, reward):
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        r = float(reward)
        self.As[action] = self.As[action] + np.outer(x, x)
        self.bs[action] = self.bs[action] + r * x
        self.counts[action] += 1

    def save(self, path:str):
        np.savez(path, As=self.As, bs=self.bs, counts=self.counts,
                 n_actions=self.cfg.n_actions, d=self.cfg.d, alpha=self.cfg.alpha, l2=self.cfg.l2)

    def load(self, path:str):
        d = np.load(path, allow_pickle=False)
        self.As = d["As"]; self.bs = d["bs"]; self.counts = d["counts"]
        self.cfg.n_actions = int(d["n_actions"]); self.cfg.d = int(d["d"])
        self.cfg.alpha = float(d["alpha"]); self.cfg.l2 = float(d["l2"])
