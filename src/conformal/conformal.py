"""Conformal prediction écrite à la main, pour comprendre (et vérifier) MAPIE.

Principe du split conformal
---------------------------
1. Un modèle déjà entraîné prédit ŷ.
2. Sur un ensemble de calibration jamais vu, on calcule des scores de non-conformité,
   par exemple s_i = |y_i - ŷ_i|.
3. On prend le quantile q de ces scores au niveau ceil((n+1)(1-α)) / n.
4. Intervalle pour une nouvelle donnée : [ŷ - q, ŷ + q].

Garantie : si les données de calibration et la nouvelle donnée sont échangeables,
P(y ∈ intervalle) ≥ 1 - α, quelle que soit la loi des données et quel que soit le modèle.
"""
import numpy as np
import pandas as pd


def sample_cuts(df: pd.DataFrame, n_cuts: int = 20, min_cycle: int = 31, seed: int = 42) -> pd.DataFrame:
    """Imite le protocole du jeu de test : chaque moteur est « coupé » à un cycle aléatoire.

    Les trajectoires de calibration vont jusqu'à la panne, alors que celles du test
    sont coupées avant (au moins 31 cycles observés). Comme les features glissantes
    n'utilisent que le passé, couper un moteur au cycle t revient simplement à prendre
    sa ligne au cycle t. On tire `n_cuts` cycles par moteur, sans remise.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for _, d in df[df["cycle"] >= min_cycle].groupby("unit"):
        k = min(n_cuts, len(d))
        parts.append(d.iloc[np.sort(rng.choice(len(d), size=k, replace=False))])
    return pd.concat(parts).reset_index(drop=True)


def conformal_quantile(scores, alpha: float) -> float:
    """Quantile conformal avec correction d'échantillon fini : le k-ième plus petit score,
    avec k = ceil((n+1)(1-α)). Renvoie +inf si n est trop petit pour garantir 1-α."""
    s = np.sort(np.asarray(scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.inf) if k > n else float(s[k - 1])


class SplitConformal:
    """Split conformal à score absolu : intervalle symétrique de largeur constante 2q."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, pred_cal, y_cal):
        self.q_ = conformal_quantile(np.abs(np.asarray(y_cal) - np.asarray(pred_cal)), self.alpha)
        return self

    def predict(self, pred):
        pred = np.asarray(pred)
        return np.clip(pred - self.q_, 0, None), pred + self.q_


class MondrianConformal:
    """Split conformal par groupe (Mondrian) : un quantile différent par tranche de RUL PRÉDITE.

    On ne peut pas grouper selon la vraie RUL (inconnue au moment de prédire), mais on
    peut grouper selon la prédiction. La garantie devient conditionnelle à chaque groupe :
    P(y ∈ intervalle | groupe) ≥ 1 - α.
    """

    def __init__(self, alpha: float = 0.1, bins=(0, 25, 50, 75, 100, np.inf)):
        self.alpha, self.bins = alpha, np.asarray(bins, dtype=float)

    def _group(self, pred):
        return np.clip(np.digitize(pred, self.bins[1:-1]), 0, len(self.bins) - 2)

    def fit(self, pred_cal, y_cal):
        pred_cal, y_cal = np.asarray(pred_cal), np.asarray(y_cal)
        g = self._group(pred_cal)
        scores = np.abs(y_cal - pred_cal)
        self.q_ = np.array([conformal_quantile(scores[g == j], self.alpha) for j in range(len(self.bins) - 1)])
        self.n_ = np.bincount(g, minlength=len(self.bins) - 1)
        return self

    def predict(self, pred):
        pred = np.asarray(pred)
        q = self.q_[self._group(pred)]
        return np.clip(pred - q, 0, None), pred + q


class LowerBoundConformal:
    """Borne basse unilatérale : « avec probabilité ≥ 1-α, il reste AU MOINS L cycles ».

    Score signé s = ŷ - y (positif quand le modèle surestime la RUL, le cas dangereux).
    L = ŷ - q, avec q le quantile conformal des scores signés.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def fit(self, pred_cal, y_cal):
        self.q_ = conformal_quantile(np.asarray(pred_cal) - np.asarray(y_cal), self.alpha)
        return self

    def predict(self, pred):
        return np.clip(np.asarray(pred) - self.q_, 0, None)


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))
