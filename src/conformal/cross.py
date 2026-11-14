"""Conformal prediction par validation croisée sur les moteurs (CV+).

Problème du split conformal de l'étape 5 : seuls 15 moteurs servent à la calibration,
les 70 autres ne servent qu'à l'entraînement. Le quantile q dépend donc beaucoup du
choix de ces 15 moteurs.

Idée du CV+ (Barber et al., 2021, « Predictive inference with the jackknife+ ») :
1. On répartit les 85 moteurs (train + calib) en K groupes.
2. Pour chaque groupe k, on entraîne un modèle μ_{-k} sur les K-1 autres groupes.
3. Chaque moteur du groupe k est prédit par μ_{-k}, un modèle qui ne l'a jamais vu :
   on obtient un résidu « hors échantillon » R_i pour TOUS les points de calibration.
4. Pour une nouvelle donnée x, chaque point i propose un intervalle
   [μ_{-k(i)}(x) - R_i, μ_{-k(i)}(x) + R_i], et on combine ces n propositions par quantiles.

Garantie théorique : couverture ≥ 1 - 2α (dans le pire cas) ; en pratique ≈ 1 - α.
Avantage : les 85 moteurs servent à la fois à entraîner et à calibrer.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.conformal.conformal import conformal_quantile, sample_cuts


def assign_folds(units, n_folds: int = 5, seed: int = 0) -> dict:
    """Répartit les moteurs en n_folds groupes de taille égale (aléatoirement)."""
    units = np.array(sorted(units))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    return {int(u): i % n_folds for i, u in enumerate(units)}


def fit_fold_models(pool: pd.DataFrame, feats, folds: dict, params: dict, verbose=True):
    """Entraîne un modèle LightGBM par groupe, chacun SANS les moteurs de ce groupe."""
    models = []
    for k in sorted(set(folds.values())):
        train_units = [u for u, f in folds.items() if f != k]
        d = pool[pool.unit.isin(train_units)]
        m = lgb.LGBMRegressor(**params).fit(d[feats], d.RUL)
        models.append(m)
        if verbose:
            print(f"groupe {k} : entraîné sur {len(train_units)} moteurs, {len(d)} lignes")
    return models


def oof_predictions(pool: pd.DataFrame, feats, folds: dict, models, n_cuts=20, min_cycle=31, seed=0):
    """Coupe chaque moteur à n_cuts cycles aléatoires et le prédit avec le modèle qui ne l'a pas vu."""
    cuts = sample_cuts(pool, n_cuts=n_cuts, min_cycle=min_cycle, seed=seed)
    cuts["fold"] = cuts.unit.map(folds)
    cuts["pred_oof"] = np.nan
    for k, m in enumerate(models):
        mask = cuts.fold == k
        cuts.loc[mask, "pred_oof"] = np.clip(m.predict(cuts.loc[mask, feats]), 0, None)
    return cuts


def fold_predictions(models, X) -> np.ndarray:
    """Matrice (n_points, K) : prédiction de chacun des K modèles."""
    return np.column_stack([np.clip(m.predict(X), 0, None) for m in models]).astype(np.float32)


class CVPlus:
    """Intervalles CV+ (codés à la main).

    Pour une nouvelle donnée x et chaque point de calibration i (du groupe k(i)) :
        bas_i  = μ_{-k(i)}(x) - R_i      haut_i = μ_{-k(i)}(x) + R_i
    Borne basse = le ⌊α(n+1)⌋-ième plus petit bas_i
    Borne haute = le ⌈(1-α)(n+1)⌉-ième plus petit haut_i
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, pred_oof, y, fold_idx):
        self.R_ = np.abs(np.asarray(y) - np.asarray(pred_oof)).astype(np.float32)
        self.fold_idx_ = np.asarray(fold_idx, dtype=int)
        n = len(self.R_)
        self.k_lo_ = max(int(np.floor(self.alpha * (n + 1))), 1) - 1
        self.k_hi_ = min(int(np.ceil((1 - self.alpha) * (n + 1))), n) - 1
        return self

    def predict(self, fold_preds: np.ndarray, chunk: int = 2000):
        """fold_preds : sortie de fold_predictions (n_points, K)."""
        lo, hi = [], []
        for s in range(0, len(fold_preds), chunk):
            mu = fold_preds[s:s + chunk][:, self.fold_idx_]           # (m, n) : μ_{-k(i)}(x)
            lo.append(np.partition(mu - self.R_, self.k_lo_, axis=1)[:, self.k_lo_])
            hi.append(np.partition(mu + self.R_, self.k_hi_, axis=1)[:, self.k_hi_])
        return np.clip(np.concatenate(lo), 0, None), np.concatenate(hi)


class MondrianCV:
    """Mondrian avec résidus hors échantillon : un quantile par tranche de RUL prédite.

    Prédiction ponctuelle = moyenne des K modèles. Les quantiles sont estimés sur les
    1700 résidus hors échantillon (85 moteurs) au lieu de 300 (15 moteurs).
    Donne aussi une borne basse unilatérale adaptative par tranche.
    """

    def __init__(self, alpha: float = 0.1, alpha_lower: float = 0.05, bins=(0, 25, 50, 75, 100, np.inf)):
        self.alpha, self.alpha_lower = alpha, alpha_lower
        self.bins = np.asarray(bins, dtype=float)
        self.labels = [f"{int(a)}-{int(b)}" if np.isfinite(b) else f"{int(a)}+" for a, b in zip(bins[:-1], bins[1:])]

    def group(self, pred):
        return np.clip(np.digitize(pred, self.bins[1:-1]), 0, len(self.bins) - 2)

    def fit(self, pred_oof, y):
        pred_oof, y = np.asarray(pred_oof), np.asarray(y)
        g = self.group(pred_oof)
        G = len(self.bins) - 1
        self.q_ = np.array([conformal_quantile(np.abs(y - pred_oof)[g == j], self.alpha) for j in range(G)])
        self.q_lower_ = np.array([conformal_quantile((pred_oof - y)[g == j], self.alpha_lower) for j in range(G)])
        self.n_ = np.bincount(g, minlength=G)
        return self

    def predict(self, pred):
        pred = np.asarray(pred)
        q = self.q_[self.group(pred)]
        return np.clip(pred - q, 0, None), pred + q

    def lower_bound(self, pred):
        pred = np.asarray(pred)
        return np.clip(pred - self.q_lower_[self.group(pred)], 0, None)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"tranche prédite": self.labels, "points de calibration": self.n_,
                             "demi-largeur (90 %)": self.q_.round(1), "recul borne basse (95 %)": self.q_lower_.round(1)})
