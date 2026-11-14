"""Transformation des trajectoires en fenêtres temporelles pour le LSTM."""
import numpy as np
import pandas as pd


def make_windows(df: pd.DataFrame, features: list[str], window: int = 30, target: str = "RUL"):
    """Construit une fenêtre des `window` derniers cycles pour CHAQUE ligne.

    Pour la ligne (moteur u, cycle t), la fenêtre contient les cycles t-window+1 ... t
    du moteur u : uniquement le passé et le présent, jamais le futur.
    En début de vie (t < window), on complète à gauche en répétant le premier
    cycle observé (padding), ce qui permet de prédire dès le premier cycle,
    comme LightGBM.

    Retourne
    --------
    X : array (n_lignes, window, n_features)
    y : array (n_lignes,)  ou None si la colonne cible est absente
    meta : DataFrame (unit, cycle) aligné avec X, pour revenir aux moteurs
    """
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    Xs, metas = [], []
    for _, d in df.groupby("unit", sort=False):
        values = d[features].to_numpy(dtype=np.float32)
        pad = np.repeat(values[:1], window - 1, axis=0)
        padded = np.vstack([pad, values])               # (window-1 + T, f)
        # fenêtres glissantes : la fenêtre i se termine au cycle i
        idx = np.arange(len(values))[:, None] + np.arange(window)[None, :]
        Xs.append(padded[idx])                          # (T, window, f)
        metas.append(d[["unit", "cycle"]])
    X = np.concatenate(Xs)
    meta = pd.concat(metas).reset_index(drop=True)
    y = df[target].to_numpy(dtype=np.float32) if target in df else None
    return X, y, meta
