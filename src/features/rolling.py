"""Features glissantes calculées par moteur, en n'utilisant QUE le passé."""
import numpy as np
import pandas as pd


def add_rolling_features(df: pd.DataFrame, sensors: list[str], windows=(5, 10, 20)) -> pd.DataFrame:
    """Ajoute pour chaque capteur et chaque fenêtre :
    - moyenne glissante (lisse le bruit)
    - écart-type glissant (instabilité du capteur)
    - pente sur la fenêtre (vitesse de dégradation)

    Les fenêtres ne regardent que les cycles passés et courant : aucune fuite
    du futur. Au début de vie, la fenêtre est plus courte (min_periods=1).
    """
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit", group_keys=False)
    new_cols = {}
    for w in windows:
        for s in sensors:
            roll = g[s].rolling(w, min_periods=1)
            new_cols[f"{s}_mean{w}"] = roll.mean().reset_index(level=0, drop=True)
            new_cols[f"{s}_std{w}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
            # pente = cov(cycle, capteur) / var(cycle) sur la fenêtre
            cov = (g[[s, "cycle"]].rolling(w, min_periods=2).cov()
                   .xs(s, level=-1)["cycle"].reset_index(level=0, drop=True))
            var = g["cycle"].rolling(w, min_periods=2).var().reset_index(level=0, drop=True)
            new_cols[f"{s}_slope{w}"] = (cov / var).fillna(0.0)
    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return out.replace([np.inf, -np.inf], 0.0)


def last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Garde la dernière ligne observée de chaque moteur (protocole d'évaluation officiel)."""
    return df.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
