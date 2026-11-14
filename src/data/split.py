"""Découpage des données PAR MOTEUR (jamais par ligne) pour éviter les fuites."""
import numpy as np
import pandas as pd


def split_units(df: pd.DataFrame, fracs=(0.70, 0.15, 0.15), seed: int = 42) -> dict[str, np.ndarray]:
    """Répartit les moteurs en trois groupes disjoints : train / calib / val.

    - train : entraînement des modèles
    - calib : calibration de la conformal prediction (jamais vue à l'entraînement)
    - val   : choix des hyperparamètres, early stopping, évaluation intermédiaire
    Le jeu de test officiel (test_FD001) reste intouché jusqu'à l'évaluation finale.
    """
    assert abs(sum(fracs) - 1) < 1e-9, "Les proportions doivent sommer à 1"
    units = np.sort(df["unit"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n = len(units)
    n_train = int(round(fracs[0] * n))
    n_calib = int(round(fracs[1] * n))
    return {
        "train": np.sort(units[:n_train]),
        "calib": np.sort(units[n_train:n_train + n_calib]),
        "val": np.sort(units[n_train + n_calib:]),
    }


def apply_split(df: pd.DataFrame, groups: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    """Renvoie un DataFrame par groupe de moteurs."""
    return {name: df[df["unit"].isin(u)].reset_index(drop=True) for name, u in groups.items()}
