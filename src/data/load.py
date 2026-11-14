"""Chargement du dataset NASA C-MAPSS (Turbofan Engine Degradation Simulation)."""
from pathlib import Path

import pandas as pd

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Racine du projet = deux niveaux au-dessus de src/data/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _read_txt(path: Path) -> pd.DataFrame:
    # Les fichiers sont séparés par des espaces, sans en-tête,
    # avec des espaces en fin de ligne : sep=r"\s+" gère tout ça.
    return pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)


def load_cmapss(subset: str = "FD001", data_dir: Path = RAW_DIR):
    """Charge un sous-ensemble C-MAPSS.

    Retourne
    --------
    train : DataFrame des trajectoires complètes jusqu'à la panne
    test : DataFrame des trajectoires tronquées avant la panne
    rul_test : Series de la vraie RUL au dernier cycle de chaque moteur de test
    """
    data_dir = Path(data_dir)
    train = _read_txt(data_dir / f"train_{subset}.txt")
    test = _read_txt(data_dir / f"test_{subset}.txt")
    rul_test = pd.read_csv(data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None).iloc[:, 0]
    rul_test.index = range(1, len(rul_test) + 1)  # index = numéro du moteur
    rul_test.name = "RUL"
    return train, test, rul_test


def add_rul(train: pd.DataFrame, clip: int | None = 125) -> pd.DataFrame:
    """Ajoute la colonne RUL = dernier cycle du moteur - cycle courant.

    clip : plafond de la RUL (RUL "piecewise linear"). Un moteur neuf ne montre
    aucun signe de dégradation, donc on ne demande pas au modèle de distinguer
    RUL=300 de RUL=200. Mettre None pour garder la RUL brute.
    """
    df = train.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    if clip is not None:
        df["RUL"] = df["RUL"].clip(upper=clip)
    return df


def constant_columns(df: pd.DataFrame, cols=None, tol: float = 1e-8) -> list[str]:
    """Liste des colonnes (quasi) constantes, sans information pour la prédiction."""
    cols = cols or SETTING_COLS + SENSOR_COLS
    return [c for c in cols if df[c].std() < tol]


def add_rul_test(test: pd.DataFrame, rul_test: pd.Series, clip: int | None = 125) -> pd.DataFrame:
    """Ajoute la vraie RUL à chaque ligne du jeu de test.

    RUL(ligne) = RUL au dernier cycle observé + (dernier cycle observé - cycle courant)
    """
    df = test.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = df["unit"].map(rul_test) + (last_cycle - df["cycle"])
    if clip is not None:
        df["RUL"] = df["RUL"].clip(upper=clip)
    return df
