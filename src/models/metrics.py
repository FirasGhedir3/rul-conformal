"""Métriques d'évaluation de la RUL."""
import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """Score asymétrique du challenge PHM08 (plus bas = meilleur).

    d = prédiction - vérité
    - d < 0 (prédiction trop précoce) : pénalité exp(-d/13) - 1
    - d > 0 (prédiction trop tardive) : pénalité exp(d/10) - 1, plus forte

    Prédire une panne trop tard est plus dangereux que trop tôt :
    le moteur pourrait casser avant la maintenance.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
