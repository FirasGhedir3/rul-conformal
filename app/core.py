"""Logique du tableau de bord, séparée de l'interface (testable sans Streamlit)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.conformal.conformal import coverage, mean_width          # noqa: E402
from src.conformal.cross import MondrianCV, fold_predictions        # noqa: E402
from src.features.preprocess import FEATURES_FD001                 # noqa: E402
from src.features.rolling import add_rolling_features              # noqa: E402

RUL_CAP = 125  # la RUL est plafonnée à 125 : la borne haute ne peut pas dépasser cette valeur
MIN_CYCLE = 31  # comme dans le jeu de test : au moins 31 cycles observés

STATUS = {
    "critical": {"label": "Maintenance immédiate", "icon": "⛔", "color": "#d03b3b"},
    "warning": {"label": "Maintenance à planifier", "icon": "⚠️", "color": "#fab219"},
    "good": {"label": "Opérationnel", "icon": "✅", "color": "#0ca30c"},
}


def load_artifacts():
    """Charge les 5 modèles LightGBM et les résidus de calibration (notebook 05b)."""
    return joblib.load(ROOT / "models" / "conformal_cv_fd001.joblib")


def load_dataset(name: str) -> pd.DataFrame:
    """name : 'test' (moteurs officiels, coupés avant la panne) ou 'val' (suivis jusqu'à la panne)."""
    df = pd.read_csv(ROOT / "data" / "processed" / f"FD001_{name}.csv")
    return add_rolling_features(df, FEATURES_FD001, windows=(5, 10, 20))


def fit_conformal(art, confidence: float, lower_confidence: float) -> MondrianCV:
    """Recalibre Mondrian CV au niveau choisi (instantané : on réutilise les résidus sauvegardés)."""
    cal = art["calibration"]
    return MondrianCV(alpha=1 - confidence, alpha_lower=1 - lower_confidence).fit(cal.pred_oof, cal.RUL)


def predict(art, df: pd.DataFrame, mcv: MondrianCV) -> pd.DataFrame:
    """Ajoute prédiction (moyenne des 5 modèles), intervalle et borne basse à chaque ligne."""
    out = df[["unit", "cycle", "RUL"]].copy()
    out["pred"] = fold_predictions(art["models"], df[art["features"]]).mean(axis=1)
    lo, hi = mcv.predict(out["pred"])
    out["lo"], out["hi"] = lo, np.minimum(hi, RUL_CAP)
    out["L"] = mcv.lower_bound(out["pred"])
    return out


def status_of(lower_bound: float, threshold: float) -> str:
    """Décision prise sur la borne basse garantie, pas sur la prédiction :
    on agit dès que le scénario prudent passe sous le seuil."""
    if lower_bound <= threshold:
        return "critical"
    if lower_bound <= 2 * threshold:
        return "warning"
    return "good"


def fleet_snapshot(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """État de chaque moteur à son dernier cycle observé."""
    last = pred.sort_values(["unit", "cycle"]).groupby("unit").tail(1).copy()
    last["statut"] = [status_of(L, threshold) for L in last["L"]]
    return last.sort_values("L").reset_index(drop=True)


def reliability_curve(art, test_last: pd.DataFrame, levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    """Couverture observée sur le jeu de test pour plusieurs niveaux de confiance."""
    fp = fold_predictions(art["models"], test_last[art["features"]]).mean(axis=1)
    rows = []
    for lv in levels:
        m = fit_conformal(art, lv, 0.95)
        lo, hi = m.predict(fp)
        hi = np.minimum(hi, RUL_CAP)
        rows.append({"niveau visé": lv, "couverture observée": coverage(test_last.RUL, lo, hi),
                     "largeur moyenne": mean_width(lo, hi)})
    return pd.DataFrame(rows)


def coverage_by_bin(pred_last: pd.DataFrame) -> pd.DataFrame:
    t = pred_last.copy()
    t["tranche"] = pd.cut(t.RUL, bins=[-1, 25, 50, 75, 100, 125], labels=["0-25", "25-50", "50-75", "75-100", "100-125"])
    t["couvert"] = (t.RUL >= t.lo) & (t.RUL <= t.hi)
    t["largeur"] = t.hi - t.lo
    return t.groupby("tranche", observed=True).agg(couverture=("couvert", "mean"), largeur=("largeur", "mean"),
                                                   moteurs=("RUL", "size")).reset_index()
