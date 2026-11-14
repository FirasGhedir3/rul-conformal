"""Sélection des capteurs et normalisation (ajustée sur le train uniquement)."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Conclusions de l'EDA sur FD001 (notebook 01) :
# - op3, s1, s5, s10, s16, s18, s19 : constants -> aucune information
# - s6 : ne prend que 2 valeurs (21.60 / 21.61), bruit de quantification -> supprimé
# - op1, op2 : une seule condition de vol dans FD001, variations = bruit -> supprimés
DROP_COLS_FD001 = ["op1", "op2", "op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]
FEATURES_FD001 = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
                  "s13", "s14", "s15", "s17", "s20", "s21"]


class SensorScaler:
    """StandardScaler appliqué aux capteurs, ajusté sur le train SEULEMENT.

    Ajuster sur toutes les données ferait fuiter des informations
    (moyenne, écart-type) des moteurs de calibration/validation/test.
    """

    def __init__(self, features=FEATURES_FD001):
        self.features = list(features)
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "SensorScaler":
        self.scaler.fit(df[self.features])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.features] = self.scaler.transform(df[self.features])
        return out
