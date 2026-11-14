import numpy as np
import pandas as pd

from src.data.load import add_rul, add_rul_test
from src.data.split import split_units


def _fake(n_units=20, length=10):
    return pd.DataFrame({
        "unit": np.repeat(np.arange(1, n_units + 1), length),
        "cycle": np.tile(np.arange(1, length + 1), n_units),
    })


def test_split_disjoint_and_complete():
    df = _fake()
    g = split_units(df)
    tr, ca, va = set(g["train"]), set(g["calib"]), set(g["val"])
    assert not (tr & ca) and not (tr & va) and not (ca & va)   # aucun moteur partagé
    assert tr | ca | va == set(df["unit"])                     # aucun moteur perdu


def test_rul_train():
    df = add_rul(_fake(n_units=1, length=200), clip=125)
    assert df["RUL"].iloc[-1] == 0          # dernier cycle = panne
    assert df["RUL"].max() == 125           # plafonnement


def test_rul_test():
    df = _fake(n_units=2, length=5)
    rul = pd.Series([10, 3], index=[1, 2])
    out = add_rul_test(df, rul, clip=None)
    assert out.loc[out.unit == 1, "RUL"].tolist() == [14, 13, 12, 11, 10]


def test_rolling_uses_only_past():
    from src.features.rolling import add_rolling_features
    df = pd.DataFrame({"unit": [1] * 6, "cycle": range(1, 7), "s2": [0, 0, 0, 0, 0, 100.0]})
    out = add_rolling_features(df, ["s2"], windows=(3,))
    # la valeur 100 au cycle 6 ne doit pas influencer les cycles précédents
    assert out["s2_mean3"].iloc[:5].abs().max() == 0


def test_nasa_score_late_is_worse():
    from src.models.metrics import nasa_score
    assert nasa_score([50], [60]) > nasa_score([50], [40])
