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


def test_windows_shape_and_no_future():
    from src.data.windows import make_windows
    df = pd.DataFrame({"unit": [1] * 5 + [2] * 3, "cycle": [1, 2, 3, 4, 5, 1, 2, 3],
                       "s2": [1, 2, 3, 4, 5, 10, 20, 30.0], "RUL": [4, 3, 2, 1, 0, 2, 1, 0.0]})
    X, y, meta = make_windows(df, ["s2"], window=3)
    assert X.shape == (8, 3, 1) and len(y) == 8
    assert X[2, :, 0].tolist() == [1, 2, 3]       # cycles 1-3 du moteur 1
    assert X[0, :, 0].tolist() == [1, 1, 1]       # padding en début de vie
    assert X[5, :, 0].tolist() == [10, 10, 10]    # pas de mélange entre moteurs
    assert X[4, -1, 0] == 5                        # le dernier pas = cycle courant


def test_conformal_quantile_finite_sample():
    from src.conformal.conformal import conformal_quantile
    s = np.arange(1, 10)                      # n = 9 scores : 1..9
    assert conformal_quantile(s, 0.1) == 9    # k = ceil(10 * 0.9) = 9
    assert conformal_quantile(s, 0.2) == 8    # k = ceil(10 * 0.8) = 8
    assert np.isinf(conformal_quantile(s, 0.05))  # k = 10 > n : impossible de garantir 95 %


def test_split_conformal_coverage():
    from src.conformal.conformal import SplitConformal, coverage
    rng = np.random.default_rng(0)
    y_cal, y_new = rng.normal(size=2000), rng.normal(size=20000)
    sc = SplitConformal(alpha=0.1).fit(np.zeros(2000), y_cal)
    lo, hi = sc.predict(np.zeros(20000))
    lo = np.zeros(20000) - sc.q_               # pas de clip à 0 pour ce test
    assert abs(coverage(y_new, lo, hi) - 0.9) < 0.02


def test_cvplus_coverage():
    from src.conformal.cross import CVPlus
    from src.conformal.conformal import coverage
    rng = np.random.default_rng(0)
    n, K, m = 1000, 5, 5000
    fold_idx = np.arange(n) % K
    y_cal = rng.normal(size=n)
    cv = CVPlus(alpha=0.1).fit(np.zeros(n), y_cal, fold_idx)
    lo, hi = cv.predict(np.zeros((m, K), dtype=np.float32))
    lo = -hi                                      # pas de clip à 0 pour ce test
    assert abs(coverage(rng.normal(size=m), lo, hi) - 0.9) < 0.03
