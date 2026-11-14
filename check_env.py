"""Vérifie que toutes les bibliothèques du projet sont bien installées."""
import importlib
import sys

PACKAGES = ["numpy", "pandas", "matplotlib", "seaborn", "plotly", "sklearn",
            "lightgbm", "mapie", "torch", "shap", "streamlit", "pytest"]

print(f"Python {sys.version.split()[0]}\n")
ok = True
for name in PACKAGES:
    try:
        mod = importlib.import_module(name)
        print(f"  OK   {name:<12} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  ERREUR  {name} n'est pas installé")
        ok = False

# MAPIE doit être en v1 (nouvelle API)
try:
    from mapie.regression import SplitConformalRegressor, ConformalizedQuantileRegressor  # noqa: F401
    print("\n  OK   API MAPIE v1 disponible")
except ImportError:
    print("\n  ERREUR  MAPIE trop ancien : lance  pip install -U 'mapie>=1.0'")
    ok = False

print("\nTout est prêt !" if ok else "\nCorrige les erreurs ci-dessus.")
