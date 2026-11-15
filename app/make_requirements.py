"""Crée app/requirements.txt avec les versions EXACTES installées sur ton PC.

Les modèles sauvegardés (joblib) doivent être rechargés avec les mêmes versions de
LightGBM et scikit-learn que celles qui les ont créés : on fige donc les versions.
Lancer depuis la racine du projet :  python app/make_requirements.py
"""
from importlib.metadata import version
from pathlib import Path

PACKAGES = ["streamlit", "plotly", "pandas", "numpy", "scikit-learn", "lightgbm", "joblib"]
lines = [f"{p}=={version(p)}" for p in PACKAGES]
path = Path(__file__).parent / "requirements.txt"
path.write_text("\n".join(lines) + "\n")
print(f"{path} créé :")
print("\n".join(lines))
