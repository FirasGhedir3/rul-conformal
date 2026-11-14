# Maintenance prédictive avec quantification d'incertitude

Prédiction de la durée de vie restante (RUL) de moteurs d'avion à partir du dataset NASA C-MAPSS,
avec des intervalles de prédiction à couverture garantie par conformal prediction.

## Objectif
- Prédire la RUL de chaque moteur (LightGBM vs LSTM)
- Fournir un intervalle de confiance avec garantie statistique (MAPIE : split conformal, CQR)
- Visualiser le suivi des moteurs dans un tableau de bord Streamlit

## Structure
```
data/raw/          données brutes C-MAPSS (non versionnées)
data/processed/    données transformées
notebooks/         exploration et expériences
src/data/          chargement et découpage par moteur
src/features/      features glissantes
src/models/        LightGBM, LSTM
src/conformal/     conformal prediction
app/               tableau de bord Streamlit
tests/             tests unitaires
reports/figures/   figures du README
```

## Installation
```bash
conda create -n rul python=3.11
conda activate rul
pip install -r requirements.txt
python check_env.py
```

## Résultats
_À compléter._

## Données
A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
NASA Ames Prognostics Data Repository.
