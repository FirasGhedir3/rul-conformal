"""Tableau de bord de maintenance prédictive – lancer avec :  streamlit run app/app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (MIN_CYCLE, ROOT, RUL_CAP, STATUS, coverage_by_bin, fit_conformal, fleet_snapshot,
                  load_artifacts, load_dataset, predict, reliability_curve, status_of)

st.set_page_config(page_title="Maintenance prédictive – RUL", page_icon="✈️", layout="wide")

# Couleurs (palette de référence, validée pour le daltonisme)
C_PRED, C_BAND = "#2a78d6", "rgba(42,120,214,0.18)"
C_LOWER, C_TRUTH, C_MUTED, C_GRID = "#eb6834", "#0b0b0b", "#898781", "#e8e7e3"


def style(fig, height=420):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
                      font=dict(color="#52514e"))
    fig.update_xaxes(gridcolor=C_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=C_GRID, zeroline=False)
    return fig


# ---------------------------------------------------------------- chargement
@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def get_dataset(name):
    return load_dataset(name)


@st.cache_data
def get_predictions(name, confidence, lower_confidence):
    art = get_artifacts()
    mcv = fit_conformal(art, confidence, lower_confidence)
    df = get_dataset(name)
    return predict(art, df[df.cycle >= MIN_CYCLE], mcv), mcv.table()


if not (ROOT / "models" / "conformal_cv_fd001.joblib").exists():
    st.error("Modèles introuvables. Exécute d'abord les notebooks 02, 03 et 05b pour créer "
             "`data/processed/` et `models/conformal_cv_fd001.joblib`.")
    st.stop()

# ---------------------------------------------------------------- barre latérale
st.sidebar.title("✈️ Réglages")
dataset = st.sidebar.radio("Moteurs", ["test", "val"], format_func=lambda x: {
    "test": "Test officiel NASA (100 moteurs, coupés avant la panne)",
    "val": "Validation (15 moteurs, suivis jusqu'à la panne)"}[x])
confidence = st.sidebar.select_slider("Niveau de confiance de l'intervalle", [0.80, 0.90, 0.95], value=0.90,
                                      format_func=lambda v: f"{v:.0%}")
lower_conf = st.sidebar.select_slider("Garantie de la borne basse", [0.90, 0.95, 0.99], value=0.95,
                                      format_func=lambda v: f"{v:.0%}")
threshold = st.sidebar.slider("Seuil de maintenance (cycles)", 5, 60, 25,
                              help="Maintenance immédiate si la borne basse passe sous ce seuil, "
                                   "à planifier si elle passe sous le double.")
show_truth = st.sidebar.checkbox("Afficher la vraie RUL (démonstration)", value=True)
st.sidebar.caption("Modèle : 5 LightGBM (validation croisée sur 85 moteurs) · "
                   "Incertitude : conformal prediction Mondrian CV · Données : NASA C-MAPSS FD001")

pred, qtable = get_predictions(dataset, confidence, lower_conf)

st.title("Maintenance prédictive de moteurs d'avion")
st.caption(f"Durée de vie restante (RUL) avec intervalle garanti à {confidence:.0%} "
           f"et borne basse garantie à {lower_conf:.0%}")

tab_engine, tab_fleet, tab_rel, tab_about = st.tabs(
    ["🔧 Suivi d'un moteur", "🛩️ Flotte", "📏 Fiabilité des intervalles", "ℹ️ Méthode"])

# ---------------------------------------------------------------- 1. suivi d'un moteur
with tab_engine:
    units = sorted(pred.unit.unique())
    c1, c2 = st.columns([1, 3])
    unit = c1.selectbox("Moteur", units)
    d = pred[pred.unit == unit].sort_values("cycle")
    cmin, cmax = int(d.cycle.min()), int(d.cycle.max())
    current = c2.slider("Rejouer l'historique : cycle actuel", cmin, cmax, cmax) if cmax > cmin else cmax
    hist = d[d.cycle <= current]
    now = hist.iloc[-1]
    s = STATUS[status_of(now.L, threshold)]

    k = st.columns(5)
    k[0].metric("Cycle actuel", int(now.cycle))
    k[1].metric("RUL prédite (cycles)", f"{now.pred:.0f}")
    k[2].metric(f"Intervalle {confidence:.0%}", f"{now.lo:.0f} – {now.hi:.0f}")
    k[3].metric(f"Borne basse {lower_conf:.0%} (cycles)", f"≥ {now.L:.0f}")
    k[4].markdown(f"**Statut**<br><span style='color:{s['color']};font-size:1.3rem;font-weight:600'>"
                  f"{s['icon']} {s['label']}</span>", unsafe_allow_html=True)

    st.info(f"Avec une garantie de {lower_conf:.0%}, ce moteur peut encore voler **au moins {now.L:.0f} cycles**. "
            f"Valeur la plus probable : {now.pred:.0f} cycles ; il y a {confidence:.0%} de chances que la vraie "
            f"durée de vie restante soit entre {now.lo:.0f} et {now.hi:.0f} cycles.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.cycle, y=hist.hi, line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=hist.cycle, y=hist.lo, fill="tonexty", fillcolor=C_BAND, line=dict(width=0),
                             name=f"Intervalle {confidence:.0%}",
                             customdata=np.c_[hist.lo, hist.hi],
                             hovertemplate="Intervalle : %{customdata[0]:.0f} – %{customdata[1]:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=hist.cycle, y=hist.pred, line=dict(color=C_PRED, width=2), name="RUL prédite",
                             hovertemplate="Prédite : %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=hist.cycle, y=hist.L, line=dict(color=C_LOWER, width=2),
                             name=f"Borne basse {lower_conf:.0%}", hovertemplate="Borne basse : %{y:.0f}<extra></extra>"))
    if show_truth:
        fig.add_trace(go.Scatter(x=hist.cycle, y=hist.RUL, line=dict(color=C_TRUTH, width=1.5, dash="dash"),
                                 name="Vraie RUL", hovertemplate="Vraie : %{y:.0f}<extra></extra>"))
    fig.add_hline(y=threshold, line=dict(color=C_MUTED, dash="dot", width=1.5),
                  annotation_text=f"seuil {threshold}", annotation_position="top left")
    fig.update_layout(xaxis_title="Cycle", yaxis_title="RUL (cycles)", yaxis_range=[0, RUL_CAP + 10],
                      xaxis_range=[cmin, cmax], title=f"Moteur {unit}")
    st.plotly_chart(style(fig, 460), width="stretch")
    if dataset == "test":
        st.caption("Les moteurs de test sont coupés avant la panne : la courbe s'arrête au dernier cycle observé.")

# ---------------------------------------------------------------- 2. flotte
with tab_fleet:
    snap = fleet_snapshot(pred, threshold)
    counts = snap.statut.value_counts()
    k = st.columns(3)
    for col, key in zip(k, ["critical", "warning", "good"]):
        col.metric(f"{STATUS[key]['icon']} {STATUS[key]['label']}", int(counts.get(key, 0)))

    fig = go.Figure()
    for key in ["critical", "warning", "good"]:
        part = snap[snap.statut == key]
        fig.add_trace(go.Bar(x=part.unit.astype(str), y=part.L, marker_color=STATUS[key]["color"],
                             name=f"{STATUS[key]['icon']} {STATUS[key]['label']}",
                             customdata=np.c_[part.pred, part.lo, part.hi],
                             hovertemplate="Moteur %{x}<br>Borne basse : %{y:.0f}<br>Prédite : %{customdata[0]:.0f}"
                                           "<br>Intervalle : %{customdata[1]:.0f} – %{customdata[2]:.0f}<extra></extra>"))
    fig.add_hline(y=threshold, line=dict(color=C_MUTED, dash="dot"), annotation_text="seuil", annotation_position="top left")
    fig.add_hline(y=2 * threshold, line=dict(color=C_MUTED, dash="dot"), annotation_text="2 × seuil", annotation_position="top left")
    fig.update_layout(title=f"Borne basse garantie à {lower_conf:.0%} de chaque moteur (triés)", barmode="group",
                      xaxis=dict(title="Moteur", type="category", categoryorder="array",
                                 categoryarray=snap.unit.astype(str)),
                      yaxis_title="Cycles restants (au moins)", hovermode="closest", bargap=0.15)
    st.plotly_chart(style(fig, 420), width="stretch")

    table = snap.assign(Statut=[f"{STATUS[s]['icon']} {STATUS[s]['label']}" for s in snap.statut])
    table = table.rename(columns={"unit": "Moteur", "cycle": "Cycles observés", "pred": "RUL prédite",
                                  "lo": "Intervalle bas", "hi": "Intervalle haut", "L": "Borne basse", "RUL": "Vraie RUL"})
    cols = ["Moteur", "Statut", "Cycles observés", "Borne basse", "RUL prédite", "Intervalle bas", "Intervalle haut"]
    if show_truth:
        cols.append("Vraie RUL")
    st.dataframe(table[cols].round(0), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 3. fiabilité
with tab_rel:
    st.markdown("La garantie de la conformal prediction se **vérifie** : sur les 100 moteurs de test, jamais vus "
                "pendant l'entraînement ni la calibration, la vraie RUL doit tomber dans l'intervalle au moins aussi "
                "souvent que le niveau annoncé.")
    art = get_artifacts()
    test_df = get_dataset("test")
    test_last = test_df.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    curve = reliability_curve(art, test_last)

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.45, 1], y=[0.45, 1], line=dict(color=C_MUTED, dash="dash"), name="Couverture parfaite",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=curve["niveau visé"], y=curve["couverture observée"], mode="lines+markers",
                             line=dict(color=C_PRED, width=2), marker=dict(size=9), name="Couverture observée (test)",
                             hovertemplate="Visé %{x:.0%} → observé %{y:.0%}<extra></extra>"))
    fig.update_layout(title="Couverture annoncée contre observée", xaxis=dict(title="Niveau visé", tickformat=".0%"),
                      yaxis=dict(title="Couverture observée", tickformat=".0%"), hovermode="closest")
    c1.plotly_chart(style(fig, 380), width="stretch")

    snap_test = predict(art, test_last, fit_conformal(art, confidence, lower_conf))
    bins = coverage_by_bin(snap_test)
    fig = go.Figure(go.Bar(x=bins.tranche, y=bins.couverture, marker_color=C_PRED, customdata=np.c_[bins.largeur, bins.moteurs],
                           hovertemplate="RUL %{x}<br>Couverture : %{y:.0%}<br>Largeur moyenne : %{customdata[0]:.0f}"
                                         "<br>Moteurs : %{customdata[1]}<extra></extra>", name="Couverture"))
    fig.add_hline(y=confidence, line=dict(color=C_MUTED, dash="dot"))
    fig.update_layout(title=f"Couverture par tranche de vraie RUL (pointillés : objectif {confidence:.0%})",
                      xaxis_title="Vraie RUL", yaxis=dict(title="Couverture", tickformat=".0%", range=[0, 1.05]),
                      hovermode="closest", showlegend=False)
    c2.plotly_chart(style(fig, 380), width="stretch")

    L_ok = np.mean(snap_test.RUL >= snap_test.L)
    st.metric(f"Borne basse respectée sur le test (objectif ≥ {lower_conf:.0%})", f"{L_ok:.0%}")
    st.markdown("**Largeur de l'intervalle selon la RUL prédite** (Mondrian CV) : l'incertitude est faible quand la "
                "panne est proche, plus forte au début de la dégradation.")
    st.dataframe(qtable, hide_index=True, width="stretch")
    st.caption("Limite connue : dans la tranche 50-75 cycles, le modèle a tendance à annoncer la panne trop tard ; "
               "la garantie porte sur la prédiction, pas sur la vraie RUL inconnue, et la couverture y reste un peu "
               "sous l'objectif.")

# ---------------------------------------------------------------- 4. méthode
with tab_about:
    st.markdown(f"""
### Le problème
Prédire combien de cycles de vol il reste à un moteur avant la panne (**RUL**, *Remaining Useful Life*),
à partir de 14 capteurs (températures, pressions, vitesses). Données : NASA C-MAPSS FD001.

### Le modèle
- 141 variables : capteurs + moyennes, écarts-types et pentes glissants sur 5, 10 et 20 cycles.
- **LightGBM**, préféré à un LSTM après comparaison (RMSE au dernier cycle du test : 13,5 contre 15,7).
- 5 modèles entraînés en validation croisée sur 85 moteurs ; la prédiction affichée est leur moyenne.

### L'incertitude : conformal prediction
1. Chaque moteur de calibration est prédit par le modèle qui ne l'a jamais vu → 1 700 erreurs « honnêtes ».
2. Pour chaque tranche de RUL prédite, on prend le quantile de ces erreurs (**Mondrian**) : l'intervalle est
   étroit près de la panne et plus large au début de la dégradation.
3. **Garantie** : si les nouveaux moteurs ressemblent aux moteurs de calibration, la vraie RUL tombe dans
   l'intervalle avec la probabilité annoncée, sans aucune hypothèse sur la loi des erreurs.

### La décision de maintenance
Le statut repose sur la **borne basse garantie**, pas sur la prédiction : on planifie la maintenance dès que
le scénario prudent (« il reste *au moins* L cycles ») passe sous le seuil.
- ⛔ borne basse ≤ seuil ({threshold} cycles) : maintenance immédiate
- ⚠️ borne basse ≤ 2 × seuil : maintenance à planifier
- ✅ sinon : opérationnel
""")
