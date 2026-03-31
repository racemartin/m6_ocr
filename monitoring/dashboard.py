# =============================================================================
# monitoring/dashboard.py — Tableau de bord Streamlit de supervision
# Visualise les prédictions en temps réel depuis predictions.jsonl
# et affiche le rapport de drift Evidently AI généré par drift_analysis.py
#
# Démarrage :
#   streamlit run monitoring/dashboard.py
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import json                                       # Lecture du fichier JSONL
import sys                                        # Accès au path Python
from   pathlib import Path                        # Chemins multi-OS

# --- Bibliothèques tierces : dashboard ---------------------------------------
import streamlit          as st                   # Interface web interactive
import pandas             as pd                   # Manipulation des données
import numpy              as np                   # Calculs statistiques
import plotly.express     as px                   # Graphiques interactifs
import plotly.graph_objects as go                 # Graphiques personnalisés

# --- Configuration -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FICHIER_PREDICTIONS,   # predictions.jsonl — source des données
    RACINE_PROJET,         # Racine pour localiser le rapport drift
)

# Chemin du rapport HTML Evidently
RAPPORT_DRIFT_HTML = RACINE_PROJET / "monitoring" / "drift_report.html"


# =============================================================================
# Configuration de la page Streamlit
# =============================================================================
st.set_page_config(
    page_title = "Prêt à Dépenser — Supervision",
    page_icon  = "📊",
    layout     = "wide",
)


# ##############################################################################
# Fonctions utilitaires
# ##############################################################################

# =============================================================================
@st.cache_data(ttl=30)
def charger_predictions() -> pd.DataFrame:
    """
    Charge et met en cache les prédictions depuis predictions.jsonl.

    Le cache est invalidé toutes les 30 secondes pour refléter
    les nouvelles prédictions sans recharger manuellement.

    Returns:
        DataFrame avec toutes les prédictions, ou DataFrame vide
        si le fichier n'existe pas encore.
    """
    if not FICHIER_PREDICTIONS.exists():
        return pd.DataFrame()

    lignes = []
    with open(FICHIER_PREDICTIONS, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                try:
                    lignes.append(json.loads(ligne))
                except json.JSONDecodeError:
                    continue  # Ignore les lignes corrompues

    if not lignes:
        return pd.DataFrame()

    df                       = pd.DataFrame(lignes)
    df["horodatage"]         = pd.to_datetime(df["horodatage"], utc=True)
    df                       = df.sort_values("horodatage").reset_index(drop=True)
    return df


# =============================================================================
def calculer_metriques_cles(df: pd.DataFrame) -> dict:
    """
    Calcule les métriques principales pour les KPI du dashboard.

    Args:
        df : DataFrame des prédictions chargées.

    Returns:
        Dictionnaire avec les métriques agrégées.
    """
    nb_total      = len(df)
    nb_approuves  = (df["decision"] == "Approuvé").sum()
    nb_rejetes    = (df["decision"] == "Rejeté").sum()
    taux_rejet    = nb_rejetes / nb_total * 100 if nb_total > 0 else 0.0
    prob_moyenne  = df["probabilite_defaut"].mean()
    latence_p50   = df["latence_ms"].quantile(0.50)
    latence_p95   = df["latence_ms"].quantile(0.95)
    latence_p99   = df["latence_ms"].quantile(0.99)

    return {
        "nb_total"     : nb_total,
        "nb_approuves" : nb_approuves,
        "nb_rejetes"   : nb_rejetes,
        "taux_rejet"   : taux_rejet,
        "prob_moyenne" : prob_moyenne,
        "latence_p50"  : latence_p50,
        "latence_p95"  : latence_p95,
        "latence_p99"  : latence_p99,
    }


# ##############################################################################
# Interface principale du dashboard
# ##############################################################################

# =============================================================================
def afficher_entete() -> None:
    """Affiche l'en-tête du dashboard avec titre et description."""
    st.title("📊 Supervision — Scoring Crédit")
    st.markdown(
        "Tableau de bord de surveillance des prédictions en temps réel. "
        "Rechargement automatique toutes les **30 secondes**."
    )
    st.divider()


# =============================================================================
def afficher_kpis(metriques: dict) -> None:
    """
    Affiche les indicateurs clés de performance (KPI) en colonnes.

    Args:
        metriques : Dictionnaire de métriques calculées.
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label = "Prédictions totales",
            value = f"{metriques['nb_total']:,}",
        )
    with col2:
        st.metric(
            label = "Approuvés",
            value = f"{metriques['nb_approuves']:,}",
        )
    with col3:
        st.metric(
            label      = "Taux de rejet",
            value      = f"{metriques['taux_rejet']:.1f} %",
        )
    with col4:
        st.metric(
            label = "Prob. défaut moyenne",
            value = f"{metriques['prob_moyenne']:.3f}",
        )
    with col5:
        st.metric(
            label = "Latence p95",
            value = f"{metriques['latence_p95']:.1f} ms",
        )


# =============================================================================
def afficher_graphique_decisions(df: pd.DataFrame) -> None:
    """
    Affiche l'évolution temporelle des décisions crédit.

    Regroupe les décisions par heure pour visualiser les tendances
    d'approbation et de rejet dans le temps.

    Args:
        df : DataFrame des prédictions avec colonne horodatage.
    """
    st.subheader("Évolution des décisions dans le temps")

    # -- Agrégation par heure ------------------------------------------------
    df_temp             = df.copy()
    df_temp["heure"]    = df_temp["horodatage"].dt.floor("h")
    df_groupe           = (
        df_temp
        .groupby(["heure", "decision"])
        .size()
        .reset_index(name="count")
    )

    fig = px.line(
        df_groupe,
        x          = "heure",
        y          = "count",
        color      = "decision",
        color_discrete_map = {
            "Approuvé" : "#2ecc71",
            "Rejeté"   : "#e74c3c",
        },
        labels     = {
            "heure"  : "Heure",
            "count"  : "Nombre de décisions",
            "decision": "Décision",
        },
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
def afficher_distribution_probabilites(df: pd.DataFrame) -> None:
    """
    Affiche la distribution des probabilités de défaut.

    Visualise l'histogramme des probabilités avec le seuil de décision
    superposé pour identifier les demandes en zone grise.

    Args:
        df : DataFrame des prédictions.
    """
    st.subheader("Distribution des probabilités de défaut")


    # Intentamos obtenerlo del DF, si no existe usamos 0.35 por defecto
    if "seuil_utilise" in df.columns:
        seuil = df["seuil_utilise"].iloc[-1] if len(df) > 0 else 0.35
    else:
        seuil = 0.35  # Valor por defecto seguro

    fig = px.histogram(
        df,
        x          = "probabilite_defaut",
        color      = "decision",
        nbins      = 50,
        opacity    = 0.75,
        color_discrete_map = {
            "Approuvé" : "#2ecc71",
            "Rejeté"   : "#e74c3c",
        },
        labels     = {
            "probabilite_defaut" : "Probabilité de défaut",
            "decision"           : "Décision",
        },
    )

    # -- Ligne verticale : seuil de décision ---------------------------------
    fig.add_vline(
        x           = seuil,
        line_dash   = "dash",
        line_color  = "#f39c12",
        annotation_text     = f"Seuil = {seuil:.2f}",
        annotation_position = "top right",
    )

    fig.update_layout(height=350, bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
def afficher_latences(df: pd.DataFrame) -> None:
    """
    Affiche les métriques de latence d'inférence (p50 / p95 / p99).

    Args:
        df : DataFrame des prédictions avec colonne latence_ms.
    """
    st.subheader("Latences d'inférence (ms)")

    col1, col2 = st.columns(2)

    with col1:
        # -- Tableau récapitulatif des percentiles ---------------------------
        percentiles = {
            "p50 (médiane)" : df["latence_ms"].quantile(0.50),
            "p75"           : df["latence_ms"].quantile(0.75),
            "p95"           : df["latence_ms"].quantile(0.95),
            "p99"           : df["latence_ms"].quantile(0.99),
            "maximum"       : df["latence_ms"].max(),
        }
        df_perc = pd.DataFrame(
            list(percentiles.items()),
            columns=["Percentile", "Latence (ms)"]
        )
        df_perc["Latence (ms)"] = df_perc["Latence (ms)"].round(2)
        st.dataframe(df_perc, use_container_width=True, hide_index=True)

    with col2:
        # -- Boîte à moustaches des latences ---------------------------------
        fig = px.box(
            df,
            y      = "latence_ms",
            labels = {"latence_ms": "Latence (ms)"},
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
def afficher_rapport_drift() -> None:
    """
    Affiche le rapport HTML Evidently AI dans un iframe Streamlit.

    Si le rapport n'existe pas, propose de le générer via un bouton.
    """
    st.subheader("🔍 Rapport de drift Evidently AI")

    if not RAPPORT_DRIFT_HTML.exists():
        st.warning(
            "Aucun rapport de drift disponible. "
            "Générez-le en cliquant sur le bouton ci-dessous."
        )
        if st.button("Générer le rapport de drift"):
            import subprocess
            script = RACINE_PROJET / "scripts" / "drift_analysis.py"
            with st.spinner("Analyse en cours..."):
                subprocess.run(
                    [sys.executable, str(script)],
                    capture_output = True,
                )
            st.rerun()
        return

    # -- Lecture et affichage du rapport HTML --------------------------------
    contenu_html = RAPPORT_DRIFT_HTML.read_text(encoding="utf-8")
    st.components.v1.html(contenu_html, height=800, scrolling=True)


# =============================================================================
def afficher_donnees_brutes(df: pd.DataFrame) -> None:
    """
    Affiche les 50 dernières prédictions dans un tableau interactif.

    Args:
        df : DataFrame des prédictions.
    """
    st.subheader("Dernières prédictions")

    colonnes_affichees = [
        "horodatage",
        "age",
        "revenu",
        "montant_pret",
        "probabilite_defaut",
        "score_risque",
        "decision",
        "latence_ms",
    ]
    colonnes_disponibles = [
        c for c in colonnes_affichees if c in df.columns
    ]

    st.dataframe(
        df[colonnes_disponibles].tail(50).iloc[::-1],
        use_container_width = True,
        hide_index          = True,
    )


# ##############################################################################
# Point d'entrée principal du dashboard
# ##############################################################################

# =============================================================================
def main() -> None:
    """
    Orchestration principale du tableau de bord Streamlit.

    Charge les données, vérifie leur disponibilité, et affiche
    les différentes sections du dashboard.
    """
    afficher_entete()

    # -- Chargement des données ----------------------------------------------
    df = charger_predictions()

    # -- Vérification de la disponibilité des données -----------------------
    if df.empty:
        st.info(
            "Aucune prédiction disponible. "
            "Effectuez des requêtes sur POST /predict pour commencer."
        )
        st.stop()

    # -- Calcul des métriques ------------------------------------------------
    metriques = calculer_metriques_cles(df)

    # -- Affichage des sections du dashboard ---------------------------------
    afficher_kpis(metriques)
    st.divider()

    col_gauche, col_droite = st.columns(2)
    with col_gauche:
        afficher_graphique_decisions(df)
    with col_droite:
        afficher_distribution_probabilites(df)

    st.divider()
    afficher_latences(df)

    st.divider()
    afficher_rapport_drift()

    st.divider()
    afficher_donnees_brutes(df)

    # -- Rechargement automatique toutes les 30 secondes --------------------
    st.caption("Rechargement automatique toutes les 30 secondes.")


# =============================================================================
if __name__ == "__main__":
    main()
