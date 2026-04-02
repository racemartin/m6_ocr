# =============================================================================
# scripts/drift_analysis.py — Analyse de drift avec Evidently AI
# Compare la distribution des prédictions récentes (predictions.jsonl)
# avec les données d'entraînement (reference_data.csv) pour détecter
# tout drift de features ou de la sortie du modèle.
#
# Evidently AI est 100% open source (bibliothèque Python, pas le cloud).
# Produit un rapport HTML interactif + un résumé JSON optionnel.
#
# Utilisation :
#   python scripts/drift_analysis.py                  → rapport HTML
#   python scripts/drift_analysis.py --format json    → résumé JSON (API)
#   python scripts/drift_analysis.py --nb-lignes 200  → 200 dernières
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import argparse                                       # Arguments CLI
import json                                           # Sortie JSON (mode API)
import logging                                        # Journalisation
import sys                                            # Code sortie, path
from   pathlib import Path                            # Chemins multi-OS

# --- Bibliothèques tierces : données -----------------------------------------
import pandas as pd                                   # Lecture JSONL et CSV

# --- Bibliothèques tierces : Evidently AI ------------------------------------
from   evidently.report        import Report          # Rapport principal
from   evidently.metric_preset import DataDriftPreset # Preset drift features
from   evidently.metric_preset import DataQualityPreset  # Qualité données
from   evidently.metrics       import DatasetDriftMetric  # Drift global dataset

# --- Configuration -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FICHIER_PREDICTIONS,   # predictions.jsonl (données courantes)
    FICHIER_REFERENCE,     # reference_data.csv (données entraînement)
    RACINE_PROJET,         # Racine pour le rapport HTML
)


# Configuration journalisation
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
)
journal = logging.getLogger(__name__)

import os
import inspect
from src.tools.rafael.log_tool import LogTool
log = LogTool(origin="drift")
NOM_FICHIER = os.path.basename(__file__)

# Limpiar handlers duplicados
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


# Chemin du rapport HTML Evidently généré
RAPPORT_HTML = RACINE_PROJET / "monitoring" / "drift_report.html"

# Features numériques analysées pour le drift (+ sortie modèle)
FEATURES_NUMERIQUES = [
    "age",
    "revenu",
    "montant_pret",
    "duree_pret_mois",
    "jours_retard_moyen",
    "taux_incidents",
    "taux_utilisation_credit",
    "nb_comptes_ouverts",
    "probabilite_defaut",    # Drift sur la sortie du modèle = concept drift
]


# ##############################################################################
# Fonctions de chargement des données
# ##############################################################################

# =============================================================================
def analyser_arguments() -> argparse.Namespace:
    """
    Analyse les arguments de la ligne de commande.

    Returns:
        Namespace avec format de sortie et nombre de lignes à analyser.
    """
    analyseur = argparse.ArgumentParser(
        description = "Analyse de drift Evidently AI sur predictions.jsonl",
    )
    analyseur.add_argument(
        "--format",
        choices = ["html", "json"],
        default = "html",
        help    = "Format de sortie : html (rapport) ou json (résumé API)",
    )
    analyseur.add_argument(
        "--nb-lignes",
        type    = int,
        default = 0,
        help    = "Nombre de prédictions récentes à analyser (0 = toutes)",
    )
    return analyseur.parse_args()


# =============================================================================
def charger_predictions(nb_lignes: int = 0) -> pd.DataFrame:
    """
    Charge les prédictions depuis le fichier predictions.jsonl.

    Lit chaque ligne JSON indépendamment (format JSONL).
    Si nb_lignes > 0, ne garde que les N dernières prédictions.

    Args:
        nb_lignes : Nombre de lignes récentes (0 = toutes).

    Returns:
        DataFrame pandas avec toutes les prédictions chargées.

    Raises:
        FileNotFoundError : Si predictions.jsonl n'existe pas.
        ValueError        : Si le fichier est vide.
    """
    if not FICHIER_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Fichier de prédictions introuvable : {FICHIER_PREDICTIONS}\n"
            "Effectuez des requêtes sur POST /predict d'abord."
        )

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
        raise ValueError(
            "Le fichier predictions.jsonl est vide. "
            "Effectuez des requêtes sur POST /predict d'abord."
        )

    df = pd.DataFrame(lignes)

    # -- Sélection des N dernières lignes si demandé -------------------------
    if nb_lignes > 0:
        df = df.tail(nb_lignes).reset_index(drop=True)

    journal.info(
        "Prédictions chargées : %d lignes | %d colonnes",
        len(df), len(df.columns),
    )
    return df


# =============================================================================
def charger_reference() -> pd.DataFrame:
    """
    Charge les données de référence (dataset d'entraînement).

    Le fichier reference_data.csv est copié par export_best_model.py
    depuis m6_ocr lors de la préparation du déploiement.

    Returns:
        DataFrame avec les données de référence pour comparaison.

    Raises:
        FileNotFoundError : Si reference_data.csv est introuvable.
    """
    if not FICHIER_REFERENCE.exists():
        raise FileNotFoundError(
            f"Données de référence introuvables : {FICHIER_REFERENCE}\n"
            "Exécutez : python scripts/export_best_model.py"
        )

    df = pd.read_csv(FICHIER_REFERENCE)
    journal.info("Données de référence chargées : %d lignes", len(df))
    return df


# ##############################################################################
# Fonctions d'analyse
# ##############################################################################

# =============================================================================
def harmoniser_colonnes(
    df_ref : pd.DataFrame,
    df_cur : pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligne les colonnes entre les deux DataFrames pour Evidently.

    Garde uniquement les features communes présentes dans les deux
    datasets. Les colonnes absentes d'un côté sont ignorées.

    Args:
        df_ref : Données de référence (entraînement).
        df_cur : Données courantes (prédictions récentes).

    Returns:
        Tuple (df_ref, df_cur) avec colonnes communes uniquement.
    """
    features_communes = [
        col for col in FEATURES_NUMERIQUES
        if col in df_ref.columns and col in df_cur.columns
    ]

    journal.info(
        "Features communes pour le drift : %s", features_communes
    )

    return df_ref[features_communes].copy(), df_cur[features_communes].copy()


# =============================================================================
def generer_rapport_html(
    df_ref : pd.DataFrame,
    df_cur : pd.DataFrame,
) -> None:
    """
    Génère un rapport HTML Evidently complet et l'enregistre.

    Inclut :
        - DataDriftPreset  : drift de chaque feature (test KS)
        - DataQualityPreset: valeurs manquantes et statistiques descriptives

    Args:
        df_ref : Données de référence alignées.
        df_cur : Données courantes alignées.
    """
    rapport = Report(metrics=[
        DataDriftPreset(),       # Drift de toutes les features
        DataQualityPreset(),     # Qualité et distributions
    ])

    journal.info("Calcul du rapport Evidently AI en cours...")
    rapport.run(reference_data=df_ref, current_data=df_cur)

    RAPPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    rapport.save_html(str(RAPPORT_HTML))
    print(f"  Rapport HTML généré.....: {RAPPORT_HTML}")


# =============================================================================
def generer_resume_json(
    df_ref : pd.DataFrame,
    df_cur : pd.DataFrame,
) -> dict:
    """
    Génère un résumé JSON du drift pour l'endpoint GET /drift/report.

    Plus rapide que le rapport HTML complet : n'exécute que la métrique
    DatasetDriftMetric pour obtenir les indicateurs globaux.

    Args:
        df_ref : Données de référence alignées.
        df_cur : Données courantes alignées.

    Returns:
        Dictionnaire avec statut drift et métriques principales.
    """
    rapport = Report(metrics=[DatasetDriftMetric()])
    rapport.run(reference_data=df_ref, current_data=df_cur)

    resultats  = rapport.as_dict()
    metriques  = resultats.get("metrics", [{}])[0].get("result", {})

    return {
        "statut"              : "ok",
        "drift_detecte"       : metriques.get("dataset_drift", False),
        "nb_features_driftees": metriques.get("number_of_drifted_columns", 0),
        "nb_features_total"   : metriques.get("number_of_columns", 0),
        "nb_predictions"      : len(df_cur),
        "nb_reference"        : len(df_ref),
    }


# ##############################################################################
# Point d'entrée principal
# ##############################################################################

# =============================================================================
def main() -> None:
    """
    Orchestration principale du script d'analyse drift.

    Charge les deux datasets, aligne les colonnes et génère
    le rapport dans le format demandé (HTML ou JSON).
    """
    args = analyser_arguments()

    print("\n============================================================================")
    print("ANALYSE DE DRIFT — EVIDENTLY AI")
    print("============================================================================")

    # -- Chargement des données ---------------------------------------------
    try:
        df_cur = charger_predictions(args.nb_lignes)
        df_ref = charger_reference()
    except (FileNotFoundError, ValueError) as erreur:
        print(f"\n  ERREUR : {erreur}", file=sys.stderr)
        sys.exit(1)

    print(f"  Prédictions courantes...: {len(df_cur)} lignes")
    print(f"  Données de référence....: {len(df_ref)} lignes")

    # -- Harmonisation des colonnes -----------------------------------------
    df_ref_ali, df_cur_ali = harmoniser_colonnes(df_ref, df_cur)
    print(f"  Features analysées......: {len(df_ref_ali.columns)}")

    # -- Génération du rapport ----------------------------------------------
    if args.format == "json":
        resume = generer_resume_json(df_ref_ali, df_cur_ali)
        print(json.dumps(resume, ensure_ascii=False, indent=2))
    else:
        generer_rapport_html(df_ref_ali, df_cur_ali)

    print("============================================================================")
    print("ANALYSE TERMINÉE")
    print("============================================================================\n")


# =============================================================================
if __name__ == "__main__":
    main()
