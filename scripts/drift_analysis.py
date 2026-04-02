# =============================================================================
# scripts/drift_analysis.py — Analyse de drift avec Evidently AI
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import argparse
import json
import logging
import sys
import os
import inspect
import time
from pathlib import Path

# --- Bibliothèques tierces : données -----------------------------------------
import pandas as pd

# --- Bibliothèques tierces : Evidently AI ------------------------------------
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric

# --- Outil de log personnel --------------------------------------------------
from src.tools.rafael.log_tool import LogTool

# --- Configuration -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FICHIER_PREDICTIONS,
    FICHIER_DONNEES_REF,
    RACINE_PROJET,
)

# Initialisation du LogTool
log = LogTool(origin="drift")
NOM_FICHIER = os.path.basename(__file__)

# Nettoyage des handlers standard pour ne garder que LogTool
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Chemin du rapport HTML
RAPPORT_HTML = RACINE_PROJET / "monitoring" / "drift_report.html"

# Features numériques analysées
FEATURES_NUMERIQUES = [
    "age", "revenu", "montant_pret", "duree_pret_mois",
    "jours_retard_moyen", "taux_incidents", "taux_utilisation_credit",
    "nb_comptes_ouverts", "probabilite_defaut",
]

# ##############################################################################
# Fonctions de chargement
# ##############################################################################

def analyser_arguments() -> argparse.Namespace:
    analyseur = argparse.ArgumentParser(description="Analyse de drift Evidently AI")
    analyseur.add_argument("--format", choices=["html", "json"], default="html")
    analyseur.add_argument("--nb-lignes", type=int, default=0)
    return analyseur.parse_args()

def charger_predictions(nb_lignes: int = 0) -> pd.DataFrame:
    if not FICHIER_PREDICTIONS.exists():
        log.LEVEL_3_CRITICAL(NOM_FICHIER, f"Fichier introuvable : {FICHIER_PREDICTIONS}")
        raise FileNotFoundError("Predictions.jsonl absent.")

    lignes = []
    with open(FICHIER_PREDICTIONS, encoding="utf-8") as f:
        for l in f:
            try:
                lignes.append(json.loads(l.strip()))
            except: continue

    if not lignes:
        log.LEVEL_4_ERROR(NOM_FICHIER, "Le fichier predictions.jsonl est vide.")
        raise ValueError("Fichier vide.")

    df = pd.DataFrame(lignes)
    if nb_lignes > 0:
        df = df.tail(nb_lignes).reset_index(drop=True)
    
    return df

def charger_reference() -> pd.DataFrame:
    if not FICHIER_DONNEES_REF.exists():
        log.LEVEL_3_CRITICAL(NOM_FICHIER, "Données de référence manquantes.")
        raise FileNotFoundError("Reference data absent.")
    return pd.read_csv(FICHIER_DONNEES_REF)

# ##############################################################################
# Main Orchestration
# ##############################################################################

def main() -> None:
    args = analyser_arguments()
    start_time = time.perf_counter()

    log.START_CALL_MANAGER_FUNCTION("DriftAnalysis", "main", "BEGIN ANALYSIS")

    # ---- ETAPE 1: Chargement ------------------------------------------------
    log.STEP(6, "1. Chargement des données")
    try:
        df_cur = charger_predictions(args.nb_lignes)
        df_ref = charger_reference()
        
        log.DEBUG_PARAMETER_VALUE("Nb Prédictions", len(df_cur))
        log.DEBUG_PARAMETER_VALUE("Nb Référence", len(df_ref))
    except Exception as e:
        log.LEVEL_3_CRITICAL(NOM_FICHIER, f"Echec chargement : {str(e)}")
        sys.exit(1)

    # ---- ETAPE 2: Harmonisation ---------------------------------------------
    log.STEP(6, "2. Harmonisation des colonnes")
    features_communes = [c for c in FEATURES_NUMERIQUES if c in df_ref.columns and c in df_cur.columns]
    
    df_ref_ali = df_ref[features_communes].copy()
    df_cur_ali = df_cur[features_communes].copy()
    
    log.DEBUG_PARAMETER_VALUE("Features analysées", len(features_communes))

    print(f"Columnas Referencia: {df_ref.columns.tolist()}")
    print(f"Columnas Actual: {df_cur.columns.tolist()}")

    if len(df_ref.columns) == 0 or len(df_cur.columns) == 0:
        print("ERROR: ¡Uno de los DataFrames no tiene columnas!")
    
    # ---- ETAPE 3: Analyse Evidently -----------------------------------------
    log.STEP(6, f"3. Génération du rapport ({args.format})")
    
    if args.format == "json":
        # Résumé rapide
        rapport = Report(metrics=[DatasetDriftMetric()])
        rapport.run(reference_data=df_ref_ali, current_data=df_cur_ali)
        
        resultats = rapport.as_dict()
        metriques = resultats.get("metrics", [{}])[0].get("result", {})
        
        resume = {
            "drift_detecte": metriques.get("dataset_drift", False),
            "nb_features_driftees": metriques.get("number_of_drifted_columns", 0),
            "nb_features_total": metriques.get("number_of_columns", 0)
        }
        
        log.LEVEL_7_INFO(NOM_FICHIER, "Résumé JSON généré avec succès")
        print(json.dumps(resume, indent=2))
        
    else:
        # Rapport complet HTML
        rapport = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        log.LEVEL_7_INFO(NOM_FICHIER, "Calcul Evidently en cours (ceci peut prendre quelques secondes)...")
        
        rapport.run(reference_data=df_ref_ali, current_data=df_cur_ali)
        
        RAPPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
        rapport.save_html(str(RAPPORT_HTML))
        
        log.DEBUG_PARAMETER_VALUE("Lien rapport", str(RAPPORT_HTML))

    # ---- FINISH -------------------------------------------------------------
    elapsed = time.perf_counter() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, sec = divmod(rem, 60)
    exec_time = f"{int(hours):02d}:{int(minutes):02d}:{sec:09.6f}"

    log.FINISH_CALL_MANAGER_FUNCTION("DriftAnalysis", "main", f"FINISH Exec: {exec_time}")

if __name__ == "__main__":
    main()