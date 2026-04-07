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
import joblib
from config import DOSSIER_ARTEFACT # Asegúrate de que apunte a model_artifact/

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

import warnings

# Filtrar el aviso específico de Pydantic antes de que cargue el resto del sistema
warnings.filterwarnings("ignore", message='.*protected namespace "model_".*')

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

    # ---- ETAPE 2: Harmonisation via Preprocesseur ---------------------------
    log.STEP(6, "2. Harmonisation via Preprocesseur")

    # -- Chargement du preprocesseur ------------------------------------------
    preprocessor = joblib.load(DOSSIER_ARTEFACT / "preprocessor.pkl")

    try:
        cols_requises = list(preprocessor.feature_names_in_)
    except AttributeError:
        log.LEVEL_4_ERROR(NOM_FICHIER, "No se pudo extraer feature_names_in_ del preprocesador.")
        sys.exit(1)

    noms_features = list(preprocessor.get_feature_names_out())

    # -- Mapeo predictions.jsonl (noms Python) → colonnes du preprocesseur ----
    MAPPING_REVERSO = {
        "age"                     : "days_birth",
        "type_residence"          : "name_housing_type",
        "revenu"                  : "amt_income_total",
        "montant_pret"            : "amt_credit",
        "duree_pret_mois"         : "duree_pret_mois",
        "type_pret"               : "name_contract_type",
        "objet_pret"              : "name_type_suite",
        "jours_retard_moyen"      : "avg_dpd_per_delinquency",
        "taux_incidents"          : "delinquency_ratio",
        "taux_utilisation_credit" : "credit_utilization_ratio",
        "nb_comptes_ouverts"      : "num_open_accounts",
    }

    # -- Colonnes catégorielles (ne pas forcer float sur ces colonnes) ---------
    # Le preprocesseur attend des chaînes pour l'OneHotEncoder.
    COLS_CATEGORIES = {"name_housing_type", "name_contract_type", "name_type_suite"}

    # -- Construction df_full_cur (predictions → espace preprocesseur) --------
    # Initialiser avec None (pas float) pour éviter de casser les catégorielles
    # df_full_cur = pd.DataFrame(index=df_cur.index, columns=cols_requises, dtype=object)

    # Por esto (o asegúrate de convertir después):
    df_full_cur = pd.DataFrame(index=df_cur.index, columns=cols_requises)
    df_full_cur = df_full_cur.astype(float, errors='ignore')

    # Convertir age en jours négatifs (convention Home Credit)
    if "age" in df_cur.columns:
        df_cur = df_cur.copy()
        df_cur["age"] = -(df_cur["age"] * 365)

    # Remplir colonne par colonne en respectant le type
    for col_json, col_preproc in MAPPING_REVERSO.items():
        if col_json in df_cur.columns and col_preproc in df_full_cur.columns:
            if col_preproc in COLS_CATEGORIES:
                # Chaîne — laisser comme object pour l'OHE
                df_full_cur[col_preproc] = df_cur[col_json].astype(str)
            else:
                # Numérique — convertir en float
                df_full_cur[col_preproc] = pd.to_numeric(df_cur[col_json], errors="coerce")

    # Colonnes numériques non mappées → NaN float (l'imputer les remplira)
    for col in cols_requises:
        if col not in COLS_CATEGORIES and col not in MAPPING_REVERSO.values():
            df_full_cur[col] = pd.to_numeric(df_full_cur[col], errors="coerce")

    # -- Transformation df_cur_ali --------------------------------------------
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            X_cur_transforme = preprocessor.transform(df_full_cur)
        df_cur_ali = pd.DataFrame(X_cur_transforme, columns=noms_features)
        log.DEBUG_PARAMETER_VALUE("df_cur_ali shape", df_cur_ali.shape)
    except Exception as e:
        log.LEVEL_4_ERROR(NOM_FICHIER, f"Erreur transformation predictions : {str(e)}")
        raise

    # -- Transformation df_ref_ali (reference_data.csv → même espace) --------
    # Le CSV est déjà dans l'espace du preprocesseur (noms colonnes identiques).
    # On filtre les colonnes communes et on transforme directement.
    try:
        cols_ref_presentes = [c for c in cols_requises if c in df_ref.columns]
        df_ref_pour_transform = pd.DataFrame(index=df_ref.index, columns=cols_requises, dtype=object)
        for col in cols_requises:
            if col in df_ref.columns:
                if col in COLS_CATEGORIES:
                    df_ref_pour_transform[col] = df_ref[col].astype(str)
                else:
                    df_ref_pour_transform[col] = pd.to_numeric(df_ref[col], errors="coerce")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            X_ref_transforme = preprocessor.transform(df_ref_pour_transform)
        df_ref_ali = pd.DataFrame(X_ref_transforme, columns=noms_features)
        log.DEBUG_PARAMETER_VALUE("df_ref_ali shape", df_ref_ali.shape)
    except Exception as e:
        log.LEVEL_4_ERROR(NOM_FICHIER, f"Erreur transformation reference : {str(e)}")
        raise

    log.DEBUG_PARAMETER_VALUE("Colonnes alignees", len(noms_features))

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