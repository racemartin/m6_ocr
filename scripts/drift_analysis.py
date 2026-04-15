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

# =============================================================================
# SELECCIÓN DE FEATURES PARA MONITOREO DE DRIFT (Top 20 SHAP)
# Se usan los nombres de las columnas originales para alineación con el dataset de referencia.
# =============================================================================

COLONNES_INTERET = [
    # 1-3: Scores externos (Críticos)
    "ext_source_2",
    "ext_source_3",
    "ext_source_1",

    # 4-8: Comportamiento de pagos y antigüedad
    "install_payment_ratio_mean",
    "days_birth",                # Representa 'age'
    "cc_amt_drawings_current_mean",
    "install_payment_delay_mean",
    "pos_months_balance_mean",

    # 9-11: Precio y Variables Categóricas (OHE)
    "amt_goods_price",
    "name_education_type",       # Se monitoriza la columna original antes del OHE
    "code_gender",               # Se monitoriza la columna original antes del OHE

    # 12-16: Créditos y Bureau
    "bureau_amt_credit_sum_total",
    "install_dpd_max",
    "amt_credit",
    "amt_annuity",
    "cc_amt_balance_mean",

    # 17-20: Estabilidad y Región
    "days_employed",
    "days_last_phone_change",
    "region_rating_client",
    "bureau_amt_credit_sum_debt_mean"
]

# Mapeo: Nombre en la API (Pydantic) -> Nombre en el Dataset Original (CSV)
MAPPING_REVERSO = {
    "ext_source_1": "ext_source_1",
    "ext_source_2": "ext_source_2",
    "ext_source_3": "ext_source_3",
    "paymnt_ratio_mean": "install_payment_ratio_mean",
    "age": "days_birth",
    "cc_drawings_mean": "cc_amt_drawings_current_mean",
    "paymnt_delay_mean": "install_payment_delay_mean",
    "pos_months_mean": "pos_months_balance_mean",
    "goods_price": "amt_goods_price",
    "education_type": "name_education_type",
    "code_gender": "code_gender",
    "bureau_credit_total": "bureau_amt_credit_sum_total",
    "max_dpd": "install_dpd_max",
    "amt_credit": "amt_credit",
    "amt_annuity": "amt_annuity",
    "cc_balance_mean": "cc_amt_balance_mean",
    "years_employed": "days_employed",
    "phone_change_days": "days_last_phone_change",
    "region_rating": "region_rating_client",
    "bureau_debt_mean": "bureau_amt_credit_sum_debt_mean"
}

# Identificación de tipos para Evidently
CATEGORICAL_FEATURES = ["name_education_type", "code_gender", "region_rating_client"]

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
    log.STEP(6, "2. Harmonisation via Preprocesseur (Top 20 SHAP)")

    # -- Mapeo Actualizado: API (Python) → Dataset Original (Preprocesador) ----
    MAPPING_REVERSO = {
        "ext_source_1": "ext_source_1",
        "ext_source_2": "ext_source_2",
        "ext_source_3": "ext_source_3",
        "paymnt_ratio_mean": "install_payment_ratio_mean",
        "age": "days_birth",
        "cc_drawings_mean": "cc_amt_drawings_current_mean",
        "paymnt_delay_mean": "install_payment_delay_mean",
        "pos_months_mean": "pos_months_balance_mean",
        "goods_price": "amt_goods_price",
        "education_type": "name_education_type",
        "code_gender": "code_gender",
        "bureau_credit_total": "bureau_amt_credit_sum_total",
        "max_dpd": "install_dpd_max",
        "amt_credit": "amt_credit",
        "amt_annuity": "amt_annuity",
        "cc_balance_mean": "cc_amt_balance_mean",
        "years_employed": "days_employed",
        "phone_change_days": "days_last_phone_change",
        "region_rating": "region_rating_client",
        "bureau_debt_mean": "bureau_amt_credit_sum_debt_mean"
    }

    # Categorías que el OneHotEncoder del preprocesador espera como String
    COLS_CATEGORIES = {"name_education_type", "code_gender"}

    # -- Construcción df_full_cur (Alinear con el espacio del preprocesador) --
    df_full_cur = pd.DataFrame(index=df_cur.index, columns=cols_requises)

    # 1. Aplicar transformaciones de unidades antes del mapeo
    df_cur_prep = df_cur.copy()
    if "age" in df_cur_prep.columns:
        df_cur_prep["age"] = -(df_cur_prep["age"] * 365)
    if "years_employed" in df_cur_prep.columns:
        df_cur_prep["years_employed"] = -(df_cur_prep["years_employed"] * 365)

    # 2. Rellenar df_full_cur usando el nuevo mapeo
    for api_col, preproc_col in MAPPING_REVERSO.items():
        if api_col in df_cur_prep.columns and preproc_col in df_full_cur.columns:
            if preproc_col in COLS_CATEGORIES:
                df_full_cur[preproc_col] = df_cur_prep[api_col].astype(str)
            else:
                df_full_cur[preproc_col] = pd.to_numeric(df_cur_prep[api_col], errors="coerce")

    # 3. Transformación final hacia el espacio de entrenamiento (df_cur_ali)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            X_cur_transforme = preprocessor.transform(df_full_cur)

        df_cur_ali = pd.DataFrame(X_cur_transforme, columns=noms_features)
        log.DEBUG_PARAMETER_VALUE("df_cur_ali shape", df_cur_ali.shape)
    except Exception as e:
        log.LEVEL_4_ERROR(NOM_FICHIER, f"Erreur transformation : {str(e)}")
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