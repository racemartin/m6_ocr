"""
Pipeline - Phase 1 : Préparation des Données (Data Preparation)
==============================================================
Centralise le traitement du notebook 01-eda-initial.ipynb en un flux exécutable.

Usage :
    python -m src.pipelines.phase1_preparation
"""

# --- BIBLIOTHÈQUES STANDARDS ---
import sys                                    # Gestion des paramètres système
import traceback                              # Trace détaillée des erreurs
from   pathlib import Path                    # Gestion robuste des chemins

# --- BIBLIOTHÈQUES TIERS (DATA) ---
import pandas  as pd                          # Manipulation de données

# --- MODULES INTERNES DU PROJET ---
from   src.database             import get_engine             # Connexion DB
from   notebooks.AttritionCleaner import AttritionCleaner       # Logique métier


# ##############################################################################
# FONCTION PRINCIPALE : PRÉPARATION DES DONNÉES
# ##############################################################################
def executer_phase1_preparation(verbose: bool = True) -> pd.DataFrame:
    """
    Exécute la Phase 1 : Chargement, harmonisation, ingestion et nettoyage.
    
    Args:
        verbose : Si True, affiche l'état d'avancement du pipeline.
        
    Returns:
        pd.DataFrame : Données finales prêtes pour l'analyse ou le modelage.
    """
    if verbose:
        print("\n============================================================")
        print("🚀 PHASE 1 : PRÉPARATION DES DONNÉES")
        print("============================================================")

    # --------------------------------------------------------------------------
    # CONFIGURATION DES ENVIRONNEMENTS
    # --------------------------------------------------------------------------
    # On définit la racine du projet (le dossier m5_ocr)
    FEATURE_TARGET           = "attrition_binary"

    RACINE_PROJET         = Path(__file__).resolve().parents[2]
    RAW_DATA_DIR          = RACINE_PROJET / "data" / "raw"  # Répertoire source des CSV
    moteur_sql            = get_engine()         # Moteur SQLAlchemy
    nettoyeur             = AttritionCleaner(moteur_sql)

    # --------------------------------------------------------------------------
    # ÉTAPE 1 : CHARGEMENT DES FICHIERS CSV (RAW DATA)
    # --------------------------------------------------------------------------
    if verbose:
        print("\n📂 [1/6] Chargement des données brutes...")

    df_sirh_raw    = pd.read_csv(RAW_DATA_DIR / "extrait_sirh.csv")
    df_evals_raw   = pd.read_csv(RAW_DATA_DIR / "extrait_eval.csv")
    df_sondage_raw = pd.read_csv(RAW_DATA_DIR / "extrait_sondage.csv")

    if verbose:
        print(f"  Effectif SIRH..........: {len(df_sirh_raw)} lignes")
        print(f"  Effectif Évaluations...: {len(df_evals_raw)} lignes")
        print(f"  Effectif Sondage.......: {len(df_sondage_raw)} lignes")

    # --------------------------------------------------------------------------
    # ÉTAPE 2 : HARMONISATION DES IDENTIFIANTS (CLEF PRIMAIRE)
    # --------------------------------------------------------------------------
    if verbose:
        print("\n🔧 [2/6] Harmonisation des IDs (emp_id)...")

    # Normalisation SIRH : Suppression des espaces
    df_sirh_raw['emp_id']    = df_sirh_raw['id_employee'].astype(str).str.strip()

    # Normalisation EVAL : Suppression du préfixe 'E_'
    df_evals_raw['emp_id']   = (df_evals_raw['eval_number'].astype(str)
                                .str.replace('E_', '', regex=False)
                                .str.strip())

    # Normalisation SONDAGE : Suppression des zéros non significatifs
    df_sondage_raw['emp_id'] = (df_sondage_raw['code_sondage'].astype(str)
                                .str.lstrip('0').str.strip())

    if verbose:
        print("  Statut.................: IDs convertis au format numérique")

    # --------------------------------------------------------------------------
    # ÉTAPE 3 : CRÉATION DES TABLES ET INGESTION (POSTGRESQL)
    # --------------------------------------------------------------------------
    if verbose:
        print("\n🗄️  [3/6] Ingestion des données dans la base PostgreSQL...")

    # Ingestion des DataFrames normalisés
    nettoyeur.ingest_sirh(df_sirh_raw)
    nettoyeur.ingest_evals(df_evals_raw)
    nettoyeur.ingest_sondage(df_sondage_raw)

    if verbose:
        print("  Tables créées..........: raw_sirh, raw_evals, raw_sondage")

    # --------------------------------------------------------------------------
    # ÉTAPE 4 : NETTOYAGE ET TRANSFORMATIONS (SQL VIEWS)
    # --------------------------------------------------------------------------
    if verbose:
        print("\n🧹 [4/6] Exécution des procédures de nettoyage SQL...")

    nettoyeur.clean_sirh()
    nettoyeur.clean_evals()
    nettoyeur.clean_sondage()

    if verbose:
        print("  Vues générées..........: v_clean_sirh, v_clean_evals, ...")

    # --------------------------------------------------------------------------
    # ÉTAPE 5 : CONSOLIDATION DU DATASET MAÎTRE (JOIN)
    # --------------------------------------------------------------------------
    if verbose:
        print("\n🔗 [5/6] Fusion des tables (Master Join)...")

    nettoyeur.create_master_view()
    df_maitre = nettoyeur.get_master_data()

    if verbose:
        print(f"  Shape Master View......: {df_maitre.shape}")

    # --------------------------------------------------------------------------
    # ÉTAPE 6 : INGÉNIERIE DES CARACTÉRISTIQUES (FEATURE ENGINEERING)
    # --------------------------------------------------------------------------
    if verbose:
        print("\n🧪 [6/6] Création des nouvelles variables (Features)...")

    nettoyeur.create_features_view()
    df_features = nettoyeur.get_features_data()

    if verbose:
        cols_fe = [c for c in df_features.columns if c.startswith('fe')]
        print(f"  Features calculées.....: {len(cols_fe)} indicateurs")
        print(f"  Aperçu des colonnes....: {', '.join(cols_fe[:3])}...")


    # --------------------------------------------------------------------------
    # 💾 ENREGISTREMENT DANS LA TABLE 'datasets' (MÉTADONNÉES)
    # --------------------------------------------------------------------------
    if verbose:
        print("\n📝 Enregistrement des métadonnées dans la base de données...")

    from sqlalchemy import text
    import json

    try:
        # Préparation des données de suivi (dictionnaire aligné)
        dataset_info = {
            "file_path"     : "data/interim/phase1_features.csv",
            "description"   : "Dataset consolidé après nettoyage et feature engineering (Phase 1)",
            "version"       : "1.0.0",
            "row_count"     : int(df_features.shape[0]),
            "feature_count" : int(df_features.shape[1]),
            "metadata"      : json.dumps({
                "source_files"    : ["extrait_sirh.csv", "extrait_eval.csv", "extrait_sondage.csv"],
                "target_variable" : FEATURE_TARGET,
                "null_count"      : int(df_features.isnull().sum().sum())
            })
        }

        # Requête d'insertion SQL
        query = text("""
            INSERT INTO datasets (file_path, description, version, row_count, feature_count, metadata)
            VALUES (:file_path, :description, :version, :row_count, :feature_count, :metadata)
        """)

        # Exécution via le moteur SQL (moteur_sql doit être défini dans ton scope)
        with moteur_sql.begin() as conn:
            conn.execute(query, dataset_info)

        if verbose:
            print("  Statut.................: Métadonnées enregistrées avec succès")
            print(f"  Version enregistrée....: {dataset_info['version']}")

    except Exception as e:
        if verbose:
            print("  ⚠️ Attention...........: Impossible d'enregistrer les métadonnées")
            print(f"  Détail de l'erreur.....: {e}")

    # --------------------------------------------------------------------------
    # RAPPORT FINAL DE LA PHASE 1
    # --------------------------------------------------------------------------
    if verbose:
        print("\n============================================================")
        print("✅ PHASE 1 TERMINÉE AVEC SUCCÈS")
        print("============================================================")
        print(f"  Effectif final.........: {df_features.shape[0]} employés")
        print(f"  Total variables........: {df_features.shape[1]}")
        print(f"  Target variable........: {FEATURE_TARGET}")
        print("  Destination SQL........: v_features_engineering")

    return df_features


# ##############################################################################
# POINT D'ENTRÉE DU SCRIPT
# ##############################################################################
def main():
    """Point d'entrée pour l'exécution en ligne de commande."""
    try:
        # Exécution du pipeline
        donnees_finales = executer_phase1_preparation(verbose=True)

        # Sauvegarde intermédiaire pour inspection rapide
        chemin_sortie   = Path("data/interim/phase1_features.csv")
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

        donnees_finales.to_csv(chemin_sortie, index=False)
        print(f"  Archive CSV............: {chemin_sortie}")

        return 0  # Code de sortie : Succès

    except Exception as erreur:
        print(f"\n❌ ERREUR CRITIQUE : {erreur}", file=sys.stderr)
        traceback.print_exc()
        return 1  # Code de sortie : Échec


if __name__ == "__main__":
    sys.exit(main())


# uv run python -m src.pipelines.executer_phase1_preparation
