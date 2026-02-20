from __future__ import annotations                # Compatibilité types futurs

"""
src/pipelines/phase1_preparation.py
=====================================
Phase 1 — Ingestion & Préparation SQL

Orchestration complète :
    0. Connexion à la base de données (SQLite ou PostgreSQL)
    1. Ingestion des 7 CSV → tables raw_*
    2. Création des vues clean (v_clean_*)
    3. Création des vues agrégées (v_agg_*)
    4. Création de la vue maîtresse (v_master)
    5. Création de la vue feature engineering (v_features_engineering)
    6. Export d'un CSV intermédiaire (optionnel)
    7. Rapport de synthèse

Usage :
    python -m src.pipelines.phase1_preparation
    python -m src.pipelines.phase1_preparation --data-dir data/raw --db data/credit_scoring.db
"""

"""
src/pipelines/phase1_preparation.py
=====================================
PHASE 1 — Ingestion et Préparation SQL des données Home Credit.
"""


# ----------------------------------------------------------------------------
# BIBLIOTHÈQUES STANDARDS ET DATA
# ----------------------------------------------------------------------------
import argparse                                   # Analyse des arguments CLI
import time                                       # Mesure des temps d'exécution
from   pathlib import Path                        # Gestion robuste des chemins
import pandas  as pd                              # Manipulation des DataFrames
import numpy   as np                              # Calculs numériques

# ----------------------------------------------------------------------------
# MODULES INTERNES DU PROJET
# ----------------------------------------------------------------------------
from   src.database                  import get_engine               # DB
from   src.data.credit_scoring_cleaner import CreditScoringCleaner     # Nettoyage
from   src.data.schema               import REGISTRY                 # Schéma
from   src.features.generate_enums   import generate_enum_classes    # Enums


# ############################################################################
# PIPELINE PHASE 1 : ORCHESTRATION PRINCIPALE
# ############################################################################

def run_phase1(
    data_dir:   str  = "data/raw",                # Dossier des CSV sources
    db_url:     str  = "sqlite:///data/db.db",    # URL de secours (SQLite)
    export_csv: bool = True,                      # Flag d'export interim
    output_dir: str  = "data/interim",            # Dossier de sortie CSV
    gen_enums:  bool = True,                      # Flag génération Enums
    limit_rows: int  = None,                      # Limite pour tests rapides
) -> dict:
    """
    Exécute le pipeline complet de préparation des données.
    """
    start_total = time.time()                     # Top chrono global
    results     = {}                              # Dictionnaire de synthèse

    print("\n============================================================================")
    print("PHASE 1 — PRÉPARATION DES DONNÉES")
    print("============================================================================")

    # ------------------------------------------------------------------------
    # [0] CONNEXION À LA BASE DE DONNÉES
    # ------------------------------------------------------------------------
    print("\n[0] Connexion à la base de données ...")
    
    engine  = get_engine()                        # Récupération moteur Docker
    cleaner = CreditScoringCleaner(               # Instance du préparateur
        engine   = engine, 
        registry = REGISTRY, 
        verbose  = True
    )

    # ------------------------------------------------------------------------
    # [1] INGESTION DES CSV BRUTS (data/raw -> SQL)
    # ------------------------------------------------------------------------
    # print("\n[1] Ingestion des CSV → tables raw_* ...")
    # t0 = time.time()
    # cleaner.ingest_all(data_dir = data_dir)
    # results["t_ingestion"] = round(time.time() - t0, 1)
    # print(f"    ⏱  Temps d'ingestion..: {results['t_ingestion']}s")

    # ------------------------------------------------------------------------
    # [2] CRÉATION DES VUES SQL (Logique métier)
    # ------------------------------------------------------------------------
    print("\n[2] Création des vues SQL ...")
    t0 = time.time()
    cleaner.create_all_views()
    results["t_views"] = round(time.time() - t0, 1)
    print(f"    ⏱  Temps SQL..........: {results['t_views']}s")


    
    # ------------------------------------------------------------------------
    # [3] CHARGEMENT ET VÉRIFICATION DE LA VUE MAÎTRESSE
    # ------------------------------------------------------------------------
    print("\n[3] Vérification de la vue maîtresse ...")
    t0 = time.time()
    df_train, df_test = cleaner.load_train_test(use_features_view=True)

    results["train_shape"] = df_train.shape       # Dimensions du train
    results["test_shape"]  = df_test.shape        # Dimensions du test
    results["t_load"]      = round(time.time() - t0, 1)

    print(f"    Train shape...........: {df_train.shape}")
    print(f"    Test shape............: {df_test.shape}")
    print(f"    ⏱  Temps chargement...: {results['t_load']}s")

    # Analyse des valeurs manquantes
    miss_train = (df_train.isnull().sum() / 
                  len(df_train) * 100)            # Calcul du % de nulls
    high_miss  = miss_train[miss_train > 50]      # Seuil critique 50%

    if len(high_miss) > 0:
        print(f"\n    ⚠️  {len(high_miss)} colonnes avec > 50% de nans :")
        for col, pct in high_miss.head(5).items():
            print(f"      {col:<20} : {pct:.1f}%")

    # Distribution de la cible (Imbalance check)
    if "target" in df_train.columns:
        dist = df_train["target"].value_counts(normalize=True)
        print(f"\n    📊 Distribution TARGET :")
        print(f"      Classe 0 (Sain).....: {dist.get(0, 0)*100:.1f}%")
        print(f"      Classe 1 (Défaut)...: {dist.get(1, 0)*100:.1f}%")
        results["default_rate"] = float(dist.get(1, 0))


    # ------------------------------------------------------------------------
    # [4] EXPORT DES CSV INTERMÉDIAIRES
    # ------------------------------------------------------------------------
    if export_csv:
        print("\n[4] Export CSV intermédiaire ...")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        path_train = out_path / "master_train.csv"
        path_test  = out_path / "master_test.csv"

        df_train.to_csv(path_train, index=False)
        df_test.to_csv(path_test, index=False)

        print(f"    ✅ Export Train.......: {path_train}")
        print(f"    ✅ Export Test........: {path_test}")

    # ------------------------------------------------------------------------
    # [5] GÉNÉRATION DES CLASSES ENUM
    # ------------------------------------------------------------------------
    if gen_enums:
        print("\n[5] Génération des classes Enum ...")
        enums = generate_enum_classes(
            registry   = REGISTRY, 
            output_dir = "src/features/enums"
        )
        results["enum_files"] = len(enums)

    # ------------------------------------------------------------------------
    # [6] SYNCHRONISATION DU REGISTRE
    # ------------------------------------------------------------------------
    print("\n[6] Export du registre d'attributs ...")
    conf_path = Path("config")
    conf_path.mkdir(exist_ok=True)
    
    REGISTRY.to_yaml(str(conf_path / "feature_registry.yaml"))
    REGISTRY.sync_to_db(engine)                   # Synchro métadonnées DB

    # ------------------------------------------------------------------------
    # [7] RÉSUMÉ FINAL
    # ------------------------------------------------------------------------
    print("\n============================================================================")
    print("RAPPORT DE LA PHASE 1")
    print("============================================================================")
    
    db_sum = cleaner.get_db_summary()
    results["t_total"] = round(time.time() - start_total, 1)

    print(f"  Durée totale.........: {results['t_total']}s")
    print(f"  Tables brutes........: {len(db_sum['tables'])}")
    print(f"  Vues créées..........: {len(db_sum['views'])}")
    print(f"  Lignes Train.........: {results.get('train_shape', (0,0))[0]}")
    print(f"  Colonnes finales.....: {results.get('train_shape', (0,0))[1]}")
    print("============================================================================\n")

    return results


# ############################################################################
# INTERFACE LIGNE DE COMMANDE (CLI)
# ############################################################################

def _parse_args() -> argparse.Namespace:
    """
    Configure les arguments de la ligne de commande.
    """
    parser = argparse.ArgumentParser(description="Phase 1 — Ingestion SQL")
    
    parser.add_argument(
        "--data-dir",   default="data/raw",       # Source des données
        help="Répertoire des CSV bruts"
    )
    parser.add_argument(
        "--db",         default=None,             # Utilise get_engine si None
        help="URL de connexion DB (optionnel)"
    )
    parser.add_argument(
        "--output-dir", default="data/interim",   # Cible des exports
        help="Répertoire d'export CSV"
    )
    parser.add_argument(
        "--no-export",  action="store_true",      # Désactive l'export CSV
        help="Ne pas exporter de CSV"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    
    run_phase1(
        data_dir   = args.data_dir,
        db_url     = args.db,
        export_csv = not args.no_export,
        output_dir = args.output_dir
    )