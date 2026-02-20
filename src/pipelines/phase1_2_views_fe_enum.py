"""
src/pipelines/phase1_2_views_fe_enum.py
=========================================
Phase 1.2 — Vues SQL + Feature Engineering + Enums + Export

Prérequis : phase1_1_inject_raw.py doit avoir été exécuté
            (tables raw_* doivent exister dans PostgreSQL)

Étapes :
    2. Création des vues clean, agrégées, master, features_engineering
    3. Vérification de la vue maîtresse (shape, nulls, target)
    4. Export CSV intermédiaires (optionnel)
    5. Génération des classes Enum depuis le REGISTRY
    6. Synchronisation du REGISTRY → PostgreSQL + YAML
    7. Rapport de synthèse

Usage :
    python -m src.pipelines.phase1_2_views_fe_enum
    python -m src.pipelines.phase1_2_views_fe_enum --no-export --no-enums
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.database                    import get_engine
from src.data.credit_scoring_cleaner import CreditScoringCleaner
from src.data.schema                 import REGISTRY
from src.features.generate_enums    import generate_enum_classes


# =============================================================================
# PIPELINE PHASE 1.2
# =============================================================================

def run_phase1_2(
    export_csv: bool = True,
    output_dir: str  = "data/interim",
    gen_enums:  bool = True,
) -> dict:
    """
    Étapes 2 à 7 : vues SQL → vérification → export → enums → registre.

    Args:
        export_csv : exporter master_train.csv / master_test.csv
        output_dir : répertoire d'export CSV
        gen_enums  : générer src/features/enums/*.py

    Returns:
        dict avec shapes, timings et résumé DB
    """
    results     = {}
    start_total = time.time()

    print("\n" + "=" * 70)
    print(f"{'PHASE 1.2 — VUES SQL + FE + ENUMS':^70}")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # [0] CONNEXION À LA BASE DE DONNÉES
    # ------------------------------------------------------------------------
    print("\n[0] Connexion à PostgreSQL ...")
    engine  = get_engine()                        # Récupération moteur Docker
    cleaner = CreditScoringCleaner(               # Instance du préparateur
        engine   = engine, 
        registry = REGISTRY, 
        verbose  = True
    )
    print(f"  ✅ Connecté : {engine.url.host}:{engine.url.port}/{engine.url.database}")

    # Vérification que les tables raw_* existent
    db_init    = cleaner.get_db_summary()
    raw_tables = [t for t in db_init["tables"] if t.startswith("raw_")]
    if not raw_tables:
        raise RuntimeError(
            "❌ Aucune table raw_* trouvée dans PostgreSQL.\n"
            "   Exécuter d'abord :\n"
            "   python -m src.pipelines.phase1_1_inject_raw"
        )
    print(f"  ✅ {len(raw_tables)} tables raw_* trouvées")

    # ------------------------------------------------------------------------
    # [2] CRÉATION DES VUES SQL (Logique métier)
    # ------------------------------------------------------------------------
    print("\n[2] Création des vues SQL ...")
    print("    v_clean_* → v_agg_* → v_master → v_features_engineering")
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

    results["train_shape"] = df_train.shape
    results["test_shape"]  = df_test.shape
    results["t_load"]      = round(time.time() - t0, 1)

    print(f"    Train : {df_train.shape[0]:>7,} lignes × {df_train.shape[1]} colonnes")
    print(f"    Test  : {df_test.shape[0]:>7,} lignes × {df_test.shape[1]} colonnes")
    print(f"    ⏱  Temps chargement : {results['t_load']}s")

    # Analyse des valeurs manquantes
    missing_pct = (df_train.isnull().sum() / len(df_train) * 100).sort_values(ascending=False)
    high_miss   = missing_pct[missing_pct > 50]
    if len(high_miss) > 0:
        print(f"\n    ⚠️  {len(high_miss)} colonnes avec > 50% de nulls :")
        for col, pct in high_miss.head(10).items():
            print(f"       {col:<40} : {pct:.1f}%")
    else:
        print("    ✅ Aucune colonne avec > 50% de nulls")

    # Distribution de la target
    if "target" in df_train.columns:
        dist = df_train["target"].value_counts(normalize=True)
        print(f"\n    📊 Distribution TARGET (train) :")
        print(f"       Classe 0 (remboursé) : {dist.get(0, 0)*100:.1f}%")
        print(f"       Classe 1 (défaut)    : {dist.get(1, 0)*100:.1f}%")
        results["default_rate"] = float(dist.get(1, 0))

    # ------------------------------------------------------------------------
    # [4] EXPORT DES CSV INTERMÉDIAIRES
    # ------------------------------------------------------------------------
    if export_csv:
        print("\n[4] Export CSV intermédiaires ...")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        path_train = out_path / "master_train.csv"
        path_test  = out_path / "master_test.csv"

        df_train.to_csv(path_train, index=False)
        df_test.to_csv(path_test,   index=False)

        results["csv_train"] = str(path_train)
        results["csv_test"]  = str(path_test)
        print(f"    ✅ {path_train}  ({df_train.shape[0]:,} lignes)")
        print(f"    ✅ {path_test}   ({df_test.shape[0]:,} lignes)")
    else:
        print("\n[4] Export CSV ignoré (--no-export)")

    # ------------------------------------------------------------------------
    # [5] GÉNÉRATION DES CLASSES ENUM
    # ------------------------------------------------------------------------
    if gen_enums:
        print("\n[5] Génération des classes Enum ...")
        enum_files = generate_enum_classes(
            registry   = REGISTRY,
            output_dir = "src/features/enums",
        )
        results["enum_files"] = len(enum_files)
    else:
        print("\n[5] Génération des Enums ignorée (--no-enums)")

    # ------------------------------------------------------------------------
    # [6] SYNCHRONISATION DU REGISTRE
    # ------------------------------------------------------------------------
    print("\n[6] Synchronisation du REGISTRY ...")

    # Export YAML local
    config_path = Path("config")
    config_path.mkdir(exist_ok=True)
    REGISTRY.to_yaml(str(config_path / "feature_registry.yaml"))
    print(f"  ✅ YAML exporté → config/feature_registry.yaml")

    # Sync vers PostgreSQL (engine SQLAlchemy → pandas to_sql fonctionne directement)
    REGISTRY.sync_to_db(engine)

    # ------------------------------------------------------------------------
    # [7] RÉSUMÉ FINAL
    # ------------------------------------------------------------------------
    print("\n[7] Résumé de la base de données ...")
    db_summary           = cleaner.get_db_summary()
    results["db_tables"] = db_summary["tables"]
    results["db_views"]  = db_summary["views"]
    results["t_total"]   = round(time.time() - start_total, 1)

    raw_count = len([t for t in results["db_tables"] if t.startswith("raw_")])

    print("\n" + "=" * 70)
    print(f"{'✅  PHASE 1.2 TERMINÉE':^70}")
    print("=" * 70)
    print(f"  Tables raw         : {raw_count}")
    print(f"  Vues créées        : {len(results['db_views'])}")
    for v in sorted(results["db_views"]):
        print(f"    • {v}")
    print(f"  Train shape        : {results.get('train_shape', 'N/A')}")
    print(f"  Test  shape        : {results.get('test_shape',  'N/A')}")
    if "default_rate" in results:
        print(f"  Taux de défaut     : {results['default_rate']*100:.1f}%")
    if "enum_files" in results:
        print(f"  Enums générés      : {results['enum_files']} fichiers")
    print(f"  Durée totale       : {results['t_total']}s")
    print("=" * 70)
    print("\n  ➡️  Suite : python -m src.pipelines.phase2_feature_engineering\n")

    return results


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1.2 — Vues SQL + Feature Engineering + Enums"
    )
    parser.add_argument(
        "--output-dir", default="data/interim",
        help="Répertoire d'export CSV (default: data/interim)"
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Ne pas exporter de CSV intermédiaires"
    )
    parser.add_argument(
        "--no-enums", action="store_true",
        help="Ne pas générer les classes Enum"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_phase1_2(
        export_csv = not args.no_export,
        output_dir = args.output_dir,
        gen_enums  = not args.no_enums,
    )
