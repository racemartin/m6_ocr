"""
src/pipelines/phase1_1_inject_raw.py
======================================
Phase 1.1 — Ingestion CSV → tables raw_*

Étapes :
    0. Connexion à PostgreSQL (via src.database.get_engine)
    1. Ingestion des 8 CSV → tables raw_* dans PostgreSQL

À exécuter UNE SEULE FOIS (ou si les CSV sources changent).
Dure plusieurs minutes sur le dataset complet Kaggle.

Usage :
    python -m src.pipelines.phase1_1_inject_raw
    python -m src.pipelines.phase1_1_inject_raw --data-dir data/raw
"""

from __future__ import annotations

import argparse
import time

from src.database                    import get_engine
from src.data.credit_scoring_cleaner import CreditScoringCleaner
from src.data.schema                 import REGISTRY


# =============================================================================
# PIPELINE PHASE 1.1
# =============================================================================

def run_phase1_1(data_dir: str = "data/raw") -> dict:
    """
    Étape 0 + 1 : connexion PostgreSQL + ingestion CSV → raw_*

    Args:
        data_dir : répertoire contenant les 8 CSV bruts Kaggle

    Returns:
        dict avec timings et liste des tables créées
    """
    results     = {}
    start_total = time.time()

    print("\n" + "=" * 70)
    print(f"{'PHASE 1.1 — INGESTION CSV → TABLES RAW':^70}")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # [0] CONNEXION À LA BASE DE DONNÉES PostgreSQL
    # ------------------------------------------------------------------------
    print("\n[0] Connexion à PostgreSQL ...")
    engine  = get_engine()                        # Récupération moteur Docker
    cleaner = CreditScoringCleaner(               # Instance du préparateur
        engine   = engine, 
        registry = REGISTRY, 
        verbose  = True
    )
    print(f"  ✅ Connecté : {engine.url.host}:{engine.url.port}/{engine.url.database}")

    # ------------------------------------------------------------------------
    # [1] INGESTION DES CSV BRUTS (data/raw -> SQL) Ingestion CSV → raw_* 
    # ------------------------------------------------------------------------
    print("\n[1] Ingestion des CSV → tables raw_* ...")
    print("    ⏳ Peut durer plusieurs minutes sur le dataset complet Kaggle.")
    t0 = time.time()

    cleaner.ingest_all(data_dir=data_dir)

    results["t_ingestion"] = round(time.time() - t0, 1)
    print(f"    ⏱  Temps ingestion : {results['t_ingestion']}s")

    # ── Résumé ─────────────────────────────────────────────────────────────
    db_summary           = cleaner.get_db_summary()
    results["db_tables"] = db_summary["tables"]
    results["t_total"]   = round(time.time() - start_total, 1)

    print("\n" + "=" * 70)
    print(f"{'✅  PHASE 1.1 TERMINÉE':^70}")
    print("=" * 70)
    print(f"  Tables raw créées : {len(results['db_tables'])}")
    for t in sorted(results["db_tables"]):
        print(f"    • {t}")
    print(f"  Durée totale      : {results['t_total']}s")
    print("=" * 70)
    print("\n  ➡️  Suite : python -m src.pipelines.phase1_2_views_fe_enum\n")

    return results


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1.1 — Ingestion CSV → tables raw_* dans PostgreSQL"
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Répertoire des CSV bruts (default: data/raw)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_phase1_1(data_dir=args.data_dir)
