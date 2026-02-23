"""
src/pipelines/phase2_feature_engineering.py
=============================================
Phase 2 — Feature Engineering & Preprocessing sklearn

Orchestration :
    1. Chargement depuis PostgreSQL (v_features_engineering)
       ou CSV intermédiaires (data/interim/)
    2. Validation du DataFrame contre le REGISTRY
    3. Fit du FeatureConfigurator sur le train (anti-leakage)
    4. Target encoding optionnel (colonnes haute cardinalité)
    5. Transform train + test
    6. Alignement des colonnes OHE (train vs test)
    7. Contrôles qualité (NaN, constantes, infinis)
    8. Sauvegarde des artefacts sklearn
    9. Export des datasets finaux (X_train, X_test, y_train)
   10. Rapport de synthèse

Usage :
    uv run python -m src.pipelines.phase2_feature_engineering
    uv run python -m src.pipelines.phase2_feature_engineering --source csv
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# Bibliothèques standard Python
# ----------------------------------------------------------------------------
import argparse                                   # Arguments ligne de commande
import time                                       # Mesure des durées
from   pathlib import Path                        # Chemins portables OS
from   typing  import Optional                    # Typage optionnel

# ----------------------------------------------------------------------------
# Calcul scientifique et manipulation de données
# ----------------------------------------------------------------------------
import numpy  as np                               # Calculs numériques, infinis
import pandas as pd                               # DataFrames et lecture CSV

# ----------------------------------------------------------------------------
# Modules internes — base de données, registre, pipeline preprocessing
# ----------------------------------------------------------------------------
from   src.database          import get_engine          # DB PostgreSQL
from   src.data.schema       import REGISTRY            # Registre des features
from   src.data.schema       import ColumnType
from   src.features.registry import FeatureConfigurator # Pipeline sklearn



# ############################################################################
# CHARGEMENT DES DONNÉES
# ############################################################################

DEBUG_MODE  = True
DEBUG_LIMIT = 10000  # or None


def charger_donnees(
    source:      str = "db",
    interim_dir: str = "data/interim",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge train et test depuis PostgreSQL ou des CSV intermédiaires.

    Le mode "db" lit la vue v_features_engineering contenant
    toutes les features engineerées et agrégées prêtes pour sklearn.
    Le mode "csv" relit les exports produits par la Phase 1.2.

    Args:
        source      : "db"  → v_features_engineering via PostgreSQL
                      "csv" → CSV intermédiaires data/interim/
        interim_dir : Répertoire des CSV (ignoré si source="db")

    Returns:
        Tuple (df_train, df_test) non transformés
    """
    if source == "db":
        # ── Chargement depuis PostgreSQL ─────────────────────────────────
        msg = "v_features_engineering"

        debug_limit = DEBUG_LIMIT if DEBUG_MODE else None
        
        if debug_limit:
            msg += f" [DEBUG MODE: {debug_limit} rows]"
        print(f"  Chargement depuis {msg} ...")
        
        engine = get_engine()

        if DEBUG_MODE:
            print(f" ⚡ MODO FLASH ACTIVADO: Cargando {DEBUG_LIMIT} filas...")
            # Usamos una subconsulta que obliga a Postgres a trabajar solo con 5000 IDs
            # Esto "engaña" a la vista para que no procese el resto de la tabla.
            query_train = f"""
                SELECT * FROM v_features_engineering 
                WHERE sk_id_curr IN (SELECT sk_id_curr FROM raw_application_train LIMIT {DEBUG_LIMIT})
            """
            query_test = f"""
                SELECT * FROM v_features_engineering 
                WHERE sk_id_curr IN (SELECT sk_id_curr FROM raw_application_test LIMIT {DEBUG_LIMIT // 5})
            """
        else:
            query_train = "SELECT * FROM v_features_engineering WHERE split = 'train'"
            query_test  = "SELECT * FROM v_features_engineering WHERE split = 'test'"

        print(f"{query_train} ...")
        df_train = pd.read_sql(query_train, engine)
        
        print(f"{query_test} ...")
        df_test  = pd.read_sql(query_test, engine)
    else:
        # ── Chargement depuis les CSV intermédiaires ──────────────────────
        print("  Chargement depuis les CSV intermédiaires ...")
        base     = Path(interim_dir)
        df_train = pd.read_csv(base / "master_train.csv", low_memory=False)
        df_test  = pd.read_csv(base / "master_test.csv",  low_memory=False)

    nb_cols = df_train.shape[1]                   # Nombre de colonnes commun
    print(f"  Train..: {df_train.shape[0]:>8,} lignes × {nb_cols} colonnes")
    print(f"  Test...: {df_test.shape[0]:>8,} lignes × {nb_cols} colonnes")
    return df_train, df_test


# ############################################################################
# ALIGNEMENT DES COLONNES TRAIN / TEST
# ############################################################################

def _aligner_colonnes(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligne les colonnes du test sur celles du train après OHE.

    Après OneHotEncoding, certaines catégories absentes du test
    créent un décalage de colonnes entre train et test.
    Cette fonction corrige ce problème de manière systématique.

    Règles appliquées :
      - Colonne présente dans train, absente du test → ajout à 0.0
      - Colonne présente dans test, absente du train → suppression
      - Ordre des colonnes du test aligné sur celui du train

    Args:
        X_train : DataFrame train transformé par FeatureConfigurator
        X_test  : DataFrame test  transformé par FeatureConfigurator

    Returns:
        (X_train, X_test) avec colonnes strictement identiques
    """
    cols_train = set(X_train.columns)
    cols_test  = set(X_test.columns)

    # ── Colonnes manquantes dans le test ─────────────────────────────────
    manquantes = cols_train - cols_test
    if manquantes:
        for col in manquantes:
            X_test[col] = 0.0               # Catégorie absente du test → 0
        print(f"    {len(manquantes)} colonnes ajoutées au test (à zéro)")

    # ── Colonnes en surplus dans le test ──────────────────────────────────
    en_trop = cols_test - cols_train
    if en_trop:
        X_test = X_test.drop(columns=list(en_trop))
        print(f"    {len(en_trop)} colonnes supprimées du test")

    # ── Réordonner le test exactement comme le train ──────────────────────
    X_test = X_test[X_train.columns]

    return X_train, X_test


# ############################################################################
# CONTRÔLES QUALITÉ POST-TRANSFORMATION
# ############################################################################

def _controles_qualite(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: Optional[pd.Series],
) -> None:
    """
    Vérifie la qualité des matrices finales après preprocessing.
    
    Contrôles effectués :
      1. NaN résiduels dans X_train et X_test
      2. Colonnes à variance nulle (constantes — inutiles pour le modèle)
      3. Valeurs infinies (overflow lors de transformations log)
      4. Distribution de la target (déséquilibre de classes)

    Args:
        X_train : Matrice features train transformée
        X_test  : Matrice features test  transformée
        y_train : Série target (None si non disponible)
    
    """
    import numpy as np
    import pandas as pd

    # ─────────────────────────────────────────────────────────────────────
    # 1. NaN résiduels
    # ─────────────────────────────────────────────────────────────────────
    nan_train = int(X_train.isnull().sum().sum())
    nan_test  = int(X_test.isnull().sum().sum())

    if nan_train > 0:
        cols_nan = X_train.columns[X_train.isnull().any()].tolist()
        print(f"    NaN dans X_train........: {nan_train} ({cols_nan[:3]})")
    else:
        print("    Pas de NaN dans X_train")

    if nan_test > 0:
        print(f"    NaN dans X_test.........: {nan_test}")
    else:
        print("    Pas de NaN dans X_test")

    # ─────────────────────────────────────────────────────────────────────
    # 2. Colonnes constantes
    # ─────────────────────────────────────────────────────────────────────
    # FUNDAMENTAL: Solo calculamos std() sobre columnas numéricas.
    # Esto evita el error con 'housing', 'self-employed', etc.
    X_num = X_train.select_dtypes(include=[np.number])
    
    cols_cst = [c for c in X_num.columns if X_num[c].std() == 0]
    
    if cols_cst:
        print(f"    Colonnes constantes.....: {len(cols_cst)} ({cols_cst[:3]})")
    else:
        print("    Pas de colonnes constantes détectées")

    # ─────────────────────────────────────────────────────────────────────
    # 3. Valeurs infinies
    # ─────────────────────────────────────────────────────────────────────
    # isinf solo funciona en datos numéricos
    cols_inf = X_num.columns[np.isinf(X_num).any()].tolist()
    if cols_inf:
        print(f"    Valeurs infinies........: {cols_inf[:3]}")
        X_train[cols_inf] = X_train[cols_inf].replace(
            [np.inf, -np.inf], np.nan
        )

    # ─────────────────────────────────────────────────────────────────────
    # 4. Distribution de la target
    # ─────────────────────────────────────────────────────────────────────
    if y_train is not None:
        taux_defaut = y_train.mean()
        print(f"    Taux de défaut (train)..: {taux_defaut * 100:.1f}%")
        if taux_defaut < 0.05:
            print("    Données très déséquilibrées")
            print("    → Utiliser class_weight='balanced' en Phase 3")


# ############################################################################
# PIPELINE PHASE 2 — ORCHESTRATION PRINCIPALE
# ############################################################################

def run_phase2(
    source:        str  = "db",
    interim_dir:   str  = "data/interim",
    output_dir:    str  = "data/processed",
    artifacts_dir: str  = "models/preprocessor",
    export_csv:    bool = True,
) -> dict:
    """
    Exécute la Phase 2 complète : du DataFrame brut aux matrices ML.

    Le FeatureConfigurator est entraîné UNIQUEMENT sur le train
    pour garantir l'absence de fuite d'information (anti-leakage).
    Les paramètres appris sont ensuite appliqués à l'identique sur le test.

    Args:
        source        : "db" (PostgreSQL) ou "csv" (fichiers interim)
        interim_dir   : Répertoire CSV intermédiaires (si source="csv")
        output_dir    : Répertoire de sortie des matrices finales
        artifacts_dir : Répertoire de sauvegarde des artefacts sklearn
        export_csv    : Exporter X_train, X_test, y_train en CSV

    Returns:
        Dictionnaire de synthèse {shapes, nb_features, timings}
    """
    debut_total = time.time()                     # Horodatage de départ
    resultats   = {}                              # Collecte des métriques

    print("\n" + "=" * 76)
    print("PHASE 2 — FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 76)

    # ── [1] Chargement ───────────────────────────────────────────────────
    print("\n[1] Chargement des données ...")
    df_train, df_test = charger_donnees(source, interim_dir)
    resultats["shape_train_brut"] = df_train.shape
    resultats["shape_test_brut"]  = df_test.shape

    # ─── NUEVO BLOQUE [1.5] INSERTAR AQUÍ ───────────────────────────
    print("\n[1.5] Sincronizando tipos categóricos con AttributeSpec...")
    
    # Extraemos nombres de columnas que definiste como categóricas en schema.py
    cat_cols = [attr.name_technique for attr in REGISTRY.attributes if attr.col_type == ColumnType.CATEGORICAL]
    
    for col in cat_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype(str).replace(['None', 'nan', 'NULL'], np.nan)
        if col in df_test.columns:
            df_test[col]  = df_test[col].astype(str).replace(['None', 'nan', 'NULL'], np.nan)
    # ────────────────────────────────────────────────────────────────
    
    # ── [2] Validation du DataFrame contre le REGISTRY ───────────────────
    print("\n[2] Validation contre le REGISTRY ...")
    config     = FeatureConfigurator(registry=REGISTRY, verbose=True)
    validation = REGISTRY.validate_dataframe(list(df_train.columns))

    if not validation["ok"]:
        inconnues  = validation["unknown_in_df"][:5]
        manquantes = validation["missing_from_df"][:5]
        print(f"    Colonnes inconnues......: {inconnues}")
        print(f"    Colonnes manquantes.....: {manquantes}")
    else:
        print("    DataFrame cohérent avec le REGISTRY")

    # ── [3] Fit sur le train ──────────────────────────────────────────────
    # Apprend médianes, μ/σ, bornes robustes UNIQUEMENT sur le train
    print("\n[3] Fit du FeatureConfigurator sur le train ...")
    t0 = time.time()
    config.fit(df_train)
    resultats["t_fit"] = round(time.time() - t0, 1)
    print(f"    Durée fit...............: {resultats['t_fit']}s")

    # ── [4] Target encoding (colonnes haute cardinalité) ──────────────────
    if config.cols_target_active:
        nb_cibles = len(config.cols_target_active)
        print(f"\n[4] Target encoding ({nb_cibles} colonne(s)) ...")
        # smoothing=1.0 : régularisation bayésienne pour éviter l'overfitting
        config.fit_target_encoding(df_train, smoothing=1.0)
    else:
        print("\n[4] Pas de target encoding (aucune colonne concernée)")

    # ── [5] Transform train ───────────────────────────────────────────────
    print("\n[5] Transformation du train ...")
    t0      = time.time()
    X_train = config.transform(df_train)
    y_train = config.get_target(df_train)
    resultats["t_transform_train"] = round(time.time() - t0, 1)
    print(f"    X_train shape...........: {X_train.shape}"
          f"  ({resultats['t_transform_train']}s)")

    # ── [6] Transform test (mêmes paramètres que le train) ────────────────
    print("\n[6] Transformation du test ...")
    t0     = time.time()
    X_test = config.transform(df_test)
    resultats["t_transform_test"] = round(time.time() - t0, 1)
    print(f"    X_test  shape...........: {X_test.shape}"
          f"  ({resultats['t_transform_test']}s)")

    # ── [7] Alignement colonnes OHE train vs test ─────────────────────────
    print("\n[7] Alignement colonnes train / test ...")
    X_train, X_test = _aligner_colonnes(X_train, X_test)
    nb_features     = X_train.shape[1]
    print(f"    Features finales........: {nb_features}")
    resultats["nb_features"] = nb_features

    # ── [8] Contrôles qualité ─────────────────────────────────────────────
    print("\n[8] Contrôles qualité ...")
    _controles_qualite(X_train, X_test, y_train)

    # ── [9] Sauvegarde des artefacts sklearn ──────────────────────────────
    # Persiste le preprocessor.pkl et les paramètres appris (anti-leakage)
    print("\n[9] Sauvegarde des artefacts ...")
    config.save_artifacts(artifacts_dir)

    # ── [10] Export des datasets finaux ───────────────────────────────────
    if export_csv:
        print("\n[10] Export des datasets finaux ...")
        dossier_sortie = Path(output_dir)
        dossier_sortie.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(dossier_sortie / "X_train.csv", index=False)
        X_test.to_csv( dossier_sortie / "X_test.csv",  index=False)

        if y_train is not None:
            y_train.to_csv(dossier_sortie / "y_train.csv", index=False)

        # Identifiants Kaggle pour la soumission finale
        if "sk_id_curr" in df_test.columns:
            df_test[["sk_id_curr"]].to_csv(
                dossier_sortie / "test_ids.csv", index=False
            )

        shape_y = y_train.shape if y_train is not None else "N/A"
        print(f"    X_train.csv............: {X_train.shape}")
        print(f"    X_test.csv.............: {X_test.shape}")
        print(f"    y_train.csv............: {shape_y}")

        resultats["dossier_sortie"] = str(dossier_sortie)
        resultats["shape_X_train"]  = X_train.shape
        resultats["shape_X_test"]   = X_test.shape

    # ── Résumé du REGISTRY ────────────────────────────────────────────────
    REGISTRY.summary()

    # ── Synthèse finale ───────────────────────────────────────────────────
    resultats["t_total"] = round(time.time() - debut_total, 1)

    print("\n" + "=" * 76)
    print("RAPPORT DE LA PHASE 2")
    print("=" * 76)
    print(f"  Features finales........: {resultats['nb_features']}")
    print(f"  X_train shape...........: {X_train.shape}")
    print(f"  X_test  shape...........: {X_test.shape}")
    print(f"  Durée totale............: {resultats['t_total']}s")
    print("=" * 76)
    print("\n  Suite : python -m src.pipelines.phase3_training")
    print("  X_train = pd.read_csv('data/processed/X_train.csv')\n")

    return resultats


# ############################################################################
# INTERFACE LIGNE DE COMMANDE (CLI)
# ############################################################################

def _parse_args() -> argparse.Namespace:
    """Configure et analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Phase 2 — Feature Engineering & Preprocessing sklearn"
    )
    parser.add_argument(
        "--source",
        choices = ["db", "csv"],
        default = "db",
        help    = "Source : 'db' (PostgreSQL) ou 'csv' (default: db)"
    )
    parser.add_argument(
        "--interim-dir",
        default = "data/interim",
        help    = "Répertoire CSV intermédiaires (default: data/interim)"
    )
    parser.add_argument(
        "--output-dir",
        default = "data/processed",
        help    = "Répertoire datasets finaux (default: data/processed)"
    )
    parser.add_argument(
        "--artifacts-dir",
        default = "models/preprocessor",
        help    = "Répertoire artefacts sklearn"
    )
    parser.add_argument(
        "--no-export",
        action  = "store_true",
        help    = "Ne pas exporter les CSV finaux"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_phase2(
        source        = args.source,
        interim_dir   = args.interim_dir,
        output_dir    = args.output_dir,
        artifacts_dir = args.artifacts_dir,
        export_csv    = not args.no_export,
    )