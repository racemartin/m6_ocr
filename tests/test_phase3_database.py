"""
tests/test_phase3_database.py
==============================
Tests — Base de données (Phase 3)

Ressource testée : Les interactions avec PostgreSQL —
chargement depuis v_features_engineering, enregistrement du champion
dans model_versions, robustesse en cas d'indisponibilité DB.

IMPORTANT : Ces tests utilisent une DB SQLite en mémoire pour ne pas
nécessiter PostgreSQL. La vue v_features_engineering est simulée
par une table temporaire de même structure.

Groupes de tests :
    1. TestLoadFromDB          → Chargement v_features_engineering
    2. TestTargetColumn        → Colonne 'target' : présence, type, distribution
    3. TestSplitFilter         → Filtre split='train', exclusion split='test'
    4. TestModelVersionsTable  → INSERT dans model_versions
    5. TestDBFallback          → Bascule CSV si DB indisponible

Usage :
    pytest tests/test_phase3_database.py -v
    pytest tests/test_phase3_database.py -v -k "test_target"
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy  as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipelines.phase3_model_training_mlflow import Phase3Pipeline


# =============================================================================
# FIXTURES : BASE DE DONNÉES EN MÉMOIRE
# =============================================================================

@pytest.fixture
def sqlite_engine():
    """Engine SQLite en mémoire — remplace PostgreSQL pour les tests."""
    return create_engine("sqlite:///:memory:", echo=False)


@pytest.fixture
def db_with_view(sqlite_engine, df_train_synth):
    """
    SQLite avec une table 'v_features_engineering' simulant la vue PostgreSQL.
    """
    import logging
    logger = logging.getLogger(__name__)

    # --- DEBUG 1: Inspeccionar el origen ---
    logger.info(f"DEBUG: Columnas en df_train_synth: {df_train_synth.columns.tolist()}")

    # 1. Normalizar df_train: Asegurar que la target se llame 'target' (minúsculas)
    df_train = df_train_synth.copy()

    df_train["split"] = "train"  # Sin esto, la query WHERE split='train' devuelve 0 filas
    
    # Normalizamos nombres para evitar el error de duplicados
    df_train = df_train.rename(columns={"TARGET": "target", "SK_ID_CURR": "sk_id_curr"})
    
    if "TARGET" in df_train.columns and "target" not in df_train.columns:
        df_train = df_train.rename(columns={"TARGET": "target"})
    
    # --- DEBUG 2: Evitar duplicados antes de seguir ---
    df_train = df_train.loc[:, ~df_train.columns.duplicated()]

    # 2. Données test (reutilizando las mismas columnas para evitar desalineación)
    rng = np.random.default_rng(999)
    # Seleccionamos solo las columnas de features (excluyendo las especiales)
    special_cols = ["target", "split", "sk_id_curr", "TARGET"]
    feature_cols = [c for c in df_train.columns if c not in special_cols]
    
    df_test = pd.DataFrame({
        col: rng.normal(0, 1, 50) for col in feature_cols
    })
    
    df_test["target"]     = None
    df_test["split"]      = "test"
    df_test["sk_id_curr"] = np.arange(10000, 10050)

    # 3. Combiner train + test
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # --- DEBUG 3: El "Sanity Check" final ---
    # Este paso elimina cualquier columna duplicada por error de tipografía (Target vs TARGET)
    df_all.columns = [c.lower() for c in df_all.columns] # Normalizar todo a minúsculas
    df_all = df_all.loc[:, ~df_all.columns.duplicated()]
    
    # Si quieres ver qué se va a escribir en la DB:
    print(f"\n[DEBUG DB] Columnas finales: {df_all.columns.tolist()}")

    # 4. Escribir en la DB
    df_all.to_sql(
        "v_features_engineering", 
        sqlite_engine,
        if_exists="replace", 
        index=False
    )

    return sqlite_engine


@pytest.fixture
def db_with_model_versions(sqlite_engine):
    """SQLite avec la table model_versions créée."""
    with sqlite_engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name  TEXT NOT NULL,
                version     TEXT NOT NULL UNIQUE,
                mlflow_run_id TEXT,
                algorithm   TEXT,
                hyperparameters TEXT,
                metrics     TEXT,
                model_path  TEXT,
                metadata_path TEXT,
                status      TEXT DEFAULT 'trained'
            )
        """))
    return sqlite_engine


@pytest.fixture
def df_train_synth(X_train_synth, y_train_synth):
    """Reconstruye un DataFrame completo (X + y) para simular la tabla de DB."""
    df = X_train_synth.copy()
    df["TARGET"] = y_train_synth.values
    # Añadimos la columna SK_ID_CURR si el test la requiere para las claves
    if "SK_ID_CURR" not in df.columns:
        df["SK_ID_CURR"] = range(len(df))
    return df

# =============================================================================
# 1. TESTS CHARGEMENT DEPUIS LA DB
# =============================================================================

class TestLoadFromDB:
    """Vérifie le chargement des données depuis v_features_engineering."""

    def test_chargement_retourne_un_dataframe(self, db_with_view):
        """La requête SQL doit retourner un DataFrame non vide."""
        df = pd.read_sql(
            "SELECT * FROM v_features_engineering WHERE split = 'train'",
            db_with_view,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_colonnes_cles_sont_presentes(self, db_with_view):
        """Les colonnes metadata doivent être présentes."""
        df = pd.read_sql(
            "SELECT * FROM v_features_engineering WHERE split = 'train'",
            db_with_view,
        )
        assert "target"     in df.columns, "Colonne 'target' manquante"
        assert "split"      in df.columns, "Colonne 'split' manquante"
        assert "sk_id_curr" in df.columns, "Colonne 'sk_id_curr' manquante"

    def test_requete_debug_limite_les_lignes(self, db_with_view):
        """Le mode debug doit charger moins de lignes via une sous-requête LIMIT."""
        debug_limit = 50
        query = f"""
            SELECT * FROM v_features_engineering
            WHERE split = 'train'
              AND sk_id_curr IN (
                  SELECT sk_id_curr FROM v_features_engineering LIMIT {debug_limit}
              )
        """
        df = pd.read_sql(query, db_with_view)
        assert len(df) <= debug_limit

    def test_pipeline_csv_load_fonctionne(
        self, tmp_processed_dir, X_train_synth, y_train_synth, monkeypatch
    ):
        """_load_from_csv() doit charger correctement les fichiers Phase 2."""
        monkeypatch.chdir(tmp_processed_dir.parent)

        pipeline = Phase3Pipeline(source="csv", verbose=False)
        pipeline.processed_dir = tmp_processed_dir
        pipeline._load_from_csv()

        assert pipeline.df_raw is not None
        assert len(pipeline.df_raw) == len(X_train_synth)
        assert "target" in pipeline.df_raw.columns

    def test_pipeline_csv_raise_si_fichier_absent(self, tmp_path, monkeypatch):
        """Une erreur explicite doit être levée si X_train.csv est absent."""
        monkeypatch.chdir(tmp_path)
        empty_dir = tmp_path / "processed"
        empty_dir.mkdir()

        pipeline = Phase3Pipeline(source="csv", verbose=False)
        pipeline.processed_dir = empty_dir

        with pytest.raises(FileNotFoundError, match="X_train.csv"):
            pipeline._load_from_csv()


# =============================================================================
# 2. TESTS COLONNE TARGET
# =============================================================================

class TestTargetColumn:
    """Vérifie la gestion de la colonne 'target' (= TARGET dans schema.py)."""

    def test_target_col_est_bien_nomme_target(self):
        """Le nom technique dans schema.py est 'target' (minuscule)."""
        pipeline = Phase3Pipeline(source="csv", verbose=False)
        assert pipeline.target_col == "target", \
            "target_col doit être 'target' — correspondance avec v_features_engineering"

    def test_target_est_convertie_en_int(self, db_with_view):
        """PostgreSQL peut retourner 0.0/1.0 (float) → doit être converti en int."""
        df = pd.read_sql(
            "SELECT * FROM v_features_engineering WHERE split = 'train'",
            db_with_view,
        )
        # Simuler la conversion du pipeline
        df["target"] = pd.to_numeric(df["target"], errors="coerce").astype(int)
        assert df["target"].dtype in [np.int64, np.int32, int], \
            f"Target doit être entier, obtenu {df['target'].dtype}"

    def test_target_a_uniquement_valeurs_0_et_1(self, db_with_view):
        """La target binaire ne doit contenir que 0 et 1."""
        df = pd.read_sql(
            "SELECT * FROM v_features_engineering WHERE split = 'train'",
            db_with_view,
        )
        valeurs = set(df["target"].dropna().astype(int).unique())
        assert valeurs.issubset({0, 1}), \
            f"Target contient des valeurs inattendues : {valeurs}"

    def test_distribution_target_reproduit_desequilibre_home_credit(self, df_train_synth):
        """Le taux de défaut doit être autour de 8% (±5%)."""
        # Convertimos todas las columnas a minúsculas temporalmente para el cálculo
        df_temp = df_train_synth.copy()
        df_temp.columns = [c.lower() for c in df_temp.columns]
        
        taux = df_temp["target"].mean()
        assert 0.03 <= taux <= 0.15, \
            f"Taux de défaut = {taux:.1%} — attendu entre 3% et 15%"

    def test_validation_leve_erreur_si_target_absente(self, tmp_path, monkeypatch):
        """_validate_data() doit lever ValueError si 'target' est absente."""
        # CSV sans colonne target
        X_sans_target = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        processed = tmp_path / "processed"
        processed.mkdir()
        X_sans_target.to_csv(processed / "X_train.csv", index=False)
        pd.Series([0, 1]).to_csv(processed / "y_train.csv", index=False)

        monkeypatch.chdir(tmp_path)
        pipeline = Phase3Pipeline(source="csv", verbose=False)
        pipeline.processed_dir = processed
        pipeline._load_from_csv()

        # df_raw ne contient pas de colonne 'target' ici (on a chargé X sans y assemblé)
        # Le test vérifie que la validation détecte l'absence
        # (On simule un df_raw sans target)
        pipeline.df_raw = X_sans_target  # Pas de colonne 'target'
        with pytest.raises(ValueError, match="target"):
            pipeline._validate_data()

    def test_lignes_sans_target_sont_exclues(self, db_with_view):
        """Les lignes test Kaggle (TARGET=NULL) ne doivent pas contaminer le train."""
        df_all = pd.read_sql(
            "SELECT * FROM v_features_engineering",
            db_with_view,
        )
        # Simuler la validation
        df_train = df_all[df_all["target"].notna()].copy()
        df_test  = df_all[df_all["target"].isna()]

        assert len(df_train) > 0, "Train ne doit pas être vide"
        assert len(df_test)  > 0, "Test existe dans la vue"
        assert df_train["target"].isna().sum() == 0, \
            "Aucun NaN ne doit subsister dans le train"


# =============================================================================
# 3. TESTS FILTRE split='train'
# =============================================================================

class TestSplitFilter:
    """
    Vérifie que le pipeline ne charge que split='train' depuis la DB.
    Le test Kaggle (sans TARGET) ne doit jamais entrer dans l'entraînement.
    """

    def test_requete_ne_charge_que_split_train(self, db_with_view):
        """La requête SQL doit filtrer WHERE split='train'."""
        df_train = pd.read_sql(
            "SELECT * FROM v_features_engineering WHERE split = 'train'",
            db_with_view,
        )
        assert (df_train["split"] == "train").all(), \
            "Des lignes split='test' ont été chargées — erreur de filtre SQL"

    def test_split_test_est_exclu_du_chargement(self, db_with_view):
        """Aucune ligne split='test' ne doit être dans le DataFrame chargé."""
        df_train = pd.read_sql(
            "SELECT * FROM v_features_engineering WHERE split = 'train'",
            db_with_view,
        )
        assert "test" not in df_train["split"].values, \
            "Des lignes Kaggle test ont été incluses dans le train"

    def test_split_strategique_80_20_est_interne(
        self, tmp_processed_dir, monkeypatch
    ):
        """
        Le split train/eval est interne (fait par Phase3Pipeline.step2_split()).
        Il n'est PAS imposé par la requête SQL (contrairement au split Phase 2).
        """
        monkeypatch.chdir(tmp_processed_dir.parent)
        pipeline = Phase3Pipeline(source="csv", eval_ratio=0.20, verbose=False)
        pipeline.processed_dir = tmp_processed_dir
        pipeline._load_from_csv()
        pipeline._validate_data()
        pipeline.step2_split()

        n_total = len(pipeline.X_train) + len(pipeline.X_eval)
        ratio_eval = len(pipeline.X_eval) / n_total

        assert abs(ratio_eval - 0.20) < 0.05, \
            f"Ratio eval={ratio_eval:.2%} — attendu 20% ±5%"

    def test_split_est_stratifie_sur_la_target(
        self, tmp_processed_dir, monkeypatch
    ):
        """Train et eval doivent avoir approximativement le même taux de défaut."""
        monkeypatch.chdir(tmp_processed_dir.parent)
        pipeline = Phase3Pipeline(source="csv", eval_ratio=0.20, verbose=False)
        pipeline.processed_dir = tmp_processed_dir
        pipeline._load_from_csv()
        pipeline._validate_data()
        pipeline.step2_split()

        taux_train = pipeline.y_train.mean()
        taux_eval  = pipeline.y_eval.mean()

        assert abs(taux_train - taux_eval) < 0.05, \
            f"Stratification incorrecte : train={taux_train:.1%}, eval={taux_eval:.1%}"


# =============================================================================
# 4. TESTS TABLE model_versions
# =============================================================================

class TestModelVersionsTable:
    """
    Vérifie l'enregistrement du champion dans PostgreSQL.
    Utilise SQLite en mémoire (même structure, sans jsonb).
    """

    def _insert_champion(self, engine, model_name="logistic_regression"):
        """Helper : insère un champion de test dans model_versions."""
        version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        info = {
            "name":       model_name,
            "version":    version,
            "run_id":     "abc123def456",
            "algo":       "LogisticRegression",
            "params":     json.dumps({"C": "0.1", "class_weight": "balanced"}),
            "metrics":    json.dumps({"f1": 0.47, "f2": 0.52, "roc_auc": 0.78}),
            "model_path": "/models/logistic_regression_model.joblib",
            "meta_path":  "/models/logistic_regression_metadata.json",
            "status":     "trained",
        }
        # SQLite ne supporte pas ::jsonb → on insère du TEXT
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO model_versions
                    (model_name, version, mlflow_run_id, algorithm,
                     hyperparameters, metrics, model_path, metadata_path, status)
                VALUES
                    (:name, :version, :run_id, :algo,
                     :params, :metrics, :model_path, :meta_path, :status)
            """), info)
        return version, info

    def test_insert_champion_dans_model_versions(self, db_with_model_versions):
        """L'INSERT du champion doit réussir sans erreur."""
        version, _ = self._insert_champion(db_with_model_versions)
        df = pd.read_sql("SELECT * FROM model_versions", db_with_model_versions)
        assert len(df) == 1
        assert df.iloc[0]["model_name"] == "logistic_regression"

    def test_version_est_unique(self, db_with_model_versions):
        """Deux insertions avec la même version doivent lever une erreur."""
        import time
        import sqlalchemy.exc

        version, _ = self._insert_champion(db_with_model_versions, "lr_v1")
        # Forcer la même version
        with pytest.raises(Exception):  # IntegrityError ou OperationalError
            with db_with_model_versions.begin() as conn:
                conn.execute(text("""
                    INSERT INTO model_versions (model_name, version, status)
                    VALUES (:name, :version, :status)
                """), {"name": "lr_v1", "version": version, "status": "trained"})

    def test_champion_a_les_metriques_metier(self, db_with_model_versions):
        """Les métriques enregistrées doivent contenir F2 et AUC-ROC."""
        self._insert_champion(db_with_model_versions)
        df = pd.read_sql("SELECT * FROM model_versions", db_with_model_versions)
        metrics_json = json.loads(df.iloc[0]["metrics"])

        assert "f2"      in metrics_json, "F2-Score manquant dans les métriques DB"
        assert "roc_auc" in metrics_json, "AUC-ROC manquant dans les métriques DB"

    def test_model_path_est_enregistre(self, db_with_model_versions):
        """Le chemin vers le fichier .joblib doit être sauvegardé."""
        self._insert_champion(db_with_model_versions)
        df = pd.read_sql("SELECT * FROM model_versions", db_with_model_versions)
        assert ".joblib" in df.iloc[0]["model_path"], \
            "model_path doit pointer vers un fichier .joblib"

    def test_status_est_trained_apres_phase3(self, db_with_model_versions):
        """Le statut après Phase 3 doit être 'trained' (pas 'production')."""
        self._insert_champion(db_with_model_versions)
        df = pd.read_sql("SELECT * FROM model_versions", db_with_model_versions)
        assert df.iloc[0]["status"] == "trained"

    def test_mlflow_run_id_est_enregistre(self, db_with_model_versions):
        """Le run_id MLflow doit être tracé pour le lien DB ↔ MLflow."""
        self._insert_champion(db_with_model_versions)
        df = pd.read_sql("SELECT * FROM model_versions", db_with_model_versions)
        assert df.iloc[0]["mlflow_run_id"] is not None
        assert len(str(df.iloc[0]["mlflow_run_id"])) > 0


# =============================================================================
# 5. TESTS BASCULE CSV (DB UNAVAILABLE)
# =============================================================================

class TestDBFallback:
    """
    Vérifie la robustesse du pipeline quand la DB est indisponible.
    Le pipeline doit basculer sur les CSV de Phase 2 (data/processed/).
    """

    def test_source_csv_ne_necessite_pas_de_db(
        self, tmp_processed_dir, monkeypatch
    ):
        """source='csv' doit fonctionner sans aucune connexion DB."""
        monkeypatch.chdir(tmp_processed_dir.parent)
        pipeline = Phase3Pipeline(source="csv", verbose=False)
        pipeline.processed_dir = tmp_processed_dir

        # Pas de get_engine() appelé
        with patch("src.pipelines.phase3_model_training_mlflow.DB_AVAILABLE", False):
            pipeline._load_from_csv()
            assert pipeline.df_raw is not None

    def test_pipeline_bascule_csv_si_db_indisponible(
        self, tmp_processed_dir, monkeypatch
    ):
        """Avec source='db' et DB indisponible, doit basculer sur CSV."""
        monkeypatch.chdir(tmp_processed_dir.parent)
        pipeline = Phase3Pipeline(source="db", verbose=False)
        pipeline.processed_dir = tmp_processed_dir

        with patch("src.pipelines.phase3_model_training_mlflow.DB_AVAILABLE", False):
            pipeline._load_from_db()
            assert pipeline.source == "csv"
            assert pipeline.df_raw is not None

    def test_step7_register_db_est_non_bloquant_si_db_absente(
        self, tmp_processed_dir, monkeypatch
    ):
        """
        step7_register_db() ne doit pas faire planter le pipeline
        si la DB est indisponible — l'enregistrement est non bloquant.
        """
        monkeypatch.chdir(tmp_processed_dir.parent)
        pipeline = Phase3Pipeline(source="csv", verbose=False)
        pipeline.best_model_name = "logistic_regression"

        with patch("src.pipelines.phase3_model_training_mlflow.DB_AVAILABLE", False):
            # Ne doit pas lever d'exception
            pipeline.step7_register_db()   # Doit logguer WARNING et continuer
