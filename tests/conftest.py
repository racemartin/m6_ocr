"""
tests/conftest.py
==================
Fixtures partagées — Phase 3 : Entraînement des modèles

Ce fichier est chargé automatiquement par pytest avant tous les tests.
Il fournit les données, modèles et pipelines synthétiques nécessaires
pour que les tests s'exécutent sans base de données ni MLflow réels.

Architecture des fixtures :
    • Données synthétiques (X_train, X_eval, y_train, y_eval)
      → reproduisent le déséquilibre Home Credit (8% de défauts)

    • Modèles entraînés légers (LogisticRegression, DummyClassifier)
      → pour tester le logging MLflow et la sauvegarde sans recalcul

    • Mock results dict
      → structure exacte retournée par ClassificationModeler.entrainer_modele()

    • Phase3Pipeline pré-configurée (source="csv")
      → pour les tests d'intégration sans DB

Usage :
    pytest tests/ -v
    pytest tests/test_phase3_models.py -v
    pytest tests/test_phase3_mlflow.py -v
    pytest tests/test_phase3_database.py -v
    pytest tests/test_phase3_pipeline.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy  as np
import pandas as pd
import pytest
from sklearn.dummy         import DummyClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONSTANTES DE TEST
# =============================================================================

N_TRAIN        = 400     # Taille du train synthétique
N_EVAL         = 100     # Taille de l'eval synthétique
N_FEATURES     = 20      # Nb features (réduit pour la rapidité)
TARGET_RATE    = 0.08    # 8% de défauts — comme Home Credit
RANDOM_STATE   = 42
IMBALANCE_RATIO = int((1 - TARGET_RATE) / TARGET_RATE)   # ≈ 11


# =============================================================================
# FIXTURES : DONNÉES SYNTHÉTIQUES
# =============================================================================

@pytest.fixture(scope="session")
def feature_names() -> list:
    """Noms des 20 features synthétiques."""
    return [
        "amt_credit", "amt_income_total", "days_birth", "days_employed",
        "ext_source_1", "ext_source_2", "ext_source_3",
        "amt_annuity", "amt_goods_price", "days_registration",
        "region_population_relative", "cnt_children", "cnt_fam_members",
        "obs_30_cnt_social_circle", "def_30_cnt_social_circle",
        "obs_60_cnt_social_circle", "def_60_cnt_social_circle",
        "flag_own_car", "flag_own_realty", "flag_document_3",
    ]


@pytest.fixture(scope="session")
def raw_synthetic_data(feature_names):
    """Générateur unique avec SCOPE SESSION et retour PANDAS."""
    TOTAL_ROWS = 500
    rng = np.random.default_rng(RANDOM_STATE)
    
    # Generar X como DataFrame
    data = {col: rng.normal(0, 1, TOTAL_ROWS) for col in feature_names}
    X = pd.DataFrame(data)
    
    # Generar y como Series
    y = pd.Series(
        rng.choice([0, 1], size=TOTAL_ROWS, p=[1 - TARGET_RATE, TARGET_RATE]),
        name="target"
    )
    
    from sklearn.model_selection import train_test_split
    # El split de sklearn mantiene el tipo DataFrame/Series si la entrada lo es
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ¡IMPORTANTE! Todas deben ser scope="session"
@pytest.fixture(scope="session")
def X_train_synth(raw_synthetic_data): return raw_synthetic_data[0]

@pytest.fixture(scope="session")
def X_eval_synth(raw_synthetic_data): return raw_synthetic_data[1]

@pytest.fixture(scope="session")
def y_train_synth(raw_synthetic_data): return raw_synthetic_data[2]

@pytest.fixture(scope="session")
def y_eval_synth(raw_synthetic_data): return raw_synthetic_data[3]


# =============================================================================
# FIXTURES : MODÈLES ENTRAÎNÉS
# =============================================================================

@pytest.fixture(scope="session")
def trained_dummy(X_train_synth, y_train_synth) -> DummyClassifier:
    """DummyClassifier fitté sur les données synthétiques."""
    model = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    model.fit(X_train_synth, y_train_synth)
    return model


@pytest.fixture(scope="session")
def trained_lr(X_train_synth, y_train_synth) -> LogisticRegression:
    """LogisticRegression fittée (légère, rapide)."""
    model = LogisticRegression(
        C=0.1, max_iter=200, class_weight="balanced",
        random_state=RANDOM_STATE, 
        # n_jobs=1,
    )
    model.fit(X_train_synth, y_train_synth)
    return model


# =============================================================================
# FIXTURES : STRUCTURE RÉSULTATS ClassificationModeler
# =============================================================================

@pytest.fixture(scope="session")
def mock_cv_scores() -> Dict:
    """
    Structure exacte retournée par ClassificationModeler._valider_croisement().

    Chaque métrique → {train_moyenne, train_ecart, cv_moyenne, cv_ecart}
    """
    metrics = ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "log_loss"]
    return {
        m: {
            "train_moyenne": 0.72,
            "train_ecart":   0.03,
            "cv_moyenne":    0.65,
            "cv_ecart":      0.04,
        }
        for m in metrics
    }


@pytest.fixture(scope="session")
def mock_scores_train() -> Dict:
    """Structure scores_train retournée par ClassificationModeler._evaluer()."""
    return {
        "accuracy":    0.82, "precision": 0.45, "recall":    0.68,
        "f1":          0.54, "f2":        0.62, "specificite": 0.87,
        "mcc":         0.41, "cohen_kappa": 0.38,
        "roc_auc":     0.81, "avg_precision": 0.38, "log_loss": 0.42,
    }


@pytest.fixture(scope="session")
def mock_scores_test() -> Dict:
    """Structure scores_test (= eval set) retournée par ClassificationModeler._evaluer()."""
    return {
        "accuracy":    0.78, "precision": 0.35, "recall":    0.72,
        "f1":          0.47, "f2":        0.58, "specificite": 0.80,
        "mcc":         0.33, "cohen_kappa": 0.30,
        "roc_auc":     0.77, "avg_precision": 0.32, "log_loss": 0.48,
    }


@pytest.fixture(scope="session")
def mock_predictions(X_eval_synth, y_eval_synth) -> Dict:
    """Prédictions mock pour l'eval set."""
    rng = np.random.default_rng(RANDOM_STATE)
    n   = len(y_eval_synth)
    return {
        "y_train_pred":  np.random.randint(0, 2, N_TRAIN),
        "y_test_pred":   np.random.randint(0, 2, n),
        "y_train_proba": np.random.uniform(0, 1, N_TRAIN),
        "y_test_proba":  np.random.uniform(0, 1, n),
    }


@pytest.fixture(scope="session")
def mock_results_lr(
    trained_lr, 
    X_train_synth, y_train_synth, 
    X_eval_synth, y_eval_synth,
    mock_cv_scores, mock_scores_train, mock_scores_test, mock_predictions,
) -> Dict:
    """
    Résultats complets avec matrices de confusion 2x2 garanties.
    """
    from sklearn.metrics import confusion_matrix

    # 1. Utilisons les vraies prédictions du modèle entraîné sur nos données de session
    # pour que la matrice de confusion soit cohérente avec les scores
    y_train_pred = trained_lr.predict(X_train_synth)
    y_test_pred  = trained_lr.predict(X_eval_synth)

    # 2. Force les labels [0, 1] pour garantir la forme (2, 2) même si
    # le modèle est très conservateur ou l'échantillon petit.
    mc_train = confusion_matrix(y_train_synth, y_train_pred, labels=[0, 1])
    mc_test  = confusion_matrix(y_eval_synth,  y_test_pred,  labels=[0, 1])

    return {
        "id_experience":    1,
        "nom_modele":       "logistic_regression",
        "modele":           trained_lr,
        "scores_cv":        mock_cv_scores,
        "scores_train":     mock_scores_train,
        "scores_test":      mock_scores_test,
        "temps_train":      0.84,
        "surapprentissage": False,
        "diagnostics":      {"ecart_f1": 0.07, "ecart_accuracy": 0.04, "overfitting": False},
        "predictions":      mock_predictions,
        "matrice_confusion": {
            "train": mc_train,
            "test":  mc_test,
        },
        "horodatage": pd.Timestamp.now(),
    }


# =============================================================================
# FIXTURES : FICHIERS TEMPORAIRES
# =============================================================================

@pytest.fixture
def tmp_processed_dir(tmp_path, X_train_synth, y_train_synth) -> Path:
    """
    Répertoire data/processed/ temporaire avec X_train.csv + y_train.csv.
    Permet de tester le mode source="csv" sans DB.
    """
    processed = tmp_path / "processed"
    processed.mkdir()
    X_train_synth.to_csv(processed / "X_train.csv", index=False)
    y_train_synth.to_csv(processed / "y_train.csv", index=False)
    return processed


@pytest.fixture
def tmp_models_dir(tmp_path) -> Path:
    """Répertoire models/ temporaire pour la sauvegarde."""
    models = tmp_path / "models"
    models.mkdir()
    return models


@pytest.fixture
def tmp_project_dir(tmp_path, X_train_synth, y_train_synth) -> Path:
    """
    Arborescence projet complète temporaire.
    Utilisé pour les tests d'intégration de Phase3Pipeline.
    """
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    X_train_synth.to_csv(tmp_path / "data" / "processed" / "X_train.csv",  index=False)
    y_train_synth.to_csv(tmp_path / "data" / "processed" / "y_train.csv",  index=False)
    return tmp_path


# =============================================================================
# FIXTURES : PHASE3PIPELINE PRÉ-CONFIGURÉE
# =============================================================================

@pytest.fixture
def pipeline_csv(tmp_project_dir, monkeypatch) -> "Phase3Pipeline":
    """
    Phase3Pipeline en mode CSV, avec MLflow mocké.
    Prête pour les tests sans connexion externe.
    """
    # Import local pour éviter l'erreur si le module n'est pas dans PYTHONPATH
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    # On monkey-patche os.getcwd() pour pointer vers tmp_project_dir
    monkeypatch.chdir(tmp_project_dir)

    # Import du pipeline
    from phase3_model_training_mlflow import Phase3Pipeline

    pipeline = Phase3Pipeline(
        source="csv",
        eval_ratio=0.20,
        random_state=RANDOM_STATE,
        debug=False,
        verbose=False,
    )
    return pipeline
