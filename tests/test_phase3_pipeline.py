"""
tests/test_phase3_pipeline.py
==============================
Tests — Pipeline d'intégration (Phase 3)

Ressource testée : Le pipeline Phase3Pipeline dans son ensemble —
chargement → split → sauvegarde → comparaison.

Groupes de tests :
    1. TestDataSplit         → step2_split() : stratification, anti-leakage
    2. TestModelSave         → step6_save() : joblib + metadata JSON
    3. TestModelCompare      → step5_compare() : tri par F2, champion
    4. TestPipelineConfig    → Constructeur, paramètres CLI

Usage :
    pytest tests/test_phase3_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy  as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipelines.phase3_model_training_mlflow import Phase3Pipeline, MLflowConfig


# =============================================================================
# 1. TESTS SPLIT TRAIN / EVAL
# =============================================================================

class TestDataSplit:

    def _make_pipeline_with_data(self, tmp_processed_dir, monkeypatch):
        monkeypatch.chdir(tmp_processed_dir.parent)
        p = Phase3Pipeline(source="csv", eval_ratio=0.20,
                           random_state=42, verbose=False)
        p.processed_dir = tmp_processed_dir
        p._load_from_csv()
        p._validate_data()
        return p

    def test_split_produit_4_objets(self, tmp_processed_dir, monkeypatch):
        p = self._make_pipeline_with_data(tmp_processed_dir, monkeypatch)
        p.step2_split()
        assert p.X_train is not None
        assert p.X_eval  is not None
        assert p.y_train is not None
        assert p.y_eval  is not None

    def test_ratio_eval_respecte(self, tmp_processed_dir, monkeypatch):
        p = self._make_pipeline_with_data(tmp_processed_dir, monkeypatch)
        p.step2_split()
        n_total    = len(p.X_train) + len(p.X_eval)
        ratio_eval = len(p.X_eval) / n_total
        assert abs(ratio_eval - 0.20) < 0.05

    def test_pas_de_nan_dans_x_train(self, tmp_processed_dir, monkeypatch):
        p = self._make_pipeline_with_data(tmp_processed_dir, monkeypatch)
        p.step2_split()
        assert p.X_train.isnull().sum().sum() == 0

    def test_pas_de_nan_dans_x_eval(self, tmp_processed_dir, monkeypatch):
        p = self._make_pipeline_with_data(tmp_processed_dir, monkeypatch)
        p.step2_split()
        assert p.X_eval.isnull().sum().sum() == 0

    def test_feature_names_correspond_aux_colonnes_x_train(
        self, tmp_processed_dir, monkeypatch
    ):
        p = self._make_pipeline_with_data(tmp_processed_dir, monkeypatch)
        p.step2_split()
        assert p.feature_names == list(p.X_train.columns)

    def test_colonnes_meta_exclues_des_features(
        self, tmp_processed_dir, monkeypatch
    ):
        p = self._make_pipeline_with_data(tmp_processed_dir, monkeypatch)
        p.step2_split()
        for col in ["target", "split", "sk_id_curr"]:
            assert col not in p.X_train.columns, \
                f"Colonne metadata '{col}' ne doit pas être dans X_train"

    def test_x_train_et_x_eval_ont_memes_colonnes(
        self, tmp_processed_dir, monkeypatch
    ):
        p = self._make_pipeline_with_data(tmp_processed_dir, monkeypatch)
        p.step2_split()
        assert list(p.X_train.columns) == list(p.X_eval.columns)

    def test_indices_train_et_eval_sont_disjoints(
        self, tmp_processed_dir, monkeypatch
    ):
        p = self._make_pipeline_with_data(tmp_processed_dir, monkeypatch)
        p.step2_split()
        idx_train = set(p.X_train.index.tolist())
        idx_eval  = set(p.X_eval.index.tolist())
        assert len(idx_train & idx_eval) == 0, "Leakage train/eval détecté"

    def test_distribution_target_similaire_train_eval(
        self, tmp_processed_dir, monkeypatch
    ):
        p = self._make_pipeline_with_data(tmp_processed_dir, monkeypatch)
        p.step2_split()
        assert abs(p.y_train.mean() - p.y_eval.mean()) < 0.05


# =============================================================================
# 2. TESTS SAUVEGARDE (step6_save)
# =============================================================================

class TestModelSave:

    def _make_pipeline_with_results(self, tmp_models_dir, trained_lr,
                                    mock_results_lr, mock_cv_scores):
        """Pipeline prêt pour step6_save() avec un modèle fictif."""
        p = Phase3Pipeline(source="csv", verbose=False)
        p.models_dir          = tmp_models_dir
        p.feature_names       = [f"f{i}" for i in range(20)]
        p.y_train             = pd.Series([0]*368 + [1]*32, name="target")
        p.y_eval              = pd.Series([0]*92 + [1]*8,   name="target")
        p.best_model_name     = "logistic_regression"
        p.mlflow_runs         = {"logistic_regression": "abc123"}
        p.results             = {"logistic_regression": mock_results_lr}
        return p

    def test_fichier_joblib_cree(self, tmp_models_dir, trained_lr,
                                  mock_results_lr, mock_cv_scores):
        p = self._make_pipeline_with_results(
            tmp_models_dir, trained_lr, mock_results_lr, mock_cv_scores
        )
        p.step6_save(save_all=False)
        joblib_path = tmp_models_dir / "logistic_regression_model.joblib"
        assert joblib_path.exists(), f"{joblib_path} non créé"

    def test_fichier_metadata_json_cree(self, tmp_models_dir, trained_lr,
                                         mock_results_lr, mock_cv_scores):
        p = self._make_pipeline_with_results(
            tmp_models_dir, trained_lr, mock_results_lr, mock_cv_scores
        )
        p.step6_save(save_all=False)
        json_path = tmp_models_dir / "logistic_regression_metadata.json"
        assert json_path.exists()

    def test_modele_joblib_est_rechargeable(self, tmp_models_dir, trained_lr,
                                             mock_results_lr, mock_cv_scores):
        p = self._make_pipeline_with_results(
            tmp_models_dir, trained_lr, mock_results_lr, mock_cv_scores
        )
        p.step6_save(save_all=False)
        path = tmp_models_dir / "logistic_regression_model.joblib"
        model_rechargé = joblib.load(path)
        assert hasattr(model_rechargé, "predict")
        assert hasattr(model_rechargé, "predict_proba")

    def test_metadata_json_contient_les_cles_requises(
        self, tmp_models_dir, trained_lr, mock_results_lr, mock_cv_scores
    ):
        p = self._make_pipeline_with_results(
            tmp_models_dir, trained_lr, mock_results_lr, mock_cv_scores
        )
        p.step6_save(save_all=False)
        path = tmp_models_dir / "logistic_regression_metadata.json"
        with open(path) as f:
            meta = json.load(f)

        required = [
            "model_name", "model_class", "is_best",
            "train_metrics", "eval_metrics",
            "cv_f1_mean", "cv_f1_std",
            "feature_names", "n_features",
            "train_samples", "eval_samples", "target_rate",
            "mlflow_run_id", "model_path", "saved_at",
        ]
        for key in required:
            assert key in meta, f"Clé '{key}' manquante dans metadata JSON"

    def test_is_best_est_true_pour_champion(self, tmp_models_dir, trained_lr,
                                             mock_results_lr, mock_cv_scores):
        p = self._make_pipeline_with_results(
            tmp_models_dir, trained_lr, mock_results_lr, mock_cv_scores
        )
        p.step6_save(save_all=False)
        with open(tmp_models_dir / "logistic_regression_metadata.json") as f:
            meta = json.load(f)
        assert meta["is_best"] is True

    def test_feature_names_dans_metadata(self, tmp_models_dir, trained_lr,
                                          mock_results_lr, mock_cv_scores):
        p = self._make_pipeline_with_results(
            tmp_models_dir, trained_lr, mock_results_lr, mock_cv_scores
        )
        p.step6_save(save_all=False)
        with open(tmp_models_dir / "logistic_regression_metadata.json") as f:
            meta = json.load(f)
        assert meta["n_features"] == 20
        assert len(meta["feature_names"]) == 20

    def test_saved_model_paths_est_mis_a_jour(self, tmp_models_dir, trained_lr,
                                               mock_results_lr, mock_cv_scores):
        p = self._make_pipeline_with_results(
            tmp_models_dir, trained_lr, mock_results_lr, mock_cv_scores
        )
        p.step6_save(save_all=False)
        assert "logistic_regression" in p.saved_model_paths
        assert p.saved_model_paths["logistic_regression"].endswith(".joblib")

    def test_save_all_cree_tous_les_fichiers(self, tmp_models_dir,
                                              trained_lr, trained_dummy,
                                              mock_results_lr, mock_cv_scores):
        """save_all=True doit sauvegarder tous les modèles, pas seulement le champion."""
        from sklearn.dummy import DummyClassifier

        mock_results_dummy = {
            **mock_results_lr,
            "modele":     trained_dummy,
            "nom_modele": "dummy_baseline",
        }
        p = Phase3Pipeline(source="csv", verbose=False)
        p.models_dir      = tmp_models_dir
        p.feature_names   = [f"f{i}" for i in range(20)]
        p.y_train         = pd.Series([0]*368 + [1]*32)
        p.y_eval          = pd.Series([0]*92  + [1]*8)
        p.best_model_name = "logistic_regression"
        p.mlflow_runs     = {
            "logistic_regression": "run_lr",
            "dummy_baseline":      "run_dummy",
        }
        p.results = {
            "logistic_regression": mock_results_lr,
            "dummy_baseline":      mock_results_dummy,
        }
        p.step6_save(save_all=True)

        assert (tmp_models_dir / "logistic_regression_model.joblib").exists()
        assert (tmp_models_dir / "dummy_baseline_model.joblib").exists()


# =============================================================================
# 3. TESTS COMPARAISON ET SÉLECTION DU CHAMPION (step5_compare)
# =============================================================================

class TestModelCompare:

    def _make_pipeline_with_multiple_results(self, mock_results_lr, mock_cv_scores):
        """Pipeline avec plusieurs résultats pour tester la comparaison."""
        from sklearn.dummy import DummyClassifier

        p = Phase3Pipeline(source="csv", verbose=False)
        p.mlflow_client = MagicMock()  # Mock le client MLflow

        # Résultats LR (meilleur F2)
        p.results = {"logistic_regression": mock_results_lr}
        p.mlflow_runs = {"logistic_regression": "run_lr_001"}

        # Résultats Dummy (moins bon)
        mock_results_dummy = {
            **mock_results_lr,
            "scores_test":  {**mock_results_lr["scores_test"],
                             "f2": 0.12, "f1": 0.12, "roc_auc": 0.52},
            "scores_train": {**mock_results_lr["scores_train"],
                             "f2": 0.12, "f1": 0.12},
            "modele":       DummyClassifier(),
            "nom_modele":   "dummy_baseline",
        }
        p.results["dummy_baseline"] = mock_results_dummy
        p.mlflow_runs["dummy_baseline"] = "run_dummy_001"
        return p

    def test_compare_retourne_un_dataframe(self, mock_results_lr, mock_cv_scores):
        p = self._make_pipeline_with_multiple_results(mock_results_lr, mock_cv_scores)
        df = p.step5_compare()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_champion_est_en_tete_du_tableau(self, mock_results_lr, mock_cv_scores):
        """Le modèle avec le meilleur F2 doit être en position 0."""
        p = self._make_pipeline_with_multiple_results(mock_results_lr, mock_cv_scores)
        df = p.step5_compare()
        assert df.iloc[0]["Modèle"] == "logistic_regression"

    def test_champion_a_le_rang_trophy(self, mock_results_lr, mock_cv_scores):
        p = self._make_pipeline_with_multiple_results(mock_results_lr, mock_cv_scores)
        df = p.step5_compare()
        assert "🏆" in df.iloc[0]["Rang"]

    def test_tri_est_par_f2_score(self, mock_results_lr, mock_cv_scores):
        """Le tableau doit être trié par F2 décroissant."""
        p = self._make_pipeline_with_multiple_results(mock_results_lr, mock_cv_scores)
        df = p.step5_compare()
        f2_col = "F2(eval)"
        assert df[f2_col].is_monotonic_decreasing, \
            "Tableau doit être trié par F2-Score décroissant"

    def test_best_model_name_est_mis_a_jour(self, mock_results_lr, mock_cv_scores):
        p = self._make_pipeline_with_multiple_results(mock_results_lr, mock_cv_scores)
        p.step5_compare()
        assert p.best_model_name == "logistic_regression"

    def test_auc_roc_est_presente_dans_le_tableau(
        self, mock_results_lr, mock_cv_scores
    ):
        """AUC-ROC doit être visible dans le tableau comparatif."""
        p = self._make_pipeline_with_multiple_results(mock_results_lr, mock_cv_scores)
        df = p.step5_compare()
        assert any("AUC" in col for col in df.columns), \
            "AUC-ROC manquant dans le tableau de comparaison"

    def test_recall_est_presente_dans_le_tableau(
        self, mock_results_lr, mock_cv_scores
    ):
        p = self._make_pipeline_with_multiple_results(mock_results_lr, mock_cv_scores)
        df = p.step5_compare()
        assert any("Recall" in col or "recall" in col for col in df.columns)

    def test_overfitting_flag_est_present(self, mock_results_lr, mock_cv_scores):
        p = self._make_pipeline_with_multiple_results(mock_results_lr, mock_cv_scores)
        df = p.step5_compare()
        assert "Overfitting" in df.columns

    def test_compare_retourne_df_vide_si_pas_de_resultats(self):
        p = Phase3Pipeline(source="csv", verbose=False)
        p.mlflow_client = MagicMock()
        df = p.step5_compare()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_mlflow_run_id_est_present_dans_tableau(
        self, mock_results_lr, mock_cv_scores
    ):
        """Le run_id MLflow doit être visible pour le lien UI ↔ tableau."""
        p = self._make_pipeline_with_multiple_results(mock_results_lr, mock_cv_scores)
        df = p.step5_compare()
        assert "MLflow Run" in df.columns
        assert df.iloc[0]["MLflow Run"] == "run_lr_0"[:8]  # 8 premiers chars


# =============================================================================
# 4. TESTS CONFIGURATION DU PIPELINE
# =============================================================================

class TestPipelineConfig:

    def test_valeurs_par_defaut(self):
        p = Phase3Pipeline()
        assert p.source        == "db"
        assert p.eval_ratio    == 0.20
        assert p.random_state  == 42
        assert p.debug         is False
        assert p.verbose       is True

    def test_experiment_name_defaut_est_correct(self):
        p = Phase3Pipeline()
        assert p.experiment_name == MLflowConfig.EXPERIMENT_NAME
        assert "futurisys" not in p.experiment_name.lower()
        assert "attrition" not in p.experiment_name.lower()

    def test_target_col_est_target(self):
        """target_col doit correspondre à la colonne de v_features_engineering."""
        p = Phase3Pipeline()
        assert p.target_col == "target"

    def test_experiment_name_personnalise(self):
        p = Phase3Pipeline(experiment_name="Mon_Experience_Test")
        assert p.experiment_name == "Mon_Experience_Test"

    def test_source_csv_est_accepte(self):
        p = Phase3Pipeline(source="csv")
        assert p.source == "csv"

    def test_debug_mode(self):
        p = Phase3Pipeline(debug=True, debug_limit=500)
        assert p.debug        is True
        assert p.debug_limit  == 500

    def test_eval_ratio_personnalise(self):
        p = Phase3Pipeline(eval_ratio=0.30)
        assert p.eval_ratio == 0.30

    def test_attributs_resultats_initialises_a_none(self):
        p = Phase3Pipeline()
        assert p.X_train        is None
        assert p.X_eval         is None
        assert p.y_train        is None
        assert p.y_eval         is None
        assert p.modeler        is None
        assert p.best_model_name is None

    def test_dicts_resultats_initialises_vides(self):
        p = Phase3Pipeline()
        assert p.results          == {}
        assert p.mlflow_runs      == {}
        assert p.saved_model_paths == {}

    def test_verbose_false(self):
        """En mode silencieux, _log() ne doit rien afficher (test indirect)."""
        p = Phase3Pipeline(verbose=False)
        assert p.verbose is False
        # Appel _log() ne doit pas lever d'exception
        p._log("test message", "INFO")
        p._sep()
