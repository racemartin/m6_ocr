"""
tests/test_phase3_mlflow.py
============================
Tests — Intégration MLflow (Phase 3)

Ressource testée : Le tracking MLflow — configuration, logging des runs,
structure des paramètres et métriques loggués, artefacts.

IMPORTANT : Ces tests utilisent un serveur MLflow local en mémoire
(mlflow.set_tracking_uri("file://...")) pour ne pas dépendre
d'un serveur MLflow externe sur :5000.

Groupes de tests :
    1. TestMLflowConfig        → MLflowConfig : attributs, setup()
    2. TestMLflowRun           → Structure d'un run : tags, params, métriques
    3. TestMLflowParamsLogging → Params loggués : modèle + contexte
    4. TestMLflowMetricsLogging → Métriques : train / eval / CV / overfitting
    5. TestMLflowArtifacts     → Feature importance CSV loggué
    6. TestMLflowExperiment    → Création / restauration de l'expérience

Usage :
    pytest tests/test_phase3_mlflow.py -v
    pytest tests/test_phase3_mlflow.py -v -k "test_metrics"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy  as np
import pandas as pd
import pytest
import mlflow
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipelines.phase3_model_training_mlflow import MLflowConfig, Phase3Pipeline


# =============================================================================
# FIXTURE : TRACKING URI LOCALE (pas de serveur requis)
# =============================================================================


@pytest.fixture
def local_mlflow(tmp_path):
    """
    Configure un serveur MLflow local en utilisant un répertoire temporaire.
    Correction spéciale pour Windows (compatibilité URI).
    """
    import mlflow
    
    # Creamos la ruta del directorio para mlruns
    tracking_path = tmp_path / "mlruns"
    tracking_path.mkdir(parents=True, exist_ok=True)
    
    # En Windows, MLflow prefiere la ruta absoluta pura o con triple barra.
    # Esta es la forma más segura de convertir Path a URI compatible:
    tracking_uri = tracking_path.as_uri() 
    
    # Si as_uri() devuelve algo que MLflow sigue rechazando en tu setup, 
    # forzamos el formato manual:
    # tracking_uri = f"file:///{str(tracking_path).replace(os.sep, '/')}"

    mlflow.set_tracking_uri(tracking_uri)
    
    yield tracking_uri
    
    # --- CLEANUP ---
    # Cleanup : fermer les runs actifs
    # if mlflow.active_run():
    #     mlflow.end_run()
    
    try:
        # 1. Cerramos cualquier run que haya quedado abierto por un test fallido
        while mlflow.active_run():
            mlflow.end_run()
            
        # 2. Opcional: Resetear el URI para no contaminar otros módulos de test
        mlflow.set_tracking_uri("")
    except Exception as e:
        print(f"Error durante el cleanup de MLflow: {e}")

@pytest.fixture
def local_client(local_mlflow):
    return MlflowClient()


# =============================================================================
# 1. TESTS MLFLOWCONFIG
# =============================================================================

class TestMLflowConfig:
    """Vérifie la configuration centralisée MLflow."""

    def test_tracking_uri_est_defini(self):
        assert MLflowConfig.TRACKING_URI, "TRACKING_URI ne doit pas être vide"
        assert "://" in MLflowConfig.TRACKING_URI, \
            "TRACKING_URI doit être une URL valide (ex: http://...)"

    def test_experiment_name_est_defini_et_non_vide(self):
        assert MLflowConfig.EXPERIMENT_NAME
        assert len(MLflowConfig.EXPERIMENT_NAME) > 3

    def test_experiment_name_est_adapte_au_projet_home_credit(self):
        """Le nom de l'expérience ne doit PAS référencer l'ancien projet."""
        nom = MLflowConfig.EXPERIMENT_NAME.lower()
        assert "futurisys" not in nom,  "Nom expérience non adapté — contient 'Futurisys'"
        assert "attrition" not in nom,  "Nom expérience non adapté — contient 'Attrition'"

    def test_artifacts_root_est_un_path(self):
        """ARTIFACTS_ROOT est requis par setup_mlflow() pour créer l'expérience."""
        assert hasattr(MLflowConfig, "ARTIFACTS_ROOT"), \
            "ARTIFACTS_ROOT manquant — provoque AttributeError dans setup_mlflow()"
        assert isinstance(MLflowConfig.ARTIFACTS_ROOT, Path), \
            "ARTIFACTS_ROOT doit être un Path"

    def test_run_tags_contiennent_les_metadonnees_projet(self):
        tags = MLflowConfig.RUN_TAGS
        assert "project"          in tags
        assert "model_type"       in tags
        assert "pipeline_version" in tags
        assert "target"           in tags

    def test_run_tags_mentionnent_home_credit(self):
        tags_str = " ".join(str(v).lower() for v in MLflowConfig.RUN_TAGS.values())
        assert "home credit" in tags_str or "home_credit" in tags_str or \
               "prêt" in tags_str or "credit" in tags_str, \
            "Les tags MLflow doivent référencer le projet Home Credit"


# =============================================================================
# 2. TESTS STRUCTURE D'UN RUN MLFLOW
# =============================================================================

class TestMLflowRun:
    """Vérifie la structure d'un run MLflow complet."""

    def test_run_se_cree_et_se_ferme_proprement(self, local_mlflow):
        """Un run MLflow doit s'ouvrir et se fermer sans erreur."""
        exp_id = mlflow.create_experiment("test_run_structure")
        with mlflow.start_run(experiment_id=exp_id, run_name="test_run") as run:
            assert run.info.run_id
            assert run.info.status == "RUNNING"

        # Après le with, le run doit être terminé
        client = MlflowClient()
        run_data = client.get_run(run.info.run_id)
        assert run_data.info.status == "FINISHED"

    def test_run_actif_est_ferme_avant_nouveau_run(self, local_mlflow):
        """
        Si un run est actif (crash précédent), il doit être fermé avant
        de créer un nouveau run. Vérifie la robustesse du pipeline.
        """
        exp_id = mlflow.create_experiment("test_run_cleanup")

        # Simuler un run non fermé
        run1 = mlflow.start_run(experiment_id=exp_id, run_name="orphan")
        assert mlflow.active_run() is not None

        # Fermer le run actif (comme le fait _train_one_model)
        if mlflow.active_run():
            mlflow.end_run()

        assert mlflow.active_run() is None

        # Maintenant on peut démarrer un nouveau run
        with mlflow.start_run(experiment_id=exp_id, run_name="clean") as run2:
            assert run2.info.run_id != run1.info.run_id

    def test_run_name_contient_le_nom_du_modele(self, local_mlflow):
        """Chaque run doit avoir un nom identifiable avec le nom du modèle."""
        exp_id = mlflow.create_experiment("test_run_names")
        model_names = ["logistic_regression", "random_forest", "xgboost"]

        run_ids = []
        for nom in model_names:
            with mlflow.start_run(
                experiment_id=exp_id, run_name=f"{nom}_run"
            ) as run:
                run_ids.append(run.info.run_id)

        # Vérifier les noms via le client
        client = MlflowClient()
        for run_id, nom in zip(run_ids, model_names):
            run_data = client.get_run(run_id)
            assert f"{nom}_run" in run_data.info.run_name, \
                f"Run name doit contenir '{nom}_run'"


# =============================================================================
# 3. TESTS LOGGING DES PARAMÈTRES
# =============================================================================

class TestMLflowParamsLogging:
    """Vérifie que les paramètres loggués sont complets et corrects."""

    CONTEXT_PARAMS = [
        "random_state", "cv_folds", "eval_ratio",
        "n_features", "train_samples", "eval_samples", "target_rate",
    ]

    def test_params_contexte_sont_loggues(self, local_mlflow):
        """Les paramètres de contexte (dataset, CV, déséquilibre) doivent être loggués."""
        exp_id = mlflow.create_experiment("test_params_context")

        with mlflow.start_run(experiment_id=exp_id) as run:
            params = {
                "random_state": 42, "cv_folds": 5, "eval_ratio": 0.20,
                "n_features": 131, "train_samples": 246008,
                "eval_samples": 61502, "target_rate": 0.0807,
            }
            mlflow.log_params(params)

        client   = MlflowClient()
        run_data = client.get_run(run.info.run_id)
        logged   = run_data.data.params

        for key in self.CONTEXT_PARAMS:
            assert key in logged, f"Paramètre de contexte '{key}' non loggué"

    def test_params_modele_ont_prefixe_model_(self, local_mlflow):
        """Les hyperparamètres du modèle sont préfixés 'model_' pour éviter les collisions."""
        exp_id = mlflow.create_experiment("test_params_prefix")

        with mlflow.start_run(experiment_id=exp_id) as run:
            model_params = {"C": 0.1, "solver": "lbfgs", "class_weight": "balanced"}
            safe_params  = {f"model_{k}": str(v) for k, v in model_params.items()}
            mlflow.log_params(safe_params)

        client   = MlflowClient()
        run_data = client.get_run(run.info.run_id)
        logged   = run_data.data.params

        assert "model_C"            in logged
        assert "model_solver"       in logged
        assert "model_class_weight" in logged
        # Pas de collision avec les params de contexte
        assert "C"            not in logged
        assert "class_weight" not in logged

    def test_params_longs_sont_tronques_a_250_chars(self, local_mlflow):
        """MLflow limite la longueur des valeurs de paramètres."""
        exp_id = mlflow.create_experiment("test_params_truncation")

        long_value = "x" * 500

        with mlflow.start_run(experiment_id=exp_id) as run:
            # Le pipeline tronque à 250 chars : str(v)[:250]
            mlflow.log_params({"model_long": long_value[:250]})

        client   = MlflowClient()
        run_data = client.get_run(run.info.run_id)
        assert len(run_data.data.params["model_long"]) <= 250

    def test_target_rate_est_logguee(self, local_mlflow):
        """Le taux de défaut doit être loggué pour traçabilité du déséquilibre."""
        exp_id = mlflow.create_experiment("test_target_rate")

        with mlflow.start_run(experiment_id=exp_id) as run:
            mlflow.log_params({"target_rate": "0.0807"})

        client   = MlflowClient()
        run_data = client.get_run(run.info.run_id)
        assert "target_rate" in run_data.data.params
        assert float(run_data.data.params["target_rate"]) < 0.15, \
            "Taux de défaut Home Credit doit être < 15%"


# =============================================================================
# 4. TESTS LOGGING DES MÉTRIQUES
# =============================================================================

class TestMLflowMetricsLogging:
    """Vérifie que toutes les métriques requises sont loggées dans MLflow."""

    def _log_full_metrics(self, exp_id, scores_train, scores_test, scores_cv):
        """Helper : logue les métriques complètes d'un run."""
        with mlflow.start_run(experiment_id=exp_id) as run:
            for k, v in scores_train.items():
                if isinstance(v, float) and not np.isnan(v):
                    mlflow.log_metric(f"train_{k}", v)
            for k, v in scores_test.items():
                if isinstance(v, float) and not np.isnan(v):
                    mlflow.log_metric(f"eval_{k}", v)
            for metric_name, d in scores_cv.items():
                mlflow.log_metric(f"cv_{metric_name}_mean",       d["cv_moyenne"])
                mlflow.log_metric(f"cv_{metric_name}_std",        d["cv_ecart"])
                mlflow.log_metric(f"cv_train_{metric_name}_mean", d["train_moyenne"])
            mlflow.log_metric("overfitting_f1",
                scores_train["f1"] - scores_test["f1"])
            mlflow.log_metric("train_time_s", 0.84)
        return run.info.run_id

    def test_metriques_eval_metier_sont_logguees(
        self, local_mlflow, mock_scores_train, mock_scores_test, mock_cv_scores
    ):
        """AUC-ROC, F2, F1, Recall doivent être loggués sur l'eval set."""
        exp_id = mlflow.create_experiment("test_metrics_metier")
        run_id = self._log_full_metrics(exp_id, mock_scores_train,
                                         mock_scores_test, mock_cv_scores)

        client   = MlflowClient()
        run_data = client.get_run(run_id)
        metrics  = run_data.data.metrics

        for m in ["eval_roc_auc", "eval_f2", "eval_f1", "eval_recall"]:
            assert m in metrics, f"Métrique métier '{m}' non logguée dans MLflow"

    def test_metriques_cv_sont_logguees_avec_mean_et_std(
        self, local_mlflow, mock_scores_train, mock_scores_test, mock_cv_scores
    ):
        """Les métriques CV doivent avoir mean ET std pour évaluer la stabilité."""
        exp_id = mlflow.create_experiment("test_metrics_cv")
        run_id = self._log_full_metrics(exp_id, mock_scores_train,
                                         mock_scores_test, mock_cv_scores)

        client  = MlflowClient()
        metrics = client.get_run(run_id).data.metrics

        for m in ["f1", "roc_auc", "f2"]:
            assert f"cv_{m}_mean" in metrics, f"cv_{m}_mean non loggué"
            assert f"cv_{m}_std"  in metrics, f"cv_{m}_std non loggué"

    def test_overfitting_f1_est_logguee(
        self, local_mlflow, mock_scores_train, mock_scores_test, mock_cv_scores
    ):
        """L'indicateur d'overfitting doit être loggué pour analyse post-entraînement."""
        exp_id = mlflow.create_experiment("test_metrics_overfitting")
        run_id = self._log_full_metrics(exp_id, mock_scores_train,
                                         mock_scores_test, mock_cv_scores)

        client  = MlflowClient()
        metrics = client.get_run(run_id).data.metrics

        assert "overfitting_f1" in metrics
        expected = mock_scores_train["f1"] - mock_scores_test["f1"]
        assert abs(metrics["overfitting_f1"] - expected) < 1e-4

    def test_train_time_est_logguee(
        self, local_mlflow, mock_scores_train, mock_scores_test, mock_cv_scores
    ):
        exp_id = mlflow.create_experiment("test_metrics_time")
        run_id = self._log_full_metrics(exp_id, mock_scores_train,
                                         mock_scores_test, mock_cv_scores)

        client  = MlflowClient()
        metrics = client.get_run(run_id).data.metrics

        assert "train_time_s" in metrics
        assert metrics["train_time_s"] > 0

    def test_metriques_nan_ne_sont_pas_logguees(self, local_mlflow):
        """Les NaN (ex: AUC-ROC pour DummyClassifier) ne doivent pas planter le log."""
        exp_id  = mlflow.create_experiment("test_metrics_nan")
        scores  = {"f1": 0.0, "roc_auc": float("nan"), "recall": 0.0}

        with mlflow.start_run(experiment_id=exp_id) as run:
            for k, v in scores.items():
                if isinstance(v, float) and not np.isnan(v):
                    mlflow.log_metric(f"eval_{k}", v)  # NaN ignoré

        client  = MlflowClient()
        metrics = client.get_run(run.info.run_id).data.metrics

        assert "eval_f1"      in metrics
        assert "eval_roc_auc" not in metrics   # NaN n'est pas loggué


# =============================================================================
# 5. TESTS ARTEFACTS MLFLOW
# =============================================================================

class TestMLflowArtifacts:
    """Vérifie la gestion des artefacts (feature importance, modèle sklearn)."""

    def test_feature_importance_est_un_csv_avec_colonnes_correctes(self, tmp_path):
        """Le CSV feature importance doit avoir 'feature' et 'importance'."""
        from sklearn.ensemble import RandomForestClassifier

        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0])

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        fi_path = tmp_path / "random_forest_feature_importance.csv"
        pd.DataFrame({
            "feature":    list(X.columns),
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).to_csv(fi_path, index=False)

        # Vérification du fichier produit
        df = pd.read_csv(fi_path)
        assert "feature"    in df.columns
        assert "importance" in df.columns
        assert len(df) == len(X.columns)
        assert df["importance"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_feature_importance_est_triee_descendante(self, tmp_path):
        """Les features les plus importantes doivent être en tête du CSV."""
        fi_path = tmp_path / "fi.csv"
        pd.DataFrame({
            "feature":    ["f3", "f1", "f2"],
            "importance": [0.6, 0.3, 0.1],
        }).sort_values("importance", ascending=False).to_csv(fi_path, index=False)

        df = pd.read_csv(fi_path)
        assert df["importance"].is_monotonic_decreasing, \
            "Feature importance doit être triée par ordre décroissant"

    def test_modele_sklearn_est_loggable_dans_mlflow(self, local_mlflow, trained_lr):
        """Un modèle sklearn doit pouvoir être loggué sans erreur."""
        exp_id = mlflow.create_experiment("test_artifact_model")

        with mlflow.start_run(experiment_id=exp_id) as run:
            mlflow.sklearn.log_model(trained_lr, "model")

        client   = MlflowClient()
        run_data = client.get_run(run.info.run_id)
        artifacts = client.list_artifacts(run.info.run_id)
        artifact_names = [a.path for a in artifacts]

        assert "model" in artifact_names, "Dossier 'model' manquant dans les artefacts"

    def test_dummy_classifier_est_loggable_sans_signature(
        self, local_mlflow, trained_dummy
    ):
        """DummyClassifier doit être loggué même si la signature échoue."""
        exp_id = mlflow.create_experiment("test_artifact_dummy")

        with mlflow.start_run(experiment_id=exp_id) as run:
            try:
                from mlflow.models import infer_signature
                sig = infer_signature(
                    pd.DataFrame({"f": [0.0]}),
                    trained_dummy.predict(pd.DataFrame({"f": [0.0]})),
                )
                mlflow.sklearn.log_model(trained_dummy, "model", signature=sig)
            except Exception:
                # Fallback sans signature
                mlflow.sklearn.log_model(trained_dummy, "model")

        artifacts = MlflowClient().list_artifacts(run.info.run_id)
        assert any("model" in a.path for a in artifacts)


# =============================================================================
# 6. TESTS GESTION DE L'EXPÉRIENCE
# =============================================================================

class TestMLflowExperiment:
    """Vérifie la création et gestion de l'expérience MLflow."""

    def test_creation_nouvelle_experience(self, local_mlflow):
        """Une nouvelle expérience doit être créée si elle n'existe pas."""
        client  = MlflowClient()
        exp_nom = "test_nouvelle_experience_phase3"

        # Vérifier qu'elle n'existe pas
        assert client.get_experiment_by_name(exp_nom) is None

        # Créer
        exp_id = client.create_experiment(
            name=exp_nom,
            artifact_location=str(MLflowConfig.ARTIFACTS_ROOT),
        )
        assert exp_id is not None

        exp = client.get_experiment_by_name(exp_nom)
        assert exp.name == exp_nom

    def test_experience_existante_est_reutilisee(self, local_mlflow):
        """Si l'expérience existe déjà, son ID doit être réutilisé."""
        client  = MlflowClient()
        exp_nom = "test_exp_reutilisee"

        # Première création
        exp_id1 = client.create_experiment(exp_nom)

        # Récupération (simulation de ce que fait setup_mlflow())
        exp = client.get_experiment_by_name(exp_nom)
        exp_id2 = exp.experiment_id

        assert exp_id1 == exp_id2, "L'expérience existante doit être réutilisée, pas recréée"

    def test_tag_best_model_est_pose_sur_le_champion(self, local_mlflow):
        """Le champion doit recevoir le tag 'best_model=true'."""
        client  = MlflowClient()
        exp_id  = mlflow.create_experiment("test_tag_best_model")

        run_ids = {}
        for nom in ["logistic_regression", "random_forest"]:
            with mlflow.start_run(experiment_id=exp_id, run_name=f"{nom}_run") as run:
                mlflow.log_metric("eval_f2", 0.5 if nom == "logistic_regression" else 0.3)
                run_ids[nom] = run.info.run_id

        # Poser le tag sur le champion
        champion = "logistic_regression"
        client.set_tag(run_ids[champion], "best_model", "true")

        # Vérifier
        run_data = client.get_run(run_ids[champion])
        assert run_data.data.tags.get("best_model") == "true"

        run_data_other = client.get_run(run_ids["random_forest"])
        assert "best_model" not in run_data_other.data.tags

    def test_plusieurs_runs_dans_la_meme_experience(self, local_mlflow):
        """8 runs (un par modèle) doivent coexister dans la même expérience."""
        client  = MlflowClient()
        exp_id  = mlflow.create_experiment("test_8_runs")
        modeles = ["dummy_baseline", "logistic_regression", "decision_tree",
                   "random_forest", "gradient_boosting", "xgboost", "lightgbm", "mlp"]

        run_ids = []
        for nom in modeles:
            with mlflow.start_run(experiment_id=exp_id, run_name=f"{nom}_run") as run:
                mlflow.log_metric("eval_f2", np.random.uniform(0, 0.6))
                run_ids.append(run.info.run_id)

        runs = client.search_runs(experiment_ids=[exp_id])
        assert len(runs) == len(modeles), \
            f"Attendu {len(modeles)} runs, obtenu {len(runs)}"
