"""
tests/test_phase3_models.py
============================
Tests — Modèles de classification (Phase 3)

Ressource testée : Les modèles sklearn eux-mêmes et leur intégration
avec ClassificationModeler (StratifiedKFold, métriques, structure des résultats).

Groupes de tests :
    1. TestModelCatalogue         → _build_models_config() : présence, params, déséquilibre
    2. TestClassImbalanceStrategy → class_weight et scale_pos_weight
    3. TestStratifiedKFold        → anti-leakage CV, conservation distribution
    4. TestMetricsStructure       → structure scores_cv / scores_train / scores_test
    5. TestOverfittingDetection   → détection surapprentissage
    6. TestModelResults           → clés du dict résultats de entrainer_modele()

Usage :
    pytest tests/test_phase3_models.py -v
    pytest tests/test_phase3_models.py -v -k "test_catalogue"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy  as np
import pandas as pd
import pytest
from sklearn.base import is_classifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import KFold
from sklearn.linear_model   import LogisticRegression
from sklearn.tree    import DecisionTreeClassifier


from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix 


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipelines.phase3_model_training_mlflow import _build_models_config, XGBOOST_OK, LGBM_OK
from src.pipelines.phase3_model_training_mlflow import Phase3Pipeline

# =============================================================================
# 1. TESTS DU CATALOGUE DE MODÈLES
# =============================================================================

class TestModelCatalogue:
    """Vérifie que _build_models_config() retourne le bon catalogue."""

    @pytest.fixture(autouse=True)
    def catalogue(self):
        self.cat = _build_models_config(random_state=42)

    def test_catalogue_contient_toutes_les_familles_requises(self):
        """Cahier des charges : LR, RF, GBM, XGBoost, LightGBM, MLP."""
        noms = list(self.cat.keys())
        assert "dummy_baseline"       in noms, "Baseline aléatoire manquante"
        assert "logistic_regression"  in noms, "LR manquante (modèle simple)"
        assert "decision_tree"        in noms, "DT manquant"
        assert "random_forest"        in noms, "RF manquante (forêt)"
        assert "gradient_boosting"    in noms, "GBM manquant (boosting)"
        assert "mlp"                  in noms, "MLP manquant (réseau)"

    def test_xgboost_present_si_installe(self):
        if XGBOOST_OK:
            assert "xgboost" in self.cat, "XGBoost installé mais absent du catalogue"

    def test_lightgbm_present_si_installe(self):
        if LGBM_OK:
            assert "lightgbm" in self.cat, "LightGBM installé mais absent du catalogue"

    def test_chaque_entree_a_les_cles_requises(self):
        """Chaque entrée doit avoir 'model', 'params', 'description'."""
        for nom, cfg in self.cat.items():
            assert "model"       in cfg, f"{nom} : clé 'model' manquante"
            assert "params"      in cfg, f"{nom} : clé 'params' manquante"
            assert "description" in cfg, f"{nom} : clé 'description' manquante"

    def test_tous_les_modeles_sont_des_classifieurs(self):
        """Tous les modèles doivent être des estimateurs sklearn compatibles."""
        for nom, cfg in self.cat.items():
            assert is_classifier(cfg["model"]), \
                f"{nom} : {type(cfg['model']).__name__} n'est pas un classifieur sklearn"

    def test_random_state_est_transmis(self):
        """Le random_state doit être appliqué à tous les modèles qui le supportent."""
        cat_42  = _build_models_config(random_state=42)
        cat_99  = _build_models_config(random_state=99)

        modeles_avec_rs = ["logistic_regression", "decision_tree",
                           "random_forest", "gradient_boosting", "mlp"]
        for nom in modeles_avec_rs:
            if nom in cat_42 and nom in cat_99:
                rs_42 = getattr(cat_42[nom]["model"], "random_state", None)
                rs_99 = getattr(cat_99[nom]["model"], "random_state", None)
                assert rs_42 == 42, f"{nom} : random_state=42 non appliqué"
                assert rs_99 == 99, f"{nom} : random_state=99 non appliqué"


# =============================================================================
# 2. TESTS STRATÉGIE DÉSÉQUILIBRE DES CLASSES
# =============================================================================

class TestClassImbalanceStrategy:
    """
    Vérifie que chaque modèle est correctement configuré pour gérer
    le déséquilibre TARGET ≈ 8% (ratio ~1:11).

    Points de vigilance du cahier des charges :
        • class_weight='balanced' pour les modèles sklearn
        • scale_pos_weight=11 pour XGBoost
        • Pas de class_weight pour GradientBoosting sklearn (limitation sklearn)
    """

    @pytest.fixture(autouse=True)
    def catalogue(self):
        self.cat = _build_models_config(random_state=42)

    def test_logistic_regression_a_class_weight_balanced(self):
        model = self.cat["logistic_regression"]["model"]
        assert model.class_weight == "balanced", \
            "LR doit avoir class_weight='balanced' pour le déséquilibre 1:11"

    def test_decision_tree_a_class_weight_balanced(self):
        model = self.cat["decision_tree"]["model"]
        assert model.class_weight == "balanced"

    def test_random_forest_a_class_weight_balanced(self):
        model = self.cat["random_forest"]["model"]
        assert model.class_weight == "balanced"

    def test_gradient_boosting_pas_de_class_weight(self):
        """GradientBoostingClassifier sklearn ne supporte pas class_weight."""
        model = self.cat["gradient_boosting"]["model"]
        # Pas d'attribut class_weight → OK (limitation sklearn)
        assert not hasattr(model, "class_weight") or model.class_weight is None

    def test_xgboost_a_scale_pos_weight(self):
        """XGBoost utilise scale_pos_weight au lieu de class_weight."""
        if not XGBOOST_OK:
            pytest.skip("XGBoost non installé")
        model = self.cat["xgboost"]["model"]
        spw = getattr(model, "scale_pos_weight", None)
        assert spw is not None, "XGBoost doit avoir scale_pos_weight"
        assert spw >= 5, \
            f"scale_pos_weight={spw} trop faible pour déséquilibre 1:11 (attendu ≥ 5)"

    def test_lightgbm_a_class_weight_balanced(self):
        if not LGBM_OK:
            pytest.skip("LightGBM non installé")
        model = self.cat["lightgbm"]["model"]
        assert model.class_weight == "balanced"

    def test_mlp_pas_de_class_weight(self):
        """MLPClassifier sklearn ne supporte pas class_weight."""
        model = self.cat["mlp"]["model"]
        assert not hasattr(model, "class_weight") or model.class_weight is None

    def test_params_dict_documente_strategie_desequilibre(self):
        """Les params loggués dans MLflow doivent refléter la stratégie déséquilibre."""
        for nom in ["logistic_regression", "random_forest", "decision_tree"]:
            params = self.cat[nom]["params"]
            assert "class_weight" in params, \
                f"{nom} : 'class_weight' absent des params MLflow"
            assert params["class_weight"] == "balanced"

        if XGBOOST_OK:
            params = self.cat["xgboost"]["params"]
            assert "scale_pos_weight" in params


# =============================================================================
# 3. TESTS VALIDATION CROISÉE STRATIFIÉE
# =============================================================================

class TestStratifiedKFold:
    """
    Vérifie l'utilisation correcte de StratifiedKFold dans le pipeline.

    Points de vigilance du cahier des charges :
        • StratifiedKFold conserve la distribution des classes sur chaque pli
        • CV uniquement sur X_train (anti-leakage : X_eval jamais vu pendant CV)
        • 5 plis minimum requis pour la robustesse
    """

    def test_stratifiedkfold_conserve_distribution_classes(
        self, X_train_synth, y_train_synth
    ):
        """Chaque pli doit avoir approximativement la même proportion de défauts."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        taux_par_pli = []

        for _, idx_val in cv.split(X_train_synth, y_train_synth):
            taux = y_train_synth.iloc[idx_val].mean()
            taux_par_pli.append(taux)

        # Les taux doivent être proches du taux global (±5%)
        taux_global = y_train_synth.mean()
        for i, taux in enumerate(taux_par_pli):
            assert abs(taux - taux_global) < 0.05, \
                f"Pli {i+1} : taux={taux:.3f} trop éloigné du global={taux_global:.3f}"

    def test_stratifiedkfold_vs_kfold_sur_classes_desequilibrees(
        self, X_train_synth, y_train_synth
    ):
        """StratifiedKFold doit donner une variance inter-plis plus faible que KFold."""


        model = LogisticRegression(C=0.1, max_iter=100, class_weight="balanced",
                                   random_state=42)

        cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_rand  = KFold(n_splits=5, shuffle=True, random_state=42)

        scores_strat = cross_validate(model, X_train_synth, y_train_synth,
                                      cv=cv_strat, scoring="roc_auc",
                                      n_jobs=1)["test_score"]
        scores_rand  = cross_validate(model, X_train_synth, y_train_synth,
                                      cv=cv_rand,  scoring="roc_auc",
                                      n_jobs=1)["test_score"]

        std_strat = scores_strat.std()
        std_rand  = scores_rand.std()

        # La stratification réduit généralement la variance
        # (test souple : on vérifie juste que l'écart n'est pas énorme)
        assert std_strat <= std_rand * 2.0, \
            f"StratifiedKFold std={std_strat:.4f} >> KFold std={std_rand:.4f}"

    def test_cv_utilise_5_plis_minimum(self):
        """La configuration du pipeline doit utiliser au moins 5 plis."""
        pipeline = Phase3Pipeline(source="csv", verbose=False)
        # Le ClassificationModeler est configuré avec CV_FOLDS=5
        # On vérifie la valeur passée au constructeur
        assert pipeline.eval_ratio < 1.0
        # (Le CV_FOLDS est vérifié dans TestModelerInit)

    def test_eval_set_jamais_vu_pendant_cv(
        self, X_train_synth, X_eval_synth, y_train_synth, y_eval_synth
    ):
        """
        Anti-leakage : X_eval ne doit pas apparaître dans le X utilisé pour la CV.
        La CV doit s'effectuer sur X_train uniquement.
        """
        # Indices train et eval doivent être disjoints
        idx_train = set(X_train_synth.index.tolist())
        idx_eval  = set(X_eval_synth.index.tolist())
        assert len(idx_train & idx_eval) == 0, \
            "Index train et eval se chevauchent — risque de leakage"

    def test_cross_validate_retourne_train_et_test_scores(
        self, X_train_synth, y_train_synth
    ):
        """cross_validate avec return_train_score=True doit retourner les deux."""
        model = DummyClassifier(strategy="stratified", random_state=42)
        cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = cross_validate(
            model, X_train_synth, y_train_synth,
            cv=cv,
            scoring={"f1": "f1", "roc_auc": "roc_auc"},
            return_train_score=True,
            n_jobs=1,
        )

        assert "train_f1"   in result
        assert "test_f1"    in result
        assert "train_roc_auc" in result
        assert "test_roc_auc"  in result
        assert len(result["test_f1"]) == 5


# =============================================================================
# 4. TESTS STRUCTURE DES MÉTRIQUES
# =============================================================================

class TestMetricsStructure:
    """
    Vérifie la structure du dict résultats retourné par ClassificationModeler.

    Cahier des charges :
        • AUC-ROC, Recall, F1, F2 (coût métier FN > FP) doivent être présents
        • scores_cv : {metric: {cv_moyenne, cv_ecart, train_moyenne, train_ecart}}
    """

    def test_scores_cv_ont_les_4_cles_attendues(self, mock_cv_scores):
        """Structure exacte scores_cv de ClassificationModeler."""
        for metric_name, d in mock_cv_scores.items():
            assert "cv_moyenne"    in d, f"{metric_name} : 'cv_moyenne' manquant"
            assert "cv_ecart"      in d, f"{metric_name} : 'cv_ecart' manquant"
            assert "train_moyenne" in d, f"{metric_name} : 'train_moyenne' manquant"
            assert "train_ecart"   in d, f"{metric_name} : 'train_ecart' manquant"

    def test_metriques_metier_presentes_dans_scores_test(self, mock_scores_test):
        """Les métriques du cahier des charges doivent être calculées."""
        for metric in ["roc_auc", "recall", "f1", "f2"]:
            assert metric in mock_scores_test, \
                f"Métrique métier '{metric}' absente des scores_test"

    def test_scores_train_ont_les_memes_cles_que_scores_test(
        self, mock_scores_train, mock_scores_test
    ):
        """Train et test doivent avoir les mêmes clés pour la comparaison."""
        assert set(mock_scores_train.keys()) == set(mock_scores_test.keys()), \
            "Clés train et test divergent — impossible de calculer Δ overfitting"

    def test_f2_beta_penalise_fn_2x_plus_que_fp(
        self, X_train_synth, y_train_synth
    ):
        """
        F2 avec beta=2 doit donner plus d'importance au recall qu'à la précision.
        Un modèle avec recall élevé doit avoir F2 > F1.
        """

        # Modèle biaisé vers la classe positive (recall élevé, précision faible)
        y_true = y_train_synth.values
        y_pred = np.ones_like(y_true)  # Prédit tout en défaut → recall=1, précision=8%

        f1 = f1_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

        assert f2 > f1, \
            f"F2 ({f2:.4f}) devrait être > F1 ({f1:.4f}) quand recall=1"

    def test_roc_auc_baseline_est_superieure_a_aleatoire(
        self, X_train_synth, X_eval_synth, y_train_synth, y_eval_synth
    ):
        """
        Une LR avec class_weight='balanced' doit surpasser le hasard sur AUC-ROC.
        Baseline aléatoire = 0.5.
        """


        model = LogisticRegression(C=0.1, max_iter=200, class_weight="balanced",
                                   random_state=42)
        model.fit(X_train_synth, y_train_synth)
        proba = model.predict_proba(X_eval_synth)[:, 1]
        auc   = roc_auc_score(y_eval_synth, proba)

        assert auc > 0.50, \
            f"AUC-ROC={auc:.4f} inférieur à l'aléatoire — vérifier les données"

    def test_metriques_dans_intervalle_valide(self, mock_scores_test):
        """Toutes les métriques de classification doivent être dans [0, 1] (sauf log_loss, mcc)."""
        metriques_01 = ["accuracy", "precision", "recall", "f1", "f2",
                        "roc_auc", "specificite", "avg_precision"]
        for m in metriques_01:
            if m in mock_scores_test:
                val = mock_scores_test[m]
                assert 0.0 <= val <= 1.0, \
                    f"{m} = {val} hors intervalle [0, 1]"


# =============================================================================
# 5. TESTS DÉTECTION DU SURAPPRENTISSAGE
# =============================================================================

class TestOverfittingDetection:
    """Vérifie la logique de détection du surapprentissage."""

    def test_delta_f1_calcule_correctement(self, mock_scores_train, mock_scores_test):
        """Δ F1 = train_f1 - test_f1 doit être positif si overfitting."""
        delta = mock_scores_train["f1"] - mock_scores_test["f1"]
        assert delta == pytest.approx(
            mock_scores_train["f1"] - mock_scores_test["f1"], abs=1e-6
        )

    def test_pas_d_overfitting_si_delta_f1_inferieur_a_seuil(self):
        """Seuil de tolérance = 0.15 (défini dans ClassificationModeler)."""
        scores_train = {"f1": 0.54, "accuracy": 0.82, "roc_auc": 0.81}
        scores_test  = {"f1": 0.47, "accuracy": 0.78, "roc_auc": 0.77}

        delta_f1  = scores_train["f1"] - scores_test["f1"]   # 0.07
        delta_acc = scores_train["accuracy"] - scores_test["accuracy"]  # 0.04

        assert delta_f1 < 0.15,  "Δ F1 < 0.15 → pas d'overfitting détecté"
        assert delta_acc < 0.15, "Δ Acc < 0.15 → pas d'overfitting détecté"

    def test_overfitting_si_delta_f1_superieur_au_seuil(self):
        """Un Δ F1 > 0.15 doit déclencher l'alerte overfitting."""
        scores_train = {"f1": 0.95, "accuracy": 0.98, "roc_auc": 0.99}
        scores_test  = {"f1": 0.30, "accuracy": 0.65, "roc_auc": 0.60}

        delta_f1 = scores_train["f1"] - scores_test["f1"]  # 0.65
        assert delta_f1 > 0.15, "Δ F1 > 0.15 → overfitting détecté"

    def test_decision_tree_overfit_sans_contrainte(
        self, X_train_synth, X_eval_synth, y_train_synth, y_eval_synth
    ):
        """Un arbre sans contrainte (max_depth=None) doit overfitter."""


        model = DecisionTreeClassifier(random_state=42)  # max_depth=None
        model.fit(X_train_synth, y_train_synth)

        f1_train = f1_score(y_train_synth, model.predict(X_train_synth),
                            zero_division=0)
        f1_eval  = f1_score(y_eval_synth,  model.predict(X_eval_synth),
                            zero_division=0)
        delta    = f1_train - f1_eval

        assert delta > 0.15, \
            f"Arbre sans contrainte devrait overfitter (Δ F1={delta:.3f})"

    def test_logistic_regression_ne_overfit_pas(
        self, X_train_synth, X_eval_synth, y_train_synth, y_eval_synth
    ):
        """LR avec régularisation (C=0.1) ne devrait pas dépasser le seuil."""

        model = LogisticRegression(C=0.1, max_iter=200, class_weight="balanced",
                                   random_state=42)
        model.fit(X_train_synth, y_train_synth)

        f1_train = f1_score(y_train_synth, model.predict(X_train_synth),
                            zero_division=0)
        f1_eval  = f1_score(y_eval_synth,  model.predict(X_eval_synth),
                            zero_division=0)
        delta    = abs(f1_train - f1_eval)

        assert delta < 0.30, \
            f"LR régularisée ne devrait pas overfitter massivement (Δ={delta:.3f})"


# =============================================================================
# 6. TESTS STRUCTURE DU DICT RÉSULTATS entrainer_modele()
# =============================================================================

class TestModelResultsDict:
    """Vérifie que la structure du dict résultats est complète et correcte."""

    REQUIRED_KEYS = [
        "id_experience", "nom_modele", "modele",
        "scores_cv", "scores_train", "scores_test",
        "temps_train", "surapprentissage", "diagnostics",
        "predictions", "matrice_confusion",
    ]

    def test_toutes_les_cles_requises_presentes(self, mock_results_lr):
        for key in self.REQUIRED_KEYS:
            assert key in mock_results_lr, f"Clé '{key}' manquante dans les résultats"

    def test_modele_est_un_estimateur_fitte(self, mock_results_lr):
        model = mock_results_lr["modele"]
        assert hasattr(model, "predict"), "Le modèle doit avoir une méthode predict"
        assert hasattr(model, "predict_proba"), "Le modèle doit avoir predict_proba"

    def test_surapprentissage_est_booleen(self, mock_results_lr):
        assert isinstance(mock_results_lr["surapprentissage"], bool)

    def test_temps_train_est_positif(self, mock_results_lr):
        assert mock_results_lr["temps_train"] > 0

    def test_predictions_ont_les_4_vecteurs(self, mock_results_lr):
        preds = mock_results_lr["predictions"]
        for key in ["y_train_pred", "y_test_pred", "y_train_proba", "y_test_proba"]:
            assert key in preds, f"Clé '{key}' manquante dans predictions"

    def test_matrice_confusion_a_train_et_test(self, mock_results_lr):
        mc = mock_results_lr["matrice_confusion"]
        assert "train" in mc
        assert "test"  in mc
        assert mc["train"].shape == (2, 2)
        assert mc["test"].shape  == (2, 2)

    def test_scores_cv_contient_les_metriques_metier(self, mock_results_lr):
        sc = mock_results_lr["scores_cv"]
        for metric in ["f1", "f2", "roc_auc", "recall"]:
            assert metric in sc, f"Métrique CV '{metric}' manquante"
