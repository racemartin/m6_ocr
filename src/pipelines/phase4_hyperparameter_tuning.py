"""
src/pipelines/phase4_hyperparameter_tuning.py
==============================================
Phase 4 — Optimisation des hyperparamètres + Seuil métier
Projet : Prêt à Dépenser — Home Credit Default Risk

Objectif métier
───────────────
Maximiser le F2-Score du champion issu de Phase 3.
Définir le seuil de décision optimal (pas 0.5 par défaut) basé sur
le coût pondéré des erreurs : FN coûte 5× plus qu'un FP.

Deux moteurs disponibles (comparables dans MLflow)
──────────────────────────────────────────────────
    GridSearchCV  → Exhaustif, reproductible, facile à expliquer
    Optuna        → Bayésien, rapide sur grands espaces, pruning intégré

Pourquoi Optuna est différent de GridSearchCV ?
───────────────────────────────────────────────
    • GridSearchCV évalue toutes les combinaisons (n¹ × n² × … × nᵏ)
      → coûteux mais totalement reproductible et transparent.
    • Optuna utilise l'optimisation bayésienne (TPE — Tree Parzen Estimator)
      → explore intelligemment, concentre les essais là où ça performe.
    • Optuna intègre le Pruning : interrompt les essais mauvais dès les
      premières itérations (économie de temps jusqu'à 10×).
    • Optuna est asynchrone et distribué (multi-process, multi-machine).
    → Pour Phase 4, les deux sont lancés en parallèle et comparés dans MLflow.

Seuil métier (Threshold)
─────────────────────────
    Seuil 0.5 → inadapté aux 8% de défauts.
    On balaye [0.05 – 0.80] en minimisant le coût métier total :
        Coût = FN × COUT_FN + FP × COUT_FP
    avec COUT_FN = 5 (défaut non détecté = perte totale du prêt)
         COUT_FP = 1 (refus abusif = manque à gagner)

Artefacts produits
──────────────────
    models/phase4_best_model.joblib         → modèle optimisé sklearn
    models/phase4_best_model_metadata.json  → métriques + seuil + run_id
    reports/phase4_threshold_curve.csv      → courbe coût vs seuil
    mlruns/                                 → runs MLflow (expérience Phase4)

Prérequis
─────────
    uv add optuna optuna-integration
    uv add plotly   # pour les visualisations Optuna

Usage
─────
    # Terminal 1 — MLflow (si pas encore démarré)
    docker compose up -d mlflow

    # Terminal 2 — Lancer Phase 4
    uv run python -m src.pipelines.phase4_hyperparameter_tuning
    uv run python -m src.pipelines.phase4_hyperparameter_tuning --engine gridsearch
    uv run python -m src.pipelines.phase4_hyperparameter_tuning --engine optuna
    uv run python -m src.pipelines.phase4_hyperparameter_tuning --engine both
    uv run python -m src.pipelines.phase4_hyperparameter_tuning --debug
    uv run python -m src.pipelines.phase4_hyperparameter_tuning --n-trials 30
"""

from __future__ import annotations

# ── Bibliothèques standard ─────────────────────────────────────────────────
import argparse
import json
import os
import time
import traceback
import warnings
import datetime 

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Infrastructure ────────────────────────────────────────────────────────
import joblib
import mlflow
import mlflow.sklearn
import numpy  as np
import pandas as pd
from mlflow        import MlflowClient
from mlflow.models import infer_signature
from sqlalchemy    import text

# ── Scikit-learn ───────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

from sklearn.dummy          import DummyClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
    confusion_matrix,
)

# --- Scikit-Learn: Estructura y Modelos --------------------------------------
from sklearn.pipeline import Pipeline

from sklearn.base import clone as sklearn_clone

# --- Manejo de Desbalanceo (Imbalanced-Learn) --------------------------------
from imblearn.over_sampling import SMOTE

# ── Garde-fous pour les librairies optionnelles ────────────────────────────
warnings.filterwarnings("ignore")

XGBOOST_OK = False
try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    print("⚠️  XGBoost non installé → uv add xgboost")

LGBM_OK = False
try:
    from lightgbm import LGBMClassifier
    LGBM_OK = True
except ImportError:
    print("⚠️  LightGBM non installé → uv add lightgbm")

# ─────────────────────────────────────────────────────────────────────────
# OPTUNA — Installation requise : uv add optuna optuna-integration
#
# Qu'est-ce qu'Optuna ?
# ─────────────────────
# Optuna est un framework d'optimisation hyperparamétrique automatique.
# Il utilise TPE (Tree-structured Parzen Estimator) : un algorithme
# bayésien qui modélise la distribution des bons hyperparamètres
# et concentre les essais suivants dans les régions prometteuses.
#
# Concepts clés d'Optuna :
#   • Study    → l'étude globale (= une session d'optimisation)
#   • Trial    → un essai unique (= une combinaison d'hyperparamètres)
#   • Objective→ la fonction à maximiser/minimiser
#   • Pruner   → coupe les essais mauvais dès les premières itérations
#   • Sampler  → algorithme de recherche (TPE par défaut)
#
# Comparaison avec GridSearchCV :
#   GridSearchCV : évalue TOUTES les combinaisons → 4×4×4 = 64 essais
#   Optuna       : évalue n_trials essais INTELLIGENTS → 30 essais suffisent
# ─────────────────────────────────────────────────────────────────────────
OPTUNA_OK = False
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners  import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    print("⚠️  Optuna non installé → uv add optuna optuna-integration")
    print("   Les visualisations Optuna nécessitent aussi : uv add plotly")

# Connexion DB
DB_AVAILABLE = False
try:
    from src.database import get_engine
    DB_AVAILABLE = True
except ImportError:
    print("⚠️  src.database introuvable — Mode CSV")

# ── Constantes ─────────────────────────────────────────────────────────────
DEFAULT_EXP_NAME  = "Smart_Credit_Scoring_Phase4"
RANDOM_SEED       = 42
DEBUG_ROW_LIMIT   = 5000

# ── Coûts métier ───────────────────────────────────────────────────────────
# FN : défaut non détecté → perte totale du capital prêté
# FP : refus abusif       → manque à gagner (une seule mensualité)
# Ratio FN/FP ≈ 5 (ajuster selon la politique de la banque)
COUT_FN = 10        # coût d'un Faux Négatif (défaut non détecté)
COUT_FP = 1        # coût d'un Faux Positif (refus injustifié)


# ##############################################################################
# CONFIGURATION MLFLOW
# ##############################################################################

class MLflowConfig:
    """Configuration centralisée MLflow — Phase 4."""

    TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI",    "http://localhost:5001")
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXP_NAME)
    ARTIFACTS_ROOT  = Path(os.getenv("MLFLOW_ARTIFACT_PATH", "./mlflow/artifacts"))

    RUN_TAGS = {
        "project"          : "Prêt à Dépenser — Home Credit",
        "team"             : "Data Science MLOps",
        "phase"            : "phase4_optimization",
        "dataset"          : "Home Credit Default Risk",
        "target"           : "target (défaut paiement ≈ 8%)",
        "pipeline_version" : "phase4_v1",
        "infrastructure"   : "Docker-Compose + PostgreSQL",
    }


# ##############################################################################
# ESPACES D'HYPERPARAMÈTRES
# ##############################################################################

def get_gridsearch_param_grid(model_name: str) -> Dict:
    """
    Génère les grilles d'hyperparamètres pour GridSearchCV.

    Dimensionnement intentionnellement raisonnable :
    LogisticRegression  → 3×3 = 9 combinaisons
    RandomForest        → 3×3×2 = 18 combinaisons
    XGBoost             → 3×3×2 = 18 combinaisons
    LightGBM            → 3×3×2 = 18 combinaisons

    Conseil : en production, élargir les grilles et utiliser Optuna.
    
    Note: Les noms des paramètres pour le MLP doivent utiliser 
    le préfixe 'mlp__' car il est encapsulé dans un Pipeline avec SMOTE.
    """
    grids = {
        # ── 1. BASELINE ──────────────────────────────────────────────────────
        "dummy_baseline": {
            "strategy":         ["stratified", "most_frequent"]
        },

        # ── 2. MODÈLES LINÉAIRES ─────────────────────────────────────────────
        "logistic_regression": {
            "C":                [0.01, 0.1, 1.0, 10.0],
            "solver":           ["lbfgs", "liblinear"],
            "max_iter":         [500, 1000]
        },

        # ── 3. ARBRES ET ENSEMBLES ───────────────────────────────────────────
        "decision_tree": {
            "max_depth":        [3, 5, 8, 12],
            "min_samples_leaf": [20, 50, 100],
            "min_samples_split":[30, 50],
        },
        "random_forest": {
            "n_estimators":     [100, 200],
            "max_depth":        [8, 12, 16],
            "max_features":     ["sqrt", "log2"]
        },

        # ── 4. BOOSTING ──────────────────────────────────────────────────────
        "gradient_boosting": {
            "n_estimators":     [100, 200],
            "learning_rate":    [0.03, 0.05, 0.10],
            "max_depth":        [3, 4, 5]
        },
        "xgboost": {
            "n_estimators":     [200, 300],
            "learning_rate":    [0.03, 0.05, 0.10],
            "max_depth":        [4, 5, 6],
            "subsample":        [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8],
            "scale_pos_weight": [11, 12]       # Ajustement du déséquilibre
        },
        "lightgbm": {
            "n_estimators":     [200, 300],
            "learning_rate":    [0.03, 0.05, 0.10],
            "num_leaves":       [31, 50, 80],
            "max_depth":        [5, 6, 8],
        },

        # ── 5. RÉSEAU DE NEURONES (AVEC PIPELINE SMOTE) ──────────────────────
        "mlp": {
            # Équilibrage : proportion de la classe minoritaire après SMOTE
            "smote__sampling_strategy": [0.25, 0.40],    
            
            # Architecture : on reste sur des structures simples (shallow)
            "mlp__hidden_layer_sizes":  [(128, 64), (64, 32)],
            
            # Régularisation : alpha élevé pour éviter l'overfitting
            "mlp__alpha":               [0.01, 0.05, 0.1],
            
            # Optimisation : taux d'apprentissage initial
            "mlp__learning_rate_init":  [0.001, 0.01]
        }
    }
    
    return grids.get(model_name, {})

def get_optuna_search_space(trial, model_name: str) -> Dict:
    """
    Espace de recherche Optuna par modèle.

    Chaque appel à trial.suggest_*() définit un hyperparamètre :
        suggest_float(name, low, high)           → float continu
        suggest_float(name, low, high, log=True) → float en log-scale (ex: C, lr)
        suggest_int(name, low, high)             → entier
        suggest_categorical(name, choices)       → catégorie

    La log-scale est préférable pour les paramètres d'échelle (C, lr)
    car elle échantillonne uniformément sur plusieurs ordres de grandeur.
    """
    if model_name == "logistic_regression":
        return {
            "C":       trial.suggest_float("C", 1e-4, 10.0, log=True),
            "solver":  trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "max_iter": trial.suggest_int("max_iter", 300, 2000),
        }

    elif model_name == "decision_tree":
        return {
            "max_depth":         trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 10, 200),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 200),
        }

    elif model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth":    trial.suggest_int("max_depth", 5, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        }

    elif model_name == "gradient_boosting":
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth", 2, 7),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        }

    elif model_name == "xgboost":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 20.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    elif model_name == "lightgbm":
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 20, 150),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":     trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        }

    elif model_name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers   = tuple(
            trial.suggest_int(f"n_units_l{i}", 32, 256)
            for i in range(n_layers)
        )
        return {
            "hidden_layer_sizes": layers,
            "alpha":              trial.suggest_float("alpha", 1e-5, 0.1, log=True),
            "learning_rate_init": trial.suggest_float("lr_init", 1e-4, 0.01, log=True),
        }

    return {}


def build_model_from_params_V0(model_name: str, params: Dict,
                             random_state: int = 42) -> Any:
    """
    Instancie le modèle sklearn correspondant avec les hyperparamètres donnés.
    Utilisé aussi bien par GridSearchCV que par Optuna.
    """
    rs = random_state

    builders = {
        "logistic_regression": lambda p: LogisticRegression(
            **p, class_weight="balanced", random_state=rs
        ),
        "decision_tree": lambda p: DecisionTreeClassifier(
            **p, class_weight="balanced", random_state=rs
        ),
        "random_forest": lambda p: RandomForestClassifier(
            **p, class_weight="balanced", random_state=rs, n_jobs=-1
        ),
        "gradient_boosting": lambda p: GradientBoostingClassifier(
            **p, random_state=rs
        ),
        "mlp": lambda p: MLPClassifier(
            **p, max_iter=300, early_stopping=True, random_state=rs
        ),
    }

    if XGBOOST_OK:
        builders["xgboost"] = lambda p: XGBClassifier(
            **p, eval_metric="auc", random_state=rs, n_jobs=-1, verbosity=0
        )

    if LGBM_OK:
        builders["lightgbm"] = lambda p: LGBMClassifier(
            **p, random_state=rs, n_jobs=-1, verbose=-1
        )

    if model_name not in builders:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return builders[model_name](params)

def build_model_from_params(
    model_name:   str, 
    params:       Dict,
    random_state: int = 42
) -> Any:
    """
    Instancie le modèle ou pipeline correspondant aux hyperparamètres.
    Garantit la cohérence entre Phase 3 (Grid) et Phase 4 (Optuna).
    """
    rs = random_state

    # 1. Définition des constructeurs (Lambdas)
    builders = {
        # Baseline (Le point zéro)
        "dummy_baseline": lambda p: DummyClassifier(**p, random_state=rs),

        # Modèles Linéaires et Arbres simples
        "logistic_regression": lambda p: LogisticRegression(
            **p, class_weight="balanced", random_state=rs
        ),
        "decision_tree": lambda p: DecisionTreeClassifier(
            **p, class_weight="balanced", random_state=rs
        ),

        # Ensembles (Bagging & Boosting Sklearn)
        "random_forest": lambda p: RandomForestClassifier(
            **p, class_weight="balanced", random_state=rs, n_jobs=-1
        ),
        "gradient_boosting": lambda p: GradientBoostingClassifier(
            **p, random_state=rs
        ),

        # Réseau de Neurones (Cas particulier : nécessite SMOTE)
        "mlp": lambda p: Pipeline(steps=[
            ("smote", SMOTE(random_state=rs)),
            ("mlp",   MLPClassifier(
                **p, max_iter=500, early_stopping=True, random_state=rs
            ))
        ]),
    }

    # 2. Ajout des frameworks haute performance (si installés)
    if XGBOOST_OK:
        builders["xgboost"] = lambda p: XGBClassifier(
            **p, eval_metric="auc", random_state=rs, n_jobs=-1, verbosity=0
        )

    if LGBM_OK:
        builders["lightgbm"] = lambda p: LGBMClassifier(
            **p, random_state=rs, n_jobs=-1, verbose=-1
        )

    # 3. Validation et Instanciation
    if model_name not in builders:
        raise ValueError(f"❌ Modèle inconnu dans le catalogue : {model_name}")

    return builders[model_name](params)



    
# ##############################################################################
# SEUIL MÉTIER
# ##############################################################################

def optimize_threshold(
    y_true:     np.ndarray,
    y_proba:    np.ndarray,
    cout_fn:    int   = COUT_FN,
    cout_fp:    int   = COUT_FP,
    n_steps:    int   = 100,
    beta:       float = 2.0,
) -> Tuple[float, pd.DataFrame]:
    """
    Détermine le seuil de décision optimal selon deux critères :
        1. Coût métier total minimisé  (FN × cout_fn + FP × cout_fp)
        2. F2-Score maximisé           (beta=2 → recall prioritaire)

    Args:
        y_true   : Labels réels (0/1)
        y_proba  : Probabilités de la classe 1 (predict_proba[:, 1])
        cout_fn  : Coût d'un Faux Négatif (défaut non détecté)
        cout_fp  : Coût d'un Faux Positif (refus abusif)
        n_steps  : Nombre de seuils à tester entre 0.05 et 0.80
        beta     : Beta pour F-beta score (2 = double poids recall)

    Returns:
        (seuil_optimal, df_courbe)
        df_courbe : DataFrame avec colonnes
            threshold, f2, f1, recall, precision, cout_metier,
            fn, fp, tn, tp
    """
    thresholds = np.linspace(0.05, 0.80, n_steps)
    rows       = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        f2        = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)

        # Coût métier total (unités relatives)
        cout_total = fn * cout_fn + fp * cout_fp

        rows.append({
            "threshold":    round(float(t), 4),
            "f2":           round(float(f2), 4),
            "f1":           round(float(f1), 4),
            "recall":       round(float(recall), 4),
            "precision":    round(float(precision), 4),
            "cout_metier":  int(cout_total),
            "fn":           int(fn),
            "fp":           int(fp),
            "tn":           int(tn),
            "tp":           int(tp),
        })

    df = pd.DataFrame(rows)

    # Le seuil optimal minimise le coût métier
    # En cas d'égalité, on préfère le recall le plus élevé
    seuil_optimal = float(
        df.loc[df["cout_metier"].idxmin(), "threshold"]
    )

    return seuil_optimal, df

def custom_metier_scorer(y_true, y_proba):
    # ✅ siempre extraer columna positiva
    if hasattr(y_proba, 'shape') and len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1]
    
    y_pred = (y_proba >= 0.5).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    score = (10 * fn) + (1 * fp)
    return -float(score)

# ##############################################################################
# PIPELINE PHASE 4
# ##############################################################################

class Phase4Pipeline:
    """
    Pipeline Phase 4 — Optimisation hyperparamètres + Seuil métier.

    Étapes :
        step0_setup()              → config MLflow expérience Phase 4
        step1_load_champion()      → charge le champion Phase 3 (joblib + metadata)
        step2_load_data()          → recharge X_train, X_eval, y_train, y_eval
        step3_gridsearch()         → GridSearchCV sur le champion (optionnel)
        step4_optuna()             → Optuna TPE sur le champion (optionnel)
        step5_compare_engines()    → compare GridSearch vs Optuna dans MLflow
        step6_optimize_threshold() → balayage seuil + courbe coût
        step7_save()               → joblib + JSON + CSV courbe
        step8_register_db()        → update model_versions

    Attributs publics après exécution :
        .best_params        : Dict — meilleurs hyperparamètres trouvés
        .best_model         : estimateur sklearn optimisé + refitté sur tout X_train
        .optimal_threshold  : float — seuil métier optimal
        .threshold_df       : pd.DataFrame — courbe coût vs seuil
        .mlflow_run_ids     : Dict — run_ids par moteur (gridsearch, optuna)
    """

    def __init__(
        self,
        champion_name:   str   = None,
        engine:          str   = "both",          # "gridsearch" | "optuna" | "both"
        n_trials:        int   = 50,              # Optuna uniquement
        cv_folds:        int   = 5,
        eval_ratio:      float = 0.20,
        source:          str   = "db",
        random_state:    int   = RANDOM_SEED,
        debug:           bool  = False,
        debug_limit:     int   = DEBUG_ROW_LIMIT,
        experiment_name: str   = None,
        verbose:         bool  = True,
        n_jobs_gs:       int   = -1,              # GridSearchCV parallelisme
        timeout_optuna:  int   = None,            # Timeout Optuna en secondes
    ):
        self.champion_name   = champion_name
        self.engine          = engine
        self.n_trials        = n_trials
        self.cv_folds        = cv_folds
        self.eval_ratio      = eval_ratio
        self.source          = source
        self.random_state    = random_state
        self.debug           = debug
        self.debug_limit     = debug_limit
        self.experiment_name = experiment_name or MLflowConfig.EXPERIMENT_NAME
        self.verbose         = verbose
        self.n_jobs_gs       = n_jobs_gs
        self.timeout_optuna  = timeout_optuna

        # Chemins projet
        self.base_dir     = Path(os.getcwd())
        self.models_dir   = self.base_dir / "models"
        self.reports_dir  = self.base_dir / "reports"
        self.processed_dir = self.base_dir / "data" / "processed"

        # Données
        self.X_train:       Optional[pd.DataFrame] = None
        self.X_eval:        Optional[pd.DataFrame] = None
        self.y_train:       Optional[pd.Series]    = None
        self.y_eval:        Optional[pd.Series]    = None
        self.feature_names: List[str]              = []
        self.target_col     = "target"

        # Champion Phase 3
        self.champion_model    = None
        self.champion_metadata = {}

        # Résultats Phase 4
        self.gs_result:         Optional[Dict] = None    # GridSearchCV
        self.optuna_result:     Optional[Dict] = None    # Optuna
        self.best_params:       Dict           = {}
        self.best_model                        = None
        self.best_engine:       str            = ""
        self.optimal_threshold: float          = 0.5
        self.threshold_df:      Optional[pd.DataFrame] = None

        # MLflow
        self.mlflow_client:  Optional[MlflowClient] = None
        self.experiment_id:  Optional[str]          = None
        self.mlflow_run_ids: Dict[str, str]         = {}


    
        # 2. Creamos el objeto scorer oficial de sklearn
        self.business_scorer = make_scorer(
            custom_metier_scorer,
            response_method="predict_proba",  # ✅ nuevo API sklearn ≥1.4
            greater_is_better=True,
        )

        # Scorer F2 pour sklearn (GridSearchCV)
        self._scorer_f2 = make_scorer(fbeta_score, beta=2, zero_division=0)

    # ── Helpers ──────────────────────────────────────────────────────────
    def _log(self, msg: str, level: str = "INFO") -> None:
        if not self.verbose:
            return
        icons = {
            "INFO":    "ℹ️ ", "SUCCESS": "✅", "WARNING": "⚠️ ",
            "ERROR":   "❌", "STEP":    "📊", "MLFLOW":  "📈",
            "OPTUNA":  "🔬", "GRID":    "🔲",
        }
        print(f"{icons.get(level, '• ')} {msg}")

    def _sep(self, char: str = "#", n: int = 80) -> None:
        if self.verbose:
            print("\n" + char * n)

    # ==========================================================================
    # STEP 0 : CONFIGURATION MLFLOW
    # ==========================================================================

    def step0_setup(self) -> None:
        """Configure MLflow et crée/récupère l'expérience Phase 4."""
        self._sep()
        self._log("Configuration MLflow Phase 4 ...", "STEP")

        mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
        self.mlflow_client = MlflowClient()

        exp = self.mlflow_client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            self.experiment_id = mlflow.create_experiment(
                name              = self.experiment_name,
                artifact_location = str(MLflowConfig.ARTIFACTS_ROOT),
            )
            self._log(f"Expérience créée : {self.experiment_name}", "SUCCESS")
        else:
            self.experiment_id = exp.experiment_id
            self._log(f"Expérience existante : {self.experiment_name}", "INFO")

    # ==========================================================================
    # STEP 1 : CHARGEMENT DU CHAMPION PHASE 3
    # ==========================================================================

    def step1_load_champion(self) -> None:
        """
        Charge le modèle champion de Phase 3 (joblib + metadata JSON).

        Le champion peut être :
            a) Spécifié via --champion en CLI
            b) Détecté automatiquement depuis models/*_metadata.json (is_best=True)
        """
        self._sep()
        self._log("STEP 1. Chargement du champion Phase 3 ...", "STEP")

        # Détection automatique du champion
        if self.champion_name is None:
            self.champion_name = self._detect_champion()

        model_path = self.models_dir / f"{self.champion_name}_model.joblib"
        meta_path  = self.models_dir / f"{self.champion_name}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle champion introuvable : {model_path}\n"
                "→ Exécutez d'abord Phase 3 :\n"
                "  uv run python -m src.pipelines.phase3_model_training_mlflow"
            )

        self.champion_model    = joblib.load(model_path)
        self.champion_metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        self._log(
            f"Champion : {self.champion_name} "
            f"(F2 Phase3={self.champion_metadata.get('eval_metrics', {}).get('f2', 'N/A')})",
            "SUCCESS",
        )
        self._log(
            f"Classe sklearn : {type(self.champion_model).__name__}",
            "INFO",
        )

    def _detect_champion(self) -> str:
        """Détecte automatiquement le champion (is_best=True dans les metadata)."""
        for meta_path in self.models_dir.glob("*_metadata.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("is_best", False):
                    name = meta["model_name"]
                    self._log(f"Champion détecté automatiquement : {name}", "INFO")
                    return name
            except Exception:
                continue
        raise FileNotFoundError(
            "Aucun champion détecté dans models/*_metadata.json\n"
            "→ Lancez Phase 3 ou spécifiez --champion <nom>"
        )

    # ==========================================================================
    # STEP 2 : RECHARGEMENT DES DONNÉES
    # ==========================================================================

    def step2_load_data(self) -> None:
        """
        Recharge les données de Phase 2 (mêmes splits que Phase 3).
        IMPORTANT : on utilise le même random_state pour reproduire
        exactement le même split train/eval que Phase 3.
        """
        self._sep()
    
        if self.source == "db" and DB_AVAILABLE:
            self._log("STEP 2. Chargement des données (From Database)...", "STEP")
            self._load_from_db()
        else:
            self._log("STEP 2. Chargement des données (From CSV)...", "STEP")
            self._load_from_csv()
        self._split_data()

    def _load_from_db(self) -> None:
        """Charge les matrices pré-traitées (X, y) depuis PostgreSQL."""
        try:
            engine = get_engine()
            limit_str = f"LIMIT {self.debug_limit}" if self.debug else ""
    
            if self.debug:
                self._log(f"⚡ DEBUG MODE (DB) : {self.debug_limit} lignes", "WARNING")
    
            self._log("Lecture des tables ml_X_train et ml_y_train ...", "INFO")
            t0 = time.time()
    
            query_x = f'SELECT * FROM "ml_X_train" {limit_str}'
            query_y = f'SELECT * FROM "ml_y_train" {limit_str}'
    
            X = pd.read_sql(query_x, engine)
            y = pd.read_sql(query_y, engine).squeeze()
            y.name = self.target_col
    
            self._df_raw = pd.concat([X, y], axis=1)
    
            self._log(
                f"Tables SQL chargées en {time.time()-t0:.1f}s — "
                f"{self._df_raw.shape[0]:,} lignes × {self._df_raw.shape[1]} colonnes",
                "SUCCESS",
            )
        except Exception as e:
            self._log(f"DB indisponible ({e}) → bascule CSV", "WARNING")
            self.source = "csv"
            self._load_from_csv()

    def _load_from_csv(self) -> None:
        """Charge X_train.csv + y_train.csv produits par Phase 2."""
        nrows = self.debug_limit if self.debug else None
    
        if self.debug:
            self._log(f"⚡ DEBUG MODE (CSV) : {self.debug_limit} lignes", "WARNING")
    
        self._log("Lecture data/processed/X_train.csv + y_train.csv ...", "INFO")
    
        x_path = self.processed_dir / "X_train.csv"
        y_path = self.processed_dir / "y_train.csv"
    
        if not x_path.exists():
            raise FileNotFoundError(
                f"X_train.csv introuvable : {x_path}\n"
                "→ Exécutez Phase 2 : uv run python -m src.pipelines.phase2_feature_engineering"
            )
    
        X = pd.read_csv(x_path, low_memory=False, nrows=nrows)
        y = pd.read_csv(y_path, nrows=nrows).squeeze()
        y.name = self.target_col
    
        self._df_raw = pd.concat([X, y], axis=1)
    
        self._log(
            f"CSV chargés ({'DEBUG' if self.debug else 'FULL'}) — {self._df_raw.shape}",
            "SUCCESS",
        )

    def _split_data(self) -> None:
        """
        Reproduit exactement le split Phase 3 (même random_state).
        Anti-leakage : X_eval n'est jamais vu pendant la CV ni l'optimisation.
        """

        cols_meta   = {"target", "split", "sk_id_curr"}
        feat_cols   = [c for c in self._df_raw.columns if c not in cols_meta]

        X = self._df_raw[feat_cols].copy()
        y = self._df_raw[self.target_col].copy()

        # Nettoyage
        X = X.select_dtypes(exclude="object").replace([np.inf, -np.inf], np.nan).fillna(0)
        self.feature_names = list(X.columns)

        self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(
            X, y,
            test_size    = self.eval_ratio,
            random_state = self.random_state,
            stratify     = y,
        )
        self._log(
            f"Split : X_train={self.X_train.shape}  X_eval={self.X_eval.shape}  "
            f"défauts={self.y_eval.mean():.1%}",
            "SUCCESS",
        )

    # ==========================================================================
    # STEP 3 : GRIDSEARCHCV
    # ==========================================================================

    def step3_gridsearch(self) -> Dict:
        """
        GridSearchCV exhaustif sur la grille définie pour le champion.

        Fonctionnement :
            1. Récupère la grille d'hyperparamètres du champion
            2. Crée un modèle base avec les params fixes (class_weight, etc.)
            3. GridSearchCV avec StratifiedKFold 5-plis + scorer F2
            4. Extrait les meilleurs paramètres + score CV
            5. Refitte sur tout X_train avec les meilleurs params
            6. Évalue sur X_eval (holdout)
            7. Logue dans MLflow
        """
        self._sep()
        self._log("STEP 3. GRIDSEARCHCV — Optimisation exhaustive ...", "GRID")

        param_grid = get_gridsearch_param_grid(self.champion_name)
        if not param_grid:
            self._log(
                f"Grille vide pour '{self.champion_name}' — GridSearch ignoré",
                "WARNING",
            )
            return {}

        # Nombre de combinaisons
        n_combinations = 1
        for vals in param_grid.values():
            n_combinations *= len(vals)
        self._log(
            f"Grille : {n_combinations} combinaisons × {self.cv_folds} plis "
            f"= {n_combinations * self.cv_folds} fits",
            "INFO",
        )

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        # Modèle de base (params fixes hors grille)
        base_model = build_model_from_params(
            self.champion_name, {}, self.random_state
        )

        gs = GridSearchCV(
            estimator  = base_model,
            param_grid = param_grid,
            scoring    = self.business_scorer,  # self._scorer_f2
            cv         = cv,
            refit      = True,
            n_jobs     = self.n_jobs_gs,
            verbose    = 0,
        )

        t_start = time.time()
        gs.fit(self.X_train, self.y_train)
        t_elapsed = time.time() - t_start

        # Évaluation sur eval set (holdout)
        best_model   = gs.best_estimator_
        y_proba_eval = best_model.predict_proba(self.X_eval)[:, 1]
        y_pred_eval  = (y_proba_eval >= 0.5).astype(int)

        cm = confusion_matrix(self.y_eval, y_pred_eval, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Coût métier (même logique que ton scorer)
        business_cost    = (10 * fn) + (1 * fp)
        #business_cost_n  = -float(business_cost)  # version "score" (plus grand = mieux)
        
        eval_metrics = {
            "f2":            fbeta_score(self.y_eval, y_pred_eval, beta=2, zero_division=0),
            "f1":            f1_score(self.y_eval, y_pred_eval, zero_division=0),
            "recall":        recall_score(self.y_eval, y_pred_eval, zero_division=0),
            "precision":     precision_score(self.y_eval, y_pred_eval, zero_division=0),
            "roc_auc":       roc_auc_score(self.y_eval, y_proba_eval),
            # ✅ Métriques métier
            "business_cost": business_cost,        # ex: 4230 (lisible)
            "fn":            int(fn),              # Faux Négatifs — les plus dangereux
            "fp":            int(fp),              # Faux Positifs
            "tp":            int(tp),
            "tn":            int(tn),
        }
        result = {
            "engine":          "gridsearch",
            "best_params":     gs.best_params_,
            "best_cv_f2":      gs.best_score_,
            "eval_metrics":    eval_metrics,
            "train_time_s":    t_elapsed,
            "n_combinations":  n_combinations,
            "cv_results_df":   pd.DataFrame(gs.cv_results_),
            "model":           best_model,
            "y_proba_eval":    y_proba_eval,
        }
        self.gs_result = result

        self._log(
            f"GridSearch terminé en {t_elapsed:.1f}s — "
            f"CV score={gs.best_score_:.1f}  Eval F2={eval_metrics['f2']:.4f}  "
            f"Coût métier={eval_metrics['business_cost']}  "   # ✅
            f"FN={eval_metrics['fn']}  FP={eval_metrics['fp']}",
            "SUCCESS",
        )
        self._log(f"Meilleurs params : {gs.best_params_}", "INFO")

        # Sauvegarde cv_results_ pour le notebook de visualisation
        # (heatmaps F2 par paire de paramètres)
        cv_path = self.reports_dir / f"phase4_gs_cv_results_{self.champion_name}.csv"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(gs.cv_results_).to_csv(cv_path, index=False)
        self._log(f"cv_results GridSearch → {cv_path.name}", "INFO")

        # Log MLflow
        self._log_to_mlflow("gridsearch", result)
        return result

    # ==========================================================================
    # STEP 4 : OPTUNA
    # ==========================================================================

    def step4_optuna(self) -> Dict:
        """
        Optimisation bayésienne avec Optuna (TPE + Median Pruning).

        Comment fonctionne Optuna dans ce contexte ?
        ─────────────────────────────────────────────
        1. Création d'une Study (maximize F2 sur CV)
        2. Pour chaque Trial :
            a. Optuna propose des hyperparamètres via TPE Sampler
            b. On crée le modèle avec ces paramètres
            c. On évalue par StratifiedKFold 5-plis (CV F2 mean)
            d. Optuna met à jour son modèle interne
        3. Après n_trials, on récupère le meilleur trial
        4. Refit sur tout X_train avec les meilleurs params
        5. Évaluation sur X_eval (holdout)
        6. Logging MLflow complet

        Pruning (MedianPruner) :
            Si un trial est clairement mauvais après les 3 premiers plis,
            Optuna l'interrompt et passe au suivant → économie de temps.

        Note : MLFLOW_TRACKING_URI doit être défini DANS chaque trial
        car Optuna peut paralléliser sur plusieurs processus.
        """
        self._sep()
        self._log("STEP 4. OPTUNA — Optimisation bayésienne (TPE) ...", "OPTUNA")

        if not OPTUNA_OK:
            self._log(
                "Optuna non installé — installez avec : uv add optuna optuna-integration",
                "WARNING",
            )
            return {}

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        # ── Fonction objectif ──────────────────────────────────────────────
        def objective(trial) -> float:
            params = get_optuna_search_space(trial, self.champion_name)
            if not params:
                raise optuna.TrialPruned()
        
            try:
                model = build_model_from_params(
                    self.champion_name, params, self.random_state
                )
        
                scores = cross_val_score(
                    model,
                    self.X_train, self.y_train,
                    cv      = cv,
                    scoring = self.business_scorer,
                    n_jobs  = 1,
                )
                
                # ✅ DEBUG — añade esto temporalmente
                # print(f"[DEBUG] params={params}")
                # print(f"[DEBUG] scores={scores}")
                # print(f"[DEBUG] mean={scores.mean()}")
                
                mean = float(np.nanmean(scores))  # ← nanmean en lugar de mean
                # print(f"[DEBUG] nanmean={mean}")
                
                if np.isnan(mean):
                    # print("[DEBUG] ⚠️ nan detectado → retornando 0.0")
                    return 0.0  # ← red de seguridad absoluta
                    
                return mean
        
            except Exception as e:
                self._log(f"  Trial échoué : {e}", "WARNING")
                raise optuna.TrialPruned()

        
        # ✅ TEST DIRECTO DEL SCORER — antes del study

        # print("[SCORER TEST] Test directo del scorer...")
        # test_model = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
        # test_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # usa el ya importado

        #test_scores = cross_val_score(
        #    test_model,
        #    self.X_train, self.y_train,
        #    cv=test_cv,
        #    scoring=self.business_scorer,
        #    n_jobs=1
        #)
        # print(f"[SCORER TEST] scores = {test_scores}")
        # print(f"[SCORER TEST] scorer = {self.business_scorer}")
        

        # y_dummy_true = self.y_train[:1000]
        # test_model.fit(self.X_train[:5000], self.y_train[:5000])

        # y_dummy_proba = test_model.predict_proba(self.X_train[:1000])
        # print(f"[SCORER TEST] custom_metier_scorer direct = {custom_metier_scorer(y_dummy_true, y_dummy_proba)}")
        
        # ── Création et lancement de la Study ─────────────────────────────
        #
        # TPESampler  : algorithme bayésien (défaut Optuna, très efficace)
        # MedianPruner: interrompt les trials sous la médiane après 3 évals
        study = optuna.create_study(
            direction   = "maximize",
            study_name  = f"phase4_{self.champion_name}",
            sampler     = TPESampler(seed=self.random_state),
            pruner      = MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )

        t_start = time.time()

        study.optimize(
            objective,
            n_trials    = self.n_trials,
            timeout     = self.timeout_optuna,    # None = pas de limite temps
            show_progress_bar = self.verbose,
        )

        t_elapsed = time.time() - t_start

        # ── Résultats du meilleur trial ────────────────────────────────────
        best_trial  = study.best_trial
        best_params = best_trial.params

        # Ajustement : hidden_layer_sizes doit être un tuple (Optuna stocke des ints)
        if self.champion_name == "mlp" and "n_layers" in best_params:
            n_layers   = best_params.pop("n_layers")
            layer_size = {k: best_params.pop(k) for k in
                          [f"n_units_l{i}" for i in range(n_layers)]}
            best_params["hidden_layer_sizes"] = tuple(layer_size.values())

        # Refit sur tout X_train avec les meilleurs hyperparamètres
        best_model = build_model_from_params(
            self.champion_name, best_params, self.random_state
        )
        best_model.fit(self.X_train, self.y_train)

        # Évaluation sur holdout
        y_proba_eval = best_model.predict_proba(self.X_eval)[:, 1]
        y_pred_eval  = (y_proba_eval >= 0.5).astype(int)

        # ✅ Matriz siempre 2×2
        cm = confusion_matrix(self.y_eval, y_pred_eval, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        business_cost = (10 * fn) + (1 * fp)
        
        eval_metrics = {
            "f2":            fbeta_score(self.y_eval, y_pred_eval, beta=2, zero_division=0),
            "f1":            f1_score(self.y_eval, y_pred_eval, zero_division=0),
            "recall":        recall_score(self.y_eval, y_pred_eval, zero_division=0),
            "precision":     precision_score(self.y_eval, y_pred_eval, zero_division=0),
            "roc_auc":       roc_auc_score(self.y_eval, y_proba_eval),
            # ✅ Métier — identique step3
            "business_cost": business_cost,
            "fn":            int(fn),
            "fp":            int(fp),
            "tp":            int(tp),
            "tn":            int(tn),
        }

        # Statistiques de la study Optuna
        n_complete = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned   = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.PRUNED])

        result = {
            "engine":          "optuna",
            "best_params":     best_params,
            "best_cv_f2":      best_trial.value,
            "eval_metrics":    eval_metrics,
            "train_time_s":    t_elapsed,
            "n_trials":        self.n_trials,
            "n_complete":      n_complete,
            "n_pruned":        n_pruned,
            "study":           study,
            "model":           best_model,
            "y_proba_eval":    y_proba_eval,
        }
        self.optuna_result = result

        self._log(
            f"Optuna terminé en {t_elapsed:.1f}s — "
            f"{n_complete} trials complets, {n_pruned} prunés  "
            f"Best score={best_trial.value:.1f}  Eval F2={eval_metrics['f2']:.4f}  "
            f"Coût métier={eval_metrics['business_cost']}  "
            f"FN={eval_metrics['fn']}  FP={eval_metrics['fp']}",
            "SUCCESS",
        )
        self._log(
            f"Meilleur trial : CV F2={best_trial.value:.4f}  "
            f"Eval F2={eval_metrics['f2']:.4f}",
            "SUCCESS",
        )
        self._log(f"Meilleurs params : {best_params}", "INFO")

        # Export du study pour les visualisations notebook
        study_path = self.reports_dir / f"phase4_optuna_study_{self.champion_name}.pkl"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(study, study_path)
        self._log(f"Study Optuna sauvegardé → {study_path.name}", "INFO")

        # Log MLflow
        self._log_to_mlflow("optuna", result)
        return result

    # ==========================================================================
    # STEP 5 : COMPARAISON GRIDSEARCH vs OPTUNA
    # ==========================================================================

    def step5_compare_engines(self) -> None:
        """
        Sélectionne le meilleur moteur (GridSearch vs Optuna) selon F2 eval.
        Met à jour self.best_model, self.best_params, self.best_engine.
        """
        self._sep()
        self._log("STEP 5. Comparaison GridSearch vs Optuna ...", "STEP")

        candidates = {}
        if self.gs_result:
            candidates["gridsearch"] = self.gs_result
        if self.optuna_result:
            candidates["optuna"] = self.optuna_result

        if not candidates:
            self._log("Aucun résultat d'optimisation disponible.", "ERROR")
            return

        # Sélection sur F2 eval (métrique métier prioritaire)
        best_engine = max(
            candidates,
            key=lambda k: candidates[k]["eval_metrics"].get("f2", 0),
        )

        winner = candidates[best_engine]
        self.best_engine  = best_engine
        self.best_params  = winner["best_params"]
        self.best_model   = winner["model"]

        # Tableau comparatif
        rows = []
        for engine_name, r in candidates.items():
            em = r["eval_metrics"]
            cv_score = r["best_cv_f2"]
            
            # ✅ CV score legible — coste métier o F2 según el scorer
            cv_display = f"{abs(cv_score):,.0f}" if cv_score < -1 else f"{cv_score:.4f}"
            
            rows.append({
                "Moteur":      engine_name,
                "CV coût":     cv_display,          # ✅ renombrado y legible
                "Eval F2":     round(em["f2"], 4),
                "Eval AUC":    round(em["roc_auc"], 4),
                "Recall":      round(em["recall"], 4),
                "Precision":   round(em["precision"], 4),
                "Coût métier": em.get("business_cost", "-"),   # ✅ coste real holdout
                "FN":          em.get("fn", "-"),
                "FP":          em.get("fp", "-"),
                "Temps (s)":   round(r["train_time_s"], 1),
                "Gagnant":     "🏆" if engine_name == best_engine else "",
            })
        
        df = pd.DataFrame(rows).sort_values("Eval F2", ascending=False)
        print("\n" + "=" * 92)
        print("COMPARAISON MOTEURS D'OPTIMISATION — Phase 4")
        print("=" * 92)
        print(df.to_string(index=False))
        print("=" * 92)
        self._log(
            f"🏆 Gagnant : {best_engine}  "
            f"(F2 eval={winner['eval_metrics']['f2']:.4f}  "
            f"Coût={winner['eval_metrics'].get('business_cost', '?')}  "
            f"FN={winner['eval_metrics'].get('fn', '?')}  "
            f"FP={winner['eval_metrics'].get('fp', '?')})",
            "SUCCESS",
        )

        # Tag MLflow du gagnant
        if best_engine in self.mlflow_run_ids:
            self.mlflow_client.set_tag(
                self.mlflow_run_ids[best_engine], "phase4_winner", "true"
            )

    # ==========================================================================
    # STEP 5b : REFIT FINAL SUR 100% DES DONNÉES (train + eval)
    # ==========================================================================

    def step6b_final_refit(self) -> None:
        """
        Ré-entraîne le modèle gagnant sur la totalité des données disponibles
        (X_train + X_eval), c'est-à-dire 100% du split Phase 3.

        Pourquoi ce step est nécessaire ?
        ──────────────────────────────────
        Durant step3/step4, le modèle a été évalué sur X_eval (holdout 20%).
        Ce holdout était indispensable pour mesurer les performances réelles
        et trouver le seuil optimal.

        Mais une fois les hyperparamètres fixés et le seuil déterminé,
        il serait dommage de laisser 20% des données inutilisées en production.
        Le refit sur 100% donne au modèle plus d'exemples → meilleure généralisation.

        Workflow exact :
            1. clone() → repart de zéro, même architecture
            2. set_params(**best_params) → injecte les meilleurs hyperparamètres
            3. fit(X_train + X_eval) → entraîne sur 100% des données labelisées

        Attention : après ce refit, on ne peut plus évaluer sur X_eval
        (les données sont dans le modèle). C'est pour ça que le seuil optimal
        doit être calculé en step6 AVANT ce refit, ou être conservé de step6.

        Note sur l'ordre des steps :
            step6 (seuil) est calculé sur self.best_model AVANT le refit.
            step5b peut donc être appelé APRÈS step6 sans problème.
        """


        self._sep()
        self._log("STEP 5b. Refit final sur 100% des données (train + eval) ...", "STEP")

        if self.best_model is None:
            self._log("Aucun modèle gagnant disponible.", "ERROR")
            return

        # 1. Concaténer train + eval → 100% des données labelisées
        X_full = pd.concat([self.X_train, self.X_eval], axis=0).reset_index(drop=True)
        y_full = pd.concat([self.y_train, self.y_eval], axis=0).reset_index(drop=True)

        self._log(
            f"Données fusionnées : {X_full.shape[0]:,} lignes "
            f"(train={len(self.X_train):,} + eval={len(self.X_eval):,})  "
            f"défauts={y_full.mean():.2%}",
            "INFO",
        )

        # 2. Cloner l'architecture (repart de zéro, pas d'état résiduel)
        refitted = sklearn_clone(self.best_model)

        # 3. Réinjecter les meilleurs hyperparamètres trouvés
        try:
            refitted.set_params(**self.best_params)
        except Exception:
            # set_params échoue pour les pipelines imblearn → on garde le clone tel quel
            self._log("set_params ignoré (pipeline complexe) — clone utilisé tel quel", "WARNING")

        # 4. LE REFIT : entraînement sur 100% des données
        t_start = time.time()
        refitted.fit(X_full, y_full)
        t_refit = time.time() - t_start

        self.best_model = refitted

        self._log(
            f"Refit terminé en {t_refit:.1f}s sur {X_full.shape[0]:,} lignes ✓",
            "SUCCESS",
        )
        self._log(
            "Le modèle champion est prêt pour la Phase 5 (scoring Kaggle + API).",
            "SUCCESS",
        )

        # Log MLflow : mise à jour du run gagnant avec le flag refit
        if self.best_engine in self.mlflow_run_ids:
            try:
                self.mlflow_client.set_tag(
                    self.mlflow_run_ids[self.best_engine],
                    "refit_on_full_data", "true"
                )
                self.mlflow_client.log_metric(
                    self.mlflow_run_ids[self.best_engine],
                    "refit_train_size", X_full.shape[0]
                )
            except Exception:
                pass

    # ==========================================================================
    # STEP 6 : OPTIMISATION DU SEUIL MÉTIER
    # ==========================================================================

    def step6_optimize_threshold(self) -> None:
        """
        Détermine le seuil de décision optimal selon le coût métier.

        Logique :
            Seuil 0.5 = imposteur pour données déséquilibrées.
            On balaie [0.05, 0.80] et on minimise :
                Coût = FN × COUT_FN + FP × COUT_FP
            avec COUT_FN=5 (défaut non détecté) et COUT_FP=1 (refus abusif).
        """
        self._sep()
        self._log(
            f"STEP 6. Optimisation du seuil métier (coût FN={COUT_FN}× > FP={COUT_FP}×) ...",
            "STEP",
        )

        if self.best_model is None:
            self._log("Aucun modèle optimisé disponible.", "ERROR")
            return

        # Probabilités du modèle gagnant sur le holdout
        y_proba_eval = self.best_model.predict_proba(self.X_eval)[:, 1]

        self.optimal_threshold, self.threshold_df = optimize_threshold(
            y_true  = self.y_eval.values,
            y_proba = y_proba_eval,
            cout_fn = COUT_FN,
            cout_fp = COUT_FP,
            n_steps = 150,
            beta    = 2.0,
        )

        # Métriques au seuil optimal
        y_pred_optimal = (y_proba_eval >= self.optimal_threshold).astype(int)
        f2_optimal     = fbeta_score(self.y_eval, y_pred_optimal, beta=2, zero_division=0)
        recall_optimal = recall_score(self.y_eval, y_pred_optimal, zero_division=0)
        cout_optimal   = self.threshold_df.loc[
            self.threshold_df["threshold"].sub(self.optimal_threshold).abs().idxmin(),
            "cout_metier"
        ]
        cout_default   = self.threshold_df.loc[
            self.threshold_df["threshold"].sub(0.5).abs().idxmin(),
            "cout_metier"
        ]

        self._log(f"Seuil optimal   : {self.optimal_threshold:.3f}", "SUCCESS")
        self._log(f"F2 au seuil opt.: {f2_optimal:.4f}", "INFO")
        self._log(f"Recall au seuil.: {recall_optimal:.4f}", "INFO")
        self._log(
            f"Coût métier : {cout_default} → {cout_optimal} "
            f"(économie {cout_default - cout_optimal} unités)",
            "SUCCESS",
        )

        # Sauvegarde courbe pour notebook
        curve_path = self.reports_dir / "phase4_threshold_curve.csv"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_df.to_csv(curve_path, index=False)
        self._log(f"Courbe seuil → {curve_path.name}", "INFO")

        # ── Log MLflow : seuil optimal + toutes les métriques au seuil ──────
        #
        # POURQUOI ici et pas dans _log_to_mlflow ?
        # ─────────────────────────────────────────
        # _log_to_mlflow() est appelé dans step3/step4, quand le seuil
        # n'est pas encore calculé (il vaut encore 0.5 par défaut).
        # C'est seulement ici, en step6, qu'on connaît optimal_threshold.
        # On utilise mlflow_client.log_metric() sur le run DÉJÀ FERMÉ
        # (post-run logging), ce qui est supporté par l'API MLflow.
        # ─────────────────────────────────────────
        if self.best_engine in self.mlflow_run_ids:
            run_id = self.mlflow_run_ids[self.best_engine]
            try:
                # Métriques au seuil optimal (toutes)
                y_pred_at_opt   = (y_proba_eval >= self.optimal_threshold).astype(int)
                precision_optimal = precision_score(self.y_eval, y_pred_at_opt, zero_division=0)
                auc_optimal       = roc_auc_score(self.y_eval, y_proba_eval)

                self.mlflow_client.log_metric(run_id, "optimal_threshold",         self.optimal_threshold)
                self.mlflow_client.log_metric(run_id, "threshold_f2",              f2_optimal)
                self.mlflow_client.log_metric(run_id, "threshold_recall",          recall_optimal)
                self.mlflow_client.log_metric(run_id, "threshold_precision",       precision_optimal)
                self.mlflow_client.log_metric(run_id, "threshold_roc_auc",         auc_optimal)
                self.mlflow_client.log_metric(run_id, "cout_metier_at_threshold",  cout_optimal)
                self.mlflow_client.log_metric(run_id, "cout_metier_at_default_05", cout_default)
                self.mlflow_client.log_metric(run_id, "cout_saved_vs_default",     cout_default - cout_optimal)

                # Tag visible dans MLflow UI pour identifier ce run comme final
                self.mlflow_client.set_tag(run_id, "threshold_optimized", "true")
                self.mlflow_client.set_tag(run_id, "optimal_threshold_value", f"{self.optimal_threshold:.4f}")

                # Courbe seuil comme artefact du run
                self.mlflow_client.log_artifact(run_id, str(curve_path))

                self._log(
                    f"MLflow run {run_id[:8]} mis à jour : "
                    f"optimal_threshold={self.optimal_threshold:.3f}  "
                    f"threshold_f2={f2_optimal:.4f}  "
                    f"threshold_recall={recall_optimal:.4f}",
                    "MLFLOW",
                )
            except Exception as e:
                self._log(f"Log MLflow seuil ignoré : {e}", "WARNING")

    # ==========================================================================
    # STEP 7 : SAUVEGARDE
    # ==========================================================================

    def step7_save(self) -> None:
        """
        Sauvegarde le modèle optimisé et ses métadonnées complètes.

        Produit :
            models/phase4_best_model.joblib
            models/phase4_best_model_metadata.json
        """
        self._sep()
        self._log("STEP 7. Sauvegarde du modèle optimisé ...", "STEP")

        if self.best_model is None:
            self._log("Aucun modèle à sauvegarder.", "ERROR")
            return

        self.models_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.models_dir / "phase4_best_model.joblib"
        joblib.dump(self.best_model, model_path)

        # Métriques au seuil optimal
        y_proba = self.best_model.predict_proba(self.X_eval)[:, 1]
        y_pred  = (y_proba >= self.optimal_threshold).astype(int)

        eval_at_threshold = {
            "f2":        float(fbeta_score(self.y_eval, y_pred, beta=2, zero_division=0)),
            "f1":        float(f1_score(self.y_eval, y_pred, zero_division=0)),
            "recall":    float(recall_score(self.y_eval, y_pred, zero_division=0)),
            "precision": float(precision_score(self.y_eval, y_pred, zero_division=0)),
            "roc_auc":   float(roc_auc_score(self.y_eval, y_proba)),
        }

        metadata = {
            # Identité
            "model_name":          "phase4_best_model",
            "champion_phase3":     self.champion_name,
            "model_class":         type(self.best_model).__name__,
            "optimization_engine": self.best_engine,

            # Hyperparamètres
            "best_params":         {k: str(v) for k, v in self.best_params.items()},

            # Métriques Phase 3 (baseline)
            "phase3_eval_f2":      self.champion_metadata.get(
                "eval_metrics", {}
            ).get("f2", None),

            # Métriques Phase 4 (seuil 0.5)
            "phase4_eval_metrics": {
                k: round(v, 4)
                for k, v in (
                    self.gs_result or self.optuna_result or {}
                ).get("eval_metrics", {}).items()
            },

            # Métriques Phase 4 (seuil optimal)
            "phase4_eval_at_threshold": eval_at_threshold,

            # Seuil métier
            "optimal_threshold":   self.optimal_threshold,
            "cout_fn":             COUT_FN,
            "cout_fp":             COUT_FP,

            # Données
            "feature_names":       self.feature_names,
            "n_features":          len(self.feature_names),
            "train_samples":       len(self.y_train),
            "eval_samples":        len(self.y_eval),
            "target_rate":         float(self.y_train.mean()),

            # MLflow
            "mlflow_run_ids":      self.mlflow_run_ids,
            "experiment_name":     self.experiment_name,

            # Traçabilité
            "model_path":          str(model_path.absolute()),
            "saved_at":            datetime.datetime.now().isoformat(),
        }

        meta_path = self.models_dir / "phase4_best_model_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        self.eval_metrics = eval_at_threshold

        self._log(f"Modèle → {model_path.name}", "SUCCESS")
        self._log(f"Metadata → {meta_path.name}", "INFO")

        # ── Re-log du modèle FINAL dans MLflow (avec seuil comme param) ────
        # Le modèle logué en step3/step4 était évalué à seuil 0.5.
        # Ici on logue le modèle définitif (refitté sur 100% + seuil connu)
        # sous l'artifact path "model_final" dans le run gagnant.
        if self.best_engine in self.mlflow_run_ids:
            run_id = self.mlflow_run_ids[self.best_engine]
            try:
                self.mlflow_client.log_param(run_id, "optimal_threshold_final", self.optimal_threshold)
                # Logguer le .joblib et le .json comme artefacts du run gagnant
                self.mlflow_client.log_artifact(run_id, str(model_path))
                self.mlflow_client.log_artifact(run_id, str(meta_path))
                self._log(
                    f"Artefacts finaux (joblib + JSON) loggués sur run MLflow {run_id[:8]}",
                    "MLFLOW",
                )
            except Exception as e:
                self._log(f"Log MLflow artefact ignoré : {e}", "WARNING")

    # ==========================================================================
    # STEP 8 : REGISTRE POSTGRESQL
    # ==========================================================================

    def step8_register_db(self) -> None:
        """
        Met à jour model_versions avec le champion Phase 4.

        Note : on INSERT une nouvelle ligne plutôt qu'UPDATE
        pour conserver l'historique complet des versions.
        """
        if not DB_AVAILABLE or self.best_model is None:
            self._log("Enregistrement DB ignoré (DB non dispo ou aucun modèle).", "WARNING")
            return

        self._sep()
        self._log("STEP 8. Enregistrement dans PostgreSQL model_versions ...", "STEP")

        try:
            engine  = get_engine()
            version = f"v_phase4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Utiliser les métriques calculées en step7 (avant le refit sur 100%)
            # car X_eval est maintenant dans le jeu d'entraînement du modèle refitté.
            # self.eval_metrics contient les métriques au seuil optimal sur X_eval.
            m = getattr(self, "eval_metrics", {})

            metrics_dict = {
                "f2":                float(m.get("f2",      0)),
                "f1":                float(m.get("f1",      0)),
                "recall":            float(m.get("recall",  0)),
                "roc_auc":           float(m.get("roc_auc", 0)),
                "optimal_threshold": self.optimal_threshold,
                "optimization_engine": self.best_engine,
                "refit_on_full_data": True,
            }

            info = {
                "name":       f"phase4_{self.champion_name}",
                "version":    version,
                "run_id":     self.mlflow_run_ids.get(self.best_engine, ""),
                "algo":       type(self.best_model).__name__,
                "params":     json.dumps({k: str(v) for k, v in self.best_params.items()}),
                "metrics":    json.dumps(metrics_dict),
                "threshold":  self.optimal_threshold,
                "model_path": str((self.models_dir / "phase4_best_model.joblib").absolute()),
                "meta_path":  str((self.models_dir / "phase4_best_model_metadata.json").absolute()),
                "status":     "optimized",
            }

            sql = text("""
                INSERT INTO model_versions
                    (model_name, version, mlflow_run_id, algorithm, 
                     hyperparameters, metrics, optimal_threshold, 
                     model_path, metadata_path, status)
                VALUES
                    (:name, :version, :run_id, :algo, 
                     CAST(:params AS JSONB), CAST(:metrics AS JSONB), :threshold, 
                     :model_path, :meta_path, :status)
                RETURNING id  -- 🟢 Añadimos esto
            """)

            with engine.begin() as conn:
                result = conn.execute(sql, info)
                # 🟢 Guardamos el ID para que el Step 9 sepa a qué modelo asociar los benchmarks
                self.model_version_id = result.scalar()

            self._log(f"Enregistré en DB — Version : {version}", "SUCCESS")

        except Exception as exc:
            self._log(f"Erreur DB : {exc}", "ERROR")
            traceback.print_exc()


    # --------------------------------------------------------------------------
    # STEP 9 : ENREGISTREMENT DES BENCHMARKS DE MONITORING
    # --------------------------------------------------------------------------
    def step9_register_monitoring(self):
        """
        Insère les métriques de référence dans 'monitoring_metrics'.
        """
        # 1. Usamos get_engine() directamente (ya importado al inicio del script)
        db_engine = get_engine()
        
        if not DB_AVAILABLE or db_engine is None:
            print("⚠️ Monitoring DB non disponible - Saut de l'étape 9")
            return

        self._sep()
        self._log("STEP 9. Enregistrement des benchmarks de monitoring...", "STEP")

        # 2. Recuperar métricas del objeto self
        metrics = getattr(self, 'eval_metrics', {})
        
        benchmarks = {
            "ref_f2_score":  metrics.get("f2"),
            "ref_auc_roc":   metrics.get("roc_auc"),
            "ref_recall":    metrics.get("recall"),
            "ref_precision": metrics.get("precision"),
            "ref_threshold": self.optimal_threshold
        }

        try:
            # 3. Usamos el objeto db_engine
            with db_engine.begin() as conn:
                for name, value in benchmarks.items():
                    if value is None: continue
                    
                    query = text("""
                        INSERT INTO monitoring_metrics 
                        (metric_name, metric_value, metric_type, model_version_id, metadata)
                        VALUES (:name, :val, 'performance_benchmark', :mod_id, :meta)
                    """)
                    
                    conn.execute(query, {
                        "name":   name,
                        "val":    float(value),
                        "mod_id": getattr(self, 'model_version_id', None),
                        "meta":   json.dumps({
                            "phase": "4_optimization",
                            "engine": self.best_engine
                        })
                    })
            print(f"✅ {len(benchmarks)} métricas de referencia guardadas con éxito.")
            
        except Exception as e:
            print(f"❌ Erreur enregistrement monitoring : {e}")
            
    # ==========================================================================
    # LOG MLFLOW (INTERNE)
    # ==========================================================================

    def _log_to_mlflow(self, engine_name: str, result: Dict) -> None:
        """Log complet d'un run d'optimisation dans MLflow."""
        if mlflow.active_run():
            mlflow.end_run()

        em = result["eval_metrics"]

        with mlflow.start_run(
            experiment_id = self.experiment_id,
            run_name      = f"{self.champion_name}_{engine_name}_opt",
        ) as run:

            self._log(
                f"  MLflow run {run.info.run_id[:8]}... [{engine_name}]", "MLFLOW"
            )

            # Tags
            mlflow.set_tags({
                **MLflowConfig.RUN_TAGS,
                "phase":              "phase4_optimization",
                "optimization_engine": engine_name,
                "champion_phase3":    self.champion_name,
                "model_class":        type(result["model"]).__name__,
            })

            # Params
            safe_params = {f"opt_{k}": str(v)[:250] for k, v in result["best_params"].items()}
            safe_params.update({
                "engine":           engine_name,
                "champion_name":    self.champion_name,
                "cv_folds":         self.cv_folds,
                "random_state":     self.random_state,
                "n_features":       len(self.feature_names),
                "train_samples":    len(self.y_train),
                "eval_samples":     len(self.y_eval),
                "target_rate":      round(float(self.y_train.mean()), 4),
                "cout_fn":          COUT_FN,
                "cout_fp":          COUT_FP,
            })
            if engine_name == "gridsearch":
                safe_params["n_combinations"] = result.get("n_combinations", 0)
            elif engine_name == "optuna":
                safe_params["n_trials"]    = result.get("n_trials",   self.n_trials)
                safe_params["n_pruned"]    = result.get("n_pruned",   0)
                safe_params["n_complete"]  = result.get("n_complete", 0)
            mlflow.log_params(safe_params)

            # Métriques
            mlflow.log_metric("cv_f2_best",  result["best_cv_f2"])
            mlflow.log_metric("train_time_s", result["train_time_s"])
            for k, v in em.items():
                if isinstance(v, float) and not np.isnan(v):
                    mlflow.log_metric(f"eval_{k}", v)

            # Baseline Phase 3 (pour comparaison dans MLflow UI)
            phase3_f2 = self.champion_metadata.get("eval_metrics", {}).get("f2")
            if phase3_f2 is not None:
                mlflow.log_metric("phase3_baseline_f2", float(phase3_f2))
                mlflow.log_metric("delta_f2_vs_phase3", em["f2"] - float(phase3_f2))

            # Modèle sklearn avec signature
            try:
                sig = infer_signature(
                    self.X_train.head(5),
                    result["model"].predict(self.X_train.head(5)),
                )
                mlflow.sklearn.log_model(result["model"], "model", signature=sig)
            except Exception:
                mlflow.sklearn.log_model(result["model"], "model")


            # Vincular el dataset  
            dataset_train = mlflow.data.from_pandas(self.X_train, name="train_set")
            mlflow.log_input(dataset_train, context="training")
            
            dataset_eval = mlflow.data.from_pandas(self.X_eval, name="eval_set")
            mlflow.log_input(dataset_eval, context="validation")

            self.mlflow_run_ids[engine_name] = run.info.run_id


# ##############################################################################
# FONCTION PRINCIPALE
# ##############################################################################

def run_phase4(
    champion_name:   str   = None,
    engine:          str   = "both",
    n_trials:        int   = 50,
    cv_folds:        int   = 5,
    source:          str   = "db",
    eval_ratio:      float = 0.20,
    random_state:    int   = RANDOM_SEED,
    debug:           bool  = False,
    debug_limit:     int   = DEBUG_ROW_LIMIT,
    experiment_name: str   = None,
    verbose:         bool  = True,
    timeout_optuna:  int   = None,
) -> Phase4Pipeline:
    """
    Exécute la Phase 4 complète.

    Args:
        champion_name   : Nom du champion Phase 3 (détecté automatiquement si None)
        engine          : "gridsearch" | "optuna" | "both"
        n_trials        : Nombre d'essais Optuna (ignoré pour gridsearch)
        cv_folds        : Nombre de plis StratifiedKFold
        source          : "db" ou "csv"
        eval_ratio      : Fraction holdout (doit correspondre à Phase 3)
        random_state    : Graine (doit correspondre à Phase 3)
        debug           : Charger seulement debug_limit lignes
        debug_limit     : Nb lignes en mode debug
        experiment_name : Nom expérience MLflow
        verbose         : Logs détaillés
        timeout_optuna  : Timeout Optuna en secondes (None = pas de limite)

    Returns:
        Instance Phase4Pipeline avec .best_model, .optimal_threshold, etc.
    """

    # self._sep()
    print("\n" + "=" * 76)
    print("🚀 PHASE 4 — OPTIMISATION HYPERPARAMÈTRES + SEUIL MÉTIER")
    print("   Projet : Prêt à Dépenser — Home Credit Default Risk")
    print("=" * 76)

    t_start  = time.time()
    pipeline = Phase4Pipeline(
        champion_name   = champion_name,
        engine          = engine,
        n_trials        = n_trials,
        cv_folds        = cv_folds,
        source          = source,
        eval_ratio      = eval_ratio,
        random_state    = random_state,
        debug           = debug,
        debug_limit     = debug_limit,
        experiment_name = experiment_name,
        verbose         = verbose,
        timeout_optuna  = timeout_optuna,
    )

    try:
        pipeline.step0_setup()
        pipeline.step1_load_champion()
        pipeline.step2_load_data()

        if engine in ("gridsearch", "both"):
            pipeline.step3_gridsearch()

        if engine in ("optuna", "both"):
            if not OPTUNA_OK and engine == "optuna":
                raise ImportError(
                    "Optuna non installé. Installez avec : uv add optuna optuna-integration"
                )
            pipeline.step4_optuna()

        pipeline.step5_compare_engines()
        pipeline.step6_optimize_threshold()   # ← seuil calculé sur 80% (X_eval)
        pipeline.step6b_final_refit()         # ← refit sur 100% (train+eval) avec best_params
        pipeline.step7_save()                 # ← sauvegarde le modèle refitté + seuil
        pipeline.step8_register_db()
        pipeline.step9_register_monitoring()
        
        t_elapsed = time.time() - t_start
        duration  = str(datetime.timedelta(seconds=round(t_elapsed)))

        print("\n" + "=" * 76)
        print("✅ PHASE 4 TERMINÉE AVEC SUCCÈS")
        print("=" * 76)
        print(f"  Durée totale...........: {duration} (H:M:S)")
        print(f"  Champion Phase 3.......: {pipeline.champion_name}")
        print(f"  Moteur gagnant.........: {pipeline.best_engine}")
        print(f"  Meilleurs params.......: {pipeline.best_params}")
        print(f"  F2 Phase 3 (baseline)..: {pipeline.champion_metadata.get('eval_metrics', {}).get('f2', 'N/A')}")
        print(f"  F2 Phase 4 (optimisé)..: {(pipeline.gs_result or 
                                             pipeline.optuna_result or {}).get('eval_metrics', {}).get('f2', 'N/A')}")
        print(f"  Seuil optimal..........: {pipeline.optimal_threshold:.3f}  (défaut : 0.500)")
        print(f"  MLflow UI..............: {MLflowConfig.TRACKING_URI}")
        print(f"  Expérience.............: {pipeline.experiment_name}")
        print()
        print("  ⮕  Suite : Phase 5 — TODO ... ")
        print("=" * 76)

        return pipeline

    except Exception as exc:
        print(f"\n❌ ERREUR Phase 4 : {exc}")
        traceback.print_exc()
        raise


# ##############################################################################
# CLI
# ##############################################################################

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4 — Optimisation hyperparamètres + Seuil métier (Home Credit)"
    )
    p.add_argument(
        "--champion", default=None,
        help="Nom du champion Phase 3 (ex: logistic_regression). Détecté auto si absent.",
    )
    p.add_argument(
        "--engine",
        choices=["gridsearch", "optuna", "both"],
        default="both",
        help="Moteur d'optimisation : gridsearch | optuna | both (défaut: both)",
    )
    p.add_argument(
        "--n-trials", type=int, default=50,
        help="Nombre d'essais Optuna (défaut: 50). Ignoré pour gridsearch.",
    )
    p.add_argument(
        "--cv-folds", type=int, default=5,
        help="Nombre de plis StratifiedKFold (défaut: 5)",
    )
    p.add_argument(
        "--source", choices=["db", "csv"], default="db",
        help="Source données : 'db' (PostgreSQL) ou 'csv' (data/processed/)",
    )
    p.add_argument(
        "--eval-ratio", type=float, default=0.20,
        help="Fraction holdout eval — doit correspondre à Phase 3 (défaut: 0.20)",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Mode debug : charge seulement --debug-limit lignes",
    )
    p.add_argument(
        "--debug-limit", type=int, default=DEBUG_ROW_LIMIT,
        help=f"Nb lignes en mode debug (défaut: {DEBUG_ROW_LIMIT})",
    )
    p.add_argument(
        "--experiment", default=None,
        help=f"Nom expérience MLflow (défaut: {MLflowConfig.EXPERIMENT_NAME})",
    )
    p.add_argument(
        "--timeout-optuna", type=int, default=None,
        help="Timeout Optuna en secondes (None = pas de limite). Ex: --timeout-optuna 600",
    )
    p.add_argument(
        "--no-verbose", action="store_true",
        help="Désactiver les logs détaillés",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_phase4(
        champion_name   = args.champion,
        engine          = args.engine,
        n_trials        = args.n_trials,
        cv_folds        = args.cv_folds,
        source          = args.source,
        eval_ratio      = args.eval_ratio,
        debug           = args.debug,
        debug_limit     = args.debug_limit,
        experiment_name = args.experiment,
        verbose         = not args.no_verbose,
        timeout_optuna  = args.timeout_optuna,
    )
