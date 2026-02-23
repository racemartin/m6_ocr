"""
src/pipelines/phase3_model_training_mlflow.py
==============================================
Phase 3 — Entraînement des modèles de classification + MLflow tracking
Projet : Prêt à Dépenser — Home Credit Default Risk

Objectif métier
───────────────
Prédire le défaut de remboursement (target=1 ≈ 8%).
Déséquilibre ~1:11 → les Faux Négatifs coûtent bien plus que les Faux Positifs.
Métrique de décision : F2-Score (double pénalité FN).
Métrique de présentation : AUC-ROC.

Modèles entraînés (complexité croissante)
──────────────────────────────────────────
    1. DummyClassifier        → baseline plancher (aléatoire stratifié)
    2. LogisticRegression     → baseline linéaire, coefficients lisibles
    3. DecisionTreeClassifier → arbre simple, diagnostic overfitting
    4. RandomForestClassifier → bagging, robuste aux outliers
    5. GradientBoosting       → boosting sklearn, pas de class_weight natif
    6. XGBClassifier          → XGBoost, scale_pos_weight pour le déséquilibre
    7. LGBMClassifier         → LightGBM, rapide sur 300k lignes
    8. MLPClassifier          → réseau shallow non-linéaire

Validation anti-leakage
────────────────────────
    • Données chargées depuis PostgreSQL (v_features_engineering WHERE split='train')
    • Split interne stratifié 80% train / 20% eval (holdout final)
    • StratifiedKFold(5 plis) sur train uniquement (ClassificationModeler)
    • Les paramètres du preprocessor de Phase 2 ne sont PAS recalculés ici

Artefacts produits
──────────────────
    models/<nom>_model.joblib      → modèle sklearn sérialisé
    models/<nom>_metadata.json     → métriques, params, feature names
    mlruns/                        → tracking MLflow (UI sur :5000)
    PostgreSQL model_versions      → registre du champion

Usage
─────
    # Terminal 1 (prérequis)
    # mlflow ui --port 5000        # No, on utilise docker  
    docker compose up -d mlflow    # Service expose dans le port 5001

    # Terminal 2
    uv run python -m src.pipelines.phase3_model_training_mlflow
    uv run python -m src.pipelines.phase3_model_training_mlflow --source csv
    uv run python -m src.pipelines.phase3_model_training_mlflow --debug
    uv run python -m src.pipelines.phase3_model_training_mlflow --save-all
"""

from __future__ import annotations

# ── Standard ───────────────────────────────────────────────────────────────
import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# ── Infrastructure ─────────────────────────────────────────────────────────
import joblib
from sqlalchemy import text

# ── Stack scientifique ─────────────────────────────────────────────────────
import numpy  as np
import pandas as pd

# ── MLflow ─────────────────────────────────────────────────────────────────
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

# ── Scikit-learn ───────────────────────────────────────────────────────────
from sklearn.dummy          import DummyClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, fbeta_score,
    precision_score, recall_score, roc_auc_score,
)

# ── Boosting avancé ────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("⚠️  XGBoost non installé → uv add xgboost")

try:
    from lightgbm import LGBMClassifier
    LGBM_OK = True
except ImportError:
    LGBM_OK = False
    print("⚠️  LightGBM non installé → uv add lightgbm")

# ── ClassificationModeler ─────────────────────────────────────────────────
# Cherche d'abord dans src/models/ (emplacement recommandé),
# puis dans notebooks/ (emplacement legacy du projet original).
MODELER_OK = False
try:
    from src.models.ClassificationModeler import ClassificationModeler
    MODELER_OK = True
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from notebooks.ClassificationModeler import ClassificationModeler
        MODELER_OK = True
    except ImportError:
        print("❌ ClassificationModeler introuvable.")
        print("   → cp ClassificationModeler.py src/models/")

# ── Base de données ────────────────────────────────────────────────────────
DB_AVAILABLE = False
try:
    from src.database import get_engine
    DB_AVAILABLE = True
except ImportError:
    print("⚠️  src.database introuvable — mode CSV forcé")


# ----------------------------------------------------------------------------
# Bibliothèques spécifiques pour le déséquilibre
# ----------------------------------------------------------------------------
from   imblearn.over_sampling import SMOTE            # Génération synthétique
from   imblearn.pipeline      import Pipeline         # Pipeline compatible SMOTE



# ----------------------------------------------------------------------------
DEBUG_ROW_LIMIT  = 10000                          # Limite de lignes en debug
DEFAULT_EXP_NAME = "Smart_Credit_Scoring"         # Nom de l'expérience MLflow
RANDOM_SEED      = 42                             # Graine de reproductibilité

# ##############################################################################
# CONFIGURATION MLFLOW
# ##############################################################################

class MLflowConfig:
    """Configuration centralisée MLflow — Prêt à Dépenser / Home Credit."""

    # ─────────────────────────────────────────────────────────────────────
    # 1. PARAMÈTRES DE CONNEXION (DYNAMIQUE)
    # ─────────────────────────────────────────────────────────────────────
    # Leemos del .env o usamos valores por defecto seguros
    TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Smart_Credit_Scoring")
    
    # IMPORTANTE: ARTIFACTS_ROOT debe coincidir con donde MLflow guarda los archivos
    # Si usas el volumen de Docker, esta ruta es más una referencia local.
    ARTIFACTS_ROOT  = Path(os.getenv("MLFLOW_ARTIFACT_PATH", "./mlflow/artifacts"))

    # ─────────────────────────────────────────────────────────────────────
    # 2. TAGS DE L'EXPÉRIENCE
    # ─────────────────────────────────────────────────────────────────────
    RUN_TAGS = {
        "project"          : "Prêt à Dépenser — Home Credit",
        "team"             : "Data Science MLOps",
        "model_type"       : "Classification Binaire",
        "dataset"          : "Home Credit Default Risk",
        "target"           : "target (défaut paiement ≈ 8%)",
        "pipeline_version" : "phase3_v1",
        "infrastructure"   : "Docker-Compose + PostgreSQL",
    }


# ##############################################################################
# CATALOGUE DES MODÈLES
# ##############################################################################

# ############################################################################
# CONFIGURATION MLP AVEC SMOTE (VERSION DIDACTIQUE)
# ############################################################################

def get_mlp_with_smote_config(random_state: int = 42) -> Dict:
    """
    Configure un pipeline MLP incluant l'équilibrage SMOTE.
    
    Le SMOTE ne sera appliqué QUE sur les données d'entraînement 
    pendant la validation croisée, évitant ainsi tout biais d'évaluation.
    """
    
    # Création du pipeline spécifique imblearn
    # 1. SMOTE équilibre les classes
    # 2. MLPClassifier apprend sur les données équilibrées
    mlp_pipeline = Pipeline([
        ('smote', SMOTE(random_state=random_state, sampling_strategy=0.3)),
        ('mlp',   MLPClassifier(
            hidden_layer_sizes = (64, 32),
            alpha              = 0.01,
            max_iter           = 300,
            early_stopping     = True,
            random_state       = random_state
        ))
    ])

    return {
        "model"      : mlp_pipeline,
        "params"     : {
            "smote__sampling_strategy": 0.3,      # Ratio de synthèse
            "mlp__alpha"              : 0.01
        },
        "description": "MLP avec SMOTE — Pipeline robuste au déséquilibre"
    }
    

# ############################################################################
# CONFIGURATION DU CATALOGUE DE MODÈLES POUR LA PHASE 3
# ############################################################################

def _build_models_config(random_state: int = 42) -> Dict:
    """
    Retourne le catalogue optimisé des modèles pour le scoring crédit.
    
    Stratégie de déséquilibre (Ratio ≈ 1:11) :
    - Utilisation de class_weight='balanced' pour compenser les 8% de défauts.
    - Paramètres régularisés pour éviter le surapprentissage (overfitting).
    """
    cat = {}                                      # Initialisation catalogue

    # ------------------------------------------------------------------------
    # 1. BASELINE ALÉATOIRE : Point de comparaison minimal
    # ------------------------------------------------------------------------
    cat["dummy_baseline"] = {
        "model"      : DummyClassifier(
            strategy     = "stratified",          # Respecte la distribution
            random_state = random_state
        ),
        "params"     : {"strategy": "stratified"},
        "description": "Baseline plancher — aléatoire stratifié",
    }

    # ------------------------------------------------------------------------
    # 2. RÉGRESSION LOGISTIQUE : Modèle linéaire interprétable
    # ------------------------------------------------------------------------
    cat["logistic_regression"] = {
        "model"      : LogisticRegression(
            C            = 0.1,                   # Régularisation forte
            max_iter     = 1000,                  # Convergence assurée
            solver       = "lbfgs",               # Solver standard robuste
            class_weight = "balanced",            # Gestion du déséquilibre
            random_state = random_state           # Reproductibilité
        ),
        "params"     : {
            "C"           : 0.1, 
            "class_weight": "balanced"
        },
        "description": "Linéaire — stable et interprétable",
    }

    # ------------------------------------------------------------------------
    # 3. ARBRE DE DÉCISION : Analyse des interactions simples
    # ------------------------------------------------------------------------
    cat["decision_tree"] = {
        "model"      : DecisionTreeClassifier(
            max_depth         = 5,                # Limite pour généraliser
            min_samples_leaf  = 50,               # Évite les feuilles isolées
            class_weight      = "balanced",       # Poids des classes
            random_state      = random_state
        ),
        "params"     : {
            "max_depth"   : 5, 
            "class_weight": "balanced"
        },
        "description": "Arbre simple — diagnostic de structure",
    }

    # ------------------------------------------------------------------------
    # 4. RANDOM FOREST : Bagging robuste
    # ------------------------------------------------------------------------
    cat["random_forest"] = {
        "model"      : RandomForestClassifier(
            n_estimators      = 150,              # Compromis vitesse/précision
            max_depth         = 10,               # Profondeur contrôlée
            class_weight      = "balanced_subsample", # Optimal pour RF
            random_state      = random_state,
            n_jobs            = -1                # Utilisation multi-cœurs
        ),
        "params"     : {
            "n_estimators": 150, 
            "max_depth"   : 10
        },
        "description": "Bagging — réduction de la variance",
    }

    # ------------------------------------------------------------------------
    # 5. GRADIENT BOOSTING SKLEARN : Boosting séquentiel
    # ------------------------------------------------------------------------
    cat["gradient_boosting"] = {
        "model"      : GradientBoostingClassifier(
            n_estimators      = 100,              # Nombre d'itérations
            learning_rate     = 0.05,             # Pas d'apprentissage lent
            max_depth         = 4,                # Arbres peu profonds
            random_state      = random_state
        ),
        "params"     : {
            "n_estimators" : 100, 
            "learning_rate": 0.05
        },
        "description": "Boosting standard — robuste mais sans poids natif",
    }

    # ------------------------------------------------------------------------
    # 6. XGBOOST : Performance optimisée
    # ------------------------------------------------------------------------
    if XGBOOST_OK:
        cat["xgboost"] = {
            "model"      : XGBClassifier(
                n_estimators      = 200,          # Nombre d'arbres
                scale_pos_weight  = 11.5,         # Ratio neg/pos pour F2
                max_depth         = 4,            # Contrôle de complexité
                learning_rate     = 0.05,         # Apprentissage graduel
                random_state      = random_state,
                n_jobs            = -1
            ),
            "params"     : {
                "n_estimators"    : 200, 
                "scale_pos_weight": 11.5
            },
            "description": "XGBoost — calibré pour le déséquilibre",
        }

    # ------------------------------------------------------------------------
    # 7. LIGHTGBM : Efficacité sur gros volumes
    # ------------------------------------------------------------------------
    if LGBM_OK:
        cat["lightgbm"] = {
            "model"      : LGBMClassifier(
                n_estimators      = 200,          # Rapide et performant
                class_weight      = "balanced",   # Force le Recall > 0
                max_depth         = 5,            # Évite la mémorisation
                learning_rate     = 0.05,         # Stabilité
                random_state      = random_state,
                n_jobs            = -1,
                verbose           = -1
            ),
            "params"     : {
                "n_estimators": 200, 
                "class_weight": "balanced"
            },
            "description": "LightGBM — haute performance, Recall prioritaire",
        }

    # ------------------------------------------------------------------------
    # 8. MLP (NEURAL NETWORK) : Relations non-linéaires complexes
    # ------------------------------------------------------------------------
    """
    cat["mlp"] = {
        "model"      : MLPClassifier(
            hidden_layer_sizes = (64, 32),        # Architecture simplifiée
            alpha              = 0.01,            # Régularisation L2 accrue
            max_iter           = 300,             # Itérations pour convergence
            early_stopping     = True,            # Évite le surapprentissage
            random_state       = random_state
        ),
        "params"     : {
            "layers": "(64,32)", 
            "alpha" : 0.01
        },
        "description": "MLP — perceptron multicouche régularisé",
    }
    """
    # Appel de la fonction modulaire pour garder le catalogue lisible
    cat["mlp"] = get_mlp_with_smote_config(random_state = random_state)

    return cat



def _build_models_config_V0(random_state: int = 42) -> Dict:
    """
    Retourne le catalogue ordonné des modèles à entraîner.

    Stratégie déséquilibre (target ≈ 8%, ratio ~1:11) :
    ──────────────────────────────────────────────────────
    • class_weight='balanced' sur sklearn (LR, DT, RF, LightGBM, MLP)
    • scale_pos_weight=11 sur XGBoost (n_négatifs / n_positifs)
    • GradientBoosting sklearn n'a pas class_weight → géré par les data weights

    Hyperparamètres : conservateurs pour la Phase 3.
    Optimisation fine → Phase 4 (Optuna / GridSearchCV).
    """
    cat = {}

    # ── 1. Baseline aléatoire ──────────────────────────────────────────────
    cat["dummy_baseline"] = {
        "model": DummyClassifier(strategy="stratified", random_state=random_state),
        "params": {"strategy": "stratified"},
        "description": "Baseline plancher — aléatoire stratifié",
    }

    # ── 2. Régression logistique ───────────────────────────────────────────
    cat["logistic_regression"] = {
        "model": LogisticRegression(
            C=0.1, max_iter=1000, solver="lbfgs",
            class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
        "params": {"C": 0.1, "solver": "lbfgs", "class_weight": "balanced",
                   "max_iter": 1000},
        "description": "Baseline linéaire — coefficients interprétables",
    }

    # ── 3. Arbre de décision ───────────────────────────────────────────────
    cat["decision_tree"] = {
        "model": DecisionTreeClassifier(
            max_depth=8, min_samples_split=50, min_samples_leaf=20,
            class_weight="balanced", random_state=random_state,
        ),
        "params": {"max_depth": 8, "min_samples_split": 50,
                   "class_weight": "balanced"},
        "description": "Arbre simple — diagnostic de surapprentissage",
    }

    # ── 4. Forêt aléatoire ─────────────────────────────────────────────────
    cat["random_forest"] = {
        "model": RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_split=30, min_samples_leaf=10,
            max_features="sqrt", class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
        "params": {"n_estimators": 200, "max_depth": 12,
                   "max_features": "sqrt", "class_weight": "balanced"},
        "description": "Bagging — robuste, bon premier modèle non-linéaire",
    }

    # ── 5. Gradient Boosting sklearn ───────────────────────────────────────
    cat["gradient_boosting"] = {
        "model": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            min_samples_split=30, random_state=random_state,
        ),
        "params": {"n_estimators": 150, "learning_rate": 0.05,
                   "max_depth": 4, "subsample": 0.8},
        "description": "Boosting sklearn — lent, fiable, sans class_weight natif",
    }

    # ── 6. XGBoost ─────────────────────────────────────────────────────────
    if XGBOOST_OK:
        cat["xgboost"] = {
            "model": XGBClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=5, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=11,           # ratio n_neg / n_pos ≈ 11
                eval_metric="auc",
                random_state=random_state, n_jobs=-1, verbosity=0,
            ),
            "params": {"n_estimators": 300, "learning_rate": 0.05,
                       "max_depth": 5, "scale_pos_weight": 11},
            "description": "XGBoost — souvent champion sur données tabulaires",
        }
    else:
        print("  [skip] XGBoost non disponible")

    # ── 7. LightGBM ────────────────────────────────────────────────────────
    if LGBM_OK:
        cat["lightgbm"] = {
            "model": LGBMClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=6, num_leaves=50,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced",
                random_state=random_state, n_jobs=-1, verbose=-1,
            ),
            "params": {"n_estimators": 300, "learning_rate": 0.05,
                       "num_leaves": 50, "class_weight": "balanced"},
            "description": "LightGBM — rapide et efficace sur 300k+ lignes",
        }
    else:
        print("  [skip] LightGBM non disponible")

    # ── 8. MLP réseau shallow ──────────────────────────────────────────────
    cat["mlp"] = {
        "model": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            solver="adam", alpha=0.001, batch_size=256,
            learning_rate_init=0.001, max_iter=200,
            early_stopping=True, validation_fraction=0.1,
            random_state=random_state,
        ),
        "params": {"hidden_layer_sizes": "(128,64,32)", "alpha": 0.001,
                   "max_iter": 200, "early_stopping": True},
        "description": "MLP shallow — non-linéaire, sans class_weight natif",
    }

    return cat


# ##############################################################################
# PIPELINE PRINCIPAL
# ##############################################################################

class Phase3Pipeline:
    """
    Pipeline Phase 3 — Entraînement + MLflow + Sauvegarde.

    Étapes :
        step0_setup_mlflow()      → configure tracking + expérience
        step1_load_data()         → PostgreSQL v_features_engineering
        step2_split()             → stratifié 80% train / 20% eval
        step3_init_modeler()      → ClassificationModeler
        step4_train_all()         → boucle catalogue + MLflow logging
        step5_compare()           → tableau comparatif + champion
        step6_save()              → joblib + metadata JSON
        step7_register_db()       → PostgreSQL model_versions

    Attributs publics après exécution :
        .results           : dict {model_name: resultats_classificationmodeler}
        .best_model_name   : str — champion (meilleur F2-Score eval)
        .mlflow_runs       : dict {model_name: run_id}
        .saved_model_paths : dict {model_name: chemin_joblib}
    """

    def __init__(
        self,
        source:          str   = "db",
        eval_ratio:      float = 0.20,
        random_state:    int   = 42,
        debug:           bool  = False,
        debug_limit:     int   = 500,  # 2000
        experiment_name: str   = None,
        verbose:         bool  = True,
    ):
        """
        Args:
            source          : "db" → PostgreSQL, "csv" → data/processed/
            eval_ratio      : Fraction du train pour le holdout eval final
            random_state    : Reproductibilité (défaut 42)
            debug           : Charger seulement debug_limit lignes
            debug_limit     : Nb lignes en mode debug (défaut {DEBUG_ROW_LIMIT})
            experiment_name : Nom expérience MLflow
            verbose         : Afficher les logs détaillés
        """
        self.source          = source
        self.eval_ratio      = eval_ratio
        self.random_state    = random_state
        self.debug           = debug
        self.debug_limit     = debug_limit
        self.verbose         = verbose

        # Chemins
        self.base_dir      = Path(os.getcwd())
        self.processed_dir = self.base_dir / "data" / "processed"
        self.models_dir    = self.base_dir / "models"

        # Données
        self.df_raw:        Optional[pd.DataFrame] = None
        self.X_train:       Optional[pd.DataFrame] = None
        self.X_eval:        Optional[pd.DataFrame] = None
        self.y_train:       Optional[pd.Series]    = None
        self.y_eval:        Optional[pd.Series]    = None
        self.feature_names: List[str]              = []

        # La colonne TARGET dans v_features_engineering s'appelle "target"
        # (définie dans schema.py : _A("TARGET", ..., "target", ...))
        self.target_col = "target"

        # MLflow
        self.experiment_name = experiment_name or MLflowConfig.EXPERIMENT_NAME
        self.mlflow_client:  Optional[MlflowClient] = None
        self.experiment_id:  Optional[str]          = None
        self.mlflow_runs:    Dict[str, str]          = {}

        # Résultats
        self.modeler:              Optional[object]   = None
        self.results:              Dict               = {}
        self.best_model_name:      Optional[str]      = None
        self.saved_model_paths:    Dict[str, str]     = {}
        self.saved_metadata_paths: Dict[str, str]     = {}

    # ── Helpers ────────────────────────────────────────────────────────────
    def _log(self, msg: str, level: str = "INFO") -> None:
        if not self.verbose:
            return
        icons = {
            "INFO": "ℹ️ ", "SUCCESS": "✅", "WARNING": "⚠️ ",
            "ERROR": "❌", "STEP": "📊", "MLFLOW": "📈",
        }
        print(f"{icons.get(level, '• ')} {msg}")

    def _sep(self, char: str = "=", n: int = 76) -> None:
        if self.verbose:
            print("\n" + char * n)

    # ==========================================================================
    # STEP 0 : SETUP MLFLOW
    # ==========================================================================

    def step0_setup_mlflow(self) -> None:
        """
        Configure le tracking MLflow et crée / restaure l'expérience.

        Utilise MLflowConfig.ARTIFACTS_ROOT pour le chemin des artefacts.
        """
        self._sep()
        self._log("Configuration MLflow ...", "STEP")

        mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
        mlflow.sklearn.autolog(disable=True)   # Logging 100% manuel
        self.mlflow_client = MlflowClient()

        try:
            exp = self.mlflow_client.get_experiment_by_name(self.experiment_name)

            if exp:
                if exp.lifecycle_stage == "deleted":
                    self._log("Restauration expérience supprimée ...", "WARNING")
                    self.mlflow_client.restore_experiment(exp.experiment_id)
                self.experiment_id = exp.experiment_id
                self._log(f"Expérience existante (ID={self.experiment_id})", "INFO")
            else:
                self.experiment_id = self.mlflow_client.create_experiment(
                    name=self.experiment_name,
                    artifact_location=str(MLflowConfig.ARTIFACTS_ROOT),
                )
                self._log(f"Expérience créée (ID={self.experiment_id})", "SUCCESS")

            mlflow.set_experiment(self.experiment_name)
            self._log(f"MLflow prêt → {MLflowConfig.TRACKING_URI}", "SUCCESS")

        except Exception as exc:
            self._log(f"Erreur MLflow : {exc}", "ERROR")
            self._log("Vérifiez que 'mlflow ui --port 5000' est actif.", "WARNING")
            raise

    # ==========================================================================
    # STEP 1 : CHARGEMENT DEPUIS POSTGRESQL
    # ==========================================================================

    def step1_load_data(self) -> None:
        """
        Charge le dataset depuis PostgreSQL (v_features_engineering) ou CSV.

        Vue v_features_engineering :
        ─────────────────────────────
            • Features engineerées Phase 1.2 (app + bureau_agg + prev_agg +
              pos_agg + cc_agg + install_agg + fe_*)
            • Colonne 'target'     : 0/1 (NULL pour split='test' Kaggle)
            • Colonne 'split'      : 'train' | 'test'
            • Colonne 'sk_id_curr' : identifiant client

        On charge UNIQUEMENT split='train'.
        Le jeu test Kaggle (sans TARGET) sera utilisé en Phase 5 (scoring).
        """
        self._sep()
        self._log(f"Chargement données (source={self.source}) ...", "STEP")

        if self.source == "db":
            self._load_from_db()
        else:
            self._load_from_csv()

        self._validate_data()

    def _load_from_db(self) -> None:
        """Charge depuis PostgreSQL via v_features_engineering."""
        if not DB_AVAILABLE:
            self._log("DB non disponible — bascule sur CSV", "WARNING")
            self.source = "csv"
            self._load_from_csv()
            return

        engine = get_engine()

        if self.debug:
            self._log(f"⚡ DEBUG MODE : {self.debug_limit} lignes", "WARNING")
            query = f"""
                SELECT *
                FROM   v_features_engineering
                WHERE  split = 'train'
                  AND  sk_id_curr IN (
                       SELECT sk_id_curr
                       FROM   raw_application_train
                       LIMIT  {self.debug_limit}
                  )
            """
        else:
            query = "SELECT * FROM v_features_engineering WHERE split = 'train'"

        self._log("v_features_engineering WHERE split='train' ...", "INFO")
        t0 = time.time()
        self.df_raw = pd.read_sql(query, engine)
        self._log(
            f"Chargé en {time.time()-t0:.1f}s — "
            f"{self.df_raw.shape[0]:,} lignes × {self.df_raw.shape[1]} colonnes",
            "SUCCESS",
        )

    def _load_from_csv(self) -> None:
        """Fallback : charge depuis les exports de Phase 2 (data/processed/)."""
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. Configuration du mode (Full vs Debug)
        # ─────────────────────────────────────────────────────────────────────
        nrows = self.debug_limit if self.debug else None
        
        if self.debug:
            self._log(f"⚡ DEBUG MODE (CSV) : {self.debug_limit} lignes", "WARNING")
        
        self._log("Lecture data/processed/X_train.csv + y_train.csv ...", "INFO")

        x_path = self.processed_dir / "X_train.csv"
        y_path = self.processed_dir / "y_train.csv"

        if not x_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {x_path}\n"
                "Exécutez d'abord :\n"
                "  uv run python -m src.pipelines.phase2_feature_engineering"
            )

        # ─────────────────────────────────────────────────────────────────────
        # 2. Chargement optimisé
        # ─────────────────────────────────────────────────────────────────────
        # nrows=None carga todo el archivo, nrows=int limita la lectura
        X = pd.read_csv(x_path, low_memory=False, nrows=nrows)
        y = pd.read_csv(y_path, nrows=nrows).squeeze()
        
        y.name = self.target_col
        
        # Concatenación para mantener el formato df_raw esperado por el resto de la clase
        self.df_raw = pd.concat([X, y], axis=1)
        
        self._log(
            f"CSV chargé ({'DEBUG' if self.debug else 'FULL'}) — {self.df_raw.shape}", 
            "SUCCESS"
        )

    def _validate_data(self) -> None:
        """Vérifie la présence de la TARGET et affiche la distribution des classes."""
        if self.target_col not in self.df_raw.columns:
            similar = [c for c in self.df_raw.columns if "target" in c.lower()]
            raise ValueError(
                f"Colonne '{self.target_col}' absente du DataFrame.\n"
                f"Colonnes similaires : {similar}"
            )

        # Supprimer les lignes sans TARGET (test Kaggle glissé dans la requête)
        n_avant = len(self.df_raw)
        self.df_raw = self.df_raw[self.df_raw[self.target_col].notna()].copy()
        n_sup = n_avant - len(self.df_raw)
        if n_sup > 0:
            self._log(f"{n_sup} lignes sans TARGET ignorées", "INFO")

        # Forcer type int (PostgreSQL peut retourner float pour des 0/1)
        self.df_raw[self.target_col] = (
            pd.to_numeric(self.df_raw[self.target_col], errors="coerce").astype(int)
        )

        n0   = (self.df_raw[self.target_col] == 0).sum()
        n1   = (self.df_raw[self.target_col] == 1).sum()
        taux = n1 / (n0 + n1)

        self._log(f"Remboursés (0) : {n0:>7,}  ({1-taux:.1%})", "INFO")
        self._log(f"Défauts    (1) : {n1:>7,}  ({taux:.1%})", "INFO")
        self._log(
            f"Déséquilibre   : 1 défaut pour ~{n0//n1} remboursements → "
            "class_weight='balanced' / scale_pos_weight=11",
            "WARNING",
        )

    # ==========================================================================
    # STEP 2 : SPLIT STRATIFIÉ TRAIN / EVAL
    # ==========================================================================

    def step2_split(self) -> None:
        """
        Découpe le dataset train en train (80%) + eval holdout (20%) stratifiés.

        Pourquoi un split interne ?
        ───────────────────────────
        La Phase 2 a séparé train / test Kaggle.
        On crée ici un holdout eval pour :
          • Comparer les modèles sur un jeu non vu pendant la CV 5-plis
          • Fournir une estimation non biaisée avant la Phase 4
        """
        self._sep()
        self._log(
            f"Split stratifié — train {1-self.eval_ratio:.0%} / eval {self.eval_ratio:.0%} ...",
            "STEP",
        )

        cols_meta   = {"target", "split", "sk_id_curr"}
        feature_cols = [c for c in self.df_raw.columns if c not in cols_meta]

        X = self.df_raw[feature_cols].copy()
        y = self.df_raw[self.target_col].copy()

        # Éliminer les colonnes string résiduelles (garde-fou)
        cols_str = X.select_dtypes(include="object").columns.tolist()
        if cols_str:
            self._log(f"Colonnes string ignorées : {cols_str[:5]}", "WARNING")
            X = X.drop(columns=cols_str)

        # Remplacer infinis et NaN résiduels par 0 (Phase 2 les a normalement éliminés)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(
            X, y,
            test_size=self.eval_ratio,
            random_state=self.random_state,
            stratify=y,
        )
        self.feature_names = list(self.X_train.columns)

        self._log(
            f"X_train : {self.X_train.shape}  défauts={self.y_train.mean():.1%}",
            "SUCCESS",
        )
        self._log(
            f"X_eval  : {self.X_eval.shape}  défauts={self.y_eval.mean():.1%}",
            "SUCCESS",
        )
        self._log(f"Features totales : {len(self.feature_names)}", "INFO")

    # ==========================================================================
    # STEP 3 : INITIALISATION CLASSIFICATIONMODELER
    # ==========================================================================

    def step3_init_modeler(self) -> None:
        """
        Instancie le ClassificationModeler avec les données.

        Important : le paramètre X_test / y_test du ClassificationModeler
        correspond à notre eval set (holdout), PAS au test Kaggle.
        La CV 5-plis s'effectue sur X_train / y_train uniquement.
        """
        self._sep()
        self._log("Initialisation ClassificationModeler ...", "STEP")

        if not MODELER_OK:
            raise ImportError(
                "ClassificationModeler introuvable.\n"
                "Solution :\n"
                "  mkdir -p src/models\n"
                "  cp ClassificationModeler.py src/models/\n"
                "  touch src/models/__init__.py"
            )

        self.modeler = ClassificationModeler(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_eval,     # "test" du ClassificationModeler = holdout eval
            y_test=self.y_eval,
            config={
                "RANDOM_STATE": self.random_state,
                "CV_FOLDS":     5,
                "THRESHOLD":    0.5,
            },
        )
        self._log("ClassificationModeler prêt (StratifiedKFold 5-plis) ✓", "SUCCESS")

    # ==========================================================================
    # STEP 4 : ENTRAÎNEMENT BOUCLE CATALOGUE + MLFLOW
    # ==========================================================================

    def step4_train_all(self) -> None:
        """
        Boucle sur le catalogue des modèles et entraîne chacun.

        Pour chaque modèle :
          1. ClassificationModeler.entrainer_modele()
             → StratifiedKFold 5-plis sur X_train
             → fit complet sur X_train
             → métriques train / eval / CV
          2. mlflow.start_run()
             → tags, params, métriques, modèle, artefacts
        """
        self._sep()
        self._log("Entraînement de tous les modèles ...", "STEP")

        if self.modeler is None:
            self.step3_init_modeler()

        catalogue = _build_models_config(self.random_state)

        for i, (nom, cfg) in enumerate(catalogue.items(), 1):
            self._sep("-", 60)
            self._log(f"[{i}/{len(catalogue)}]  {nom.upper()}", "STEP")
            self._log(f"  {cfg['description']}", "INFO")
            try:
                self._train_one_model_with_mlflow(cfg["model"], nom, cfg["params"])
            except Exception as exc:
                self._log(f"Erreur avec '{nom}' : {exc}", "ERROR")
                traceback.print_exc()

    # ############################################################################
    # MÉTHODE : _train_one_model_with_mlflow (Architecture Robuste Phase 3)
    # ############################################################################
    
    def _train_one_model_with_mlflow(
        self,
        model,
        model_name:   str,
        model_params: Dict,
    ) -> None:
        """
        Orchestre l'entraînement et le logging MLflow complet d'un modèle.
        
        Cette méthode assure la cohérence entre le moteur d'entraînement 
        (ClassificationModeler) et le serveur de tracking (MLflow).
        """
    
        # Nettoyage préventif : Fermer tout run actif pour éviter les collisions
        if mlflow.active_run():
            mlflow.end_run()
    
        # ------------------------------------------------------------------------
        # 1. ENTRAÎNEMENT (Moteur de calcul externe)
        # ------------------------------------------------------------------------
        
        try:
            # Tentative d'entraînement : CV 5-plis + Fit final
            results = self.modeler.entrainer_modele(
                modele     = model,
                nom_modele = model_name,
                verbeux    = self.verbose,
            )
        except Exception as e:
            # En cas d'échec, on logue l'erreur et on interrompt proprement
            error_msg = str(e)
            self._log(f"  ❌ Erreur critique ({model_name}) : {error_msg}", "ERROR")
            return None  # On sort pour ne pas tenter de loguer des résultats vides
            
        # ------------------------------------------------------------------------
        # 2. TRACKING MLFLOW (Persistence et Audit)
        # ------------------------------------------------------------------------
    
        with mlflow.start_run(
            experiment_id = self.experiment_id,
            run_name      = f"{model_name}_run",
        ) as run:
    
            self._log(f"  MLflow run {run.info.run_id[:8]}...", "MLFLOW")
    
            # .................................................................
            # --- TAGS : Métadonnées d'identification ---
            # .................................................................
            mlflow.set_tags({
                **MLflowConfig.RUN_TAGS,
                "model_name":  model_name,               # Identifiant métier
                "model_class": type(model).__name__,     # Classe technique
                "debug_mode":  str(self.debug),          # État du debug
                "phase":       "phase3",                 # Étape du projet
            })
    
            # .................................................................
            # --- PARAMÈTRES : Configuration de l'expérience ---
            # .................................................................
            # Préparation des paramètres pour éviter les chaînes trop longues
            safe_params = {f"model_{k}": str(v)[:250] for k, v in model_params.items()}
            
            # Injection des paramètres de split et contexte
            safe_params.update({
                "random_state":  self.random_state,      # Graine d'aléatoire
                "cv_folds":      5,                      # Nombre de plis CV
                "eval_ratio":    self.eval_ratio,        # Part du set d'évaluation
                "n_features":    len(self.feature_names),# Nombre de variables
                "train_samples": len(self.y_train),      # Taille du set d'entraînement
                "eval_samples":  len(self.y_eval),       # Taille du set holdout
                "target_rate":   round(float(self.y_train.mean()), 4),
            })
            mlflow.log_params(safe_params)
    
            # .................................................................
            # --- MÉTRIQUES : Performance du modèle ---
            # .................................................................            
            # 1. Métriques sur Train (Performance après fit complet)
            for k, v in results["scores_train"].items():
                if isinstance(v, (int, float)) and not np.isnan(float(v)):
                    mlflow.log_metric(f"train_{k}", float(v))
            
            # 2. Métriques sur Eval (Holdout set : capacité de généralisation)
            # Note : 'scores_test' dans le modeler correspond à notre set d'évaluation
            for k, v in results["scores_test"].items():
                if isinstance(v, (int, float)) and not np.isnan(float(v)):
                    mlflow.log_metric(f"eval_{k}", float(v))
            
            # 3. Métriques de Validation Croisée (Analyse de la stabilité)
            for metric, d in results["scores_cv"].items():
                mlflow.log_metric(f"cv_{metric}_mean",       d["cv_moyenne"])
                mlflow.log_metric(f"cv_{metric}_std",        d["cv_ecart"])
                mlflow.log_metric(f"cv_train_{metric}_mean", d["train_moyenne"])
                mlflow.log_metric(f"cv_train_{metric}_std",  d["train_ecart"])
            
            # .................................................................            
            # --- INDICATEURS SYNTHÉTIQUES : Overfitting et Temps ---
            # .................................................................            
            
            # Calcul des écarts de performance (Train vs Eval)
            # Comme l'expliquerait Yann LeCun, un écart trop grand signale un sur-apprentissage.
            gap_f1 = results["scores_train"]["f1"] - results["scores_test"]["f1"]
            gap_f2 = results["scores_train"]["f2"] - results["scores_test"]["f2"]
            
            mlflow.log_metric("overfitting_f1", gap_f1)          # Stabilité F1-Score
            mlflow.log_metric("overfitting_f2", gap_f2)          # Stabilité F2-Score (Priorité)
            mlflow.log_metric("train_time_s",   results["temps_train"])
    
            # .................................................................
            # --- MODELE : Enregistrement du modèle avec signature (schéma I/O) ---
            # .................................................................
            # Enregistrement du modèle avec signature (schéma I/O)
            try:
                signature = infer_signature(
                    self.X_train.head(5),
                    results["predictions"]["y_test_pred"][:5],
                )
                mlflow.sklearn.log_model(
                    sk_model      = results["modele"],
                    artifact_path = "model",
                    signature     = signature,
                )
            except Exception:
                mlflow.sklearn.log_model(results["modele"], "model")
    
            # .................................................................
            # --- 5. ARTÉFACTS : Matrice de Confusion et Importance ----------
            # .................................................................
            
            # Définition des chemins de stockage (Alignement des :)
            cm_name : str = f"{model_name}_confusion_matrix.txt"
            fi_name : str = f"{model_name}_feature_importance.csv"
            
            cm_path = self.models_dir / cm_name            # Chemin matrice (.txt)
            fi_path = self.models_dir / fi_name            # Chemin importance (.csv)
    
            # Création du répertoire si inexistant
            self.models_dir.mkdir(parents=True, exist_ok=True)
    
            # A. Sauvegarde physique de la Matrice de Confusion
            cm_train = results['matrice_confusion']['train']
            cm_test  = results['matrice_confusion']['test']  # Set d'évaluation
    
            with open(cm_path, "w", encoding="utf-8") as f:
                f.write(f"Rapport de Matrice de Confusion : {model_name}\n")
                f.write("="*50 + "\n\n")
                f.write(f"TRAIN SET :\n{cm_train}\n\n")
                f.write(f"EVAL SET (Holdout) :\n{cm_test}\n")
    
            # Log de l'artéfact texte dans MLflow
            mlflow.log_artifact(str(cm_path))              # Envoi vers le serveur
    
            # B. Sauvegarde de l'Importance des Variables (Check de structure)
            # ----------------------------------------------------------------------------
            # Comme l'expliquerait Geoffrey Hinton, nous devons vérifier la capacité 
            # d'introspection du modèle avant d'extraire ses poids.
            # ----------------------------------------------------------------------------
            
            if hasattr(results["modele"], "feature_importances_"):
                # Cas des modèles basés sur les arbres (XGBoost, Random Forest, etc.)
                importances = results["modele"].feature_importances_
                
                df_fi = pd.DataFrame({
                    "feature":    self.feature_names,            # Noms des variables
                    "importance": importances                    # Poids de décision
                }).sort_values("importance", ascending=False)    # Tri décroissant
            
                df_fi.to_csv(fi_path, index=False)               # Export physique
                mlflow.log_artifact(str(fi_path))                # Log MLflow
                self._log("  ✓ Feature Importance enregistrée", "SUCCESS")
            
            elif hasattr(results["modele"], "coef_"):
                # Cas des modèles linéaires (Logistic Regression, SVM)
                # Note : On utilise la valeur absolue pour l'importance relative
                importances = np.abs(results["modele"].coef_[0])
                
                df_fi = pd.DataFrame({
                    "feature":    self.feature_names,
                    "importance": importances
                }).sort_values("importance", ascending=False)
            
                df_fi.to_csv(fi_path, index=False)
                mlflow.log_artifact(str(fi_path))
                self._log("  ✓ Coefficients linéaires enregistrés", "SUCCESS")
            
            else:
                # Cas des modèles sans mesure d'importance directe (ex: KNN)
                self._log("  ⚠️  Le modèle ne supporte pas l'importance des variables", "INFO")

            
            # .................................................................
            # --- DATASET : Traçabilité des données sources ---
            # .................................................................
            try:
                full_train = pd.concat([self.X_train, self.y_train], axis=1)
                mlflow.log_input(
                    mlflow.data.from_pandas(
                        full_train,
                        source = "v_features_engineering (PostgreSQL)",
                        name   = "train_data",
                    ),
                    context = "training",
                )
            except Exception:
                pass
    
            # Archivage du run_id pour usage ultérieur
            self.mlflow_runs[model_name] = run.info.run_id
    
        # ------------------------------------------------------------------------
        # 3. SYNTHÈSE ET FEEDBACK
        # ------------------------------------------------------------------------
        self.results[model_name] = results
        
        print("\n============================================================================")
        print(f"RAPPORT D'ENTRAÎNEMENT : {model_name}")
        print("============================================================================")
        print(f"  ID du Run MLflow....: {run.info.run_id}")
        print(f"  Échantillons Train..: {len(self.y_train)}")
        print(f"  Échantillons Eval...: {len(self.y_eval)}")
        print(f"  F1-Score (Eval).....: {results['scores_test']['f1']:.4f}")
        print(f"  F2-Score (Recall++).: {results['scores_test']['f2']:.4f}") # Métrique métier
        print(f"  Indicateur Overfit..: {gap_f2:.4f} (Basé sur F2)")
        print(f"  Temps d'exécution...: {results['temps_train']:.2f} secondes")
        print("============================================================================\n")

        self._log(f"  ✓ {model_name} — logué avec succès", "SUCCESS")
    
    # ==========================================================================
    # STEP 5 : COMPARAISON & SÉLECTION DU CHAMPION
    # ==========================================================================

    def step5_compare(self) -> pd.DataFrame:
        """
        Génère le tableau comparatif et sélectionne le champion.

        Critère de sélection : F2-Score sur eval set.

        Pourquoi F2 ?
        ─────────────
        beta=2 → le recall pèse 2× plus que la précision.
        Contexte crédit : un défaut non détecté (FN) coûte bien plus
        qu'un refus abusif (FP). F2 est aligné avec ce coût métier.
        AUC-ROC reste la métrique académique de référence.
        """
        self._sep()
        self._log("Comparaison des modèles ...", "STEP")

        if not self.results:
            self._log("Aucun modèle entraîné.", "ERROR")
            return pd.DataFrame()

        rows = []
        for nom, res in self.results.items():
            st = res["scores_train"]
            se = res["scores_test"]    # scores sur eval set (holdout)
            sc = res["scores_cv"]

            rows.append({
                "Modèle":           nom,
                "AUC-ROC(eval)":    round(se.get("roc_auc",   np.nan), 4),
                "F2(eval)":         round(se.get("f2",        np.nan), 4),
                "F1(eval)":         round(se.get("f1",        np.nan), 4),
                "Recall(eval)":     round(se.get("recall",    np.nan), 4),
                "Precision(eval)":  round(se.get("precision", np.nan), 4),
                "MCC(eval)":        round(se.get("mcc",       np.nan), 4),
                "F1 CV mean":       round(sc.get("f1", {}).get("cv_moyenne", np.nan), 4),
                "F1 CV std":        round(sc.get("f1", {}).get("cv_ecart",   np.nan), 4),
                "Overfit ΔF1":      round(st["f1"] - se.get("f1", 0), 4),
                "Train (s)":        round(res["temps_train"], 2),
                "Overfitting":      "❗" if res["surapprentissage"] else "✅",
                "MLflow Run":       self.mlflow_runs.get(nom, "")[:8],
            })

        df = (
            pd.DataFrame(rows)
            .sort_values("F2(eval)", ascending=False)
            .reset_index(drop=True)
        )
        df.insert(0, "Rang", ["🏆"] + [""] * (len(df) - 1))

        # Champion
        self.best_model_name = df.loc[0, "Modèle"]

        # Tag MLflow du champion
        best_run = self.mlflow_runs.get(self.best_model_name)
        if best_run:
            self.mlflow_client.set_tag(best_run, "best_model",   "true")
            self.mlflow_client.set_tag(best_run, "phase",        "phase3_champion")

        print("\n" + "=" * 110)
        print("COMPARAISON DES MODÈLES — Phase 3   (trié par F2-Score eval)")
        print("=" * 110)
        print(df.to_string(index=False))
        print("=" * 110)

        f2  = df.loc[0, "F2(eval)"]
        auc = df.loc[0, "AUC-ROC(eval)"]
        self._log(
            f"🏆 Champion : {self.best_model_name}  (F2={f2:.4f}  AUC-ROC={auc:.4f})",
            "SUCCESS",
        )
        self._log(
            "⮕  Phase 4 : Optuna pour optimiser les hyperparamètres du champion.",
            "INFO",
        )

        return df

    # ==========================================================================
    # STEP 6 : SAUVEGARDE JOBLIB + METADATA JSON
    # ==========================================================================

    def step6_save(self, save_all: bool = False) -> None:
        """
        Sauvegarde les modèles en local.

        Produit par modèle :
          • models/<nom>_model.joblib    → modèle sklearn sérialisé
          • models/<nom>_metadata.json  → métriques, params, feature names, run_id
        """
        self._sep()
        self._log("Sauvegarde des modèles ...", "STEP")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        noms = list(self.results.keys()) if save_all else [self.best_model_name]

        for nom in noms:
            if nom not in self.results:
                continue
            res = self.results[nom]
            sc  = res["scores_cv"]

            # Modèle binaire
            path_model = (self.models_dir / f"{nom}_model.joblib").absolute()
            joblib.dump(res["modele"], path_model)
            self.saved_model_paths[nom] = str(path_model)

            # Métadonnées JSON
            metadata = {
                "model_name":    nom,
                "model_class":   type(res["modele"]).__name__,
                "is_best":       (nom == self.best_model_name),
                "train_metrics": {k: float(v) if isinstance(v, (int, float)) else str(v)
                                  for k, v in res["scores_train"].items()},
                "eval_metrics":  {k: float(v) if isinstance(v, (int, float)) else str(v)
                                  for k, v in res["scores_test"].items()},
                "cv_f1_mean":    sc.get("f1", {}).get("cv_moyenne"),
                "cv_f1_std":     sc.get("f1", {}).get("cv_ecart"),
                "cv_auc_mean":   sc.get("roc_auc", {}).get("cv_moyenne"),
                "cv_auc_std":    sc.get("roc_auc", {}).get("cv_ecart"),
                "feature_names": self.feature_names,
                "n_features":    len(self.feature_names),
                "train_samples": len(self.y_train),
                "eval_samples":  len(self.y_eval),
                "target_rate":   float(self.y_train.mean()),
                "mlflow_run_id": self.mlflow_runs.get(nom),
                "model_path":    str(path_model),
                "saved_at":      datetime.now().isoformat(),
            }
            path_meta = (self.models_dir / f"{nom}_metadata.json").absolute()
            with open(path_meta, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            self.saved_metadata_paths[nom] = str(path_meta)

            self._log(f"  ✓ {nom:<30} → {path_model.name}", "INFO")

        self._log(f"{len(noms)} modèle(s) dans {self.models_dir}/", "SUCCESS")

    # ==========================================================================
    # STEP 7 : REGISTRE POSTGRESQL (model_versions)
    # ==========================================================================

    def step7_register_db(self) -> None:
        """
        Enregistre le champion dans la table PostgreSQL model_versions.

        Prérequis SQL (à créer une seule fois) :
        ─────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS model_versions (
                id              SERIAL PRIMARY KEY,
                model_name      TEXT NOT NULL,
                version         TEXT NOT NULL UNIQUE,
                mlflow_run_id   TEXT,
                algorithm       TEXT,
                hyperparameters JSONB,
                metrics         JSONB,
                model_path      TEXT,
                metadata_path   TEXT,
                status          TEXT DEFAULT 'trained',
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """
        if not DB_AVAILABLE or not self.best_model_name:
            self._log("Enregistrement DB ignoré (DB non dispo ou aucun champion).", "WARNING")
            return

        self._sep()
        self._log(
            f"Enregistrement champion [{self.best_model_name}] dans PostgreSQL ...",
            "STEP",
        )
        try:
            engine = get_engine()
            res    = self.results[self.best_model_name]
            version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 1. Préparation des données (On sérialise en string JSON ici)
            # ----------------------------------------------------------------
            params_dict = {k: str(v) for k, v in res["modele"].get_params().items()}
            
            # Note de Dario Amodei : On s'assure que les métriques sont propres
            metrics_dict = {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                           for k, v in res["scores_test"].items()}

            info = {
                "name":       self.best_model_name,
                "version":    version,
                "run_id":     self.mlflow_runs.get(self.best_model_name),
                "algo":       type(res["modele"]).__name__,
                "params":     json.dumps(params_dict),
                "metrics":    json.dumps(metrics_dict),
                "model_path": self.saved_model_paths.get(self.best_model_name, ""),
                "meta_path":  self.saved_metadata_paths.get(self.best_model_name, ""),
                "status":     "trained",
            }

            # 2. Exécution avec la syntaxe de repli sécurisée
            # ----------------------------------------------------------------
            sql = text("""
                INSERT INTO model_versions 
                    (model_name, version, mlflow_run_id, algorithm, 
                     hyperparameters, metrics, model_path, metadata_path, status)
                VALUES 
                    (:name, :version, :run_id, :algo, 
                     CAST(:params AS JSONB), CAST(:metrics AS JSONB), 
                     :model_path, :meta_path, :status)
            """)

            with engine.begin() as conn:
                conn.execute(sql, info)

            self._log(f"✅ Champion enregistré en DB (Version: {version})", "SUCCESS")

        except Exception as exc:
            self._log(f"❌ Erreur DB : {exc}", "ERROR")

# ##############################################################################
# FONCTION PRINCIPALE
# ##############################################################################

def run_phase3(
    source:          str   = "db",                # Source : 'db' ou 'csv'
    eval_ratio:      float = 0.20,                # Proportion du set d'éval
    random_state:    int   = RANDOM_SEED,         # Graine aléatoire centrale
    debug:           bool  = False,               # Mode test rapide
    debug_limit:     int   = DEBUG_ROW_LIMIT,     # Limite via constante
    save_all:        bool  = False,               # Sauvegarde de tous les modèles
    experiment_name: str   = DEFAULT_EXP_NAME,    # Nom du tracking MLflow
    verbose:         bool  = True                 # Affichage des logs
) -> Phase3Pipeline:
    """
    Exécute la Phase 3 complète.

    Args:
        source          : "db" → PostgreSQL, "csv" → data/processed/
        eval_ratio      : Fraction holdout eval (défaut 0.20)
        random_state    : Graine aléatoire (défaut 42)
        debug           : Mode debug (charge seulement debug_limit lignes)
        debug_limit     : Nb lignes en mode debug (défaut {DEBUG_ROW_LIMIT} )
        save_all        : Sauvegarder tous les modèles (défaut : champion seul)
        experiment_name : Nom expérience MLflow
        verbose         : Logs détaillés

    Returns:
        Instance Phase3Pipeline avec .results, .best_model_name, .mlflow_runs.
    """
    print("\n" + "=" * 76)
    print("🚀 PHASE 3 — CLASSIFICATION + MLFLOW TRACKING")
    print("   Projet : Prêt à Dépenser — Home Credit Default Risk")
    print("=" * 76)

    pipeline = Phase3Pipeline(
        source=source,
        eval_ratio=eval_ratio,
        random_state=random_state,
        debug=debug,
        debug_limit=debug_limit,
        experiment_name=experiment_name,
        verbose=verbose,
    )

    try:
        pipeline.step0_setup_mlflow()
        pipeline.step1_load_data()
        pipeline.step2_split()
        pipeline.step3_init_modeler()
        pipeline.step4_train_all()
        pipeline.step5_compare()
        pipeline.step6_save(save_all=save_all)
        pipeline.step7_register_db()

        print("\n" + "=" * 76)
        print("✅ PHASE 3 TERMINÉE AVEC SUCCÈS")
        print("=" * 76)
        print(f"  Modèles entraînés  : {len(pipeline.results)}")
        print(f"  Champion           : {pipeline.best_model_name}")
        print(f"  Modèles sauvegardés: {pipeline.models_dir}/")
        print(f"  MLflow UI          : {MLflowConfig.TRACKING_URI}")
        print(f"  Expérience         : {pipeline.experiment_name}")
        print()
        print("  ⮕  Suite : Phase 4 — Optimisation hyperparamètres + seuil métier")
        print("     uv run python -m src.pipelines.phase4_hyperparameter_tuning")
        print("=" * 76)

        return pipeline

    except Exception as exc:
        print(f"\n❌ ERREUR Phase 3 : {exc}")
        traceback.print_exc()
        raise


def run_phase3(
    source:          str   = "db",                # Source des données
    eval_ratio:      float = 0.20,                # Ratio évaluation
    random_state:    int   = RANDOM_SEED,         # Graine aléatoire
    debug:           bool  = False,               # Mode debug actif/inactif
    debug_limit:     int   = DEBUG_ROW_LIMIT,     # Limite de lignes
    save_all:        bool  = False,               # Sauvegarde intégrale
    experiment_name: str   = DEFAULT_EXP_NAME,    # Nom Expérience MLflow
    verbose:         bool  = True                 # Affichage détaillé
) -> Phase3Pipeline:
    """
    Exécute la Phase 3 complète.

    Args:
        source          : "db" → PostgreSQL, "csv" → data/processed/
        eval_ratio      : Fraction holdout eval (défaut 0.20)
        random_state    : Graine aléatoire (défaut 42)
        debug           : Mode debug (charge seulement debug_limit lignes)
        debug_limit     : Nb lignes en mode debug (défaut {DEBUG_ROW_LIMIT} )
        save_all        : Sauvegarder tous les modèles (défaut : champion seul)
        experiment_name : Nom expérience MLflow
        verbose         : Logs détaillés

    Returns:
        Instance Phase3Pipeline avec .results, .best_model_name, .mlflow_runs.
    """
    import time                                   # Pour le chronométrage
    import datetime                               # Pour le formatage HH:MM:SS

    print("\n" + "=" * 76)
    print("🚀 PHASE 3 — CLASSIFICATION + MLFLOW TRACKING")
    print("   Projet : Prêt à Dépenser — Home Credit Default Risk")
    print("=" * 76)

    # Capturer le temps initial
    t_start = time.time()                         # Temps de début (Epoch)

    pipeline = Phase3Pipeline(
        source          = source,
        eval_ratio      = eval_ratio,
        random_state    = random_state,
        debug           = debug,
        debug_limit     = debug_limit,
        experiment_name = experiment_name,
        verbose         = verbose,
    )

    try:
        # Exécution séquentielle des étapes du pipeline
        pipeline.step0_setup_mlflow()
        pipeline.step1_load_data()
        pipeline.step2_split()
        pipeline.step3_init_modeler()
        pipeline.step4_train_all()
        pipeline.step5_compare()
        pipeline.step6_save(save_all=save_all)
        pipeline.step7_register_db()

        # Calcul de la durée totale
        t_end    = time.time()                    # Temps de fin
        duration = t_end - t_start                # Delta en secondes
        
        # Formatage de la durée en HH:MM:SS
        d_format = str(datetime.timedelta(seconds=round(duration)))

        print("\n" + "=" * 76)
        print("✅ PHASE 3 TERMINÉE AVEC SUCCÈS")
        print("=" * 76)
        print(f"  Durée totale........: {d_format} (H:M:S)")
        print(f"  Modèles entraînés...: {len(pipeline.results)}")
        print(f"  Champion............: {pipeline.best_model_name}")
        print(f"  MLflow UI...........: {MLflowConfig.TRACKING_URI}")
        print(f"  Expérience..........: {pipeline.experiment_name}")
        print()
        print("  ⮕  Suite : Phase 4 — Optimisation hyperparamètres")
        print("=" * 76)

        return pipeline

    except Exception as exc:
        print(f"\n❌ ERREUR Phase 3 : {exc}")
        import traceback
        traceback.print_exc()
        raise
    
# ##############################################################################
# CLI
# ##############################################################################

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 — Classification + MLflow (Prêt à Dépenser)"
    )
    p.add_argument(
        "--source", choices=["db", "csv"], default="db",
        help="Source données : 'db' (PostgreSQL) ou 'csv' (data/processed/)",
    )
    p.add_argument(
        "--eval-ratio", type=float, default=0.20,
        help="Fraction holdout eval (défaut 0.20)",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Mode debug : charge seulement --debug-limit lignes",
    )
    p.add_argument(
        "--debug-limit", 
        type    = int, 
        default = DEBUG_ROW_LIMIT,                # Utilisation de la constante
        help    = f"Nb lignes en mode debug (défaut {DEBUG_ROW_LIMIT})"
    )
    
    p.add_argument(
        "--save-all", action="store_true",
        help="Sauvegarder tous les modèles (défaut : champion uniquement)",
    )
    p.add_argument(
        "--experiment", default=None,
        help=f"Nom expérience MLflow (défaut : {MLflowConfig.EXPERIMENT_NAME})",
    )
    p.add_argument(
        "--no-verbose", action="store_true",
        help="Désactiver les logs détaillés",
    )
    return p.parse_args()


if __name__ == "__main__":
    print("\n" + "=" * 76)
    print("⚠️  PRÉREQUIS : démarrer MLflow UI dans un terminal séparé")
    print("   Terminal 1 →  mlflow ui --port 5000")
    print("=" * 76)

    args = _parse_args()
    run_phase3(
        source=args.source,
        eval_ratio=args.eval_ratio,
        debug=args.debug,
        debug_limit=args.debug_limit,
        save_all=args.save_all,
        experiment_name=args.experiment,
        verbose=not args.no_verbose,
    )
