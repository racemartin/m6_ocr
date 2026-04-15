# =============================================================================
# INJECTION DE DÉPENDANCES — Sélection des adaptateurs selon l'environnement
# Fournit les instances des adaptateurs au use case via FastAPI Depends.
# Le choix du backend (ONNX ou MLflow) est piloté par MODEL_BACKEND.
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import logging  # Journalisation du choix d'adaptateur

# --- FastAPI — Injection de dépendances --------------------------------------
from fastapi import Depends  # Déclaration de dépendance dans les routes

# --- Adaptateurs concrets ----------------------------------------------------
from src.api.adapters.scorers.onnx_scorer_adapter   import OnnxScorerAdaptater
from src.api.adapters.scorers.mlflow_scorer_adapter import MlflowScoeurAdapter
from src.api.adapters.loggers.jsonl_logger_adapter  import JsonlJournaliseurAdapter

# --- Cas d'utilisation -------------------------------------------------------
from src.api.application.evaluate_credit_use_case import EvaluerDemandeCreditUseCase

# --- Ports (pour les annotations de type) ------------------------------------
from src.api.ports.i_credit_scorer     import ICreditScorer
from src.api.ports.i_prediction_logger import IJournaliseurPredictions

# --- Configuration globale ---------------------------------------------------
from config import parametres  # MODEL_BACKEND et paramètres associés
import sys
import os
import inspect
from src.tools.rafael.log_tool import LogTool

# Configuration du logger applicatif pour ce module
journalapp  = logging.getLogger(__name__)
log         = LogTool(origin="UseCase")
NOM_FICHIER = os.path.basename(__file__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# =============================================================================
# INSTANCES PARTAGÉES (pattern module-level singleton)
# Créées une seule fois au démarrage via le lifespan FastAPI.
# =============================================================================
_scoreur:      ICreditScorer | None            = None  # Adaptateur de scoring
_journaliseur: IJournaliseurPredictions | None = None  # Adaptateur de logging


# =============================================================================
# INITIALISATION DES ADAPTATEURS (appelée depuis lifespan)
# =============================================================================
def initialiser_adaptateurs() -> None:
    """
    Instancie et charge les adaptateurs selon MODEL_BACKEND.

    Appelée UNE SEULE FOIS au démarrage du serveur dans le lifespan
    FastAPI. Cette fonction garantit que le modèle est chargé en mémoire
    avant la première requête (évite le cold start par requête).

    Backends supportés
    ------------------
    "onnx"   → OnnxScoeurAdapter   (production, HuggingFace)
    "mlflow" → MlflowScoeurAdapter (développement local avec Docker MLflow)

    Lève
    ----
    ValueError
        Si MODEL_BACKEND contient une valeur inconnue.
    """
    global _scoreur, _journaliseur

    backend = parametres.model_backend.lower()
    log.START_INDETED_LEVEL(2, NOM_FICHIER, inspect.currentframe().f_code.co_name , "BEGIN")
                            
    # -- Sélection et chargement de l'adaptateur de scoring ------------------
    if backend == "onnx":
        adaptateur = OnnxScorerAdaptater()
        adaptateur.charger()
        _scoreur = adaptateur
    elif backend == "mlflow":
        adaptateur = MlflowScoeurAdapter()
        adaptateur.charger()
        _scoreur = adaptateur
    else:
        raise ValueError(
            f"MODEL_BACKEND inconnu : '{backend}'. "
            f"Valeurs acceptées : 'onnx', 'mlflow'."
        )

    # -- Instanciation du journaliseur JSONL ----------------------------------
    _journaliseur = JsonlJournaliseurAdapter()

    log.DEBUG_PARAMETER_VALUE("Adaptateur initialisé", f"scoreur      = {type(_scoreur).__name__}")
    log.DEBUG_PARAMETER_VALUE("Adaptateur initialisé", f"journaliseur = {type(_journaliseur).__name__}")

    log.FINISH_INDETED_LEVEL(2, NOM_FICHIER, inspect.currentframe().f_code.co_name ,"FINISH")

# =============================================================================
# FOURNISSEURS DE DÉPENDANCES (injectés par FastAPI dans les routes)
# =============================================================================
def obtenir_scorer() -> ICreditScorer:
    """
    Fournisseur FastAPI — retourne l'adaptateur de scoring actif.

    Utilisé via Depends(obtenir_scorer) dans les routes.
    L'instance est partagée (singleton) entre toutes les requêtes.
    """
    if _scoreur is None:
        raise RuntimeError(
            "Le scoreur n'a pas été initialisé. "
            "Vérifiez que initialiser_adaptateurs() est appelé au démarrage."
        )
    return _scoreur


def obtenir_journaliseur() -> IJournaliseurPredictions:
    """
    Fournisseur FastAPI — retourne l'adaptateur de journalisation actif.

    Utilisé via Depends(obtenir_journaliseur) dans les routes.
    """
    if _journaliseur is None:
        raise RuntimeError(
            "Le journaliseur n'a pas été initialisé. "
            "Vérifiez que initialiser_adaptateurs() est appelé au démarrage."
        )
    return _journaliseur

def obtenir_use_case(
    scoreur:      ICreditScorer            = Depends(obtenir_scorer),
    journaliseur: IJournaliseurPredictions = Depends(obtenir_journaliseur),
) -> EvaluerDemandeCreditUseCase:
    """
    Fournisseur FastAPI — construit le use case avec ses dépendances.

    FastAPI injecte automatiquement scoreur et journaliseur grâce à
    Depends(). Le use case est recréé à chaque requête (très léger :
    aucun I/O, pas de chargement modèle — les adaptateurs sont partagés).

    Paramètres
    ----------
    scoreur : ICreditScorer
        Adaptateur de scoring injecté par FastAPI.
    journaliseur : IJournaliseurPredictions
        Adaptateur de journalisation injecté par FastAPI.

    Retourne
    --------
    EvaluerDemandeCreditUseCase
        Instance prête à recevoir une demande de crédit.
    """
    return EvaluerDemandeCreditUseCase(
        scoreur      = scoreur,
        journaliseur = journaliseur,
    )
