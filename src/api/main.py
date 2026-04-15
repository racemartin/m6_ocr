# =============================================================================
# src/api/main.py — Point d'entrée FastAPI : application de scoring crédit
# Crée l'application FastAPI, enregistre les routers et gère le cycle
# de vie du serveur via lifespan (chargement/déchargement du modèle).
#
# Démarrage local :
#   uvicorn src.api.main:application --host 0.0.0.0 --port 8000 --reload
#
# Démarrage Docker / HuggingFace (port imposé = 7860) :
#   uvicorn src.api.main:application --host 0.0.0.0 --port 7860
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import logging                                        # Configuration journaux
from   contextlib import asynccontextmanager          # Lifespan async

# --- Bibliothèques tierces : API ---------------------------------------------
from   fastapi                   import FastAPI       # Cadre applicatif
from   fastapi.middleware.cors   import CORSMiddleware  # Autorisations CORS

# --- Couche API : routers ----------------------------------------------------
from src.api.routers import predict                   # POST /predict
from src.api.routers import health                    # GET  /health
from src.api.routers import drift                     # GET  /drift/report

# --- Injection des dépendances -----------------------------------------------
from src.api.dependencies import initialiser_adaptateurs  # Warm-up modèle

# --- Configuration -----------------------------------------------------------
from config import parametres                         # Titre, version, seuil

import sys
import os
import inspect
from src.tools.rafael.log_tool import LogTool
log = LogTool(origin="UseCase")
NOM_FICHIER = os.path.basename(__file__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =============================================================================
# Configuration globale de la journalisation
# =============================================================================
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
journal = logging.getLogger(__name__)


# =============================================================================
# Gestionnaire de cycle de vie — lifespan
# Remplace les décorateurs dépréciés on_event("startup"/"shutdown").
# Exécuté par FastAPI au démarrage (avant yield) et à l'arrêt (après).
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie complet de l'application FastAPI.

    Démarrage (avant le yield) :
        - Charge le modèle ONNX ou MLflow en mémoire (warm-up unique)
        - Initialise le logger JSONL

    Arrêt (après le yield) :
        - Libère les ressources si nécessaire

    Args:
        app : Instance FastAPI (paramètre imposé par le protocole lifespan).
    """
    log.START_ACTION(NOM_FICHIER, inspect.currentframe().f_code.co_name , "BEGING")

    # -- Phase démarrage : chargement du modèle en mémoire ------------------
    log.STEP(3, "DÉMARRAGE — API SCORING CRÉDIT", "PRÊT À DÉPENSER")
    log.DEBUG_PARAMETER_VALUE("Backend modèle"      , parametres.model_backend)
    log.DEBUG_PARAMETER_VALUE("Seuil de décision"   , parametres.seuil_decision)
    log.DEBUG_PARAMETER_VALUE("Version API"         , parametres.version_api)
    # log.log_io_functions()
    
    try:
        initialiser_adaptateurs()
        log.LEVEL_7_INFO(NOM_FICHIER, "Modèle chargé. Serveur prêt à recevoir des requêtes.")
        log.FINISH_ACTION(NOM_FICHIER, inspect.currentframe().f_code.co_name , "FINISH")
    except Exception as erreur:
        log.LEVEL_3_CRITICAL(NOM_FICHIER, f"Échec du chargement du modèle : {erreur}")
        raise  # Empêche le démarrage si le modèle est inaccessible

    yield  # L'application est active à partir d'ici

    # -- Phase arrêt : libération des ressources ----------------------------
    log.LEVEL_7_INFO(NOM_FICHIER, "ARRÊT — Libération des ressources.")

# =============================================================================
# Création de l'application FastAPI
# =============================================================================
application = FastAPI(
    title       = parametres.titre_api,
    version     = parametres.version_api,
    description = (
        "API de scoring crédit basée sur un modèle LightGBM entraîné "
        "sur les données Home Credit Default Risk. "
        "Déployée sur HuggingFace Spaces (format ONNX).\n\n"
        "## Endpoints disponibles\n\n"
        "- **POST /predict** — Évaluation d'une demande de crédit\n"
        "- **GET /health** — État de santé du service\n"
        "- **GET /drift/report** — Rapport de drift des données\n\n"
        "## Modèle\n\n"
        "LightGBM (Phase 4 — Optimisation bayésienne), "
        "exporté au format ONNX pour une inférence optimale (≈ 3–5 ms)."
    ),
    lifespan    = lifespan,
    docs_url    = "/docs",    # Swagger UI
    redoc_url   = "/redoc",   # ReDoc
)


# =============================================================================
# Middleware CORS
# Autorise les requêtes cross-origin depuis HuggingFace et localhost.
# En production sécurisée, restreindre allow_origins aux domaines connus.
# =============================================================================
application.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],              # À restreindre si nécessaire
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],    # Méthodes autorisées
    allow_headers     = ["*"],
)


# =============================================================================
# Enregistrement des routers — chaque router gère un groupe de routes
# =============================================================================
application.include_router(predict.routeur)  # POST /predict
application.include_router(health.routeur)   # GET  /health
application.include_router(drift.routeur)    # GET  /drift/report
