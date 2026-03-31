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
    # -- Phase démarrage : chargement du modèle en mémoire ------------------
    print("\n============================================================================")
    print("DÉMARRAGE — API SCORING CRÉDIT : PRÊT À DÉPENSER")
    print("============================================================================")
    print(f"  Backend modèle..........: {parametres.model_backend}")
    print(f"  Seuil de décision.......: {parametres.seuil_decision}")
    print(f"  Version API.............: {parametres.version_api}")
    print("============================================================================")

    try:
        initialiser_adaptateurs()
        journal.info("Modèle chargé. Serveur prêt à recevoir des requêtes.")
    except Exception as erreur:
        journal.critical("Échec du chargement du modèle : %s", erreur)
        raise  # Empêche le démarrage si le modèle est inaccessible

    yield  # L'application est active à partir d'ici

    # -- Phase arrêt : libération des ressources ----------------------------
    print("\n============================================================================")
    print("ARRÊT — Libération des ressources.")
    print("============================================================================")


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
