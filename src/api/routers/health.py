# =============================================================================
# src/api/routers/health.py — Route GET /health
# Endpoint de supervision : vérifie que le modèle est chargé et que
# le service est opérationnel. Utilisé par Docker healthcheck,
# HuggingFace Spaces et tout outil de monitoring externe.
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import logging                                        # Journalisation

# --- Bibliothèques tierces : API ---------------------------------------------
from   fastapi import APIRouter, Depends              # Router et injection

# --- Couche API : dépendances ------------------------------------------------
from src.api.dependencies import obtenir_scorer       # Scorer pour vérif état

# --- Ports -------------------------------------------------------------------
from src.api.ports.i_credit_scorer import ICreditScorer   # Interface scorer

# --- Configuration -----------------------------------------------------------
from config import parametres                         # Version, backend, seuil


# Journalisation du module
journal = logging.getLogger(__name__)

# Instance du router FastAPI
routeur = APIRouter()


# ##############################################################################
# Route : GET /health
# ##############################################################################

# =============================================================================
@routeur.get(
    "/health",
    summary     = "Vérification de l'état du service",
    description = (
        "Retourne l'état de santé du service de scoring crédit.\n\n"
        "Vérifie que le modèle est chargé et prêt à recevoir des requêtes. "
        "Utilisé par les sondes Docker et HuggingFace Spaces."
    ),
    tags        = ["Supervision"],
    status_code = 200,
)
def verifier_sante(
    scorer: ICreditScorer = Depends(obtenir_scorer),
) -> dict:
    """
    Vérifie l'état de santé du service.

    Retourne les métadonnées utiles pour le monitoring et le dashboard
    de supervision (backend actif, version, seuil configuré).

    Args:
        scorer : Scorer injecté — son attribut est_pret est consulté.

    Returns:
        Dictionnaire avec statut, version, backend et seuil de décision.
    """
    statut = "ok" if scorer.est_pret else "dégradé"

    journal.debug("GET /health | statut=%s", statut)

    return {
        "statut"         : statut,
        "scorer_pret"    : scorer.est_pret,
        "backend_modele" : parametres.backend_modele,
        "version_api"    : parametres.version_api,
        "seuil_decision" : parametres.seuil_decision,
    }
