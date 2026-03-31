# =============================================================================
# SCHÉMA DE SORTIE — Réponse Pydantic du endpoint POST /predict
# Structure la réponse JSON retournée au client après le scoring.
# Documenté automatiquement dans Swagger UI (OpenAPI).
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
from typing import Literal  # Contrainte de valeur sur un champ string

# --- Pydantic — Sérialisation et documentation de la réponse ----------------
from pydantic import BaseModel, Field  # Modèle de réponse avec descriptions


# =============================================================================
# SCHÉMA : DONNÉES DE SORTIE CLIENT
# =============================================================================
class ClientDataOutput(BaseModel):
    """
    Corps de la réponse JSON du endpoint POST /predict.

    Retourné au client après scoring réussi. Contient le résultat
    de la décision, la probabilité de défaut et la latence mesurée.

    Champs
    ------
    id_demande : str
        UUID unique de la demande traitée (utile pour le suivi).
    probabilite_defaut : float
        Probabilité brute de défaut de paiement [0.0, 1.0].
    decision : Literal["Approuvé", "Refusé"]
        Décision binaire issue du seuil métier (par défaut 0.35).
    score_risque : float
        Score normalisé [0–100] pour affichage dans les dashboards.
    latence_ms : float
        Temps d'inférence mesuré côté serveur (en millisecondes).
    """

    # -- Identifiant de traçabilité -------------------------------------------
    id_demande: str = Field(
        description = "Identifiant UUID unique de la demande évaluée.",
        examples    = ["550e8400-e29b-41d4-a716-446655440000"],
    )

    # -- Résultat du scoring --------------------------------------------------
    probabilite_defaut: float = Field(
        ge          = 0.0,
        le          = 1.0,
        description = (
            "Probabilité de défaut de paiement produite par le modèle "
            "[0.0 = risque nul, 1.0 = défaut certain]."
        ),
        examples    = [0.7231],
    )

    decision: Literal["Approuvé", "Refusé"] = Field(
        description = (
            "Décision finale : 'Approuvé' si proba < seuil, "
            "'Refusé' sinon. Seuil métier par défaut : 0.35."
        ),
        examples    = ["Refusé"],
    )

    score_risque: float = Field(
        ge          = 0.0,
        le          = 100.0,
        description = (
            "Score de risque normalisé sur 100 "
            "(100 × probabilité_défaut). Utilisé pour les dashboards."
        ),
        examples    = [72.31],
    )

    latence_ms: float = Field(
        ge          = 0.0,
        description = (
            "Temps d'inférence mesuré côté serveur, en millisecondes."
        ),
        examples    = [4.3],
    )

    # -- Exemple complet pour Swagger UI --------------------------------------
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id_demande"         : "550e8400-e29b-41d4-a716-446655440000",
                    "probabilite_defaut" : 0.7231,
                    "decision"           : "Refusé",
                    "score_risque"       : 72.31,
                    "latence_ms"         : 4.3,
                }
            ]
        }
    }
