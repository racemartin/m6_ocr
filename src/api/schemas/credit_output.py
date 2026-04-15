# =============================================================================
# src/api/schemas/credit_output.py -- Schema de reponse POST /predict
# Enrichi avec les explications SHAP pour la transparence algorithmique.
# =============================================================================

# --- Bibliotheques standard ---------------------------------------------------
from typing import List, Literal, Optional  # Annotations et contraintes

# --- Bibliotheques tierces : API ---------------------------------------------
from pydantic import BaseModel, Field       # Validation et doc Swagger


# =============================================================================
# Schema : Explication SHAP d'une feature (en langage client)
# =============================================================================
class ExplicationShapSchema(BaseModel):
    """
    Contribution SHAP d'une feature a la decision de credit.

    Retournee en langage metier (pas de noms de colonnes techniques).
    Les valeurs originales envoyees par le client sont preservees
    et retournees telles quelles -- aucune normalisation n'est visible.

    Exemple de reponse :
    {
        "feature"          : "Taux d utilisation du credit",
        "valeur_client"    : "0.45",
        "impact_shap"      : 0.312,
        "direction"        : "hausse_risque",
        "explication"      : "Un taux d utilisation de 0.45 augmente
                              significativement le risque de defaut."
    }
    """

    # -- Identification de la feature ----------------------------------------
    feature: str = Field(
        description = "Nom metier de la feature en langage client.",
        examples    = ["Taux d utilisation du credit"],
    )

    # -- Valeur originale du client (telle qu'envoyee) -----------------------
    valeur_client: str = Field(
        description = (
            "Valeur exacte envoyee par le client, en string lisible. "
            "Jamais normalisee ni encodee."
        ),
        examples    = ["0.45"],
    )

    # -- Impact SHAP agrege --------------------------------------------------
    impact_shap: float = Field(
        description = (
            "Contribution SHAP de cette feature a la probabilite de defaut. "
            "Positif -> hausse le risque. Negatif -> baisse le risque. "
            "Valeurs proches de 0 -> feature peu influente pour ce client."
        ),
        examples    = [0.312],
    )

    # -- Direction lisible ---------------------------------------------------
    direction: Literal["hausse_risque", "baisse_risque", "neutre"] = Field(
        description = (
            "Direction de l'impact : 'hausse_risque' si impact > 0, "
            "'baisse_risque' si impact < 0, 'neutre' si negligeable."
        ),
        examples    = ["hausse_risque"],
    )

    # -- Explication en langage naturel (generee par l'API) ------------------
    explication: Optional[str] = Field(
        default     = None,
        description = (
            "Explication en langage naturel de la contribution de cette "
            "feature pour ce client specifique. Generee automatiquement."
        ),
        examples    = [
            "Un taux d utilisation de 0.45 augmente le risque de defaut."
        ],
    )


# =============================================================================
# Schema : Reponse complete de l'API POST /predict avec SHAP
# =============================================================================
class ClientDataOutput(BaseModel):
    """
    Reponse JSON de POST /predict avec decision et explication SHAP.

    Structure de la reponse :
    {
        "id_demande"         : "550e8400-...",
        "probabilite_defaut" : 0.7231,
        "decision"           : "Refuse",
        "score_risque"       : 72.31,
        "latence_ms"         : 3.8,
        "seuil_utilise"      : 0.35,
        "explication_shap"   : [
            {
                "feature"       : "Taux d utilisation du credit",
                "valeur_client" : "0.45",
                "impact_shap"   : +0.312,
                "direction"     : "hausse_risque"
            },
            {
                "feature"       : "Taux d incidents de paiement",
                "valeur_client" : "0.02",
                "impact_shap"   : +0.089,
                "direction"     : "hausse_risque"
            },
            ...
        ]
    }
    """

    # -- Identifiant de tracabilite ------------------------------------------
    id_demande: str = Field(
        description = "UUID unique de la demande evaluee.",
        examples    = ["550e8400-e29b-41d4-a716-446655440000"],
    )

    # -- Resultat du scoring -------------------------------------------------
    probabilite_defaut: float = Field(
        ge          = 0.0,
        le          = 1.0,
        description = "Probabilite de defaut de paiement [0.0, 1.0].",
        examples    = [0.7231],
    )

    decision: Literal["Approuvé", "Refusé"] = Field(
        description = (
            "Decision finale : 'Approuvé' si proba < seuil, 'Refusé' sinon."
        ),
        examples    = ["Refusé"],
    )

    score_risque: float = Field(
        ge          = 0.0,
        le          = 100.0,
        description = "Score de risque normalise sur 100 (= probabilite * 100).",
        examples    = [72.31],
    )

    # -- Metriques techniques ------------------------------------------------
    latence_ms: float = Field(
        ge          = 0.0,
        description = "Temps d inference ONNX en millisecondes.",
        examples    = [3.8],
    )

    seuil_utilise: float = Field(
        ge          = 0.0,
        le          = 1.0,
        description = "Seuil de decision applique au moment du scoring.",
        examples    = [0.35],
    )

    # -- Explication SHAP (top features triees par |impact| decroissant) -----
    explication_shap: List[ExplicationShapSchema] = Field(
        default     = [],
        description = (
            "Liste des features les plus influentes pour cette decision, "
            "triees par |impact SHAP| decroissant. "
            "Les valeurs sont exprimees dans l'espace original du client "
            "(pas de normalisation ni de one-hot visible). "
            "Vide si le calcul SHAP est desactive ou a echoue."
        ),
    )

    # -- Exemple complet pour Swagger UI -------------------------------------
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id_demande"         : "550e8400-e29b-41d4-a716-446655440000",
                    "probabilite_defaut" : 0.7231,
                    "decision"           : "Refuse",
                    "score_risque"       : 72.31,
                    "latence_ms"         : 3.8,
                    "seuil_utilise"      : 0.35,
                    "explication_shap"   : [
                        {
                            "feature"       : "Taux d utilisation du credit",
                            "valeur_client" : "0.45",
                            "impact_shap"   : 0.312,
                            "direction"     : "hausse_risque",
                            "explication"   : (
                                "Un taux d utilisation de 0.45 contribue "
                                "fortement a la hausse du risque de defaut."
                            ),
                        },
                        {
                            "feature"       : "Taux d incidents de paiement",
                            "valeur_client" : "0.02",
                            "impact_shap"   : 0.089,
                            "direction"     : "hausse_risque",
                            "explication"   : None,
                        },
                        {
                            "feature"       : "Type de residence",
                            "valeur_client" : "Locataire",
                            "impact_shap"   : -0.041,
                            "direction"     : "baisse_risque",
                            "explication"   : None,
                        },
                    ],
                }
            ]
        }
    }
