# =============================================================================
# SCHÉMA D'ENTRÉE — Validation Pydantic de la requête POST /predict
# Définit les 11 features attendues, leur type et leurs contraintes.
# Ce schéma documente automatiquement l'API Swagger (OpenAPI).
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
from typing import Annotated  # Annotations avec métadonnées de validation

# --- Pydantic — Validation et documentation automatique ----------------------
from pydantic import BaseModel, Field  # Modèle de données avec contraintes


# =============================================================================
# SCHÉMA : DONNÉES D'ENTRÉE CLIENT
# =============================================================================
class ClientDataInput(BaseModel):
    """
    Corps de la requête POST /predict.

    Contient les 11 features nécessaires au modèle de scoring crédit,
    identiques à celles utilisées lors de l'entraînement (m6_ocr).

    Toutes les valeurs sont validées à la réception — une requête
    invalide retourne HTTP 422 avec le détail des erreurs Pydantic.
    """

    # -- Features numériques continues ----------------------------------------
    age: Annotated[int, Field(
        ge          = 18,
        le          = 100,
        description = "Âge du client en années (18–100).",
        examples    = [35],
    )]

    revenu: Annotated[float, Field(
        gt          = 0,
        description = "Revenu annuel déclaré en euros (> 0).",
        examples    = [45000.0],
    )]

    montant_pret: Annotated[float, Field(
        gt          = 0,
        description = "Montant total du prêt demandé en euros (> 0).",
        examples    = [15000.0],
    )]

    duree_pret_mois: Annotated[int, Field(
        ge          = 1,
        le          = 360,
        description = "Durée de remboursement en mois (1–360).",
        examples    = [48],
    )]

    # -- Features numériques issues de l'historique crédit --------------------
    jours_retard_moyen: Annotated[float, Field(
        ge          = 0.0,
        description = (
            "Moyenne des jours de retard par incident de paiement (≥ 0)."
        ),
        examples    = [0.5],
    )]

    taux_incidents: Annotated[float, Field(
        ge          = 0.0,
        le          = 1.0,
        description = "Ratio incidents / paiements totaux [0.0, 1.0].",
        examples    = [0.02],
    )]

    taux_utilisation_credit: Annotated[float, Field(
        ge          = 0.0,
        le          = 1.0,
        description = "Utilisation du crédit disponible [0.0, 1.0].",
        examples    = [0.45],
    )]

    nb_comptes_ouverts: Annotated[int, Field(
        ge          = 0,
        description = "Nombre de comptes bancaires actifs (≥ 0).",
        examples    = [3],
    )]

    # -- Features catégorielles -----------------------------------------------
    type_residence: Annotated[str, Field(
        description = (
            'Statut résidentiel : "Propriétaire", "Locataire", "Hypothèque".'
        ),
        examples    = ["Locataire"],
    )]

    objet_pret: Annotated[str, Field(
        description = (
            'Motif du prêt : "Éducation", "Immobilier", '
            '"Personnel", "Automobile", "Médical".'
        ),
        examples    = ["Personnel"],
    )]

    type_pret: Annotated[str, Field(
        description = 'Nature du prêt : "Garanti" ou "Non garanti".',
        examples    = ["Non garanti"],
    )]

    # -- Exemple complet pour Swagger UI --------------------------------------
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary"             : "Client à risque modéré",
                    "age"                 : 35,
                    "revenu"              : 45000.0,
                    "montant_pret"        : 15000.0,
                    "duree_pret_mois"     : 48,
                    "jours_retard_moyen"  : 0.5,
                    "taux_incidents"      : 0.02,
                    "taux_utilisation_credit": 0.45,
                    "nb_comptes_ouverts"  : 3,
                    "type_residence"      : "Locataire",
                    "objet_pret"          : "Personnel",
                    "type_pret"           : "Non garanti",
                }
            ]
        }
    }
