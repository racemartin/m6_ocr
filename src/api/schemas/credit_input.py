# =============================================================================
# SCHÉMA D'ENTRÉE — Validation Pydantic de la requête POST /predict
# Définit les 11 features attendues, leur type et leurs contraintes.
# Ce schéma documente automatiquement l'API Swagger (OpenAPI).
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
from typing import Annotated, Literal # Annotations avec métadonnées de validation

# --- Pydantic — Validation et documentation automatique ----------------------
from pydantic import BaseModel, Field  # Modèle de données avec contraintes


# =============================================================================
# SCHÉMA : DONNÉES D'ENTRÉE CLIENT
# =============================================================================


class ClientDataInput(BaseModel):
    # Variables críticas (Scores externos)
    ext_source_1: Annotated[float, Field(ge=0.01, le=0.96, examples=[0.5])]
    ext_source_2: Annotated[float, Field(ge=0.0, le=0.85, examples=[0.5])]
    ext_source_3: Annotated[float, Field(ge=0.0, le=0.90, examples=[0.5])]

    # Comportamiento de pagos
    paymnt_ratio_mean: Annotated[float, Field(ge=0.0, le=1.0)]
    paymnt_delay_mean: Annotated[float, Field(ge=0.0)]
    max_dpd: Annotated[float, Field(ge=0.0)]

    # Datos personales y demográficos
    age: Annotated[int, Field(ge=18, le=70)]
    years_employed: Annotated[int, Field(ge=0, le=50)]
    code_gender: Annotated[Literal["M", "F", "XNA"], Field(examples=["F"])]
    education_type: Annotated[Literal[
        "Secondary / secondary special", "Higher education",
        "Incomplete higher", "Lower secondary", "Academic degree"
    ], Field(examples=["Higher education"])]

    # Financieros
    amt_credit: Annotated[float, Field(ge=45000)]
    amt_annuity: Annotated[float, Field(ge=1600)]
    goods_price: Annotated[float, Field(ge=40000)]

    # Buró y otros
    bureau_credit_total: float
    bureau_debt_mean: float
    pos_months_mean: float
    cc_drawings_mean: float
    cc_balance_mean: float
    phone_change_days: float
    region_rating: Annotated[int, Field(ge=1, le=3)]

class ClientDataInput_V1(BaseModel):
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
