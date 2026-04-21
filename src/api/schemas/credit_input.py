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
    """
    Schéma de validation complet pour les données du simulateur.
    Chaque champ est synchronisé avec le payload de simulator.py.
    """

    # --- 1. Variables Critiques (Scores) ---
    ext_source_1: Annotated[float, Field(ge=0.01,  le=0.96, description="Score agence externe 1")]
    ext_source_2: Annotated[float, Field(ge=0.0,   le=0.85, description="Score agence externe 2")]
    ext_source_3: Annotated[float, Field(ge=0.0,   le=0.90, description="Score agence externe 3")]

    # --- 2. Données de Situation (Catégories) ---
    type_pret:      Annotated[str, Field(description="Nature du prêt (Cash/Revolving)")]
    objet_pret:     Annotated[str, Field(description="Accompagnement / Suite (name_type_suite)")]
    type_residence: Annotated[str, Field(description="Statut résidentiel (name_housing_type)")]
    code_gender:    Annotated[Literal["M", "F", "XNA"], Field(description="Genre du client")]
    education_type: Annotated[str, Field(description="Niveau d'études (name_education_type)")]

    # --- 3. Données Financières ---
    revenu:      Annotated[float, Field(gt=0,       description="Revenu annuel total déclaré")]
    amt_credit:  Annotated[float, Field(ge=45000,   description="Montant total du prêt demandé")]
    amt_annuity: Annotated[float, Field(ge=1600,    description="Annuité du prêt")]
    goods_price: Annotated[float, Field(ge=40000,   description="Prix du bien immobilier")]

    # --- 4. Historique et Comportement (Cercle Social inclus) ---
    paymnt_ratio_mean: Annotated[float, Field(ge=0.0, le=1.0, description="Ratio de paiement moyen")]
    paymnt_delay_mean: Annotated[float, Field(ge=0.0,         description="Délai moyen de paiement en jours")]
    max_dpd:           Annotated[float, Field(ge=0.0,         description="Retard maximal constaté (DPD)")]

    # --- 5. Profil Démographique ---
    age:            Annotated[int, Field(ge=18, le=70, description="Âge du client en années")]
    years_employed: Annotated[int, Field(ge=0,  le=50, description="Ancienneté professionnelle en années")]

    # --- 6. Autres Données (Bureau & Technique) ---
    bureau_credit_total: Annotated[float, Field(ge=0, description="Total crédits actifs au Bureau")]
    bureau_debt_mean:    Annotated[float, Field(description="Dette moyenne au Bureau")]
    pos_months_mean:     Annotated[float, Field(description="Moyenne des mois de balance POS")]
    cc_drawings_mean:    Annotated[float, Field(description="Retraits CB moyens")]
    cc_balance_mean:     Annotated[float, Field(description="Solde CB moyen")]
    phone_change_days:   Annotated[float, Field(description="Jours depuis le dernier changement de téléphone")]
    region_rating:       Annotated[int,   Field(ge=1, le=3, description="Note de la région (1-3)")]

    # --- Configuration pour documentation API ---
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ext_source_1": 0.5,
                "ext_source_2": 0.5,
                "ext_source_3": 0.5,
                "type_pret": "Cash loans",
                "objet_pret": "Unaccompanied",
                "type_residence": "House / apartment",
                "code_gender": "F",
                "education_type": "Higher education",
                "revenu": 50000.0,
                "amt_credit": 500000.0,
                "amt_annuity": 25000.0,
                "goods_price": 450000.0,
                "paymnt_ratio_mean": 0.1,
                "paymnt_delay_mean": 2.0,
                "max_dpd": 0.0,
                "age": 35,
                "years_employed": 10,
                "bureau_credit_total": 5.0,
                "bureau_debt_mean": 1000.0,
                "pos_months_mean": 12.0,
                "cc_drawings_mean": 0.0,
                "cc_balance_mean": 0.0,
                "phone_change_days": 365.0,
                "region_rating": 2
            }]
        }
    }
