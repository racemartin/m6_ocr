# =============================================================================
# src/api/routers/predict.py — Route POST /predict
# Point d'entrée principal : reçoit une demande de crédit JSON,
# orchestre l'évaluation via le use case et retourne la décision.
#
# Flux complet d'une requête :
#   JSON → ClientDataInput (Pydantic) → DemandeCredit (domaine)
#        → EvaluerDemandeCreditUseCase.executer()
#        → DecisionCredit (domaine) → ClientDataOutput (Pydantic) → JSON
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import logging                                        # Journalisation requêtes

# --- Bibliothèques tierces : API ---------------------------------------------
from   fastapi import APIRouter, Depends, HTTPException  # Router, injection

# --- Couche API : schémas ----------------------------------------------------
from src.api.schemas.credit_input  import ClientDataInput   # Validation entrée
from src.api.schemas.credit_output import ClientDataOutput  # Réponse typée

# --- Couche API : dépendances ------------------------------------------------
from src.api.dependencies import obtenir_use_case           # Use case injecté

# --- Application (use case) --------------------------------------------------
from src.api.application.evaluate_credit_use_case import (
    EvaluerDemandeCreditUseCase,                            # Orchestrateur métier
)

# --- Domaine -----------------------------------------------------------------
from src.api.domain.entities import DemandeCredit             # Entité domaine
import os
import inspect
from src.tools.rafael.log_tool import LogTool

# Journalisation du module
journal = logging.getLogger(__name__)

# Instance du router FastAPI pour ce groupe de routes
routeur     = APIRouter()
log         = LogTool(origin="router ")
NOM_FICHIER = os.path.basename(__file__)

# ##############################################################################
# Route : POST /predict
# ##############################################################################

# =============================================================================
@routeur.post(
    "/predict",
    response_model = ClientDataOutput,
    summary        = "Évaluation d'une demande de crédit",
    description    = (
        "Reçoit les données d'un demandeur de crédit et retourne "
        "une décision basée sur le modèle LightGBM/ONNX entraîné.\n\n"
        "**Seuil de décision** : configurable via `SEUIL_DECISION` "
        "(défaut : `0.35`).\n\n"
        "| Condition                      | Décision   |\n"
        "|-------------------------------|------------|\n"
        "| probabilité < seuil           | Approuvé   |\n"
        "| probabilité ≥ seuil           | Rejeté     |"
    ),
    tags           = ["Scoring"],
    status_code    = 200,
)
def evaluer_credit(
    donnees_entree : ClientDataInput,
    use_case       : EvaluerDemandeCreditUseCase = Depends(obtenir_use_case),
) -> ClientDataOutput:
    """
    Évalue une demande de crédit et retourne la décision.

    Args:
        donnees_entree : Corps JSON validé par Pydantic (11 features).
        use_case       : Use case injecté avec scorer + logger.

    Returns:
        ClientDataOutput avec probabilité, score, décision et latence.

    Raises:
        HTTPException 422 : Données invalides (validation Pydantic auto).
        HTTPException 500 : Erreur interne du moteur de scoring.
    """
    log.START_ACTION(NOM_FICHIER, inspect.currentframe().f_code.co_name , "POST /predict BEGING")
    log.DEBUG_PARAMETER_VALUE("age", donnees_entree.age)
    log.DEBUG_PARAMETER_VALUE("ext_source_2", donnees_entree.ext_source_2)
    log.DEBUG_PARAMETER_VALUE("amt_credit", donnees_entree.amt_credit)
    log.DEBUG_PARAMETER_VALUE("education", donnees_entree.education_type)

    # -- Conversion schéma API → entité domaine ------------------------------
    demande = _schema_vers_entite(donnees_entree)

    # -- Exécution du use case -----------------------------------------------
    try:
        decision = use_case.executer(demande)

    except ValueError as erreur:
        journal.warning("Données invalides pour le scoring : %s", erreur)
        raise HTTPException(
            status_code = 422,
            detail      = f"Données de scoring invalides : {erreur}",
        )
    except RuntimeError as erreur:
        journal.error("Erreur d'inférence : %s", erreur)
        raise HTTPException(
            status_code = 500,
            detail      = "Erreur interne du moteur de scoring.",
        )

    log.LEVEL_7_INFO(NOM_FICHIER, "Réponse /predict")
    log.DEBUG_PARAMETER_VALUE("id_demande"        , decision.id_demande )
    log.DEBUG_PARAMETER_VALUE("decision.value"    , decision.decision.value)
    log.DEBUG_PARAMETER_VALUE("score.valeur"      , decision.score.valeur)
    log.DEBUG_PARAMETER_VALUE("latence_ms"       , decision.latence_ms)

    # -- Conversion entité domaine → schéma de réponse ----------------------
    # -- Conversion entité domaine (lógica) → schéma de réponse (JSON) -------
    
    log.FINISH_ACTION(NOM_FICHIER, inspect.currentframe().f_code.co_name , "FINISH")
    
    return ClientDataOutput(
        id_demande         = str(decision.id_demande),

        # Ambos campos del JSON beben de la misma fuente en la entidad:
        probabilite_defaut = decision.score.valeur,   # Mapeo correcto
        score_risque       = decision.score.valeur,   # O el nombre que use tu esquema

        # La decisión se envía como texto ("Refusé") para el JSON
        decision           = decision.decision.value, # "Approuvé" o "Refusé" (string)

        latence_ms         = decision.latence_ms,
    
        seuil_utilise      = decision.seuil_utilise,
       
        # MAPEAMOS la lista de objetos de dominio a diccionarios compatibles con Pydantic
        explication_shap   = [
            {
                "feature":       exp.nom_feature,
                "valeur_client": str(exp.valeur_originale),
                "impact_shap":   float(exp.impact_shap),
                "direction":     exp.direction,
                "explication":   getattr(exp, 'explication', None)
            }
            for exp in decision.explications_shap
        ]
    )

# ##############################################################################
# Fonctions privées
# ##############################################################################

# =============================================================================
# EN: src/api/routers/predict.py
# =============================================================================
def _schema_vers_entite(schema: ClientDataInput) -> DemandeCredit:
    """
    Convertit un schéma Pydantic en entité domaine DemandeCredit.
    Mise à jour : Utilisation des 20 features top SHAP.
    """
    return DemandeCredit(
        # 1-3: Scores externes
        ext_source_1             = schema.ext_source_1,
        ext_source_2             = schema.ext_source_2,
        ext_source_3             = schema.ext_source_3,

        # 4-8: Comportement et âge
        paymnt_ratio_mean        = schema.paymnt_ratio_mean,
        age                      = schema.age,
        cc_drawings_mean         = schema.cc_drawings_mean,
        paymnt_delay_mean        = schema.paymnt_delay_mean,
        pos_months_mean          = schema.pos_months_mean,

        # 9-11: Prix et Catégories
        goods_price              = schema.goods_price,
        education_type           = schema.education_type,
        code_gender              = schema.code_gender,

        # 12-16: Crédit et Bureau
        bureau_credit_total      = schema.bureau_credit_total,
        max_dpd                  = schema.max_dpd,
        amt_credit               = schema.amt_credit,
        amt_annuity              = schema.amt_annuity,
        cc_balance_mean          = schema.cc_balance_mean,

        # 17-20: Emploi, Téléphone et Région
        years_employed           = schema.years_employed,
        phone_change_days        = schema.phone_change_days,
        region_rating            = schema.region_rating,
        bureau_debt_mean         = schema.bureau_debt_mean
    )
