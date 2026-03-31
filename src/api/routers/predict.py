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

# Journalisation du module
journal = logging.getLogger(__name__)

# Instance du router FastAPI pour ce groupe de routes
routeur = APIRouter()


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
    journal.info("============================================================")
    journal.info(
        "POST /predict | age=%d | revenu=%.0f € | montant=%.0f €",
        donnees_entree.age,
        donnees_entree.revenu,
        donnees_entree.montant_pret,
    )

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

    journal.info("------------------------------------------------------------")
    journal.info(
        "Réponse /predict | id=%s | décision=%s | prob=%.4f | %.1f ms",
        decision.id_demande,
        decision.decision.value,  # .value porque es un Enum
        decision.score.valeur,    # .score.valeur para entrar al float del objeto
        decision.latence_ms,
    )
    journal.info("------------------------------------------------------------")

    # -- Conversion entité domaine → schéma de réponse ----------------------
    # -- Conversion entité domaine (lógica) → schéma de réponse (JSON) -------
    return ClientDataOutput(
        id_demande         = str(decision.id_demande),

        # Ambos campos del JSON beben de la misma fuente en la entidad:
        probabilite_defaut = decision.score.valeur,   # Mapeo correcto
        score_risque       = decision.score.valeur,   # O el nombre que use tu esquema

        # La decisión se envía como texto ("Refusé") para el JSON
        decision           = decision.decision.value, # "Approuvé" o "Refusé" (string)

        latence_ms         = decision.latence_ms,
    
        seuil_utilise      = decision.seuil_utilise,
       
        explications_shap  = decision.explications_shap
    )

# ##############################################################################
# Fonctions privées
# ##############################################################################

# =============================================================================
def _schema_vers_entite(schema: ClientDataInput) -> DemandeCredit:
    """
    Convertit un schéma Pydantic en entité domaine DemandeCredit.

    Isole la couche API du domaine : DemandeCredit ne dépend pas
    de Pydantic, et ClientDataInput ne dépend pas du domaine.

    Args:
        schema : Données validées depuis le corps JSON de la requête.

    Returns:
        DemandeCredit prête à être traitée par le use case.
    """
    return DemandeCredit(
        age                      = schema.age,
        revenu                  = schema.revenu,
        montant_pret             = schema.montant_pret,
        duree_pret_mois          = schema.duree_pret_mois,
        jours_retard_moyen             = schema.jours_retard_moyen,
        taux_incidents              = schema.taux_incidents,
        taux_utilisation_credit  = schema.taux_utilisation_credit,
        nb_comptes_ouverts       = schema.nb_comptes_ouverts,
        type_residence           = schema.type_residence,
        objet_pret               = schema.objet_pret,
        type_pret                = schema.type_pret,
    )
