# =============================================================================
# CAS D'UTILISATION : ÉVALUATION D'UNE DEMANDE DE CRÉDIT
# Orchestre le scoring et la journalisation sans connaître les détails
# techniques des adaptateurs (modèle ONNX, fichier JSONL, etc.).
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import logging  # Journalisation applicative interne


# --- Entités du domaine -------------------------------------------------------
from src.api.domain.entities import DemandeCredit, DecisionCredit  # Agrégats

# --- Ports (interfaces abstraites) --------------------------------------------
from src.api.ports.i_credit_scorer     import ICreditScorer            # Modèle
from src.api.ports.i_prediction_logger import IJournaliseurPredictions  # Logging

# --- Configuration globale ----------------------------------------------------
from config import parametres  # Seuil de décision et autres réglages

# Configuration du logger applicatif pour ce module
journalapp = logging.getLogger(__name__)

import os
import inspect
from src.tools.rafael.log_tool import LogTool
log = LogTool(origin="UseCase")

def get_filename(self) -> str:
    """Devuelve el nombre del archivo actual con su extensión."""
    return os.path.basename(__file__)
 
# =============================================================================
# CAS D'UTILISATION : EvaluerDemandeCreditUseCase
# =============================================================================
class EvaluerDemandeCreditUseCase:
    """
    Cas d'utilisation principal de l'application de scoring crédit.

    Responsabilités
    ---------------
    1. Déléguer le scoring au port ICreditScorer (modèle ONNX ou MLflow).
    2. Déléguer la journalisation au port IJournaliseurPredictions (JSONL).
    3. Ne contenir aucune logique technique (pas d'import sklearn, onnx…).

    Principe SRP (Single Responsibility Principle)
    -----------------------------------------------
    Ce use case orchestre uniquement — il ne prétraite pas les données
    et ne sait pas comment le modèle charge ses poids. Ces responsabilités
    appartiennent aux adaptateurs injectés.

    Paramètres du constructeur
    --------------------------
    scoreur : ICreditScorer
        Adaptateur concret chargé de l'inférence du modèle.
    journaliseur : IJournaliseurPredictions
        Adaptateur concret chargé de l'écriture dans predictions.jsonl.
    """

    # -------------------------------------------------------------------------
    def __init__(
        self,
        scoreur:      ICreditScorer,
        journaliseur: IJournaliseurPredictions,
    ) -> None:
        """Injecte les deux dépendances via leurs interfaces abstraites."""
        self._scoreur      = scoreur       # Moteur d'inférence (ONNX/MLflow)
        self._journaliseur = journaliseur  # Backend de persistance (JSONL)

    # =========================================================================
    # MÉTHODE PRINCIPALE : EXÉCUTER LE SCORING
    # =========================================================================
    def executer(self, demande: DemandeCredit) -> DecisionCredit:
        """
        Évalue une demande de crédit et persiste le résultat.

        Séquence d'exécution
        --------------------
        1. Vérifier que le scoreur est opérationnel.
        2. Appeler le scoreur pour obtenir score + décision + latence.
        3. Journaliser la paire (demande, décision) via le journaliseur.
        4. Retourner la décision au routeur FastAPI.

        Paramètres
        ----------
        demande : DemandeCredit
            Entité entrante, construite depuis le schéma Pydantic de l'API.

        Retourne
        --------
        DecisionCredit
            Résultat complet : score, décision binaire et latence.

        Lève
        ----
        RuntimeError
            Si le scoreur signale qu'il n'est pas prêt.
        """
        
        log.START_ACTION(self.__class__.__name__, inspect.currentframe().f_code.co_name , "BEGING")
        
        # -- Vérification disponibilité du modèle -----------------------------
        if not self._scoreur.est_pret:
            raise RuntimeError(
                "Le scoreur n'est pas initialisé. "
                "Vérifiez que le modèle ONNX a bien été chargé au démarrage."
            )

        log.DEBUG_PARAMETER_VALUE("Évaluation de la demande" , demande.id_demande)
        
        log.STEP(2, "Inference" , "ONNX ou MLflow selon l'environnement")

        # -- Inférence via le port (ONNX ou MLflow selon l'environnement) -----
        decision = self._scoreur.predire(demande)

        log.DEBUG_PARAMETER_VALUE("probabilite_defaut" , demande.id_demande)
        log.DEBUG_PARAMETER_VALUE("decision.value"     , decision.decision.value )
        log.DEBUG_PARAMETER_VALUE("score.valeur"       , f"{decision.score.valeur:.4f}")
        log.DEBUG_PARAMETER_VALUE("latence_ms"         , f"{decision.latence_ms:.2f}")
        
        # -- Journalisation pour le monitoring du drift -----------------------
        log.STEP(2, "Journalisation" , "Pour le monitoring du drift")
        try:
            self._journaliseur.journaliser(demande, decision)
        except IOError as erreur:
            # On loggue l'erreur mais on ne bloque pas la réponse client
            journalapp.warning(
                "Journalisation échouée pour id=%s : %s",
                demande.id_demande,
                erreur,
            )

        log.FINISH_ACTION(self.__class__.__name__, inspect.currentframe().f_code.co_name , "FINISH")
        return decision
