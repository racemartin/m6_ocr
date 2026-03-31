# =============================================================================
# PORT PRIMAIRE : INTERFACE DE SCORING CRÉDIT
# Contrat abstrait que tout adaptateur de modèle doit implémenter.
# Le use case ne dépend que de cette interface — jamais des adaptateurs.
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
from abc import ABC, abstractmethod  # Classe de base abstraite

# --- Entités du domaine -------------------------------------------------------
from src.api.domain.entities import DemandeCredit, DecisionCredit  # Contrat métier


# =============================================================================
# INTERFACE : SCOREUR DE CRÉDIT
# =============================================================================
class ICreditScorer(ABC):
    """
    Interface abstraite pour tout moteur de scoring crédit.

    Définit le contrat entre la couche Application (use cases)
    et les adaptateurs concrets (ONNX, MLflow, mock de test…).

    Principe hexagonal
    ------------------
    Le use case EvaluerDemandeCreditUseCase reçoit une instance de cette
    interface par injection de dépendance. Il ne sait jamais si le
    modèle sous-jacent est ONNX, MLflow ou un simulateur de test.
    Cela permet de changer d'implémentation sans toucher au domaine.

    Implémentations attendues
    -------------------------
    - OnnxScoeurAdapter   : inférence via onnxruntime (production)
    - MlflowScoeurAdapter : inférence via mlflow.pyfunc (dev local)
    """

    # -------------------------------------------------------------------------
    @abstractmethod
    def predire(self, demande: DemandeCredit) -> DecisionCredit:
        """
        Effectue le scoring d'une demande de crédit.

        Paramètres
        ----------
        demande : DemandeCredit
            Entité contenant les 11 features du client à évaluer.

        Retourne
        --------
        DecisionCredit
            Résultat enrichi : score, décision et latence mesurée.

        Lève
        ----
        RuntimeError
            Si le modèle n'est pas chargé ou si l'inférence échoue.
        """
        ...

    # -------------------------------------------------------------------------
    @abstractmethod
    def est_pret(self) -> bool:
        """
        Vérifie que le modèle est chargé et opérationnel.

        Retourne
        --------
        bool
            True si le modèle est prêt à recevoir des requêtes.
        """
        ...
