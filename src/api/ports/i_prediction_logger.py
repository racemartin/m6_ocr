# =============================================================================
# PORT SECONDAIRE : INTERFACE DE JOURNALISATION DES PRÉDICTIONS
# Contrat abstrait pour persister chaque décision de scoring.
# Découple le use case de tout format de stockage concret.
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
from abc import ABC, abstractmethod  # Classe de base abstraite

# --- Entités du domaine -------------------------------------------------------
from src.api.domain.entities import DemandeCredit, DecisionCredit  # Données métier


# =============================================================================
# INTERFACE : JOURNALISEUR DE PRÉDICTIONS
# =============================================================================
class IJournaliseurPredictions(ABC):
    """
    Interface abstraite pour la journalisation des décisions de crédit.

    Chaque prédiction produite par le use case est transmise à cette
    interface pour être persistée. Le format de stockage (JSONL, CSV,
    base de données, cloud…) est entièrement délégué à l'adaptateur.

    Principe hexagonal
    ------------------
    Le use case ne connaît que cette interface. L'adaptateur concret
    (JSONLJournaliseurAdapter) est injecté depuis la couche API,
    ce qui permet de brancher n'importe quel backend de stockage
    sans modifier la logique métier.

    Utilisation pour le drift
    -------------------------
    Les données journalisées servent de données de production à
    Evidently AI pour détecter un drift par rapport au dataset
    de référence (reference_data.csv dans model_artifact/).

    Implémentations attendues
    -------------------------
    - JsonlJournaliseurAdapter : écriture ligne par ligne dans
      predictions.jsonl (format attendu par dashboard.py et
      drift_analysis.py)
    """

    # -------------------------------------------------------------------------
    @abstractmethod
    def journaliser(
        self,
        demande:  DemandeCredit,
        decision: DecisionCredit,
    ) -> None:
        """
        Persiste une paire (demande, décision) dans le journal.

        Paramètres
        ----------
        demande : DemandeCredit
            Entité d'entrée contenant toutes les features du client.
        decision : DecisionCredit
            Résultat du scoring : score, décision et latence.

        Lève
        ----
        IOError
            Si l'écriture dans le support de stockage échoue.
        """
        ...
