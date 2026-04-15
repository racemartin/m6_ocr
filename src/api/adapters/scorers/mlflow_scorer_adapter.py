# =============================================================================
# ADAPTATEUR MLFLOW — Implémentation du port ICreditScorer
# Charge le modèle depuis le registre MLflow local (dev uniquement).
# NE PAS utiliser en production ou sur HuggingFace : MLflow n'y est pas.
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import logging  # Journalisation des événements de l'adaptateur
import time     # Mesure de la latence d'inférence

# --- Bibliothèques scientifiques ----------------------------------------------
import pandas as pd  # MLflow pyfunc attend un DataFrame comme entrée

# --- MLflow — Accès au registre de modèles -----------------------------------
import mlflow.pyfunc  # Chargement universel de modèles MLflow

# --- Domaine et port ----------------------------------------------------------
from src.api.domain.entities      import DemandeCredit, DecisionCredit  # Contrat
from src.api.domain.value_objects import ScoreRisque           # Concepts
from src.api.ports.i_credit_scorer import ICreditScorer                  # Interface

# --- Configuration globale ----------------------------------------------------
from config import parametres  # URI MLflow et seuil de décision

# Configuration du logger applicatif pour ce module
journalapp = logging.getLogger(__name__)


# =============================================================================
# ADAPTATEUR : SCOREUR MLFLOW
# =============================================================================
class MlflowScoeurAdapter(ICreditScorer):
    """
    Adaptateur d'inférence basé sur mlflow.pyfunc.

    Utilisé uniquement en environnement de développement local,
    lorsque le serveur MLflow est disponible sur le réseau Docker.

    En production (HuggingFace), l'adaptateur OnnxScoeurAdapter
    est sélectionné à la place via la variable MODEL_BACKEND.

    URI du modèle
    -------------
    Exemples d'URI MLflow acceptées :
    - "runs:/<run_id>/model"
    - "models:/credit_scorer/Production"
    - "mlflow-artifacts:/..."
    """

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialise l'adaptateur sans charger le modèle immédiatement."""
        self._modele: object | None = None  # Modèle pyfunc chargé
        self._seuil:  float         = parametres.seuil_decision

    # =========================================================================
    # CHARGEMENT DU MODÈLE
    # =========================================================================
    def charger(self) -> None:
        """
        Charge le modèle pyfunc depuis le registre MLflow.

        Configure l'URI de tracking puis charge le modèle identifié
        par MODEL_URI dans la configuration.

        Lève
        ----
        mlflow.exceptions.MlflowException
            Si le run_id est invalide ou le modèle introuvable.
        """
        mlflow.set_tracking_uri(parametres.mlflow_tracking_uri)
        uri_modele = parametres.mlflow_model_uri

        journalapp.info(
            "Chargement du modèle MLflow depuis : %s", uri_modele
        )

        self._modele = mlflow.pyfunc.load_model(uri_modele)

        journalapp.info("Modèle MLflow chargé avec succès.")

    # =========================================================================
    # VÉRIFICATION DE DISPONIBILITÉ
    # =========================================================================
    def est_pret(self) -> bool:
        """Retourne True si le modèle pyfunc est chargé."""
        return self._modele is not None

    # =========================================================================
    # INFÉRENCE PRINCIPALE
    # =========================================================================
    def predire(self, demande: DemandeCredit) -> DecisionCredit:
        """
        Effectue l'inférence MLflow pour une demande de crédit.

        MLflow pyfunc attend un DataFrame avec les noms de colonnes
        correspondant exactement à ceux du pipeline d'entraînement.

        Paramètres
        ----------
        demande : DemandeCredit
            Entité avec les 11 features du client.

        Retourne
        --------
        DecisionCredit
            Score, décision et latence d'inférence.

        Lève
        ----
        RuntimeError
            Si le modèle pyfunc n'est pas initialisé.
        """
        if not self.est_pret():
            raise RuntimeError(
                "MlflowScoeurAdapter : modèle non initialisé. "
                "Appelez charger() avant d'effectuer des prédictions."
            )

        # -- Construction du DataFrame d'entrée attendu par pyfunc -----------
        noms_colonnes = [
            "age",
            "revenu",
            "montant_pret",
            "duree_pret_mois",
            "jours_retard_moyen",
            "taux_incidents",
            "taux_utilisation_credit",
            "nb_comptes_ouverts",
            "type_residence",
            "objet_pret",
            "type_pret",
        ]
        df_input = pd.DataFrame(
            [demande.vers_tableau_features()],
            columns=noms_colonnes,
        )

        # -- Inférence et mesure de latence -----------------------------------
        debut_ms  = time.perf_counter()
        resultats = self._modele.predict(df_input)
        latence_ms = (time.perf_counter() - debut_ms) * 1000.0

        # -- Extraction de la probabilité de la colonne "proba_defaut" --------
        if hasattr(resultats, "iloc"):
            # Le modèle retourne un DataFrame avec probabilités par classe
            probabilite_defaut = float(resultats.iloc[0]["proba_defaut"])
        else:
            # Tableau numpy : colonne 1 = probabilité de défaut
            probabilite_defaut = float(resultats[0][1])

        # -- Application du seuil métier --------------------------------------
        score    = ScoreRisque(probabilite_defaut)
        decision = score.vers_decision(self._seuil)

        journalapp.debug(
            "MLflow inférence — proba=%.4f, décision=%s, latence=%.1f ms",
            probabilite_defaut, decision.value, latence_ms,
        )

        return DecisionCredit(
            id_demande = demande.id_demande,
            score      = score,
            decision   = decision,
            latence_ms = round(latence_ms, 2),
        )
