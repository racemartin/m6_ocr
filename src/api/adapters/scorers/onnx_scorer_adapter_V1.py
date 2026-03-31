# =============================================================================
# ADAPTATEUR ONNX — Implémentation du port ICreditScorer
# Charge le modèle best_model.onnx depuis model_artifact/ et effectue
# l'inférence via onnxruntime. Utilisé en production et sur HuggingFace.
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import json      # Lecture du fichier de métadonnées du modèle
import logging   # Journalisation des événements de l'adaptateur
import time      # Mesure de la latence d'inférence
import joblib   # Para cargar el .pkl

# --- Bibliothèques scientifiques ----------------------------------------------
import numpy as np  # Préparation du tableau de features pour ONNX
import pandas as pd

# --- Runtime ONNX (inférence rapide, sans dépendance sklearn) ----------------
import onnxruntime as ort  # Moteur d'inférence ONNX multiplateforme

# --- Domaine et port ----------------------------------------------------------
from src.api.domain.entities      import DemandeCredit, DecisionCredit  # Contrat
from src.api.domain.value_objects import Decision, ScoreRisque           # Concepts
from src.api.ports.i_credit_scorer import ICreditScorer                  # Interface

# --- Configuration globale ----------------------------------------------------
from config import parametres  # Chemin modèle et seuil de décision

# Configuration du logger applicatif pour ce module
journalapp = logging.getLogger(__name__)


# =============================================================================
# ADAPTATEUR : SCOREUR ONNX
# =============================================================================
class OnnxScoeurAdapter(ICreditScorer):
    """
    Adaptateur d'inférence basé sur onnxruntime.

    Chargement unique au démarrage de l'application (pattern singleton
    géré par FastAPI lifespan) pour éviter le rechargement à chaque
    requête et maintenir une latence inférieure à 10 ms.

    Chemin du modèle
    ----------------
    model_artifact/best_model.onnx  (non ignoré par .gitignore)
    Le fichier est exporté depuis MLflow par scripts/export_best_model.py.

    Prétraitement embarqué
    ----------------------
    Le pipeline sklearn (encodage + imputation) doit être intégré dans
    le fichier ONNX via skl2onnx. Le tableau de features brutes est
    passé directement sans transformation Python supplémentaire.
    """

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialise l'adaptateur sans charger le modèle immédiatement."""
        self._session:        ort.InferenceSession | None = None
        self._preprocesador:   any                         = None # Para el preprocessor.pkl
        self._nom_entree:      str | None                  = None
        self._seuil:           float                       = parametres.seuil_decision

    def _charger_ressources(self) -> None:
        """Charge le modèle ONNX et le préprocesseur s'ils ne le sont pas déjà."""
        if self._session is None:
            # Carga del modelo ONNX
            self._session = ort.InferenceSession(parametres.chemin_modele_onnx)
            self._nom_entree = self._session.get_inputs()[0].name

        if self._preprocesador is None:
            # Carga del preprocesador .pkl
            # Asegúrate de añadir 'chemin_preprocesador' en tu config/parametres
            self._preprocesador = joblib.load(parametres.chemin_preprocesador)

    # =========================================================================
    # CHARGEMENT DU MODÈLE
    # =========================================================================
    def charger(self) -> None:
        """
        Charge la session ONNX depuis le fichier model_artifact/best_model.onnx.

        Appelé une seule fois dans le lifespan FastAPI au démarrage.
        Configure les providers dans l'ordre de priorité :
        CUDA (GPU si disponible) → CPU.

        Lève
        ----
        FileNotFoundError
            Si le fichier .onnx est absent de model_artifact/.
        RuntimeError
            Si onnxruntime ne parvient pas à initialiser la session.
        """
        chemin_modele = parametres.chemin_modele_onnx

        journalapp.info(
            "Chargement du modèle ONNX depuis : %s", chemin_modele
        )

        # -- Sélection automatique du provider (GPU si disponible) ------------
        providers_disponibles = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers_disponibles
            else ["CPUExecutionProvider"]
        )

        self._session    = ort.InferenceSession(
            str(chemin_modele), providers=providers
        )
        self._nom_entree = self._session.get_inputs()[0].name

        journalapp.info(
            "Modèle ONNX chargé — provider : %s", providers[0]
        )

        # -- Lecture des métadonnées pour traçabilité -------------------------
        chemin_meta = parametres.chemin_meta_modele
        if chemin_meta.exists():
            with open(chemin_meta, encoding="utf-8") as f:
                meta = json.load(f)
            journalapp.info(
                "Métadonnées modèle : run_id=%s, auc=%.4f",
                meta.get("run_id", "inconnu"),
                meta.get("auc",    0.0),
            )

        # -- Carga del preprocesador scikit-learn --
        try:
            path_preproc = parametres.chemin_preprocesador
            self._preprocesador = joblib.load(path_preproc)
            journalapp.info("Préprocesseur chargé depuis : %s", path_preproc)
        except Exception as e:
            journalapp.error("Échec du chargement du préprocesseur : %s", e)
            raise

    # =========================================================================
    # VÉRIFICATION DE DISPONIBILITÉ
    # =========================================================================
    def est_pret(self) -> bool:
        """Retourne True si la session ONNX est initialisée."""
        return self._session is not None

    # =========================================================================
    # INFÉRENCE PRINCIPALE
    # =========================================================================
    def predire(self, demande: DemandeCredit) -> DecisionCredit:
        """
        Effectue l'inférence ONNX pour une demande de crédit.

        Séquence interne
        ----------------
        1. Convertir les features en tableau numpy float32.
        2. Lancer l'inférence via onnxruntime (mesure latence).
        3. Extraire la probabilité de défaut (classe 1).
        4. Appliquer le seuil métier pour obtenir la décision.
        5. Retourner une DecisionCredit immuable.

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
            Si la session ONNX n'est pas initialisée.
        """
        if not self.est_pret():
            raise RuntimeError(
                "OnnxScoeurAdapter : session non initialisée. "
                "Appelez charger() avant d'effectuer des prédictions."
            )

        # -- Préparation du tableau de features -------------------------------
        features_brutes = demande.vers_tableau_features()
        tableau_input   = np.array(
            [features_brutes], dtype=np.float32  # Shape : (1, 11)
        )

        # -- Inférence et mesure de latence -----------------------------------
        debut_ms = time.perf_counter()
        resultats = self._session.run(
            None,                                        # Tous les outputs
            {self._nom_entree: tableau_input},           # Mapping nom→valeur
        )
        latence_ms = (time.perf_counter() - debut_ms) * 1000.0

        # -- Extraction de la probabilité de défaut (classe 1) ----------------
        # resultats[1] est le dict de probabilités : [{0: p0, 1: p1}, ...]
        probabilite_defaut = float(resultats[1][0][1])

        # -- Application du seuil métier --------------------------------------
        score    = ScoreRisque(probabilite_defaut)
        decision = score.vers_decision(self._seuil)

        journalapp.debug(
            "ONNX inférence — proba=%.4f, décision=%s, latence=%.1f ms",
            probabilite_defaut, decision.value, latence_ms,
        )

        return DecisionCredit(
            id_demande = demande.id_demande,
            score      = score,
            decision   = decision,
            latence_ms = round(latence_ms, 2),
        )
