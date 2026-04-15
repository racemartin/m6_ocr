# =============================================================================
# CONFIGURATION GLOBALE — Paramètres de l'application
# Centralise tous les réglages en un seul endroit via pydantic-settings.
# Les valeurs sont lues depuis les variables d'environnement ou le .env.
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
from pathlib import Path  # Chemins indépendants du système d'exploitation

# --- Pydantic settings (lecture .env et variables d'environnement) -----------
# --- IMPORTANTE: Configuración de Pydantic para eliminar el Warning ---
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

# =============================================================================
# CLASSE DE CONFIGURATION PRINCIPALE
# =============================================================================
class Paramètres(BaseSettings):
    """
    Paramètres globaux de l'application de scoring crédit.

    Priorité de résolution (de la plus haute à la plus basse) :
    1. Variables d'environnement du système (export MODEL_BACKEND=onnx)
    2. Fichier .env à la racine du projet
    3. Valeurs par défaut définies ici

    Variables clés
    --------------
    MODEL_BACKEND : str
        "onnx"   → OnnxScoeurAdapter    (production, HuggingFace)
        "mlflow" → MlflowScoeurAdapter  (développement local)
    SEUIL_DECISION : float
        Seuil de probabilité de défaut pour la décision binaire.
        0.35 = si proba_défaut ≥ 0.35 → crédit refusé.
    """

    # Esto le dice a Pydantic que ignore el prefijo "model_" como protegido
    # Esta es la forma oficial y única necesaria en Pydantic V2:
    model_config = ConfigDict(protected_namespaces=('settings_',))

    # -- Sélection du backend de scoring --------------------------------------
    model_backend: str = "onnx"  # "onnx" | "mlflow" — voir dependencies.py

    # -- Seuil de décision métier ---------------------------------------------
    seuil_decision: float = 0.35  # Seuil optimisé en phase 4 (m6_ocr)

    # -- Chemins des artefacts du modèle --------------------------------------
    chemin_modele_onnx: Path = Path(
        "model_artifact/best_model.onnx"         # Modèle ONNX exporté
    )
    chemin_meta_modele: Path = Path(
        "model_artifact/best_model_meta.json"    # Méta : run_id, AUC, seuil
    )
    chemin_reference_drift: Path = Path(
        "model_artifact/reference_data.csv"      # Dataset d'entraînement ref.
    )

    # -- Chemin du journal de prédictions -------------------------------------
    chemin_predictions_jsonl: Path = Path(
        "predictions.jsonl"                      # Journal de production
    )

    # -- Chemin du rapport de drift -------------------------------------------
    chemin_rapport_drift: Path = Path(
        "monitoring/drift_report.html"           # Rapport Evidently AI
    )

    # -- Configuration MLflow (uniquement pour le backend "mlflow") -----------
    mlflow_tracking_uri: str = (
        "http://mlflow:5000"                     # Serveur MLflow Docker
    )
    mlflow_model_uri: str = (
        "models:/credit_scorer/Production"       # URI du modèle enregistré
    )

    # -- Configuration de l'API FastAPI ---------------------------------------
    titre_api:       str = "API Scoring Crédit — Prêt à Dépenser"
    version_api:     str = "2.0.0"
    description_api: str = (
        "API de scoring crédit basée sur un modèle LightGBM optimisé "
        "(ONNX) — Architecture hexagonale — MLOps Partie 2/2"
    )

    # -- Logging ---------------------------------------------------------------
    niveau_log: str = "INFO"  # DEBUG | INFO | WARNING | ERROR

    # -- Pydantic : lecture du fichier .env -----------------------------------
    model_config = {
        "env_file"          : ".env",
        "env_file_encoding" : "utf-8",
        "case_sensitive"    : False,   # MODEL_BACKEND == model_backend
        "extra"             : "ignore"
    }


# =============================================================================
# INSTANCE GLOBALE — importée par tous les modules
# =============================================================================
parametres = Paramètres()

RACINE_PROJET        = Path(__file__).parent.resolve()
DOSSIER_ARTEFACT     = RACINE_PROJET / "model_artifact"
DOSSIER_ARTEFACT.mkdir(parents=True, exist_ok=True)

FICHIER_META_MODELE   = parametres.chemin_meta_modele
FICHIER_MODELE_ONNX   = parametres.chemin_modele_onnx
FICHIER_DONNEES_REF   = parametres.chemin_reference_drift

FICHIER_PREDICTIONS   = parametres.chemin_predictions_jsonl
FICHIER_RAPPORT_DRIFT = parametres.chemin_rapport_drift
