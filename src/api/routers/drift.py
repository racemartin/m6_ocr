# =============================================================================
# src/api/routers/drift.py — Route GET /drift/report
# Endpoint de supervision du drift des données.
# Lance le script drift_analysis.py en sous-processus et retourne
# un résumé JSON du drift détecté sur les prédictions récentes.
#
# Le drift est analysé en batch (pas en temps réel par requête) :
#   - Manuellement via cet endpoint
#   - Ou via un cron job quotidien sur le serveur
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import logging                                        # Journalisation
import subprocess                                     # Lancement script drift
import sys                                            # Interpréteur Python
from   pathlib import Path                            # Chemins fichiers

# --- Bibliothèques tierces : API ---------------------------------------------
from   fastapi import APIRouter, HTTPException        # Router et erreurs HTTP

# --- Configuration -----------------------------------------------------------
from config import RACINE_PROJET, FICHIER_PREDICTIONS  # Chemins


# Journalisation du module
journal = logging.getLogger(__name__)

# Instance du router FastAPI
routeur = APIRouter()

# Chemin absolu du script d'analyse drift
SCRIPT_DRIFT = RACINE_PROJET / "scripts" / "drift_analysis.py"


# ##############################################################################
# Route : GET /drift/report
# ##############################################################################

# =============================================================================
@routeur.get(
    "/drift/report",
    summary     = "Rapport de drift des données",
    description = (
        "Lance l'analyse Evidently AI sur les prédictions récentes "
        "et retourne un résumé JSON du drift détecté.\n\n"
        "**Pré-requis** : `predictions.jsonl` doit contenir "
        "au moins 10 prédictions pour une analyse minimale.\n\n"
        "Le rapport HTML complet est généré dans "
        "`monitoring/drift_report.html`."
    ),
    tags        = ["Supervision"],
    status_code = 200,
)
def obtenir_rapport_drift() -> dict:
    """
    Déclenche l'analyse de drift et retourne un résumé JSON.

    Lance drift_analysis.py en sous-processus avec --format json,
    ce qui permet à l'endpoint de retourner les métriques clés
    sans bloquer l'API pendant toute la durée de l'analyse.

    Returns:
        Dictionnaire avec statut, nombre de prédictions analysées
        et indicateur de drift détecté.

    Raises:
        HTTPException 404 : predictions.jsonl inexistant.
        HTTPException 503 : Volume insuffisant (< 10 prédictions).
        HTTPException 504 : Timeout dépassé (> 60 secondes).
        HTTPException 500 : Erreur interne du script d'analyse.
    """
    # -- Vérification de l'existence des données de prédictions -------------
    if not FICHIER_PREDICTIONS.exists():
        raise HTTPException(
            status_code = 404,
            detail      = (
                "Aucune prédiction enregistrée. "
                "Effectuez des requêtes sur POST /predict d'abord."
            ),
        )

    # -- Vérification d'un volume minimum pour l'analyse --------------------
    nb_lignes = _compter_lignes_jsonl(FICHIER_PREDICTIONS)

    if nb_lignes < 10:
        raise HTTPException(
            status_code = 503,
            detail      = (
                f"Volume insuffisant : {nb_lignes} prédictions disponibles "
                f"(minimum requis : 10)."
            ),
        )

    journal.info(
        "GET /drift/report | %d prédictions à analyser", nb_lignes
    )

    # -- Lancement du script drift en sous-processus -----------------------
    try:
        resultat = subprocess.run(
            [sys.executable, str(SCRIPT_DRIFT), "--format", "json"],
            capture_output = True,
            text           = True,
            timeout        = 60,              # Timeout 60s maximum
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code = 504,
            detail      = "L'analyse drift a dépassé le délai de 60 secondes.",
        )

    # -- Vérification du succès du script ------------------------------------
    if resultat.returncode != 0:
        journal.error(
            "Erreur drift_analysis.py : %s", resultat.stderr[:200]
        )
        raise HTTPException(
            status_code = 500,
            detail      = (
                "Erreur lors de l'analyse drift. "
                "Consultez les logs du serveur pour le détail."
            ),
        )

    # -- Parsing de la sortie JSON du script ---------------------------------
    import json
    try:
        resume = json.loads(resultat.stdout)
    except json.JSONDecodeError:
        # Le script n'a pas produit de JSON valide → réponse partielle
        resume = {
            "statut"    : "ok",
            "nb_lignes" : nb_lignes,
            "message"   : "Analyse terminée. Voir monitoring/drift_report.html",
        }

    return resume


# ##############################################################################
# Fonctions privées
# ##############################################################################

# =============================================================================
def _compter_lignes_jsonl(chemin: Path) -> int:
    """
    Compte les lignes non vides d'un fichier JSONL.

    Chaque ligne non vide représente une prédiction enregistrée.

    Args:
        chemin : Chemin vers le fichier predictions.jsonl.

    Returns:
        Nombre de prédictions disponibles (0 si lecture impossible).
    """
    try:
        with open(chemin, encoding="utf-8") as f:
            return sum(1 for ligne in f if ligne.strip())
    except OSError:
        return 0
