# =============================================================================
# scripts/export_best_model.py — Export du meilleur modèle → model_artifact/
# Script exécuté UNE SEULE FOIS localement après l'entraînement (m6_ocr).
# Récupère le meilleur run MLflow, lance la conversion ONNX et copie
# les artefacts dans model_artifact/ pour le déploiement HuggingFace.
#
# Prérequis :
#   - mlflow server actif (docker-compose up mlflow dans m6_ocr)
#   - m6_ocr/models/phase4_best_model.joblib doit exister
#
# Utilisation :
#   python scripts/export_best_model.py
#   python scripts/export_best_model.py --run-id abc123def456
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import argparse                                       # Arguments CLI
import json                                           # Écriture métadonnées
import logging                                        # Journalisation
import shutil                                         # Copie fichiers
import subprocess                                     # Lancement convert_onnx
import sys                                            # Code sortie, path
from   pathlib import Path                            # Chemins multi-OS

# --- Configuration -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RACINE_PROJET,         # Racine du projet m7_ocr
    DOSSIER_ARTEFACT,      # Destination : model_artifact/
    FICHIER_META_MODELE,   # Métadonnées JSON du modèle exporté
    FICHIER_MODELE_ONNX,   # Destination ONNX finale
)


# Configuration journalisation
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
)
journal = logging.getLogger(__name__)


# ##############################################################################
# Fonctions
# ##############################################################################

# =============================================================================
def analyser_arguments() -> argparse.Namespace:
    """
    Analyse les arguments de la ligne de commande.

    Returns:
        Namespace avec run_id optionnel et chemin données référence.
    """
    analyseur = argparse.ArgumentParser(
        description = "Export du meilleur modèle MLflow → model_artifact/",
    )
    analyseur.add_argument(
        "--run-id",
        type    = str,
        default = "",
        help    = "ID du run MLflow (auto-détecté si omis)",
    )
    analyseur.add_argument(
        "--donnees-reference",
        type    = str,
        default = "",
        help    = "Chemin vers les données d'entraînement (référence drift)",
    )
    analyseur.add_argument(
        "--mlflow-uri",
        type    = str,
        default = "http://localhost:5001",
        help    = "URL du serveur MLflow (défaut : http://localhost:5001)",
    )
    return analyseur.parse_args()


# =============================================================================
def detecter_meilleur_run(uri_tracking: str) -> tuple[str, dict]:
    """
    Identifie le meilleur run MLflow par ROC-AUC décroissant.

    Connecte au serveur MLflow, cherche dans l'expérience principale
    le run avec le score ROC-AUC le plus élevé.

    Args:
        uri_tracking : URL du serveur MLflow.

    Returns:
        Tuple (run_id, données_du_run) avec métriques et paramètres.

    Raises:
        RuntimeError : Si aucun run n'est trouvé dans MLflow.
    """
    import mlflow

    journal.info("Connexion au serveur MLflow : %s", uri_tracking)
    mlflow.set_tracking_uri(uri_tracking)

    # -- Recherche du meilleur run trié par ROC-AUC décroissant -------------
    runs = mlflow.search_runs(
        experiment_names = ["Smart_Credit_Scoring"],
        order_by         = ["metrics.eval_roc_auc DESC"],
        max_results      = 1,
    )

    if runs.empty:
        raise RuntimeError(
            "Aucun run trouvé dans l'expérience 'Smart_Credit_Scoring'.\n" # <--- Vérifiez bien le nom ici aussi
            "Vérifiez que MLflow est démarré et que m6_ocr a bien "
            "été exécuté avec un entraînement complet."
        )

    run_id = runs.iloc[0]["run_id"]

    # Mapeo exacto de los nombres de MLflow a tu diccionario local
    run_data = {
        "auc": float(runs.iloc[0]["metrics.eval_roc_auc"]),
        "f1": float(runs.iloc[0].get("metrics.eval_f1", 0.0)),
        "threshold": float(runs.iloc[0].get("metrics.seuil_metier_optimal", 0.527)), # <--- Valor 0.527
        "nb_features": runs.iloc[0].get("params.n_features", "232"),
        "algo": runs.iloc[0].get("tags.mlflow.runName", "lightgbm")
    }

    print(f"  Meilleur run détecté....: {run_id} ({run_data['algo']})")
    print(f"  ROC-AUC (eval)..........: {run_data['auc']:.4f}")
    print(f"  Seuil Optimal...........: {run_data['threshold']:.4f}")

    return run_id, run_data


# =============================================================================
def exporter_metadonnees(run_id: str, run_data: dict) -> None:
    """
    Sauvegarde les métadonnées du modèle exporté en JSON.

    Permet à l'API de connaître le contexte d'entraînement sans
    avoir besoin d'accéder à MLflow en production.

    Args:
        run_id   : Identifiant du run MLflow exporté.
        run_data : Dictionnaire des données du run (métriques, paramètres).
    """
    metadonnees = {
        "run_id"        : run_id,
        "roc_auc"       : run_data.get("auc", 0.0),        # Usa 'auc' de run_data
        "f1_score"      : run_data.get("f1", 0.0),         # Usa 'f1' de run_data
        "seuil_optimal" : run_data.get("threshold", 0.35), # Usa 'threshold' de run_data
        "nb_features"   : run_data.get("nb_features", "232"),
        "algorithme"    : run_data.get("algo", "lightgbm"),
    }

    print("\n" + "="*80)
    print("RÉSUMÉ DES MÉTADONNÉES À EXPORTER")
    print("="*80)

    # Opción A: Print formateado con indentación (Muy legible)
    # print(json.dumps(metadonnees, indent=4, ensure_ascii=False))

    print("="*80)
    # Opción B: Print línea por línea con iconos para la consola
    print(f"  🆔 Run ID         : {metadonnees['run_id']}")
    print(f"  📈 ROC-AUC        : {metadonnees['roc_auc']:.4f}")
    print(f"  🎯 F1-Score       : {metadonnees['f1_score']:.4f}")
    print(f"  ⚖️  Seuil Optimal  : {metadonnees['seuil_optimal']:.4f}")
    print(f"  🔢 Nb Features    : {metadonnees['nb_features']}")
    print(f"  🤖 Algorithme     : {metadonnees['algorithme']}")
    print("="*80 + "\n")

    with open(FICHIER_META_MODELE, "w", encoding="utf-8") as f:
        json.dump(metadonnees, f, indent=2, ensure_ascii=False)

    print(f"  Métadonnées sauvegardées: {FICHIER_META_MODELE}")


# =============================================================================
def copier_donnees_reference(chemin_source: str) -> None:
    """
    Copie les données d'entraînement comme référence pour Evidently.

    Le fichier reference_data.csv sera utilisé par drift_analysis.py
    pour comparer la distribution des prédictions récentes.

    Args:
        chemin_source : Chemin vers le CSV des données d'entraînement.
                        Si vide, recherche automatique dans m6_ocr.
    """
    if not chemin_source:
        # -- Localisation automatique depuis m6_ocr --------------------------
        candidats = [
            RACINE_PROJET.parent / "m6_ocr" / "data" / "processed" / "X_train.csv",
            RACINE_PROJET / "data" / "processed" / "X_train.csv",
        ]
        chemin_source = next(
            (str(c) for c in candidats if Path(c).exists()),
            "",
        )

    if not chemin_source or not Path(chemin_source).exists():
        journal.warning(
            "Données de référence introuvables — drift non disponible."
        )
        return

    destination = DOSSIER_ARTEFACT / "reference_data.csv"
    shutil.copy2(chemin_source, destination)
    print(f"  Données référence.......: {destination}")


# ##############################################################################
# Point d'entrée principal
# ##############################################################################

# =============================================================================
def main() -> None:
    """
    Orchestration complète de l'export du modèle.

    Étapes :
        1. Détection ou utilisation du run_id MLflow fourni
        2. Lancement de convert_onnx.py pour la conversion ONNX
        3. Export des métadonnées JSON
        4. Copie des données de référence pour Evidently AI
    """
    args = analyser_arguments()

    print("\n============================================================================")
    print("EXPORT DU MEILLEUR MODÈLE → model_artifact/")
    print("============================================================================")

    # -- Création du dossier de destination si nécessaire -------------------
    DOSSIER_ARTEFACT.mkdir(parents=True, exist_ok=True)

    # -- Détection du run MLflow --------------------------------------------
    if args.run_id:
        run_id   = args.run_id
        run_data = {}
        print(f"  Run ID fourni...........: {run_id}")
    else:
        run_id, run_data = detecter_meilleur_run(args.mlflow_uri)

    # -- Lancement de la conversion ONNX ------------------------------------
    script_onnx = Path(__file__).parent / "convert_onnx.py"

    print("\n  Lancement de la conversion ONNX...")
    resultat = subprocess.run(
        [sys.executable, str(script_onnx), "--run-id", run_id],
        capture_output = False,
    )

    if resultat.returncode != 0:
        print("\n  ERREUR : La conversion ONNX a échoué.", file=sys.stderr)
        sys.exit(1)

    # -- Export des métadonnées JSON ----------------------------------------
    if run_data:
        exporter_metadonnees(run_id, run_data)

    # -- Copie des données de référence Evidently ---------------------------
    copier_donnees_reference(args.donnees_reference)

    print("\n============================================================================")
    print("EXPORT TERMINÉ — model_artifact/ prêt pour HuggingFace")
    print("============================================================================\n")


# =============================================================================
if __name__ == "__main__":
    main()
