# =============================================================================
# scripts/export_best_model_shap.py
# Export du meilleur modele + preprocesseur + modele LightGBM pour SHAP.
#
# Produit dans model_artifact/ :
#   best_model.onnx        <- inference rapide (onnxruntime)
#   preprocessor.pkl       <- ColumnTransformer (preprocessing client)
#   best_model_lgbm.pkl    <- modele LightGBM original (pour TreeExplainer)
#   best_model_meta.json   <- metadonnees (run_id, auc, seuil_optimal)
#   reference_data.csv     <- 100 lignes pour le background SHAP
#
# Pourquoi deux fichiers modele ?
#   - best_model.onnx    : inference ~3 ms, format universel
#   - best_model_lgbm.pkl: SHAP TreeExplainer exige le modele sklearn natif
#     (il lit directement les noeuds des arbres de decision)
#
# Utilisation :
#   python scripts/export_best_model_shap.py
#   python scripts/export_best_model_shap.py --run-id abc123
# =============================================================================

# --- Bibliotheques standard ---------------------------------------------------
import argparse                                       # Arguments CLI
import json                                           # Metadonnees JSON
import logging                                        # Journalisation
import shutil                                         # Copie fichiers
import subprocess                                     # Lancement convert_onnx
import sys                                            # Code sortie
from   pathlib import Path                            # Chemins multi-OS

# --- Bibliotheques tierces ---------------------------------------------------
import joblib                                         # Serialisation .pkl

# --- Configuration -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RACINE_PROJET,
    DOSSIER_ARTEFACT,
    FICHIER_META_MODELE,
)


# Configuration journalisation
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
)
journal = logging.getLogger(__name__)


# =============================================================================
def analyser_arguments() -> argparse.Namespace:
    """Arguments de la ligne de commande."""
    analyseur = argparse.ArgumentParser(
        description = "Export modele + preprocesseur + LightGBM pour SHAP",
    )
    analyseur.add_argument("--run-id",           type=str, default="")
    analyseur.add_argument("--mlflow-uri",        type=str,
                            default="http://localhost:5001")
    analyseur.add_argument("--nom-experience",    type=str,
                            default="Smart_Credit_Scoring")
    analyseur.add_argument("--donnees-reference", type=str, default="")
    analyseur.add_argument("--nb-lignes-bg",      type=int, default=100,
                            help="Nombre de lignes background SHAP")
    return analyseur.parse_args()


# =============================================================================
def detecter_meilleur_run(
    uri_tracking   : str,
    nom_experience : str,
) -> tuple:
    """Detecte le meilleur run MLflow par ROC-AUC."""
    import mlflow

    mlflow.set_tracking_uri(uri_tracking)
    runs = mlflow.search_runs(
        experiment_names = [nom_experience],
        max_results      = 100,
    )
    if runs.empty:
        raise RuntimeError(f"Aucun run dans '{nom_experience}'.")

    # -- Detection de la metrique ROC-AUC ------------------------------------
    candidats = ["metrics.eval_roc_auc",
        "metrics.roc_auc", "metrics.roc_auc_score",
        "metrics.val_roc_auc", "metrics.auc",
    ]
    cle_metrique = next(
        (c for c in candidats
         if c in runs.columns and runs[c].max() > 0),
        None,
    )
    if not cle_metrique:
        raise RuntimeError(
            "Aucune metrique ROC-AUC trouvee. "
            f"Colonnes disponibles : {[c for c in runs.columns if 'roc' in c.lower() or 'auc' in c.lower()]}"
        )

    meilleur    = runs.sort_values(cle_metrique, ascending=False).iloc[0]
    run_id      = meilleur["run_id"]
    roc_auc     = meilleur[cle_metrique]

    print(f"  Run ID detecte..........: {run_id}")
    print(f"  ROC-AUC.................: {roc_auc:.4f}")

    return run_id, meilleur.to_dict(), cle_metrique


# =============================================================================
def exporter_pipeline_complet_V1(run_id: str) -> object:
    """
    Charge le Pipeline complet depuis MLflow et l'exporte en deux fichiers :
        - best_model_lgbm.pkl   : Pipeline complet (preprocesseur + LightGBM)
                                  utilise par SHAP TreeExplainer
        - preprocessor.pkl      : ColumnTransformer seul (preprocessing API)

    Le Pipeline complet est necessaire pour SHAP car TreeExplainer lit
    directement la structure interne des arbres LightGBM.

    Args:
        run_id : Identifiant du run MLflow.

    Returns:
        Pipeline sklearn charge.
    """
    import mlflow.sklearn

    journal.info("Chargement du Pipeline depuis MLflow : runs:/%s/model", run_id)
    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    chemin_lgbm = DOSSIER_ARTEFACT / "best_model_lgbm.pkl"
    joblib.dump(pipeline, chemin_lgbm, compress=3)
    print(f"  Pipeline LightGBM.......: {chemin_lgbm}")

    # -- Export du preprocesseur seul ----------------------------------------
    # Le Pipeline a typiquement 2 etapes : steps[0] = preprocesseur, steps[1] = modele
    if hasattr(pipeline, "steps") and len(pipeline.steps) >= 2:
        preprocesseur = pipeline.steps[0][1]   # Premier transformeur
    elif hasattr(pipeline, "steps") and len(pipeline.steps) == 1:
        preprocesseur = pipeline.steps[0][1]
    else:
        # Le pipeline EST le preprocesseur
        preprocesseur = pipeline

    chemin_preproc = DOSSIER_ARTEFACT / "preprocessor.pkl"
    joblib.dump(preprocesseur, chemin_preproc, compress=3)
    print(f"  Preprocesseur seul......: {chemin_preproc}")

    return pipeline

def exporter_pipeline_complet(run_id: str) -> object:
    import mlflow.sklearn
    journal.info("Chargement du Pipeline depuis MLflow : runs:/%s/model", run_id)
    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # 1. Guardar el pipeline completo para SHAP
    chemin_lgbm = DOSSIER_ARTEFACT / "best_model_lgbm.pkl"
    joblib.dump(pipeline, chemin_lgbm, compress=3)

    # -- Export du preprocesseur seul ----------------------------------------
    preprocesseur = None
    
    # Intento 1: Extraer del pipeline de MLflow
    if hasattr(pipeline, "steps") and len(pipeline.steps) >= 2:
        preprocesseur = pipeline.steps[0][1]
    
    # Intento 2: Carga local si MLflow falla o está incompleto
    if preprocesseur is None or not hasattr(preprocesseur, "transform"):
        journal.warning("Préprocesseur non trouvé dans le Pipeline MLflow. Tentative locale...")
        
        # Añadimos tu ruta específica a la lista de búsqueda
        candidatos = [
            RACINE_PROJET / "models" / "preprocessor" / "preprocessor.pkl", 
            RACINE_PROJET / "models" / "preprocessor.pkl",
            RACINE_PROJET / "data" / "models" / "preprocessor.pkl",
        ]
        
        for ruta in candidatos:
            if ruta.exists():
                preprocesseur = joblib.load(ruta)
                journal.info("Préprocesseur chargé avec succès depuis : %s", ruta)
                break
    
    if preprocesseur is None:
         raise RuntimeError("Impossible d'extraire un objet avec .transform() del pipeline")

    chemin_preproc = DOSSIER_ARTEFACT / "preprocessor.pkl"
    joblib.dump(preprocesseur, chemin_preproc, compress=3)
    
    return pipeline    

# =============================================================================
def exporter_donnees_reference(
    chemin_source : str,
    nb_lignes     : int = 100,
) -> None:
    """
    Exporte les donnees de reference pour le background SHAP.

    Le background SHAP est un sous-ensemble des donnees d'entrainement
    utilise pour calculer les SHAP values de reference (baseline).
    100 lignes sont suffisantes pour un background stable.

    Args:
        chemin_source : Chemin vers le CSV des donnees d'entrainement.
        nb_lignes     : Nombre de lignes a exporter (defaut 100).
    """
    import pandas as pd

    if not chemin_source:
        candidats = [
            RACINE_PROJET.parent / "m6_ocr" / "data" / "processed" / "X_train.csv",
            RACINE_PROJET / "data" / "processed" / "X_train.csv",
        ]
        chemin_source = next(
            (str(c) for c in candidats if Path(c).exists()), ""
        )

    if not chemin_source or not Path(chemin_source).exists():
        journal.warning(
            "Donnees de reference introuvables -- background SHAP nul."
        )
        return

    df              = pd.read_csv(chemin_source)
    df_sample       = df.sample(
        n           = min(nb_lignes, len(df)),
        random_state = 42,
    )
    destination     = DOSSIER_ARTEFACT / "reference_data.csv"
    df_sample.to_csv(destination, index=False)
    print(f"  Donnees reference ({nb_lignes} lignes): {destination}")


# =============================================================================
def exporter_metadonnees(
    run_id       : str,
    run_data     : dict,
    cle_metrique : str,
) -> None:
    """Sauvegarde les metadonnees du modele en JSON."""
    cles_seuil = [
        "metrics.seuil_optimal", "metrics.optimal_threshold",
        "metrics.threshold",     "metrics.best_threshold",
    ]
    seuil = next(
        (run_data.get(c) for c in cles_seuil if run_data.get(c)),
        0.35,
    )

    metadonnees = {
        "run_id"        : run_id,
        "roc_auc"       : run_data.get(cle_metrique, 0.0),
        "f1_score"      : run_data.get("metrics.f1_score",
                          run_data.get("metrics.f1", 0.0)),
        "seuil_optimal" : seuil,
        "algorithme"    : run_data.get("params.algorithme", "lightgbm"),
        "shap_disponible" : True,
    }

    with open(FICHIER_META_MODELE, "w", encoding="utf-8") as f:
        json.dump(metadonnees, f, indent=2, ensure_ascii=False)

    print(f"  Metadonnees.............: {FICHIER_META_MODELE}")


# =============================================================================
def main() -> None:
    """
    Orchestration de l'export complet pour SHAP.

    Etapes :
        1. Detection / utilisation du run MLflow
        2. Export Pipeline LightGBM complet (pour SHAP TreeExplainer)
        3. Export preprocesseur seul (pour l'API)
        4. Conversion ONNX (via convert_onnx.py)
        5. Export donnees de reference (background SHAP)
        6. Export metadonnees JSON
    """
    args = analyser_arguments()

    print("\n============================================================================")
    print("EXPORT MODELE COMPLET (ONNX + LGBM + PREPROCESSEUR + REFERENCE)")
    print("============================================================================")

    DOSSIER_ARTEFACT.mkdir(parents=True, exist_ok=True)

    # -- Detection du run MLflow --------------------------------------------
    if args.run_id:
        run_id, run_data, cle_metrique = (
            args.run_id, {}, "metrics.roc_auc"
        )
        print(f"  Run ID fourni...........: {run_id}")
    else:
        run_id, run_data, cle_metrique = detecter_meilleur_run(
            args.mlflow_uri, args.nom_experience
        )

    # -- Export Pipeline LightGBM + preprocesseur ----------------------------
    exporter_pipeline_complet(run_id)

    # -- Conversion ONNX ----------------------------------------------------
    script_onnx = Path(__file__).parent / "convert_onnx.py"
    print("\n  Conversion ONNX en cours...")
    resultat = subprocess.run(
        [sys.executable, str(script_onnx), "--run-id", run_id],
        capture_output = False,
    )
    if resultat.returncode != 0:
        print("\n  ERREUR : Conversion ONNX echouee.", file=sys.stderr)
        sys.exit(1)

    # -- Donnees de reference (background SHAP) ------------------------------
    exporter_donnees_reference(args.donnees_reference, args.nb_lignes_bg)

    # -- Metadonnees JSON ---------------------------------------------------
    if run_data:
        exporter_metadonnees(run_id, run_data, cle_metrique)

    print("\n============================================================================")
    print("EXPORT TERMINE")
    print("============================================================================")
    print("\n  Contenu de model_artifact/ :")
    for f in sorted(DOSSIER_ARTEFACT.iterdir()):
        taille = f.stat().st_size / 1024
        print(f"    {f.name:<35} {taille:8.1f} Ko")
    print()


# =============================================================================
if __name__ == "__main__":
    main()