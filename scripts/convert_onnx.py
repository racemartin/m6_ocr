# =============================================================================
# scripts/convert_onnx.py — Conversion pipeline LightGBM → format ONNX
# Exporte le modèle entraîné (sklearn Pipeline) au format ONNX pour
# une inférence 2-5x plus rapide en production, sans dépendance sklearn.
#
# Pourquoi ONNX ?
#   - Runtime léger : onnxruntime CPU ≈ 3 ms vs sklearn ≈ 8 ms
#   - Indépendant de la version scikit-learn du serveur de production
#   - Compatible HuggingFace Spaces, Docker Alpine, ARM
#   - Format standard ouvert (Microsoft/Linux Foundation)
#
# Utilisation :
#   python scripts/convert_onnx.py --run-id abc123def456
#   python scripts/convert_onnx.py --modele-path models/phase4_best_model.joblib
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import argparse                                       # Arguments CLI
import logging                                        # Journalisation
import sys                                            # Code sortie, path
import time                                           # Mesure benchmark
from   pathlib import Path                            # Chemins multi-OS

# --- Bibliothèques tierces : modèle sklearn ----------------------------------
import joblib                                         # Chargement .joblib
import numpy  as np                                   # Tenseurs benchmark
import pandas as pd

# --- Bibliothèques tierces : conversion ONNX ---------------------------------
from   skl2onnx                   import convert_sklearn   # Conversion sklearn
from   skl2onnx.common.data_types import FloatTensorType   # Type entrée ONNX

# --- Bibliothèques tierces : validation ONNX ---------------------------------
import onnxruntime as ort                             # Vérification post-conv.

# --- IMPORTACIONES CORREGIDAS ---
from skl2onnx import update_registered_converter
from lightgbm import LGBMClassifier
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm as lgbm_converter

# --- Configuration -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RACINE_PROJET,         # Racine du projet
    DOSSIER_ARTEFACT,      # Destination model_artifact/
    FICHIER_MODELE_ONNX,   # Chemin du fichier ONNX de sortie
)


# Configuration journalisation
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
)
journal = logging.getLogger(__name__)

# Nombre de features par défaut (dataset Home Credit Default Risk)
NB_FEATURES_DEFAUT = 18


# ##############################################################################
# Fonctions
# ##############################################################################

# =============================================================================
def analyser_arguments() -> argparse.Namespace:
    """
    Analyse les arguments de la ligne de commande.

    Returns:
        Namespace avec run_id ou chemin modèle, et options benchmark.
    """
    analyseur = argparse.ArgumentParser(
        description = "Conversion du pipeline LightGBM → ONNX",
    )
    analyseur.add_argument(
        "--run-id",
        type    = str,
        default = "",
        help    = "ID du run MLflow (prioritaire sur --modele-path)",
    )
    analyseur.add_argument(
        "--modele-path",
        type    = str,
        default = "",
        help    = "Chemin direct vers le fichier .joblib du modèle",
    )
    analyseur.add_argument(
        "--nb-requetes-bench",
        type    = int,
        default = 200,
        help    = "Nombre de requêtes pour le benchmark (défaut : 200)",
    )
    return analyseur.parse_args()


# =============================================================================
def charger_pipeline(run_id: str, chemin_modele: str):
    """
    Charge le pipeline sklearn depuis MLflow ou le système de fichiers.

    Priorité : MLflow si run_id fourni, sinon chemin local.
    Si aucun des deux n'est fourni, tente une localisation automatique
    du fichier phase4_best_model.joblib de m6_ocr.

    Args:
        run_id        : Identifiant du run MLflow (peut être vide).
        chemin_modele : Chemin vers le fichier .joblib (peut être vide).

    Returns:
        Pipeline sklearn chargé (preprocessor + estimateur LightGBM).

    Raises:
        FileNotFoundError : Si aucune source valide n'est trouvée.
    """
    # -- Chargement depuis MLflow si run_id fourni --------------------------
    if run_id:
        journal.info("Chargement depuis MLflow : runs:/%s/model", run_id)
        import mlflow.sklearn
        return mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # -- Résolution du chemin local -----------------------------------------
    if not chemin_modele:
        candidats = [
            RACINE_PROJET / "models" / "phase4_best_model.joblib",
            RACINE_PROJET.parent / "m6_ocr" / "models" / "phase4_best_model.joblib",
        ]
        chemin_resolu = next(
            (c for c in candidats if c.exists()), None
        )
        if not chemin_resolu:
            raise FileNotFoundError(
                "Modèle introuvable. Spécifiez --run-id ou --modele-path.\n"
                f"Chemins testés : {[str(c) for c in candidats]}"
            )
    else:
        chemin_resolu = Path(chemin_modele)

    journal.info("Chargement depuis fichier : %s", chemin_resolu)
    return joblib.load(chemin_resolu)


# =============================================================================
def convertir_en_onnx(pipeline, nb_features):
    # 1. Registro del convertidor
    # IMPORTANTE: Añadimos 'options' al registro para que acepte 'zipmap' y 'nocl'
    update_registered_converter(
        LGBMClassifier,
        'LightGbmLGBMClassifier',
        calculate_linear_classifier_output_shapes,
        lgbm_converter,
        options={'zipmap': [True, False], 'nocl': [True, False]} # <--- ESTO ES LA CLAVE
    )

    # 2. Configuración de entrada
    initial_type = [('float_input', FloatTensorType([None, nb_features]))]

    # 3. Forzamos las opciones en la conversión
    # Desactivamos zipmap (recomendado para APIs) para evitar el error de 'nocl'
    options = {id(pipeline): {'zipmap': False}}
    # Si lo anterior falla, probamos con el tipo de clase:
    options = {LGBMClassifier: {'zipmap': False}}

    print("  Info: Iniciando conversión con registro de opciones extendido...")

    dest_opset = {
        '': 17,          # Opset estándar de ONNX
        'ai.onnx.ml': 3  # <--- ESTO CORRIGE EL RUNTIME ERROR
    }

    modele_onnx = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=dest_opset,
        options=options
    )

    # 4. Retornar los bytes serializados
    return modele_onnx.SerializeToString()


# =============================================================================
def benchmarker_V1(
    pipeline_original,
    session_onnx      : ort.InferenceSession,
    nb_features       : int,reference_data,
    nb_requetes       : int = 200,
) -> dict:
    """
    Compare les performances sklearn vs ONNX sur N requêtes.

    Génère des données aléatoires float32 et mesure la latence
    d'inférence des deux moteurs pour calculer le gain ONNX.

    Args:
        pipeline_original : Pipeline sklearn d'origine.
        session_onnx      : Session ONNX initialisée.
        nb_features       : Nombre de features d'entrée.
        nb_requetes       : Nombre d'itérations de benchmark.

    Returns:
        Dictionnaire avec latences moyennes et speedup calculé.
    """
    np.random.seed(42)
    donnees = np.random.rand(nb_requetes, nb_features).astype(np.float32)

    # -- Benchmark sklearn ---------------------------------------------------
    debut_sk  = time.perf_counter()
    for i in range(nb_requetes):
        pipeline_original.predict_proba(donnees[i:i+1])
    fin_sk    = time.perf_counter()
    moy_sk_ms = (fin_sk - debut_sk) / nb_requetes * 1000

    # -- Benchmark ONNX Runtime ---------------------------------------------
    nom_entree = session_onnx.get_inputs()[0].name
    debut_onnx = time.perf_counter()
    for i in range(nb_requetes):
        session_onnx.run(None, {nom_entree: donnees[i:i+1]})
    fin_onnx   = time.perf_counter()
    moy_onnx_ms = (fin_onnx - debut_onnx) / nb_requetes * 1000

    speedup     = moy_sk_ms / moy_onnx_ms if moy_onnx_ms > 0 else 1.0

    return {
        "latence_sklearn_ms" : round(moy_sk_ms, 3),
        "latence_onnx_ms"    : round(moy_onnx_ms, 3),
        "speedup"            : round(speedup, 2),
        "nb_requetes"        : nb_requetes,
    }


def benchmarker(pipeline, session, nb_features, n_runs):
    """Mesure la latence sklearn vs ONNX Runtime."""

    # 1. Recuperar los nombres de las columnas originales del pipeline
    # Scikit-learn guarda esto en el atributo 'feature_names_in_'
    try:
        cols = pipeline.feature_names_in_
    except AttributeError:
        # Si por alguna razón no están, creamos nombres genéricos
        cols = [f"f{i}" for i in range(nb_features)]

    # 2. Generar datos aleatorios (NumPy)
    X_dummy_np = np.random.rand(1, nb_features).astype(np.float32)

    # 3. CONVERSIÓN A DATAFRAME (Esto quita el Warning)
    X_dummy_df = pd.DataFrame(X_dummy_np, columns=cols)

    # --- BENCHMARK SKLEARN ---
    start_sk = time.time()
    for _ in range(n_runs):
        # Usamos el DataFrame en lugar del array de numpy
        pipeline.predict_proba(X_dummy_df)
    latence_sk = (time.time() - start_sk) / n_runs * 1000

    # --- BENCHMARK ONNX ---
    # Nota: ONNX NO necesita DataFrame, sigue usando NumPy
    input_name = session.get_inputs()[0].name
    start_onnx = time.time()
    for _ in range(n_runs):
        session.run(None, {input_name: X_dummy_np})
    latence_onnx = (time.time() - start_onnx) / n_runs * 1000

    return {
        "latence_sklearn_ms": latence_sk,
        "latence_onnx_ms": latence_onnx,
        "speedup": latence_sk / latence_onnx
    }

# ##############################################################################
# Point d'entrée principal
# ##############################################################################

# =============================================================================
def main() -> None:
    """
    Orchestration de la conversion ONNX.

    Étapes :
        1. Chargement du pipeline sklearn (MLflow ou fichier local)
        2. Conversion vers ONNX avec skl2onnx
        3. Validation de la session ONNX Runtime
        4. Benchmark comparatif sklearn vs ONNX
        5. Sauvegarde dans model_artifact/best_model.onnx
    """
    args = analyser_arguments()

    print("\n============================================================================")
    print("CONVERSION LIGHTGBM → ONNX")
    print("============================================================================")

    # -- Chargement du pipeline source --------------------------------------
    try:
        pipeline = charger_pipeline(args.run_id, args.modele_path)
    except FileNotFoundError as erreur:
        print(f"\n  ERREUR : {erreur}", file=sys.stderr)
        sys.exit(1)

    # -- Détection du nombre de features ------------------------------------
    try:
        nb_features = pipeline.n_features_in_
    except AttributeError:
        nb_features = NB_FEATURES_DEFAUT
        journal.warning(
            "n_features_in_ non disponible — utilisation de %d features.",
            nb_features,
        )

    print(f"  Pipeline chargé.........: {type(pipeline).__name__}")
    print(f"  Nombre de features......: {nb_features}")

    # -- Conversion ONNX ----------------------------------------------------
    DOSSIER_ARTEFACT.mkdir(parents=True, exist_ok=True)
    contenu_onnx = convertir_en_onnx(pipeline, nb_features)

    # -- Sauvegarde du fichier ONNX -----------------------------------------
    with open(FICHIER_MODELE_ONNX, "wb") as f:
        f.write(contenu_onnx)

    taille_mo = FICHIER_MODELE_ONNX.stat().st_size / 1_048_576
    print(f"  Modèle ONNX sauvegardé..: {FICHIER_MODELE_ONNX}")
    print(f"  Taille du fichier.......: {taille_mo:.2f} Mo")

    # -- Validation de la session ONNX --------------------------------------
    session = ort.InferenceSession(
        str(FICHIER_MODELE_ONNX),
        providers = ["CPUExecutionProvider"],
    )
    print(f"  Session ONNX validée....: OK")

    # -- Benchmark comparatif -----------------------------------------------
    print(f"\n  Benchmark {args.nb_requetes_bench} requêtes en cours...")
    metriques = benchmarker(
        pipeline, session, nb_features, args.nb_requetes_bench
    )

    print("\n============================================================================")
    print("RÉSULTATS DU BENCHMARK")
    print("============================================================================")
    print(f"  Latence sklearn.........: {metriques['latence_sklearn_ms']:.3f} ms")
    print(f"  Latence ONNX Runtime....: {metriques['latence_onnx_ms']:.3f} ms")
    print(f"  Accélération (speedup)..: {metriques['speedup']:.1f}x")
    print("============================================================================")
    print("CONVERSION TERMINÉE AVEC SUCCÈS")
    print("============================================================================\n")


# =============================================================================
if __name__ == "__main__":
    main()
