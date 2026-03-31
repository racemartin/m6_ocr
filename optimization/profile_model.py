# =============================================================================
# optimization/profile_model.py — Profilage de performance du modèle
# Mesure la latence d'inférence avec cProfile et benchmark N requêtes.
# Permet de comparer sklearn natif vs ONNX Runtime et d'identifier
# les fonctions les plus coûteuses dans le pipeline de prédiction.
#
# Quand l'utiliser :
#   - Avant conversion ONNX : mesurer le besoin d'optimisation
#   - Après conversion ONNX : valider le gain de performance
#   - En production : détecter une dégradation de latence
#
# Utilisation :
#   python optimization/profile_model.py --nb-requetes 500
#   python optimization/profile_model.py --nb-requetes 500 --format csv
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import argparse                                   # Arguments CLI
import cProfile                                   # Profilage fonction par fonction
import io                                         # Capture sortie cProfile
import json                                       # Écriture résultats JSON
import logging                                    # Journalisation
import pstats                                     # Analyse résultats cProfile
import sys                                        # Code de sortie, path
import time                                       # Mesure latence précise
from   pathlib import Path                        # Chemins multi-OS

# --- Bibliothèques tierces : données -----------------------------------------
import numpy  as np                               # Génération données test
import pandas as pd                               # Export résultats CSV

# --- Configuration -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FICHIER_MODELE_ONNX,   # Modèle ONNX à benchmarker
    RACINE_PROJET,          # Dossier racine pour les rapports
)


# Configuration journalisation
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
)
journal = logging.getLogger(__name__)

# Répertoire de sortie des rapports de profilage
DOSSIER_RAPPORTS  = RACINE_PROJET / "optimization" / "rapports"
NB_FEATURES_DEFAUT = 18  # Nombre de features du modèle Home Credit


# =============================================================================
def analyser_arguments() -> argparse.Namespace:
    """
    Analyse les arguments de la ligne de commande.

    Returns:
        Namespace avec les paramètres de benchmark.
    """
    analyseur = argparse.ArgumentParser(
        description = "Profilage cProfile et benchmark latences du modèle",
    )
    analyseur.add_argument(
        "--nb-requetes",
        type    = int,
        default = 500,
        help    = "Nombre de requêtes de benchmark (défaut : 500)",
    )
    analyseur.add_argument(
        "--nb-features",
        type    = int,
        default = NB_FEATURES_DEFAUT,
        help    = f"Nombre de features du modèle (défaut : {NB_FEATURES_DEFAUT})",
    )
    analyseur.add_argument(
        "--format",
        choices = ["txt", "csv", "json"],
        default = "txt",
        help    = "Format du rapport de sortie",
    )
    analyseur.add_argument(
        "--top-fonctions",
        type    = int,
        default = 20,
        help    = "Nombre de fonctions à afficher dans le rapport cProfile",
    )
    return analyseur.parse_args()


# ##############################################################################
# Benchmark ONNX Runtime
# ##############################################################################

# =============================================================================
def benchmarker_onnx(
    nb_requetes : int,
    nb_features : int,
) -> dict:
    """
    Mesure les latences d'inférence ONNX sur N requêtes consécutives.

    Génère des données aléatoires float32 représentant des demandes
    de crédit et mesure le temps d'inférence individuel pour calculer
    les percentiles de latence (p50, p95, p99).

    Args:
        nb_requetes : Nombre de requêtes à effectuer.
        nb_features : Nombre de features d'entrée du modèle.

    Returns:
        Dictionnaire avec percentiles de latence et statistiques.

    Raises:
        FileNotFoundError : Si best_model.onnx est introuvable.
    """
    import onnxruntime as ort

    # -- Vérification existence du modèle ------------------------------------
    if not FICHIER_MODELE_ONNX.exists():
        raise FileNotFoundError(
            f"Modèle ONNX introuvable : {FICHIER_MODELE_ONNX}\n"
            "Exécutez d'abord : python scripts/export_best_model.py"
        )

    # -- Initialisation de la session ONNX -----------------------------------
    options                    = ort.SessionOptions()
    options.log_severity_level = 3  # Silence les logs ONNX verbeux
    session = ort.InferenceSession(
        str(FICHIER_MODELE_ONNX),
        sess_options = options,
        providers    = ["CPUExecutionProvider"],
    )
    nom_entree = session.get_inputs()[0].name

    # -- Génération des données de test aléatoires ---------------------------
    np.random.seed(42)
    donnees_test = np.random.rand(
        nb_requetes, nb_features
    ).astype(np.float32)

    # -- Phase de warm-up (5 requêtes non comptabilisées) --------------------
    for i in range(min(5, nb_requetes)):
        session.run(None, {nom_entree: donnees_test[i:i+1]})

    # -- Benchmark principal -------------------------------------------------
    latences_ms = np.zeros(nb_requetes)

    for i in range(nb_requetes):
        debut         = time.perf_counter()
        session.run(None, {nom_entree: donnees_test[i:i+1]})
        fin           = time.perf_counter()
        latences_ms[i] = (fin - debut) * 1000

    return {
        "moteur"      : "ONNX Runtime",
        "nb_requetes" : nb_requetes,
        "latence_min" : float(np.min(latences_ms)),
        "latence_p50" : float(np.percentile(latences_ms, 50)),
        "latence_p75" : float(np.percentile(latences_ms, 75)),
        "latence_p95" : float(np.percentile(latences_ms, 95)),
        "latence_p99" : float(np.percentile(latences_ms, 99)),
        "latence_max" : float(np.max(latences_ms)),
        "latence_moy" : float(np.mean(latences_ms)),
        "latences_raw": latences_ms.tolist(),
    }


# ##############################################################################
# Profilage cProfile
# ##############################################################################

# =============================================================================
def profiler_onnx(
    nb_requetes : int,
    nb_features : int,
    top_n       : int,
) -> str:
    """
    Profile l'inférence ONNX avec cProfile et retourne le rapport texte.

    Identifie les fonctions Python les plus coûteuses lors de l'inférence.
    Utile pour détecter des goulots d'étranglement inattendus (ex: encodage
    des données, appels système, conversions numpy…).

    Args:
        nb_requetes : Nombre d'inférences à profiler.
        nb_features : Nombre de features d'entrée.
        top_n       : Nombre de fonctions à inclure dans le rapport.

    Returns:
        Rapport cProfile formaté en texte.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(FICHIER_MODELE_ONNX),
        providers = ["CPUExecutionProvider"],
    )
    nom_entree   = session.get_inputs()[0].name
    donnees_test = np.random.rand(
        nb_requetes, nb_features
    ).astype(np.float32)

    # -- Fonction cible du profilage -----------------------------------------
    def executer_inferences():
        for i in range(nb_requetes):
            session.run(None, {nom_entree: donnees_test[i:i+1]})

    # -- Profilage avec cProfile ---------------------------------------------
    profileur     = cProfile.Profile()
    profileur.enable()
    executer_inferences()
    profileur.disable()

    # -- Extraction et formatage du rapport ----------------------------------
    tampon        = io.StringIO()
    statistiques  = pstats.Stats(profileur, stream=tampon)
    statistiques.sort_stats("cumulative")
    statistiques.print_stats(top_n)

    return tampon.getvalue()


# ##############################################################################
# Affichage et export des résultats
# ##############################################################################

# =============================================================================
def afficher_resultats(metriques: dict) -> None:
    """
    Affiche les résultats du benchmark dans la console.

    Args:
        metriques : Dictionnaire de métriques calculées par benchmarker_onnx.
    """
    print("\n============================================================================")
    print("RAPPORT DE BENCHMARK — INFÉRENCE ONNX RUNTIME")
    print("============================================================================")
    print(f"  Moteur d'inférence......: {metriques['moteur']}")
    print(f"  Nombre de requêtes......: {metriques['nb_requetes']:,}")
    print("----------------------------------------------------------------------------")
    print(f"  Latence minimum.........: {metriques['latence_min']:.3f} ms")
    print(f"  Latence p50 (médiane)...: {metriques['latence_p50']:.3f} ms")
    print(f"  Latence p75.............: {metriques['latence_p75']:.3f} ms")
    print(f"  Latence p95.............: {metriques['latence_p95']:.3f} ms")
    print(f"  Latence p99.............: {metriques['latence_p99']:.3f} ms")
    print(f"  Latence maximum.........: {metriques['latence_max']:.3f} ms")
    print(f"  Latence moyenne.........: {metriques['latence_moy']:.3f} ms")
    print("============================================================================")


# =============================================================================
def exporter_resultats(
    metriques  : dict,
    rapport_profil : str,
    format_sortie  : str,
) -> None:
    """
    Exporte les résultats dans le format demandé.

    Args:
        metriques      : Métriques de benchmark calculées.
        rapport_profil : Rapport texte cProfile.
        format_sortie  : "txt" | "csv" | "json"
    """
    DOSSIER_RAPPORTS.mkdir(parents=True, exist_ok=True)

    # -- Export selon le format demandé -------------------------------------
    if format_sortie == "json":
        chemin = DOSSIER_RAPPORTS / "benchmark_onnx.json"
        export = {k: v for k, v in metriques.items() if k != "latences_raw"}
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"  Résultats JSON exportés.: {chemin}")

    elif format_sortie == "csv":
        chemin = DOSSIER_RAPPORTS / "benchmark_onnx_latences.csv"
        df = pd.DataFrame({
            "requete_num" : range(len(metriques["latences_raw"])),
            "latence_ms"  : metriques["latences_raw"],
        })
        df.to_csv(chemin, index=False)
        print(f"  Latences CSV exportées..: {chemin}")

    # -- Export du rapport cProfile toujours en TXT -------------------------
    chemin_profil = DOSSIER_RAPPORTS / "cprofile_onnx.txt"
    with open(chemin_profil, "w", encoding="utf-8") as f:
        f.write(rapport_profil)
    print(f"  Rapport cProfile exporté: {chemin_profil}")


# ##############################################################################
# Point d'entrée principal
# ##############################################################################

# =============================================================================
def main() -> None:
    """
    Orchestration principale du profilage de performance.

    Étapes :
        1. Benchmark ONNX Runtime sur N requêtes
        2. Profilage cProfile de l'inférence
        3. Affichage des résultats dans la console
        4. Export dans le format demandé
    """
    args = analyser_arguments()

    print("\n============================================================================")
    print("PROFILAGE DE PERFORMANCE — MODÈLE SCORING CRÉDIT")
    print("============================================================================")
    print(f"  Nombre de requêtes......: {args.nb_requetes:,}")
    print(f"  Nombre de features......: {args.nb_features}")
    print(f"  Format de sortie........: {args.format}")

    # -- Benchmark latences ONNX --------------------------------------------
    print("\n  Benchmark ONNX en cours...")
    try:
        metriques = benchmarker_onnx(args.nb_requetes, args.nb_features)
    except FileNotFoundError as erreur:
        print(f"\n  ERREUR : {erreur}", file=sys.stderr)
        sys.exit(1)

    afficher_resultats(metriques)

    # -- Profilage cProfile -------------------------------------------------
    print("\n  Profilage cProfile en cours...")
    rapport_profil = profiler_onnx(
        args.nb_requetes,
        args.nb_features,
        args.top_fonctions,
    )
    print("\n  Top fonctions par temps cumulé (cProfile) :")
    print(rapport_profil[:2000])  # Aperçu tronqué dans la console

    # -- Export des résultats -----------------------------------------------
    exporter_resultats(metriques, rapport_profil, args.format)

    print("============================================================================")
    print("PROFILAGE TERMINÉ")
    print("============================================================================\n")


# =============================================================================
if __name__ == "__main__":
    main()
