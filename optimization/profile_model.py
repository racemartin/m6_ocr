# =============================================================================
# optimization/profile_model.py — Comparativa de Performance: Sklearn vs ONNX
# =============================================================================

# =============================================================================
# POSSIBILITÉS D'EXÉCUTION (Commandes UV / Python) :
# =============================================================================
#
# 1. COMPARAISON STANDARD (Analyse des deux moteurs avec 500 requêtes) :
#    uv run python optimization/profile_model.py
#
# 2. TESTER UNIQUEMENT LE MOTEUR "AVANT" (Scikit-Learn natif) :
#    uv run python optimization/profile_model.py --moteur sklearn
#
# 3. TESTER UNIQUEMENT LE MOTEUR "APRÈS" (ONNX optimisé) :
#    uv run python optimization/profile_model.py --moteur onnx
#
# 4. TEST DE CHARGE (Augmenter le volume pour vérifier la stabilité) :
#    uv run python optimization/profile_model.py --nb-requetes 2000
#
# 5. EXPORTER LES RÉSULTATS AU FORMAT JSON (Pour l'audit MLOps) :
#    uv run python optimization/profile_model.py --format json
#
# 6. ANALYSE APPROFONDIE (Afficher les 50 fonctions les plus lentes via le Profiler) :
#    uv run python optimization/profile_model.py --top-fonctions 50
#
# =============================================================================

import argparse
import cProfile
import io
import json
import logging
import pstats
import sys
import time
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

# --- Patch de compatibilité sklearn ------------------------------------------
# Le modèle .pkl a été entraîné avec sklearn < 1.6 qui utilisait
# force_all_finite. Sklearn >= 1.6 l'a renommé en ensure_all_finite.
# Ce patch rétablit la compatibilité sans re-entraîner le modèle.
try:
    import sklearn.utils.validation as _skval
    if not hasattr(_skval, "_check_feature_names"):
        pass  # version très ancienne, pas de patch nécessaire
    _orig_check_array = _skval.check_array
    def _patched_check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            val = kwargs.pop("force_all_finite")
            # Convertir l'ancienne valeur vers le nouveau paramètre
            if val is True:
                kwargs.setdefault("ensure_all_finite", True)
            elif val is False:
                kwargs.setdefault("ensure_all_finite", False)
            elif val == "allow-nan":
                kwargs.setdefault("ensure_all_finite", "allow-nan")
        return _orig_check_array(*args, **kwargs)
    _skval.check_array = _patched_check_array
    # Propager le patch aux modules qui importent check_array directement
    import sklearn.utils._validation as _skval2
    _skval2.check_array = _patched_check_array
except Exception:
    pass  # Si le patch échoue, on laisse l'erreur originale apparaître

# --- Configuration -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FICHIER_MODELE_ONNX, RACINE_PROJET

# Definimos la ruta del modelo original (Ajusta el nombre si es distinto)
FICHIER_MODELE_SKLEARN = RACINE_PROJET / "model_artifact" / "best_model_lgbm.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
journal = logging.getLogger(__name__)

DOSSIER_RAPPORTS = RACINE_PROJET / "optimization" / "rapports"
NB_FEATURES_DEFAUT = 232  # Basado en tu nuevo esquema de 20 variables

# =============================================================================
def analyser_arguments() -> argparse.Namespace:
    analyseur = argparse.ArgumentParser(description="Benchmark & Profiling: Sklearn vs ONNX")
    analyseur.add_argument("--nb-requetes", type=int, default=500, help="Número de predicciones para el test")
    analyseur.add_argument("--nb-features", type=int, default=NB_FEATURES_DEFAUT)
    analyseur.add_argument("--moteur", choices=["sklearn", "onnx", "both"], default="both")
    analyseur.add_argument("--format", choices=["txt", "csv", "json"], default="txt")
    analyseur.add_argument("--top-fonctions", type=int, default=15, help="Funciones a mostrar en cProfile")
    return analyseur.parse_args()

# =============================================================================
def benchmarker_moteur(moteur_type: str, nb_requetes: int, nb_features: int) -> dict:
    import joblib

    np.random.seed(42)
    donnees_test = np.random.rand(nb_requetes, nb_features).astype(np.float32)
    latences_ms = []

    if moteur_type == "sklearn":
        if not FICHIER_MODELE_SKLEARN.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {FICHIER_MODELE_SKLEARN}")

        model = joblib.load(FICHIER_MODELE_SKLEARN)

        def predict_fn(x):
            try:
                return model.predict_proba(x)
            except TypeError:
                if hasattr(model, 'steps'):
                    return model.steps[-1][1].predict_proba(x)
                raise
        nom_complet = "Scikit-Learn (Antes)"
    else:
        import onnxruntime as ort
        session = ort.InferenceSession(str(FICHIER_MODELE_ONNX), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        def predict_fn(x): return session.run(None, {input_name: x})
        nom_complet = "ONNX Runtime (Después)"

    # Warm-up — 10 requêtes non comptabilisées pour initialiser les buffers
    for i in range(min(10, nb_requetes)):
        predict_fn(donnees_test[i:i+1])

    # Test réel — mesure requête par requête
    for i in range(nb_requetes):
        debut = time.perf_counter()
        predict_fn(donnees_test[i:i+1])
        latences_ms.append((time.perf_counter() - debut) * 1000)

    return {
        "moteur"      : nom_complet,
        "moteur_code" : moteur_type,
        "nb_requetes" : nb_requetes,
        "latence_min" : float(np.min(latences_ms)),
        "latence_p50" : float(np.percentile(latences_ms, 50)),
        "latence_p75" : float(np.percentile(latences_ms, 75)),
        "latence_p95" : float(np.percentile(latences_ms, 95)),
        "latence_p99" : float(np.percentile(latences_ms, 99)),
        "latence_max" : float(np.max(latences_ms)),
        "latence_moy" : float(np.mean(latences_ms)),
        "latences_raw": latences_ms,
    }

# =============================================================================
def afficher_rapport_console(metriques: dict) -> None:
    """
    Affiche le rapport de benchmark formaté dans la console.
    Produit exactement le bloc décrit dans le tutorial :
        ============================================================
        RAPPORT DE BENCHMARK — INFÉRENCE <MOTEUR>
        ============================================================
          Moteur d'inférence......: <nom>
          Nombre de requêtes......: <N>
        ------------------------------------------------------------
          Latence minimum.........: X.XXX ms
          Latence p50 (médiane)...: X.XXX ms   <- réponse typique
          Latence p75.............: X.XXX ms
          Latence p95.............: X.XXX ms   <- pic de charge
          Latence p99.............: X.XXX ms   <- pire cas habituel
          Latence maximum.........: X.XXX ms   <- outlier
          Latence moyenne.........: X.XXX ms
        ============================================================
    """
    SEP  = "=" * 60
    SEP2 = "-" * 60
    moteur_label = metriques["moteur"]

    print(f"\n{SEP}")
    print(f"RAPPORT DE BENCHMARK — INFÉRENCE {moteur_label.upper()}")
    print(SEP)
    print(f"  Moteur d'inférence......: {moteur_label}")
    print(f"  Nombre de requêtes......: {metriques['nb_requetes']}")
    print(SEP2)
    print(f"  Latence minimum.........: {metriques['latence_min']:.3f} ms")
    print(f"  Latence p50 (médiane)...: {metriques['latence_p50']:.3f} ms"
          "      <- réponse typique")
    print(f"  Latence p75.............: {metriques['latence_p75']:.3f} ms")
    print(f"  Latence p95.............: {metriques['latence_p95']:.3f} ms"
          "      <- pic de charge")
    print(f"  Latence p99.............: {metriques['latence_p99']:.3f} ms"
          "      <- pire cas habituel")
    print(f"  Latence maximum.........: {metriques['latence_max']:.3f} ms"
          "     <- outlier")
    print(f"  Latence moyenne.........: {metriques['latence_moy']:.3f} ms")
    print(SEP)

# =============================================================================
def profiler_moteur(moteur_type: str, nb_requetes: int, nb_features: int, top_n: int) -> str:
    """Ejecuta cProfile para detectar cuellos de botella internos."""
    donnees_test = np.random.rand(nb_requetes, nb_features).astype(np.float32)

    if moteur_type == "sklearn":
        model = joblib.load(FICHIER_MODELE_SKLEARN)
        def target(): [model.predict_proba(donnees_test[i:i+1]) for i in range(nb_requetes)]
    else:
        import onnxruntime as ort
        session = ort.InferenceSession(str(FICHIER_MODELE_ONNX), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        def target(): [session.run(None, {input_name: donnees_test[i:i+1]}) for i in range(nb_requetes)]

    prof = cProfile.Profile()
    prof.enable()
    target()
    prof.disable()

    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    ps.print_stats(top_n)
    return s.getvalue()

# =============================================================================
def exporter_resultats(metriques: dict, rapport_profil: str, format_ext: str) -> None:
    """
    Exporte les résultats selon le format demandé.

    Fichiers produits :
        --format txt  (défaut) : profiling_{slug}.txt  — rapport cProfile
        --format csv           : profiling_{slug}.txt  — rapport cProfile
                                 benchmark_{slug}_latences.csv
                                     colonnes : requete_num, latence_ms
        --format json          : profiling_{slug}.txt  — rapport cProfile
                                 bench_{slug}.json
                                     métriques résumées (sans latences_raw)
    Le rapport cProfile est TOUJOURS généré quel que soit le format.
    """
    DOSSIER_RAPPORTS.mkdir(parents=True, exist_ok=True)
    slug = metriques["moteur_code"]

    # -- CSV : série temporelle avec numéro de requête -----------------------
    if format_ext == "csv":
        df_latencias = pd.DataFrame({
            "requete_num": range(len(metriques["latences_raw"])),
            "latence_ms" : metriques["latences_raw"],
        })
        ruta_csv = DOSSIER_RAPPORTS / f"benchmark_{slug}_latences.csv"
        df_latencias.to_csv(ruta_csv, index=False)
        journal.info(f"CSV exportado: {ruta_csv}")

    # -- JSON : métriques résumées (latences_raw omis) -----------------------
    if format_ext == "json":
        metriques_export = {
            k: v for k, v in metriques.items()
            if k not in ("latences_raw", "moteur_code")
        }
        ruta_json = DOSSIER_RAPPORTS / f"bench_{slug}.json"
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(metriques_export, f, indent=2)
        journal.info(f"JSON exportado: {ruta_json}")

    # -- cProfile TXT : toujours généré --------------------------------------
    ruta_txt = DOSSIER_RAPPORTS / f"profiling_{slug}.txt"
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(rapport_profil)
    journal.info(f"cProfile exportado: {ruta_txt}")

# =============================================================================
def main():
    args = analyser_arguments()
    moteurs_a_tester = ["sklearn", "onnx"] if args.moteur == "both" else [args.moteur]
    resultats = []

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPARATIVO: SKLEARN vs ONNX ({args.nb_requetes} req)")
    print("=" * 70)

    for m in moteurs_a_tester:
        print(f"\nAnalizando motor: {m.upper()}...")
        try:
            res      = benchmarker_moteur(m, args.nb_requetes, args.nb_features)
            prof_res = profiler_moteur(m, args.nb_requetes, args.nb_features, args.top_fonctions)

            # Rapport console complet
            afficher_rapport_console(res)

            exporter_resultats(res, prof_res, args.format)
            resultats.append(res)
        except Exception as e:
            print(f"   Error: {e}")

    # Tabla comparativa final (solo en modo --moteur both)
    if len(resultats) > 1:
        print("\n" + "RESUMEN FINAL".center(70, "-"))
        gain = resultats[0]["latence_p50"] / resultats[1]["latence_p50"]
        for r in resultats:
            print(f"   {r['moteur']:<30} | p50: {r['latence_p50']:>6.3f} ms"
                  f"  p95: {r['latence_p95']:>6.3f} ms")
        print("-" * 70)
        print(f"MEJORA DE VELOCIDAD: x{gain:.2f} veces más rápido con ONNX")
        print("=" * 70 + "\n")

if __name__ == "__main__":
    main()