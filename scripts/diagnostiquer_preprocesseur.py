# =============================================================================
# scripts/diagnostiquer_preprocesseur.py
# Script de diagnostic a executer UNE FOIS dans m6_ocr pour connaitre :
#   - Le type exact du preprocesseur
#   - Les noms des colonnes d'entree attendues
#   - Le nombre de features en sortie
#   - La structure interne (ColumnTransformer, Pipeline...)
#
# Utilisation (depuis la racine de m6_ocr) :
#   uv run python scripts/diagnostiquer_preprocesseur.py
# =============================================================================

# --- Bibliotheques standard ---------------------------------------------------
import json                                           # Lecture feature_names.json
import sys                                            # Code de sortie
from   pathlib import Path                            # Chemins multi-OS

# --- Bibliotheques tierces ---------------------------------------------------
import joblib                                         # Chargement .pkl
import numpy  as np                                   # Test de transformation


# =============================================================================
def inspecter_preprocesseur(chemin_pkl: Path) -> None:
    """
    Charge et inspecte le preprocesseur sklearn.

    Affiche le type, les colonnes d'entree et la forme de sortie
    pour permettre le cablage correct dans l'adaptateur ONNX.

    Args:
        chemin_pkl : Chemin vers preprocessor.pkl
    """
    print("\n============================================================================")
    print("DIAGNOSTIC DU PREPROCESSEUR")
    print("============================================================================")

    # -- Chargement ----------------------------------------------------------
    print(f"\n  Fichier.................:{chemin_pkl}")
    preproc = joblib.load(chemin_pkl)

    # -- Type ----------------------------------------------------------------
    type_preproc = type(preproc).__name__
    print(f"  Type....................: {type_preproc}")
    print(f"  Module..................: {type(preproc).__module__}")

    # -- Colonnes d'entree ---------------------------------------------------
    if hasattr(preproc, "feature_names_in_"):
        cols = list(preproc.feature_names_in_)
        print(f"\n  Colonnes d'entree.......: {len(cols)}")
        print("  Noms :")
        for i, col in enumerate(cols):
            print(f"    [{i:3d}] {col}")

    elif hasattr(preproc, "transformers_"):
        print(f"\n  Transformers (ColumnTransformer) :")
        for nom, trans, colonnes in preproc.transformers_:
            nb  = len(colonnes) if hasattr(colonnes, "__len__") else "?"
            print(f"    '{nom}' -> {type(trans).__name__} | {nb} colonnes")
            if hasattr(colonnes, "__iter__") and not isinstance(colonnes, str):
                for col in colonnes:
                    print(f"      - {col}")

    # -- Test de transformation sur donnees fictives -------------------------
    print("\n  Test de transformation sur une ligne fictive...")
    try:
        import pandas as pd

        # Ligne fictive avec des valeurs generiques
        ligne_test = {col: 0 for col in (preproc.feature_names_in_
                      if hasattr(preproc, "feature_names_in_") else [])}

        if ligne_test:
            df_test    = pd.DataFrame([ligne_test])
            sortie     = preproc.transform(df_test)
            print(f"  Shape entree............: {df_test.shape}")
            print(f"  Shape sortie............: {sortie.shape}")
            print(f"  Type sortie.............: {type(sortie).__name__}")
            print(f"  dtype sortie............: {sortie.dtype}")
        else:
            print("  (impossible sans feature_names_in_)")

    except Exception as erreur:
        print(f"  Erreur test.............: {erreur}")

    print("\n============================================================================")


# =============================================================================
def lire_feature_names(chemin_json: Path) -> None:
    """
    Lit et affiche le contenu de feature_names.json si disponible.

    Args:
        chemin_json : Chemin vers feature_names.json
    """
    if not chemin_json.exists():
        print(f"\n  feature_names.json introuvable : {chemin_json}")
        return

    with open(chemin_json, encoding="utf-8") as f:
        donnees = json.load(f)

    print("\n============================================================================")
    print("CONTENU DE feature_names.json")
    print("============================================================================")
    print(f"  Type contenu............: {type(donnees).__name__}")

    if isinstance(donnees, list):
        print(f"  Nombre de features......: {len(donnees)}")
        print("  10 premieres features :")
        for i, nom in enumerate(donnees[:10]):
            print(f"    [{i:3d}] {nom}")
        if len(donnees) > 10:
            print(f"    ... ({len(donnees) - 10} autres)")

    elif isinstance(donnees, dict):
        print(f"  Cles disponibles........: {list(donnees.keys())}")
        for cle, val in donnees.items():
            if isinstance(val, list):
                print(f"  '{cle}' -> {len(val)} elements")
                for i, nom in enumerate(val[:5]):
                    print(f"    [{i}] {nom}")

    print("============================================================================\n")


# =============================================================================
def main() -> None:
    """Point d'entree : localise et inspecte le preprocesseur."""

    # -- Localisation du preprocesseur --------------------------------------
    candidats_pkl = [
        Path("models/preprocessor/preprocessor.pkl"),
        Path("models/preprocessor.pkl"),
        Path("model_artifact/preprocessor.pkl"),
    ]
    candidats_json = [
        Path("models/preprocessor/feature_names.json"),
        Path("models/feature_names.json"),
    ]

    chemin_pkl = next((c for c in candidats_pkl if c.exists()), None)
    chemin_json = next((c for c in candidats_json if c.exists()), None)

    if chemin_pkl is None:
        print("\n  ERREUR : preprocessor.pkl introuvable.")
        print(f"  Chemins testes : {[str(c) for c in candidats_pkl]}")
        sys.exit(1)

    inspecter_preprocesseur(chemin_pkl)

    if chemin_json:
        lire_feature_names(chemin_json)


# =============================================================================
if __name__ == "__main__":
    main()
