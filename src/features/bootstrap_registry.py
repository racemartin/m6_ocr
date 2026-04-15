"""
src/features/bootstrap_registry.py
=====================================
Génération automatique du code REGISTRY depuis les fichiers CSV bruts.

Fonctionnement :
  1. Inspecte chaque fichier CSV brut (types, cardinalité, nulls, valeurs uniques)
  2. Charge HomeCredit_columns_description.csv → description officielle par colonne
  3. Déduit automatiquement ColumnType, EncodingType, TransformType, source_table
  4. Génère le code Python REGISTRY prêt à coller dans schema.py

Usage :
    python -m src.features.bootstrap_registry \\
        --data-dir   data/raw \\
        --desc-file  data/raw/HomeCredit_columns_description.csv \\
        --output     src/data/schema_bootstrap.py

Workflow recommandé :
    1. Lancer ce script → génère schema_bootstrap.py
    2. Réviser name_metier (nom métier humain) dans le fichier généré
    3. Compléter / ajuster les valeurs_possibles si besoin
    4. Copier le contenu dans schema.py (REGISTRY)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# CONFIG : SEUILS DE DÉTECTION AUTOMATIQUE
# =============================================================================

# Si cardinalité <= ce seuil ET type object → ONE_HOT
OHE_CARDINALITY_MAX    = 15
# Si cardinalité > ce seuil ET type object → TARGET_ENCODING (haute cardinalité)
TARGET_ENC_CARDINALITY = 50
# Entre les deux → ORDINAL candidat (à réviser manuellement)
ORDINAL_CARDINALITY    = 30

# Si % de nulls > ce seuil → signalé dans les commentaires
NULL_ALERT_THRESHOLD   = 30.0

# Colonnes connues comme identifiants (pas de transform)
ID_COLUMNS = {"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"}

# Colonnes connues comme binaires Y/N (object mais pas OHE)
BINARY_YN_COLUMNS = {
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "FLAG_LAST_APPL_PER_CONTRACT", "EMERGENCYSTATE_MODE",
}

# Préfixes → transformation automatique
LOG_PREFIXES      = {"AMT_", "BUREAU_AMT", "CC_AMT", "PREV_AMT", "INSTALL_AMT"}
ROBUST_PREFIXES   = {"OBS_", "DEF_", "CNT_SOCIAL"}
DAYS_COLUMNS      = {"DAYS_"}  # → STANDARD (valeurs négatives déjà centrées)


# =============================================================================
# TABLE → source_table dans le Registry
# =============================================================================

TABLE_MAP = {
    "application_{train|test}.csv": "application",
    "bureau.csv":                   "bureau",
    "bureau_balance.csv":           "bureau_balance",
    "previous_application.csv":     "previous_application",
    "POS_CASH_balance.csv":         "pos_cash",
    "credit_card_balance.csv":      "credit_card",
    "installments_payments.csv":    "installments",
}

CSV_FILES = {
    "application": "application_train.csv",
    "bureau":      "bureau.csv",
    "bureau_balance":           "bureau_balance.csv",
    "previous_application":     "previous_application.csv",
    "pos_cash":    "POS_CASH_balance.csv",
    "credit_card": "credit_card_balance.csv",
    "installments":"installments_payments.csv",
}


# =============================================================================
# HELPERS : INFÉRENCE DES TYPES ET TRANSFORMATIONS
# =============================================================================

def _infer_source_table(desc_table: str) -> str:
    return TABLE_MAP.get(desc_table.strip(), "application")


def _infer_column_type(col: str, series: pd.Series) -> str:
    """Déduit ColumnType depuis le nom de colonne et la série."""
    if col in ID_COLUMNS:
        return "ColumnType.IDENTIFIER"
    if col == "TARGET":
        return "ColumnType.BINARY"
    dtype = str(series.dtype)
    if dtype == "object":
        if col in BINARY_YN_COLUMNS:
            return "ColumnType.BINARY"
        uniq = set(series.dropna().unique())
        if uniq <= {"Y", "N", "Yes", "No", "0", "1", 0, 1}:
            return "ColumnType.BINARY"
        return "ColumnType.CATEGORICAL"
    if dtype in ("int64", "float64"):
        # Colonnes binaires 0/1 numériques
        if col.startswith("FLAG_") or col.startswith("NFLAG_"):
            uniq = set(series.dropna().unique())
            if uniq <= {0, 1, 0.0, 1.0}:
                return "ColumnType.BINARY"
        return "ColumnType.NUMERICAL"
    return "ColumnType.NUMERICAL"


def _infer_encoding(col: str, series: pd.Series, col_type: str) -> str:
    """Déduit EncodingType."""
    if "CATEGORICAL" not in col_type:
        return "EncodingType.NONE"
    if col in BINARY_YN_COLUMNS:
        return "EncodingType.BINARY"
    card = series.nunique()
    if card <= OHE_CARDINALITY_MAX:
        return "EncodingType.ONE_HOT"
    if card >= TARGET_ENC_CARDINALITY:
        return "EncodingType.TARGET_ENC"
    return "EncodingType.ORDINAL"  # à réviser manuellement


def _infer_transform(col: str, series: pd.Series, col_type: str) -> str:
    """Déduit TransformType."""
    if "NUMERICAL" not in col_type:
        return "TransformType.NONE"
    # LOG : colonnes montants (distributions très asymétriques)
    for pref in LOG_PREFIXES:
        if col.startswith(pref):
            return "TransformType.LOG"
    # ROBUST : colonnes avec outliers extrêmes (cercle social, comptages)
    for pref in ROBUST_PREFIXES:
        if col.startswith(pref):
            return "TransformType.ROBUST"
    # DAYS : normalisation standard (déjà relatives, valeurs négatives)
    for pref in DAYS_COLUMNS:
        if col.startswith(pref):
            return "TransformType.STANDARD"
    # EXT_SOURCE : standard
    if col.startswith("EXT_SOURCE"):
        return "TransformType.STANDARD"
    # Par défaut pour numériques : STANDARD
    return "TransformType.STANDARD"


def _infer_role(col: str) -> str:
    if col in ID_COLUMNS:
        return "ColumnRole.IDENTIFIER"
    if col == "TARGET":
        return "ColumnRole.TARGET"
    return "ColumnRole.FEATURE"


def _build_valeurs_possibles(col: str, series: pd.Series, col_type: str) -> Optional[dict]:
    """
    Génère le dict valeurs_possibles pour les catégoriels à faible cardinalité.
    name_raw → name_technique (snake_case)
    """
    if "CATEGORICAL" not in col_type and "BINARY" not in col_type:
        return None
    uniq = sorted(series.dropna().unique().tolist())
    if len(uniq) > OHE_CARDINALITY_MAX:
        return None  # Trop de valeurs → TARGET_ENCODING, pas de mapping statique
    if len(uniq) == 0:
        return None

    result = {}
    for v in uniq:
        v_str = str(v)
        # Technique : snake_case nettoyé
        tech = re.sub(r"[^a-zA-Z0-9]", "_", v_str.lower().strip())
        tech = re.sub(r"_+", "_", tech).strip("_")
        if tech and tech[0].isdigit():
            tech = "v_" + tech
        result[v_str] = tech or "unknown"
    return result


def _to_name_technique(col_raw: str) -> str:
    """Convertit le nom brut en snake_case technique."""
    return col_raw.lower()


def _null_comment(pct: float, col: str) -> str:
    if pct >= NULL_ALERT_THRESHOLD:
        return f"  # ⚠️ {pct:.0f}% nulls"
    return ""


# =============================================================================
# INSPECTION DES CSV
# =============================================================================

def inspect_csv(path: Path, source_table: str) -> list[dict]:
    """
    Inspecte un fichier CSV et retourne une liste de dicts par colonne.
    """
    df = pd.read_csv(path, nrows=50_000)  # Échantillon suffisant pour les stats
    results = []

    for col in df.columns:
        series = df[col]
        dtype  = str(series.dtype)
        n_null = series.isna().sum()
        pct_null = round(n_null / len(df) * 100, 1)

        col_type   = _infer_column_type(col, series)
        encoding   = _infer_encoding(col, series, col_type)
        transform  = _infer_transform(col, series, col_type)
        role       = _infer_role(col)
        valeurs    = _build_valeurs_possibles(col, series, col_type)

        # Stats pour commentaires
        if "NUMERICAL" in col_type and n_null < len(df):
            try:
                stat_min = round(float(series.min()), 2)
                stat_max = round(float(series.max()), 2)
                stat_str = f"min={stat_min}, max={stat_max}"
            except Exception:
                stat_str = ""
        elif "CATEGORICAL" in col_type or "BINARY" in col_type:
            card = series.nunique()
            stat_str = f"card={card}"
        else:
            stat_str = ""

        results.append({
            "name_raw":          col,
            "name_technique":    _to_name_technique(col),
            "source_table":      source_table,
            "col_type":          col_type,
            "role":              role,
            "encoding":          encoding,
            "transform":         transform,
            "valeurs_possibles": valeurs,
            "pct_null":          pct_null,
            "stat_str":          stat_str,
            "dtype":             dtype,
        })

    return results


# =============================================================================
# CHARGEMENT DES DESCRIPTIONS OFFICIELLES
# =============================================================================

def load_descriptions(desc_file: Path) -> dict[str, str]:
    """
    Charge HomeCredit_columns_description.csv.
    Retourne un dict {COL_NAME: description_en_anglais}
    """
    try:
        df = pd.read_csv(desc_file, encoding="latin1")
        # Colonnes : Table, Row, Description
        result = {}
        for _, row in df.iterrows():
            col_name = str(row.get("Row", "")).strip()
            desc     = str(row.get("Description", "")).strip()
            if col_name and desc and desc != "nan":
                result[col_name] = desc
        print(f"   📖 {len(result)} descriptions chargées depuis {desc_file.name}")
        return result
    except Exception as e:
        print(f"   ⚠️ Impossible de charger {desc_file}: {e}")
        return {}


def _description_to_name_metier(col_raw: str, description: str) -> str:
    """
    Génère un name_metier court depuis la description officielle anglaise.
    Exemples :
        'Income of the client'                    → 'Revenu client'
        'Number of children the client has'       → 'Nb enfants'
        'How many days before the application...' → 'DAYS_EMPLOYED (à renommer)'
    Stratégie : prendre les 4 premiers mots significatifs de la description.
    """
    # Heuristique simple : 4 premiers mots, nettoyés
    desc_clean = re.sub(r"\(.*?\)", "", description).strip()
    words = desc_clean.split()[:5]
    short = " ".join(words)
    # Tronquer à 40 caractères
    if len(short) > 40:
        short = short[:37] + "..."
    return short if short else col_raw


# =============================================================================
# GÉNÉRATION DU CODE PYTHON
# =============================================================================

def _format_valeurs_possibles(valeurs: Optional[dict], indent: int = 12) -> str:
    if not valeurs:
        return "None"
    pad = " " * indent
    lines = ["{"]
    for k, v in valeurs.items():
        lines.append(f'{pad}    "{k}": "{v}",')
    lines.append(f"{pad}}}")
    return "\n".join(lines)


def _generate_attribute_code(col_info: dict, descriptions: dict) -> str:
    """Génère le code build_attribute_spec(...) pour une colonne."""
    col     = col_info["name_raw"]
    tech    = col_info["name_technique"]
    src     = col_info["source_table"]
    ctype   = col_info["col_type"]
    role    = col_info["role"]
    enc     = col_info["encoding"]
    tr      = col_info["transform"]
    valeurs = col_info["valeurs_possibles"]
    pct_n   = col_info["pct_null"]
    stat    = col_info["stat_str"]

    # name_metier : depuis description officielle ou col_raw
    raw_desc = descriptions.get(col, "")
    name_metier = _description_to_name_metier(col, raw_desc) if raw_desc else col

    # Commentaire inline
    comment_parts = []
    if stat:
        comment_parts.append(stat)
    if pct_n >= NULL_ALERT_THRESHOLD:
        comment_parts.append(f"⚠️ {pct_n:.0f}% null")
    comment = f"  # {' | '.join(comment_parts)}" if comment_parts else ""

    # Corps de build_attribute_spec
    lines = [
        '        build_attribute_spec(',
        f'            "{col}",',
        f'            "{name_metier}",  # TODO: réviser name_metier',
        f'            "{tech}",',
        f'            source_table="{src}",',
    ]
    if role != "ColumnRole.FEATURE":
        lines.append(f'            role={role},')
    if ctype != "ColumnType.NUMERICAL":
        lines.append(f'            col_type={ctype},')
    if enc != "EncodingType.NONE":
        lines.append(f'            encoding={enc},')
    if tr != "TransformType.NONE":
        lines.append(f'            transform={tr},')
    if valeurs:
        vp_str = _format_valeurs_possibles(valeurs, indent=12)
        lines.append(f'            valeurs_possibles={vp_str},')
    lines.append(f'        ),{comment}')

    return "\n".join(lines)


def generate_registry_code(
    data_dir:  Path,
    desc_file: Path,
) -> str:
    """
    Génère le code Python complet du REGISTRY.
    """
    descriptions = load_descriptions(desc_file)

    all_attributes: list[str] = []
    # seen_cols: set[str] = set()
    seen_table_cols: set[tuple[str, str]] = set()
    
    print(f"\n🔍 Inspection des fichiers CSV dans {data_dir}/")
    print(f"   {'─' * 60}")

    for source_table, filename in CSV_FILES.items():
        csv_path = data_dir / filename
        if not csv_path.exists():
            print(f"   ⚠️  {filename} introuvable → ignoré")
            continue

        cols_info = inspect_csv(csv_path, source_table)
        # new_cols  = [c for c in cols_info if c["name_raw"] not in seen_cols]
        # 2. La lógica de filtrado ahora es por par (tabla, columna)
        new_cols = [
            c for c in cols_info 
            if (source_table, c["name_raw"]) not in seen_table_cols
        ]
        print(f"   ✅ {filename:<35} → {len(cols_info)} cols ({len(new_cols)} nouvelles)")
        
        for col_info in new_cols:
            code = _generate_attribute_code(col_info, descriptions)
            all_attributes.append((source_table, col_info["name_raw"], code))
            #seen_cols.add(col_info["name_raw"])
            # 3. Guardamos el par para evitar duplicados reales dentro de la MISMA tabla
            seen_table_cols.add((source_table, col_info["name_raw"]))

    # Grouper par source_table pour organiser le REGISTRY
    table_order = list(CSV_FILES.keys())
    grouped: dict[str, list[str]] = {t: [] for t in table_order}
    for src, col, code in all_attributes:
        grouped[src].append((col, code))

    # Assembler le fichier final
    header = '''"""
src/data/schema_bootstrap.py
================================
FICHIER AUTO-GÉNÉRÉ par bootstrap_registry.py
NE PAS UTILISER DIRECTEMENT — réviser puis copier dans schema.py

Workflow :
  1. Réviser tous les "TODO: réviser name_metier" → nom métier humain court
  2. Vérifier les valeurs_possibles générées automatiquement
  3. Ajuster ColumnType / EncodingType si nécessaire
  4. Copier le bloc REGISTRY dans schema.py

Légende des commentaires inline :
  ⚠️ X% null  → taux de valeurs manquantes élevé
  card=N       → cardinalité (nb valeurs uniques)
  min=X, max=Y → plage de valeurs
"""

from __future__ import annotations
from typing import Optional, Dict, List
from dataclasses import dataclass, field

# --- Importer depuis schema.py ---
from src.data.schema import (
    AttributeSpec, FeatureRegistry, ColumnType, ColumnRole,
    EncodingType, TransformType, build_attribute_spec
)


# ═══════════════════════════════════════════════════════════════════════
# REGISTRY BOOTSTRAP — généré automatiquement, à réviser avant usage
# ═══════════════════════════════════════════════════════════════════════

REGISTRY_BOOTSTRAP = FeatureRegistry(
    attributes=[
'''

    body_lines = []

    for source_table in table_order:
        attrs = grouped.get(source_table, [])
        if not attrs:
            continue

        section_title = source_table.upper().replace("_", " ")
        body_lines.append(f"\n        # {'═' * 60}")
        body_lines.append(f"        # {section_title}")
        body_lines.append(f"        # {'═' * 60}\n")

        for col_name, code in attrs:
            body_lines.append(code)
            body_lines.append("")

    footer = '''    ]
)


# Pour tester le registry généré :
if __name__ == "__main__":
    print(f"Total attributs : {len(REGISTRY_BOOTSTRAP.attributes)}")
    print(f"Cols OHE        : {REGISTRY_BOOTSTRAP.cols_ohe}")
    print(f"Cols LOG        : {REGISTRY_BOOTSTRAP.cols_log}")
    print(f"Cols STANDARD   : {REGISTRY_BOOTSTRAP.cols_standard[:5]}...")
'''

    return header + "\n".join(body_lines) + "\n" + footer


# =============================================================================
# RAPPORT DE SYNTHÈSE
# =============================================================================

def print_summary(data_dir: Path) -> None:
    """Affiche un résumé statistique rapide de tous les CSV."""
    print(f"\n📊 Résumé statistique — {data_dir}/")
    print(f"   {'─' * 80}")
    print(f"   {'Fichier':<35} {'Lignes':>10} {'Cols':>6} {'Nulls%':>8} {'Catégor.':>10} {'Numér.':>8}")
    print(f"   {'─' * 80}")

    for source_table, filename in CSV_FILES.items():
        csv_path = data_dir / filename
        if not csv_path.exists():
            continue

        df   = pd.read_csv(csv_path, nrows=50_000)
        rows = len(df)
        cols = len(df.columns)
        null_pct = round(df.isna().mean().mean() * 100, 1)
        n_cat = sum(1 for c in df.columns if str(df[c].dtype) == "object")
        n_num = cols - n_cat

        print(f"   {filename:<35} {rows:>10,} {cols:>6} {null_pct:>7.1f}% {n_cat:>10} {n_num:>8}")

    print(f"   {'─' * 80}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap du REGISTRY depuis les CSV bruts"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Répertoire contenant les CSV bruts",
    )
    parser.add_argument(
        "--desc-file",
        type=Path,
        default=Path("data/raw/HomeCredit_columns_description.csv"),
        help="Fichier de descriptions des colonnes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/data/schema_bootstrap.py"),
        help="Fichier Python de sortie",
    )
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  BOOTSTRAP REGISTRY — Home Credit Default Risk")
    print("═" * 70)

    # Résumé des fichiers
    print_summary(args.data_dir)

    # Génération du code
    code = generate_registry_code(
        data_dir=args.data_dir,
        desc_file=args.desc_file,
    )

    # Écriture
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"\n✅ Fichier généré : {args.output}")
    print("   → Réviser les 'TODO: réviser name_metier' avant d'utiliser")
    print("   → Copier le contenu dans schema.py\n")


if __name__ == "__main__":
    main()
