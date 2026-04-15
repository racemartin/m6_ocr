"""
src/features/generate_enums.py
================================
Génération automatique des classes Enum depuis le FeatureRegistry.

Pour chaque AttributeSpec avec valeurs_possibles, génère un fichier Python
avec une classe Enum (valeur_metier → valeur_technique) qui permet :
    - Validation des valeurs à l'entrée de l'API
    - Mapping bidirectionnel métier ↔ technique
    - Autocomplétion IDE

Usage :
    python -m src.features.generate_enums
    # Génère src/features/enums/*.py
"""

from __future__ import annotations

import re
from pathlib import Path

from src.data.schema import REGISTRY, FeatureRegistry


# =============================================================================
# HELPERS
# =============================================================================

def _to_class_name(name_technique: str) -> str:
    """Convertit un name_technique en PascalCase."""
    parts = re.sub(r"[^a-zA-Z0-9]", "_", name_technique).split("_")
    return "".join(p.capitalize() for p in parts if p)


def _to_enum_key(value: str) -> str:
    """Convertit une valeur en constante UPPER_SNAKE_CASE valide pour Enum."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").upper()
    if cleaned and cleaned[0].isdigit():
        cleaned = "V_" + cleaned
    return cleaned or "UNKNOWN"


# =============================================================================
# GÉNÉRATION D'UNE CLASSE ENUM
# =============================================================================

def _generate_enum_file(
    name_technique: str,
    name_metier: str,
    description: str,
    valeurs_possibles: dict,
) -> str:
    """
    Génère le contenu Python d'un fichier Enum.

    Args:
        name_technique:    Nom technique de la colonne
        name_metier:       Nom métier compréhensible
        description:       Description de l'attribut
        valeurs_possibles: Dict {valeur_metier: valeur_technique}

    Returns:
        Contenu du fichier Python (str)
    """
    class_name = _to_class_name(name_technique) + "Enum"

    # Génère les entrées de l'Enum
    enum_entries = []
    for metier_val, technique_val in valeurs_possibles.items():
        key = _to_enum_key(metier_val)
        enum_entries.append((key, metier_val, technique_val))

    # Déduplique les clés en cas de collision
    seen_keys = {}
    deduped_entries = []
    for key, metier, technique in enum_entries:
        if key in seen_keys:
            key = f"{key}_{seen_keys[key]}"
            seen_keys[key] = seen_keys.get(key, 0) + 1
        else:
            seen_keys[key] = 1
        deduped_entries.append((key, metier, technique))

    # Mapping métier → technique (utilisé dans les méthodes de classe)
    mapping_str = ", ".join([
        f'"{m}": "{t}"' for _, m, t in deduped_entries
    ])
    reverse_mapping_str = ", ".join([
        f'"{t}": "{m}"' for _, m, t in deduped_entries
    ])

    lines = [
        '"""',
        'Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT',
        f'Attribut : {name_technique} ({name_metier})',
        f'Description : {description}',
        '"""',
        "",
        "from enum import Enum",
        "from typing import Optional",
        "",
        "",
        f"class {class_name}(str, Enum):",
        '    """',
        f"    Valeurs de {name_metier}.",
        "    Mapping : valeur_metier → valeur_technique",
        '    """',
        "",
    ]

    # Membres de l'Enum (valeur = valeur métier d'origine)
    for key, metier_val, _ in deduped_entries:
        lines.append(f'    {key} = "{metier_val}"')

    # Méthodes utilitaires
    lines += [
        "",
        "    # ─────────────────────────────────────────────────────────────",
        "",
        "    @classmethod",
        "    def to_technique(cls, metier_value: str) -> str:",
        '        """Convertit une valeur métier en valeur technique (pour le modèle)."""',
        f"        mapping = {{{mapping_str}}}",
        "        return mapping.get(str(metier_value).strip(), metier_value)",
        "",
        "    @classmethod",
        "    def to_metier(cls, technique_value: str) -> Optional[str]:",
        '        """Convertit une valeur technique en valeur métier (pour l\'API)."""',
        f"        reverse = {{{reverse_mapping_str}}}",
        "        return reverse.get(str(technique_value).strip())",
        "",
        "    @classmethod",
        "    def all_metier_values(cls) -> list:",
        '        """Liste de toutes les valeurs métier valides."""',
        "        return [e.value for e in cls]",
        "",
        "    @classmethod",
        "    def all_technique_values(cls) -> list:",
        '        """Liste de toutes les valeurs techniques correspondantes."""',
        f"        mapping = {{{mapping_str}}}",
        "        return list(mapping.values())",
        "",
        "    @classmethod",
        "    def is_valid(cls, value: str) -> bool:",
        '        """Vérifie si une valeur métier est dans l\'Enum."""',
        "        return value in cls._value2member_map_",
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# GÉNÉRATION DE L'INDEX DES ENUMS
# =============================================================================

def _generate_enums_init(enum_names: list[tuple[str, str]]) -> str:
    """
    Génère le fichier __init__.py du package enums.

    Args:
        enum_names: Liste de (module_name, class_name)
    """
    lines = [
        '"""',
        "Package enums — Auto-généré par generate_enums.py",
        "Contient une classe Enum par variable catégorielle du projet.",
        '"""',
        "",
    ]
    for module_name, class_name in sorted(enum_names):
        lines.append(f"from .{module_name} import {class_name}")

    lines += [
        "",
        "",
        "ALL_ENUMS = {",
    ]
    for module_name, class_name in sorted(enum_names):
        lines.append(f'    "{module_name}": {class_name},')
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================

def generate_enum_classes(
    registry: FeatureRegistry = REGISTRY,
    output_dir: str = "src/features/enums",
) -> list[str]:
    """
    Génère toutes les classes Enum depuis le FeatureRegistry.

    Args:
        registry:   FeatureRegistry source
        output_dir: Répertoire de sortie

    Returns:
        Liste des fichiers générés
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []
    enum_index = []

    print(f"\n🔧 Génération des Enums → {output_path}/")
    print(f"   {'-' * 50}")

    for attr in registry.attributes:
        if not attr.valeurs_possibles:
            continue

        module_name = attr.name_technique.lower()
        class_name  = _to_class_name(attr.name_technique) + "Enum"

        content = _generate_enum_file(
            name_technique=attr.name_technique,
            name_metier=attr.name_metier,
            description=attr.description,
            valeurs_possibles=attr.valeurs_possibles,
        )

        filepath = output_path / f"{module_name}.py"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        generated_files.append(str(filepath))
        enum_index.append((module_name, class_name))
        print(f"   ✅ {class_name:45s} → {filepath.name}")

    # Génère l'index __init__.py
    init_content = _generate_enums_init(enum_index)
    init_path = output_path / "__init__.py"
    with open(init_path, "w", encoding="utf-8") as f:
        f.write(init_content)

    print(f"\n   📄 {len(generated_files)} fichiers générés + __init__.py")
    print("   Import : from src.features.enums import ContractTypeEnum\n")

    return generated_files


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    generate_enum_classes(
        registry=REGISTRY,
        output_dir="src/features/enums"
    )
