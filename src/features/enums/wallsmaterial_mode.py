"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : wallsmaterial_mode (Normalized information about building...)
Description : 
"""

from enum import Enum
from typing import Optional


class WallsmaterialModeEnum(str, Enum):
    """
    Valeurs de Normalized information about building....
    Mapping : valeur_metier → valeur_technique
    """

    BLOCK = "Block"
    MIXED = "Mixed"
    MONOLITHIC = "Monolithic"
    OTHERS = "Others"
    PANEL = "Panel"
    STONE_BRICK = "Stone, brick"
    WOODEN = "Wooden"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Block": "block", "Mixed": "mixed", "Monolithic": "monolithic", "Others": "others", "Panel": "panel", "Stone, brick": "stone_brick", "Wooden": "wooden"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"block": "Block", "mixed": "Mixed", "monolithic": "Monolithic", "others": "Others", "panel": "Panel", "stone_brick": "Stone, brick", "wooden": "Wooden"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Block": "block", "Mixed": "mixed", "Monolithic": "monolithic", "Others": "others", "Panel": "panel", "Stone, brick": "stone_brick", "Wooden": "wooden"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
