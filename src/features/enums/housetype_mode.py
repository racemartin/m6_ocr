"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : housetype_mode (Normalized information about building...)
Description : 
"""

from enum import Enum
from typing import Optional


class HousetypeModeEnum(str, Enum):
    """
    Valeurs de Normalized information about building....
    Mapping : valeur_metier → valeur_technique
    """

    BLOCK_OF_FLATS = "block of flats"
    SPECIFIC_HOUSING = "specific housing"
    TERRACED_HOUSE = "terraced house"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"block of flats": "block_of_flats", "specific housing": "specific_housing", "terraced house": "terraced_house"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"block_of_flats": "block of flats", "specific_housing": "specific housing", "terraced_house": "terraced house"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"block of flats": "block_of_flats", "specific housing": "specific_housing", "terraced house": "terraced_house"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
