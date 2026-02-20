"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_client_type (Was the client old or)
Description : 
"""

from enum import Enum
from typing import Optional


class NameClientTypeEnum(str, Enum):
    """
    Valeurs de Was the client old or.
    Mapping : valeur_metier → valeur_technique
    """

    NEW = "New"
    REFRESHED = "Refreshed"
    REPEATER = "Repeater"
    XNA = "XNA"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"New": "new", "Refreshed": "refreshed", "Repeater": "repeater", "XNA": "xna"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"new": "New", "refreshed": "Refreshed", "repeater": "Repeater", "xna": "XNA"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"New": "new", "Refreshed": "refreshed", "Repeater": "repeater", "XNA": "xna"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
