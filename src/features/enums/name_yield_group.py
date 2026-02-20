"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_yield_group (Grouped interest rate into small)
Description : 
"""

from enum import Enum
from typing import Optional


class NameYieldGroupEnum(str, Enum):
    """
    Valeurs de Grouped interest rate into small.
    Mapping : valeur_metier → valeur_technique
    """

    XNA = "XNA"
    HIGH = "high"
    LOW_ACTION = "low_action"
    LOW_NORMAL = "low_normal"
    MIDDLE = "middle"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"XNA": "xna", "high": "high", "low_action": "low_action", "low_normal": "low_normal", "middle": "middle"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"xna": "XNA", "high": "high", "low_action": "low_action", "low_normal": "low_normal", "middle": "middle"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"XNA": "xna", "high": "high", "low_action": "low_action", "low_normal": "low_normal", "middle": "middle"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
