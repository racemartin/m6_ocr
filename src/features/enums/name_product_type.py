"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_product_type (Was the previous application x-sell)
Description : 
"""

from enum import Enum
from typing import Optional


class NameProductTypeEnum(str, Enum):
    """
    Valeurs de Was the previous application x-sell.
    Mapping : valeur_metier → valeur_technique
    """

    XNA = "XNA"
    WALK_IN = "walk-in"
    X_SELL = "x-sell"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"XNA": "xna", "walk-in": "walk_in", "x-sell": "x_sell"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"xna": "XNA", "walk_in": "walk-in", "x_sell": "x-sell"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"XNA": "xna", "walk-in": "walk_in", "x-sell": "x_sell"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
