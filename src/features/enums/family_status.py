"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : family_status (Statut familial)
Description : 
"""

from enum import Enum
from typing import Optional


class FamilyStatusEnum(str, Enum):
    """
    Valeurs de Statut familial.
    Mapping : valeur_metier → valeur_technique
    """

    SINGLE_NOT_MARRIED = "Single / not married"
    MARRIED = "Married"
    CIVIL_MARRIAGE = "Civil marriage"
    WIDOW = "Widow"
    SEPARATED = "Separated"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Single / not married": "single", "Married": "married", "Civil marriage": "civil", "Widow": "widow", "Separated": "separated"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"single": "Single / not married", "married": "Married", "civil": "Civil marriage", "widow": "Widow", "separated": "Separated"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Single / not married": "single", "Married": "married", "Civil marriage": "civil", "Widow": "widow", "Separated": "separated"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
