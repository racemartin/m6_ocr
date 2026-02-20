"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_family_status (Family status of the client)
Description : 
"""

from enum import Enum
from typing import Optional


class NameFamilyStatusEnum(str, Enum):
    """
    Valeurs de Family status of the client.
    Mapping : valeur_metier → valeur_technique
    """

    CIVIL_MARRIAGE = "Civil marriage"
    MARRIED = "Married"
    SEPARATED = "Separated"
    SINGLE_NOT_MARRIED = "Single / not married"
    UNKNOWN = "Unknown"
    WIDOW = "Widow"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Civil marriage": "civil_marriage", "Married": "married", "Separated": "separated", "Single / not married": "single_not_married", "Unknown": "unknown", "Widow": "widow"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"civil_marriage": "Civil marriage", "married": "Married", "separated": "Separated", "single_not_married": "Single / not married", "unknown": "Unknown", "widow": "Widow"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Civil marriage": "civil_marriage", "Married": "married", "Separated": "separated", "Single / not married": "single_not_married", "Unknown": "unknown", "Widow": "widow"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
