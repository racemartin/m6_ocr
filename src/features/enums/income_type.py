"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : income_type (Type revenu)
Description : 
"""

from enum import Enum
from typing import Optional


class IncomeTypeEnum(str, Enum):
    """
    Valeurs de Type revenu.
    Mapping : valeur_metier → valeur_technique
    """

    WORKING = "Working"
    STATE_SERVANT = "State servant"
    COMMERCIAL_ASSOCIATE = "Commercial associate"
    PENSIONER = "Pensioner"
    UNEMPLOYED = "Unemployed"
    STUDENT = "Student"
    BUSINESSMAN = "Businessman"
    MATERNITY_LEAVE = "Maternity leave"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Working": "working", "State servant": "state_servant", "Commercial associate": "commercial", "Pensioner": "pensioner", "Unemployed": "unemployed", "Student": "student", "Businessman": "businessman", "Maternity leave": "maternity"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"working": "Working", "state_servant": "State servant", "commercial": "Commercial associate", "pensioner": "Pensioner", "unemployed": "Unemployed", "student": "Student", "businessman": "Businessman", "maternity": "Maternity leave"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Working": "working", "State servant": "state_servant", "Commercial associate": "commercial", "Pensioner": "pensioner", "Unemployed": "unemployed", "Student": "student", "Businessman": "businessman", "Maternity leave": "maternity"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
