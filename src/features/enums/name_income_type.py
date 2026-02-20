"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_income_type (Clients income type)
Description : 
"""

from enum import Enum
from typing import Optional


class NameIncomeTypeEnum(str, Enum):
    """
    Valeurs de Clients income type.
    Mapping : valeur_metier → valeur_technique
    """

    BUSINESSMAN = "Businessman"
    COMMERCIAL_ASSOCIATE = "Commercial associate"
    MATERNITY_LEAVE = "Maternity leave"
    PENSIONER = "Pensioner"
    STATE_SERVANT = "State servant"
    STUDENT = "Student"
    UNEMPLOYED = "Unemployed"
    WORKING = "Working"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Businessman": "businessman", "Commercial associate": "commercial_associate", "Maternity leave": "maternity_leave", "Pensioner": "pensioner", "State servant": "state_servant", "Student": "student", "Unemployed": "unemployed", "Working": "working"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"businessman": "Businessman", "commercial_associate": "Commercial associate", "maternity_leave": "Maternity leave", "pensioner": "Pensioner", "state_servant": "State servant", "student": "Student", "unemployed": "Unemployed", "working": "Working"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Businessman": "businessman", "Commercial associate": "commercial_associate", "Maternity leave": "maternity_leave", "Pensioner": "pensioner", "State servant": "state_servant", "Student": "student", "Unemployed": "unemployed", "Working": "working"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
