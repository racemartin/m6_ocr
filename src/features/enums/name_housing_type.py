"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_housing_type (What is the housing situation)
Description : 
"""

from enum import Enum
from typing import Optional


class NameHousingTypeEnum(str, Enum):
    """
    Valeurs de What is the housing situation.
    Mapping : valeur_metier → valeur_technique
    """

    CO_OP_APARTMENT = "Co-op apartment"
    HOUSE_APARTMENT = "House / apartment"
    MUNICIPAL_APARTMENT = "Municipal apartment"
    OFFICE_APARTMENT = "Office apartment"
    RENTED_APARTMENT = "Rented apartment"
    WITH_PARENTS = "With parents"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Co-op apartment": "co_op_apartment", "House / apartment": "house_apartment", "Municipal apartment": "municipal_apartment", "Office apartment": "office_apartment", "Rented apartment": "rented_apartment", "With parents": "with_parents"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"co_op_apartment": "Co-op apartment", "house_apartment": "House / apartment", "municipal_apartment": "Municipal apartment", "office_apartment": "Office apartment", "rented_apartment": "Rented apartment", "with_parents": "With parents"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Co-op apartment": "co_op_apartment", "House / apartment": "house_apartment", "Municipal apartment": "municipal_apartment", "Office apartment": "office_apartment", "Rented apartment": "rented_apartment", "With parents": "with_parents"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
