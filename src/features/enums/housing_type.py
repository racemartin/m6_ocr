"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : housing_type (Type logement)
Description : 
"""

from enum import Enum
from typing import Optional


class HousingTypeEnum(str, Enum):
    """
    Valeurs de Type logement.
    Mapping : valeur_metier → valeur_technique
    """

    HOUSE_APARTMENT = "House / apartment"
    RENTED_APARTMENT = "Rented apartment"
    WITH_PARENTS = "With parents"
    MUNICIPAL_APARTMENT = "Municipal apartment"
    OFFICE_APARTMENT = "Office apartment"
    CO_OP_APARTMENT = "Co-op apartment"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"House / apartment": "house_apartment", "Rented apartment": "rented", "With parents": "with_parents", "Municipal apartment": "municipal", "Office apartment": "office", "Co-op apartment": "coop"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"house_apartment": "House / apartment", "rented": "Rented apartment", "with_parents": "With parents", "municipal": "Municipal apartment", "office": "Office apartment", "coop": "Co-op apartment"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"House / apartment": "house_apartment", "Rented apartment": "rented", "With parents": "with_parents", "Municipal apartment": "municipal", "Office apartment": "office", "Co-op apartment": "coop"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
