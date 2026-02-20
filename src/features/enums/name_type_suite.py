"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_type_suite (Who accompanied client when applying)
Description : 
"""

from enum import Enum
from typing import Optional


class NameTypeSuiteEnum(str, Enum):
    """
    Valeurs de Who accompanied client when applying.
    Mapping : valeur_metier → valeur_technique
    """

    CHILDREN = "Children"
    FAMILY = "Family"
    GROUP_OF_PEOPLE = "Group of people"
    OTHER_A = "Other_A"
    OTHER_B = "Other_B"
    SPOUSE_PARTNER = "Spouse, partner"
    UNACCOMPANIED = "Unaccompanied"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Children": "children", "Family": "family", "Group of people": "group_of_people", "Other_A": "other_a", "Other_B": "other_b", "Spouse, partner": "spouse_partner", "Unaccompanied": "unaccompanied"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"children": "Children", "family": "Family", "group_of_people": "Group of people", "other_a": "Other_A", "other_b": "Other_B", "spouse_partner": "Spouse, partner", "unaccompanied": "Unaccompanied"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Children": "children", "Family": "family", "Group of people": "group_of_people", "Other_A": "other_a", "Other_B": "other_b", "Spouse, partner": "spouse_partner", "Unaccompanied": "unaccompanied"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
