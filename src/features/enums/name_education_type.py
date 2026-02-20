"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_education_type (Level of highest education the)
Description : 
"""

from enum import Enum
from typing import Optional


class NameEducationTypeEnum(str, Enum):
    """
    Valeurs de Level of highest education the.
    Mapping : valeur_metier → valeur_technique
    """

    ACADEMIC_DEGREE = "Academic degree"
    HIGHER_EDUCATION = "Higher education"
    INCOMPLETE_HIGHER = "Incomplete higher"
    LOWER_SECONDARY = "Lower secondary"
    SECONDARY_SECONDARY_SPECIAL = "Secondary / secondary special"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Academic degree": "academic_degree", "Higher education": "higher_education", "Incomplete higher": "incomplete_higher", "Lower secondary": "lower_secondary", "Secondary / secondary special": "secondary_secondary_special"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"academic_degree": "Academic degree", "higher_education": "Higher education", "incomplete_higher": "Incomplete higher", "lower_secondary": "Lower secondary", "secondary_secondary_special": "Secondary / secondary special"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Academic degree": "academic_degree", "Higher education": "higher_education", "Incomplete higher": "incomplete_higher", "Lower secondary": "lower_secondary", "Secondary / secondary special": "secondary_secondary_special"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
