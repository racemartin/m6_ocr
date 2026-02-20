"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : education_type (Éducation)
Description : 
"""

from enum import Enum
from typing import Optional


class EducationTypeEnum(str, Enum):
    """
    Valeurs de Éducation.
    Mapping : valeur_metier → valeur_technique
    """

    LOWER_SECONDARY = "Lower secondary"
    SECONDARY_SECONDARY_SPECIAL = "Secondary / secondary special"
    INCOMPLETE_HIGHER = "Incomplete higher"
    HIGHER_EDUCATION = "Higher education"
    ACADEMIC_DEGREE = "Academic degree"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Lower secondary": "1_lower_secondary", "Secondary / secondary special": "2_secondary", "Incomplete higher": "3_incomplete_higher", "Higher education": "4_higher", "Academic degree": "5_academic"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"1_lower_secondary": "Lower secondary", "2_secondary": "Secondary / secondary special", "3_incomplete_higher": "Incomplete higher", "4_higher": "Higher education", "5_academic": "Academic degree"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Lower secondary": "1_lower_secondary", "Secondary / secondary special": "2_secondary", "Incomplete higher": "3_incomplete_higher", "Higher education": "4_higher", "Academic degree": "5_academic"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
