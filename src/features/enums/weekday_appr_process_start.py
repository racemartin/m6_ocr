"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : weekday_appr_process_start (On which day of the)
Description : 
"""

from enum import Enum
from typing import Optional


class WeekdayApprProcessStartEnum(str, Enum):
    """
    Valeurs de On which day of the.
    Mapping : valeur_metier → valeur_technique
    """

    FRIDAY = "FRIDAY"
    MONDAY = "MONDAY"
    SATURDAY = "SATURDAY"
    SUNDAY = "SUNDAY"
    THURSDAY = "THURSDAY"
    TUESDAY = "TUESDAY"
    WEDNESDAY = "WEDNESDAY"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"FRIDAY": "friday", "MONDAY": "monday", "SATURDAY": "saturday", "SUNDAY": "sunday", "THURSDAY": "thursday", "TUESDAY": "tuesday", "WEDNESDAY": "wednesday"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"friday": "FRIDAY", "monday": "MONDAY", "saturday": "SATURDAY", "sunday": "SUNDAY", "thursday": "THURSDAY", "tuesday": "TUESDAY", "wednesday": "WEDNESDAY"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"FRIDAY": "friday", "MONDAY": "monday", "SATURDAY": "saturday", "SUNDAY": "sunday", "THURSDAY": "thursday", "TUESDAY": "tuesday", "WEDNESDAY": "wednesday"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
