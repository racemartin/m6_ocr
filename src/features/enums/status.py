"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : status (Status of Credit Bureau loan)
Description : 
"""

from enum import Enum
from typing import Optional


class StatusEnum(str, Enum):
    """
    Valeurs de Status of Credit Bureau loan.
    Mapping : valeur_metier → valeur_technique
    """

    V_0 = "0"
    V_1 = "1"
    V_2 = "2"
    V_3 = "3"
    V_4 = "4"
    V_5 = "5"
    C = "C"
    X = "X"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"0": "v_0", "1": "v_1", "2": "v_2", "3": "v_3", "4": "v_4", "5": "v_5", "C": "c", "X": "x"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"v_0": "0", "v_1": "1", "v_2": "2", "v_3": "3", "v_4": "4", "v_5": "5", "c": "C", "x": "X"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"0": "v_0", "1": "v_1", "2": "v_2", "3": "v_3", "4": "v_4", "5": "v_5", "C": "c", "X": "x"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
