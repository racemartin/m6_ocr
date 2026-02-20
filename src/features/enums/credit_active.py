"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : credit_active (Status of the Credit Bureau)
Description : 
"""

from enum import Enum
from typing import Optional


class CreditActiveEnum(str, Enum):
    """
    Valeurs de Status of the Credit Bureau.
    Mapping : valeur_metier → valeur_technique
    """

    ACTIVE = "Active"
    BAD_DEBT = "Bad debt"
    CLOSED = "Closed"
    SOLD = "Sold"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Active": "active", "Bad debt": "bad_debt", "Closed": "closed", "Sold": "sold"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"active": "Active", "bad_debt": "Bad debt", "closed": "Closed", "sold": "Sold"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Active": "active", "Bad debt": "bad_debt", "Closed": "closed", "Sold": "sold"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
