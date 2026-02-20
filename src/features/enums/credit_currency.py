"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : credit_currency (Recoded currency of the Credit)
Description : 
"""

from enum import Enum
from typing import Optional


class CreditCurrencyEnum(str, Enum):
    """
    Valeurs de Recoded currency of the Credit.
    Mapping : valeur_metier → valeur_technique
    """

    CURRENCY_1 = "currency 1"
    CURRENCY_2 = "currency 2"
    CURRENCY_3 = "currency 3"
    CURRENCY_4 = "currency 4"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"currency 1": "currency_1", "currency 2": "currency_2", "currency 3": "currency_3", "currency 4": "currency_4"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"currency_1": "currency 1", "currency_2": "currency 2", "currency_3": "currency 3", "currency_4": "currency 4"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"currency 1": "currency_1", "currency 2": "currency_2", "currency 3": "currency_3", "currency 4": "currency_4"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
