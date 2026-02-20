"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_contract_type (Contract product type of the)
Description : 
"""

from enum import Enum
from typing import Optional


class NameContractTypeEnum(str, Enum):
    """
    Valeurs de Contract product type of the.
    Mapping : valeur_metier → valeur_technique
    """

    CASH_LOANS = "Cash loans"
    CONSUMER_LOANS = "Consumer loans"
    REVOLVING_LOANS = "Revolving loans"
    XNA = "XNA"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Cash loans": "cash_loans", "Consumer loans": "consumer_loans", "Revolving loans": "revolving_loans", "XNA": "xna"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"cash_loans": "Cash loans", "consumer_loans": "Consumer loans", "revolving_loans": "Revolving loans", "xna": "XNA"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Cash loans": "cash_loans", "Consumer loans": "consumer_loans", "Revolving loans": "revolving_loans", "XNA": "xna"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
