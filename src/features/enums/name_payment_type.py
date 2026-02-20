"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_payment_type (Payment method that client chose)
Description : 
"""

from enum import Enum
from typing import Optional


class NamePaymentTypeEnum(str, Enum):
    """
    Valeurs de Payment method that client chose.
    Mapping : valeur_metier → valeur_technique
    """

    CASH_THROUGH_THE_BANK = "Cash through the bank"
    CASHLESS_FROM_THE_ACCOUNT_OF_THE_EMPLOYER = "Cashless from the account of the employer"
    NON_CASH_FROM_YOUR_ACCOUNT = "Non-cash from your account"
    XNA = "XNA"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Cash through the bank": "cash_through_the_bank", "Cashless from the account of the employer": "cashless_from_the_account_of_the_employer", "Non-cash from your account": "non_cash_from_your_account", "XNA": "xna"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"cash_through_the_bank": "Cash through the bank", "cashless_from_the_account_of_the_employer": "Cashless from the account of the employer", "non_cash_from_your_account": "Non-cash from your account", "xna": "XNA"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Cash through the bank": "cash_through_the_bank", "Cashless from the account of the employer": "cashless_from_the_account_of_the_employer", "Non-cash from your account": "non_cash_from_your_account", "XNA": "xna"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
