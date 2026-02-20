"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : channel_type (Through which channel we acquired)
Description : 
"""

from enum import Enum
from typing import Optional


class ChannelTypeEnum(str, Enum):
    """
    Valeurs de Through which channel we acquired.
    Mapping : valeur_metier → valeur_technique
    """

    AP_CASH_LOAN = "AP+ (Cash loan)"
    CAR_DEALER = "Car dealer"
    CHANNEL_OF_CORPORATE_SALES = "Channel of corporate sales"
    CONTACT_CENTER = "Contact center"
    COUNTRY_WIDE = "Country-wide"
    CREDIT_AND_CASH_OFFICES = "Credit and cash offices"
    REGIONAL_LOCAL = "Regional / Local"
    STONE = "Stone"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"AP+ (Cash loan)": "ap_cash_loan", "Car dealer": "car_dealer", "Channel of corporate sales": "channel_of_corporate_sales", "Contact center": "contact_center", "Country-wide": "country_wide", "Credit and cash offices": "credit_and_cash_offices", "Regional / Local": "regional_local", "Stone": "stone"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"ap_cash_loan": "AP+ (Cash loan)", "car_dealer": "Car dealer", "channel_of_corporate_sales": "Channel of corporate sales", "contact_center": "Contact center", "country_wide": "Country-wide", "credit_and_cash_offices": "Credit and cash offices", "regional_local": "Regional / Local", "stone": "Stone"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"AP+ (Cash loan)": "ap_cash_loan", "Car dealer": "car_dealer", "Channel of corporate sales": "channel_of_corporate_sales", "Contact center": "contact_center", "Country-wide": "country_wide", "Credit and cash offices": "credit_and_cash_offices", "Regional / Local": "regional_local", "Stone": "stone"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
