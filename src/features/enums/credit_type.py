"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : credit_type (Type of Credit Bureau credit)
Description : 
"""

from enum import Enum
from typing import Optional


class CreditTypeEnum(str, Enum):
    """
    Valeurs de Type of Credit Bureau credit.
    Mapping : valeur_metier → valeur_technique
    """

    ANOTHER_TYPE_OF_LOAN = "Another type of loan"
    CAR_LOAN = "Car loan"
    CONSUMER_CREDIT = "Consumer credit"
    CREDIT_CARD = "Credit card"
    LOAN_FOR_BUSINESS_DEVELOPMENT = "Loan for business development"
    LOAN_FOR_WORKING_CAPITAL_REPLENISHMENT = "Loan for working capital replenishment"
    MICROLOAN = "Microloan"
    MORTGAGE = "Mortgage"
    REAL_ESTATE_LOAN = "Real estate loan"
    UNKNOWN_TYPE_OF_LOAN = "Unknown type of loan"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Another type of loan": "another_type_of_loan", "Car loan": "car_loan", "Consumer credit": "consumer_credit", "Credit card": "credit_card", "Loan for business development": "loan_for_business_development", "Loan for working capital replenishment": "loan_for_working_capital_replenishment", "Microloan": "microloan", "Mortgage": "mortgage", "Real estate loan": "real_estate_loan", "Unknown type of loan": "unknown_type_of_loan"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"another_type_of_loan": "Another type of loan", "car_loan": "Car loan", "consumer_credit": "Consumer credit", "credit_card": "Credit card", "loan_for_business_development": "Loan for business development", "loan_for_working_capital_replenishment": "Loan for working capital replenishment", "microloan": "Microloan", "mortgage": "Mortgage", "real_estate_loan": "Real estate loan", "unknown_type_of_loan": "Unknown type of loan"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Another type of loan": "another_type_of_loan", "Car loan": "car_loan", "Consumer credit": "consumer_credit", "Credit card": "credit_card", "Loan for business development": "loan_for_business_development", "Loan for working capital replenishment": "loan_for_working_capital_replenishment", "Microloan": "microloan", "Mortgage": "mortgage", "Real estate loan": "real_estate_loan", "Unknown type of loan": "unknown_type_of_loan"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
