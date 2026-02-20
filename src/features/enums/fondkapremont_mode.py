"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : fondkapremont_mode (Normalized information about building...)
Description : 
"""

from enum import Enum
from typing import Optional


class FondkapremontModeEnum(str, Enum):
    """
    Valeurs de Normalized information about building....
    Mapping : valeur_metier → valeur_technique
    """

    NOT_SPECIFIED = "not specified"
    ORG_SPEC_ACCOUNT = "org spec account"
    REG_OPER_ACCOUNT = "reg oper account"
    REG_OPER_SPEC_ACCOUNT = "reg oper spec account"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"not specified": "not_specified", "org spec account": "org_spec_account", "reg oper account": "reg_oper_account", "reg oper spec account": "reg_oper_spec_account"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"not_specified": "not specified", "org_spec_account": "org spec account", "reg_oper_account": "reg oper account", "reg_oper_spec_account": "reg oper spec account"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"not specified": "not_specified", "org spec account": "org_spec_account", "reg oper account": "reg_oper_account", "reg oper spec account": "reg_oper_spec_account"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
