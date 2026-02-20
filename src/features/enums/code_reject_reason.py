"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : code_reject_reason (Why was the previous application)
Description : 
"""

from enum import Enum
from typing import Optional


class CodeRejectReasonEnum(str, Enum):
    """
    Valeurs de Why was the previous application.
    Mapping : valeur_metier → valeur_technique
    """

    CLIENT = "CLIENT"
    HC = "HC"
    LIMIT = "LIMIT"
    SCO = "SCO"
    SCOFR = "SCOFR"
    SYSTEM = "SYSTEM"
    VERIF = "VERIF"
    XAP = "XAP"
    XNA = "XNA"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"CLIENT": "client", "HC": "hc", "LIMIT": "limit", "SCO": "sco", "SCOFR": "scofr", "SYSTEM": "system", "VERIF": "verif", "XAP": "xap", "XNA": "xna"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"client": "CLIENT", "hc": "HC", "limit": "LIMIT", "sco": "SCO", "scofr": "SCOFR", "system": "SYSTEM", "verif": "VERIF", "xap": "XAP", "xna": "XNA"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"CLIENT": "client", "HC": "hc", "LIMIT": "limit", "SCO": "sco", "SCOFR": "scofr", "SYSTEM": "system", "VERIF": "verif", "XAP": "xap", "XNA": "xna"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
