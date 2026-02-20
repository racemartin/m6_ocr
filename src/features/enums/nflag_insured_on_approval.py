"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : nflag_insured_on_approval (Did the client requested insurance)
Description : 
"""

from enum import Enum
from typing import Optional


class NflagInsuredOnApprovalEnum(str, Enum):
    """
    Valeurs de Did the client requested insurance.
    Mapping : valeur_metier → valeur_technique
    """

    V_0_0 = "0.0"
    V_1_0 = "1.0"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"0.0": "v_0_0", "1.0": "v_1_0"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"v_0_0": "0.0", "v_1_0": "1.0"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"0.0": "v_0_0", "1.0": "v_1_0"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
