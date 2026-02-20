"""
Auto-généré par generate_enums.py — NE PAS ÉDITER MANUELLEMENT
Attribut : name_seller_industry (The industry of the seller)
Description : 
"""

from enum import Enum
from typing import Optional


class NameSellerIndustryEnum(str, Enum):
    """
    Valeurs de The industry of the seller.
    Mapping : valeur_metier → valeur_technique
    """

    AUTO_TECHNOLOGY = "Auto technology"
    CLOTHING = "Clothing"
    CONNECTIVITY = "Connectivity"
    CONSTRUCTION = "Construction"
    CONSUMER_ELECTRONICS = "Consumer electronics"
    FURNITURE = "Furniture"
    INDUSTRY = "Industry"
    JEWELRY = "Jewelry"
    MLM_PARTNERS = "MLM partners"
    TOURISM = "Tourism"
    XNA = "XNA"

    # ─────────────────────────────────────────────────────────────

    @classmethod
    def to_technique(cls, metier_value: str) -> str:
        """Convertit une valeur métier en valeur technique (pour le modèle)."""
        mapping = {"Auto technology": "auto_technology", "Clothing": "clothing", "Connectivity": "connectivity", "Construction": "construction", "Consumer electronics": "consumer_electronics", "Furniture": "furniture", "Industry": "industry", "Jewelry": "jewelry", "MLM partners": "mlm_partners", "Tourism": "tourism", "XNA": "xna"}
        return mapping.get(str(metier_value).strip(), metier_value)

    @classmethod
    def to_metier(cls, technique_value: str) -> Optional[str]:
        """Convertit une valeur technique en valeur métier (pour l'API)."""
        reverse = {"auto_technology": "Auto technology", "clothing": "Clothing", "connectivity": "Connectivity", "construction": "Construction", "consumer_electronics": "Consumer electronics", "furniture": "Furniture", "industry": "Industry", "jewelry": "Jewelry", "mlm_partners": "MLM partners", "tourism": "Tourism", "xna": "XNA"}
        return reverse.get(str(technique_value).strip())

    @classmethod
    def all_metier_values(cls) -> list:
        """Liste de toutes les valeurs métier valides."""
        return [e.value for e in cls]

    @classmethod
    def all_technique_values(cls) -> list:
        """Liste de toutes les valeurs techniques correspondantes."""
        mapping = {"Auto technology": "auto_technology", "Clothing": "clothing", "Connectivity": "connectivity", "Construction": "construction", "Consumer electronics": "consumer_electronics", "Furniture": "furniture", "Industry": "industry", "Jewelry": "jewelry", "MLM partners": "mlm_partners", "Tourism": "tourism", "XNA": "xna"}
        return list(mapping.values())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Vérifie si une valeur métier est dans l'Enum."""
        return value in cls._value2member_map_
