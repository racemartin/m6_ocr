# =============================================================================
# OBJETS VALEUR DU DOMAINE — Scoring Crédit
# Objets immuables représentant des concepts métier fondamentaux.
# Aucune dépendance externe : ce module est pur Python standard.
# Enrichi avec ExplicationShap pour la transparence algorithmique.

# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
from dataclasses import dataclass  # Décorateur pour classes de données
from enum        import Enum       # Énumération typée


# =============================================================================
# ÉNUMÉRATION : DÉCISION DE CRÉDIT
# Représente la décision binaire issue du scoring.
# =============================================================================
class Decision(str, Enum):
    """Décision finale accordée ou refusée pour un dossier de crédit."""

    APPROUVE = "Approuvé"  # Crédit accordé (probabilité < seuil)
    REFUSE   = "Refusé"    # Crédit refusé  (probabilité ≥ seuil)


# =============================================================================
# OBJET VALEUR : SCORE DE RISQUE
# Encapsule la probabilité de défaut et garantit qu'elle reste dans [0, 1].
# =============================================================================
@dataclass(frozen=True)
class ScoreRisque:
    """
    Probabilité de défaut de paiement produite par le modèle.

    Attributs
    ---------
    valeur : float
        Probabilité entre 0.0 (risque nul) et 1.0 (défaut certain).

    Lève
    ----
    ValueError
        Si la valeur est hors de l'intervalle [0, 1].
    """

    valeur: float  # Probabilité de défaut — intervalle [0.0, 1.0]

    # -------------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Valide que la probabilité est bien comprise entre 0 et 1."""
        if not (0.0 <= self.valeur <= 1.0):
            raise ValueError(
                f"ScoreRisque invalide : {self.valeur!r} "
                f"doit être dans [0.0, 1.0]."
            )

    # -------------------------------------------------------------------------
    def vers_decision(self, seuil: float) -> Decision:
        """
        Convertit le score en décision binaire selon un seuil métier.

        Paramètres
        ----------
        seuil : float
            Valeur limite (ex. 0.35) — au-delà, le crédit est refusé.

        Retourne
        --------
        Decision
            APPROUVE si valeur < seuil, REFUSE sinon.
        """
        if self.valeur < seuil:
            return Decision.APPROUVE
        return Decision.REFUSE

# =============================================================================
# Value Object : Explication SHAP d'une feature
# Represente la contribution d'une feature originale a la decision finale.
# =============================================================================
@dataclass(frozen=True)
class ExplicationShap:
    """
    Contribution SHAP d'une feature originale (en espace client).
 
    Contrairement aux SHAP values brutes qui operent sur N features
    encodees (apres OneHotEncoder + StandardScaler), ce value object
    est exprime dans l'espace des 11 features originales envoyees
    par le client. L'agregation est realisee dans l'adaptateur ONNX.
 
    Attributes
    ----------
    nom_feature : str
        Nom de la feature en langage metier (ex : "taux_utilisation_credit").
    valeur_originale : str
        Valeur telle qu'envoyee par le client, en string lisible
        (ex : "0.45", "Locataire", "35 ans").
    impact_shap : float
        Contribution SHAP agregee sur l'espace original.
        Positif -> hausse le risque de defaut.
        Negatif -> baisse le risque de defaut.
    direction : str
        "hausse_risque" si impact_shap > 0, "baisse_risque" sinon,
        "neutre" si abs(impact) < epsilon.
    """
    nom_feature     : str    # Nom metier de la feature
    valeur_originale: str    # Valeur envoyee par le client (string)
    impact_shap     : float  # Contribution SHAP agregee
    direction       : str    # "hausse_risque" | "baisse_risque" | "neutre"
 
    @classmethod
    def construire(
        cls,
        nom_feature      : str,
        valeur_originale : object,
        impact_shap      : float,
        epsilon          : float = 0.001,
    ) -> "ExplicationShap":
        """
        Constructeur avec calcul automatique de la direction.
 
        Args:
            nom_feature      : Nom de la feature originale.
            valeur_originale : Valeur brute (int, float ou str).
            impact_shap      : Valeur SHAP agregee.
            epsilon          : Seuil en-dessous duquel l'impact est neutre.
 
        Returns:
            ExplicationShap avec direction calculee.
        """
        # -- Formatage de la valeur en string lisible -------------------------
        if isinstance(valeur_originale, float):
            val_str = f"{valeur_originale:.4g}"   # 0.45, 45000.0, 0.023...
        else:
            val_str = str(valeur_originale)
 
        # -- Calcul de la direction -------------------------------------------
        if abs(impact_shap) < epsilon:
            direction = "neutre"
        elif impact_shap > 0:
            direction = "hausse_risque"
        else:
            direction = "baisse_risque"
 
        return cls(
            nom_feature      = nom_feature,
            valeur_originale = val_str,
            impact_shap      = round(impact_shap, 5),
            direction        = direction,
        )
 
