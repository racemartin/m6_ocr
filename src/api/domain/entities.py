# =============================================================================
# ENTITÉS DU DOMAINE — Scoring Crédit
# Agrégats métier portant identité et cycle de vie d'une demande de crédit.
# Aucune dépendance externe : ce module est pur Python standard.
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
from __future__ import annotations        # Annotations différées (Python 3.9)
from dataclasses import dataclass, field  # Classes de données immuables
from datetime    import datetime          # Horodatage de la demande
from uuid        import UUID, uuid4       # Identifiant unique de la demande

# --- Objets valeur du domaine -------------------------------------------------
from src.api.domain.value_objects import Decision, ScoreRisque  # Concepts métier

from enum import Enum

class TypeResidence(str, Enum):
    HOUSE = "House / apartment"
    PARENTS = "With parents"
    RENTED = "Rented apartment"
    MUNICIPAL = "Municipal apartment"
    # Ajoutez les autres si nécessaire...

class ObjetPret(str, Enum):
    UNACCOMPANIED = "Unaccompanied"
    SPOUSE = "Spouse, partner"
    FAMILY = "Family"
    CHILDREN = "Children"

class TypePret(str, Enum):
    CASH = "Cash loans"
    REVOLVING = "Revolving loans"

# =============================================================================
# ENTITÉ : DEMANDE DE CRÉDIT
# Représente la requête entrante avec toutes les caractéristiques du client.
# C'est l'objet qui traverse toutes les couches de l'architecture hexagonale.
# =============================================================================
@dataclass
class DemandeCredit:
    """
    Demande de crédit soumise par un client.

    Regroupe les 11 features nécessaires au modèle de scoring,
    telles qu'elles sont définies dans le feature registry (m6_ocr).

    Attributs
    ---------
    id_demande : UUID
        Identifiant unique généré automatiquement à la création.
    age : int
        Âge du client en années.
    revenu : float
        Revenu annuel déclaré en euros.
    montant_pret : float
        Montant du prêt demandé en euros.
    duree_pret_mois : int
        Durée de remboursement souhaitée en mois.
    jours_retard_moyen : float
        Moyenne des jours de retard par incident de paiement.
    taux_incidents : float
        Ratio incidents / total paiements [0.0, 1.0].
    taux_utilisation_credit : float
        Utilisation du crédit disponible [0.0, 1.0].
    nb_comptes_ouverts : int
        Nombre de comptes bancaires actifs.
    type_residence : str
        Statut résidentiel : "Propriétaire", "Locataire", "Hypothèque".
    objet_pret : str
        Motif du prêt : "Éducation", "Immobilier", "Personnel", etc.
    type_pret : str
        Nature du prêt : "Garanti" ou "Non garanti".
    horodatage : datetime
        Date et heure de la soumission de la demande.
    """

    # -- Identité -------------------------------------------------------------
    id_demande: UUID     = field(default_factory=uuid4)  # UUID auto-généré
    horodatage: datetime = field(
        default_factory=datetime.utcnow               # UTC à la création
    )

    # -- Features numériques --------------------------------------------------
    age:                     int   = 0      # Âge du client (années)
    revenu:                  float = 0.0   # Revenu annuel (€)
    montant_pret:            float = 0.0   # Montant demandé (€)
    duree_pret_mois:         int   = 0      # Durée en mois
    jours_retard_moyen:      float = 0.0   # Moyenne jours retard / incident
    taux_incidents:          float = 0.0   # Ratio incidents [0, 1]
    taux_utilisation_credit: float = 0.0   # Utilisation crédit [0, 1]
    nb_comptes_ouverts:      int   = 0      # Comptes bancaires actifs

    # -- Features catégorielles -----------------------------------------------
    type_residence: TypeResidence = TypeResidence.HOUSE
    objet_pret:     ObjetPret     = ObjetPret.UNACCOMPANIED
    type_pret:      TypePret      = TypePret.CASH

    # -------------------------------------------------------------------------
    def vers_tableau_features(self) -> list:
        """
        Sérialise la demande en liste ordonnée pour l'inférence ONNX.

        L'ordre doit correspondre exactement à l'ordre des colonnes
        utilisé lors de l'entraînement du modèle (voir feature_registry.yaml).

        Retourne
        --------
        list
            Liste de 11 valeurs dans l'ordre attendu par le modèle.
        """
        return [
            self.age,
            self.revenu,
            self.montant_pret,
            self.duree_pret_mois,
            self.jours_retard_moyen,
            self.taux_incidents,
            self.taux_utilisation_credit,
            self.nb_comptes_ouverts,
            self.type_residence,
            self.objet_pret,
            self.type_pret,
        ]

# =============================================================================
# ENTITÉ : DÉCISION DE CRÉDIT
# Résultat enrichi produit par le use case après scoring et logging.
# =============================================================================
@dataclass(frozen=True)
class DecisionCredit:
    """
    Résultat complet du scoring pour une demande de crédit.

    Attributs
    ---------
    id_demande : UUID
        Référence vers la DemandeCredit ayant généré ce résultat.
    score : ScoreRisque
        Probabilité de défaut encapsulée et validée.
    decision : Decision
        APPROUVE ou REFUSE selon le seuil métier appliqué.
    latence_ms : float
        Temps d'inférence mesuré en millisecondes.
    """

    id_demande: UUID        # Référence à la demande d'origine
    score:      ScoreRisque # Probabilité de défaut validée
    decision:   Decision    # Décision binaire issue du seuil
    latence_ms: float       # Durée d'inférence (ms)
    seuil_utilise: float    # Le seuil appliqué pour cette décision
    explications_shap: List[ExplicationFeature] = field(default_factory=list)
