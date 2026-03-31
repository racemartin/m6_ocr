# =============================================================================
# ADAPTATEUR JSONL — Implémentation du port IJournaliseurPredictions
# Écrit chaque décision de scoring dans predictions.jsonl, une ligne
# par prédiction. Ce fichier alimente Streamlit et Evidently AI.
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import json      # Sérialisation de chaque ligne du journal
import logging   # Journalisation interne de l'adaptateur
import threading # Verrou pour accès concurrent (multi-workers Uvicorn)
from pathlib import Path  # Gestion des chemins de fichier

# --- Domaine et port ----------------------------------------------------------
from src.api.domain.entities       import DemandeCredit, DecisionCredit  # Contrat
from src.api.ports.i_prediction_logger import IJournaliseurPredictions    # Interface

# --- Configuration globale ----------------------------------------------------
from config import parametres  # Chemin du fichier predictions.jsonl

# Configuration du logger applicatif pour ce module
journalapp = logging.getLogger(__name__)


# =============================================================================
# ADAPTATEUR : JOURNALISEUR JSONL
# =============================================================================
class JsonlJournaliseurAdapter(IJournaliseurPredictions):
    """
    Journalise chaque prédiction dans un fichier .jsonl (JSON Lines).

    Format du fichier
    -----------------
    Une ligne JSON par prédiction, structure :
    {
        "horodatage"            : "2025-03-27T10:23:11.452Z",
        "id_demande"            : "uuid-...",
        "age"                   : 35,
        "revenu"                : 45000.0,
        "montant_pret"          : 15000.0,
        "duree_pret_mois"       : 48,
        "jours_retard_moyen"    : 0.5,
        "taux_incidents"        : 0.02,
        "taux_utilisation_credit": 0.45,
        "nb_comptes_ouverts"    : 3,
        "type_residence"        : "Locataire",
        "objet_pret"            : "Personnel",
        "type_pret"             : "Non garanti",
        "probabilite_defaut"    : 0.7231,
        "decision"              : "Refusé",
        "latence_ms"            : 4.3
    }

    Sécurité concurrente
    --------------------
    Un threading.Lock protège l'écriture lorsque plusieurs workers
    Uvicorn (ou threads) écrivent simultanément dans le même fichier.

    Usage par les outils de monitoring
    -----------------------------------
    - monitoring/dashboard.py  : lit ce fichier pour les graphiques
    - scripts/drift_analysis.py : compare avec reference_data.csv
    """

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialise le chemin et le verrou d'écriture concurrente."""
        self._chemin_journal = parametres.chemin_predictions_jsonl
        self._verrou         = threading.Lock()  # Sécurité multi-threads

        # -- Création du répertoire parent si nécessaire ----------------------
        self._chemin_journal.parent.mkdir(parents=True, exist_ok=True)
        journalapp.info(
            "JournaliseurJSONL initialisé — fichier : %s",
            self._chemin_journal,
        )

    # =========================================================================
    # JOURNALISATION D'UNE PRÉDICTION
    # =========================================================================
    def journaliser(
        self,
        demande:  DemandeCredit,
        decision: DecisionCredit,
    ) -> None:
        """
        Écrit une ligne JSON dans predictions.jsonl.

        Utilise un verrou threading pour garantir l'intégrité du fichier
        en cas d'accès simultané (mode multi-workers Uvicorn).

        Paramètres
        ----------
        demande : DemandeCredit
            Données d'entrée du client (toutes les features).
        decision : DecisionCredit
            Résultat du scoring (score, décision, latence).

        Lève
        ----
        IOError
            Si l'écriture dans le fichier échoue.
        """
        # -- Construction du dictionnaire à sérialiser -------------------------
        ligne = {
            "horodatage"             : demande.horodatage.isoformat() + "Z",
            "id_demande"             : str(demande.id_demande),
            # -- Features numériques ------------------------------------------
            "age"                    : demande.age,
            "revenu"                 : demande.revenu,
            "montant_pret"           : demande.montant_pret,
            "duree_pret_mois"        : demande.duree_pret_mois,
            "jours_retard_moyen"     : demande.jours_retard_moyen,
            "taux_incidents"         : demande.taux_incidents,
            "taux_utilisation_credit": demande.taux_utilisation_credit,
            "nb_comptes_ouverts"     : demande.nb_comptes_ouverts,
            # -- Features catégorielles ---------------------------------------
            "type_residence"         : demande.type_residence,
            "objet_pret"             : demande.objet_pret,
            "type_pret"              : demande.type_pret,
            # -- Résultat du scoring ------------------------------------------
            "probabilite_defaut"     : round(decision.score.valeur, 6),
            "decision"               : decision.decision.value,
            "latence_ms"             : decision.latence_ms,
        }

        # -- Écriture thread-safe dans le fichier JSONL -----------------------
        with self._verrou:
            with open(
                self._chemin_journal, "a", encoding="utf-8"
            ) as fichier:
                fichier.write(json.dumps(ligne, ensure_ascii=False) + "\n")

        journalapp.debug(
            "Prédiction journalisée — id=%s, décision=%s",
            demande.id_demande,
            decision.decision.value,
        )
