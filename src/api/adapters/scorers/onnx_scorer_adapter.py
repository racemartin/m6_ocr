# =============================================================================
# src/adapters/scorers/onnx_scorer_adapter.py
#
# Adaptateur ONNX avec explication SHAP intégrée.
#
# Pipeline complet par requête :
#
#   DemandeCredit (11 features originales avec strings)
#       |
#       v  _construire_dataframe()
#   DataFrame pandas (noms colonnes préprocesseur)
#       |
#       v  preprocesseur.transform()
#   X_float32 (N features encodées + normalisées)
#       |
#       v  onnxruntime.run()         + shap.TreeExplainer.shap_values()
#   probabilité (float)               shap_values (ndarray shape N)
#       |                                   |
#       v                                   v  _agreger_shap()
#   décision métier               shap_agrege (dict 11 features originales)
#       |                                   |
#       +-----------------------------------+
#       v
#   DecisionCredit avec explication SHAP en espace original
#
# SHAP TreeExplainer :
#   - Fonctionne sur le modèle sklearn original (LightGBM Pipeline)
#   - Chargé en mémoire au démarrage (pas de recalcul du background)
#   - ~5-20 ms par requête selon le nombre de features
#   - Retourne shap_values[1] : contributions pour la classe 1 (défaut)
#
# Prérequis dans model_artifact/ :
#   best_model.onnx           <- inférence rapide
#   preprocessor.pkl          <- ColumnTransformer de m6_ocr phase 2
#   best_model_lgbm.pkl       <- modèle LightGBM original pour SHAP
#   best_model_meta.json      <- métadonnées (seuil, run_id, auc)
# =============================================================================

# --- Bibliothèques standard ---------------------------------------------------
import json                                          # Lecture des métadonnées
import logging                                       # Journalisation
import os                                            # Nom du fichier courant
import time                                          # Mesure de la latence
import warnings                                      # Suppression des UserWarning
import inspect                                       # Introspection des frames
from   typing  import Dict, List, Optional           # Annotations de type

# --- Bibliothèques tierces : données -----------------------------------------
import joblib                                        # Chargement des .pkl
import numpy  as np                                  # Tenseurs ONNX
import pandas as pd                                  # DataFrame préprocesseur

# --- Bibliothèques tierces : ONNX Runtime ------------------------------------
import onnxruntime as ort                            # Inférence rapide ONNX

# --- Bibliothèques tierces : SHAP --------------------------------------------
import shap                                          # Explicabilité SHAP

# --- Domaine -----------------------------------------------------------------
from src.api.domain.entities      import DemandeCredit, DecisionCredit
from src.api.domain.value_objects import (
    ScoreRisque,
    ExplicationShap,
)
from src.api.ports.i_credit_scorer import ICreditScorer

# --- Configuration et outils -------------------------------------------------
from config                        import parametres, DOSSIER_ARTEFACT
from src.tools.rafael.log_tool     import LogTool

# -----------------------------------------------------------------------------
# Journalisation du module et suppression des handlers dupliqués
# -----------------------------------------------------------------------------
journalapp  = logging.getLogger(__name__)             # Logger standard Python
log         = LogTool(origin="adapter")               # Logger métier enrichi
NOM_FICHIER = os.path.basename(__file__)              # Nom court pour les logs

# Nettoyage des handlers racine pour éviter la duplication des messages
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


# =============================================================================
# MAPPING_COLONNES
#
# Associe les champs Python de DemandeCredit aux noms de colonnes attendus
# par le ColumnTransformer de m6_ocr lors du fit().
#
# Convention de nommage du préprocesseur de m6_ocr :
#   - Colonnes catégorielles OHE  : "ohe__<nom_colonne>_<valeur>"
#     ex. : "ohe__name_contract_type_cash_loans"
#   - Colonnes numériques (scaler): "ss__<nom_colonne>"
#     ex. : "ss__days_birth"
#
# Ce mapping sert de pont entre l'API métier et le pipeline sklearn.
# =============================================================================
MAPPING_COLONNES_V1 = {
    # Champ Python              : Nom colonne dans le préprocesseur (m6_ocr)
    "age"                     : "days_birth",
    "revenu"                  : "amt_income_total",
    "montant_pret"            : "amt_credit",
    "duree_pret_mois"         : "duree_pret_mois",
    "jours_retard_moyen"      : "avg_dpd_per_delinquency",
    "taux_incidents"          : "delinquency_ratio",
    "taux_utilisation_credit" : "credit_utilization_ratio",
    "nb_comptes_ouverts"      : "num_open_accounts",
    "type_residence"          : "name_housing_type",
    "objet_pret"              : "name_type_suite",
    "type_pret"               : "name_contract_type",
}

MAPPING_COLONNES = {
    # Scores Críticos
    "ext_source_1": "ext_source_1",
    "ext_source_2": "ext_source_2",
    "ext_source_3": "ext_source_3",

    # Categorías (Selectboxes)
    "type_pret": "name_contract_type",
    "objet_pret": "name_type_suite",
    "type_residence": "name_housing_type",
    "code_gender": "code_gender",
    "education_type": "name_education_type",

    # Datos Financieros
    "revenu": "amt_income_total",
    "montant_pret": "amt_credit",
    "amt_annuity": "amt_annuity",
    "amt_goods_price": "amt_goods_price",

    # Historial y Comportamiento
    "paymnt_ratio_mean": "install_payment_ratio_mean",
    "paymnt_delay_mean": "install_payment_delay_mean",
    "max_dpd": "install_dpd_max",

    # Perfil y Otros
    "age": "days_birth",
    "days_employed": "days_employed",
    "bureau_credit_total": "bureau_credit_count",
    "bureau_debt_mean": "bureau_amt_credit_sum_debt_mean",
    "pos_months_mean": "pos_months_balance_mean",
    "cc_drawings_mean": "cc_amt_drawings_current_mean",
    "cc_balance_mean": "cc_amt_balance_mean",
    "phone_change_days": "days_last_phone_change",
    "region_rating": "region_rating_client"
}

# Mapping inverse : nom colonne préprocesseur -> champ Python
MAPPING_INVERSE = {v: k for k, v in MAPPING_COLONNES.items()}

# Noms métier lisibles pour la réponse client (affichage front-end)
NOMS_METIER_V1 = {
    "age"                     : "Âge",
    "revenu"                  : "Revenu annuel",
    "montant_pret"            : "Montant du prêt",
    "duree_pret_mois"         : "Durée (mois)",
    "jours_retard_moyen"      : "Retard moyen (jours/incident)",
    "taux_incidents"          : "Taux d'incidents de paiement",
    "taux_utilisation_credit" : "Taux d'utilisation du crédit",
    "nb_comptes_ouverts"      : "Nombre de comptes actifs",
    "type_residence"          : "Type de résidence",
    "objet_pret"              : "Objet du prêt",
    "type_pret"               : "Type de prêt",
}

NOMS_METIER = {
    "ext_source_1": "Score Externe 1",
    "ext_source_2": "Score Externe 2",
    "ext_source_3": "Score Externe 3",
    "type_pret": "Type de prêt",
    "objet_pret": "Objet du prêt",
    "type_residence": "Type de résidence",
    "code_gender": "Genre",
    "education_type": "Niveau d'études",
    "revenu": "Revenu annuel",
    "montant_pret": "Montant du prêt",
    "amt_annuity": "Annuité",
    "amt_goods_price": "Prix du bien",
    "paymnt_ratio_mean": "Ratio de paiement",
    "paymnt_delay_mean": "Retard moyen (jours)",
    "max_dpd": "Retard max (DPD)",
    "age": "Âge",
    "days_employed": "Ancienneté pro",
    "bureau_credit_total": "Nombre de crédits actifs",
    "bureau_debt_mean": "Dette moyenne Bureau",
    "pos_months_mean": "Moyenne mois POS",
    "cc_drawings_mean": "Retraits CB moyens",
    "cc_balance_mean": "Solde CB moyen",
    "phone_change_days": "Ancienneté téléphone",
    "region_rating": "Note Région"
}


# ##############################################################################
# Classe OnnxScorerAdaptateur
# ##############################################################################
class OnnxScorerAdaptater(ICreditScorer):
    """
    Adaptateur d'inférence ONNX avec explication SHAP.

    Charge au démarrage (via lifespan FastAPI) :
        - Session ONNX Runtime      (inférence ~3 ms)
        - ColumnTransformer sklearn  (preprocessing)
        - Modèle LightGBM original   (pour SHAP TreeExplainer)
        - Background SHAP            (100 lignes du dataset de référence)

    Par requête :
        - Preprocessing + inférence ONNX
        - Calcul SHAP values sur le modèle LightGBM original
        - Agrégation des SHAP sur les features originales
        - Retour de la décision + explication lisible par le client

    Latence additionnelle SHAP : ~5 à 20 ms selon le modèle.
    Désactivable via parametres.activer_shap = False.
    """

    # =========================================================================
    def __init__(self) -> None:
        """Initialise l'adaptateur sans charger les ressources (lazy)."""
        self._session           : Optional[ort.InferenceSession] = None
        self._preprocesseur     : Optional[object]               = None
        self._modele_lgbm       : Optional[object]               = None
        self._explainer_shap    : Optional[shap.TreeExplainer]   = None
        self._background_shap   : Optional[np.ndarray]           = None
        self._nom_entree        : Optional[str]                  = None
        self._seuil             : float    = parametres.seuil_decision
        self._noms_features_enc : List[str] = []  # Noms bruts du préprocesseur
        self._nb_features       : int      = 0    # Renseigné au chargement
        self._columnas_originales: List[str] = [] # feature_names_in_ du preproc
        self._n_esperado_onnx   : int      = 0    # Nb features attendu par ONNX
        self._sobrantes         : List[str] = []  # Colonnes en trop vs ONNX

        # Mode de sortie SHAP :
        #   "probability" : valeurs dans [0, 1]
        #   "raw"         : log-odds → sigmoid appliqué dans _calculer_shap
        self._mode_shap         : str      = "probability"

    # =========================================================================
    def charger(self) -> None:
        """
        Charge toutes les ressources au démarrage du serveur.

        Ordre de chargement :
            1. Session ONNX Runtime      (inférence)
            2. ColumnTransformer sklearn  (preprocessing)
            3. Modèle LightGBM original   (pour SHAP)
            4. Background SHAP            (100 lignes de référence)
            5. TreeExplainer SHAP         (compilé une seule fois)
            6. Métadonnées                (seuil optimal, run_id, auc)
            7. Colonnes originales        (feature_names_in_ du préprocesseur)

        Lève
        ----
        FileNotFoundError
            Si best_model.onnx, preprocessor.pkl ou best_model_lgbm.pkl
            sont absents de model_artifact/.
        """
        log.START_CALL_MANAGER_FUNCTION(
            self.__class__.__name__,
            inspect.currentframe().f_code.co_name,
            "BEGIN",
        )

        # Idempotence : ne recharge pas si déjà initialisé
        if self._session is not None:
            journalapp.debug("Ressources déjà chargées — ignoré.")
            return

        # ---------------------------------------------------------------------
        # Résolution des chemins d'artefacts
        # ---------------------------------------------------------------------
        chemin_onnx    = DOSSIER_ARTEFACT / "best_model.onnx"
        chemin_preproc = DOSSIER_ARTEFACT / "preprocessor.pkl"
        chemin_lgbm    = DOSSIER_ARTEFACT / "best_model_lgbm.pkl"
        chemin_ref     = DOSSIER_ARTEFACT / "reference_data.csv"
        chemin_meta    = DOSSIER_ARTEFACT / "best_model_meta.json"
        chemin_dims    = DOSSIER_ARTEFACT / "model_input_dims.json"

        # Vérification de l'existence des fichiers obligatoires
        for chemin, label in [
            (chemin_onnx,    "Modèle ONNX"),
            (chemin_preproc, "Préprocesseur"),
            (chemin_lgbm,    "Modèle LightGBM (pour SHAP)"),
        ]:
            if not chemin.exists():
                raise FileNotFoundError(
                    f"{label} introuvable : {chemin}\n"
                    "Exécutez : python scripts/export_best_model.py"
                )

        # ---------------------------------------------------------------------
        # charger Étape 1 : Session ONNX Runtime
        # ---------------------------------------------------------------------
        log.STEP(6, "charger Étape 1 : Session ONNX Runtime", chemin_onnx)

        providers_dispo = ort.get_available_providers()
        providers       = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers_dispo
            else ["CPUExecutionProvider"]
        )

        options                    = ort.SessionOptions()
        options.log_severity_level = 3               # Réduit le bruit ONNX

        self._session    = ort.InferenceSession(
            str(chemin_onnx),
            sess_options = options,
            providers    = providers,
        )
        self._nom_entree      = self._session.get_inputs()[0].name
        self._n_esperado_onnx = self._session.get_inputs()[0].shape[1]

        log.DEBUG_PARAMETER_VALUE(
            "provider actif",
            self._session.get_providers()[0],
        )
        log.DEBUG_PARAMETER_VALUE("nom entrée ONNX",    self._nom_entree)
        log.DEBUG_PARAMETER_VALUE("features attendues", self._n_esperado_onnx)

        # ---------------------------------------------------------------------
        # charger Étape 2 : ColumnTransformer sklearn (préprocesseur)
        # ---------------------------------------------------------------------
        log.STEP(6, "charger Étape 2 : Préprocesseur sklearn", chemin_preproc)

        self._preprocesseur     = joblib.load(chemin_preproc)
        self._noms_features_enc = self._extraire_noms_features_encodees()
        self._nb_features       = len(self._noms_features_enc)

        log.DEBUG_PARAMETER_VALUE(
            "features après encodage",
            self._nb_features,
        )
        noms_apercu = self._noms_features_enc[:15]
        log.DEBUG_PARAMETER_VALUE("15 premiers noms encodés", noms_apercu)

        # ---------------------------------------------------------------------
        # charger Étape 3 : Alignement JSON ↔ ONNX (détection sobrantes)
        #
        # Le fichier model_input_dims.json peut contenir N+1 noms si le
        # préprocesseur a été ré-entraîné après l'export ONNX.
        # On calcule la liste recoupée pour garantir shape == _n_esperado_onnx.
        # ---------------------------------------------------------------------
        log.STEP(6, "charger Étape 3 : Alignement JSON ↔ ONNX", chemin_dims)

        if chemin_dims.exists():
            with open(chemin_dims, "r", encoding="utf-8") as f:
                config_cols    = json.load(f)
            noms_json          = config_cols["feature_names"]
            nb_json            = len(noms_json)
            difference         = nb_json - self._n_esperado_onnx

            if difference > 0:
                # Les colonnes en trop sont retirées lors du transform
                self._sobrantes         = noms_json[-difference:]
                self._noms_features_enc = noms_json[:self._n_esperado_onnx]
                log.DEBUG_PARAMETER_VALUE(
                    "colonnes sobrantes détectées",
                    self._sobrantes,
                )
            else:
                self._noms_features_enc = noms_json

            log.DEBUG_PARAMETER_VALUE("nb noms JSON",   nb_json)
            log.DEBUG_PARAMETER_VALUE("nb noms alignés", len(self._noms_features_enc))

        # ---------------------------------------------------------------------
        # charger Étape 4 : Modèle LightGBM original (requis par SHAP)
        # ---------------------------------------------------------------------
        log.STEP(6, "charger Étape 4 : Modèle LightGBM (SHAP)", chemin_lgbm)

        self._modele_lgbm = joblib.load(chemin_lgbm)

        # ---------------------------------------------------------------------
        # charger Étape 5 : Background SHAP (référence 100 lignes)
        #
        # Un background dataset permet à TreeExplainer d'estimer l'impact
        # moyen de chaque feature par rapport à une distribution de référence.
        # 100 lignes offrent un bon compromis stabilité / vitesse.
        # ---------------------------------------------------------------------
        log.STEP(6, "charger Étape 5 : Background SHAP", chemin_ref)

        if chemin_ref.exists():
            df_ref                = pd.read_csv(chemin_ref)
            X_ref                 = df_ref.values
            if hasattr(X_ref, "toarray"):
                X_ref             = X_ref.toarray()
            nb_bg                 = min(100, len(X_ref))
            self._background_shap = shap.sample(X_ref, nb_bg)

            log.DEBUG_PARAMETER_VALUE("lignes background SHAP", nb_bg)
        else:
            # Sans référence, SHAP fonctionne mais les valeurs sont moins précises
            log.DEBUG_PARAMETER_VALUE(
                "background SHAP",
                "reference_data.csv absent — précision réduite",
            )
            self._background_shap = None

        # ---------------------------------------------------------------------
        # charger Étape 6 : TreeExplainer SHAP (compilé une seule fois)
        #
        # Tentative 1 — mode "probability" avec background data.
        # Échoue si le modèle LightGBM contient des splits catégoriels natifs.
        #
        # Tentative 2 — mode "raw" (tree_path_dependent) sans background.
        # Contraintes : data=None, model_output="raw" (log-odds).
        # La conversion log-odds → [0,1] est faite dans _calculer_shap.
        # ---------------------------------------------------------------------
        log.STEP(6, "charger Étape 6 : TreeExplainer SHAP")

        try:
            self._explainer_shap = shap.TreeExplainer(
                model         = self._modele_lgbm,
                data          = self._background_shap,
                feature_names = self._noms_features_enc,
                model_output  = "probability",
            )
            self._mode_shap = "probability"
            log.DEBUG_PARAMETER_VALUE("mode SHAP", "probability (standard)")

        except Exception as err_std:
            # Repli obligatoire pour les modèles avec catégories natives LightGBM
            log.DEBUG_PARAMETER_VALUE(
                "mode standard échoué — repli",
                str(err_std)[:80],
            )
            self._explainer_shap = shap.TreeExplainer(
                model                = self._modele_lgbm,
                data                 = None,
                feature_perturbation = "tree_path_dependent",
                model_output         = "raw",
            )
            self._mode_shap = "raw"
            log.DEBUG_PARAMETER_VALUE("mode SHAP", "raw (tree_path_dependent)")

        # ---------------------------------------------------------------------
        # charger Étape 7 : Métadonnées (seuil, run_id, auc)
        # ---------------------------------------------------------------------
        log.STEP(6, "charger Étape 7 : Métadonnées", chemin_meta)

        if chemin_meta.exists():
            with open(chemin_meta, encoding="utf-8") as f:
                meta       = json.load(f)
            seuil_meta     = meta.get("seuil_optimal", 0.0)
            if seuil_meta > 0:
                self._seuil = seuil_meta

            log.DEBUG_PARAMETER_VALUE(
                "métadonnées",
                "run_id=%s | auc=%.4f | seuil=%.3f" % (
                    meta.get("run_id", "?"),
                    meta.get("roc_auc", 0.0),
                    self._seuil,
                ),
                )

        # ---------------------------------------------------------------------
        # charger Étape 8 : Colonnes originales (feature_names_in_)
        #
        # Le préprocesseur expose la liste exacte des colonnes qu'il a vues
        # lors du fit(). Cette liste est utilisée dans _construire_dataframe
        # pour initialiser le DataFrame à zéro avant l'injection des données.
        # ---------------------------------------------------------------------
        log.STEP(6, "charger Étape 8 : Colonnes originales (feature_names_in_)")

        if hasattr(self._preprocesseur, "feature_names_in_"):
            self._columnas_originales = list(
                self._preprocesseur.feature_names_in_
            )
            nb_cols = len(self._columnas_originales)
            apercu  = self._columnas_originales[:15]
            log.DEBUG_PARAMETER_VALUE("nb colonnes originales", nb_cols)
            log.DEBUG_PARAMETER_VALUE("15 premières colonnes",  apercu)
        else:
            journalapp.error(
                "Le préprocesseur ne possède pas 'feature_names_in_'. "
                "Vérifiez la version de scikit-learn (>= 1.0 requis)."
            )

        log.FINISH_CALL_MANAGER_FUNCTION(
            self.__class__.__name__,
            inspect.currentframe().f_code.co_name,
            "FINISH",
        )

    # =========================================================================
    @property
    def est_pret(self) -> bool:
        """True si toutes les ressources critiques sont chargées."""
        return (
                self._session        is not None
                and self._preprocesseur  is not None
                and self._explainer_shap is not None
        )

    # =========================================================================
    def predire(self, demande: "DemandeCredit") -> "DecisionCredit":
        """
        Inférence complète avec explication SHAP.

        Séquence :
            1. Construire le DataFrame (mapping champs → colonnes)
            2. Preprocessing sklearn (OHE + scaling)
            3. Inférence ONNX Runtime (probabilité + latence)
            4. Extraction de la probabilité de défaut
            5. Calcul SHAP values sur le modèle LightGBM original
            6. Agrégation SHAP : N features encodées → 11 originales
            7. Construction de la DecisionCredit enrichie

        Args:
            demande : DemandeCredit avec les 11 features brutes.

        Returns:
            DecisionCredit avec probabilité, décision, latence
            et liste d'ExplicationShap triées par |impact| décroissant.
        """
        t0 = time.perf_counter()

        log.START_CALL_MANAGER_FUNCTION(
            self.__class__.__name__,
            inspect.currentframe().f_code.co_name,
            "BEGIN",
        )

        if not self.est_pret:
            raise RuntimeError(
                "OnnxScorerAdaptateur non initialisé. "
                "Appelez charger() avant d'effectuer des prédictions."
            )

        # ---------------------------------------------------------------------
        # predire Étape 1 : Construction du DataFrame d'entrée
        #
        # Convertit les champs Python de DemandeCredit en un DataFrame dont
        # les colonnes correspondent exactement à feature_names_in_ du
        # préprocesseur. Les colonnes non renseignées sont initialisées à 0.
        # ---------------------------------------------------------------------
        log.STEP(6, "predire Étape 1 : Construction du DataFrame d'entrée")

        df_entree = self._construire_dataframe(demande)

        log.DEBUG_PARAMETER_VALUE("nb colonnes df_entree", len(df_entree.columns))
        log.DEBUG_PARAMETER_VALUE(
            "15 premières colonnes",
            list(df_entree.columns[:15]),
        )

        t1 = time.perf_counter()

        # ---------------------------------------------------------------------
        # predire Étape 2 : Transformation sklearn (OHE + scaling)
        #
        # Le ColumnTransformer applique en séquence :
        #   - OneHotEncoder sur les colonnes catégorielles
        #   - StandardScaler (ou autre) sur les colonnes numériques
        # La matrice résultante peut être sparse → conversion en dense.
        # Les UserWarning sklearn (imputation) sont silencieux en production.
        # ---------------------------------------------------------------------
        log.STEP(6, "predire Étape 2 : Transformation sklearn (OHE + scaling)")

        debut_sk = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category = UserWarning,
                    module   = "sklearn.impute",
                )
                X_transforme = self._preprocesseur.transform(df_entree)
        except Exception as erreur:
            journalapp.critical(
                "Échec preprocessing : %s", erreur
            )
            raise ValueError(
                f"Échec preprocessing : {erreur}\n"
                f"Colonnes fournies : {list(df_entree.columns)}"
            ) from erreur

        duree_sk_ms = (time.perf_counter() - debut_sk) * 1000.0

        # Conversion sparse → dense si nécessaire (certains pipelines sklearn)
        if hasattr(X_transforme, "toarray"):
            X_transforme = X_transforme.toarray()

        # Suppression des colonnes sobrantes détectées lors du chargement
        if self._sobrantes:
            noms_actuels     = list(self._preprocesseur.get_feature_names_out())
            indices_a_retirer = [
                noms_actuels.index(col)
                for col in self._sobrantes
                if col in noms_actuels
            ]
            if indices_a_retirer:
                X_transforme = np.delete(
                    X_transforme, indices_a_retirer, axis=1
                )
                log.DEBUG_PARAMETER_VALUE(
                    "colonnes sobrantes retirées",
                    len(indices_a_retirer),
                )

        X_float32 = X_transforme.astype(np.float32)

        log.DEBUG_PARAMETER_VALUE("durée sklearn (ms)", f"{duree_sk_ms:.3f}")
        log.DEBUG_PARAMETER_VALUE("shape X_float32",    X_float32.shape)

        t2 = time.perf_counter()

        # ---------------------------------------------------------------------
        # predire Étape 3 : Inférence ONNX Runtime
        #
        # La session ONNX exécute le modèle sur le vecteur encodé.
        # La latence est mesurée en millisecondes pour le monitoring.
        # ---------------------------------------------------------------------
        log.STEP(6, "predire Étape 3 : Inférence ONNX Runtime")

        debut_onnx = time.perf_counter()
        resultats  = self._session.run(
            None,
            {self._nom_entree: X_float32},
        )
        latence_ms = (time.perf_counter() - debut_onnx) * 1000.0

        log.DEBUG_PARAMETER_VALUE("latence ONNX (ms)", f"{latence_ms:.3f}")



        # ---------------------------------------------------------------------
        # predire Étape 4 : Extraction de la probabilité de défaut
        #
        # La sortie ONNX peut prendre 3 formats selon la version skl2onnx :
        #   - zipmap=False : ndarray shape (1, 2)
        #   - zipmap=True  : liste de dicts [{0: p0, 1: p1}]
        #   - sortie unique: resultats[0][0]
        # La méthode _extraire_probabilite gère les 3 cas.
        # ---------------------------------------------------------------------
        log.STEP(6, "predire Étape 4 : Extraction probabilité de défaut")

        probabilite_defaut = self._extraire_probabilite(resultats)

        log.DEBUG_PARAMETER_VALUE(
            "probabilité défaut",
            f"{probabilite_defaut:.4f}",
        )

        t3 = time.perf_counter()

        # ---------------------------------------------------------------------
        # predire Étape 5 : Calcul SHAP values
        #
        # Les SHAP values quantifient la contribution de chaque feature encodée
        # à la probabilité de défaut. Les avertissements LightGBM sont captés
        # pour ne pas polluer les logs de production.
        # ---------------------------------------------------------------------
        log.STEP(6, "predire Étape 5 : Calcul SHAP values")

        with warnings.catch_warnings(record=True) as avertissements:
            warnings.simplefilter("always")
            shap_values = self._calculer_shap(X_transforme)

            for avert in avertissements:
                msg = str(avert.message)
                if "LightGBM binary classifier" in msg:
                    log.DEBUG_PARAMETER_VALUE("avis SHAP", msg[:80])

        log.DEBUG_PARAMETER_VALUE(
            "nb SHAP values calculées",
            len(shap_values),
        )

        # ---------------------------------------------------------------------
        # predire Étape 6 : Agrégation SHAP → espace original
        #
        # Les features OHE génèrent plusieurs colonnes par variable catégorielle.
        # L'agrégation somme les contributions pour retrouver l'impact net
        # de la variable originale (ex. "type_residence" = somme de ses OHE).
        # ---------------------------------------------------------------------
        log.STEP(6, "predire Étape 6 : Agrégation SHAP → espace original")

        explications = self._agreger_shap(shap_values, demande)

        nb_explications = len(explications)
        log.DEBUG_PARAMETER_VALUE("nb explications SHAP", nb_explications)

        if explications:
            top           = explications[0]
            log.DEBUG_PARAMETER_VALUE("top SHAP feature", top.nom_feature)
            log.DEBUG_PARAMETER_VALUE(
                "top SHAP impact",
                f"{top.impact_shap:+.4f}",
            )
        else:
            log.DEBUG_PARAMETER_VALUE("explications SHAP", "aucune disponible")

        t4 = time.perf_counter()
        # --- DEBUG DE RENDIMIENTO (Una línea por etapa) ---
        # Calculamos los tiempos en milisegundos (ms)
        dur_df      = (t1 - t0) * 1000
        dur_preproc = (t2 - t1) * 1000
        dur_onnx    = (t3 - t2) * 1000
        dur_shap    = (t4 - t3) * 1000
        dur_total   = (t4 - t0) * 1000

        # Registro individual por etapa para identificar cuellos de botella
        log.DEBUG_PARAMETER_VALUE("PERF_STEP_1_DATAFRAME" ,f"{dur_df:.3f} ms")
        log.DEBUG_PARAMETER_VALUE("PERF_STEP_2_PREPROC"   ,f"{dur_preproc:.3f} ms")
        log.DEBUG_PARAMETER_VALUE("PERF_STEP_3_ONNX_RUN"  ,f"{dur_onnx:.3f} ms")
        log.DEBUG_PARAMETER_VALUE("PERF_STEP_4_SHAP_GEN"  ,f"{dur_shap:.3f} ms")
        log.DEBUG_PARAMETER_VALUE("PERF_TOTAL_LATENCY"    ,f"{dur_total:.3f} ms")

        # ---------------------------------------------------------------------
        # predire Étape 7 : Construction de la DecisionCredit
        #
        # Le ScoreRisque encapsule la probabilité brute.
        # La décision métier (accord / refus) est déterminée par comparaison
        # avec le seuil optimal issu des métadonnées du modèle.
        # ---------------------------------------------------------------------
        log.STEP(6, "predire Étape 7 : Construction de la décision")

        score    = ScoreRisque(valeur=probabilite_defaut)
        decision = score.vers_decision(self._seuil)

        log.DEBUG_PARAMETER_VALUE("décision",   decision.value)
        log.DEBUG_PARAMETER_VALUE("seuil utilisé", f"{self._seuil:.3f}")
        log.DEBUG_PARAMETER_VALUE("latence totale (ms)", f"{latence_ms:.2f}")

        log.FINISH_CALL_MANAGER_FUNCTION(
            self.__class__.__name__,
            inspect.currentframe().f_code.co_name,
            "FINISH",
        )

        return DecisionCredit(
            id_demande        = demande.id,
            score             = score,
            decision          = decision,
            latence_ms        = round(latence_ms, 2),
            seuil_utilise     = self._seuil,
            explications_shap = explications,
        )

    # =========================================================================
    # Méthodes privées
    # =========================================================================

    # -------------------------------------------------------------------------
    def _construire_dataframe(self, demande: DemandeCredit) -> pd.DataFrame:
        """
        Convertit DemandeCredit en DataFrame pour le ColumnTransformer.

        Stratégie :
            1. Initialise toutes les colonnes à 0.0 (noms originaux du fit)
            2. Injecte les valeurs métier avec leur transformation d'unité
            3. Gère les Enums via getattr pour les features catégorielles
            4. Retourne le DataFrame avec les colonnes dans l'ordre exact du fit

        La conversion age → days_birth suit la convention Home Credit :
            days_birth = -(age_annees × 365)  [valeur négative]

        Args:
            demande : Entité domaine avec les features brutes du formulaire.

        Returns:
            DataFrame une ligne, colonnes = feature_names_in_ du préprocesseur.
        """
        # Toutes les colonnes initialisées à 0.0 (valeur neutre pour l'imputer)
        donnees = {col: 0.0 for col in self._columnas_originales}

        # Injection des valeurs numériques avec transformation d'unité si besoin
        mapeo = {
            "ext_source_1"                 : demande.ext_source_1,
            "ext_source_2"                 : demande.ext_source_2,
            "ext_source_3"                 : demande.ext_source_3,
            "install_payment_ratio_mean"   : demande.paymnt_ratio_mean,
            "days_birth"                   : float(-(demande.age * 365)),
            "cc_amt_drawings_current_mean" : demande.cc_drawings_mean,
            "install_payment_delay_mean"   : demande.paymnt_delay_mean,
            "pos_months_balance_mean"      : demande.pos_months_mean,
            "amt_goods_price"              : demande.goods_price,
            "bureau_amt_credit_sum_total"  : demande.bureau_credit_total,
            "install_dpd_max"              : demande.max_dpd,
            "amt_credit"                   : demande.amt_credit,
            "amt_annuity"                  : demande.amt_annuity,
            "cc_amt_balance_mean"          : demande.cc_balance_mean,
            "days_employed"                : float(-(demande.years_employed * 365)),
            "days_last_phone_change"       : demande.phone_change_days,
            "region_rating_client"         : float(demande.region_rating),
            "bureau_amt_credit_sum_debt_mean": demande.bureau_debt_mean,
        }
        donnees.update(mapeo)

        # Variables catégorielles : extraction de la valeur string depuis l'Enum
        donnees["name_education_type"] = getattr(
            demande.education_type, "value", demande.education_type
        )
        donnees["code_gender"] = getattr(
            demande.code_gender, "value", demande.code_gender
        )

        # Retour avec l'ordre exact imposé par le préprocesseur lors du fit
        return pd.DataFrame([donnees])[self._columnas_originales]

    # -------------------------------------------------------------------------
    def _calculer_shap(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule les SHAP values pour la classe 1 (défaut).

        Gère les deux formats de retour selon la version SHAP installée :
          - Nouvelle API (>= 0.40) : liste [ndarray_cl0, ndarray_cl1]
          - Ancienne API            : ndarray 2D shape (1, N)

        Mode "raw" (tree_path_dependent) :
            Applique sigmoid pour convertir les log-odds en [0, 1].

        En cas d'échec non bloquant, retourne un vecteur de zéros.

        Args:
            X : np.ndarray shape (1, N) — sortie du préprocesseur.

        Returns:
            SHAP values shape (N,) pour la classe 1.
        """
        try:
            sv = self._explainer_shap.shap_values(X)

            # ------------------------------------------------------------------
            # Extraction de la classe 1 selon le format de retour
            # ------------------------------------------------------------------
            if isinstance(sv, list):
                # Nouvelle API : [ndarray_classe0, ndarray_classe1]
                arr        = np.array(sv[1])
                sv_classe1 = arr[0] if arr.ndim == 2 else arr.flatten()

            elif isinstance(sv, np.ndarray) and sv.ndim == 2:
                # Ancienne API : shape (1, N) directement
                sv_classe1 = sv[0]

            else:
                sv_classe1 = np.array(sv).flatten()

            # ------------------------------------------------------------------
            # Conversion log-odds → [0, 1] pour le mode raw
            # ------------------------------------------------------------------
            if self._mode_shap == "raw":
                sv_classe1 = 1.0 / (1.0 + np.exp(-sv_classe1))

            return sv_classe1

        except Exception as erreur:
            journalapp.warning("Échec calcul SHAP (non bloquant) : %s", erreur)
            nb_features = (
                X.shape[1] if hasattr(X, "shape") and X.ndim == 2 else len(X)
            )
            return np.zeros(nb_features)

    # -------------------------------------------------------------------------
    def _agreger_shap(
            self,
            shap_values : np.ndarray,
            demande     : "DemandeCredit",
    ) -> List[ExplicationShap]:
        """
        Agrège les SHAP values de l'espace encodé vers l'espace original.

        Après OneHotEncoding, une feature catégorielle comme "type_residence"
        devient plusieurs colonnes binaires :
            name_housing_type_owned    -> shap = -0.05
            name_housing_type_rented   -> shap = +0.09
            name_housing_type_mortgage -> shap =  0.00

        L'agrégation somme ces contributions pour retrouver l'impact net
        de la feature originale "type_residence" = +0.04.

        Pour les features numériques scalées, la colonne correspond
        directement à la feature originale (ratio 1:1).

        Args:
            shap_values : SHAP values brutes shape (N,) — espace encodé.
            demande     : Demande originale pour les valeurs client.

        Returns:
            Liste d'ExplicationShap triée par |impact_shap| décroissant,
            limitée aux features ayant un impact non nul.
        """
        # Initialisation des impacts agrégés à 0.0 pour chaque feature originale
        impacts: Dict[str, float] = {
            nom_orig: 0.0 for nom_orig in MAPPING_COLONNES.keys()
        }

        # ------------------------------------------------------------------
        # Sommation des SHAP values par feature originale
        # ------------------------------------------------------------------
        for idx, nom_enc in enumerate(self._noms_features_enc):
            if idx >= len(shap_values):
                break

            valeur_shap = float(shap_values[idx])
            nom_orig    = self._trouver_feature_originale(nom_enc)

            if nom_orig in impacts:
                impacts[nom_orig] += valeur_shap   # Sommation OHE

        # ------------------------------------------------------------------
        # Valeurs originales du client pour chaque feature
        # ------------------------------------------------------------------
        valeurs_client = {
            "ext_source_1"              : demande.ext_source_1,
            "ext_source_2"              : demande.ext_source_2,
            "ext_source_3"              : demande.ext_source_3,
            "paymnt_ratio_mean"         : demande.paymnt_ratio_mean,
            "age"                       : demande.age,
            "cc_drawings_mean"          : demande.cc_drawings_mean,
            "paymnt_delay_mean"         : demande.paymnt_delay_mean,
            "pos_months_mean"           : demande.pos_months_mean,
            "goods_price"               : demande.goods_price,
            "education_type"            : demande.education_type,
            "code_gender"               : demande.code_gender,
            "bureau_credit_total"       : demande.bureau_credit_total,
            "max_dpd"                   : demande.max_dpd,
            "amt_credit"                : demande.amt_credit,
            "amt_annuity"               : demande.amt_annuity,
            "cc_balance_mean"           : demande.cc_balance_mean,
            "years_employed"            : demande.years_employed,
            "phone_change_days"         : demande.phone_change_days,
            "region_rating"             : demande.region_rating,
            "bureau_debt_mean"          : demande.bureau_debt_mean,
        }

        # ------------------------------------------------------------------
        # Construction des objets ExplicationShap
        # ------------------------------------------------------------------
        explications = []
        for nom_orig, impact in impacts.items():
            val_client = valeurs_client.get(nom_orig, "?")
            nom_metier = NOMS_METIER.get(nom_orig, nom_orig)

            expl = ExplicationShap.construire(
                nom_feature      = nom_metier,
                valeur_originale = val_client,
                impact_shap      = impact,
            )
            explications.append(expl)

        # Tri par |impact| décroissant → les features les plus déterminantes
        explications.sort(key=lambda e: abs(e.impact_shap), reverse=True)

        return explications

    # -------------------------------------------------------------------------
    def _trouver_feature_originale(self, nom_encode: str) -> str:
        """
        Retrouve la feature originale depuis le nom brut d'une colonne encodée.

        Convention de nommage du préprocesseur m6_ocr :
          - Numériques (StandardScaler) : "ss__<nom_colonne>"
            ex. : "ss__days_birth" -> "age"
          - Catégorielles (OHE)         : "ohe__<nom_colonne>_<valeur>"
            ex. : "ohe__name_contract_type_cash_loans" -> "type_pret"

        Stratégie :
            Extraction de la partie après le préfixe (ss__ ou ohe__), puis
            recherche dans MAPPING_COLONNES de la valeur qui est un préfixe
            de cette partie (match direct ou match par préfixe OHE).

        Args:
            nom_encode : Nom brut de la colonne encodée.

        Returns:
            Champ Python correspondant (ex. "type_residence"), ou "inconnu".
        """
        # Suppression du préfixe du step sklearn (ohe__, ss__, num__, cat__…)
        partie = nom_encode.split("__", 1)[1] if "__" in nom_encode else nom_encode

        # Match direct : feature numérique scalée (partie == nom_colonne)
        if partie in MAPPING_INVERSE:
            return MAPPING_INVERSE[partie]

        # Match par préfixe : feature OHE (partie commence par nom_colonne + "_")
        for nom_col_preproc, nom_orig in MAPPING_INVERSE.items():
            if (
                    partie.startswith(nom_col_preproc + "_")
                    or partie == nom_col_preproc
            ):
                return nom_orig

        return "inconnu"

    # -------------------------------------------------------------------------
    def _extraire_noms_features_encodees(self) -> List[str]:
        """
        Récupère les noms des features après transformation sklearn.

        Conserve les noms BRUTS tels que produits par get_feature_names_out()
        (ex. : "ohe__name_contract_type_cash_loans", "ss__days_birth").
        Le décodage des préfixes est délégué à _trouver_feature_originale().

        Fallback : si get_feature_names_out() échoue, génère des noms
        génériques "feature_0", "feature_1", … depuis la shape ONNX.

        Returns:
            Liste des noms bruts de colonnes dans l'espace transformé.
        """
        try:
            if hasattr(self._preprocesseur, "get_feature_names_out"):
                return list(self._preprocesseur.get_feature_names_out())
        except Exception as erreur:
            journalapp.warning(
                "get_feature_names_out indisponible : %s", erreur
            )

        # Fallback : noms génériques indexés depuis la dimension ONNX
        try:
            n_out = self._session.get_inputs()[0].shape[1]
            return [f"feature_{i}" for i in range(n_out or 0)]
        except Exception:
            return []

    # -------------------------------------------------------------------------
    @staticmethod
    def _extraire_probabilite(resultats: list) -> float:
        """
        Extrait la probabilité de défaut (classe 1) depuis la sortie ONNX.

        Gère les 3 formats possibles selon la version skl2onnx :
            Cas 1 — zipmap=False : probas ndarray shape (1, 2)
            Cas 2 — zipmap=True  : liste de dicts [{0: p0, 1: p1}]
            Cas 3 — sortie seule : resultats[0][0]

        Args:
            resultats : Sortie brute de ort.InferenceSession.run().

        Returns:
            Probabilité de défaut (classe 1) dans [0, 1].
        """
        if len(resultats) >= 2:
            probas = resultats[1]

            if isinstance(probas, np.ndarray) and probas.ndim == 2:
                # Cas 1 : ndarray (1, 2) — proba classe 1 en position [0][1]
                return float(probas[0][1])

            if isinstance(probas, list) and isinstance(probas[0], dict):
                # Cas 2 : liste de dicts — clé 1 (int ou str selon skl2onnx)
                d   = probas[0]
                cle = 1 if 1 in d else "1"
                return float(d[cle])

        # Cas 3 : sortie unique — repli sur le premier élément
        journalapp.warning(
            "Structure ONNX inattendue : len=%d — repli resultats[0][0]",
            len(resultats),
        )
        return float(resultats[0][0])
