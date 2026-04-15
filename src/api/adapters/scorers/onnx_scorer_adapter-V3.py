# =============================================================================
# src/adapters/scorers/onnx_scorer_adapter.py
# Adaptateur ONNX avec explication SHAP integree.
#
# Pipeline complet par requete :
#   DemandeCredit (11 features originales avec strings)
#       |
#       v  _construire_dataframe()
#   DataFrame pandas (noms colonnes preprocesseur)
#       |
#       v  preprocesseur.transform()
#   X_float32 (N features encodees + normalisees)
#       |
#       v  onnxruntime.run()         + shap.TreeExplainer.shap_values()
#   probabilite (float)               shap_values (ndarray shape N)
#       |                                   |
#       v                                   v  _agreger_shap()
#   decision metier               shap_agrege (dict 11 features originales)
#       |                                   |
#       +-----------------------------------+
#       v
#   DecisionCredit avec explication SHAP en espace original
#
# SHAP TreeExplainer :
#   - Fonctionne sur le modele sklearn original (LightGBM Pipeline)
#   - Charge en memoire au demarrage (pas de recalcul du background)
#   - ~5-20 ms par requete selon le nombre de features
#   - Retourne shap_values[1] : contributions pour la classe 1 (defaut)
#
# Prerequis dans model_artifact/ :
#   best_model.onnx           <- inference rapide
#   preprocessor.pkl          <- ColumnTransformer de m6_ocr phase 2
#   best_model_lgbm.pkl       <- modele LightGBM original pour SHAP
#   best_model_meta.json      <- metadonnees (seuil, run_id, auc)
# =============================================================================

# --- Bibliotheques standard ---------------------------------------------------
import json                                           # Lecture metadonnees
import logging                                        # Journalisation
import time                                           # Mesure latence
from   typing  import Dict, List, Optional            # Annotations

# --- Bibliotheques tierces : donnees -----------------------------------------
import joblib                                         # Chargement .pkl
import numpy  as np                                   # Tenseurs ONNX
import pandas as pd                                   # DataFrame preprocesseur

# --- Bibliotheques tierces : ONNX Runtime ------------------------------------
import onnxruntime as ort                             # Inference rapide

# --- Bibliotheques tierces : SHAP --------------------------------------------
import shap                                           # Explicabilite SHAP

# --- Domaine -----------------------------------------------------------------
from src.api.domain.entities      import DemandeCredit, DecisionCredit
from src.api.domain.value_objects import (
    ScoreRisque,
    ExplicationShap,
)
from src.api.ports.i_credit_scorer import ICreditScorer

# --- Configuration -----------------------------------------------------------
from config import parametres, DOSSIER_ARTEFACT
import warnings
import os
import inspect
from src.tools.rafael.log_tool import LogTool

# Journalisation du module
journalapp = logging.getLogger(__name__)

log = LogTool(origin="adapter")
NOM_FICHIER = os.path.basename(__file__)

# =============================================================================
# Mapping : champs Python -> noms colonnes du ColumnTransformer de m6_ocr.
# Ajustez si les noms de colonnes different dans votre feature registry.
# =============================================================================
MAPPING_COLONNES = {
    "age"                     : "DAYS_BIRTH",
    "revenu"                  : "AMT_INCOME_TOTAL",
    "montant_pret"            : "AMT_CREDIT",
    "duree_pret_mois"              : "duree_pret_mois",
    "jours_retard_moyen"      : "avg_dpd_per_delinquency",
    "taux_incidents"          : "delinquency_ratio",
    "taux_utilisation_credit" : "credit_utilization_ratio",
    "nb_comptes_ouverts"      : "num_open_accounts",
    "type_residence"          : "residence_type",
    "objet_pret"              : "loan_purpose",
    "type_pret"               : "loan_type",
}

# Mapping inverse : nom colonne preprocesseur -> nom feature originale
MAPPING_INVERSE = {v: k for k, v in MAPPING_COLONNES.items()}

# Noms metier lisibles pour la reponse client (en francais)
NOMS_METIER = {
    "age"                     : "Age",
    "revenu"                  : "Revenu annuel",
    "montant_pret"            : "Montant du pret",
    "duree_pret_mois"              : "Duree (mois)",
    "jours_retard_moyen"      : "Retard moyen (jours/incident)",
    "taux_incidents"          : "Taux d incidents de paiement",
    "taux_utilisation_credit" : "Taux d utilisation du credit",
    "nb_comptes_ouverts"      : "Nombre de comptes actifs",
    "type_residence"          : "Type de residence",
    "objet_pret"              : "Objet du pret",
    "type_pret"               : "Type de pret",
}


# ##############################################################################
# Adaptateur : OnnxScorerAdaptateur
# ##############################################################################

# =============================================================================
class OnnxScorerAdaptater(ICreditScorer):
    """
    Adaptateur d'inference ONNX avec explication SHAP.

    Charge au demarrage (via lifespan FastAPI) :
        - Session ONNX Runtime     (inference ~3 ms)
        - ColumnTransformer sklearn (preprocessing)
        - Modele LightGBM original (pour SHAP TreeExplainer)
        - Background SHAP          (100 lignes du dataset de reference)

    Par requete :
        - Preprocessing + inference ONNX
        - Calcul SHAP values sur le modele LightGBM original
        - Agregation des SHAP sur les features originales
        - Retour de la decision + explication client-lisible

    Latence additionnelle SHAP : ~5 a 20 ms selon le modele.
    Desactivable via parametres.activer_shap = False.
    """

    # =========================================================================
    def __init__(self) -> None:
        """Initialise l'adaptateur sans charger les ressources (lazy)."""
        self._session        : Optional[ort.InferenceSession] = None
        self._preprocesseur  : Optional[object]               = None
        self._modele_lgbm    : Optional[object]               = None
        self._explainer_shap : Optional[shap.TreeExplainer]   = None
        self._background_shap: Optional[np.ndarray]           = None
        self._nom_entree     : Optional[str]                  = None
        self._seuil          : float  = parametres.seuil_decision
        self._noms_features_enc: List[str] = []  # Noms apres OHE
        self._nb_features    : int   = 0                   # Renseigne au chargement

    # =========================================================================
    def charger(self) -> None:
        """
        Charge toutes les ressources au demarrage du serveur.

        Ordre de chargement :
            1. Session ONNX Runtime (inference)
            2. ColumnTransformer sklearn (preprocessing)
            3. Modele LightGBM original (pour SHAP)
            4. Background SHAP (100 lignes reference)
            5. TreeExplainer SHAP (compile une fois)
            6. Metadonnees (seuil optimal, run_id)

        Leve
        ----
        FileNotFoundError
            Si best_model.onnx, preprocessor.pkl ou best_model_lgbm.pkl
            sont absents de model_artifact/.
        """
        log.START_CALL_MANAGER_FUNCTION(self.__class__.__name__, inspect.currentframe().f_code.co_name , "BEGING")
        
        if self._session is not None:
            journalapp.debug("Ressources deja chargees -- ignore.")
            return

        chemin_onnx    = DOSSIER_ARTEFACT / "best_model.onnx"
        chemin_preproc = DOSSIER_ARTEFACT / "preprocessor.pkl"
        chemin_lgbm    = DOSSIER_ARTEFACT / "best_model_lgbm.pkl"
        chemin_ref     = DOSSIER_ARTEFACT / "reference_data.csv"
        chemin_meta    = DOSSIER_ARTEFACT / "best_model_meta.json"

        # -- Verification d'existence -----------------------------------------
        for chemin, label in [
            (chemin_onnx,    "Modele ONNX"),
            (chemin_preproc, "Preprocesseur"),
            (chemin_lgbm,    "Modele LightGBM (pour SHAP)"),
        ]:
            if not chemin.exists():
                raise FileNotFoundError(
                    f"{label} introuvable : {chemin}\n"
                    "Executez : python scripts/export_best_model.py"
                )

        # -- 1. Session ONNX -------------------------------------------------
        log.STEP(6, "1. Chargement ONNX", chemin_onnx)
        
        providers_dispo = ort.get_available_providers()
        providers       = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers_dispo
            else ["CPUExecutionProvider"]
        )
        options                    = ort.SessionOptions()
        options.log_severity_level = 3

        self._session    = ort.InferenceSession(
            str(chemin_onnx),
            sess_options = options,
            providers    = providers,
        )
        self._nom_entree = self._session.get_inputs()[0].name
        log.DEBUG_PARAMETER_VALUE("ONNX pret | provide", providers[0])

        # -- 2. Preprocesseur sklearn ----------------------------------------
        log.STEP(6, "2. Chargement preprocesseur", chemin_preproc)
        
        self._preprocesseur = joblib.load(chemin_preproc)

        # -- Detection du nombre de features apres transformation ------------
        try:
            nb_sortie = self._session.get_inputs()[0].shape[1]
            if nb_sortie:
                self._nb_features = nb_sortie
        except (IndexError, TypeError):
            self._nb_features = 0
            
        # -- Recuperation des noms de features apres transformation ----------
        self._noms_features_enc = self._extraire_noms_features_encodees()
        log.DEBUG_PARAMETER_VALUE("Preprocesseur charge ", "Done!")
        log.DEBUG_PARAMETER_VALUE("Features apres encodage", self._nb_features)

        # -- 3. Modele LightGBM original pour SHAP ---------------------------
        log.STEP(6, "3. Chargement LightGBM (SHAP)", chemin_lgbm)
       
        self._modele_lgbm = joblib.load(chemin_lgbm)

        # -- 4. Background SHAP (sous-ensemble du dataset de reference) ------
        
        if chemin_ref.exists():
            log.STEP(6, "4. Chargement background (SHAP)", chemin_ref)
            df_ref             = pd.read_csv(chemin_ref)
            #X_ref              = self._preprocesseur.transform(df_ref)
            X_ref = df_ref.values
            if hasattr(X_ref, "toarray"):
                X_ref          = X_ref.toarray()
            # 100 lignes suffisent pour un background SHAP stable
            nb_bg              = min(100, len(X_ref))
            self._background_shap = shap.sample(X_ref, nb_bg)
            log.DEBUG_PARAMETER_VALUE("Background SHAP. lignes", nb_bg)
        else:
            # Pas de donnees de reference : background nul (moins precis)
            log.LEVEL_5_WARNING(
                "reference_data.csv absent -- "
                "background SHAP nul (moins precis)."
            )
            self._background_shap = None

        # -- 5. TreeExplainer SHAP (compilé une seule fois au démarrage) -----
        # Tentative 1 — mode standard : background data + output en probabilité.
        # Échoue si le modèle LightGBM contient des splits catégoriels natifs
        # (LightGBM "categorical_feature") : SHAP ne peut pas perturber ces
        # features avec un background dataset dans ce cas.
        log.STEP(6, "5. Initialisation TreeExplainer SHAP", "(mode standard)")
        
    
        try:
            self._explainer_shap = shap.TreeExplainer(
                model         = self._modele_lgbm,
                data          = self._background_shap,
                feature_names = self._noms_features_enc,
                model_output  = "probability",
            )
            self._mode_shap = "probability"
            log.LEVEL_7_INFO(NOM_FICHIER, "TreeExplainer SHAP initialisé — mode : probability.")
 
        except Exception as err_std:
            # Tentative 2 — mode compatible catégories natives LightGBM.
            # Contraintes imposées par SHAP dans ce cas :
            #   - data=None             (pas de background)
            #   - feature_perturbation  = "tree_path_dependent"
            #   - model_output          = "raw"  (log-odds bruts, pas probabilité)
            # La conversion log-odds → probabilité est faite dans _calculer_shap
            # via la fonction sigmoid.
            # SHAP opère alors sur le DataFrame ORIGINAL (avant OHE/scaling),
            # car LightGBM encode les catégories en interne.
            log.LEVEL_5_WARNING(NOM_FICHIER, f"Mode standard échoué ({err_std}) — repli sur tree_path_dependent.")

            self._explainer_shap = shap.TreeExplainer(
                model                = self._modele_lgbm,
                data                 = None,
                feature_perturbation = "tree_path_dependent",
                model_output         = "raw",
            )
            self._mode_shap = "raw"
            log.LEVEL_7_INFO(NOM_FICHIER, "TreeExplainer SHAP initialisé — mode : raw (tree_path_dependent).")
        
        # -- 6. Metadonnees --------------------------------------------------
        
        
        if chemin_meta.exists():
            log.STEP(6, "6. Metadonnees", chemin_meta)
            with open(chemin_meta, encoding="utf-8") as f:
                meta = json.load(f)
            seuil_meta = meta.get("seuil_optimal", 0.0)
            if seuil_meta > 0:
                self._seuil = seuil_meta

        log.DEBUG_PARAMETER_VALUE("Metadonnees", "run_id=%s | auc=%.4f | seuil=%.3f" % (
                        meta.get("run_id", "?"),
                        meta.get("roc_auc", 0.0),
                        self._seuil))
        
        # EXTRACTION AUTOMATIQUE DES COLONNES
        # 'feature_names_in_' contient les noms des colonnes originales
        # que le préprocesseur a vues lors du fit()
        
        log.STEP(6, "7. Colonnes attendues", chemin_meta)
        
        if hasattr(self._preprocesseur, "feature_names_in_"):
            self._columnas_originales = list(self._preprocesseur.feature_names_in_)
            log.DEBUG_PARAMETER_VALUE("Colonnes", f"{self._nb_features} attendues par le préprocesseur")
        else:
            # Si pour une raison quelconque l'attribut n'est pas défini (anciennes versions de sklearn)
            log.LEVEL_4_ERROR("Le préprocesseur ne possède pas l'attribut feature_names_in_")

        log.FINISH_CALL_MANAGER_FUNCTION(self.__class__.__name__, inspect.currentframe().f_code.co_name , "FINISH")


    # =========================================================================
    @property
    def est_pret(self) -> bool:
        """True si toutes les ressources sont chargees."""
        return (
            self._session is not None
            and self._preprocesseur is not None
            and self._explainer_shap is not None
        )

    # =========================================================================
    def predire(self, demande: "DemandeCredit") -> "DecisionCredit":
        """
        Inference complete avec explication SHAP.

        Sequence :
            1. Construire le DataFrame (mapping champs -> colonnes)
            2. Preprocessing sklearn (OHE + scaling)
            3. Inference ONNX Runtime (probabilite + latence)
            4. Calcul SHAP values sur le modele LightGBM original
            5. Agregation SHAP : N features encodees -> 11 originales
            6. Construction de la DecisionCredit enrichie

        Args:
            demande : DemandeCredit avec les 11 features brutes.

        Returns:
            DecisionCredit avec probabilite, decision, latence
            et liste d'ExplicationShap triees par |impact| decroissant.
        """
        log.START_CALL_MANAGER_FUNCTION(self.__class__.__name__, inspect.currentframe().f_code.co_name , "BEGING")

        
        if not self.est_pret:
            raise RuntimeError(
                "OnnxScorerAdaptateur non initialise. "
                "Appelez charger() avant d'effectuer des predictions."
            )

        # ---- Etape 1: DataFrame avec les noms attendus par le preprocesseur -
        log.STEP(6, "1. Construire dataframe depuis la demande")
        df_entree = self._construire_dataframe(demande)

        # ---- Etape 2: Transformation sklearn --------------------------------
        log.STEP(6, "2. Transformation sklearn")
        
        try:
            # Capturamos los avisos de sklearn para que no ensucien el terminal
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute")
                X_transforme = self._preprocesseur.transform(df_entree)
                
        except Exception as erreur:
            # Log de error crítico antes de lanzar la excepción
            log.LEVEL_3_CRITICAL(NOM_FICHIER, f"Echec preprocessing : {erreur}")
            raise ValueError(
                f"Echec preprocessing : {erreur}\n"
                f"Colonnes fournies : {list(df_entree.columns)}"
            ) from erreur
        
        # Conversión de sparse a dense si es necesario
        if hasattr(X_transforme, "toarray"):
            X_transforme = X_transforme.toarray()
        
        # --- AJUSTE DINÁMICO DE COLUMNAS ---
        # Si el modelo espera 232 y tenemos 233, recortamos la última (posible TARGET/ID)
        if X_transforme.shape[1] > self._nb_features:
            msg_ajuste = f"Eliminación de columna extra. De {X_transforme.shape[1]} a {self._nb_features}"
            
            # Usamos tu LogTool para que el aviso sea visible y con color
            log.LEVEL_5_WARNING(NOM_FICHIER, msg_ajuste)
            
            X_transforme = X_transforme[:, :self._nb_features]
        
        X_float32 = X_transforme.astype(np.float32)
        log.DEBUG_PARAMETER_VALUE("Shape final X", X_float32.shape)

        # ---- Etape 3: Inference ONNX + latence ------------------------------
        log.STEP(6, "3. Inference ONNX + latence")
        debut_ms  = time.perf_counter()
        resultats = self._session.run(
            None,
            {self._nom_entree: X_float32},
        )
        latence_ms = (time.perf_counter() - debut_ms) * 1000.0


        # ---- Etape 4: extraction de la probabilite de defaut ----------------
        log.STEP(6, "4. extraction de la probabilite de defaut")
        
        probabilite_defaut = self._extraire_probabilite(resultats)
        
        # ---- Etape 5: Calcul SHAP values ------------------------------------
        log.STEP(6, "5. Calcul SHAP values")
        
        # Activamos el modo "record" para interceptar los avisos en una lista
        with warnings.catch_warnings(record=True) as w:
            # No filtramos, queremos que se registre en 'w'
            shap_values = self._calculer_shap(X_transforme)
            
            # Revisamos si se generó algún aviso durante el cálculo
            for warning in w:
                msg = str(warning.message)
                # Filtramos para capturar solo el de LightGBM/TreeExplainer
                if "LightGBM binary classifier" in msg:
                    # Lo enviamos a tu LogTool con el formato que definimos
                    log.LEVEL_5_WARNING(NOM_FICHIER, f"Avis SHAP détecté : {msg}")

        # ---- Etape 6 : Agregation SHAP espace original -----------------------
        log.STEP(6, "6. Obtenir explications")
        explications = self._agreger_shap(shap_values, demande)

        if explications:
            # Iteramos sobre todas las explicaciones encontradas
            for i, exp in enumerate(explications, start=1):
                # Formateamos el nombre con un índice para que sea legible
                label_feat   = f"SHAP Top-{i} Feature"
                label_impact = f"SHAP Top-{i} Impact"
                
                log.DEBUG_PARAMETER_VALUE(label_feat,   exp.nom_feature)
                log.DEBUG_PARAMETER_VALUE(label_impact, f"{exp.impact_shap:+.4f}")
                
            # Opcional: un separador visual al final si hay muchas
            log.DEBUG_PARAMETER_VALUE("Total Features", len(explications))
        else:
            log.LEVEL_5_WARNING(NOM_FICHIER, "Aucune explication SHAP disponible")
            
        
        # ---- Etape 7: Construire la decision --------------------------------
        log.STEP(6, "7. Construire la decision")
        score = ScoreRisque(valeur=probabilite_defaut)
        
        decision = score.vers_decision(self._seuil)


        # -- Registro de la respuesta en LogTool --------------------------------------
        log.LEVEL_7_INFO(NOM_FICHIER, "Prediction")
        
        # Parámetros básicos
        log.DEBUG_PARAMETER_VALUE("probabilite_defaut" , f"{probabilite_defaut:.4f}")
        log.DEBUG_PARAMETER_VALUE("decision.value"     , decision.value)
        log.DEBUG_PARAMETER_VALUE("latence_ms"         , f"{latence_ms:.2f}")
        
        # Explicabilidad SHAP (Traducción de lo que hacía el journalapp)
        if explications:
            top_feat   = explications[0].nom_feature if explications else "?"
            top_impact = explications[0].impact_shap if explications else 0.0
            log.DEBUG_PARAMETER_VALUE("top_shap_name"   ,  top_feat)
            log.DEBUG_PARAMETER_VALUE("top_shap_impact" , f"+{top_impact:.3f}")
        else:
            log.DEBUG_PARAMETER_VALUE("top_shap",       "Aucune explication disponible")
    
        log.FINISH_CALL_MANAGER_FUNCTION(self.__class__.__name__, inspect.currentframe().f_code.co_name , "FINISH")

        return DecisionCredit(
            id_demande         = demande.id_demande,
            score              = score,
            decision           = decision,
            latence_ms         = round(latence_ms, 2),
            seuil_utilise      = self._seuil,
            explications_shap  = explications,
        )

    # -------------------------------------------------------------------------
    def _construire_dataframe_V1(
        self,
        demande: "DemandeCredit",
    ) -> pd.DataFrame:
        """
        Convertit DemandeCredit en DataFrame pour le ColumnTransformer.

        Applique la conversion age -> DAYS_BIRTH (negatif, convention
        Home Credit Default Risk) et mappe les noms de champs Python
        vers les noms de colonnes attendus par le preprocesseur.

        Args:
            demande : Entite domaine avec les 11 features originales.

        Returns:
            DataFrame une ligne avec les noms de colonnes preprocesseur.
        """
        ligne = {
            MAPPING_COLONNES["age"]                    : -(demande.age * 365),
            MAPPING_COLONNES["revenu"]                 : demande.revenu,
            MAPPING_COLONNES["montant_pret"]           : demande.montant_pret,
            MAPPING_COLONNES["duree_pret_mois"]             : demande.duree_pret_mois,
            MAPPING_COLONNES["jours_retard_moyen"]     : demande.jours_retard_moyen,
            MAPPING_COLONNES["taux_incidents"]         : demande.taux_incidents,
            MAPPING_COLONNES["taux_utilisation_credit"]: demande.taux_utilisation_credit,
            MAPPING_COLONNES["nb_comptes_ouverts"]     : demande.nb_comptes_ouverts,
            MAPPING_COLONNES["type_residence"]         : demande.type_residence,
            MAPPING_COLONNES["objet_pret"]             : demande.objet_pret,
            MAPPING_COLONNES["type_pret"]              : demande.type_pret,
        }
        return pd.DataFrame([ligne])

    def _construire_dataframe(self, demande: DemandeCredit) -> pd.DataFrame:
        # 1. Creamos un diccionario base con todas las columnas en 0.0 (floats)
        datos = {col: 0.0 for col in self._columnas_originales}

        # 2. Preparamos los valores específicos de la solicitud
        # Usamos getattr para manejar Enums de forma segura
        mapeo = {
            "days_birth"        : float(-(demande.age * 365)),
            "amt_income_total"  : float(demande.revenu),
            "amt_credit"        : float(demande.montant_pret),
            "name_housing_type" : getattr(demande.type_residence, "value", demande.type_residence),
            "name_type_suite"   : getattr(demande.objet_pret, "value", demande.objet_pret),
            "name_contract_type": getattr(demande.type_pret, "value", demande.type_pret),
        }

        # 3. Actualizamos el diccionario con los datos reales
        datos.update(mapeo)

        # 4. Creamos el DataFrame a partir de una lista de un solo diccionario
        # Al hacerlo así, Pandas no genera el FutureWarning
        return pd.DataFrame([datos])
    
    # -------------------------------------------------------------------------
    def _calculer_shap(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule les SHAP values pour la classe 1 (defaut).

        Le TreeExplainer de SHAP opère directement sur le modele LightGBM
        sklearn (arbre de decision exact, pas approximation lineaire).
        C'est beaucoup plus precis que KernelExplainer ou GradientExplainer.

        Args:
            X : Features transformees shape (1, N), dtype float64.

        Returns:
            SHAP values shape (N,) pour la classe 1.
            Chaque valeur represente la contribution de la feature i
            a la deviation par rapport a la prediction moyenne.
        """
        try:
            # shap_values retourne shape (1, N) pour une seule observation
            sv = self._explainer_shap.shap_values(X)

            # LightGBM binaire : sv est shape (1, N) directement
            if isinstance(sv, np.ndarray) and sv.ndim == 2:
                return sv[0]                    # shape (N,)

            # Certaines versions retournent une liste [classe_0, classe_1]
            if isinstance(sv, list) and len(sv) == 2:
                return np.array(sv[1])[0]      # classe 1, premiere obs

            # Fallback
            return np.array(sv).flatten()

        except Exception as erreur:
            journalapp.warning(
                "Echec calcul SHAP (non bloquant) : %s", erreur
            )
            # Retourner des zeros plutot que de bloquer la reponse
            nb_features = X.shape[1] if X.ndim == 2 else len(X)
            return np.zeros(nb_features)

    # -------------------------------------------------------------------------
    def _agreger_shap(
        self,
        shap_values : np.ndarray,
        demande     : "DemandeCredit",
    ) -> List[ExplicationShap]:
        """
        Agrege les SHAP values de l'espace encode vers l'espace original.

        Apres OneHotEncoding, une feature categorielle comme "type_residence"
        devient plusieurs colonnes binaires :
            residence_type_Owned    -> shap = -0.05
            residence_type_Rented   -> shap = +0.09
            residence_type_Mortgage -> shap =  0.00

        L'agregation consiste a sommer ces contributions pour retrouver
        l'impact net de la feature originale "type_residence" = +0.04.

        Pour les features numeriques scalees, la colonne correspond
        directement a la feature originale (ratio 1:1).

        Args:
            shap_values : SHAP values brutes shape (N,) -- espace encode.
            demande     : Demande originale pour les valeurs clients.

        Returns:
            Liste d'ExplicationShap triee par |impact_shap| decroissant,
            limitee aux features ayant un impact non nul.
        """
        # -- Construction d'un dictionnaire feature_originale -> impact_shap -
        impacts: Dict[str, float] = {
            nom_orig: 0.0 for nom_orig in MAPPING_COLONNES.keys()
        }

        # -- Iteration sur les features encodees avec leur SHAP value --------
        for idx, nom_enc in enumerate(self._noms_features_enc):
            if idx >= len(shap_values):
                break

            valeur_shap = float(shap_values[idx])

            # -- Recherche de la feature originale correspondante ------------
            nom_orig = self._trouver_feature_originale(nom_enc)

            if nom_orig in impacts:
                impacts[nom_orig] += valeur_shap   # Sommation des OHE
            # else : colonne inconnue (feature generee par le preprocesseur)
            # -> ignoree proprement

        # -- Valeurs originales du client pour chaque feature ----------------
        valeurs_client = {
            "age"                     : demande.age,
            "revenu"                  : demande.revenu,
            "montant_pret"            : demande.montant_pret,
            "duree_pret_mois"              : demande.duree_pret_mois,
            "jours_retard_moyen"      : demande.jours_retard_moyen,
            "taux_incidents"          : demande.taux_incidents,
            "taux_utilisation_credit" : demande.taux_utilisation_credit,
            "nb_comptes_ouverts"      : demande.nb_comptes_ouverts,
            "type_residence"          : demande.type_residence,
            "objet_pret"              : demande.objet_pret,
            "type_pret"               : demande.type_pret,
        }

        # -- Construction des ExplicationShap --------------------------------
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

        # -- Tri par |impact| decroissant -> les features les plus importantes
        explications.sort(key=lambda e: abs(e.impact_shap), reverse=True)

        return explications

    # -------------------------------------------------------------------------
    def _trouver_feature_originale(self, nom_encode: str) -> str:
        """
        Retrouve la feature originale depuis le nom d'une colonne encodee.

        Logique :
            - Si le nom est directement dans MAPPING_INVERSE : match direct
              (features numeriques scalees, ex: "AMT_CREDIT" -> "montant_pret")
            - Sinon, chercher un prefixe correspondant a un nom du mapping
              (features OHE, ex: "residence_type_Owned" -> "type_residence")

        Args:
            nom_encode : Nom de la colonne apres transformation sklearn.

        Returns:
            Nom de la feature originale correspondante, ou "inconnu".
        """
        # -- Match direct (features numeriques) ------------------------------
        if nom_encode in MAPPING_INVERSE:
            return MAPPING_INVERSE[nom_encode]

        # -- Match par prefixe (features OHE) --------------------------------
        # Ex: "residence_type_Owned" commence par "residence_type"
        for nom_col_preprocesseur, nom_orig in MAPPING_INVERSE.items():
            if nom_encode.startswith(nom_col_preprocesseur + "_"):
                return nom_orig
            if nom_encode.startswith(nom_col_preprocesseur):
                return nom_orig

        return "inconnu"

    # -------------------------------------------------------------------------
    def _extraire_noms_features_encodees(self) -> List[str]:
        """
        Recupere les noms des features apres transformation sklearn.

        Le ColumnTransformer sklearn expose get_feature_names_out()
        depuis sklearn >= 1.0. En cas d'echec, genere des noms generiques.

        Returns:
            Liste des noms de colonnes dans l'espace transforme.
        """
        try:
            if hasattr(self._preprocesseur, "get_feature_names_out"):
                noms = list(self._preprocesseur.get_feature_names_out())
                # Nettoyer les prefixes "num__" et "cat__" ajoutes par sklearn
                noms = [
                    n.replace("num__", "").replace("cat__", "")
                    .replace("remainder__", "")
                    for n in noms
                ]
                return noms
        except Exception as e:
            journalapp.warning(
                "get_feature_names_out indisponible : %s", e
            )

        # -- Fallback : noms generiques feature_0, feature_1, ... -----------
        try:
            n_out = self._session.get_inputs()[0].shape[1]
            return [f"feature_{i}" for i in range(n_out or 0)]
        except Exception:
            return []

    # -------------------------------------------------------------------------
    @staticmethod
    def _extraire_probabilite(resultats: list) -> float:
        """
        Extrait la probabilite de defaut (classe 1) depuis la sortie ONNX.

        Gere les 3 formats possibles selon la version skl2onnx :
            Cas 1 - zipmap=False : probas ndarray shape (1, 2)
            Cas 2 - zipmap=True  : liste de dicts [{0: p0, 1: p1}]
            Cas 3 - sortie seule : resultats[0][0]
        """
        if len(resultats) >= 2:
            probas = resultats[1]
            if isinstance(probas, np.ndarray) and probas.ndim == 2:
                return float(probas[0][1])
            if isinstance(probas, list) and isinstance(probas[0], dict):
                d   = probas[0]
                cle = 1 if 1 in d else "1"
                return float(d[cle])

        journalapp.warning(
            "Structure ONNX inattendue : len=%d -- repli resultats[0][0]",
            len(resultats),
        )
        return float(resultats[0][0])
