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
journalapp  = logging.getLogger(__name__)
log         = LogTool(origin="adapter")
NOM_FICHIER = os.path.basename(__file__)

# Limpiar handlers duplicados
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
# =============================================================================
# Mapping : champs Python -> noms colonnes du ColumnTransformer de m6_ocr.
#
# Ces noms correspondent aux colonnes vues par le preprocesseur au moment
# du fit() dans m6_ocr. Vérifiables via le log DIAGNOSTIC au démarrage.
#
# Convention de nommage du preprocesseur de m6_ocr :
#   - Colonnes catégorielles OHE : "ohe__<nom_colonne>_<valeur>"
#     ex. : "ohe__name_contract_type_cash_loans"
#   - Colonnes numériques StandardScaler : "ss__<nom_colonne>"
#     ex. : "ss__days_birth"
#
# MAPPING_COLONNES : champ Python -> préfixe de colonne preprocesseur
# (sans le préfixe ohe__/ss__ ni le suffixe _<valeur> pour les OHE)
# =============================================================================
MAPPING_COLONNES = {
    # Champ Python              : Nom colonne dans le preprocesseur (m6_ocr)
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
        self._session          : Optional[ort.InferenceSession] = None
        self._preprocesseur    : Optional[object]               = None
        self._modele_lgbm      : Optional[object]               = None
        self._explainer_shap   : Optional[shap.TreeExplainer]   = None
        self._background_shap  : Optional[np.ndarray]           = None
        self._nom_entree       : Optional[str]                  = None
        self._seuil            : float  = parametres.seuil_decision
        self._noms_features_enc: List[str] = []  # Noms bruts du preprocesseur
        self._nb_features      : int    = 0      # Renseigne au chargement
        # "probability" : sortie SHAP en espace probabilité [0,1]
        # "raw"         : sortie SHAP en log-odds → sigmoid appliqué
        self._mode_shap        : str    = "probability"

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
        nb_sortie        = self._session.get_inputs()[0].shape[1]
        log.DEBUG_PARAMETER_VALUE("self._nom_entree" , self._nom_entree)
        log.DEBUG_PARAMETER_VALUE("nb_sortie"        , nb_sortie)
        log.DEBUG_PARAMETER_VALUE("ONNX pret | available provider 0", providers[0])

        # 2. Log del proveedor de ejecución (Execution Provider)
        # Usamos el primer elemento de la lista de providers activos
        log.DEBUG_PARAMETER_VALUE("ONNX pret | session get provider 0", self._session.get_providers()[0])

        # 3. Inspección de cada entrada (input) esperada por el grafo ONNX
        for i in self._session.get_inputs():
            # log.DEBUG_PARAMETER_VALUE acepta 2 parámetros: etiqueta y valor
            log.DEBUG_PARAMETER_VALUE(f"input {i}", i.name)

        print("-" * 30)
        print(f"Total de parámetros detectados: {len(self._session.get_inputs())}")

        # Inspección profunda del shape
        for i in self._session.get_inputs():
            nombre = i.name
            forma = i.shape  # Esto devolverá algo como ['batch', 232]
            print(f"DEBUG: El input '{nombre}' espera un vector de tamaño: {forma[1]}")
            self._esperado_onnx = forma[1] # Guardamos ese 232

        # -- 2. Preprocesseur sklearn ----------------------------------------
        log.STEP(6, "2. Chargement preprocesseur", chemin_preproc)
        
        self._preprocesseur = joblib.load(chemin_preproc)

        # -- DIAGNOSTIC: Nombres de features encodees ------------------------
        try:
            # Extraer los nombres reales del preprocesador
            noms_encodees = self._preprocesseur.get_feature_names_out()
            self._noms_features_enc = noms_encodees  # Guardar para SHAP

            # Para ver los 232 nombres en tu log de depuración:
            for i, nombre in enumerate(noms_encodees):
                log.DEBUG_PARAMETER_VALUE(f"feat[{i:03}]", nombre)
            
            log.LEVEL_7_INFO(NOM_FICHIER, f"DIAGNOSTIC noms features encodees ({min(30, len(noms_encodees))} premiers)")
            
            # Iterar y loguear con el formato exacto que pediste
            for i, nombre in enumerate(noms_encodees[:30]):
                # f"feat[{i:03}]" asegura el formato 000, 001...
                log.DEBUG_PARAMETER_VALUE(f"feat[{i:03}]", nombre)
                
        except Exception as e:
            log.LEVEL_5_WARNING(NOM_FICHIER, f"Impossible d'extraire les noms de features: {e}")

        
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


        # --- En el método charger() ---

        # A. Cargar las columnas desde tu JSON
        with open(DOSSIER_ARTEFACT / "model_input_dims.json", 'r') as f:
            config_cols = json.load(f)
            self._nombres_json = config_cols["feature_names"]
        
        # B. Ver cuántas espera ONNX realmente
        self._n_esperado_onnx = self._session.get_inputs()[0].shape[1] 
        
        
        print(f"DEBUG: JSON tiene {len(self._nombres_json)} columnas.")
        print(f"DEBUG: ONNX espera {self._n_esperado_onnx} columnas.")
        
        # C. Identificar la diferencia
        if len(self._nombres_json) > self._n_esperado_onnx:
            # Si el JSON tiene 233 y ONNX 232, la "intrusa" es probablemente la última
            # Pero para estar seguros, calculamos cuántas sobran
            diferencia = len(self._nombres_json) - self._n_esperado_onnx
            self.sobrantes = self._nombres_json[-diferencia:] 
            
            log.LEVEL_5_WARNING(NOM_FICHIER, f"¡DESVÍO DETECTADO! Sobran estas columnas para ONNX: {self.sobrantes}")
            
            # Guardamos la lista 'recortada' para que SHAP y el resto no fallen
            self._noms_features_enc = self._nombres_json[:self._n_esperado_onnx]
        else:
            self._noms_features_enc = self._nombres_json

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

        # ✅ DIAGNÓSTICO TEMPORAL — eliminar tras fix
        cols_en_df  = set(df_entree.columns)
        cols_modelo = set(self._columnas_originales)
        inyectadas  = cols_en_df - cols_modelo   # claves del mapeo que NO existen → se descartan
        faltantes   = cols_modelo - cols_en_df   # columnas reales que se quedan en 0
        log.DEBUG_PARAMETER_VALUE("DIAG cols inyectadas (descartadas)", str(sorted(inyectadas)))
        log.DEBUG_PARAMETER_VALUE("DIAG cols faltantes  (quedan en 0)", str(sorted(faltantes)))
        log.DEBUG_PARAMETER_VALUE("DIAG total df_entree cols",  len(cols_en_df))
        log.DEBUG_PARAMETER_VALUE("DIAG total modelo    cols",  len(cols_modelo))
        # ✅ FIN DIAGNÓSTICO
        
        # ---- Etape 2: Transformation sklearn --------------------------------
        log.STEP(6, "2. Transformation sklearn")
        start_sk = time.perf_counter()
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

        elapsed      = time.perf_counter() - start_sk
        hours, rem   = divmod(elapsed, 3600)
        minutes, sec = divmod(rem, 60)
        exec_time    = f"{int(hours):02d}:{int(minutes):02d}:{sec:09.6f}"

        log.DEBUG_PARAMETER_VALUE("Durée Preproc", exec_time)

        # Conversión de sparse a dense si es necesario
        if hasattr(X_transforme, "toarray"):
            X_transforme = X_transforme.toarray()

        # --- AJUSTE DINÁMICO DE COLUMNAS ---
        # 2. RETIRADA DINÁMICA DE SOBRANTES
        # Obtenemos los nombres de las columnas que acaba de generar el preprocesador
        nombres_actuales = list(self._preprocesseur.get_feature_names_out())
    
        if hasattr(self, 'sobrantes') and self.sobrantes:
            indices_a_borrar = []
            for col in self.sobrantes:
                if col in nombres_actuales:
                    indices_a_borrar.append(nombres_actuales.index(col))
            
            if indices_a_borrar:
                # Borramos las columnas por su índice en el eje 1 (columnas)
                X_transforme = np.delete(X_transforme, indices_a_borrar, axis=1)
                log.LEVEL_7_INFO(NOM_FICHIER, f"Eliminadas {len(indices_a_borrar)} columnas sobrantes para ONNX")
    
        # 3. Ahora el shape será exactamente (1, 232)
        X_float32 = X_transforme.astype(np.float32)
        log.DEBUG_PARAMETER_VALUE("Shape final X", X_float32.shape)


        # --- En el método charger() ---
        print(f"DEBUG: JSON tiene {len(self._nombres_json)} columnas.")
        print(f"DEBUG: ONNX espera {self._n_esperado_onnx} columnas.")
        
        # C. Identificar la diferencia
        if len(self._nombres_json) > self._n_esperado_onnx:
            # Si el JSON tiene 233 y ONNX 232, la "intrusa" es probablemente la última
            # Pero para estar seguros, calculamos cuántas sobran
            diferencia = len(self._nombres_json) - self._n_esperado_onnx
            self.sobrantes = self._nombres_json[-diferencia:] 
            
            log.LEVEL_5_WARNING(NOM_FICHIER, f"¡DESVÍO DETECTADO! Sobran estas columnas para ONNX: {self.sobrantes}")
            
            # Guardamos la lista 'recortada' para que SHAP y el resto no fallen
            self._noms_features_enc = self._nombres_json[:self._n_esperado_onnx]
        else:
            self._noms_features_enc = self._nombres_json
        
        # DIAGNÓSTICO COLUMNA EXTRA — eliminar tras fix
        noms_out = list(self._preprocesseur.get_feature_names_out())


        # 1. Los nombres que genera tu preprocesador ACTUAL (los 233)
        noms_preproc_actual = list(self._preprocesseur.get_feature_names_out())
        
        # 2. Los nombres que tú tienes guardados en 'self._noms_features_enc' (los 232 originales)
        noms_originales = self._noms_features_enc
        
        # 3. Encontrar la diferencia exacta
        set_actual = set(noms_preproc_actual)
        set_original = set(noms_originales)
        
        self.sobrantes = list(set_actual - set_original)
        faltantes = list(set_original - set_actual)
        
        log.LEVEL_5_WARNING(NOM_FICHIER, f"COLUMNAS SOBRANTES: {self.sobrantes}")
        log.LEVEL_5_WARNING(NOM_FICHIER, f"COLUMNAS FALTANTES: {faltantes}")

        
        # Convertimos ambas listas a sets para poder comparar
        set_esperado = set(self._noms_features_enc)
        set_actual   = set(noms_out)
        
        # 1. ¿Qué tiene el preprocesador que NO espera el modelo? (El culpable del 233 vs 232)
        self.sobrantes = set_actual - set_esperado
        
        # 2. ¿Qué espera el modelo que el preprocesador NO ha generado?
        faltantes = set_esperado - set_actual
        
        # 3. Mostrar resultados
        if self.sobrantes:
            print(f"⚠️ COLUMNAS SOBRANTES EN EL PREPROCESADOR ({len(self.sobrantes)}): {self.sobrantes}")
        if faltantes:
            print(f"❌ COLUMNAS FALTANTES EN EL PREPROCESADOR ({len(faltantes)}): {faltantes}")
        if not self.sobrantes and not faltantes:
            print("✅ ¡Perfecto! Las columnas coinciden exactamente.")
        
        for i, nombre in enumerate(noms_out):
                log.DEBUG_PARAMETER_VALUE(f"feat_enc[{i:03}]", nombre)
        
        log.DEBUG_PARAMETER_VALUE("DIAG features tras transform", len(noms_out))
        
        # Buscar columnas sospechosas (infrequent, unknown, extra)
        sospechosas = [n for n in noms_out if "infrequent" in n or "remainder" in n]
        log.DEBUG_PARAMETER_VALUE("DIAG cols sospechosas", str(sospechosas))
        
        # Mostrar las últimas 5 (la extra suele estar al final)
        log.DEBUG_PARAMETER_VALUE("DIAG últimas 5 cols", str(noms_out[-5:]))


        
        # ---- Etape 3: Inference ONNX + latence ------------------------------
        log.STEP(6, "3. Inference ONNX + latence")
        log.DEBUG_PARAMETER_VALUE("self._nom_entree", self._nom_entree)
        debut_ms  = time.perf_counter()
        resultats = self._session.run(
            None,
            {self._nom_entree: X_float32},    #   _noms_features_enc
        )
        latence_ms = (time.perf_counter() - debut_ms) * 1000.0

        elapsed      = time.perf_counter() - debut_ms
        hours, rem   = divmod(elapsed, 3600)
        minutes, sec = divmod(rem, 60)
        exec_time    = f"{int(hours):02d}:{int(minutes):02d}:{sec:09.6f}"
        
        log.DEBUG_PARAMETER_VALUE("Durée Inference", exec_time)
        log.DEBUG_PARAMETER_VALUE("latence_ms", latence_ms)

        # ---- Etape 4: extraction de la probabilite de defaut ----------------
        log.STEP(6, "4. extraction de la probabilite de defaut")
        
        probabilite_defaut = self._extraire_probabilite(resultats)
        
        # ---- Etape 5: Calcul SHAP values ------------------------------------
        log.STEP(6, "5. Calcul SHAP values")
        
        # Activamos el modo "record" para interceptar los avisos en una lista
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always") # Forzamos a que se registren
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
            id_demande         = demande.id,
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

    def _construire_dataframe_V2(self, donnees):
        # 1. Extraer datos (Pydantic v2/v1)
        data_dict = donnees.model_dump() if hasattr(donnees, "model_dump") else donnees.dict()

        # 2. Crear DataFrame con las 11 columnas del formulario
        df = pd.DataFrame([data_dict])
        df = df.rename(columns=MAPPING_COLONNES)

        # 3. RELLENO SINTÉTICO (Para las 122 columnas faltantes)
        # Obtenemos la lista de todas las columnas que espera el preprocesador
        # Estas vienen de la propiedad .feature_names_in_ del objeto sklearn
        try:
            columnas_totales = self._preprocesseur.feature_names_in_

            # Añadimos las columnas que faltan con valor por defecto
            for col in columnas_totales:
                if col not in df.columns:
                    # Si es una columna de tipo objeto (categoría), usamos 'XNA' o 'Unaccompanied'
                    # Si es numérica, usamos 0.0
                    df[col] = 0.0  # El preprocesador suele manejar bien el 0.0 si hay un Imputer

            # 4. Asegurar el orden exacto de las 133+ columnas
            df_final = df[columnas_totales]

        except AttributeError:
            # Si el preprocesador no tiene feature_names_in_, usamos el orden del mapping

            log.LEVEL_5_WARNING(
                "ONNXAdapter",
                "Le préprocesseur n'a pas de métadonnées de colonnes, utilisation du mapping réduit."
            )
            journalapp.warning("Le préprocesseur n'a pas de métadonnées de colonnes, utilisation du mapping réduit.")
            df_final = df[list(MAPPING_COLONNES.values())]

        return df_final

    def _construire_dataframe_V1(self, demande: DemandeCredit) -> pd.DataFrame:
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

    def _construire_dataframe_V3(self, demande: DemandeCredit) -> pd.DataFrame:
        """
        Método robusto: Inicializa todas las columnas esperadas y mapea
        los datos del formulario a los nombres técnicos del modelo.
        """
        # 1. Inicialización con ceros (o valores neutros)
        # self._columnas_originales debe contener las 133+ columnas del modelo
        datos = {col: 0.0 for col in self._columnas_originales}

        # 2. Mapeo explícito y transformación de unidades (Edad a días negativos)
        # Usamos los nombres técnicos que el preprocessor espera
        mapeo_real = {
            "days_birth": float(-(demande.age * 365)),
            "amt_income_total": float(demande.revenu),
            "amt_credit": float(demande.montant_pret),
            "duree_pret_mois": float(demande.duree_pret_mois),
            "avg_dpd_per_delinquency": float(demande.jours_retard_moyen),
            "delinquency_ratio": float(demande.taux_incidents),
            "credit_utilization_ratio": float(demande.taux_utilisation_credit),
            "num_open_accounts": float(demande.nb_comptes_ouverts),
            # Manejo seguro de Enums o Strings
            "name_housing_type": getattr(demande.type_residence, "value", demande.type_residence),
            "name_type_suite": getattr(demande.objet_pret, "value", demande.objet_pret),
            "name_contract_type": getattr(demande.type_pret, "value", demande.type_pret),
        }

        # 3. Inyectar los datos reales en la estructura completa
        datos.update(mapeo_real)

        # 4. Crear DataFrame garantizando el orden exacto de las columnas
        # El orden es vital para que el preprocesador no confunda variables
        return pd.DataFrame([datos])[self._columnas_originales]

    def _construire_dataframe_V4(self, demande: DemandeCredit) -> pd.DataFrame:
        # 1. Inicializar todas las columnas que el modelo espera en 0.0
        # self._columnas_originales debe venir de self._preprocesseur.feature_names_in_
        datos = {col: 0.0 for col in self._columnas_originales}

        # 2. Mapeo Directo (Numéricas con sus transformaciones prefijadas)
        mapeo = {
            "standard__ext_source_1": demande.ext_source_1,
            "standard__ext_source_2": demande.ext_source_2,
            "standard__ext_source_3": demande.ext_source_3,
            "standard__install_payment_ratio_mean": demande.paymnt_ratio_mean,
            "standard__days_birth": float(-(demande.age * 365)),
            "log__cc_amt_drawings_current_mean": demande.cc_drawings_mean,
            "robust__install_payment_delay_mean": demande.paymnt_delay_mean,
            "standard__pos_months_balance_mean": demande.pos_months_mean,
            "log__amt_goods_price": demande.goods_price,
            "log__bureau_amt_credit_sum_total": demande.bureau_credit_total,
            "robust__install_dpd_max": demande.max_dpd,
            "log__amt_credit": demande.amt_credit,
            "log__amt_annuity": demande.amt_annuity,
            "log__cc_amt_balance_mean": demande.cc_balance_mean,
            "standard__days_employed": float(-(demande.years_employed * 365)),
            "standard__days_last_phone_change": demande.phone_change_days,
            "standard__region_rating_client": float(demande.region_rating),
            "log__bureau_amt_credit_sum_debt_mean": demande.bureau_debt_mean,
        }
        datos.update(mapeo)

        # 3. Lógica One-Hot Encoding (OHE) manual
        if demande.education_type == "Higher education":
            datos["ohe__name_education_type_higher_education"] = 1.0

        if demande.code_gender == "F":
            datos["ohe__code_gender_f"] = 1.0

        # 4. Crear DF respetando el orden exacto de entrenamiento
        return pd.DataFrame([datos])[self._columnas_originales]

    def _construire_dataframe(self, demande: DemandeCredit) -> pd.DataFrame:
        # 1. Todas las columnas a 0.0 (nombres originales, sin prefijo)
        datos = {col: 0.0 for col in self._columnas_originales}
    
        # 2. Mapeo con nombres ORIGINALES (sin standard__, log__, robust__)
        mapeo = {
            "ext_source_1":                    demande.ext_source_1,
            "ext_source_2":                    demande.ext_source_2,
            "ext_source_3":                    demande.ext_source_3,
            "install_payment_ratio_mean":      demande.paymnt_ratio_mean,
            "days_birth":                      float(-(demande.age * 365)),
            "cc_amt_drawings_current_mean":    demande.cc_drawings_mean,
            "install_payment_delay_mean":      demande.paymnt_delay_mean,
            "pos_months_balance_mean":         demande.pos_months_mean,
            "amt_goods_price":                 demande.goods_price,
            "bureau_amt_credit_sum_total":     demande.bureau_credit_total,
            "install_dpd_max":                 demande.max_dpd,
            "amt_credit":                      demande.amt_credit,
            "amt_annuity":                     demande.amt_annuity,
            "cc_amt_balance_mean":             demande.cc_balance_mean,
            "days_employed":                   float(-(demande.years_employed * 365)),
            "days_last_phone_change":          demande.phone_change_days,
            "region_rating_client":            float(demande.region_rating),
            "bureau_amt_credit_sum_debt_mean": demande.bureau_debt_mean,
        }
        datos.update(mapeo)
    
        # 3. OHE categóricas — verificar los nombres exactos en feature_names_in_
        edu = getattr(demande.education_type, "value", demande.education_type)
        datos["name_education_type"] = edu  # columna original string, no OHE manual
    
        gender = getattr(demande.code_gender, "value", demande.code_gender)
        datos["code_gender"] = gender
    
        return pd.DataFrame([datos])[self._columnas_originales]
    
    # -------------------------------------------------------------------------
    def _calculer_shap(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule les SHAP values pour la classe 1 (defaut).

        Gère les deux formats de retour selon la version SHAP installée :
          - Nouvelle API (>= 0.40) : liste [ndarray_cl0, ndarray_cl1]
          - Ancienne API : ndarray 2D shape (1, N)

        Mode "raw" (tree_path_dependent) :
            Applique sigmoid pour convertir les log-odds en [0, 1].

        Args:
            X : np.ndarray shape (1, N) — sortie du preprocesseur.

        Returns:
            SHAP values shape (N,) pour la classe 1.
        """
        try:
            sv = self._explainer_shap.shap_values(X)

            # -- Extraction de la classe 1 -----------------------------------
            if isinstance(sv, list):
                # Nouvelle API : [ndarray_classe0, ndarray_classe1]
                # sv[1] peut être shape (1, N) ou (N,) selon la version
                arr = np.array(sv[1])
                sv_classe1 = arr[0] if arr.ndim == 2 else arr.flatten()

            elif isinstance(sv, np.ndarray) and sv.ndim == 2:
                # Ancienne API : shape (1, N) directement
                sv_classe1 = sv[0]

            else:
                sv_classe1 = np.array(sv).flatten()

            # -- Conversion log-odds → [0, 1] pour mode raw -----------------
            if self._mode_shap == "raw":
                sv_classe1 = 1.0 / (1.0 + np.exp(-sv_classe1))

            return sv_classe1

        except Exception as erreur:
            journalapp.warning(
                "Echec calcul SHAP (non bloquant) : %s", erreur
            )
            nb_features = X.shape[1] if hasattr(X, "shape") and X.ndim == 2 else len(X)
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
            "ext_source_1": demande.ext_source_1,
            "ext_source_2": demande.ext_source_2,
            "ext_source_3": demande.ext_source_3,
            "paymnt_ratio_mean": demande.paymnt_ratio_mean,
            "age": demande.age,
            "cc_drawings_mean": demande.cc_drawings_mean,
            "paymnt_delay_mean": demande.paymnt_delay_mean,
            "pos_months_mean": demande.pos_months_mean,
            "goods_price": demande.goods_price,
            "education_type": demande.education_type,
            "code_gender": demande.code_gender,
            "bureau_credit_total": demande.bureau_credit_total,
            "max_dpd": demande.max_dpd,
            "amt_credit": demande.amt_credit,
            "amt_annuity": demande.amt_annuity,
            "cc_balance_mean": demande.cc_balance_mean,
            "years_employed": demande.years_employed,
            "phone_change_days": demande.phone_change_days,
            "region_rating": demande.region_rating,
            "bureau_debt_mean": demande.bureau_debt_mean
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
        Retrouve la feature originale depuis le nom brut d'une colonne encodee.

        Convention de nommage du préprocesseur m6_ocr :
          - Numériques (StandardScaler) : "ss__<nom_colonne>"
            ex. : "ss__days_birth" -> "age"
          - Catégorielles (OHE)         : "ohe__<nom_colonne>_<valeur>"
            ex. : "ohe__name_contract_type_cash_loans" -> "type_pret"

        Stratégie :
            Pour chaque nom brut, on extrait la partie après le préfixe
            (ss__ ou ohe__), puis on cherche dans MAPPING_COLONNES la valeur
            (nom colonne preprocesseur) qui est un préfixe de cette partie.

        Args:
            nom_encode : Nom brut de la colonne (ex. "ohe__name_housing_type_rented").

        Returns:
            Champ Python correspondant (ex. "type_residence"), ou "inconnu".
        """
        # -- Supprimer le préfixe du step sklearn (ohe__, ss__, num__, cat__...) -
        # On coupe au premier __ pour obtenir le nom de colonne brut
        if "__" in nom_encode:
            partie = nom_encode.split("__", 1)[1]   # "name_housing_type_rented"
        else:
            partie = nom_encode

        # -- Match direct (feature numérique scalée : partie == nom_colonne) --
        if partie in MAPPING_INVERSE:
            return MAPPING_INVERSE[partie]

        # -- Match par préfixe (feature OHE : partie commence par nom_colonne) -
        # "name_housing_type_rented" commence par "name_housing_type"
        for nom_col_preproc, nom_orig in MAPPING_INVERSE.items():
            if partie.startswith(nom_col_preproc + "_") or partie == nom_col_preproc:
                return nom_orig

        return "inconnu"

    # -------------------------------------------------------------------------
    def _extraire_noms_features_encodees(self) -> List[str]:
        """
        Recupere les noms des features apres transformation sklearn.

        Conserve les noms BRUTS tels que produits par get_feature_names_out()
        (ex. : "ohe__name_contract_type_cash_loans", "ss__days_birth").
        Le nettoyage des prefixes est délégué à _trouver_feature_originale()
        qui connaît la convention de nommage du preprocesador de m6_ocr.

        Returns:
            Liste des noms bruts de colonnes dans l'espace transforme.
        """
        try:
            if hasattr(self._preprocesseur, "get_feature_names_out"):
                return list(self._preprocesseur.get_feature_names_out())
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
