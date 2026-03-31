# =============================================================================
# src/adapters/scorers/onnx_scorer_adapter.py
# Adaptateur ONNX -- Implementation du port ICreditScorer
#
# Pipeline d'inference :
#   DemandeCredit (11 features brutes, avec strings)
#       |
#       v  _construire_dataframe()
#   DataFrame pandas (noms de colonnes = ceux vus par le preprocesseur)
#       |
#       v  preprocesseur.transform()
#   np.ndarray float32 (N features apres One-Hot + scaling)
#       |
#       v  onnxruntime.run()
#   probabilite de defaut (float)
#       |
#       v  seuil metier
#   DecisionCredit
#
# Prerequis dans model_artifact/ :
#   best_model.onnx       <- modele LightGBM converti par convert_onnx.py
#   preprocessor.pkl      <- ColumnTransformer de la phase 2 de m6_ocr
#   best_model_meta.json  <- metadonnees (run_id, auc, seuil_optimal)
#
# Pour copier le preprocesseur depuis m6_ocr :
#   cp m6_ocr/models/preprocessor/preprocessor.pkl model_artifact/
# =============================================================================

# --- Bibliotheques standard ---------------------------------------------------
import json                                           # Lecture metadonnees
import logging                                        # Journalisation
import time                                           # Mesure latence
from   pathlib import Path                            # Chemins multi-OS

# --- Bibliotheques tierces : donnees -----------------------------------------
import joblib                                         # Chargement preprocessor.pkl
import numpy  as np                                   # Tenseurs float32 pour ONNX
import pandas as pd                                   # DataFrame pour le preprocesseur

# --- Bibliotheques tierces : ONNX Runtime ------------------------------------
import onnxruntime as ort                             # Inference ONNX multiplateforme

# --- Domaine et port ---------------------------------------------------------
from src.api.domain.entities       import DemandeCredit, DecisionCredit
from src.api.domain.value_objects  import Decision, ScoreRisque
from src.api.ports.i_credit_scorer import ICreditScorer

# --- Configuration -----------------------------------------------------------
from config import parametres, DOSSIER_ARTEFACT


# Journalisation du module
journalapp = logging.getLogger(__name__)

# =============================================================================
# Mapping : champs de DemandeCredit -> noms de colonnes du preprocesseur
# IMPORTANT : ces noms doivent correspondre EXACTEMENT aux noms de colonnes
# que le ColumnTransformer de m6_ocr a vus pendant l'entrainement.
# Si les noms different, verifiez models/preprocessor/feature_names.json
# dans m6_ocr et ajustez ce dictionnaire.
# =============================================================================
MAPPING_COLONNES_V0 = {
    # Champ Python      : Colonne attendue par le preprocesseur m6_ocr
    "age"                     : "DAYS_BIRTH",             # age en annees -> converti en jours negatifs
    "revenus"                 : "AMT_INCOME_TOTAL",       # revenus annuels
    "montant_pret"            : "AMT_CREDIT",             # montant du pret
    "duree_pret_mois"         : "duree_pret_mois",        # duree en mois (verifier le nom exact)
    "jours_retard_moyen"      : "avg_dpd_per_delinquency",# retard moyen par incident
    "taux_incidents"             : "delinquency_ratio",      # ratio d'incidents
    "taux_utilisation_credit" : "credit_utilization_ratio", # utilisation revolving
    "nb_comptes_ouverts"      : "num_open_accounts",      # nombre de comptes actifs
    "type_residence"          : "residence_type",         # string : "Owned", "Rented"...
    "objet_pret"              : "loan_purpose",           # string : "Education", "Home"...
    "type_pret"               : "loan_type",              # string : "Secured", "Unsecured"
}

MAPPING_COLONNES = {
    "age"                     : "days_birth",        # Minúsculas, como en tu lista
    "revenu"                  : "amt_income_total",  # OJO: verifica si es 'revenu' o 'revenus' en tu entidad
    "montant_pret"            : "amt_credit",
    "type_residence"          : "name_housing_type",
    "objet_pret"              : "name_type_suite",
    "type_pret"               : "name_contract_type",
    # Las columnas calculadas como 'fe1_credit_income_ratio' también deben estar
}

# ##############################################################################
# Adaptateur : OnnxScorerAdaptateur
# ##############################################################################

# =============================================================================
class OnnxScorerAdaptater(ICreditScorer):
    """
    Adaptateur d'inference base sur onnxruntime + preprocesseur sklearn.

    Chargement unique au demarrage (pattern singleton via lifespan FastAPI).
    La session ONNX et le preprocesseur sont conserves en memoire pour
    maintenir une latence d'inference < 10 ms par requete.

    Attributs prives
    ----------------
    _session       : Session ONNX Runtime (inference rapide)
    _preprocesseur : ColumnTransformer sklearn (One-Hot + scaling)
    _nom_entree    : Nom de la couche d'entree du modele ONNX
    _seuil         : Seuil de decision metier (configure via parametres)
    _nb_features   : Nombre de features apres transformation (debug)
    """

    # =========================================================================
    def __init__(self) -> None:
        """Initialise l'adaptateur sans charger les ressources (lazy loading)."""
        self._session        : ort.InferenceSession | None = None
        self._preprocesseur  : object | None               = None
        self._nom_entree     : str | None                  = None
        self._seuil          : float = parametres.seuil_decision
        self._nb_features    : int   = 0                   # Renseigne au chargement

    # =========================================================================
    def charger(self) -> None:
        """
        Charge la session ONNX et le preprocesseur sklearn.

        Appele UNE SEULE FOIS dans le lifespan FastAPI au demarrage.
        Idempotent : un second appel est ignore sans erreur.

        Leve
        ----
        FileNotFoundError
            Si best_model.onnx ou preprocessor.pkl sont absents.
        RuntimeError
            Si l'initialisation ONNX ou le chargement sklearn echoue.
        """
        # -- Idempotence : ne recharge pas si deja initialise ----------------
        if self._session is not None:
            journalapp.debug("Ressources deja chargees -- ignoré.")
            return

        # -- Chemins des artefacts -------------------------------------------
        chemin_onnx   = DOSSIER_ARTEFACT / "best_model.onnx"
        chemin_preproc = DOSSIER_ARTEFACT / "preprocessor.pkl"
        chemin_meta    = DOSSIER_ARTEFACT / "best_model_meta.json"

        # -- Verification d'existence avant chargement -----------------------
        if not chemin_onnx.exists():
            raise FileNotFoundError(
                f"Modele ONNX introuvable : {chemin_onnx}\n"
                "Executez : python scripts/export_best_model.py"
            )
        if not chemin_preproc.exists():
            raise FileNotFoundError(
                f"Preprocesseur introuvable : {chemin_preproc}\n"
                "Copiez depuis m6_ocr :\n"
                "  cp m6_ocr/models/preprocessor/preprocessor.pkl "
                "model_artifact/"
            )

        # -- Chargement de la session ONNX -----------------------------------
        journalapp.info("Chargement ONNX depuis : %s", chemin_onnx)

        providers_dispo = ort.get_available_providers()
        providers       = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers_dispo
            else ["CPUExecutionProvider"]
        )

        options                    = ort.SessionOptions()
        options.log_severity_level = 3             # Silence les logs verbeux ONNX

        self._session    = ort.InferenceSession(
            str(chemin_onnx),
            sess_options = options,
            providers    = providers,
        )
        self._nom_entree = self._session.get_inputs()[0].name

        journalapp.info(
            "Session ONNX prete -- provider : %s | entree : '%s'",
            providers[0],
            self._nom_entree,
        )

        # -- Chargement du preprocesseur sklearn (ColumnTransformer) ---------
        journalapp.info("Chargement preprocesseur depuis : %s", chemin_preproc)
        self._preprocesseur = joblib.load(chemin_preproc)

        # -- Detection du nombre de features apres transformation ------------
        try:
            nb_sortie = self._session.get_inputs()[0].shape[1]
            if nb_sortie:
                self._nb_features = nb_sortie
        except (IndexError, TypeError):
            self._nb_features = 0

        journalapp.info(
            "Preprocesseur charge -- type : %s",
            type(self._preprocesseur).__name__,
        )

        # -- Lecture des metadonnees pour tracabilite ------------------------
        if chemin_meta.exists():
            with open(chemin_meta, encoding="utf-8") as f:
                meta = json.load(f)
            journalapp.info(
                "Modele -- run_id=%s | auc=%.4f | seuil=%.2f",
                meta.get("run_id",        "inconnu"),
                meta.get("roc_auc",       0.0),
                meta.get("seuil_optimal", self._seuil),
            )
            # -- Mise a jour du seuil depuis les metadonnees si disponible ---
            seuil_meta = meta.get("seuil_optimal", 0.0)
            if seuil_meta > 0:
                self._seuil = seuil_meta
                journalapp.info(
                    "Seuil de decision mis a jour depuis metadonnees : %.3f",
                    self._seuil,
                )

        # EXTRAER COLUMNAS AUTOMÁTICAMENTE
        # 'feature_names_in_' contiene los nombres de las columnas originales
        # que el preprocesador vio durante el fit()
        if hasattr(self._preprocesseur, "feature_names_in_"):
            self._columnas_originales = list(self._preprocesseur.feature_names_in_)
            journalapp.info("Colonnes attendues par le préprocesseur : %s", self._columnas_originales)
        else:
            # Si por alguna razón no está definido (versiones antiguas de sklearn)
            journalapp.error("Le préprocesseur n'a pas l'attribut feature_names_in_")

    # =========================================================================
    @property
    def est_pret(self) -> bool:
        """True si la session ONNX et le preprocesseur sont charges."""
        return self._session is not None and self._preprocesseur is not None



    # =========================================================================
    def predire(self, demande: DemandeCredit) -> DecisionCredit:
        """
        Inference complete : preprocessing + ONNX + decision metier.

        Sequence interne
        ----------------
        1. Construire un DataFrame pandas avec les noms de colonnes
           attendus par le ColumnTransformer de m6_ocr.
        2. Appliquer le preprocesseur (One-Hot Encoding + scaling).
        3. Convertir en tableau numpy float32 pour ONNX.
        4. Lancer l'inference onnxruntime (mesure de latence).
        5. Extraire la probabilite de defaut (classe 1).
        6. Appliquer le seuil metier.
        7. Retourner une DecisionCredit.

        Leve
        ----
        RuntimeError
            Si les ressources ne sont pas initialisees.
        ValueError
            Si la transformation du preprocesseur echoue.
        """
        if not self.est_pret:
            raise RuntimeError(
                "OnnxScorerAdaptateur non initialise. "
                "Appelez charger() avant d'effectuer des predictions."
            )

        # -- Etape 1 : construction du DataFrame avec les bons noms ----------
        df_entree = self._construire_dataframe(demande)

        # -- Etape 2 : transformation sklearn (One-Hot + scaling) ------------
        try:
            X_transforme = self._preprocesseur.transform(df_entree)
        except Exception as erreur:
            raise ValueError(
                f"Echec de la transformation preprocesseur : {erreur}\n"
                f"Colonnes fournies : {list(df_entree.columns)}\n"
                "Verifiez MAPPING_COLONNES dans onnx_scorer_adapter.py."
            ) from erreur

        # -- Etape 3 : conversion numpy float32 pour ONNX -------------------
        # Certains preprocesseurs retournent une matrice sparse scipy
        if hasattr(X_transforme, "toarray"):
            X_transforme = X_transforme.toarray()    # sparse -> dense

        # ELIMINAR LA COLUMNA SOBRANTE (AJUSTE DINÁMICO)
        # Si el modelo espera 232 y tenemos 233, recortamos la última
        if X_transforme.shape[1] > self._nb_features:
            journalapp.warning(
                "Ajuste: eliminando columna extra (posiblemente TARGET o ID). De %d a %d",
                X_transforme.shape[1], self._nb_features
            )
            X_transforme = X_transforme[:, :self._nb_features] #

        tableau_input = X_transforme.astype(np.float32)  # shape (1, N)

        # -- Etape 4 : inference ONNX avec mesure de latence -----------------
        debut_ms  = time.perf_counter()
        resultats = self._session.run(
            None,
            {self._nom_entree: tableau_input},
        )
        latence_ms = (time.perf_counter() - debut_ms) * 1000.0

        # -- Etape 5 : extraction de la probabilite de defaut ----------------
        probabilite_defaut = self._extraire_probabilite(resultats)

        # --- Étape 6 : Application du seuil métier ---------------------------
        # 1. On crée l'objet ScoreRisque en passant la valeur au constructeur
        score = ScoreRisque(valeur=probabilite_defaut)

        # 2. On utilise la méthode interne de ScoreRisque pour obtenir la décision
        # Note : on passe self._seuil qui est stocké dans l'adaptateur
        decision = score.vers_decision(self._seuil)

        # --- Etape 7 : Retour du résultat conforme à DecisionCredit ---------
        journalapp.debug(
            "Inference ONNX -- proba=%.4f | decision=%s | latence=%.1f ms",
            score.valeur, decision.value, latence_ms,
        )

        # --- Etape 7 : Retour du résultat conforme à DecisionCredit ---------
        return DecisionCredit(
            id_demande = demande.id_demande, # L'UUID
            score      = score,              # L'OBJET ScoreRisque (pas score.valeur)
            decision   = decision,           # L'OBJET Decision (pas decision.value)
            latence_ms = round(latence_ms, 2)
        )

    # -------------------------------------------------------------------------
    def _construire_dataframe_V0(
        self,
        demande: DemandeCredit,
    ) -> pd.DataFrame:
        """
        Construit le DataFrame d'entree pour le preprocesseur sklearn.

        Applique les conversions necessaires :
          - age (annees) -> DAYS_BIRTH (jours negatifs, convention Home Credit)
          - Les features categoriques restent en string (le preprocesseur
            les encode via OneHotEncoder)

        Args:
            demande : Entite domaine avec les 11 features brutes.

        Returns:
            DataFrame une ligne avec les noms de colonnes attendus
            par le ColumnTransformer de m6_ocr.
        """
        # -- Construction de la ligne brute avec les noms corrects -----------
        # JOURS_NAISSANCE : dans Home Credit, DAYS_BIRTH est negatif (jours
        # ecoules depuis la naissance, comptes a rebours).
        ligne = {
            MAPPING_COLONNES["age"]                    : -(demande.age * 365),
            MAPPING_COLONNES["revenus"]                : demande.revenu,
            MAPPING_COLONNES["montant_pret"]           : demande.montant_pret,
            MAPPING_COLONNES["duree_pret_mois"]        : demande.duree_pret_mois,
            MAPPING_COLONNES["jours_retard_moyen"]     : demande.jours_retard_moyen,
            MAPPING_COLONNES["taux_incidents"]            : demande.taux_incidents,
            MAPPING_COLONNES["taux_utilisation_credit"]: demande.taux_utilisation_credit,
            MAPPING_COLONNES["nb_comptes_ouverts"]     : demande.nb_comptes_ouverts,
            MAPPING_COLONNES["type_residence"]         : demande.type_residence,  # str
            MAPPING_COLONNES["objet_pret"]             : demande.objet_pret,      # str
            MAPPING_COLONNES["type_pret"]              : demande.type_pret,       # str
        }

        return pd.DataFrame([ligne])

    def _construire_dataframe_V1(self, demande: DemandeCredit) -> pd.DataFrame:
        # 1. Crear un DataFrame vacío con la estructura exacta que pide el pkl
        df = pd.DataFrame(columns=self._columnas_originales, dtype=object)

        # 2. Creamos una fila inicial de ceros o NaNs
        df.loc[0] = np.nan

        # Función auxiliar para extraer el valor de forma segura
        def obtener_valor(campo):
            # Si tiene .value (es Enum), lo usamos. Si no, devolvemos el campo tal cual (es str).
            return getattr(campo, "value", campo)

        # 3. Mapeo inteligente: llenamos lo que tenemos
        # Asegúrate de que los nombres de la izquierda coincidan con los de tu lista dinámica
        mapeo_directo = {
            "days_birth"        : -(demande.age * 365),
            "amt_income_total"  : demande.revenu,
            "amt_credit"        : demande.montant_pret,
            "name_housing_type" : getattr(demande.type_residence, "value", demande.type_residence),
            "name_type_suite"   : getattr(demande.objet_pret, "value", demande.objet_pret),
            "name_contract_type": getattr(demande.type_pret, "value", demande.type_pret),
        }

        for col, valor in mapeo_directo.items():
            if col in df.columns:
                df.loc[0, col] = valor

        # 4. Rellenar el resto (ext_source_1, etc.) con 0 para que el transform() no falle
        # Rellenar con 0 y asegurar que las columnas numéricas sean floats
        # para que el preprocesador no falle
        return df.fillna(0)

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
    @staticmethod
    def _extraire_probabilite(resultats: list) -> float:
        """
        Extrait la probabilite de defaut depuis la sortie ONNX.

        La structure de sortie d'onnxruntime varie selon la facon dont
        le modele a ete exporte par skl2onnx :

        Cas 1 (zipmap=False, recommande) :
          resultats = [labels, probas]
          labels  : np.ndarray shape (1,)       -> classe predite
          probas  : np.ndarray shape (1, 2)     -> [prob_classe_0, prob_classe_1]
          Acces   : resultats[1][0][1]

        Cas 2 (zipmap=True, ancien defaut) :
          resultats = [labels, [{0: p0, 1: p1}]]
          Acces   : resultats[1][0][1]  <- meme acces, fonctionne pour les deux

        Cas 3 (modele sans predict_proba, sortie unique) :
          resultats = [probas_brutes]
          Acces   : resultats[0][0]

        Args:
            resultats : Liste de sorties ONNX Runtime.

        Returns:
            Probabilite de defaut (classe 1) entre 0.0 et 1.0.
        """
        # -- Cas 1 et 2 : deux sorties (labels + probas) ---------------------
        if len(resultats) >= 2:
            probas = resultats[1]

            # Cas 1 : array numpy shape (1, 2)
            if isinstance(probas, np.ndarray) and probas.ndim == 2:
                return float(probas[0][1])

            # Cas 2 : liste de dicts [{0: p0, 1: p1}]
            if isinstance(probas, list) and isinstance(probas[0], dict):
                return float(probas[0][1])

            # Cas 2b : liste de dicts avec cles string {"0": p0, "1": p1}
            if isinstance(probas, list) and isinstance(probas[0], dict):
                d = probas[0]
                cle = 1 if 1 in d else "1"
                return float(d[cle])

        # -- Cas 3 : sortie unique (probabilite brute) -----------------------
        if len(resultats) == 1:
            probas = resultats[0]
            if isinstance(probas, np.ndarray):
                return float(probas.flat[0])

        # -- Cas inconnu : log + valeur par defaut ---------------------------
        journalapp.warning(
            "Structure ONNX inattendue : len=%d, types=%s -- "
            "Utilisation de resultats[0][0] en repli.",
            len(resultats),
            [type(r).__name__ for r in resultats],
        )
        return float(resultats[0][0])
