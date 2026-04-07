"""
src/features/registry.py
=========================
FeatureConfigurator — Pont entre le FeatureRegistry (schema.py)
et le pipeline sklearn (ColumnTransformer).

Responsabilités :
    1. Charger/filtrer les listes de colonnes selon le DataFrame réel
    2. Valider que le DataFrame est cohérent avec le registre
    3. Construire le ColumnTransformer à partir des listes dérivées
    4. Apprendre les paramètres (médiane, mode, bornes) sur le train → anti-leakage
    5. Sauvegarder / charger les paramètres appris (artefacts)
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS - LIBRAIRIES STANDARD PYTHON
# ==============================================================================
import json                                 # Sérialisation des artefacts
import logging                              # Logger du module
import warnings                             # Avertissements contrôlés
from   pathlib import Path                  # Chemins de fichiers
from   typing  import Dict, List, Optional, Tuple  # Typage
import warnings
# ==============================================================================
# IMPORTS - LIBRAIRIES TIERS (DATA)
# ==============================================================================
import numpy as np                          # Calculs numériques
import pandas as pd                         # DataFrames

# ==============================================================================
# IMPORTS - SCIKIT-LEARN (PREPROCESSING)
# ==============================================================================
from   sklearn.compose      import ColumnTransformer  # Pipeline colonnes
from   sklearn.impute       import SimpleImputer      # Imputation
from   sklearn.pipeline     import Pipeline           # Pipeline sklearn
from   sklearn.preprocessing import (                 # Encodage + scaling
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

# ==============================================================================
# IMPORTS - MODULES INTERNES (REGISTRE MÉTIER ↔ TECHNIQUE)
# ==============================================================================
from   src.data.schema import (                  # Registre et enums du projet
    ColumnRole,
    ColumnType,
    EncodingType,
    FeatureRegistry,
    REGISTRY,
    TransformType,
)

logger = logging.getLogger(__name__)


# ##############################################################################
# FEATURE CONFIGURATOR
# ##############################################################################


# ==============================================================================
# FONCTIONS UTILITAIRES MODULE-LEVEL
# (doivent être au niveau module pour être picklables)
# ==============================================================================

def _coerce_numeric_columns(X):
    """
    Convertit les colonnes d'un array numpy en float.
    Gère les valeurs 'y'/'n'/'Y'/'N' retournées par PostgreSQL
    comme TEXT même après CASE WHEN ... THEN 1 ELSE 0.
    Définie au niveau MODULE (pas dans une méthode) pour être picklable.
    """
    df_tmp = pd.DataFrame(X)
    for c in df_tmp.columns:
        df_tmp[c] = pd.to_numeric(df_tmp[c], errors="coerce")
    return df_tmp.values.astype(float)

# ##############################################################################
# UTILITAIRES DE TRANSFORMATION (GLOBAL POUR PICKLE)
# ##############################################################################

def safe_log1p(X):
    """
    Calcule log(1+x) en protégeant contre les valeurs négatives.
    Défini au niveau global pour permettre la sérialisation via pickle.
    """
    # Aseguramos que X sea un array de numpy con tipo flotante
    X_arr = np.asanyarray(X).astype(float)
    return np.log1p(np.maximum(X_arr, 0))

class FeatureConfigurator:
    """
    Construit et gère le pipeline de preprocessing Phase 2.

    Usage :
        config = FeatureConfigurator(registry=REGISTRY)
        config.fit(df_train)                          # Apprend les paramètres
        X_train = config.transform(df_train)
        X_test  = config.transform(df_test)
        config.save_artifacts("models/preprocessor/")
    """

    def __init__(
        self,
        registry: FeatureRegistry = REGISTRY,
        verbose: bool = True
    ):
        self.registry = registry
        self.verbose  = verbose
        self._fitted  = False

        # Listes dérivées (remplies après fit)
        self.cols_ohe_active: List[str]      = []
        self.cols_ordinal_active: List[str]  = []
        self.cols_target_active: List[str]   = []
        self.cols_log_active: List[str]      = []
        self.cols_standard_active: List[str] = []
        self.cols_robust_active: List[str]   = []
        self.cols_binary_active: List[str]   = []

        # Paramètres appris (anti-leakage)
        self.learned_medians: Dict[str, float] = {}
        self.learned_modes: Dict[str, str]     = {}
        self.learned_winsor: Dict[str, Tuple]  = {}
        self.learned_log_dec: Dict[str, float] = {}

        # Pipeline sklearn
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_out: List[str]              = []
        self.columns_seen_at_fit: List[str]            = []

    # ─────────────────────────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # =========================================================================
    # FIT — apprend tous les paramètres sur le train uniquement
    # =========================================================================

    def fit(self, df_train: pd.DataFrame) -> "FeatureConfigurator":
        """
        Apprend les paramètres depuis df_train.
        Met à jour le registre avec les valeurs apprises.

        Args:
            df_train: DataFrame train (inclut TARGET)

        Returns:
            self
        """
        self._log("\n============================================================================")
        self._log("FEATURE CONFIGURATOR — FIT")
        self._log("============================================================================")

        # 1. Validation du DataFrame contre le registre
        validation = self.registry.validate_dataframe_robust(list(df_train.columns))
        if validation["unknown_in_df"]:
            self._log(f"\n  ⚠️  Colonnes inconnues dans le DF ({len(validation['unknown_in_df'])}) :")
            self._log(f"     {validation['unknown_in_df'][:10]}")
        if validation["missing_from_df"]:
            self._log(f"  ℹ️  Features du registre absentes du DF ({len(validation['missing_from_df'])}) :")
            self._log(f"     {validation['missing_from_df'][:10]}")

        # Colonnes présentes (pour filtrer les listes)
        df_cols = list(df_train.columns)
        self.columns_seen_at_fit = df_cols

        # 2. Dérivation des listes actives (registre filtré par colonnes présentes)
        self.cols_ohe_active      = self.registry.get_columns(encoding=EncodingType.ONE_HOT,    present_in=df_cols)
        self.cols_ordinal_active  = self.registry.get_columns(encoding=EncodingType.ORDINAL,    present_in=df_cols)
        self.cols_target_active   = self.registry.get_columns(encoding=EncodingType.TARGET_ENC, present_in=df_cols)
        self.cols_log_active      = self.registry.get_columns(transform=TransformType.LOG,      present_in=df_cols)
        self.cols_standard_active = self.registry.get_columns(transform=TransformType.STANDARD, present_in=df_cols)
        self.cols_robust_active   = self.registry.get_columns(transform=TransformType.ROBUST,   present_in=df_cols)
        self.cols_binary_active   = self.registry.get_columns(col_type=ColumnType.BINARY,       present_in=df_cols)

        # Exclure target et identifiants des features
        _exclude = set(self.registry.cols_drop + self.registry.cols_identifiers)
        if self.registry.col_target:
            _exclude.add(self.registry.col_target)
        _exclude.add("split")

        for attr_list in [
            "cols_ohe_active", "cols_ordinal_active", "cols_target_active",
            "cols_log_active", "cols_standard_active", "cols_robust_active", "cols_binary_active"
        ]:
            setattr(self, attr_list, [c for c in getattr(self, attr_list) if c not in _exclude])

        self._log(f"\n  Listes dérivées automatiquement depuis le registre :")
        self._log(f"    OHE           : {len(self.cols_ohe_active)} cols")
        self._log(f"    Ordinal       : {len(self.cols_ordinal_active)} cols")
        self._log(f"    Target Enc.   : {len(self.cols_target_active)} cols")
        self._log(f"    Log           : {len(self.cols_log_active)} cols")
        self._log(f"    Standard      : {len(self.cols_standard_active)} cols")
        self._log(f"    Robust        : {len(self.cols_robust_active)} cols")
        self._log(f"    Binaires      : {len(self.cols_binary_active)} cols")

        # 3. Apprentissage des paramètres (sur train uniquement)
        self._learn_imputation_params(df_train)
        self._learn_log_params(df_train)
        self._learn_winsor_params(df_train)

        # 4. Mise à jour du registre avec les valeurs apprises
        self._update_registry_learned_params()

        # 5. Construction du ColumnTransformer
        self._build_preprocessor(df_train)

        self._fitted = True
        self._log("\n  ✅ FeatureConfigurator fitted.\n")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # APPRENTISSAGE DES PARAMÈTRES
    # ─────────────────────────────────────────────────────────────────────────

    def _learn_imputation_params(self, df: pd.DataFrame) -> None:
        """Apprend médianes (numériques) et modes (catégoriels) sur le train."""
        self._log("\n  ── Apprentissage imputation ...")
    
        # Silenciamos los warnings de NumPy solo durante este cálculo
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            
            num_cols = self.cols_log_active + self.cols_standard_active + self.cols_robust_active
            for col in num_cols:
                if col in df.columns and df[col].notna().any():
                    # El float() de una columna vacía tras median() da error o warning
                    val = df[col].median()
                    if pd.notna(val):
                        self.learned_medians[col] = float(val)

            cat_cols = self.cols_ohe_active + self.cols_ordinal_active + self.cols_target_active
            for col in cat_cols:
                if col in df.columns and df[col].notna().any():
                    mode_val = df[col].mode()
                    self.learned_modes[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "unknown"
            
            # Binaires → médiane (0 ou 1)
            # pd.to_numeric : gère les valeurs 'y'/'n' ou 'Y'/'N' qui arrivent
            # en string depuis PostgreSQL même après CASE WHEN … THEN 1 ELSE 0
            for col in self.cols_binary_active:
                if col in df.columns and df[col].notna().any():
                    serie_num = pd.to_numeric(df[col], errors="coerce")
                    val = serie_num.median()
                    if pd.notna(val):
                        self.learned_medians[col] = float(val)
    
        self._log(f"     {len(self.learned_medians)} médianes · {len(self.learned_modes)} modes")


    def _learn_log_params(self, df: pd.DataFrame) -> None:
        """Apprend le décalage log (pour gérer les valeurs ≤ 0)."""
        self._log("  ── Apprentissage transformation log ...")
        for col in self.cols_log_active:
            if col not in df.columns:
                continue
            col_min = df[col].min()
            decalage = max(0.0, -float(col_min) + 1.0) if (col_min is not None and not np.isnan(float(col_min) if col_min is not None else float('nan'))) else 1.0
            self.learned_log_dec[col] = decalage
            # Mise à jour dans le registre
            attr = self.registry.get_attr(col)
            if attr:
                attr.learned_log_decalage = decalage

    def _learn_winsor_params(self, df: pd.DataFrame) -> None:
        """Apprend les bornes de winsorisation (percentiles 1%-99%)."""
        self._log("  ── Apprentissage winsorisation ...")
        for col in self.cols_robust_active:
            if col not in df.columns:
                continue
            q01 = float(df[col].quantile(0.01))
            q99 = float(df[col].quantile(0.99))
            self.learned_winsor[col] = (q01, q99)
            attr = self.registry.get_attr(col)
            if attr:
                attr.learned_winsor_low  = q01
                attr.learned_winsor_high = q99

    def _update_registry_learned_params(self) -> None:
        """Synchronise les paramètres appris vers le FeatureRegistry."""
        for col, median in self.learned_medians.items():
            attr = self.registry.get_attr(col)
            if attr:
                attr.learned_median = median

        for col, mode in self.learned_modes.items():
            attr = self.registry.get_attr(col)
            if attr:
                attr.learned_mode = mode

    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRUCTION DU COLUMN TRANSFORMER
    # ─────────────────────────────────────────────────────────────────────────

    def _build_preprocessor(self, df_train: pd.DataFrame) -> None:
        """Construit le ColumnTransformer con deduplicación estricta."""
        self._log("\n  ── Construction ColumnTransformer ...")

        
        # 1. Limpieza de duplicados internos de cada lista
        self._log("\n     ── 1. Limpieza de duplicados internos de cada lista ...")
        
        self.cols_ohe_active      = list(dict.fromkeys(self.cols_ohe_active))
        self.cols_ordinal_active  = list(dict.fromkeys(self.cols_ordinal_active))
        self.cols_log_active      = list(dict.fromkeys(self.cols_log_active))
        self.cols_standard_active = list(dict.fromkeys(self.cols_standard_active))
        self.cols_robust_active   = list(dict.fromkeys(self.cols_robust_active))
        self.cols_binary_active   = list(dict.fromkeys(self.cols_binary_active))
        self.cols_target_active   = list(dict.fromkeys(self.cols_target_active))

        # 2. Gestión de Colisiones (Prioridad)
        # Una columna solo puede estar en UNA categoría. 
        # Definimos un orden de prioridad: Log > Robust > Standard > Binary
        
        seen = set()
        
        def _get_unique(current_list, already_seen):
            unique = [c for c in current_list if c not in already_seen]
            already_seen.update(unique)
            return unique

        # Aplicamos la exclusión mutua según jerarquía técnica
        self.cols_log_active      = _get_unique(self.cols_log_active, seen)
        self.cols_robust_active   = _get_unique(self.cols_robust_active, seen)
        self.cols_standard_active = _get_unique(self.cols_standard_active, seen)
        self.cols_binary_active   = _get_unique(self.cols_binary_active, seen)
        self.cols_ohe_active      = _get_unique(self.cols_ohe_active, seen)
        self.cols_ordinal_active  = _get_unique(self.cols_ordinal_active, seen)
        self.cols_target_active   = _get_unique(self.cols_target_active, seen)

        transformers = []

        # ── 1. OHE ────────────────────────────────────────────────────────
        self._log("\n     ── 1. OHE ...")

        if self.cols_ohe_active:
            ohe_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("ohe",     OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None
                ))
            ])
            transformers.append(("ohe", ohe_pipe, self.cols_ohe_active))

        # ── 2. Ordinal ────────────────────────────────────────────────────
        self._log("\n     ── 2. Ordinal ...")
        if self.cols_ordinal_active:
            cat_list = []
            for col in self.cols_ordinal_active:
                attr = self.registry.get_attr(col)
                
                # Verificamos si tenemos categorías reales en el registro
                if attr and attr.valeurs_possibles and len(attr.valeurs_possibles) > 0:
                    cat_list.append(sorted(attr.valeurs_possibles.values()))
                else:
                    # SI NO HAY, necesitamos una lista vacía o 'auto' para ESA columna.
                    # Pero Sklearn falla si mezclas. La mejor práctica es:
                    # Si falta una, usamos 'auto' para TODAS o extraemos las categorías del DF
                    cat_list = "auto"
                    break # Salimos del bucle, usaremos detección automática
            
            ord_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("ordinal", OrdinalEncoder(
                    categories=cat_list, 
                    handle_unknown="use_encoded_value", 
                    unknown_value=-1
                ))
            ])
            transformers.append(("ordinal", ord_pipe, self.cols_ordinal_active))
        
        # ── 3. Log → Standard (VERSION SÉRIALISABLE  ──────────────────────
        # ── sin lambda x: np.sign(x) * np.log1p(np.abs(x))) ───────────────
        self._log("\n     ── 3. Log → Standard ...")
        
        if self.cols_log_active:
            log_pipe = Pipeline([
                # Imputation à 0 : sûr pour counts et amounts agrégées
                ("imputer",  SimpleImputer(strategy="constant", fill_value=0)),
                # sign * log1p(|x|) : gère les valeurs négatives légitimes
                # (ex: AMT_CREDIT_SUM_DEBT min=-4.7M, AMT_CREDIT_SUM_LIMIT min=-586k)
                ("log",      FunctionTransformer(
                                 safe_log1p,  # <--- Usamos la función global
                                 validate=False,
                                 feature_names_out="one-to-one"
                             )),
                ("scaler",   StandardScaler())
            ])
            transformers.append(("log", log_pipe, self.cols_log_active))
            
        # ── 4. Standard (Version avec Indicateur de Valeurs Manquantes) ────
        self._log("\n     ── 4. Standard (Imputation Médiane + Indicateur) ...")
        
        if self.cols_standard_active:
            std_pipe = Pipeline([
                # add_indicator=True : Crée une colonne binaire supplémentaire 
                # pour chaque feature où une valeur manquante a été détectée.
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler",  StandardScaler())
            ])
            transformers.append(("standard", std_pipe, self.cols_standard_active))        
        
        # ── 5. Robust ─────────────────────────────────────────────────────
        self._log("\n     ── 5. Robust ...")
        if self.cols_robust_active:
            rob_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  RobustScaler())
            ])
            transformers.append(("robust", rob_pipe, self.cols_robust_active))

        # ── 6. Binaires (pass-through avec imputation) ────────────────────
        self._log("\n     ── 6. Binaires (pass-through avec imputation) ...")
        if self.cols_binary_active:
            # _coerce_numeric_columns est définie au niveau MODULE
            # pour être picklable par save_artifacts()
            bin_pipe = Pipeline([
                ("coerce", FunctionTransformer(
                    _coerce_numeric_columns,
                    validate=False,
                    feature_names_out="one-to-one",
                )),
                # ("imputer", SimpleImputer(strategy="most_frequent")),
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ])
            transformers.append(("binary", bin_pipe, self.cols_binary_active))
        
        # ── 7. Target encoding (simple mean encoding) ─────────────────────
        self._log("\n     ── 7. Target encoding (simple mean encoding) ...")
        # Traité séparément dans transform() car nécessite la target
        # Ici on laisse passer ces colonnes en string pour info
        if self.cols_target_active:
            te_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ])
            transformers.append(("target_enc", te_pipe, self.cols_target_active))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            n_jobs=-1
        )

        self._log("\n     ── Fit du ColumnTransformer sur les données d'entrainement ...")
        # Fit du ColumnTransformer sur les données d'entrainement
        X_train = self._prepare_df(df_train)
        
        # Les colonnes 100% NaN (ex: FLAG_DOCUMENT_* constants dans le test)
        # sont correctement gérées par fill_value=0 — le warning sklearn est informatif
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values",
                category=UserWarning,
            )
            self.preprocessor.fit(X_train)

        self._log("\n     ── preprocessor.fit(X_train) Done!")
        
        # Noms de colonnes output
        self.feature_names_out = self._get_feature_names()
        self._log(f"     ✅ {len(self.feature_names_out)} features en sortie")

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare le DataFrame (sélection des colonnes features uniquement)."""
        all_feature_cols = (
            self.cols_ohe_active + self.cols_ordinal_active + self.cols_target_active +
            self.cols_log_active + self.cols_standard_active +
            self.cols_robust_active + self.cols_binary_active
        )
        # Garder seulement les colonnes présentes
        cols_present = [c for c in all_feature_cols if c in df.columns]
        return df[cols_present].copy()

    def _get_feature_names(self) -> List[str]:
        """Récupère les noms de features en sortie du ColumnTransformer."""
        if self.preprocessor is None:
            return []
        try:
            return list(self.preprocessor.get_feature_names_out())
        except Exception:
            # Fallback : reconstituer manuellement depuis les transformers
            names = []
            for name, trans, cols in self.preprocessor.transformers_:
                if name == "remainder":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    names.extend(trans.get_feature_names_out(cols))
                elif name == "ohe":
                    # OHE : utiliser les catégories apprises
                    ohe = trans.named_steps.get("ohe")
                    if ohe is not None:
                        for i, col in enumerate(cols):
                            for cat in ohe.categories_[i]:
                                names.append(f"{col}_{cat}")
                    else:
                        names.extend([f"{col}_enc" for col in cols])
                else:
                    # Transformers scalaires : 1 feature par colonne d'entrée
                    names.extend(cols)
            return names if names else [f"feature_{i}" for i in range(100)]

    # =========================================================================
    # TRANSFORM
    # =========================================================================

    def transform(
        self,
        df: pd.DataFrame,
        return_dataframe: bool = True
    ) -> pd.DataFrame:
        """
        Transforme un DataFrame (train ou test) avec les paramètres appris.

        Args:
            df:               DataFrame à transformer
            return_dataframe: True → retourne un DataFrame nommé, False → np.ndarray

        Returns:
            DataFrame ou array transformé
        """
        if not self._fitted or self.preprocessor is None:
            raise RuntimeError("FeatureConfigurator non fitted. Appelez fit() d'abord.")

        X = self._prepare_df(df)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values",
                category=UserWarning,
            )
            X_transformed = self.preprocessor.transform(X)

        if not return_dataframe:
            return X_transformed

        df_out = pd.DataFrame(
            X_transformed,
            columns=self.feature_names_out,
            index=df.index
        )
        return df_out

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform en une seule étape (pour le train uniquement)."""
        self.fit(df_train)
        return self.transform(df_train)

    def get_target(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Extrait la variable cible du DataFrame."""
        target_col = self.registry.col_target
        if target_col and target_col in df.columns:
            return df[target_col]
        return None

    def get_X_y(
        self,
        df: pd.DataFrame,
        transform: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retourne (X, y) depuis un DataFrame.

        Args:
            df:        DataFrame
            transform: True → applique les transformations; False → X brut

        Returns:
            (X, y)
        """
        y = self.get_target(df)
        if y is None:
            raise ValueError("Colonne TARGET absente du DataFrame.")
        X = self.transform(df) if transform else self._prepare_df(df)
        return X, y

    # =========================================================================
    # TARGET ENCODING (ajouté séparément, anti-leakage)
    # =========================================================================

    def fit_target_encoding(
        self,
        df_train: pd.DataFrame,
        smoothing: float = 1.0
    ) -> "FeatureConfigurator":
        """
        Calcule les mappings de target encoding (mean encoding avec lissage).
        À appeler APRÈS fit() si des colonnes target_enc sont présentes.

        Args:
            df_train:  DataFrame train (avec TARGET)
            smoothing: Paramètre de lissage bayésien

        Returns:
            self
        """
        if not self.cols_target_active:
            return self

        target_col = self.registry.col_target
        if target_col not in df_train.columns:
            warnings.warn("TARGET absent, target encoding ignoré.")
            return self

        self.target_encoding_maps = {}
        global_mean = df_train[target_col].mean()

        for col in self.cols_target_active:
            if col not in df_train.columns:
                continue
            stats = df_train.groupby(col)[target_col].agg(["mean", "count"])
            # Lissage : (count * mean + k * global_mean) / (count + k)
            k = 1.0 / smoothing
            stats["smoothed"] = (
                (stats["count"] * stats["mean"] + k * global_mean) /
                (stats["count"] + k)
            )
            self.target_encoding_maps[col] = stats["smoothed"].to_dict()
            self._log(f"     Target enc. {col}: {len(stats)} modalités")

        return self

    # =========================================================================
    # SAUVEGARDE / CHARGEMENT DES ARTEFACTS
    # =========================================================================

    def save_artifacts(self, output_dir: str = "models/preprocessor") -> None:
        """
        Sauvegarde les artefacts du preprocessing :
            - preprocessor.pkl (ColumnTransformer fitté)
            - learned_params.json (valeurs apprises)
            - feature_names.json (noms colonnes output)
            - registry.yaml (snapshot du registre au moment du fit)
        """
        import pickle
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. ColumnTransformer
        with open(output_path / "preprocessor.pkl", "wb") as f:
            pickle.dump(self.preprocessor, f)

        # 2. Paramètres appris
        params = {
            "learned_medians":          self.learned_medians,
            "learned_modes":            self.learned_modes,
            "learned_winsor":           {k: list(v) for k, v in self.learned_winsor.items()},
            "learned_log_decalages":    self.learned_log_dec,
            "cols_ohe":                 self.cols_ohe_active,
            "cols_ordinal":             self.cols_ordinal_active,
            "cols_target_enc":          self.cols_target_active,
            "cols_log":                 self.cols_log_active,
            "cols_standard":            self.cols_standard_active,
            "cols_robust":              self.cols_robust_active,
            "cols_binary":              self.cols_binary_active,
            "columns_seen_at_fit":      self.columns_seen_at_fit,
        }
        if hasattr(self, "target_encoding_maps"):
            params["target_encoding_maps"] = self.target_encoding_maps

        with open(output_path / "learned_params.json", "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

        # 3. Noms de features output
        with open(output_path / "feature_names.json", "w") as f:
            json.dump({"feature_names_out": self.feature_names_out}, f, indent=2)

        # 4. Snapshot du registre
        self.registry.to_yaml(str(output_path / "registry_snapshot.yaml"))

        self._log(f"\n  💾 Artefacts sauvegardés : {output_path}/")
        self._log(f"     preprocessor.pkl · learned_params.json · feature_names.json · registry_snapshot.yaml")

    @classmethod
    def load_artifacts(
        cls,
        artifacts_dir: str = "models/preprocessor",
        registry: FeatureRegistry = REGISTRY
    ) -> "FeatureConfigurator":
        """
        Charge un FeatureConfigurator fitté depuis des artefacts.

        Returns:
            FeatureConfigurator prêt à transformer (sans re-fitter)
        """
        import pickle
        artifacts_path = Path(artifacts_dir)

        config = cls(registry=registry)

        # ColumnTransformer
        with open(artifacts_path / "preprocessor.pkl", "rb") as f:
            config.preprocessor = pickle.load(f)

        # Paramètres appris
        with open(artifacts_path / "learned_params.json", "r", encoding="utf-8") as f:
            params = json.load(f)

        config.learned_medians          = params.get("learned_medians", {})
        config.learned_modes            = params.get("learned_modes", {})
        config.learned_winsor           = {k: tuple(v) for k, v in params.get("learned_winsor", {}).items()}
        config.learned_log_dec          = params.get("learned_log_decalages", {})
        config.cols_ohe_active          = params.get("cols_ohe", [])
        config.cols_ordinal_active      = params.get("cols_ordinal", [])
        config.cols_target_active       = params.get("cols_target_enc", [])
        config.cols_log_active          = params.get("cols_log", [])
        config.cols_standard_active     = params.get("cols_standard", [])
        config.cols_robust_active       = params.get("cols_robust", [])
        config.cols_binary_active       = params.get("cols_binary", [])
        config.columns_seen_at_fit      = params.get("columns_seen_at_fit", [])

        if "target_encoding_maps" in params:
            config.target_encoding_maps = params["target_encoding_maps"]

        # Noms de features
        with open(artifacts_path / "feature_names.json", "r") as f:
            config.feature_names_out = json.load(f).get("feature_names_out", [])

        config._fitted = True
        return config

    # =========================================================================
    # RÉSUMÉ
    # =========================================================================

    def summary(self) -> None:
        """Affiche un résumé complet du configurateur."""
        self.registry.summary()
        if self._fitted:
            print("\n============================================================================")
            print("ÉTAT DU FIT")
            print("============================================================================")
            print(f"  Features OHE actives........: {self.cols_ohe_active}")
            print(f"  Features Log actives........: {self.cols_log_active}")
            print(f"  Features Standard (extrait).: {self.cols_standard_active[:5]}...")
            print(f"  Features en sortie..........: {len(self.feature_names_out)}")