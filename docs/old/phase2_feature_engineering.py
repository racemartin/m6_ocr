"""
Pipeline - Phase 2 : Feature Engineering & Preprocessing
=========================================================
Transforme les données brutes en dataset prêt pour l'entraînement.

⚠️ ATTENTION : Cette phase NE FAIT PAS de train_test_split
Le split sera fait en Phase 3 (train/test/evaluation)

Étapes :
    1. Chargement des données de Phase 1
    2. Sélection et nettoyage des features
    3. Construction du preprocessing pipeline sklearn
    4. Fit_transform sur TOUTES les données
    5. Sauvegarde du dataset final + ColumnTransformer

Usage :
    python -m src.pipelines.phase2_feature_engineering
"""

# --- BIBLIOTHÈQUES STANDARDS ---
import sys
import traceback
from pathlib import Path
from typing import Dict, List

# --- BIBLIOTHÈQUES TIERS (DATA) ---
import pandas as pd
import numpy as np
import joblib

# --- SCIKIT-LEARN : PREPROCESSING ---
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    FunctionTransformer
)

# --- MODULES INTERNES DU PROJET ---
from src.database import get_engine


moteur_sql            = get_engine()         # Moteur SQLAlchemy

# ##############################################################################
# CONFIGURATION DES FEATURES
# ##############################################################################
class FeatureConfig:
    """Configuration centralisée des features pour le preprocessing"""

    # --- VARIABLE CIBLE ---
    TARGET = "target_attrition"
    TARGET_INITIAL = "a_quitte_l_entreprise"

    # --- COLONNES À SUPPRIMER ---
    COLS_IDENTIFIANTS = [
        'id',
        'emp_id',  # Garder pour référence mais supprimer du training
        'id_employee',
        'eval_number',
        'code_sondage'
    ]

    COLS_CONSTANT = [
        'nombre_heures_travailless',
        'nombre_employee_sous_responsabilite',
        'ayant_enfants'
    ]

    COLS_MISSING_EXCESIF = []  # À définir selon analyse
    COLS_REDONDANTS = []       # À définir selon corrélation

    # --- FEATURES CATÉGORIELLES (ONE-HOT ENCODING) ---
    COLS_ONE_HOT_ENCODING = [
        'genre',
        'statut_marital',
        'poste',
        'domaine_etude',
        'frequence_deplacement'
    ]

    # --- FEATURES NUMÉRIQUES ---

    # Variables nécessitant transformation log (distribution asymétrique)
    COLS_TO_LOG = [
        'revenu_mensuel',
        'annee_experience_totale',
        'annees_dans_l_entreprise',
        'annees_depuis_la_derniere_promotion'
    ]

    # Variables robustes aux outliers (si nécessaire)
    COLS_TO_ROBUST = []

    # Variables pour standardisation classique
    COLS_STANDARD = [
        'age',
        'nombre_experiences_precedentes',
        'annees_dans_le_poste_actuel',
        'satisfaction_employee_environnement',
        'note_evaluation_precedente',
        'satisfaction_employee_nature_travail',
        'satisfaction_employee_equipe',
        'satisfaction_employee_equilibre_pro_perso',
        'note_evaluation_actuelle',
        'heure_supplementaires',
        'augementation_salaire_precedente',
        'nombre_participation_pee',
        'nb_formations_suivies',
        'distance_domicile_travail',
        'niveau_education',
        'annes_sous_responsable_actuel',
        # Features engineered (fe*)
        'fe1_ratio_stagnation',
        'fe2_stabilite_manager',
        'fe3_indice_job_hopping',
        'fe4_anciennete_relative',
        'fe5_satisfaction_globale',
        'fe6_risque_overwork',
        'fe7_penibilite_trajet',
        'fe8_valeur_experience'
    ]

    @classmethod
    def get_all_to_remove(cls) -> List[str]:
        """Retourne toutes les colonnes à supprimer AVANT preprocessing"""
        return (
            cls.COLS_IDENTIFIANTS +
            cls.COLS_CONSTANT +
            cls.COLS_MISSING_EXCESIF +
            cls.COLS_REDONDANTS
        )


# 1. Definir la función a nivel de módulo (fuera de la clase)
def safe_log_transform(x):
    return np.log1p(np.maximum(x, 0))



# ##############################################################################
# CLASSE PRINCIPALE : FEATURE ENGINEERING PIPELINE
# ##############################################################################
class FeatureEngineeringPipeline:
    """
    Pipeline de preprocessing pour préparer les données.
    
    ⚠️ IMPORTANT : Ne fait PAS de train/test split
    Le split sera fait en Phase 3 pour avoir train/test/evaluation
    
    Responsabilités :
    - Chargement des données preprocessed Phase 1
    - Nettoyage et sélection des features
    - Construction du ColumnTransformer sklearn
    - Fit_transform sur TOUTES les données
    - Sauvegarde du dataset final + pipeline
    """

    def __init__(
        self,
        config: FeatureConfig = None,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialise le pipeline de feature engineering.
        
        Args:
            config: Configuration des features (défaut: FeatureConfig)
            random_state: Graine aléatoire pour reproductibilité
            verbose: Affichage des logs
        """
        self.config = config or FeatureConfig()
        self.random_state = random_state
        self.verbose = verbose

        # Stockage des données
        self.df_raw = None
        self.df_cleaned = None
        self.X_prepared = None  # Features avant transformation
        self.y = None           # Target
        self.X_final = None     # Features après transformation
        self.data_output = None # X_final + y (pour sauvegarde)

        # Pipeline
        self.preprocessing_pipeline = None
        self.column_transformer = None

        # Métadonnées
        self.feature_names_in = []
        self.feature_names_out = []
        self.metadata = {}

    def _log(self, message: str, level: str = "INFO"):
        """Affiche un message si verbose activé"""
        if self.verbose:
            symbols = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "STEP": "📊"
            }
            symbol = symbols.get(level, "•")
            print(f"{symbol} {message}")



    # ==========================================================================
    # ÉTAPE 1 : CHARGEMENT DES DONNÉES
    # ==========================================================================

    def load_data(self, source: str = "database") -> pd.DataFrame:
        """
        Charge les données de Phase 1.
        
        Args:
            source: "database" ou "csv"
            
        Returns:
            DataFrame brut
        """
        self._log("Chargement des données de Phase 1...", "STEP")

        if source == "database":
            # Chargement depuis PostgreSQL
            engine = get_engine()
            query = "SELECT * FROM v_features_engineering"

            self.df_raw = pd.read_sql(query, engine)
            self._log(f"  Données chargées depuis PostgreSQL: {self.df_raw.shape}", "INFO")

        elif source == "csv":
            # Chargement depuis CSV
            csv_path = Path("data/interim/phase1_features.csv")

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Fichier {csv_path} introuvable. "
                    "Exécutez d'abord phase1_preparation.py"
                )

            self.df_raw = pd.read_csv(csv_path)
            self._log(f"  Données chargées depuis CSV: {self.df_raw.shape}", "INFO")

        else:
            raise ValueError(f"Source '{source}' non reconnue. Utilisez 'database' ou 'csv'")

        # Validation de la target
        if self.config.TARGET not in self.df_raw.columns:
            if self.config.TARGET_INITIAL in self.df_raw.columns:
                # Créer la target binaire si nécessaire
                self._log("  Création de la target binaire...", "INFO")
                self.df_raw[self.config.TARGET] = (
                    self.df_raw[self.config.TARGET_INITIAL].map({
                        'Oui': 1,
                        'Yes': 1,
                        'Non': 0,
                        'No': 0
                    })
                )
            else:
                raise ValueError(
                    f"Variable cible '{self.config.TARGET}' ou "
                    f"'{self.config.TARGET_INITIAL}' introuvable"
                )

        return self.df_raw

    # ==========================================================================
    # ÉTAPE 2 : NETTOYAGE ET SÉLECTION DES FEATURES
    # ==========================================================================

    def clean_and_select_features(self) -> tuple:
        """
        Supprime les colonnes inutiles et sépare X et y.
        
        ⚠️ NE FAIT PAS de train/test split ici !
        
        Returns:
            (X_prepared, y) : Features préparées et target
        """
        self._log("Nettoyage et sélection des features...", "STEP")

        if self.df_raw is None:
            raise ValueError("Appelez d'abord load_data()")

        df = self.df_raw.copy()

        # Colonnes à supprimer
        cols_to_remove = self.config.get_all_to_remove()

        # Ne pas supprimer la target initiale si elle diffère
        if self.config.TARGET_INITIAL != self.config.TARGET:
            cols_to_remove.append(self.config.TARGET_INITIAL)

        cols_to_remove_present = [c for c in cols_to_remove if c in df.columns]

        self._log(f"  Suppression de {len(cols_to_remove_present)} colonnes...", "INFO")

        # Séparation X et y AVANT suppression
        if self.config.TARGET not in df.columns:
            raise ValueError(f"Target '{self.config.TARGET}' introuvable")

        self.y = df[self.config.TARGET].copy()

        # Suppression des colonnes inutiles (incluant la target)
        df = df.drop(columns=cols_to_remove_present + [self.config.TARGET], errors='ignore')

        # Stockage
        self.X_prepared = df
        self.feature_names_in = df.columns.tolist()

        self._log(f"  Features préparées: {len(self.feature_names_in)}", "SUCCESS")
        self._log(f"  Shape X: {self.X_prepared.shape}", "INFO")
        self._log(f"  Shape y: {self.y.shape}", "INFO")

        return self.X_prepared, self.y

    # ==========================================================================
    # ÉTAPE 3 : DÉTECTION AUTOMATIQUE DES TYPES
    # ==========================================================================

    def detect_feature_types(self) -> Dict[str, List[str]]:
        """
        Détecte automatiquement les types de features présentes dans X_prepared.
        
        Returns:
            Dictionnaire {type: [colonnes]}
        """
        self._log("Détection automatique des types de features...", "STEP")

        if self.X_prepared is None:
            raise ValueError("Appelez d'abord clean_and_select_features()")

        df = self.X_prepared

        # Détection
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols  = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Filtrage selon config (colonnes qui existent réellement)
        categorical_to_encode = [
            c for c in self.config.COLS_ONE_HOT_ENCODING
            if c in categorical_cols
        ]

        numerical_to_log = [
            c for c in self.config.COLS_TO_LOG
            if c in numerical_cols
        ]

        numerical_to_robust = [
            c for c in self.config.COLS_TO_ROBUST
            if c in numerical_cols
        ]

        numerical_to_standard = [
            c for c in self.config.COLS_STANDARD
            if c in numerical_cols and c not in numerical_to_log and c not in numerical_to_robust
        ]

        # Colonnes restantes (passthrough)
        all_transformed = categorical_to_encode + numerical_to_log + numerical_to_robust + numerical_to_standard
        remaining_cols = [c for c in df.columns if c not in all_transformed]

        feature_types = {
            'categorical': categorical_cols,
            'numerical': numerical_cols,
            'to_encode': categorical_to_encode,
            'to_log': numerical_to_log,
            'to_robust': numerical_to_robust,
            'to_standard': numerical_to_standard,
            'remaining': remaining_cols
        }

        self._log(f"  Catégorielles détectées....: {len(categorical_cols)}", "INFO")
        self._log(f"  Numériques détectées.......: {len(numerical_cols)}", "INFO")
        self._log(f"  À encoder (One-Hot)........: {len(categorical_to_encode)}", "INFO")
        self._log(f"  À transformer (log)........: {len(numerical_to_log)}", "INFO")
        self._log(f"  À standardiser.............: {len(numerical_to_standard)}", "INFO")
        self._log(f"  Restantes (passthrough)....: {len(remaining_cols)}", "INFO")

        return feature_types

    # ==========================================================================
    # ÉTAPE 4 : CONSTRUCTION DU PREPROCESSING PIPELINE
    # ==========================================================================

    def build_preprocessing_pipeline(self) -> Pipeline:
        """
        Construit le pipeline de preprocessing sklearn.
        
        Structure :
            Pipeline([
                ('preproc', ColumnTransformer([
                    ('c_ohe', OneHotEncoder, categorical_cols),
                    ('n_log', log_transform + StandardScaler, numerical_log_cols),
                    ('n_std', StandardScaler, numerical_standard_cols)
                ]))
            ])
        
        Returns:
            Pipeline sklearn configuré
        """
        self._log("Construction du pipeline de preprocessing...", "STEP")

        feature_types = self.detect_feature_types()

        # --- TRANSFORMERS ---

        # 1. Catégorielles : OneHotEncoder
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        # 2. Numériques avec log : log1p + StandardScaler
        #def safe_log_transform(X):
        #    """Applique log(1 + x) de manière sûre"""
        #    return np.log1p(np.clip(X, 0, None))

        numerical_log_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(safe_log_transform, validate=False)),
            ('scaler', StandardScaler())
        ])

        # 3. Numériques avec robust : RobustScaler
        numerical_robust_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        # 4. Numériques standard : StandardScaler
        numerical_standard_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # --- COLUMN TRANSFORMER ---
        transformers_list = []

        if feature_types['to_encode']:
            transformers_list.append(('c_ohe', categorical_transformer, feature_types['to_encode']))

        if feature_types['to_log']:
            transformers_list.append(('n_log', numerical_log_transformer, feature_types['to_log']))

        if feature_types['to_robust']:
            transformers_list.append(('n_robust', numerical_robust_transformer, feature_types['to_robust']))

        if feature_types['to_standard']:
            transformers_list.append(('n_std', numerical_standard_transformer, feature_types['to_standard']))

        column_transformer = ColumnTransformer(
            transformers=transformers_list,
            # remainder='passthrough',  # Garder les colonnes non transformées
            remainder='drop', # Si algo se escapó, mejor que no entre al modelo
            verbose_feature_names_out=True  # Préfixes explicites
        )

        # --- PIPELINE COMPLET ---
        preprocessing_pipeline = Pipeline(steps=[
            ('preproc', column_transformer)
        ])

        # Stockage
        self.preprocessing_pipeline = preprocessing_pipeline
        self.column_transformer = column_transformer

        self._log(f"  Pipeline créé avec {len(transformers_list)} transformers", "SUCCESS")

        return preprocessing_pipeline

    # ==========================================================================
    # ÉTAPE 5 : FIT_TRANSFORM SUR TOUTES LES DONNÉES
    # ==========================================================================

    def fit_transform_all(self) -> pd.DataFrame:
        """
        Fit et transform le pipeline sur TOUTES les données.
        
        ⚠️ Pas de split ici - on transforme tout le dataset !
        
        Returns:
            DataFrame transformé (X_final)
        """
        self._log("Fit_transform du pipeline sur toutes les données...", "STEP")

        if self.preprocessing_pipeline is None:
            self.build_preprocessing_pipeline()

        if self.X_prepared is None:
            raise ValueError("Appelez d'abord clean_and_select_features()")

        # Configuration de sortie Pandas
        self.preprocessing_pipeline.set_output(transform="pandas")

        # Fit_transform sur TOUT le dataset
        self.X_final = self.preprocessing_pipeline.fit_transform(self.X_prepared, self.y)

        # Récupération des noms de features
        try:
            self.feature_names_out = self.X_final.columns.tolist()
            self._log(f"  Features après transformation: {len(self.feature_names_out)}", "SUCCESS")
        except Exception:
            self._log("  Impossible de récupérer les noms des features", "WARNING")

        self._log(f"  Shape finale: {self.X_final.shape}", "INFO")

        # Vérification NaNs
        nans_count = self.X_final.isna().sum().sum()
        if nans_count > 0:
            self._log(f"  ⚠️ Attention: {nans_count} NaNs détectés après transformation", "WARNING")
        else:
            self._log("  ✓ Aucun NaN après transformation", "INFO")

        return self.X_final

    # ==========================================================================
    # ÉTAPE 6 : CONSOLIDATION X + y
    # ==========================================================================

    def consolidate_dataset(self) -> pd.DataFrame:
        """
        Fusionne X_final et y en un seul DataFrame pour sauvegarde.
        
        Returns:
            DataFrame consolidé (X_final + y)
        """
        self._log("Consolidation du dataset (X + y)...", "STEP")

        if self.X_final is None or self.y is None:
            raise ValueError("Appelez d'abord fit_transform_all()")

        # Reset des index pour éviter les problèmes
        X_reset = self.X_final.reset_index(drop=True)
        y_reset = self.y.reset_index(drop=True)

        # Concaténation
        self.data_output = pd.concat([X_reset, y_reset], axis=1)

        self._log(f"  Dataset consolidé: {self.data_output.shape}", "SUCCESS")

        return self.data_output

    # ==========================================================================
    # ÉTAPE 7 : SAUVEGARDE DES ARTEFACTS
    # ==========================================================================

    def auditer_features_evolution(self, df_phase2):
        import pandas as pd
        from sqlalchemy import text
        from src.database import get_engine
        import re

        try:
            moteur_sql = get_engine()

            # 1. Recuperar el PATH del fichero de la Phase 1 desde la DB
            with moteur_sql.connect() as conn:
                res = conn.execute(text("SELECT file_path FROM datasets ORDER BY created_at DESC LIMIT 1"))
                row = res.fetchone()
                path_p1 = row[0] if row else None

            if not path_p1:
                print("⚠️ Phase 1 file path non trouvé dans la base.")
                return

            # 2. Extraer nombres "ancianos" directamente del fichero
            # Leemos solo la primera fila (header) para no cargar todo el CSV
            df_p1_header = pd.read_csv(path_p1, nrows=0)
            nombres_ancianos = set(df_p1_header.columns)

            # 3. Limpiar prefijos de los nombres nuevos (Phase 2)
            nombres_p2_raw = [c for c in df_phase2.columns if c != "Attrition"]

            # Función para limpiar: quita 'n_std__', 'c_ohe__', 'n_log__', etc.
            def limpiar_prefijo(nombre):
                return re.sub(r'^[a-z]_[a-z]+__', '', nombre)

            # Mapeo: {Nombre_Limpio: Nombre_Con_Prefijo}
            mapeo_p2 = {limpiar_prefijo(c): c for c in nombres_p2_raw}
            nombres_p2_limpios = set(mapeo_p2.keys())

            # 4. Comparar
            nuevas_reales = nombres_p2_limpios - nombres_ancianos

            if self.verbose:
                print("\n--- 🔍 AUDIT D'ÉVOLUTION (SOURCE: CSV) ---")
                print(f"  Fichier P1: {path_p1}")
                print(f"  Variables initiales : {len(nombres_ancianos)}")
                print(f"  Variables actuelles  : {len(nombres_p2_raw)}")

                if nuevas_reales:
                    print("  ✨ NUEVAS FEATURES (Post-Engineering) :")
                    for limpia in sorted(nuevas_reales):
                        print(f"     + {mapeo_p2[limpia]} (derivada de {limpia})")

                print("------------------------------------------\n")

            # 1. Identificar las variables de Feature Engineering (tus creaciones)
            # Buscamos las que tienen el patrón 'fe' seguido de un número
            verdaderas_nuevas = [c for c in df_phase2.columns if "__fe" in c]

            # 2. Mostrar solo las 6 "reales"
            print("\n--- 💎 LAS CREACIONES DEL DATA SCIENTIST ---")
            if verdaderas_nuevas:
                for i, col in enumerate(sorted(verdaderas_nuevas), 1):
                    # Extraemos el nombre original sin prefijos para que sea legible
                    nombre_limpio = col.split("__")[-1]
                    print(f"  {i}. {col}  -> (Concepto: {nombre_limpio})")
            print("----------------------------------------------\n")

        except Exception as e:
            print(f"⚠️ Erreur audit : {e}")

    def save_artifacts(self, output_dir: Path = None) -> Dict[str, Path]:
        """
        Sauvegarde tous les artefacts nécessaires pour Phase 3.
        
        Args:
            output_dir: Dossier de sortie
            
        Returns:
            Dictionnaire des chemins de sauvegarde
        """
        self._log("Sauvegarde des artefacts...", "STEP")

        if output_dir is None:
            output_dir = Path("data/processed")

        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # 1. Dataset final (X + y)
        if self.data_output is None:
            self.consolidate_dataset()

        final_csv_path = output_dir / "phase2_data_final.csv"
        self.data_output.to_csv(final_csv_path, index=False)
        saved_paths['data_final'] = final_csv_path
        self._log(f"  ✓ Dataset final: {final_csv_path}", "INFO")

        # 2. ColumnTransformer seul (pour production)
        transformer_path = output_dir / "phase2_BEST_COLUMNTRANSFORMER_PACK.joblib"
        export_data = {
            'preprocessor': self.column_transformer
        }
        joblib.dump(export_data, transformer_path)
        saved_paths['column_transformer'] = transformer_path
        self._log(f"  ✓ ColumnTransformer: {transformer_path}", "INFO")

        # 3. Pipeline complet
        pipeline_path = output_dir / "phase2_preprocessing_pipeline.joblib"
        joblib.dump(self.preprocessing_pipeline, pipeline_path)
        saved_paths['pipeline'] = pipeline_path
        self._log(f"  ✓ Pipeline complet: {pipeline_path}", "INFO")

        # 4. Métadonnées
        metadata = {
            'feature_names_in': self.feature_names_in,
            'feature_names_out': self.feature_names_out,
            'target': self.config.TARGET,
            'shape_input': self.X_prepared.shape,
            'shape_output': self.X_final.shape,
            'total_samples': len(self.data_output),
            'attrition_rate': float(self.y.mean()),
            'categorical_encoded': self.detect_feature_types()['to_encode'],
            'numerical_log': self.detect_feature_types()['to_log'],
            'numerical_standard': self.detect_feature_types()['to_standard'],
        }

        metadata_path = output_dir / "phase2_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        saved_paths['metadata'] = metadata_path
        self._log(f"  ✓ Métadonnées: {metadata_path}", "INFO")

        # 5. X et y séparés (pour Phase 3)
        X_path = output_dir / "phase2_X_preprocessed.csv"
        y_path = output_dir / "phase2_y_target.csv"

        self.X_final.to_csv(X_path, index=False)
        self.y.to_csv(y_path, index=False)

        saved_paths['phase2_X_preprocessed'] = X_path
        saved_paths['phase2_y_target']       = y_path
        self._log("  ✓ X et y séparés sauvegardés", "INFO")


        # ----------------------------------------------------------------------
        # 💾 REGISTRE DANS LE FEATURE STORE (MÉTADONNÉES DE CONTRÔLE)
        # ----------------------------------------------------------------------
        # Antes de la inserción SQL:
        self.auditer_features_evolution(self.data_output)

        from sqlalchemy import text
        import json
        from src.database import get_engine

        if self.verbose:
            print("\n🚀 Publication des métadonnées dans le Feature Store...")

        try:
            moteur_sql = get_engine()

            # 1. Préparation des variables de chemin et noms
            chemin_final_str = str(final_csv_path.absolute())
            target_name      = "Attrition"

            # 2. Calcul des métriques du fichier (X + y)
            df_final      = self.data_output
            total_rows    = int(len(df_final))

            # Extraire les noms des features (X) uniquement
            columnas_x = df_final.columns.tolist()
            if target_name in columnas_x:
                columnas_x.remove(target_name)

            num_features = len(columnas_x)

            with moteur_sql.begin() as conn:
                # 3. Récupération de l'ID du dataset parent
                res = conn.execute(text("SELECT id FROM datasets ORDER BY created_at DESC LIMIT 1"))
                row = res.fetchone()

                if row is None:
                    raise ValueError("Aucun enregistrement parent trouvé dans 'datasets'.")

                db_dataset_id = row[0]

                # 4. Insertion dans feature_store
                query = text("""
                    INSERT INTO feature_store 
                    (dataset_id, file_path, row_count, feature_count, feature_names, target_name)
                    VALUES 
                    (:ds_id, :path, :rows, :f_count, :f_names, :target)
                """)

                conn.execute(query, {
                    "ds_id"   : db_dataset_id,
                    "path"    : chemin_final_str,
                    "rows"    : total_rows,
                    "f_count" : num_features,
                    "f_names" : json.dumps(columnas_x),
                    "target"  : target_name
                })

            if self.verbose:
                print("  ✅ Feature Store mis à jour avec succès.")
                print(f"  📊 Dimensions : {total_rows} lignes x {num_features} features.")
                print(f"  📍 Référence  : {chemin_final_str}")

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Erreur d'archivage SQL : {e}")



        return saved_paths


# ##############################################################################
# FONCTION PRINCIPALE : EXÉCUTION COMPLÈTE PHASE 2
# ##############################################################################
def executer_phase2_feature_engineering(
    data_source: str = "database",
    random_state: int = 42,
    verbose: bool = True
) -> FeatureEngineeringPipeline:
    """
    Exécute la Phase 2 complète : Feature Engineering & Preprocessing.
    
    ⚠️ NE FAIT PAS de train/test split - c'est pour Phase 3 !
    
    Args:
        data_source: "database" ou "csv"
        random_state: Graine aléatoire
        verbose: Affichage des logs
        
    Returns:
        Instance du pipeline configurée
    """
    if verbose:
        print("\n" + "="*70)
        print("🚀 PHASE 2 : FEATURE ENGINEERING & PREPROCESSING")
        print("="*70)

    # Initialisation du pipeline
    pipeline = FeatureEngineeringPipeline(
        random_state=random_state,
        verbose=verbose
    )

    # Exécution séquentielle
    try:
        # 1. Chargement
        pipeline.load_data(source=data_source)

        # 2. Nettoyage et séparation X/y
        pipeline.clean_and_select_features()

        # 3. Construction du pipeline
        pipeline.build_preprocessing_pipeline()

        # 4. Fit_transform sur TOUTES les données
        pipeline.fit_transform_all()

        # 5. Consolidation
        pipeline.consolidate_dataset()

        # 6. Sauvegarde
        saved_paths = pipeline.save_artifacts()

        # Rapport final
        if verbose:
            print("\n" + "="*70)
            print("✅ PHASE 2 TERMINÉE AVEC SUCCÈS")
            print("="*70)
            print(f"  Total samples........: {len(pipeline.data_output)}")
            print(f"  Features initiales...: {len(pipeline.feature_names_in)}")
            print(f"  Features finales.....: {len(pipeline.feature_names_out)}")
            print(f"  Taux attrition.......: {pipeline.y.mean():.2%}")
            print("\n  📦 Artefacts sauvegardés:")
            for name, path in saved_paths.items():
                print(f"     • {name}: {path.name}")
            print("\n  ➡️  Les données sont prêtes pour Phase 3 (train/test/eval split)")

        return pipeline

    except Exception as e:
        if verbose:
            print(f"\n❌ ERREUR DURANT PHASE 2: {e}")
            traceback.print_exc()
        raise


# ##############################################################################
# POINT D'ENTRÉE DU SCRIPT
# ##############################################################################
def main():
    """Point d'entrée pour l'exécution en ligne de commande."""
    try:
        # Exécution du pipeline
        _ = executer_phase2_feature_engineering(
            data_source="database",  # ou "csv"
            random_state=42,
            verbose=True
        )

        print("\n✅ Pipeline Phase 2 terminé avec succès !")
        print("   → Dataset final: data/processed/data_final.csv")
        print("   → ColumnTransformer: data/processed/0_BEST_COLUMNTRANSFORMER_PACK.joblib")
        print("   → Prêt pour Phase 3 (train/test/evaluation split + training)")

        return 0  # Code de sortie : Succès

    except Exception as erreur:
        print(f"\n❌ ERREUR CRITIQUE : {erreur}", file=sys.stderr)
        traceback.print_exc()
        return 1  # Code de sortie : Échec


if __name__ == "__main__":
    sys.exit(main())


# Usage:
# uv run python -m src.pipelines.phase2_feature_engineering
