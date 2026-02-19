"""
Module de prétraitement des données pour le Feature Engineering.
Implémente les étapes de nettoyage initial et d'élimination des colonnes problématiques.

Auteur: Rafael CEREZO MARTIN 
Date: 2025
"""

import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.neighbors       import BallTree
import math                                   # Calculs mathématiques de base


from   sklearn.preprocessing import (
    StandardScaler,                           # Normalisation des données
                           # Transformations personnalisées
)


# from seattle_custom_funtions      import calculer_distances_points_cles
# from seattle_custom_funtions      import calcular_densite_voisinage_with_reference_tree

class DataCleaner:

    def ____CONSTRUCTOR(self): pass

    """
    Classe principale pour le nettoyage et l'analyse préliminaire des données.
    """
    def __init__(self, df: pd.DataFrame = None, verbose: bool = True):
        """
        Initialise le nettoyeur de données.
        
        Args:
            df: DataFrame à nettoyer
            verbose: Afficher les messages de progression
        """
        # Conversion automatique Polars vers Pandas si nécessaire
        # Verificación de Polars
        if df is not None and hasattr(df, 'to_pandas'):
            df = df.to_pandas()

        # PROTECCIÓN: Solo copiamos si df no es None
        self.df_original = df.copy() if df is not None else None
        self.df          = df.copy() if df is not None else None
        self.verbose     = verbose

        # Historique des opérations
        self.history = []
        self.columns_suppressed = []
        self.columns_added = []
        self.features_creadas = []


        self.params = {} # Pour stocker les moyennes, bins, etc.

        self.FEATURE_TARGET = 'SiteEnergyUse(kBtu)'

        # --
        self.COLS_SPECIFIQUES         = {
                                         'ZipCode': 98101,       # Code postal par défaut de la ville
                                         'NumberofBuildings': 1  # Par défaut, 1 bâtiment
                                        }

        # --- LISTE DE COLUMNAS (Pour selection) ---
        self.FEATURES_TO_REMOVE            = []  # COLS_IDENTIFIANTS + COLS_CONSTANT + COLS_MISSING_EXCESIF + self.COLS_REDONDANTS
        self.COLS_IDENTIFIANTS             = []
        self.COLS_CONSTANT                 = []
        self.COLS_MISSING_EXCESIF          = []
        self.COLS_REDONDANTS               = []

        # --- CATEGORICAL  ---
        self.COLS_CATEGORICAL              = []  # Todas las categóricas detectadas
        self.COLS_ONE_HOT_ENCODING         = []
        self.COLS_BINARY_ENCODING          = []
        self.COLS_TARGET_ENCODING          = []
        self.COLS_TARGET_ADVANCED_ENCODING = []

        # --- NUMERICAL ---
        self.COLS_NUMERICAL                = []  # Todas las numéricas detectadas
        self.COLS_TO_LOG                   = []  # Candidatas a log(1+x) par asymétrie
        self.COLS_TO_ROBUST                = []
        self.COLS_STANDARD                 = []


        # --- 2. IMPUTACIÓN (CONNAISSANCE APPRISE) ---
        self.learned_medians  = {}  # {colonne: valeur_médiane}
        self.learned_modes    = {}  # {colonne: valeur_mode}

        # --- 3. OUTLIERS ET DISTRIBUTIONS ---
        self.learned_winsor   = {}  # {colonne: (min_val, max_val)}

        # --- 4. FEATURES SPATIALES ET INDICES ---
        self.reference_tree   = None
        self.media_referencia_SiteEUI = None

        # --- 5. ENCODAGE ET CATÉGORIES ---
        self.ohe_columns      = []  # La "mémoire" du fit pour One-Hot
        self.mappings         = {}  # Dictionnaires par catégorie (Target Encoding)
        self.global_means      = {}  # Moyenne globale pour catégories inconnues

        self.columns_metadata = {}

        # --- 6. MISE À L'ÉCHELLE (SCALING) ---
        self.learned_scaler   = {}  # {colonne: {'mean': m, 'std': s}}

    # =========================================================================
    # TOOLS
    # =========================================================================
    def ____TOOLS(self): pass

    def get_dataframe(self) -> pd.DataFrame:
        """Retourne le DataFrame nettoyé."""
        return self.df.copy()

    def get_history(self) -> List[Dict]:
        """Retourne l'historique des opérations."""
        return self.history

    def reset(self) -> 'DataCleaner':
        """Réinitialise le DataFrame à son état original."""
        self.df = self.df_original.copy()
        self.history = []
        self.columns_suppressed = []
        return self

    # =========================================================================
    # Analyse_preliminaire
    # =========================================================================
    def ____Analyse_preliminaire(self): pass

    def display_feature_summary(self):
        """
        Rapport d'architecture aligné : Les deux points sont fixés sur une colonne verticale.
        """
        # 1. RÉCUPÉRATION DES DONNÉES DE BASE
        res = self._identifier_colonnes_par_types()

        # 2. REMPLISSAGE DES LISTES MAÎTRES (Usando tu lógica de identificación)
        self.COLS_NUMERICAL   = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.COLS_CATEGORICAL = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        # 3. ESTRUCTURA DE SECCIONES
        sections = {
            "🔴 REMOVAL (Drop candidates)": {
                "🗑️ Total Identifiants": self.COLS_IDENTIFIANTS,
                "🗑️ Constant Features": self.COLS_CONSTANT,
                "🗑️ Excessive Missing": self.COLS_MISSING_EXCESIF,
                "🗑️ Redundant/Correlated": self.COLS_REDONDANTS
            },
            "🔹 CATEGORICAL STRATEGIES": {
                "One-Hot Encoding": self.COLS_ONE_HOT_ENCODING,
                "Binary Encoding": self.COLS_BINARY_ENCODING,
                "Target Encoding": self.COLS_TARGET_ENCODING,
                "Advanced Target": self.COLS_TARGET_ADVANCED_ENCODING
            },
            "🔸 NUMERICAL STRATEGIES": {
                "Log(1+x) Transform": self.COLS_TO_LOG,
                "Robust Scaling": self.COLS_TO_ROBUST,
                "Standard Scaling": self.COLS_STANDARD
            }
        }

        # 4. CONFIGURACIÓN DE ALINEACIÓN
        W_TEXT = 35  # Aumentado un poco para dar aire a los emojis
        W_NUM  = 5

        print("\n" + "="*95)
        print(f"{'ARCHITECTURES DES FEATURES & STRATÉGIES':^95}")
        print("="*95)

        grand_total = 0

        for section_name, subsections in sections.items():
            print(f"\n{section_name}")
            print("-" * 95)
            section_count = 0

            for sub_name, col_list in subsections.items():
                count = len(col_list)
                section_count += count
                preview = ", ".join(col_list[:3]) + ("..." if count > 15 else "")

                # LA CLAVE: El ':' está fuera de las llaves de ancho fijo
                # {sub_name:<{W_TEXT}} -> Nombre a la izquierda
                # :                    -> Separador fijo
                # {count:>{W_NUM}}    -> Número a la derecha
                print(f"  {sub_name:<{W_TEXT}} : {count:>{W_NUM}} | {preview}")

            print(f"  {'-'*W_TEXT}")
            print(f"  {'> TOTAL SECTION':<{W_TEXT}} : {section_count:>{W_NUM}}")
            grand_total += section_count

        print("\n" + "="*95)
        print("📊 RÉCAPITULATIF FINAL")
        # Alineación idéntica para el resumen final
        print(f"  {'Total 🔸 Numerical Detectées':<{W_TEXT}} : {len(self.COLS_NUMERICAL):>{W_NUM}}")
        print(f"  {'Total 🔹 Categorical Detectées':<{W_TEXT}} : {len(self.COLS_CATEGORICAL):>{W_NUM}}")
        print(f"  {'Total ✅ Features Classées':<{W_TEXT}} : {grand_total:>{W_NUM}}")
        print("="*95)

        return grand_total


    # ############################################################################
    # MÉTHODE : effectuer_audit_final(X_train, X_test, y_train, y_test)
    # ############################################################################
    def effectuer_audit_final(self, X_train, X_test, y_train, y_test):
        """
        Audit de sécurité critique avant la phase d'apprentissage.
        """
        # ------------------------------------------------------------------------
        # I. TESTS DE SÉCURITÉ CRITIQUES (BREAKPOINTS)
        # ------------------------------------------------------------------------
        # Test A: Correspondance stricte entre Features et Target
        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
            raise ValueError(f"❌ ERREUR CRITIQUE : Désalignement X/y ! "
                             f"Train: {len(X_train)}/{len(y_train)}")

        # Test B: Présence de valeurs manquantes
        nan_count = X_train.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"❌ ERREUR CRITIQUE : {nan_count} NaNs détectés !")

        # Test C: Intégrité des colonnes (Train vs Test)
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("❌ ERREUR CRITIQUE : Les colonnes ne sont pas "
                             "identiques ou leur ordre diffère.")

        # ------------------------------------------------------------------------
        # II. CALCULS DES INDICATEURS D'AUDIT
        # ------------------------------------------------------------------------
        total_samples    = X_train.shape[0] + X_test.shape[0]
        nb_features      = X_train.shape[1]
        nb_obj_train     = X_train.select_dtypes(include=['object']).shape[1]

        # Analyse de la standardisation
        num_cols         = X_train.select_dtypes(include=[np.number]).columns
        non_binary       = [c for c in num_cols if X_train[c].nunique() > 2]
        mean_check       = (X_train[non_binary].mean().abs() < 0.1).mean() * 100
        std_check        = ((X_train[non_binary].std() - 1).abs() < 0.1).mean() * 100

        # ------------------------------------------------------------------------
        # III. AFFICHAGE DU RAPPORT DE VALIDATION
        # ------------------------------------------------------------------------
        print(f"\n{'='*70}")
        print("VÉRIFICATIONS FINALES")
        print(f"{'='*70}")

        print(f"\n✅ 1. Dimensions identiques\n   Train : {X_train.shape}\n   "
              f"Test  : {X_test.shape}")
        print(f"\n✅ 2. Colonnes alignées\n   Nombre de features : {nb_features}")
        print("\n✅ 3. Pas de valeurs manquantes\n   Status : 0 missing")

        print("\n📊 4. État des types de données")
        print(f"   Colonnes textuelles : {nb_obj_train}")
        if nb_obj_train > 0:
            print(f"   ⚠️ ALERTE : {nb_obj_train} colonnes nécessitent "
                  f"un encodage via le Pipeline.")

        print("\n✅ 5. Cibles alignées avec les features")
        print(f"   Train : {len(X_train)} samples OK")

        print("\n✅ 6. Standardisation (colonnes non-binaires)")
        print(f"   Colonnes avec μ ≈ 0 : {mean_check:.1f}%")
        print(f"   Colonnes avec σ ≈ 1 : {std_check:.1f}%")

        if mean_check < 90 or std_check < 50:
            print("   ⚠️ ALERTE : La standardisation semble incomplète.")

        print(f"\n✅ 7. Répartition Train/Test\n   Ratio : "
              f"{(len(X_train)/total_samples):.1%} / "
              f"{(len(X_test)/total_samples):.1%}")

        print("\n✅ 8. Distribution de la cible")
        print(f"   Train - μ: {y_train.mean():.2f}, σ: {y_train.std():.2f}")
        print(f"   Test  - μ: {y_test.mean():.2f}, σ: {y_test.std():.2f}")

        # ------------------------------------------------------------------------
        # IV. RÉSUMÉ ET DÉCISION
        # ------------------------------------------------------------------------
        print(f"\n{'='*70}")
        print("✅ TOUTES LES VÉRIFICATIONS SONT PASSÉES !")
        print(f"{'='*70}")
        print(f"🚀 Prêt pour la modélisation avec {nb_features} features")
        print(f"📊 Total : {total_samples} observations")
        print(f"{'='*70}\n")

    def normaliser_categories(self, df):
        """
        Normalise toutes les colonnes de type 'object' : 
        Majuscules, suppression des espaces inutiles et gestion des NaN.
        """
        # On identifie les colonnes textuelles
        cols_texte = df.select_dtypes(include=['object']).columns

        for col in cols_texte:
            # 1. Convertir en string (pour éviter les erreurs sur les types mixtes)
            # 2. Tout en majuscules
            # 3. Supprimer les espaces au début et à la fin (strip)
            df[col] = df[col].astype(str).str.upper().str.strip()

            # 4. (Optionnel) Reconvertir les chaînes 'NAN' ou 'NONE' en vrais NaN de Numpy
            df[col] = df[col].replace(['NAN', 'NONE', 'NULL', ''], np.nan)

        print(f"✨ Normalisation terminée sur {len(cols_texte)} colonnes.")
        return df


    def missing_summary(self, df, threshold=100.0):
            """
            Génère un résumé statistique et identifie les colonnes critiques.
            
            Args:
                df: Le DataFrame à analyser.
                threshold: Le seuil (%) pour classer une colonne comme 'excessive'.
                
            Returns:
                missing_stats (pd.DataFrame): Résumé trié des nuls.
                cols_excessives (list): Liste des colonnes >= threshold.
            """
            # 1. Calcul des statistiques de base
            null_counts = df.isnull().sum().values
            total_len   = len(df)

            missing_stats = pd.DataFrame({
                'Column': df.columns,
                'Missing_Count': null_counts,
                'Missing_Pct': (null_counts / total_len * 100),
                'Dtype': df.dtypes.values
            })

            # 2. Filtrage pour le résumé (uniquement celles qui ont des nuls)
            missing_stats_filtered = missing_stats[missing_stats['Missing_Count'] > 0]
            missing_stats_filtered = missing_stats_filtered.sort_values('Missing_Pct', ascending=False)

            # 3. Identification locale des colonnes excessives
            cols_excessives = missing_stats[missing_stats['Missing_Pct'] >= threshold]['Column'].tolist()

            # 4. Affichage de l'alerte si des colonnes dépassent le seuil
            if cols_excessives:
                print(f"\n🚨 ALERTE : {len(cols_excessives)} colonnes dépassent le seuil de {threshold}% de nuls.")
                print(f"Colonnes à supprimer : {cols_excessives}")

            return missing_stats_filtered.reset_index(drop=True), cols_excessives


    # ##############################################################################
    # FONCTION : MISSING_HEATMAP
    # ##############################################################################

    def missing_heatmap(self, df, figsize=(14, 10), cmap='viridis'):
        """
        Crée une cartographie visuelle des patterns de valeurs manquantes.
        """
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()                   # Conversion de sécurité

        # Création de la matrice binaire (True si manquant)
        missing_matrix = df.isnull()

        # Identification des colonnes problématiques
        cols_with_miss = missing_matrix.columns[missing_matrix.any()].tolist()

        if not cols_with_miss:
            print("⚠️ Info..................: Aucune valeur manquante détectée.")
            return None

        # --------------------------------------------------------------------------
        # GÉNÉRATION DU GRAPHIQUE SEABORN
        # --------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            missing_matrix[cols_with_miss].T,     # Transposition pour lecture en Y
            cmap      = cmap,                     # Palette de couleurs
            cbar_kws  = {'label': 'Manquant (1)'},# Légende de la barre latérale
            yticklabels = True,                   # Affichage des noms de variables
            xticklabels = False,                  # Masquage des index individuels
            ax        = ax                        # Axe de dessin
        )

        ax.set_title('Cartographie des Valeurs Manquantes', fontsize=14, pad=20)

        plt.tight_layout()
        return fig

    # ##############################################################################
    # FONCTION : MISSING_BY_TYPE
    # ##############################################################################

    def missing_by_type(self, df):
        """
        Analyse séparée des manquants selon le type : numérique vs catégoriel.
        """
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()

        # Séparation des types de données
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # --------------------------------------------------------------------------
        # FONCTION INTERNE DE GÉNÉRATION DE RÉSUMÉ
        # --------------------------------------------------------------------------
        def _create_sub_summary(columns):
            if not columns: return pd.DataFrame() # Sécurité si liste vide

            counts  = df[columns].isnull().sum().values
            pcts    = (counts / len(df) * 100)

            res     = pd.DataFrame({
                'Column'        : columns,
                'Missing_Count' : counts,
                'Missing_Pct'   : pcts
            })
            return res[res['Missing_Count'] > 0].sort_values('Missing_Pct',
                                                           ascending=False)

        num_missing = _create_sub_summary(num_cols)
        cat_missing = _create_sub_summary(cat_cols)

        return num_missing.reset_index(drop=True), cat_missing.reset_index(drop=True)

    # ##############################################################################
    # FONCTION : MISSING_THRESHOLD_FILTER
    # ##############################################################################

    def missing_threshold_filter(self, df, threshold=50.0, return_type='list'):
        """
        Identifie les colonnes dépassant un seuil critique de données manquantes (en %).
        Uniformisé avec missing_summary pour utiliser une échelle de 0 à 100.
        """
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()

        # 1. Calcul du pourcentage de perte (0-100)
        missing_pct = (df.isnull().sum() / len(df)) * 100

        # 2. Filtrage selon le seuil (ex: 50.0)
        high_missing = missing_pct[missing_pct >= threshold]

        if return_type == 'list':
            return high_missing.index.tolist()

        # 3. Rapport détaillé
        result = pd.DataFrame({
            'Column'      : high_missing.index,
            'Missing_Pct' : high_missing.values.round(2),
            'Threshold'   : f'{threshold}%'
        })

        return result.sort_values('Missing_Pct', ascending=False).reset_index(drop=True)

    def constant_columns_analysis(
        self,
        df,
        unique_threshold: int = 1,
        variance_threshold: float = 0.01
    ) -> Dict:
        """
        Identifie les colonnes constantes ou à faible variance.
        
        Args:
            unique_threshold: Seuil pour considérer une colonne comme constante
            variance_threshold: Seuil de variance normalisée pour quasi-constantes
            
        Returns:
            Dictionnaire contenant les listes de colonnes et le DataFrame de synthèse
        """
        results = []

        for col in df.columns:
            v_counts = df[col].value_counts(dropna=False)
            n_unique = len(v_counts)
            n_missing = df[col].isnull().sum()
            dtype = df[col].dtype

            # Analyse de la variance pour types numériques
            variance = None
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].notna().sum() > 1:
                    mean_val = df[col].mean()
                    variance = (df[col].std() / abs(mean_val)
                               if mean_val != 0 else df[col].std())

            # Valeur dominante
            dom_val = v_counts.index[0] if n_unique > 0 else None
            dom_freq = v_counts.values[0] if n_unique > 0 else 0
            dom_pct = (dom_freq / len(df) * 100)

            # Classifications
            is_const = (n_unique <= unique_threshold)
            is_quasi = (not is_const and variance is not None
                       and variance < variance_threshold)

            results.append({
                'Column': col,
                'N_Unique': n_unique,
                'Missing_Pct': (n_missing / len(df) * 100),
                'Dtype': str(dtype),
                'Var_Norm': variance,
                'Dominant_Pct': dom_pct,
                'Is_Constant': is_const,
                'Is_Quasi': is_quasi
            })

        summary_df = pd.DataFrame(results)

        const_cols = summary_df[summary_df['Is_Constant']]['Column'].tolist()
        q_const = summary_df[summary_df['Is_Quasi']]['Column'].tolist()

        summary_df = summary_df.sort_values(
            ['Is_Constant', 'N_Unique'],
            ascending=[False, True]
        )

        return {
            'constant_cols': const_cols,
            'quasi_constant_cols': q_const,
            'summary_df': summary_df.reset_index(drop=True)
        }

    def generer_synthese_suppression(
        self,
        df,
        missing_threshold: float = 0.95
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Crée un rapport consolidé des colonnes candidates à l'élimination.
        
        Args:
            missing_threshold: Seuil de manquants au-delà duquel supprimer
            
        Returns:
            Tuple (liste des colonnes à supprimer, DataFrame de synthèse)
        """
        # Récupération des analyses
        missing_df = self.missing_summary()
        variance_res = self.constant_columns_analysis(df)

        # Colonnes avec trop de manquants
        cols_missing = missing_df[
            missing_df['Missing_Pct'] > (missing_threshold * 100)
        ]
        list_missing = cols_missing['Column'].tolist()

        # Colonnes constantes
        list_const = variance_res['constant_cols']

        # Construction du résumé
        synthese_data = []
        tous_candidats = list(set(list_missing + list_const))

        for col in tous_candidats:
            raison = []
            if col in list_missing:
                raison.append(f"Manquants > {missing_threshold*100:.0f}%")
            if col in list_const:
                raison.append("Constante (Unique)")

            synthese_data.append({
                'Column': col,
                'Raison_Principale': " & ".join(raison),
                'Dtype': str(df[col].dtype),
                'Impact_Potentiel': "Perte d'information nulle"
            })

        df_candidats = pd.DataFrame(synthese_data)

        if not df_candidats.empty:
            df_candidats = df_candidats.sort_values('Column')

        if self.verbose:
            print("\n" + "=" * 80)
            print("SYNTHÈSE DES CANDIDATS À LA SUPPRESSION")
            print("=" * 80)
            print(f" Colonnes analysées........: {len(df.columns)}")
            print(f" Colonnes à retirer........: {len(tous_candidats)}")
            print("-" * 80)

        return tous_candidats, df_candidats

    # =========================================================================
    # ÉTAPE 1:
    # =========================================================================
    def ____1_Nettoyage_Initial_et_Elimination(self): pass


    def supprimer_colonnes_constantes(self,
        df: pd.DataFrame,) -> pd.DataFrame:
        """
        Supprime les colonnes identifiées comme constantes.
        
        Returns:
            Self pour chaînage des méthodes
        """
        variance_res = self.constant_columns_analysis(df)
        cols_to_drop = variance_res['constant_cols']

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.columns_suppressed.extend(cols_to_drop)
            self.history.append({
                'operation': 'suppression_constantes',
                'colonnes': cols_to_drop,
                'nb_colonnes': len(cols_to_drop)
            })

            if self.verbose:
                print(f"\n✓ Suppression de {len(cols_to_drop)} colonnes constantes")
                print(f"  Colonnes: {', '.join(cols_to_drop)}")

        return df

    def supprimer_colonnes_par_motif(
        self,
        df: pd.DataFrame,
        motifs: Union[List[str], str],
        case_sensitive: bool = False,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Supprime les colonnes dont le nom contient l'un des motifs.
        
        Args:
            df: DataFrame à traiter
            motifs: Motif unique ou liste de motifs à chercher
            case_sensitive: Si True, sensible à la casse
            verbose: Si True, affiche les informations
            
        Returns:
            DataFrame sans les colonnes trouvées
        
        Exemple:
            df_clean = DataCleaner.supprimer_colonnes_par_motif(
                df, ['Electricity', 'Gas', 'Steam']
            )
        """
        # Convertir en liste si c'est un seul motif
        if isinstance(motifs, str):
            motifs = [motifs]

        # Trouver les colonnes qui contiennent l'un des motifs
        cols_to_drop = []
        for col in df.columns:
            col_check = col if case_sensitive else col.lower()
            for motif in motifs:
                motif_check = motif if case_sensitive else motif.lower()
                if motif_check in col_check:
                    cols_to_drop.append(col)
                    break  # Éviter les doublons

        if cols_to_drop:
            df_result = df.drop(columns=cols_to_drop)

            if verbose:
                print(f"\n✅ Suppression de {len(cols_to_drop)} colonnes contenant: {motifs}")
                print(f"   Colonnes trouvées: {', '.join(cols_to_drop)}")
                print(f"   Shape: {df.shape} → {df_result.shape}")

            return df_result
        else:
            if verbose:
                print(f"\n⚠️  Aucune colonne trouvée avec les motifs: {motifs}")

            return df.copy()


    def supprimer_colonnes_manquantes(
        self,
        threshold: float = 0.95
    ) -> 'DataCleaner':
        """
        Supprime les colonnes avec un taux de manquants excessif.
        
        Args:
            threshold: Seuil de manquants (0.95 = 95%)
            
        Returns:
            Self pour chaînage des méthodes
        """
        missing_df = self.missing_summary()
        cols_to_drop = missing_df[
            missing_df['Missing_Pct'] > (threshold * 100)
        ]['Column'].tolist()

        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.columns_suppressed.extend(cols_to_drop)
            self.history.append({
                'operation': 'suppression_manquants',
                'colonnes': cols_to_drop,
                'nb_colonnes': len(cols_to_drop),
                'seuil': threshold
            })

            if self.verbose:
                print(f"\n✓ Suppression de {len(cols_to_drop)} colonnes "
                      f"avec >{threshold*100:.0f}% de manquants")
                print(f"  Colonnes: {', '.join(cols_to_drop)}")

        return self


    def supprier_colonnes_REDONDANTS(self, df, meta_dict, cols_to_remove, reason="Redondance"):
        """
        Désactive les colonnes redondantes dans le dictionnaire de métadonnées fourni
        et les ajoute à la liste de suppression physique de l'instance.
        """
        # Aseguramos que cols_to_remove sea una lista
        if isinstance(cols_to_remove, str):
            cols_to_remove = [cols_to_remove]

        print(f"\n🗑️ Nettoyage de la redondance : {len(cols_to_remove)} colonnes à traiter.")

        for col in cols_to_remove:
            if col in meta_dict:
                # --- VALIDACIÓN DE SEGURIDAD ---
                # Si ya está marcada, saltamos para evitar duplicar notas o procesos
                if meta_dict[col].get('type_encodage') == '❌ Suppression':
                    print(f"  ℹ️ Info : '{col}' est déjà marqué pour suppression. Passage...")
                    continue

                # 1. ACTUALIZACIÓN DE METADATOS
                meta_dict[col]['a_supprimer']        = False  # Marcamos como positivo para borrar
                meta_dict[col]['type_encodage']      = '❌ Suppression'
                meta_dict[col]['raison_suppression'] = reason # Usamos la variable, no el string 'reason'

                # Concatenamos la razón a la nota de forma limpia
                #current_note = meta_dict[col].get('note', '')
                #meta_dict[col]['note'] = f"{reason} | {current_note}".strip(" | ")

                # 2. SINCRONIZACIÓN CON LA LISTA FÍSICA
                if col not in self.FEATURES_TO_REMOVE:
                    self.FEATURES_TO_REMOVE.append(col)

                # --- 3. SUPPRESSION PHYSIQUE (DataFrame) ---
                if col in df.columns:
                    df = df.drop(columns=[col])
                    print(f"  • {col:<20} | Statut : ❌ Supprimé du DataFrame")

                # print(f"  • {col:<20} | Statut : ❌ Marqué pour suppression")
            else:
                print(f"  ⚠️ Attention : '{col}' introuvable dans le dictionnaire.")

        return df, meta_dict

    def supprimer_colonnes_specifiques(self, df: pd.DataFrame, colonnes: list) -> pd.DataFrame:
        """
        Supprime des colonnes ciblées et archive l'opération dans l'historique global.
        """
        if not colonnes:
            return df

        # Identification des cibles réellement présentes dans le flux actuel
        cols_presentes    = [c for c in colonnes if c in df.columns]

        if cols_presentes:
            # 1. Exécution de la suppression physique
            df            = df.drop(columns=cols_presentes, errors='ignore')

            # 2. Archivage pour le rapport de synthèse final (Liste plate)
            self.columns_suppressed.extend(cols_presentes)

            # 3. AJOUT DE LA TRACE DANS L'HISTORIQUE (Point manquant corrigé)
            # Permet de reconstruire la généalogie du nettoyage étape par étape.
            self.history.append({
                'operation'   : 'suppression_specifique',       # Type d'action
                'colonnes'    : cols_presentes,                 # Noms des variables évacuées
                'nb_colonnes' : len(cols_presentes)             # Volume de données supprimées
            })

            # 4. Feedback utilisateur (si verbose=True)
            if self.verbose:
                print(f"\n✓ ACTION : Suppression de {len(cols_presentes)} colonnes spécifiques.")
                print(f"  DÉTAIL : {', '.join(cols_presentes)}")
        else:
            if self.verbose:
                print("\nℹ️ INFO : Aucune des colonnes cibles n'était présente pour suppression.")

        return df


    def nettoyage_initial_complet(
        self,
        missing_threshold: float = 0.95,
        colonnes_specifiques: Optional[List[str]] = None
    ) -> 'DataCleaner':
        """
        Exécute la séquence complète de nettoyage initial (Étape 1).
        
        Args:
            missing_threshold: Seuil de manquants
            colonnes_specifiques: Colonnes supplémentaires à retirer
            
        Returns:
            Self pour chaînage des méthodes
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("ÉTAPE 1: NETTOYAGE INITIAL ET ÉLIMINATION")
            print("=" * 80)
            print(f"Shape initiale: {self.df.shape}")

        # 1.1 Suppression des constantes
        self.supprimer_colonnes_constantes()

        # 1.2 Élimination des manquants excessifs
        self.supprimer_colonnes_manquantes(threshold=missing_threshold)

        # Suppression des colonnes spécifiques si fournies
        if colonnes_specifiques:
            self.supprimer_colonnes_specifiques(colonnes_specifiques)

        if self.verbose:
            print(f"\nShape finale: {self.df.shape}")
            print(f"Total colonnes supprimées: {len(self.columns_suppressed)}")
            print("=" * 80)

        return self

    # =========================================================================
    # ÉTAPE 2: GESTION DES VALEURS MANQUANTES
    # =========================================================================
    def ____2_Gestion_des_Valeurs_Manquantes(self): pass

    def _identifier_colonnes_par_type(self) -> Dict[str, List[str]]:
        """
        Identifie et sépare les colonnes numériques et catégorielles.
        
        Returns:
            Dict avec 'numeriques' et 'categoriques'
        """
        numeriques   = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categoriques = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        return {
            'numeriques': numeriques,
            'categoriques': categoriques
        }

    def _identifier_colonnes_par_types(self) -> Dict[str, List[str]]:
        """
        Identifie, sépare et TRANSFORME les colonnes par type de données.
        Les binaires (Oui/Non, etc.) sont converties en 0/1.
        """
        # 1. Types de base pandas
        numeriques    = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categoriques  = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()

        # 2. Identifier les booléens (bool natif pandas)
        booleens = self.df.select_dtypes(include=['bool']).columns.tolist()

        # 3. Mappages professionnels pour la transformation réelle
        PRO_MAPPINGS = {
            frozenset({'Oui', 'Non'}):       {'Oui': 1, 'Non': 0},
            frozenset({'Yes', 'No'}):        {'Yes': 1, 'No': 0},
            frozenset({'Y', 'N'}):           {'Y': 1, 'N': 0},
            frozenset({'True', 'False'}):    {'True': 1, 'False': 0},
            frozenset({'Homme', 'Femme'}):   {'Homme': 1, 'Femme': 0},
            frozenset({'Male', 'Female'}):   {'Male': 1, 'Female': 0}
        }

        # 4. Identifier et TRANSFORMER les colonnes binaires
        binaires = []
        for col in self.df.columns:
            n_unique = self.df[col].nunique()
            if n_unique == 2 and col not in booleens:
                valeurs_set = set(self.df[col].dropna().unique())

                # A. Vérifier si c'est un pattern connu pour transformation
                mapping_trouve = None
                for pattern_set, mapping in PRO_MAPPINGS.items():
                    if valeurs_set == pattern_set:
                        mapping_trouve = mapping
                        break

                # B. Si pattern trouvé OU si c'est une binaire déjà numérique (0/1)
                if mapping_trouve or (n_unique == 2 and col in numeriques):
                    binaires.append(col)

                    # --- TRANSFORMACIÓN REAL (ACTIVA) ---
                    if mapping_trouve:
                        print(f"🔄 Transformation binaire : {col} ({valeurs_set} -> 0/1)")
                        self.df[col] = self.df[col].map(mapping_trouve).astype(int)

                    # Nettoyage des listes pour l'exclusivité
                    if col in categoriques: categoriques.remove(col)
                    if col in numeriques:   numeriques.remove(col)

        # 5. Identifier les numériques discrètes (restantes après extraction binaires)
        numeriques_discretes = []
        numeriques_continues = []
        for col in numeriques:
            if col not in binaires and col not in booleens:
                n_unique = self.df[col].nunique()
                if str(self.df[col].dtype).startswith('int') and n_unique < 20:
                    numeriques_discretes.append(col)
                else:
                    numeriques_continues.append(col)

        # 6. Identifier les IDs
        ids = [col for col in self.df.columns if any(p in col.lower() for p in ['id', 'code'])
               and self.df[col].nunique() / len(self.df) > 0.95]




        return {
            'numeriques': numeriques,
            'numeriques_continues': numeriques_continues,
            'numeriques_discretes': numeriques_discretes,
            'categoriques': categoriques,
            'booleens': booleens,
            'binaires': binaires,
            'datetime': datetime_cols,
            'identifiants': ids
        }

    def show_dictionary_classified_BAK(self, meta_dict):
        """Affiche un résumé structuré du dictionnaire de métadonnées."""
        # 1. Variables a suprimir (Filtramos las que tienen el flag True)
        cols_a_borrar = [col for col, info in meta_dict.items() if info.get('a_supprimer')]

        print(f"\n--- 🗑️ COLUMNAS A SUPRIMIR ({len(cols_a_borrar)}) ---")
        for col in cols_a_borrar:
            razon = meta_dict[col].get('raison_suppression', 'No especificada')
            print(f"• {col:<45} | Raison: {razon}")

        # 2. Variables Activas por Clasificación Técnica
        # El orden es importante para la estrategia de modelado posterior
        categorias_orden = [
            'booleens', 'binaires', 'datetime',
            'numeriques_continues', 'numeriques_discretes', 'categoriques'
        ]

        for cat in categorias_orden:
            activas = [col for col, info in meta_dict.items()
                       if info.get('classification') == cat and not info.get('a_supprimer')]

            if activas:
                print(f"\n--- 📦 CATEGORÍA TÉCNICA: {cat.upper()} ({len(activas)}) ---")
                for col in activas:
                    desc = meta_dict[col].get('description', 'Sin descripción')
                    # Mostramos también el tipo de encodage si existe
                    enc = meta_dict[col].get('type_encodage', 'None')
                    print(f"• {col:<45} | {desc[:50]:<50} | Enc: {enc}")

    def show_dictionary_classified(self, meta_dict):
        """Affiche un résumé structuré, avec modalités, encodage et notes techniques."""

        # 1. Variables a suprimir
        cols_a_borrar = [col for col, info in meta_dict.items() if info.get('a_supprimer')]

        print(f"\n--- 🗑️ COLUMNAS A SUPRIMIR ({len(cols_a_borrar)}) ---")

        header = f"  {'COLONNE':<45} | {'ACTION/ENC':<20} | {'NOTE / CONTENU'}"
        print(header)
        print("  " + "-" * len(header))

        for col in cols_a_borrar:
            razon = meta_dict[col].get('raison_suppression', 'No especificada')
            note  = meta_dict[col].get('note', '')
            action = "❌ Suppresion"
            line = f"• {col:<45} | {str(action):<20} | {razon}"
            if note: line += f" | 📝 {note}"
            print(line)

        # 2. Variables Activas por Clasificación Técnica
        categorias_orden = [
            'booleens', 'binaires', 'datetime',
            'numeriques_continues', 'numeriques_discretes', 'categoriques'
        ]

        for cat in categorias_orden:
            activas = [col for col, info in meta_dict.items()
                       if info.get('classification') == cat and not info.get('a_supprimer')]

            if activas:
                print(f"\n--- 📦 CATEGORÍA TÉCNICA: {cat.upper()} ({len(activas)}) ---")
                header = f"  {'COLONNE':<45} | {'ACTIONENC':<20} | {'NOTE / CONTENU'}"
                print(header)
                print("  " + "-" * len(header))

                for col in activas:
                    info = meta_dict[col]
                    enc = info.get('type_encodage', 'None')
                    note = info.get('note', '')
                    razon = meta_dict[col].get('raison_suppression', None)

                    # Prioridad: Modalidades > Descripción
                    modalites = info.get('valeurs_possibles', [])
                    if modalites and isinstance(modalites, list):
                        content = ", ".join([str(m) for m in modalites])
                    else:
                        content = info.get('description', 'Sin descripción')


                    # --- LÓGICA DE TRUNCADO ---
                    if len(content) > 35:
                        content = content[:32] + "..."
                    razon_txt = ""
                    if razon and len(razon) > 0:
                        razon_txt = "(" + razon + ")"
                    # Formatear la línea final incluyendo la nota si existe
                    line = f"• {col:<45} | {str(enc):<20} | {razon_txt} {content} "
                    if note:
                        line += f" | 📝 {note}"

                    print(line)

    def enriquecer_y_clasificar_meta(self, df, meta_dict, types_cols):
        """
        1. Modifica meta_dict_enrichi agregando la clave 'classification' técnica.
        2. Imprime un listado organizado por el destino de cada variable.
        # description
        # type             objet, "int64"
        # classification  'numeriques', 'categoriques', 'booleens', 'datetime', 'binaires'
        # categorie        User SET:  "Éducation", "Management", "Démographique",
        # unite            None, "nombre", "€", "km", "heures/semaine"
        # a_supprimer
        
        # raison_suppression
        
        # encodage_necessaire
        # type_encodage              LabelEncoding, OneHotEncoding, OrdinalEncoding,...
        # valeurs_possibles  User SET: ["Ventes", "R&D", "RH", "Marketing", "Finance"] , ["Homme", "Femme"]
        # note
        """
        # Mapeo de la clasificación técnica al diccionario
        # types_cols es un dict tipo: {'numeriques': [...], 'categoriques': [...], etc.}
        for categoria_tecnica, columnas in types_cols.items():
            for col in columnas:
                if col in meta_dict:
                    meta_dict[col]['classification'] = categoria_tecnica

        # Llamamos al reporte visual
        self.show_dictionary_classified(meta_dict)

    def creer_indicateurs_missing(
        self,
        threshold: float = 0.50,
        suffix: str = '_Manquant'
    ) -> 'DataCleaner':
        """
        ÉTAPE 2.1: Création d'Indicateurs de Valeurs Manquantes.
        
        Crée des features binaires [Col]_Manquant pour colonnes >threshold% missing.
        TRAITEMENT SÉPARÉ par type de variable (numérique vs catégorielle).
        
        JUSTIFICATION:
        - L'absence peut être informative (ex: pas d'équipement, pas de certification)
        - Préserve l'information AVANT imputation
        - Améliore les performances prédictives du modèle
        
        Args:
            threshold: Seuil de missing au-delà duquel créer l'indicateur (0.50 = 50%)
            suffix: Suffixe pour les nouvelles colonnes
            
        Returns:
            Self pour chaînage des méthodes
        """
        # Calcul du taux de missing par colonne
        missing_ratio = self.df.isnull().sum() / len(self.df)
        candidats = missing_ratio[missing_ratio > threshold]

        if candidats.empty:
            if self.verbose:
                print(f"\n⚠️  Aucune colonne avec >{threshold*100:.0f}% de valeurs manquantes")
            return self

        # Séparation par type
        types_cols = self._identifier_colonnes_par_type()

        # Filtrage des candidats par type
        candidats_num = [c for c in candidats.index if c in types_cols['numeriques']]
        candidats_cat = [c for c in candidats.index if c in types_cols['categoriques']]

        indicateurs_crees = []

        # =====================================================================
        # CRÉATION DES INDICATEURS - NUMÉRIQUES
        # =====================================================================
        for col in candidats_num:
            nom_indicateur = f"{col}{suffix}"
            self.df[nom_indicateur] = self.df[col].isnull().astype(int)
            indicateurs_crees.append({
                'colonne_origine': col,
                'indicateur': nom_indicateur,
                'type': 'numérique',
                'missing_pct': missing_ratio[col] * 100
            })

        # =====================================================================
        # CRÉATION DES INDICATEURS - CATÉGORIELLES
        # =====================================================================
        for col in candidats_cat:
            nom_indicateur = f"{col}{suffix}"
            self.df[nom_indicateur] = self.df[col].isnull().astype(int)
            indicateurs_crees.append({
                'colonne_origine': col,
                'indicateur': nom_indicateur,
                'type': 'catégorielle',
                'missing_pct': missing_ratio[col] * 100
            })

        # Enregistrement dans l'historique
        if indicateurs_crees:
            self.history.append({
                'operation': 'creation_indicateurs_missing',
                'threshold': threshold,
                'nb_indicateurs_num': len(candidats_num),
                'nb_indicateurs_cat': len(candidats_cat),
                'indicateurs_numeriques': candidats_num,
                'indicateurs_categoriques': candidats_cat,
                'details': indicateurs_crees
            })

            if self.verbose:
                print("\n" + "=" * 80)
                print("ÉTAPE 2.1 : CRÉATION D'INDICATEURS DE VALEURS MANQUANTES")
                print("=" * 80)
                print(f"Seuil appliqué............: >{threshold*100:.0f}%")
                print(f"Indicateurs numériques....: {len(candidats_num)}")
                if candidats_num:
                    for col in candidats_num:
                        print(f"  • {col} ({missing_ratio[col]*100:.2f}%) → {col}{suffix}")

                print(f"Indicateurs catégoriels...: {len(candidats_cat)}")
                if candidats_cat:
                    for col in candidats_cat:
                        print(f"  • {col} ({missing_ratio[col]*100:.2f}%) → {col}{suffix}")

                print(f"Total créés...............: {len(indicateurs_crees)}")
                print("=" * 80)

        return self


    def imputer_categoriques(
        self,
        valeur_defaut: str = 'INCONNU',
        colonnes_specifiques: Optional[Dict[str, str]] = None
    ) -> 'DataCleaner':
        """
        ÉTAPE 2.2: Imputation Catégorielle.
        
        Remplace les valeurs manquantes dans les colonnes catégorielles.
        
        STRATÉGIE:
        - Valeur par défaut: 'INCONNU' ou 'AUCUN'
        - Permet de spécifier des valeurs personnalisées par colonne
        
        Args:
            valeur_defaut: Valeur à utiliser par défaut pour toutes les catégorielles
            colonnes_specifiques: Dict {nom_colonne: valeur_specifique} pour exceptions
            
        Returns:
            Self pour chaînage des méthodes
            
        Example:
            cleaner.imputer_categoriques(
                valeur_defaut='INCONNU',
                colonnes_specifiques={
                    'PropertyUseType': 'AUCUN',
                    'Neighborhood': 'NON_SPECIFIE'
                }
            )
        """
        # Identification des colonnes catégorielles avec valeurs manquantes
        types_cols = self._identifier_colonnes_par_type()
        cat_cols = types_cols['categoriques']

        # Filtrer uniquement celles qui ont des valeurs manquantes
        cat_with_missing = [
            col for col in cat_cols
            if self.df[col].isnull().sum() > 0
        ]

        if not cat_with_missing:
            if self.verbose:
                print("\n⚠️  Aucune colonne catégorielle avec valeurs manquantes")
            return self

        imputations = []

        for col in cat_with_missing:
            # Déterminer la valeur d'imputation
            if colonnes_specifiques and col in colonnes_specifiques:
                valeur_imputation = colonnes_specifiques[col]
            else:
                valeur_imputation = valeur_defaut

            # Compter les valeurs manquantes avant imputation
            nb_missing = self.df[col].isnull().sum()
            pct_missing = (nb_missing / len(self.df)) * 100

            # Imputation
            self.df[col] = self.df[col].fillna(valeur_imputation)

            imputations.append({
                'colonne': col,
                'valeur_imputation': valeur_imputation,
                'nb_values_imputees': nb_missing,
                'pct_impute': pct_missing
            })

        # Enregistrement dans l'historique
        self.history.append({
            'operation': 'imputation_categoriques',
            'valeur_defaut': valeur_defaut,
            'nb_colonnes_imputees': len(cat_with_missing),
            'colonnes': cat_with_missing,
            'details': imputations
        })

        if self.verbose:
            print("\n" + "=" * 80)
            print("ÉTAPE 2.2 : IMPUTATION CATÉGORIELLE")
            print("=" * 80)
            print(f"Valeur par défaut.........: '{valeur_defaut}'")
            print(f"Colonnes traitées.........: {len(cat_with_missing)}")
            print()
            for imp in imputations:
                valeur_affichee = imp['valeur_imputation']
                if valeur_affichee != valeur_defaut:
                    valeur_affichee = f"'{valeur_affichee}' (spécifique)"
                else:
                    valeur_affichee = f"'{valeur_affichee}'"

                print(f"  • {imp['colonne']:<40} → {valeur_affichee}")
                print(f"    Valeurs imputées: {imp['nb_values_imputees']:>5} "
                      f"({imp['pct_impute']:>6.2f}%)")
            print("=" * 80)

        return self



    def imputer_numeriques(
        self,
        strategie: str = 'mediane',
        colonnes_specifiques: Optional[Dict[str, float]] = None
    ) -> 'DataCleaner':
        """
        ÉTAPE 2.3: Imputation Numérique avec la Médiane.
        
        Remplace les valeurs manquantes dans les colonnes numériques 
        en utilisant la médiane GLOBALE de chaque colonne.
        
        STRATÉGIE:
        - Calcul de la médiane sur TOUS les valeurs non-NaN de la colonne
        - Imputation de TOUS les NaN avec cette médiane unique
        - Robuste aux outliers (contrairement à la moyenne)
        - Les indicateurs créés en 2.1 sont PRÉSERVÉS
        
        Args:
            strategie: Type d'imputation ('mediane' uniquement pour l'instant)
            colonnes_specifiques: Dict {nom_colonne: valeur} pour imputation manuelle
            
        Returns:
            Self pour chaînage des méthodes
        """
        if strategie != 'mediane':
            raise ValueError("Seule la stratégie 'mediane' est implémentée")

        # Identification des colonnes numériques avec valeurs manquantes
        types_cols = self._identifier_colonnes_par_type()
        num_cols = types_cols['numeriques']

        num_with_missing = [
            col for col in num_cols
            if self.df[col].isnull().sum() > 0
        ]

        if not num_with_missing:
            if self.verbose:
                print("\n⚠️  Aucune colonne numérique avec valeurs manquantes")
            return self

        imputations = []

        for col in num_with_missing:
            nb_missing = self.df[col].isnull().sum()
            pct_missing = (nb_missing / len(self.df)) * 100
            nb_existants = self.df[col].notna().sum()

            # Déterminer la valeur d'imputation
            if colonnes_specifiques and col in colonnes_specifiques:
                valeur_imputation = colonnes_specifiques[col]
                source = 'manuelle'
            else:
                valeur_imputation = self.df[col].median()
                source = 'mediane'

            # Statistiques AVANT imputation
            stats_avant = {
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'mean': self.df[col].mean(),
                'std': self.df[col].std()
            }

            # IMPUTATION
            self.df[col] = self.df[col].fillna(valeur_imputation)

            # Vérification
            nb_missing_apres = self.df[col].isnull().sum()

            imputations.append({
                'colonne': col,
                'strategie': source,
                'valeur_imputation': round(valeur_imputation, 4),
                'nb_values_imputees': nb_missing,
                'pct_impute': pct_missing,
                'nb_values_existants': nb_existants,
                'stats_avant': stats_avant,
                'verification': nb_missing_apres == 0
            })

        # Historique
        self.history.append({
            'operation': 'imputation_numeriques',
            'strategie': strategie,
            'nb_colonnes_imputees': len(num_with_missing),
            'colonnes': num_with_missing,
            'details': imputations
        })

        if self.verbose:
            print("\n" + "=" * 80)
            print("ÉTAPE 2.3 : IMPUTATION NUMÉRIQUE")
            print("=" * 80)
            print(f"Stratégie.................: {strategie.upper()}")
            print(f"Colonnes traitées.........: {len(num_with_missing)}")
            print()

            for imp in imputations:
                source_label = "médiane" if imp['strategie'] == 'mediane' else "manuelle"

                print(f"  • {imp['colonne']:<45}")
                print(f"    Valeur d'imputation: {imp['valeur_imputation']:>12.4f} ({source_label})")
                print(f"    Valeurs imputées...: {imp['nb_values_imputees']:>5} "
                      f"({imp['pct_impute']:>6.2f}%) sur {imp['nb_values_existants']} existantes")
                print(f"    Vérification.......: {'✓ OK' if imp['verification'] else '✗ ERREUR'}")
                print()

            print("=" * 80)

        return self

    def gestion_valeurs_manquantes_complete(
        self,
        threshold_indicateurs: float = 0.50,
        valeur_cat_defaut: str = 'INCONNU',
        strategie_num: str = 'mediane',
        colonnes_cat_specifiques: Optional[Dict[str, str]] = None,
        colonnes_num_specifiques: Optional[Dict[str, float]] = None
    ) -> 'DataCleaner':
        """
        Exécute la séquence complète de gestion des valeurs manquantes (Étape 2).
        
        PIPELINE: 2.1 → 2.2 → 2.3
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("ÉTAPE 2: GESTION COMPLÈTE DES VALEURS MANQUANTES")
            print("=" * 80)

            total_missing = self.df.isnull().sum().sum()
            pct_missing = (total_missing / (len(self.df) * len(self.df.columns))) * 100
            print("État initial:")
            print(f"  • Valeurs manquantes: {total_missing} ({pct_missing:.2f}%)")

        self.creer_indicateurs_missing(threshold=threshold_indicateurs)
        self.imputer_categoriques(
            valeur_defaut=valeur_cat_defaut,
            colonnes_specifiques=colonnes_cat_specifiques
        )
        self.imputer_numeriques(
            strategie=strategie_num,
            colonnes_specifiques=colonnes_num_specifiques
        )

        if self.verbose:
            total_missing_final = self.df.isnull().sum().sum()
            print("\n" + "-" * 80)
            print("BILAN FINAL:")
            print(f"  • Valeurs manquantes: {total_missing_final}")
            print(f"  • Réduction: {total_missing - total_missing_final} valeurs")
            print("=" * 80)

        return self



    # =========================================================================
    # ÉTAPE 3: TRAITEMENT DE L'ASYMÉTRIE ET OUTLIERS
    # =========================================================================
    def ____Analyse_Univariee_Variables_Numeriques(self): pass

    def analyser_statistiques_globales_BAK(self, df, sort_by='Skewness'):
        """
        Automatise l'interprétation du describe() et identifie les points critiques.
        Ajout : Tri dynamique et protection des variables binaires.
        """
        # Sélection exclusive des variables numériques
        df_num = df.select_dtypes(include=[np.number])

        # --------------------------------------------------------------------------
        # 1. GÉNÉRATION DES STATISTIQUES DESCRIPTIVES ÉTENDUES
        # --------------------------------------------------------------------------
        desc            = df_num.describe().T
        desc['Range']   = desc['max'] - desc['min']
        desc['Skewness']= df_num.skew()
        desc['CV']      = desc['std'] / desc['mean'].abs().replace(0, np.nan)

        # --------------------------------------------------------------------------
        # 2. IDENTIFICATION DES ÉCHELLES ET ASYMÉTRIES
        # --------------------------------------------------------------------------
        moy_globale     = desc['mean'].abs().mean()

        # Identification des échelles extrêmes
        echelles_ext    = []
        for col in desc.index:
            ratio = desc.loc[col, 'mean'] / moy_globale if moy_globale != 0 else 1
            if ratio > 100 or ratio < 0.01:
                echelles_ext.append(col)

        # Identification de l'asymétrie critique
        # 💡 Note : On filtre pour ne pas inclure les binaires (0/1) qui ont souvent un skew élevé par nature
        cols_asym = []
        for col in desc.index:
            is_binary = df[col].nunique() == 2
            if abs(desc.loc[col, 'Skewness']) > 1 and not is_binary:
                cols_asym.append(col)

        # --------------------------------------------------------------------------
        # 3. SYSTÈME EXPERT : COLONNE D'OBSERVATIONS
        # --------------------------------------------------------------------------
        def _generer_recommandation(row):
            actions = []
            is_binary = df[row.name].nunique() == 2

            if is_binary:
                return "Binary (No Transform)"

            if row.name in cols_asym: actions.append("Log Transform")
            if row.name in echelles_ext: actions.append("Scaling")
            if row['CV'] > 2: actions.append("Check Outliers")

            return " | ".join(actions) if actions else "RAS (Standardize)"

        desc['Action_Recommandee'] = desc.apply(_generer_recommandation, axis=1)

        # --- NOUVEAU : Tri dynamique avant affichage y retorno ---
        desc = desc.sort_values(by=sort_by, ascending=False)

        # --------------------------------------------------------------------------
        # 4. AFFICHAGE DES ALERTES (LOGGING)
        # --------------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("ÉTAPE 3.1 : ANALYSE STATISTIQUE ET DIAGNOSTIC DES ÉCHELLES")
        print("=" * 80)
        print(f" Variables numériques analysées.......: {len(df_num.columns)}")
        print(f" Variables à échelles critiques.......: {len(echelles_ext)}")
        print(f" Variables fortement asymétriques.....: {len(cols_asym)}")

        if echelles_ext:
            print("\n⚠️ ALERTE ÉCHELLES :")
            for col in echelles_ext[:20]:
                print(f"   - {col:30} : Moyenne = {desc.loc[col, 'mean']:.2e}")

        if cols_asym:
            print("\n📊 ALERTE ASYMÉTRIE (Hors Binaires) :")
            for col in cols_asym[:20]:
                print(f"   - {col:30} : Skewness = {desc.loc[col, 'Skewness']:.2f}")

        print("-" * 80)

        # Retourne strictement les 3 objets pour maintenir la compatibilité
        return desc, echelles_ext, cols_asym

    def get_numeric_blueprint_BAK(self, df, meta_dict, mapping_expert):
        """
        Analyse, classifie et synchronise les stratégies de transformation numérique.
        Remplit les listes de contrôle de l'instance et met à jour le meta_dict par référence.
        """
        # 1. Récupération des listes issues du diagnostic expert
        cols_log    = mapping_expert.get('num_log', [])
        cols_robust = mapping_expert.get('num_robust', [])
        cols_std    = mapping_expert.get('num_std', [])

        # 2. Initialisation des listes de contrôle de l'instance
        self.COLS_TO_LOG    = cols_log
        self.COLS_TO_ROBUST = cols_robust
        self.COLS_STANDARD  = cols_std

        print("\n🔍 ANALYSE ET SYNCHRONISATION DU BLUEPRINT NUMÉRIQUE")
        print(f"{'Colonne':<28} | {'Stratégie':<22} | {'Métriques (Skew/CV)'}")
        print("-" * 85)

        # Structure de sortie (Blueprint)
        blueprint = {
            'log': cols_log,
            'robust': cols_robust,
            'standard': cols_std,
            'details': {}
        }

        # On parcourt toutes les colonnes numériques identifiées
        all_num_cols = cols_log + cols_robust + cols_std

        for col in all_num_cols:
            # Récupération des métriques pour le log/affichage
            skew_val = df[col].skew()
            cv_val   = (df[col].std() / df[col].mean()) if df[col].mean() != 0 else 0

            # --- Détermination de la stratégie pour le meta_dict ---
            if col in cols_log:
                strategy = "🚀 Log Transform"
                params = f"Skew: {skew_val:.2f}"
            elif col in cols_robust:
                strategy = "🛡️ Robust Scaling"
                params = f"CV: {cv_val:.2f}"
            else:
                strategy = "🔔 Standardize (RAS)"
                params = "Normal/Binaire"

            # --- Mise à jour du meta_dict (par référence) ---
            if col in meta_dict:
                meta_dict[col].update({
                    'transformation_type': strategy,
                    'is_skewed': col in cols_log,
                    'has_outliers': col in cols_robust,
                    'stats_point': {'skew': skew_val, 'cv': cv_val}
                })

            # Metadatos para el blueprint de retorno
            blueprint['details'][col] = {
                'strategy': strategy,
                'metrics': params
            }

            # --- Visualisation propre ---
            print(f"{col:<28} | {strategy:<22} | {params}")

        print("-" * 85)
        print(f"✅ Sync Numérique terminée. Log: {len(cols_log)} | Robust: {len(cols_robust)} | Std: {len(cols_std)}")

        return blueprint


    def get_numerical_encoding_blueprint(self, df, meta_dict, mapping_expert):
        """
        Analyse et synchronise les stratégies numériques avec sécurité anti-oubli.
        """
        # 1. Récupération des listes
        cols_log    = mapping_expert.get('num_log', [])
        cols_robust = mapping_expert.get('num_robust', [])
        cols_std    = mapping_expert.get('num_std', [])
        all_classified = set(cols_log + cols_robust + cols_std)

        # --- SÉCURITÉ : Détection des "Orphelines" ---
        # Buscamos qué columnas numéricas del DF no están en nuestras listas
        num_in_df = set(df.select_dtypes(include=[np.number]).columns)
        orphans = list(num_in_df - all_classified)

        if orphans:
            print(f"⚠️ ATTENTION : {len(orphans)} colonnes numériques non classées. Ajout à 'Standard'.")
            cols_std.extend(orphans) # Las rescatamos y mandamos a Standard
            all_classified.update(orphans)

        # 2. Synchronisation des listes d'instance
        self.COLS_TO_LOG    = cols_log
        self.COLS_TO_ROBUST = cols_robust
        self.COLS_STANDARD  = cols_std

        print("\n🔍 ANALYSE ET SYNCHRONISATION DU BLUEPRINT NUMÉRIQUE")
        print(f"{'Colonne':<40} | {'Stratégie':<22} | {'Métriques (Skew/CV)'}")
        print("-" * 85)

        # 3. Boucle de mise à jour (avec la logique de préservation que nous avons vue)
        for col in all_classified:
            skew_val = df[col].skew()
            cv_val   = (df[col].std() / df[col].mean()) if df[col].mean() != 0 else 0

            if col in cols_log:
                strategy = "🚀 Log Transform"
                params = f"Skew: {skew_val:.2f}"
            elif col in cols_robust:
                strategy = "🛡️ Robust Scaling"
                params = f"CV: {cv_val:.2f}"
            else:
                strategy = "🔔 Standardize (RAS)"
                params = "Normal/Binaire/Orphan"

            # --- Mise à jour du meta_dict (par référence) ---
            if col in meta_dict:
                # Aseguramos que sea un diccionario para evitar el AttributeError: 'str'

                # print(f"  Debug col : {col}")

                if isinstance(meta_dict[col], str):
                    meta_dict[col] = {'type_original': meta_dict[col]}

                # Actualización con la misma estructura que el Encoding Blueprint
                meta_dict[col].update({
                    'type_encodage': strategy,             # Ej: "🚀 Log Transform"
                    'encodage_necessaire': True,           # AUNQUE No es categórica, es numérica

                    'is_skewed': col in cols_log,
                    'has_outliers': col in cols_robust,
                    'metrics_detectées': {
                        'skewness': round(skew_val, 3),
                        'cv': round(cv_val, 3)
                    },
                    'transformation_active': True        # Para que el pipeline sepa que debe actuar
                })
            else:
                # Si la columna es nueva (ej. creada en Feature Engineering)
                meta_dict[col] = {
                    'transformation_type': strategy,
                    'metrics_detectées': {'skewness': skew_val, 'cv': cv_val},
                    'encodage_necessaire': False,
                    'note': 'Ajouté dynamiquement (Numérique)'
                }

            print(f"{col:<40} | {strategy:<22} | {params}")

        print("-" * 85)
        return {"log": cols_log, "robust": cols_robust, "standard": cols_std}



    def analyser_statistiques_globales(self, df, sort_by='Skewness', card_threshold=10):
        """
        Analyse complète : Statistiques, Échelles, Asymétrie, Nuls et Cardinalité.
        """
        # Sélection exclusive des variables numériques
        df_num = df.select_dtypes(include=[np.number])

        # --------------------------------------------------------------------------
        # 1. GÉNÉRATION DES STATISTIQUES ET NULOS
        # --------------------------------------------------------------------------
        desc = df_num.describe().T

        # Cálculo de Missing Values (como en tu ejemplo)
        desc['Missing_Count'] = df_num.isnull().sum()
        desc['Missing_Pct']   = (desc['Missing_Count'] / len(df)) * 100

        # Métricas avanzadas
        desc['Range']    = desc['max'] - desc['min']
        desc['Skewness'] = df_num.skew()
        desc['CV']       = desc['std'] / desc['mean'].abs().replace(0, np.nan)

        # Cardinalidad y Tipo Técnico
        desc['Uniques']  = df_num.nunique()
        desc['Nature']   = desc['Uniques'].apply(
            lambda x: 'Binaire' if x == 2 else ('Discrète/Ordinale' if x <= card_threshold else 'Continue')
        )

        # --------------------------------------------------------------------------
        # 2. IDENTIFICATION DES ÉCHELLES ET ASYMÉTRIES
        # --------------------------------------------------------------------------
        moy_globale = desc['mean'].abs().mean()



       # --- CLASSIFICATION POUR LE PIPELINE ---

        # A. LOG : Continues avec forte asymétrie
        cols_log = desc[
            (desc['Skewness'].abs() > 1) & (desc['Nature'] == 'Continue')
        ].index.tolist()

        # B. ROBUST : Non-log mais avec forte dispersion (CV > 2)
        cols_robust = desc[
            (~desc.index.isin(cols_log)) & (desc['CV'] > 2) & (desc['Nature'] != 'Binaire')
        ].index.tolist()

        # C. STANDARD (Le reste) : Binaires, Discrètes et Continues "saines"
        # On prend tout ce qui n'est pas dans LOG ou ROBUST
        cols_std = desc[
            (~desc.index.isin(cols_log)) & (~desc.index.isin(cols_robust))
        ].index.tolist()

        # para print de alertas
        echelles_ext = desc[
            (desc['mean'] / moy_globale > 100) | (desc['mean'] / moy_globale < 0.01)
        ].index.tolist()

         # Identification de l'asymétrie critique
        cols_asym       = desc[desc['Skewness'].abs() > 1].index.tolist()

        # --------------------------------------------------------------------------
        # 3. SYSTÈME EXPERT : COLONNE D'OBSERVATIONS
        # --------------------------------------------------------------------------

        #def _generer_recommandation(row):
        #    actions = []
        #    if row.name in cols_asym: actions.append("Log Transform")
        #    if row.name in echelles_ext: actions.append("Scaling")
        #    if row['CV'] > 2: actions.append("Check Outliers")
        #    return " | ".join(actions) if actions else "RAS (Standardize)"

        # 3. SYSTÈME EXPERT : RECOMMANDATIONS VISUELLES
        def _generer_recommandation(row):
            actions = []

            # PRIORITÉ 0 : Niveaux de vide (Le plus critique)
            if row['Missing_Pct'] > 50:
                return "⚠️ DROP (Too many nulls)"

            # PRIORITÉ 1 : Nature binaire (Pas de transformation de forme)
            if row['Nature'] == 'Binaire':
                actions.append("Binary (No Transform)")
            else:
                # PRIORITÉ 2 : Classification de transformation pour les variables continues/discrètes
                if row.name in cols_log:
                    actions.append("Log Transform")
                elif row.name in cols_robust:
                    actions.append("Robust Scaling")
                else:
                    actions.append("Standardize (RAS)")

            # NOTES COMPLÉMENTAIRES (Diagnostic d'échelle et d'outliers)
            if row.name in echelles_ext:
                actions.append("High Scale Alert")

            # On n'alerte sur les outliers que si on n'a pas déjà prévu un RobustScaler
            if row['CV'] > 2 and row.name not in cols_robust and row['Nature'] != 'Binaire':
                actions.append("Check Outliers")

            return " | ".join(actions)

        desc['Action_Recommandee'] = desc.apply(_generer_recommandation, axis=1)

        # Tri dynamique (podrías ordenar por 'Missing_Pct' para ver nulos primero)
        desc = desc.sort_values(by=sort_by, ascending=False)

        # --------------------------------------------------------------------------
        # 4. AFFICHAGE DES ALERTES (LOGGING)
        # --------------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("DIAGNOSTIC GLOBAL ET CLASSIFICATION DES COLONNES")
        print("=" * 80)
        print(f" Variables analysées : {len(df_num.columns):<5} | Nuls critiques (>10%): {len(desc[desc['Missing_Pct']>10])}")
        print(f" Variables Log (Asymétriques)..: {len(cols_log)}")
        print(f" Variables Robust (Outliers)..: {len(cols_robust)}")
        print(f" Variables Standard (RAS).....: {len(cols_std)}")
        print(f" Alertes......................: {len(echelles_ext)}")
        print("-" * 80)

        # Mostrar resumen de nulos si existen
        nulos = desc[desc['Missing_Count'] > 0][['Missing_Count', 'Missing_Pct']].sort_values('Missing_Pct', ascending=False)
        if not nulos.empty:
            print("\n📌 COLONNES AVEC VALEURS NULLES :")
            print(nulos.head(10).to_markdown())

        return desc, cols_log, cols_robust, cols_std, echelles_ext

    def plot_distributions_grille(self, df, columns, n_cols=3, figsize_unit=(5, 4)):
        """
        Crée une grille d'histogrammes avec courbe de densité (KDE).
        """
        if not columns:
            print("⚠️ Info..................: Aucune colonne à visualiser.")
            return None

        # Calcul du nombre de lignes nécessaires
        n_vars  = len(columns)
        n_rows  = math.ceil(n_vars / n_cols)

        # Configuration de la taille de la figure
        fig_size = (n_cols * figsize_unit[0], n_rows * figsize_unit[1])

        # --------------------------------------------------------------------------
        # CRÉATION DE LA FIGURE ET DES SUBPLOTS
        # --------------------------------------------------------------------------
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        axes = axes.flatten()                     # Aplatir pour itération facile

        for i, col in enumerate(columns):
            sns.histplot(
                df[col],
                kde=True,                         # Ajout de la courbe de densité
                ax=axes[i],
                color='steelblue',
                edgecolor='white'
            )

            # Esthétique et titres
            axes[i].set_title(f"Distribution : {col}", fontsize=12, fontweight='bold')
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Fréquence")

            # Calcul de la médiane pour comparaison visuelle avec la moyenne
            axes[i].axvline(df[col].median(), color='red', linestyle='--',
                            label=f"Médiane: {df[col].median():.2f}")
            axes[i].legend(fontsize=8)

        # Nettoyage des axes vides (si n_vars < n_rows * n_cols)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        return fig



    def interpreter_proprietes_vars(self, df):
        """
        Réalise un diagnostic textuel automatique des variables numériques.
        """
        df_num = df.select_dtypes(include=[np.number])
        diagnostics = []

        for col in df_num.columns:
            data = df_num[col].dropna()
            if data.empty: continue

            # 1. Calcul des métriques avancées
            skew     = data.skew()                 # Asymétrie
            kurt     = data.kurtosis()             # Épaisseur des queues

            # 2. Détection d'Outliers (Méthode IQR)
            q1, q3   = data.quantile(0.25), data.quantile(0.75)
            iqr      = q3 - q1
            outliers = data[(data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))]
            out_pct  = (len(outliers) / len(data)) * 100

            # ----------------------------------------------------------------------
            # 3. MOTEUR D'INTERPRÉTATION (LOGIQUE SÉMANTIQUE)
            # ----------------------------------------------------------------------

            # A. Analyse de l'Asymétrie
            if abs(skew) < 0.5:  msg_skew = "Symétrique (Normale)"
            elif skew > 0.5:      msg_skew = f"Asymétrie Positive ({skew:.1f})"
            else:                 msg_skew = f"Asymétrie Négative ({skew:.1f})"

            # B. Analyse des Queues (Kurtosis / Tail)
            if kurt > 1:         msg_tail = "Queues lourdes (Outliers probables)"
            elif kurt < -1:       msg_tail = "Distribution plate (Uniforme)"
            else:                 msg_tail = "Queues normales"

            # C. Diagnostic Outliers
            if out_pct > 5:      msg_out  = f"Critique ({out_pct:.1f}%)"
            elif out_pct > 0:    msg_out  = f"Modérée ({out_pct:.1f}%)"
            else:                msg_out  = "Aucun"

            diagnostics.append({
                'Variable'      : col,
                'Asymétrie'     : msg_skew,
                'Type_Queues'   : msg_tail,
                'Outliers_IQR'  : msg_out,
                'Action_Data'   : "Log + Robust Scaling" if out_pct > 5 else "Standardize"
            })

        # --------------------------------------------------------------------------
        # 4. PRÉSENTATION DES RÉSULTATS
        # --------------------------------------------------------------------------
        df_diag = pd.DataFrame(diagnostics)

        print("\n" + "=" * 80)
        print("ÉTAPE 3.5 : INTERPRÉTATION AUTOMATIQUE DU DATASET")
        print("=" * 80)

        return df_diag

    # =========================================================================
    # ÉTAPE 3: TRAITEMENT DE L'ASYMÉTRIE ET OUTLIERS
    # =========================================================================
    def ____3_Traitement_de_Asymetrie_et_Outliers(self): pass

    def transformation_logarithmique(
        self,
        colonnes: Optional[List[str]] = None,
        auto_detect: bool = True,
        skew_threshold: float = 1.0,
        suffix: str = '_log'
    ) -> 'DataCleaner':
        """
        ÉTAPE 3.1: Transformation Logarithmique.
        
        Applique log(x + 1) aux variables asymétriques pour:
        - Réduire l'asymétrie (skewness)
        - Normaliser les distributions
        - Stabiliser la variance
        - Gérer les outliers
        
        IMPORTANT: Ajoute 1 avant log pour gérer les valeurs nulles.
        
        Args:
            colonnes: Liste de colonnes à transformer (si None, auto-détection)
            auto_detect: Détecter automatiquement les colonnes asymétriques
            skew_threshold: Seuil de skewness pour auto-détection (1.0 par défaut)
            suffix: Suffixe pour les nouvelles colonnes
            
        Returns:
            Self pour chaînage des méthodes
        """
        # Identification des colonnes numériques
        types_cols = self._identifier_colonnes_par_type()
        num_cols   = types_cols['numeriques']

        # Exclure les colonnes indicateurs
        num_cols = [c for c in num_cols if '_Manquant' not in c]

        if not num_cols:
            if self.verbose:
                print("\n⚠️  Aucune colonne numérique disponible")
            return self

        # Déterminer les colonnes à transformer
        if auto_detect and colonnes is None:
            # Auto-détection basée sur le skewness
            cols_a_transformer = []
            skewness_info      = []

            for col in num_cols:
                skew_val = self.df[col].skew()

                if abs(skew_val) > skew_threshold:
                    cols_a_transformer.append(col)
                    skewness_info.append({
                        'colonne'  : col,
                        'skewness' : skew_val
                    })

        elif colonnes is not None:
            # Utiliser les colonnes spécifiées
            cols_a_transformer = [c for c in colonnes if c in num_cols]
            skewness_info      = [
                {'colonne': c, 'skewness': self.df[c].skew()}
                for c in cols_a_transformer
            ]

        else:
            if self.verbose:
                print("\n⚠️  Aucune colonne à transformer")
            return self

        if not cols_a_transformer:
            if self.verbose:
                print(f"\n⚠️  Aucune colonne avec |skewness| > {skew_threshold}")
            return self

        # Application de la transformation
        transformations = []

        for col in cols_a_transformer:
            # Statistiques AVANT transformation
            skew_avant = self.df[col].skew()
            min_avant  = self.df[col].min()
            max_avant  = self.df[col].max()

            # Vérifier si la colonne contient des valeurs négatives
            has_negative = (self.df[col] < 0).any()

            # Manejo de Negativos: detecta si hay valores negativos y aplica
            # un "shift" (desplazamiento, decalage) antes del logaritmo.
            # Esto es vital para evitar errores fatales en el pipeline.

            if has_negative:
                # Décalage pour rendre toutes les valeurs positives
                decalage            = abs(self.df[col].min()) + 1
                nom_col_transformed = f"{col}{suffix}"

                self.df[nom_col_transformed] = np.log1p(self.df[col] + decalage)

                transformations.append({
                    'colonne'       : col,
                    'colonne_log'   : nom_col_transformed,
                    'skew_avant'    : skew_avant,
                    'skew_apres'    : self.df[nom_col_transformed].skew(),
                    'decalage'      : decalage,
                    'has_negative'  : True,
                    'min_avant'     : min_avant,
                    'max_avant'     : max_avant
                })
            else:
                # Transformation directe avec log1p
                nom_col_transformed = f"{col}{suffix}"

                self.df[nom_col_transformed] = np.log1p(self.df[col])

                transformations.append({
                    'colonne'       : col,
                    'colonne_log'   : nom_col_transformed,
                    'skew_avant'    : skew_avant,
                    'skew_apres'    : self.df[nom_col_transformed].skew(),
                    'decalage'      : 0,
                    'has_negative'  : False,
                    'min_avant'     : min_avant,
                    'max_avant'     : max_avant
                })

        # Enregistrement dans l'historique
        self.history.append({
            'operation'            : 'transformation_logarithmique',
            'auto_detect'          : auto_detect,
            'skew_threshold'       : skew_threshold,
            'nb_transformations'   : len(transformations),
            'colonnes_originales'  : cols_a_transformer,
            'details'              : transformations
        })

        if self.verbose:
            print("\n============================================================================")
            print("ÉTAPE 3.1 : TRANSFORMATION LOGARITHMIQUE")
            print("============================================================================")
            print(f"Méthode..................: {'Auto-détection' if auto_detect else 'Manuel'}")
            if auto_detect:
                print(f"Seuil de skewness........: |skew| > {skew_threshold}")
            print(f"Colonnes transformées....: {len(transformations)}")
            print()

            # Affichage détaillé
            for trans in transformations:
                print(f"  • {trans['colonne']:<40} → {trans['colonne_log']}")
                print(f"    Skewness avant.......: {trans['skew_avant']:>8.3f}")
                print(f"    Skewness après.......: {trans['skew_apres']:>8.3f}")
                print(f"    Amélioration.........: {abs(trans['skew_avant']) - abs(trans['skew_apres']):>8.3f}")

                if trans['has_negative']:
                    print(f"    ⚠️ Valeurs négatives → Décalage: +{trans['decalage']:.2f}")

                print()

            print("============================================================================")

        return self

    def winsorisation(
        self,
        colonnes: Optional[List[str]] = None,
        percentile_bas: float = 0.01,
        percentile_haut: float = 0.99,
        inplace: bool = False,
        suffix: str = '_wins'
    ) -> 'DataCleaner':
        """
        ÉTAPE 3.2: Winsorisation/Écrêtage des Outliers.
        
        Cappe les valeurs extrêmes aux percentiles spécifiés pour:
        - Réduire l'impact des outliers
        - Préserver la structure des données (vs suppression)
        - Améliorer la robustesse des modèles
        
        Args:
            colonnes: Liste de colonnes à traiter (si None, toutes les numériques)
            percentile_bas: Percentile inférieur (0.01 = 1%)
            percentile_haut: Percentile supérieur (0.99 = 99%)
            inplace: Modifier les colonnes originales (True) ou créer nouvelles (False)
            suffix: Suffixe pour les nouvelles colonnes (si inplace=False)
            
        Returns:
            Self pour chaînage des méthodes
        """
        # Identification des colonnes numériques
        types_cols = self._identifier_colonnes_par_type()
        num_cols   = types_cols['numeriques']

        # Exclure les indicateurs et colonnes log
        num_cols = [
            c for c in num_cols
            if '_Manquant' not in c and '_log' not in c
        ]

        # Déterminer les colonnes à traiter
        if colonnes is not None:
            cols_a_traiter = [c for c in colonnes if c in num_cols]
        else:
            cols_a_traiter = num_cols

        if not cols_a_traiter:
            if self.verbose:
                print("\n⚠️  Aucune colonne à winsoriser")
            return self

        # Application de la winsorisation
        winsorisations = []

        for col in cols_a_traiter:
            # Calcul des bornes
            q_bas  = self.df[col].quantile(percentile_bas)
            q_haut = self.df[col].quantile(percentile_haut)

            # Comptage des valeurs cappées
            nb_bas   = (self.df[col] < q_bas).sum()
            nb_haut  = (self.df[col] > q_haut).sum()
            nb_total = nb_bas + nb_haut
            pct_cappe = (nb_total / len(self.df)) * 100

            # Statistiques AVANT
            min_avant  = self.df[col].min()
            max_avant  = self.df[col].max()
            mean_avant = self.df[col].mean()
            std_avant  = self.df[col].std()

            # Application du capping
            if inplace:
                # Modification directe
                self.df[col] = self.df[col].clip(lower=q_bas, upper=q_haut)
                col_finale   = col
            else:
                # Création d'une nouvelle colonne
                col_finale            = f"{col}{suffix}"
                self.df[col_finale]   = self.df[col].clip(lower=q_bas, upper=q_haut)

            # Statistiques APRÈS
            min_apres  = self.df[col_finale].min()
            max_apres  = self.df[col_finale].max()
            mean_apres = self.df[col_finale].mean()
            std_apres  = self.df[col_finale].std()

            winsorisations.append({
                'colonne'        : col,
                'colonne_finale' : col_finale,
                'borne_inf'      : q_bas,
                'borne_sup'      : q_haut,
                'nb_cappe_bas'   : nb_bas,
                'nb_cappe_haut'  : nb_haut,
                'nb_total_cappe' : nb_total,
                'pct_cappe'      : pct_cappe,
                'stats_avant'    : {
                    'min'  : min_avant,
                    'max'  : max_avant,
                    'mean' : mean_avant,
                    'std'  : std_avant
                },
                'stats_apres'    : {
                    'min'  : min_apres,
                    'max'  : max_apres,
                    'mean' : mean_apres,
                    'std'  : std_apres
                }
            })

        # Enregistrement dans l'historique
        self.history.append({
            'operation'           : 'winsorisation',
            'percentile_bas'      : percentile_bas,
            'percentile_haut'     : percentile_haut,
            'inplace'             : inplace,
            'nb_colonnes_traitees': len(winsorisations),
            'colonnes'            : cols_a_traiter,
            'details'             : winsorisations
        })

        if self.verbose:
            print("\n============================================================================")
            print("ÉTAPE 3.2 : WINSORISATION/ÉCRÊTAGE")
            print("============================================================================")
            print(f"Percentiles..............: {percentile_bas*100:.0f}% - {percentile_haut*100:.0f}%")
            print(f"Mode.....................: {'Inplace' if inplace else 'Nouvelles colonnes'}")
            print(f"Colonnes traitées........: {len(winsorisations)}")
            print()

            # Affichage détaillé
            for wins in winsorisations:
                col_affichage = wins['colonne']
                if not inplace:
                    col_affichage = f"{wins['colonne']} → {wins['colonne_finale']}"

                print(f"  • {col_affichage}")
                print(f"    Bornes...............: [{wins['borne_inf']:.2f}, {wins['borne_sup']:.2f}]")
                print(f"    Valeurs cappées......: {wins['nb_total_cappe']} ({wins['pct_cappe']:.2f}%)")
                print(f"      - Bas (< {wins['borne_inf']:.2f})...: {wins['nb_cappe_bas']}")
                print(f"      - Haut (> {wins['borne_sup']:.2f}).: {wins['nb_cappe_haut']}")
                print()

            print("============================================================================")

        return self

    def traitement_asymetrie_complet(
        self,
        log_colonnes: Optional[List[str]] = None,
        log_auto_detect: bool = True,
        log_skew_threshold: float = 1.0,
        wins_colonnes: Optional[List[str]] = None,
        wins_percentiles: tuple = (0.01, 0.99),
        wins_inplace: bool = False
    ) -> 'DataCleaner':
        """
        Exécute la séquence complète de traitement de l'asymétrie (Étape 3).
        
        PIPELINE COMPLET:
        3.1 → Transformation logarithmique
        3.2 → Winsorisation
        
        Args:
            log_colonnes: Colonnes pour transformation log (None = auto)
            log_auto_detect: Auto-détection des colonnes asymétriques
            log_skew_threshold: Seuil de skewness
            wins_colonnes: Colonnes pour winsorisation (None = toutes)
            wins_percentiles: Tuple (percentile_bas, percentile_haut)
            wins_inplace: Modifier colonnes originales ou créer nouvelles
            
        Returns:
            Self pour chaînage des méthodes
        """
        if self.verbose:
            print("\n============================================================================")
            print("ÉTAPE 3: TRAITEMENT COMPLET DE L'ASYMÉTRIE ET OUTLIERS")
            print("============================================================================")

        # 3.1 Transformation logarithmique
        self.transformation_logarithmique(
            colonnes=log_colonnes,
            auto_detect=log_auto_detect,
            skew_threshold=log_skew_threshold
        )

        # 3.2 Winsorisation
        self.winsorisation(
            colonnes=wins_colonnes,
            percentile_bas=wins_percentiles[0],
            percentile_haut=wins_percentiles[1],
            inplace=wins_inplace
        )

        if self.verbose:
            print("\n============================================================================")
            print("BILAN ÉTAPE 3 COMPLÈTE")
            print("============================================================================")
            print(f"Shape finale.............: {self.df.shape}")

            # Compter les nouvelles colonnes
            nb_cols_log  = len([c for c in self.df.columns if '_log' in c])
            nb_cols_wins = len([c for c in self.df.columns if '_wins' in c])

            print(f"Colonnes log créées......: {nb_cols_log}")
            print(f"Colonnes wins créées.....: {nb_cols_wins}")
            print("============================================================================")

        return self


    # =============================================================================
    # ANÁLISIS PREVIO: ¿QUÉ VARIABLES NECESITAN TRANSFORMACIÓN?
    # =============================================================================

    def analizar_asimetria_y_outliers(self, df, umbral_skew=1.0, umbral_outliers=0.05):
        """
        Analiza qué columnas necesitan transformación logarítmica o winsorización.
        
        Args:
            df: DataFrame a analizar
            umbral_skew: Umbral de skewness para considerar transformación log
            umbral_outliers: % de outliers para considerar winsorización
            
        Returns:
            Dict con recomendaciones
        """
        print("\n" + "="*80)
        print("ANÁLISIS PREVIO: DETECCIÓN DE ASIMETRÍA Y OUTLIERS")
        print("="*80)

        # IMPORTANTE: Usamos self.df
        df_interno = self.df

        # Identificar columnas numéricas (excluir indicadores)
        num_cols = df_interno.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if '_Manquant' not in c]

        resultados = {
            'necesitan_log': [],
            'necesitan_winsorisation': [],
            'detalles': []
        }

        for col in num_cols:
            # Calcular métricas
            skewness = df[col].skew()

            # Detectar outliers usando IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            pct_outliers = (n_outliers / len(df)) * 100

            # Recomendaciones
            recomendacion_log = abs(skewness) > umbral_skew
            recomendacion_wins = pct_outliers > (umbral_outliers * 100)

            if recomendacion_log:
                resultados['necesitan_log'].append(col)

            if recomendacion_wins:
                resultados['necesitan_winsorisation'].append(col)

            resultados['detalles'].append({
                'colonne': col,
                'skewness': skewness,
                'pct_outliers': pct_outliers,
                'recomendacion_log': '✓' if recomendacion_log else '✗',
                'recomendacion_wins': '✓' if recomendacion_wins else '✗'
            })

        # Mostrar resultados
        df_analisis = pd.DataFrame(resultados['detalles'])
        df_analisis = df_analisis.sort_values('skewness', key=abs, ascending=False)

        print("\nRÉSUMÉ DES RECOMMANDATIONS:")
        print("-" * 80)
        print(f"Variables nécessitant LOG........: {len(resultados['necesitan_log'])}")
        print(f"Variables nécessitant WINSORISATION: {len(resultados['necesitan_winsorisation'])}")
        print()

        print("\nDÉTAIL PAR VARIABLE:")
        print("-" * 80)
        print(f"{'Colonne':<35} {'Skewness':>10} {'Outliers%':>10} {'LOG':>5} {'WINS':>5}")
        print("-" * 80)

        for _, row in df_analisis.iterrows():
            print(f"{row['colonne']:<35} "
                  f"{row['skewness']:>10.3f} "
                  f"{row['pct_outliers']:>9.2f}% "
                  f"{row['recomendacion_log']:>5} "
                  f"{row['recomendacion_wins']:>5}")

        print("="*80)

        return resultados


    def visualizar_distribuciones(self, df, colonnes, max_cols=6):
        """
        Visualiza las distribuciones de las columnas seleccionadas.
        """

        df = self.df

        colonnes_validas = [c for c in colonnes if c in df.columns]
        n_cols = min(len(colonnes_validas), max_cols)

        if n_cols == 0:
            print("⚠️  Aucune colonne valide à visualiser")
            return

        fig, axes = plt.subplots(n_cols, 2, figsize=(14, 4*n_cols))
        if n_cols == 1:
            axes = axes.reshape(1, -1)

        for idx, col in enumerate(colonnes_validas[:max_cols]):
            # Histograma
            axes[idx, 0].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx, 0].set_title(f'{col} - Distribution')
            axes[idx, 0].set_xlabel('Valeur')
            axes[idx, 0].set_ylabel('Fréquence')

            # Boxplot
            axes[idx, 1].boxplot(df[col].dropna(), vert=True)
            axes[idx, 1].set_title(f'{col} - Boxplot')
            axes[idx, 1].set_ylabel('Valeur')

            # Añadir skewness
            skew = df[col].skew()
            axes[idx, 0].text(0.02, 0.98, f'Skewness: {skew:.3f}',
                             transform=axes[idx, 0].transAxes,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()


    def obtener_resumen_transformaciones(self) -> pd.DataFrame:
        """
        Genera un resumen de todas las transformaciones aplicadas en Étape 3.
        
        Returns:
            DataFrame con el resumen de transformaciones LOG y WINS
        """
        # Buscar operaciones de Étape 3 en el historial
        ops_etape3 = [
            op for op in self.history
            if op['operation'] in ['transformation_logarithmique', 'winsorisation']
        ]

        if not ops_etape3:
            print("⚠️  Aucune transformation de l'Étape 3 trouvée")
            return pd.DataFrame()

        resumen_data = []

        for op in ops_etape3:
            if op['operation'] == 'transformation_logarithmique':
                for detail in op['details']:
                    resumen_data.append({
                        'Transformation': 'LOG',
                        'Colonne_Origine': detail['colonne'],
                        'Colonne_Résultat': detail['colonne_log'],
                        'Skewness_Avant': round(detail['skew_avant'], 3),
                        'Skewness_Après': round(detail['skew_apres'], 3),
                        'Amélioration': round(abs(detail['skew_avant']) - abs(detail['skew_apres']), 3),
                        'Décalage': detail['decalage'] if detail['has_negative'] else 0
                    })

            elif op['operation'] == 'winsorisation':
                for detail in op['details']:
                    resumen_data.append({
                        'Transformation': 'WINS',
                        'Colonne_Origine': detail['colonne'],
                        'Colonne_Résultat': detail['colonne_finale'],
                        'Valeurs_Cappées': detail['nb_total_cappe'],
                        'Pct_Cappé': round(detail['pct_cappe'], 2),
                        'Borne_Inf': round(detail['borne_inf'], 2),
                        'Borne_Sup': round(detail['borne_sup'], 2)
                    })

        df_resumen = pd.DataFrame(resumen_data)

        if self.verbose and not df_resumen.empty:
            print("\n" + "="*80)
            print("RÉSUMÉ DES TRANSFORMATIONS ÉTAPE 3")
            print("="*80)

            # Separar por tipo
            df_log = df_resumen[df_resumen['Transformation'] == 'LOG']
            df_wins = df_resumen[df_resumen['Transformation'] == 'WINS']

            if not df_log.empty:
                print("\nTRANSFORMATIONS LOGARITHMIQUES:")
                print("-"*80)
                for _, row in df_log.iterrows():
                    print(f"  • {row['Colonne_Origine']} → {row['Colonne_Résultat']}")
                    print(f"    Skewness: {row['Skewness_Avant']} → {row['Skewness_Après']} "
                          f"(amélioration: {row['Amélioration']})")

            if not df_wins.empty:
                print("\nWINSORISATIONS:")
                print("-"*80)
                for _, row in df_wins.iterrows():
                    print(f"  • {row['Colonne_Origine']} → {row['Colonne_Résultat']}")
                    print(f"    Cappées: {row['Valeurs_Cappées']} ({row['Pct_Cappé']}%)")
                    print(f"    Bornes: [{row['Borne_Inf']}, {row['Borne_Sup']}]")

            print("="*80)

        return df_resumen


    def identifier_features_racines(self, suffixes: list[str] = ['_log', '_wins']) -> dict[str, list[str]]:
        """
        Analyse le DataFrame pour trouver les variables 'racines' qui possèdent 
        des versions transformées (log, wins, etc.).
        
        Returns:
            Dict: { 'colonne_racine': ['colonne_derivee_1', 'colonne_derivee_2'] }
        """
        cols_presentes = self.df.columns.tolist()
        mappage_redondance = {}

        for col in cols_presentes:
            for suffix in suffixes:
                if col.endswith(suffix):
                    # Déduire la racine
                    racine = col.replace(suffix, '')

                    # Vérifier si la racine existe encore dans le DF
                    if racine in cols_presentes:
                        if racine not in mappage_redondance:
                            mappage_redondance[racine] = []
                        mappage_redondance[racine].append(col)

        if self.verbose:
            print(f"\n{'─'*70}")
            print(f"{'AUDIT DE REDONDANCE (RACINES VS DÉRIVÉES)':^70}")
            print(f"{'─'*70}")

            if not mappage_redondance:
                print("✅ Aucune redondance détectée (les racines ont déjà été supprimées ou n'existent pas).")
            else:
                for racine, derives in mappage_redondance.items():
                    print(f"Variable Racine : {racine:30s} ➔ Dérivée(s) : {derives}")
            print(f"{'─'*70}\n")

        return mappage_redondance

    # =========================================================================
    # 4. Création de Nouvelles Features
    # =========================================================================
    def ____4_Creation_de_Nouvelles_Features(self): pass

    def ajouter_feature(
        self,
        nom: str,
        calcul: callable,
        description: str = '',
        dtype: Optional[type] = None,
        validar: bool = True
    ) -> 'FeatureEngineer':
        """
        Método genérico para añadir cualquier feature.
        
        Args:
            nom: Nombre de la nueva columna
            calcul: Función lambda o callable que recibe df y retorna Series
            description: Descripción de la feature
            dtype: Tipo de datos a forzar (opcional)
            validar: Si True, valida que no exista ya la columna
            
        Returns:
            Self para chaînage
            
        Example:
            >>> engineer.ajouter_feature(
            ...     nom='Parking_Ratio',
            ...     calcul=lambda df: df['PropertyGFAParking'] / df['PropertyGFATotal'],
            ...     description='Ratio parking/total'
            ... )
        """

        # Validación
        if validar and nom in self.df.columns:
            if self.verbose:
                print(f"⚠️  Feature '{nom}' déjà existante - ignorée")
            return self

        # Cálculo
        try:
            nueva_col = calcul(self.df)

            # Forzar tipo si se especifica
            if dtype is not None:
                nueva_col = nueva_col.astype(dtype)

            # Añadir al DataFrame
            self.df[nom] = nueva_col

            # Tracking
            self.columns_added.append({
                'nom': nom,
                'description': description,
                'dtype': str(nueva_col.dtype),
                'n_unique': nueva_col.nunique(),
                'missing': nueva_col.isnull().sum()
            })

            # Tracking
            self.features_creadas.append({
                'nom': nom,
                'description': description,
                'dtype': str(nueva_col.dtype),
                'n_unique': nueva_col.nunique(),
                'missing': nueva_col.isnull().sum()
            })

            # Histórico
            self.history.append({
                'operation': 'ajouter_feature',
                'feature': nom,
                'description': description,
                'dtype': str(nueva_col.dtype),
            })

            if self.verbose:
                print(f"✅ Feature créée: {nom}")
                if description:
                    print(f"   Description: {description}")
                print(f"   Type: {nueva_col.dtype}, "
                      f"Unique: {nueva_col.nunique()}, "
                      f"Missing: {nueva_col.isnull().sum()}")

        except Exception as e:
            print(f"❌ Erreur création '{nom}': {e}")

        return self


    def _generer_features_ingenierie(self, df):
        """
        SOURCE UNIQUE DE VÉRITÉ : Cette méthode contient toute la logique mathématique.
        Elle est appelée par .fit() (sur df_fit) et par .transform() (sur df_clean).
        """
        # 1. Copie locale pour éviter les effets de bord sur le DataFrame original
        df_temp = df.copy()

        # --------------------------------------------------------------------------
        # 4.1 & 4.2 Distance au Centre et au Port (Calcul géospatial)
        # --------------------------------------------------------------------------
        # Note : Ces calculs sont déterministes et ne dépendent pas du fit.
        dist_results = calculer_distances_points_cles(
            df_temp, col_lat='Latitude', col_lon='Longitude'
        )
        df_temp['F1_Distance_Centre_m'] = dist_results[0]
        df_temp['F2_Distance_Port_m']   = dist_results[1]

        # --------------------------------------------------------------------------
        # 4.3 Densite_Voisinage_{rayon}m (Analyse de proximité)
        # --------------------------------------------------------------------------
        RADII_METERS = [500, 1000, 1500, 2000]
        if self.reference_tree is not None:
            for rayon in RADII_METERS:
                # La méthode capture l'état de self.reference_tree appris lors du fit
                df_temp[f'F3_Densite_Voisinage_{rayon}m'] = calcular_densite_voisinage_with_reference_tree(
                    df_temp, rayon_m=rayon, reference_tree=self.reference_tree
                )

        # --------------------------------------------------------------------------
        # 4.4 Taille_Batiment_Ordinale (Binning et Mapping)
        # --------------------------------------------------------------------------
        # Seuils pour la maille "Taille du bâtiment" (en pieds carrés - sqft)
        SURFACE_SMALL_THRESHOLD = 20000
        SURFACE_LARGE_THRESHOLD = 100000
        MAP_TAILLE = {'Petit' : 0, 'Moyen' : 1, 'Grand' : 2}

        # ÉTAPE : GÉNÉRATION DES MAILLES DE TAILLE (BINNING)
        categories = pd.cut(
            df_temp['PropertyGFATotal'],
            bins    = [0, SURFACE_SMALL_THRESHOLD, SURFACE_LARGE_THRESHOLD, np.inf],
            labels  = ['Petit', 'Moyen', 'Grand']
        )
        # ÉTAPE : CONVERSION NUMÉRIQUE DES MAILLES DE TAILLE
        df_temp['F4_Taille_Batiment_Ordinale'] = categories.map(MAP_TAILLE).astype(int)

        # --------------------------------------------------------------------------
        # 4.5 Création de 'Usage_Diversity'
        # --------------------------------------------------------------------------
        # Calcule si le bâtiment a un usage unique, double ou triple
        df_temp['F5_Usage_Diversity'] = (
            (df_temp['SecondLargestPropertyUseType'] != 'INCONNU').astype(int) +
            (df_temp['ThirdLargestPropertyUseType']  != 'INCONNU').astype(int)
        )

        # --------------------------------------------------------------------------
        # 4.6 & 4.7 Ratios et Indicateurs de Parking
        # --------------------------------------------------------------------------
        # Mesure la proportion de la surface dédiée au parking par rapport à la surface totale
        df_temp['F6_Parking_Ratio'] = np.where(
            df_temp['PropertyGFATotal'] > 0,
            df_temp['PropertyGFAParking'] / df_temp['PropertyGFATotal'],
            0
        )
        # Variable binaire indiquant si le bâtiment possède un parking ou non
        df_temp['F7_Has_Parking'] = (df_temp['PropertyGFAParking'] > 0).astype(int)

        # --------------------------------------------------------------------------
        # 4.8 Binning: Discrétiser ENERGYSTARScore
        # --------------------------------------------------------------------------
        # Transforme une variable continue en catégories qualitatives.
        df_temp['F8_ENERGYSTARScore_Category'] = pd.cut(
            df_temp['ENERGYSTARScore'],
            bins=[0, 50, 75, 90, 100],
            labels=['Faible', 'Moyen', 'Bon', 'Excellent'],
            include_lowest=True
        ).astype('category')

        # --------------------------------------------------------------------------
        # 4.9. Indice d'efficacité relative (Utilisation de la moyenne apprise)
        # --------------------------------------------------------------------------
        #if hasattr(self, 'media_referencia_SiteEUI') and self.media_referencia_SiteEUI:
        #    # Nous créons la feature en utilisant la moyenne sauvegardée dans le fit
        #    df_temp['F9_Energy_Efficiency_Index'] = df_temp['SiteEUI(kBtu/sf)'] / self.media_referencia_SiteEUI

        # --------------------------------------------------------------------------
        # 4.10. Ratio d'usage principal
        # --------------------------------------------------------------------------
        # Mesure la dominance de l'usage principal par rapport à la surface totale
        df_temp['F10_Primary_Use_Ratio'] = np.where(
            df_temp['PropertyGFATotal'] > 0,
            df_temp['LargestPropertyUseTypeGFA'] / df_temp['PropertyGFATotal'],
            0
        )

        # --------------------------------------------------------------------------
        # 4.11 Building_Age (Âge du bâtiment)
        # --------------------------------------------------------------------------
        # L'âge est souvent plus corrélé à la consommation que l'année de construction brute.
        df_temp['F11_Building_Age'] = 2016 - df_temp['YearBuilt']

        # --------------------------------------------------------------------------
        # 4.12 Floors_per_Building (Densité verticale moyenne)
        # --------------------------------------------------------------------------
        # Calcule le nombre moyen d'étages par structure physique sur la propriété.
        df_temp['F12_floors_by_building_mean'] = np.where(
            df_temp['NumberofBuildings'] > 0,
            df_temp['NumberofFloors'] / df_temp['NumberofBuildings'],
            df_temp['NumberofFloors']
        )


        return df_temp

    # =========================================================================
    # 5. Codification Catégorielle
    # =========================================================================
    def ____5_Codification_Categorielle(self): pass

    def suggest_encoding_strategies(self, df, max_ohe=12, max_binary=18):
        """
        Segmenta las columnas categóricas en diferentes estrategias de encoding
        basadas en su cardinalidad.
        """
        candidatas = df.select_dtypes(include=['object', 'category']).columns.tolist()

        report = {
            'ohe': [],
            'binary': [],
            'target_avancé': [],
            'ignored': []
        }

        print("🔍 Analyse des Stratégies d'Encodage")
        print(f"{'Colonne':<30} | {'Uniques':<8} | {'Stratégie Recommandée'}")
        print("-" * 75)

        for col in candidatas:
            n_uniques = df[col].nunique()

            # 1. Caso OHE: Baja cardinalidad (Interpretación directa)
            if n_uniques <= max_ohe:
                strategy = "✅ ONE-HOT ENCODING"
                report['ohe'].append(col)

            # 2. Caso BINARY: Cardinalidad media (Compresión logarítmica)
            elif max_ohe < n_uniques <= max_binary:
                strategy = "🔷 BINARY ENCODING"
                report['binary'].append(col)

            # 3. Caso TARGET: Alta cardinalidad (Evitar explosión de columnas)
            else:
                strategy = "🚀 TARGET ENCODING (AVANCÉ)"
                report['target_avancé'].append(col)

            print(f"{col:<30} | {n_uniques:<8} | {strategy}")

        print("-" * 75)
        for k, v in report.items():
            if v: print(f"TOTAL {k.upper()}: {len(v)} colonnes")

        return report


    def get_categorical_encoding_blueprint(self, df, meta_dict, max_o_ohe=12, max_binary=18):
        """
        Analyse, classifie et synchronise les stratégies d'encodage.
        Remplit les listes de contrôle et met à jour le dictionnaire par référence.
        """

        candidatas = df.select_dtypes(include=['object', 'category']).columns.tolist()

        print("🔍 ANALYSE ET SYNCHRONISATION DU BLUEPRINT")
        print(f"{'Colonne':<28} | {'Stratégie':<22} | {'Aperçu Modalités'}")
        print("-" * 85)

        # Estructura de salida profesional
        blueprint = {
            'ohe': {},            # {col: [valeurs]}
            'binary': [],         # Liste simple (le binaire gère ses propres mapping)
            'target_avancé': [],  # Liste simple
            'details': {}         # Infos complémentaires
        }

        for col in candidatas:
            valeurs = sorted(df[col].dropna().unique().tolist())
            n_uniques = len(valeurs)

            # --- Logique de décision de stratégie ---
            if n_uniques <= max_o_ohe:
                strategy = "✅ OneHotEncoding"
                self.COLS_ONE_HOT_ENCODING.append(col)
                blueprint['ohe'][col] = valeurs
            elif max_o_ohe < n_uniques <= max_binary:
                strategy = "🔷 BinaryEncoding"
                self.COLS_BINARY_ENCODING.append(col)
                blueprint['binary'].append(col)
            else:
                strategy = "🚀 TargetAdvancedEncoding"
                self.COLS_TARGET_ADVANCED_ENCODING.append(col)
                blueprint['target_avancé'].append(col)

            # --- Mise à jour du meta_dict (par référence) ---
            if col in meta_dict:
                meta_dict[col].update({
                    'valeurs_possibles': valeurs,
                    'n_uniques_detectes': n_uniques,
                    'type_encodage': strategy,
                    'encodage_necessaire': True
                })

            # Guardamos metadatos adicionales para tu meta_dict_enrichi
            blueprint['details'][col] = {
                'n_uniques': n_uniques,
                'modalites': valeurs,
                'strategy': strategy
            }

            # --- Visualisation propre ---
            mod_str = ", ".join(map(str, valeurs))
            if len(mod_str) > 65: mod_str = mod_str[:32] + "..."
            print(f"{col:<28} | {strategy:<22} | {mod_str}")

        print("-" * 85)
        print(f"✅ Sync terminée. OHE: {len(self.COLS_ONE_HOT_ENCODING)} | "
              f"Binary: {len(self.COLS_BINARY_ENCODING)} | Target: {len(self.COLS_TARGET_ADVANCED_ENCODING)}")
        return blueprint


    def get_encoding_blueprint_BAK(self, df, max_ohe=12, max_binary=18) -> Dict:
        """
        Analyse les colonnes et extrait les modalités (valeurs possibles) 
        pour chaque stratégie d'encodage.
        """
        candidatas = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Estructura de salida profesional
        blueprint = {
            'ohe': {},            # {col: [valeurs]}
            'binary': [],         # Liste simple (le binaire gère ses propres mapping)
            'target_avancé': [],  # Liste simple
            'details': {}         # Infos complémentaires
        }

        print("🔍 Construction du Blueprint d'Encodage")
        print(f"{'Colonne':<30} | {'Uniques':<8} | {'Stratégie'}")
        print("-" * 75)

        for col in candidatas:
            # Extraer valores únicos ordenados (para consistencia)
            valeurs = sorted(df[col].dropna().unique().tolist())
            n_uniques = len(valeurs)

            # 1. Caso OHE: Extraemos las categorías exactas
            if n_uniques <= max_ohe:
                strategy = "✅ OHE"
                blueprint['ohe'][col] = valeurs

            # 2. Caso BINARY
            elif max_ohe < n_uniques <= max_binary:
                strategy = "🔷 BINARY"
                blueprint['binary'].append(col)

            # 3. Caso TARGET
            else:
                strategy = "🚀 TARGET"
                blueprint['target_avancé'].append(col)

            print(f"{col:<30} | {n_uniques:<8} | {strategy}")

            # Guardamos metadatos adicionales para tu meta_dict_enrichi
            blueprint['details'][col] = {
                'n_uniques': n_uniques,
                'modalites': valeurs
            }

        return blueprint

    def convert_percentages_to_float(self, df, meta_dict, columns=None):
        """
        Identifie et convertit les pourcentages en float.
        Met à jour le meta_dict_enrichi pour refléter le nouveau type technique.
        """
        # Si no se especifican columnas, buscamos patrones de '%' en las tipo object
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()

        for col in columns:
            # Verificamos si la columna existe y tiene el símbolo %
            if col in df.columns and df[col].dtype == 'object' and df[col].str.contains('%', na=False).any():

                # --- TRANSFORMACIÓN DE DATOS ---
                print(f"📊 Transformation : {col} (Categoriel % -> Numerique)")

                # Limpieza robusta: quita %, espacios, cambia coma por punto y divide por 100
                df[col] = (
                    df[col]
                    .str.replace('%', '', regex=False)
                    .str.replace(' ', '', regex=False)
                    .str.replace(',', '.', regex=False)
                    .astype(float) / 100
                )

                # --- ACTUALIZACIÓN DE METADATOS (Tu parámetro) ---
                if col in meta_dict:
                    meta_dict[col]['type'] = 'float64'
                    meta_dict[col]['classification'] = 'numeriques_continues'
                    meta_dict[col]['encodage_necessaire'] = False
                    meta_dict[col]['type_encodage'] = None
                    meta_dict[col]['note'] = "Converti automatiquement de string % à float"
                    # Ya no necesitamos valores_possibles porque ahora es un rango continuo
                    meta_dict[col]['valeurs_possibles'] = f"Range: [{df[col].min()}, {df[col].max()}]"

        return df, meta_dict

    def identifier_candidates_ohe(self, df, max_cardinality: int = 15) -> list:
        """
        Identifie les colonnes catégorielles éligibles au One-Hot Encoding.
        
        Args:
            max_cardinality: Nombre maximum de catégories uniques acceptées pour le OHE.
        """
        # 1. Sélectionner les colonnes non-numériques (object ou category)
        candidatas = df.select_dtypes(include=['object', 'category']).columns.tolist()

        sugerencias = []

        print(f"🔍 Examen des colonnes pour One-Hot Encoding (Limite : {max_cardinality} catégories)")
        print("-" * 60)

        for col in candidatas:
            n_uniques = df[col].nunique()

            if n_uniques <= max_cardinality:
                status = "✅ OHE RECOMMANDÉ"
                sugerencias.append(col)
            else:
                status = "⚠️ HAUTE CARDINALITÉ (Trop de colonnes)"

            print(f"{col:<30} | Uniques: {n_uniques:>3} | {status}")

        print("-" * 60)
        print(f"Total de colonnes suggérées : {len(sugerencias)}")

        return sugerencias

    def analyser_pour_ohe(self, col: str):
        """
        Analyse une colonne pour évaluer l'impact d'un One-Hot Encoding.
        """
        if col not in self.df.columns:
            print(f"❌ La colonne '{col}' n'existe pas.")
            return

        uniques = self.df[col].unique()
        n_uniques = len(uniques)
        counts = self.df[col].value_counts()

        print(f"📊 Analyse de la feature : {col}")
        print("---")
        print(f"• Nombre de catégories uniques : {n_uniques}")
        print(f"• Type de données : {self.df[col].dtype}")
        print(f"• Top 5 des catégories :\n{counts.head(5)}")

        # Conseil d'expert (Style Yann LeCun)
        if n_uniques > 15:
            print(f"\n⚠️ ATTENTION : Cette variable a une forte cardinalité ({n_uniques}).")
            print("Le OHE va créer beaucoup de colonnes. Envisagez de regrouper les catégories rares.")
        else:
            print("\n✅ Faible cardinalité. Le OHE est recommandé ici.")

    def apply_one_hot_encoding(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drop_first: bool = True,
        prefix_sep: str = '_'
    ) -> 'FeatureEngineer':
        """
        Applique One-Hot Encoding aux colonnes spécifiées.
        
        Paramètres :
        -----------
        df : pd.DataFrame
            dataframe du dataset
        columns : List[str]
            Liste des colonnes catégorielles à encoder
        drop_first : bool (default=True)
            Si True, supprime la première catégorie (évite multicolinéarité)
        prefix_sep : str (default='_')
            Séparateur entre le nom de colonne et la catégorie
        
        Retourne :
        ---------
        Self pour chaînage
        
        Example:
            >>> engineer.apply_one_hot_encoding(
            ...     columns=['BuildingType', 'ComplianceStatus'],
            ...     drop_first=True
            ... )
        """

        if self.verbose:
            print(f"\n{'='*60}")
            print("ONE-HOT ENCODING")
            print(f"{'='*60}")

        df_res = df.copy()

        new_columns_total = []
        encoding_details = []

        for col in columns:
            if col not in df_res.columns:
                if self.verbose:
                    print(f"⚠️  Colonne '{col}' introuvable - ignorée")
                continue

            # Información pre-encoding
            n_categories = df_res[col].nunique()
            categories   = df_res[col].unique().tolist()
            n_missing    = df_res[col].isnull().sum()

            # Créer les dummies
            dummies = pd.get_dummies(
                df_res[col],
                prefix=col,
                prefix_sep=prefix_sep,
                drop_first=drop_first,
                dtype=int
            )

            # Ajouter au DataFrame
            df_res = pd.concat([df_res, dummies], axis=1)
            new_cols = dummies.columns.tolist()
            new_columns_total.extend(new_cols)

            # Tracking détaillé
            encoding_details.append({
                'colonne_originale': col,
                'n_categories': n_categories,
                'categories': categories[:5] if len(categories) > 5 else categories,  # Max 5 pour le log
                'n_missing_avant': n_missing,
                'drop_first': drop_first,
                'nouvelles_colonnes': new_cols,
                'n_colonnes_creees': len(new_cols)
            })

            # Registrar cada nueva columna
            for new_col in new_cols:
                self.columns_added.append({
                    'nom': new_col,
                    'description': f'OHE de {col}',
                    'dtype': 'int64',
                    'n_unique': 2,  # Binaire por definición
                    'missing': 0,
                    'source': 'one_hot_encoding'
                })

            # Supprimer la colonne originale
            df_res.drop(columns=[col], inplace=True)

            # Tracking de suppression
            self.columns_suppressed.append({
                'nom': col,
                'raison': 'One-Hot Encoding appliqué',
                'type': 'categorical',
                'remplacé_par': new_cols
            })

            if self.verbose:
                print(f"\n✅ {col}")
                print(f"   Catégories : {n_categories} → {len(new_cols)} colonnes binaires")
                if n_missing > 0:
                    print(f"   ⚠️  Valeurs manquantes : {n_missing}")
                print(f"   Nouvelles colonnes : {', '.join(new_cols[:3])}"
                      f"{'...' if len(new_cols) > 3 else ''}")

        # Histórico global
        self.history.append({
            'operation': 'one_hot_encoding',
            'n_colonnes_encodees': len(encoding_details),
            'colonnes': [d['colonne_originale'] for d in encoding_details],
            'n_colonnes_creees': len(new_columns_total),
            'nouvelles_colonnes': new_columns_total,
            'drop_first': drop_first,
            'details': encoding_details
        })

        if self.verbose:
            print(f"\n{'='*60}")
            print("RÉSUMÉ OHE")
            print(f"{'='*60}")
            print(f"Colonnes encodées    : {len(encoding_details)}")
            print(f"Colonnes créées      : {len(new_columns_total)}")
            print(f"Colonnes supprimées  : {len(encoding_details)}")
            print(f"Forme DataFrame      : {df_res.shape}")
            print(f"{'='*60}\n")

        return df_res

    def identifier_candidates_target_encode(self, min_cardinality: int = 10, max_cardinality: int = 100) -> list:
        """
        Identifie les colonnes avec une cardinalité trop élevée pour le OHE
        mais potentiellement utiles pour le Target Encoding.
        """
        candidates = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        suggestions = []

        print(f"🔍 Recherche de candidats pour Target Encoding (>{min_cardinality} catégories)")
        print("-" * 70)

        for col in candidates:
            n_uniques = self.df[col].nunique()
            if n_uniques >= min_cardinality:
                print(f"✅ {col:<30} | Uniques: {n_uniques:>3} | PRUDENCE : Risque d'overfitting élevé")
                suggestions.append(col)
            else:
                print(f"❌ {col:<30} | Uniques: {n_uniques:>3} | Trop faible (utilisez OHE)")

        return suggestions

    def analyser_pour_target_encode(self, col: str, target: str = 'SiteEnergyUse(kBtu)_log'):
        """
        Analyse la pertinence du Target Encoding avec un diagnostic précis des colonnes.
        """
        # 1. Verificación precisa de existencia
        missing_cols = []
        if col not in self.df.columns:
            missing_cols.append(f"Feature: '{col}'")
        if target not in self.df.columns:
            missing_cols.append(f"Target: '{target}'")

        if missing_cols:
            print(f"❌ Erreur de référence : {', '.join(missing_cols)} introuvable(s) dans le DataFrame.")
            print(f"💡 Colonnes disponibles (extrait) : {list(self.df.columns[:5])}...")
            return

        # 2. Cálculo de estadísticas
        stats = self.df.groupby(col)[target].agg(['mean', 'std', 'count']).sort_values(by='mean')

        print(f"📊 Analyse de la relation : {col} ➔ {target}")
        print("-" * 60)
        display(stats)

        # 3. Visualización (Boxplot)
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        # Ordenamos el boxplot por la media para que sea legible
        order = stats.index
        sns.boxplot(
                    x=col,
                    y=target,
                    data=self.df,
                    order=order,
                    hue=col,          # Asignamos la variable X al color
                    legend=False,     # Quitamos la leyenda porque es redundante con el eje X
                    palette='viridis'
                )

        plt.xticks(rotation=45, ha='right')
        plt.title(f"Impact de {col} sur {target} (ordonné par moyenne)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def apply_target_encode_cv(
        self,
        df: pd.DataFrame,
        column: str,
        target: str,
        n_folds: int = 5,
        smoothing: float = 10.0,
        suffix: str = '_Encoded'
    ) -> 'FeatureEngineer':
        """
        Applique Target Encoding avec Cross-Validation pour éviter le leakage.
        
        Paramètres :
        -----------
        df : pd.DataFrame
            dataframe du dataset
        column : str
            Nom de la colonne à encoder
        target : str
            Nom de la colonne cible
        n_folds : int (default=5)
            Nombre de folds pour la CV
        smoothing : float (default=10.0)
            Paramètre de lissage (régularisation bayésienne)
        suffix : str (default='_Encoded')
            Suffixe pour la nouvelle colonne
        
        Retourne :
        ---------
        Self pour chaînage
        
        Example:
            >>> engineer.target_encode_cv(
            ...     column='Neighborhood',
            ...     target='SiteEUI(kBtu/sf)',
            ...     n_folds=5,
            ...     smoothing=10.0
            ... )
        """
        from sklearn.model_selection import KFold

        df_res = df.copy()

        if column not in df_res.columns:
            print(f"❌ Colonne '{column}' introuvable")
            return self

        if target not in df_res.columns:
            print(f"❌ Target '{target}' introuvable")
            return self

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TARGET ENCODING : {column}")
            print(f"{'='*60}")

        # Información pre-encoding
        n_categories = df_res[column].nunique()
        n_missing = df_res[column].isnull().sum()
        categories_sample = df_res[column].value_counts().head(10).to_dict()

        # Inicializar
        encoded = pd.Series(index=df_res.index, dtype=float, name=f'{column}{suffix}')
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Moyenne globale de la cible
        global_mean = df_res[target].mean()

        # Estadísticas por categoría (para tracking)
        category_stats_full = df_res.groupby(column)[target].agg(['mean', 'count', 'std'])

        # Cross-Validation
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(df_res), 1):
            # Calculer les moyennes sur le fold d'entraînement
            train_df = df_res.iloc[train_idx]

            # Compter les occurrences par catégorie
            category_stats = train_df.groupby(column)[target].agg(['mean', 'count'])

            # Appliquer le smoothing (régularisation bayésienne)
            category_stats['smoothed'] = (
                (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
                (category_stats['count'] + smoothing)
            )

            # Mapper les valeurs du fold de validation
            encoded.iloc[val_idx] = df_res.iloc[val_idx][column].map(
                category_stats['smoothed']
            ).fillna(global_mean)

        # Ajouter la nouvelle colonne
        new_col_name = f'{column}{suffix}'
        df_res[new_col_name] = encoded

        # Tracking
        self.columns_added.append({
            'nom': new_col_name,
            'description': f'Target Encoding de {column} (CV={n_folds})',
            'dtype': 'float64',
            'n_unique': encoded.nunique(),
            'missing': encoded.isnull().sum(),
            'source': 'target_encoding',
            'smoothing': smoothing,
            'global_mean': global_mean
        })

        # Supprimer la colonne originale
        df_res.drop(columns=[column], inplace=True)

        self.columns_suppressed.append({
            'nom': column,
            'raison': 'Target Encoding appliqué',
            'type': 'categorical',
            'remplacé_par': [new_col_name],
            'n_categories': n_categories
        })

        # Histórico
        self.history.append({
            'operation': 'target_encoding',
            'colonne': column,
            'target': target,
            'nouvelle_colonne': new_col_name,
            'n_folds': n_folds,
            'smoothing': smoothing,
            'global_mean': round(global_mean, 4),
            'n_categories': n_categories,
            'n_missing_avant': n_missing,
            'categories_sample': categories_sample,
            'stats_summary': {
                'mean_encoded': round(encoded.mean(), 4),
                'std_encoded': round(encoded.std(), 4),
                'min_encoded': round(encoded.min(), 4),
                'max_encoded': round(encoded.max(), 4)
            }
        })

        if self.verbose:
            print(f"Target               : {target}")
            print(f"Catégories           : {n_categories}")
            print(f"Global Mean          : {global_mean:.4f}")
            print(f"Smoothing            : {smoothing}")
            print(f"CV Folds             : {n_folds}")
            print(f"\nNouvelle colonne     : {new_col_name}")
            print(f"  Mean               : {encoded.mean():.4f}")
            print(f"  Std                : {encoded.std():.4f}")
            print(f"  Range              : [{encoded.min():.4f}, {encoded.max():.4f}]")
            print(f"  Missing            : {encoded.isnull().sum()}")

            if n_missing > 0:
                print(f"\n⚠️  {n_missing} valeurs manquantes → remplacées par global_mean")

            print(f"\n{'='*60}\n")

        return df_res


    # =========================================================================
    # 6. Réduction de la Redondance
    # =========================================================================
    def ____6_Reduction_de_la_Redondance(self): pass

    def rapport_redondance(self, df, threshold=0.90):
        """
        Identifie les paires de variables avec une corrélation supérieure au seuil.
        Retourne un dictionnaire avec les paires et leur score.
        """
        # Convertir a Pandas para el cálculo de correlación
        df_pd = df.to_pandas() if hasattr(df, 'to_pandas') else df
        corr_matrix = df_pd.select_dtypes(include=['number']).corr()

        redondances = []
        cols = corr_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                score = corr_matrix.iloc[i, j]
                if abs(score) > threshold:
                    redondances.append({
                        'var1': cols[i],
                        'var2': cols[j],
                        'correlation': round(score, 4)
                    })

        return redondances


    def filtrer_features_pertinentes(self, df, target="SiteEnergyUse(kBtu)", threshold=0.3):
        """
        Identifie automatiquement les variables numériques qui ont une corrélation
        significative avec la variable cible (target).
        """
        # Conversion en Pandas pour la matrice de corrélation
        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df

        # Calcul de la corrélation par rapport à la cible
        corr_matrix = df_pd.select_dtypes(include=['number']).corr()
        corr_target = corr_matrix[target].abs().sort_values(ascending=False)

        # On ne garde que celles qui dépassent le seuil de significativité
        features_pertinentes = corr_target[corr_target > threshold].index.tolist()

        return features_pertinentes

    def plot_correlation_heatmap(self, df, features, meta_dict):
        """
        Génère une Heatmap lisible avec des noms de variables traduits.
        """
        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df

        # Extraction et renommage des colonnes
        data_corr = df_pd[features].rename(columns=meta_dict)
        corr = data_corr.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                    linewidths=.5, cbar_kws={"shrink": .8})

        plt.title("Matrice de Corrélation : Variables Clés du Modèle", fontsize=15, fontweight='bold')
        plt.show()

    def analyser_correlations(
        self,
        df,
        target: Optional[str] = None,
        seuil: float = 0.90,
        method: str = 'pearson',
        inclure_target: bool = True,
        top_n: int = 20,
        visualiser: bool = True,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Dict[str, any]:
        """
        Analyse la matrice de corrélation et identifie les colonnes redondantes.
        
        Paramètres :
        -----------
        target : str, optional
            Nom de la colonne cible pour prioriser les corrélations
        seuil : float (default=0.90)
            Seuil de corrélation pour considérer deux variables redondantes
        method : str (default='pearson')
            Méthode de corrélation : 'pearson', 'spearman', 'kendall'
        inclure_target : bool (default=True)
            Si True, inclut la target dans l'analyse de corrélation
        top_n : int (default=20)
            Nombre de paires les plus corrélées à afficher
        visualiser : bool (default=True)
            Si True, génère un heatmap des corrélations
        figsize : Tuple[int, int]
            Taille de la figure pour le heatmap
        
        Retourne :
        ---------
        Dict contenant :
            - matrice_corr : DataFrame de corrélation complète
            - paires_correlees : Liste des paires |ρ| > seuil
            - candidats_suppression : Colonnes recommandées pour suppression
            - stats : Statistiques de corrélation
        
        Example:
            >>> results = engineer.analyser_correlations(
            ...     target='SiteEUI(kBtu/sf)',
            ...     seuil=0.90,
            ...     visualiser=True
            ... )
            >>> print(f"Candidats à supprimer : {results['candidats_suppression']}")
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        from itertools import combinations

        if self.verbose:
            print(f"\n{'='*70}")
            print("ANALYSE DE CORRÉLATION")
            print(f"{'='*70}")

        # Sélectionner uniquement les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclure la target de l'analyse de redondance si demandé
        if target and not inclure_target and target in numeric_cols:
            numeric_cols_analysis = [c for c in numeric_cols if c != target]
        else:
            numeric_cols_analysis = numeric_cols

        if len(numeric_cols_analysis) < 2:
            print("⚠️  Pas assez de colonnes numériques pour l'analyse")
            return {}

        # Calculer la matrice de corrélation
        corr_matrix = df[numeric_cols_analysis].corr(method=method)

        # Identifier les paires hautement corrélées
        paires_correlees = []
        upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if upper_triangle[i, j]:
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= seuil:
                        paires_correlees.append({
                            'var1': col1,
                            'var2': col2,
                            'correlation': corr_val,
                            'abs_correlation': abs(corr_val)
                        })

        # Trier par corrélation absolue décroissante
        paires_correlees = sorted(
            paires_correlees,
            key=lambda x: x['abs_correlation'],
            reverse=True
        )

        # Identifier les candidats à la suppression
        candidats_suppression = []
        colonnes_a_garder = set()

        if target and target in df.columns:
            # Calculer la corrélation avec la target
            target_corr = df[numeric_cols_analysis].corrwith(
                df[target],
                method=method
            ).abs()

            # Pour chaque paire, garder celle la plus corrélée avec target
            for paire in paires_correlees:
                var1, var2 = paire['var1'], paire['var2']

                # Si aucune des deux n'est déjà marquée à garder
                if var1 not in colonnes_a_garder and var2 not in colonnes_a_garder:

                    # --- MODIFICATION ICI : Extraction sécurisée des valeurs ---
                    # On utilise .loc[var] pour être précis et on prend la première valeur [0]
                    # au cas où il y aurait des doublons d'index.
                    try:
                        val1 = target_corr.loc[var1]
                        val2 = target_corr.loc[var2]

                        # Si c'est une Series, on prend la première valeur, sinon la valeur directe
                        c1 = val1.iloc[0] if isinstance(val1, pd.Series) else val1
                        c2 = val2.iloc[0] if isinstance(val2, pd.Series) else val2

                        # Conversion forcée en float et gestion des NaNs
                        corr1 = float(c1) if pd.notna(c1) else 0.0
                        corr2 = float(c2) if pd.notna(c2) else 0.0
                    except Exception:
                        corr1, corr2 = 0.0, 0.0
                    # -----------------------------------------------------------

                    if corr1 >= corr2:
                        colonnes_a_garder.add(var1)
                        candidats_suppression.append({
                            'colonne': var2,
                            'correlée_avec': var1,
                            'correlation': paire['correlation'],
                            'corr_target_gardée': corr1,
                            'corr_target_supprimée': corr2,
                            'raison': f'Redondante avec {var1} (ρ={paire["correlation"]:.3f})'
                        })
                    else:
                        colonnes_a_garder.add(var2)
                        candidats_suppression.append({
                            'colonne': var1,
                            'correlée_avec': var2,
                            'correlation': paire['correlation'],
                            'corr_target_gardée': corr2,
                            'corr_target_supprimée': corr1,
                            'raison': f'Redondante avec {var2} (ρ={paire["correlation"]:.3f})'
                        })
        else:
            # Sans target, approche arbitraire : garder la première de chaque paire
            for paire in paires_correlees:
                var1, var2 = paire['var1'], paire['var2']
                if var1 not in colonnes_a_garder and var2 not in colonnes_a_garder:
                    colonnes_a_garder.add(var1)
                    candidats_suppression.append({
                        'colonne': var2,
                        'correlée_avec': var1,
                        'correlation': paire['correlation'],
                        'raison': f'Redondante avec {var1} (ρ={paire["correlation"]:.3f})'
                    })

        # Statistiques générales
        stats = {
            'n_colonnes_numeriques': len(numeric_cols_analysis),
            'n_paires_totales': len(list(combinations(numeric_cols_analysis, 2))),
            'n_paires_correlees': len(paires_correlees),
            'n_candidats_suppression': len(candidats_suppression),
            'seuil': seuil,
            'method': method,
            'target': target
        }

        # Affichage des résultats
        if self.verbose:
            print(f"\nMéthode              : {method.capitalize()}")
            print(f"Seuil                : |ρ| ≥ {seuil}")
            print(f"Colonnes numériques  : {len(numeric_cols_analysis)}")
            print(f"Paires totales       : {stats['n_paires_totales']}")
            print(f"Paires corrélées     : {len(paires_correlees)}")
            print(f"Candidats suppression: {len(candidats_suppression)}")

            if paires_correlees:
                print(f"\n{'─'*70}")
                print(f"TOP {min(top_n, len(paires_correlees))} PAIRES LES PLUS CORRÉLÉES")
                print(f"{'─'*70}")

                for idx, paire in enumerate(paires_correlees[:top_n], 1):
                    print(f"\n{idx}. ρ = {paire['correlation']:+.4f}")
                    print(f"   {paire['var1']}")
                    print(f"   {paire['var2']}")

            if candidats_suppression:
                print(f"\n{'─'*70}")
                print(f"CANDIDATS À LA SUPPRESSION ({len(candidats_suppression)})")
                print(f"{'─'*70}")

                for idx, candidat in enumerate(candidats_suppression[:top_n], 1):
                    print(f"\n{idx}. {candidat['colonne']}")
                    print(f"   Corrélée avec : {candidat['correlée_avec']} "
                          f"(ρ={candidat['correlation']:+.4f})")
                    if 'corr_target_gardée' in candidat:
                        print(f"   Corr. target gardée   : {candidat['corr_target_gardée']:.4f}")
                        print(f"   Corr. target supprimée: {candidat['corr_target_supprimée']:.4f}")

            print(f"\n{'='*70}\n")

        # Visualisation
        if visualiser and len(paires_correlees) > 0:
            # Sélectionner les colonnes impliquées dans les corrélations élevées
            cols_to_plot = set()
            for paire in paires_correlees[:30]:  # Top 30 paires
                cols_to_plot.add(paire['var1'])
                cols_to_plot.add(paire['var2'])

            if target and target in df.columns:
                cols_to_plot.add(target)

            cols_to_plot = list(cols_to_plot)[:50]  # Limiter à 50 colonnes max

            # Heatmap
            plt.figure(figsize=figsize)
            mask = np.triu(np.ones_like(corr_matrix.loc[cols_to_plot, cols_to_plot], dtype=bool))

            sns.heatmap(
                corr_matrix.loc[cols_to_plot, cols_to_plot],
                mask=mask,
                annot=False,
                fmt='.2f',
                cmap='RdBu_r',
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Corrélation"}
            )

            plt.title(
                f'Matrice de Corrélation ({method.capitalize()})\n'
                f'Colonnes avec |ρ| ≥ {seuil} (Top {len(cols_to_plot)})',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            plt.xlabel('')
            plt.ylabel('')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

        # Construir resultado
        resultado = {
            'matrice_corr': corr_matrix,
            'paires_correlees': paires_correlees,
            'candidats_suppression': candidats_suppression,
            'colonnes_a_garder': list(colonnes_a_garder),
            'stats': stats
        }

        # Histórico
        self.history.append({
            'operation': 'analyser_correlations',
            'method': method,
            'seuil': seuil,
            'n_paires_correlees': len(paires_correlees),
            'n_candidats_suppression': len(candidats_suppression),
            'target': target,
            'stats': stats
        })

        return resultado


    def visualiser_top_correlations(
        self,
        target: str,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Visualise les colonnes les plus corrélées avec la target.
        
        Paramètres :
        -----------
        target : str
            Nom de la colonne cible
        top_n : int (default=20)
            Nombre de features à afficher
        figsize : Tuple[int, int]
            Taille de la figure
        """
        import matplotlib.pyplot as plt

        if target not in self.df.columns:
            print(f"❌ Target '{target}' introuvable")
            return

        # Calculer corrélations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target]

        correlations = self.df[numeric_cols].corrwith(self.df[target]).abs()
        correlations = correlations.sort_values(ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=figsize)
        colors = ['#d62728' if x < 0 else '#2ca02c'
                  for x in self.df[correlations.index].corrwith(self.df[target])]

        plt.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
        plt.yticks(range(len(correlations)), correlations.index)
        plt.xlabel('|Corrélation| avec Target', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Features Corrélées avec {target}',
                  fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

        if self.verbose:
            print(f"\nTop {top_n} corrélations avec '{target}':")
            for col, corr in correlations.items():
                print(f"  {col:50s} : {corr:.4f}")

    def eliminer_correlees(
        self,
        colonnes: List[str],
        raison: str = 'Corrélation élevée avec autre variable',
        valider: bool = True
    ) -> 'FeatureEngineer':
        """
        Élimine les colonnes hautement corrélées de manière supervisée.
        
        Paramètres :
        -----------
        colonnes : List[str]
            Liste des colonnes à supprimer
        raison : str
            Raison de la suppression (pour tracking)
        valider : bool (default=True)
            Si True, demande confirmation avant suppression
        
        Retourne :
        ---------
        Self pour chaînage
        
        Example:
            >>> # Après avoir analysé les corrélations
            >>> results = engineer.analyser_correlations(target='SiteEUI(kBtu/sf)')
            >>> colonnes_a_supprimer = [c['colonne'] for c in results['candidats_suppression']]
            >>> engineer.eliminer_correlees(colonnes_a_supprimer)
        """

        if not colonnes:
            if self.verbose:
                print("⚠️  Aucune colonne à supprimer")
            return self

        # Filtrer les colonnes qui existent vraiment
        colonnes_existantes = [c for c in colonnes if c in self.df.columns]
        colonnes_manquantes = [c for c in colonnes if c not in self.df.columns]

        if colonnes_manquantes and self.verbose:
            print(f"⚠️  Colonnes introuvables : {colonnes_manquantes}")

        if not colonnes_existantes:
            if self.verbose:
                print("⚠️  Aucune colonne valide à supprimer")
            return self

        # Validation interactive
        if valider and self.verbose:
            print(f"\n{'='*60}")
            print(f"SUPPRESSION DE {len(colonnes_existantes)} COLONNES CORRÉLÉES")
            print(f"{'='*60}")
            print("\nColonnes à supprimer :")
            for idx, col in enumerate(colonnes_existantes, 1):
                print(f"  {idx:2d}. {col}")

            confirmation = input("\nConfirmer la suppression ? [o/N] : ").strip().lower()

            if confirmation not in ['o', 'oui', 'y', 'yes']:
                print("❌ Suppression annulée")
                return self

        # Información pre-suppression
        shape_avant = self.df.shape

        # Supprimer
        for col in colonnes_existantes:
            # Info de la colonne
            col_info = {
                'nom': col,
                'raison': raison,
                'dtype': str(self.df[col].dtype),
                'n_unique': self.df[col].nunique(),
                'missing': self.df[col].isnull().sum(),
                'type': 'correlation_redundancy'
            }

            # Tracking
            self.columns_suppressed.append(col_info)

            # Supprimer
            self.df.drop(columns=[col], inplace=True)

        shape_apres = self.df.shape

        # Histórico
        self.history.append({
            'operation': 'eliminer_correlees',
            'n_colonnes_supprimees': len(colonnes_existantes),
            'colonnes': colonnes_existantes,
            'raison': raison,
            'shape_avant': shape_avant,
            'shape_apres': shape_apres
        })

        if self.verbose:
            print(f"\n{'='*60}")
            print("RÉSUMÉ SUPPRESSION")
            print(f"{'='*60}")
            print(f"Colonnes supprimées  : {len(colonnes_existantes)}")
            print(f"Forme avant          : {shape_avant}")
            print(f"Forme après          : {shape_apres}")
            print(f"{'='*60}\n")

        return self


    def pipeline_reduction_redondance(
        self,
        df,
        target: str,
        seuil: float = 0.99,         # seuil: float = 0.90,
        method: str = 'pearson',
        auto_eliminer: bool = False,
        visualiser: bool = True
    ) -> Dict[str, any]:
        """
        Pipeline complet de réduction de redondance.
        
        Paramètres :
        -----------
        target : str
            Nom de la colonne cible
        seuil : float (default=0.90)
            Seuil de corrélation pour considérer redondance
        method : str (default='pearson')
            Méthode de corrélation
        auto_eliminer : bool (default=False)
            Si True, élimine automatiquement sans demander confirmation
        visualiser : bool (default=True)
            Si True, génère visualisations
        
        Retourne :
        ---------
        Résultats de l'analyse de corrélation
        
        Example:
            >>> results = engineer.pipeline_reduction_redondance(
            ...     target='SiteEUI(kBtu/sf)',
            ...     seuil=0.90,
            ...     auto_eliminer=False  # Demande confirmation
            ... )
        """

        if self.verbose:
            print(f"\n{'#'*70}")
            print("# PIPELINE : RÉDUCTION DE LA REDONDANCE")
            print(f"{'#'*70}\n")

        # 1. Analyser les corrélations
        results = self.analyser_correlations(df,
            target=target,
            seuil=seuil,
            method=method,
            visualiser=visualiser
        )

        # 2. Éliminer si demandé
        if results.get('candidats_suppression'):
            colonnes_a_supprimer = [c['colonne'] for c in results['candidats_suppression']]

            # --- PROTECCIÓN PARA EL FIT ---
            # Solo eliminamos físicamente si auto_eliminer es True.
            # Si es False, solo informamos y dejamos que el fit guarde la lista.
            if auto_eliminer:
                self.eliminer_correlees(
                    colonnes=colonnes_a_supprimer,
                    raison=f'Corrélation |ρ| ≥ {seuil}',
                    valider=False # Ya que es automático
                )
            else:
                if self.verbose:
                    print(f"ℹ️ [FIT] {len(colonnes_a_supprimer)} colonnes identifiées comme redondantes.")
                    print("ℹ️ Elles seront conservées pendant le FIT et supprimées lors du TRANSFORM.")
        else:
            if self.verbose:
                print("✅ Aucune colonne redondante détectée")

        # 3. Histórico del pipeline
        self.history.append({
            'operation': 'pipeline_reduction_redondance',
            'target': target,
            'seuil': seuil,
            'method': method,
            'n_candidats': len(results.get('candidats_suppression', [])),
            'auto_eliminer': auto_eliminer
        })

        return results



    # =========================================================================
    # 7. Mise à l'Échelle Finale (Scaling)
    # =========================================================================
    def ____7_Mise_a_Echelle_Finale(self): pass

    def verifier_colonnes_scaling(
        self,
        target: Optional[str] = None,
        seuil_binaire: int = 2,
        afficher_details: bool = True
    ) -> Dict[str, any]:
        """
        Vérifie et catégorise les colonnes avant la standardisation.
        
        Identifie :
        - Colonnes numériques à scaler
        - Colonnes binaires (à exclure)
        - Colonnes catégorielles/textuelles (à exclure)
        - Target (à exclure)
        - Colonnes avec variance nulle (à exclure)
        
        Paramètres :
        -----------
        target : str, optional
            Nom de la colonne cible
        seuil_binaire : int (default=2)
            Nombre de valeurs uniques max pour considérer une colonne binaire
        afficher_details : bool (default=True)
            Si True, affiche le détail de chaque catégorie
        
        Retourne :
        ---------
        Dict contenant :
            - a_scaler : Liste des colonnes à standardiser
            - binaires : Liste des colonnes binaires
            - categoriques : Liste des colonnes catégorielles
            - variance_nulle : Colonnes sans variance
            - target : Nom de la target
            - stats : Statistiques générales
        
        Example:
            >>> verif = engineer.verifier_colonnes_scaling(target='SiteEUIkBtu/sf')
            >>> print(f"À scaler : {len(verif['a_scaler'])} colonnes")
            >>> print(f"À exclure : {len(verif['binaires']) + len(verif['categoriques'])}")
        """

        if self.verbose:
            print(f"\n{'='*70}")
            print("VÉRIFICATION PRÉ-SCALING")
            print(f"{'='*70}")

        # Initialisation des catégories
        a_scaler = []
        binaires = []
        categoriques = []
        variance_nulle = []
        target_col = None

        # Informations détaillées
        details = {
            'a_scaler': [],
            'binaires': [],
            'categoriques': [],
            'variance_nulle': []
        }

        # Analyser chaque colonne
        for col in self.df.columns:

            # 1. Target : à exclure
            if target and col == target:
                target_col = col
                continue

            dtype = self.df[col].dtype
            n_unique = self.df[col].nunique()
            n_missing = self.df[col].isnull().sum()

            # 2. Colonnes catégorielles/textuelles : à exclure
            if dtype == 'object' or dtype.name == 'category':
                categoriques.append(col)
                details['categoriques'].append({
                    'nom': col,
                    'dtype': str(dtype),
                    'n_unique': n_unique,
                    'n_missing': n_missing
                })
                continue

            # 3. Colonnes booléennes : à exclure
            if dtype == 'bool':
                binaires.append(col)
                details['binaires'].append({
                    'nom': col,
                    'dtype': 'bool',
                    'n_unique': n_unique,
                    'valeurs': self.df[col].unique().tolist()
                })
                continue

            # 4. Colonnes numériques
            if dtype in ['int64', 'int32', 'float64', 'float32', 'int8', 'int16']:

                # 4a. Vérifier variance nulle
                variance = self.df[col].var()
                if variance == 0 or pd.isna(variance):
                    variance_nulle.append(col)
                    details['variance_nulle'].append({
                        'nom': col,
                        'dtype': str(dtype),
                        'valeur_unique': self.df[col].unique()[0] if n_unique == 1 else None
                    })
                    continue

                # 4b. Vérifier si binaire (0/1 ou deux valeurs uniques)
                valeurs_uniques = self.df[col].dropna().unique()

                if n_unique <= seuil_binaire:
                    # Vérifier si c'est vraiment binaire (0/1 ou deux valeurs)
                    if set(valeurs_uniques).issubset({0, 1}):
                        binaires.append(col)
                        details['binaires'].append({
                            'nom': col,
                            'dtype': str(dtype),
                            'n_unique': n_unique,
                            'valeurs': sorted(valeurs_uniques.tolist())
                        })
                        continue

                # 4c. Colonne numérique à scaler
                a_scaler.append(col)
                details['a_scaler'].append({
                    'nom': col,
                    'dtype': str(dtype),
                    'n_unique': n_unique,
                    'n_missing': n_missing,
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'range': self.df[col].max() - self.df[col].min()
                })

        # Statistiques générales
        stats = {
            'total_colonnes': len(self.df.columns),
            'n_a_scaler': len(a_scaler),
            'n_binaires': len(binaires),
            'n_categoriques': len(categoriques),
            'n_variance_nulle': len(variance_nulle),
            'n_target': 1 if target_col else 0,
            'n_a_exclure': len(binaires) + len(categoriques) + len(variance_nulle) + (1 if target_col else 0)
        }

        # Affichage détaillé
        if self.verbose:
            print(f"\nTotal colonnes       : {stats['total_colonnes']}")
            print(f"{'─'*70}")

            # Target
            if target_col:
                print("\n🎯 TARGET (à exclure)")
                print(f"{'─'*70}")
                print(f"  • {target_col}")
                print(f"    dtype={self.df[target_col].dtype}, "
                      f"unique={self.df[target_col].nunique()}, "
                      f"range=[{self.df[target_col].min():.2f}, {self.df[target_col].max():.2f}]")

            # Colonnes à scaler
            print(f"\n✅ COLONNES À SCALER ({len(a_scaler)})")
            print(f"{'─'*70}")

            if afficher_details and a_scaler:
                for idx, info in enumerate(details['a_scaler'][:10], 1):  # Limiter à 10
                    print(f"\n  {idx:2d}. {info['nom']}")
                    print(f"      dtype={info['dtype']}, unique={info['n_unique']}, "
                          f"missing={info['n_missing']}")
                    print(f"      μ={info['mean']:.2f}, σ={info['std']:.2f}, "
                          f"range=[{info['min']:.2f}, {info['max']:.2f}]")

                if len(a_scaler) > 10:
                    print(f"\n  ... et {len(a_scaler) - 10} autres colonnes")
            else:
                print(f"  {', '.join(a_scaler[:5])}")
                if len(a_scaler) > 5:
                    print(f"  ... et {len(a_scaler) - 5} autres")

            # Colonnes binaires
            if binaires:
                print(f"\n🔘 COLONNES BINAIRES (à exclure : {len(binaires)})")
                print(f"{'─'*70}")

                if afficher_details:
                    for idx, info in enumerate(details['binaires'][:10], 1):
                        print(f"  {idx:2d}. {info['nom']:45s} | "
                              f"valeurs={info['valeurs']}")

                    if len(binaires) > 10:
                        print(f"  ... et {len(binaires) - 10} autres")
                else:
                    print(f"  {', '.join(binaires[:5])}")
                    if len(binaires) > 5:
                        print(f"  ... et {len(binaires) - 5} autres")

            # Colonnes catégorielles
            if categoriques:
                print(f"\n📝 COLONNES CATÉGORIELLES (à exclure : {len(categoriques)})")
                print(f"{'─'*70}")

                for idx, info in enumerate(details['categoriques'], 1):
                    print(f"  {idx:2d}. {info['nom']:45s} | "
                          f"dtype={info['dtype']}, unique={info['n_unique']}")

            # Colonnes variance nulle
            if variance_nulle:
                print(f"\n⚠️  COLONNES VARIANCE NULLE (à exclure : {len(variance_nulle)})")
                print(f"{'─'*70}")

                for idx, info in enumerate(details['variance_nulle'], 1):
                    val = info['valeur_unique']
                    print(f"  {idx:2d}. {info['nom']:45s} | "
                          f"valeur={val}")

            # Résumé
            print(f"\n{'='*70}")
            print("RÉSUMÉ")
            print(f"{'='*70}")
            print(f"✅ À scaler          : {stats['n_a_scaler']} colonnes")
            print(f"❌ À exclure (total) : {stats['n_a_exclure']} colonnes")
            print(f"   - Target          : {stats['n_target']}")
            print(f"   - Binaires        : {stats['n_binaires']}")
            print(f"   - Catégorielles   : {stats['n_categoriques']}")
            print(f"   - Variance nulle  : {stats['n_variance_nulle']}")
            print(f"{'='*70}\n")

        # Resultado
        resultado = {
            'a_scaler': a_scaler,
            'binaires': binaires,
            'categoriques': categoriques,
            'variance_nulle': variance_nulle,
            'target': target_col,
            'stats': stats,
            'details': details
        }

        # Histórico
        self.history.append({
            'operation': 'verifier_colonnes_scaling',
            'target': target_col,
            'stats': stats
        })

        return resultado


    def recommander_exclure_scaling(
        self,
        target: Optional[str] = None
    ) -> List[str]:
        """
        Retourne la liste recommandée de colonnes à exclure du scaling.
        
        Paramètres :
        -----------
        target : str, optional
            Nom de la colonne cible
        
        Retourne :
        ---------
        Liste des colonnes à exclure
        
        Example:
            >>> exclure = engineer.recommander_exclure_scaling(target='SiteEUIkBtu/sf')
            >>> engineer.standardiser_features(exclure=exclure)
        """

        verif = self.verifier_colonnes_scaling(target=target, afficher_details=False)

        exclure = []

        # Target
        if verif['target']:
            exclure.append(verif['target'])

        # Binaires
        exclure.extend(verif['binaires'])

        # Catégorielles
        exclure.extend(verif['categoriques'])

        # Variance nulle
        exclure.extend(verif['variance_nulle'])

        if self.verbose:
            print(f"\n📋 RECOMMANDATION : Exclure {len(exclure)} colonnes du scaling")

        return exclure

    def gestionar_outliers(
        self,
        colonnes: Optional[List[str]] = None,
        metodo: str = 'quantile',  # 'iqr', 'zscore', 'quantile'
        umbral_quantile: Tuple[float, float] = (0.01, 0.99),
        umbral_zscore: float = 3.0,
        umbral_iqr: float = 1.5,
        accion: str = 'clip',  # 'clip', 'remove', 'nan'
        exclure: Optional[List[str]] = None
    ) -> 'DataCleaner':
        """
        Gestiona outliers con diferentes métodos, privilegiando quantiles para 
        evitar pérdida excesiva de datos.
        
        Paramètres:
        -----------
        colonnes : List[str], optional
            Colonnes à traiter. Si None, toutes les numériques.
        metodo : str (default='quantile')
            Méthode de détection:
            - 'quantile': Basé sur percentiles (moins agressif, recommandé)
            - 'iqr': Interquartile Range (modérément agressif)
            - 'zscore': Z-score (plus agressif)
        umbral_quantile : Tuple[float, float] (default=(0.01, 0.99))
            Quantiles min et max pour clipper (ex: 1% et 99%)
        umbral_zscore : float (default=3.0)
            Seuil pour z-score (valeurs au-delà de ±umbral)
        umbral_iqr : float (default=1.5)
            Multiplicateur pour IQR
        accion : str (default='clip')
            Action à prendre:
            - 'clip': Limiter aux seuils (conserve toutes les lignes)
            - 'remove': Supprimer les lignes avec outliers
            - 'nan': Remplacer outliers par NaN
        exclure : List[str], optional
            Colonnes à exclure du traitement
        
        Returns:
        --------
        Self pour chaînage
        
        Example:
            >>> # Approche recommandée: quantiles avec clipping
            >>> cleaner.gestionar_outliers(
            ...     metodo='quantile',
            ...     umbral_quantile=(0.01, 0.99),
            ...     accion='clip'
            ... ).standardiser_features()

            
            # 1. Gestionar outliers primero (menos agresivo)
            cleaner.gestionar_outliers(
                metodo='quantile',
                umbral_quantile=(0.01, 0.99),  # Conserva 98% de datos
                accion='clip',  # No elimina filas
                exclure=['target_variable']
            )
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"GESTION DES OUTLIERS - Méthode: {metodo.upper()}")
            print(f"{'='*70}")

        # Sélectionner colonnes
        if colonnes is None:
            colonnes = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if exclure:
            colonnes = [c for c in colonnes if c not in exclure]

        colonnes = [c for c in colonnes if c in self.df.columns]

        if not colonnes:
            if self.verbose:
                print("⚠️  Aucune colonne à traiter")
            return self

        shape_avant = self.df.shape
        outliers_detectes = {}

        for col in colonnes:
            serie = self.df[col].copy()
            n_avant = serie.notna().sum()

            # Détection selon méthode
            if metodo == 'quantile':
                q_low = serie.quantile(umbral_quantile[0])
                q_high = serie.quantile(umbral_quantile[1])
                mask_outliers = (serie < q_low) | (serie > q_high)

            elif metodo == 'iqr':
                Q1 = serie.quantile(0.25)
                Q3 = serie.quantile(0.75)
                IQR = Q3 - Q1
                q_low = Q1 - umbral_iqr * IQR
                q_high = Q3 + umbral_iqr * IQR
                mask_outliers = (serie < q_low) | (serie > q_high)

            elif metodo == 'zscore':
                z_scores = np.abs((serie - serie.mean()) / serie.std())
                mask_outliers = z_scores > umbral_zscore
                q_low = serie[~mask_outliers].min()
                q_high = serie[~mask_outliers].max()

            else:
                raise ValueError(f"Méthode inconnue: {metodo}")

            n_outliers = mask_outliers.sum()

            # Action sur outliers
            if accion == 'clip':
                self.df[col] = serie.clip(lower=q_low, upper=q_high)
            elif accion == 'nan':
                self.df.loc[mask_outliers, col] = np.nan
            elif accion == 'remove':
                # Se hará después para todas las columnas
                pass

            outliers_detectes[col] = {
                'n_outliers': n_outliers,
                'pourcentage': (n_outliers / n_avant * 100),
                'seuil_bas': q_low,
                'seuil_haut': q_high
            }

        # Si remove, eliminar filas
        if accion == 'remove':
            mask_total = pd.Series(False, index=self.df.index)
            for col in colonnes:
                serie = self.df[col]
                if metodo == 'quantile':
                    q_low = serie.quantile(umbral_quantile[0])
                    q_high = serie.quantile(umbral_quantile[1])
                    mask_total |= (serie < q_low) | (serie > q_high)

            self.df = self.df[~mask_total].copy()

        # Histórico
        self.history.append({
            'operation': 'gestionar_outliers',
            'metodo': metodo,
            'accion': accion,
            'n_colonnes': len(colonnes),
            'colonnes': colonnes,
            'shape_avant': shape_avant,
            'shape_apres': self.df.shape,
            'outliers_detectes': outliers_detectes
        })

        if self.verbose:
            print(f"\nMéthode          : {metodo}")
            print(f"Action           : {accion}")
            print(f"Colonnes traitées: {len(colonnes)}")

            if metodo == 'quantile':
                print(f"Quantiles        : {umbral_quantile[0]:.1%} - {umbral_quantile[1]:.1%}")

            print(f"\n{'─'*70}")
            print("OUTLIERS DÉTECTÉS PAR COLONNE")
            print(f"{'─'*70}")

            total_outliers = 0
            for col, stats in list(outliers_detectes.items())[:10]:
                print(f"{col:30s} : {stats['n_outliers']:5d} ({stats['pourcentage']:5.2f}%) "
                      f"[{stats['seuil_bas']:.2f}, {stats['seuil_haut']:.2f}]")
                total_outliers += stats['n_outliers']

            if accion == 'remove':
                lignes_supprimees = shape_avant[0] - self.df.shape[0]
                print(f"\n⚠️  Lignes supprimées : {lignes_supprimees} "
                      f"({lignes_supprimees/shape_avant[0]*100:.2f}%)")

            print(f"\nForme finale : {self.df.shape}")
            print(f"{'='*70}\n")

        return self


    def standardiser_features(
        self,
        colonnes: Optional[List[str]] = None,
        exclure: Optional[List[str]] = None,
        garder_originales: bool = False,
        suffix: str = '_scaled'
    ) -> 'DataCleaner':
        """
        Applique StandardScaler (z-score normalization) aux features numériques.
        
        Formule : z = (x - μ) / σ
        
        Paramètres :
        -----------
        target : str, optional
            Le nom de la variable cible à exclure systématiquement.
        colonnes : List[str], optional
            Liste des colonnes à standardiser. Si None, toutes les numériques.
        exclure : List[str], optional
            Liste des colonnes à NE PAS standardiser (ex: binaires)
        garder_originales : bool (default=False)
            Si True, crée nouvelles colonnes avec suffix au lieu de remplacer
        suffix : str (default='_scaled')
            Suffixe pour les nouvelles colonnes si garder_originales=True
        
        Retourne :
        ---------
        Self pour chaînage
        
        Example:
            >>> # Standardiser toutes les features sauf target et binaires
            >>> engineer.standardiser_features(
            ...     target='SiteEnergyUsekBtu_wins',
            ...     exclure=['Has_Parking'],
            ...     garder_originales=False
            ... )
        """

        if self.verbose:
            print(f"\n{'='*70}")
            print("STANDARDISATION (StandardScaler)")
            print(f"{'='*70}")

        # Sélectionner colonnes numériques
        if colonnes is None:
            colonnes = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclure colonnes spécifiées
        if exclure:
            colonnes = [c for c in colonnes if c not in exclure]


        # Filtrer colonnes existantes
        colonnes = [c for c in colonnes if c in self.df.columns]

        if not colonnes:
            if self.verbose:
                print("⚠️  Aucune colonne à standardiser")
            return self

        # Información pre-scaling
        shape_avant = self.df.shape
        stats_avant = {}

        for col in colonnes:
            stats_avant[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max()
            }

        # Appliquer StandardScaler
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(self.df[colonnes])

        # Créer DataFrame avec valeurs standardisées
        scaled_df = pd.DataFrame(
            scaled_values,
            columns=[f'{c}{suffix}' for c in colonnes] if garder_originales else colonnes,
            index=self.df.index
        )

        if garder_originales:
            # Ajouter nouvelles colonnes
            self.df = pd.concat([self.df, scaled_df], axis=1)
            nouvelles_colonnes = scaled_df.columns.tolist()

            # Tracking
            for col_orig, col_scaled in zip(colonnes, nouvelles_colonnes):
                self.columns_added.append({
                    'nom': col_scaled,
                    'description': f'Standardisation de {col_orig}',
                    'dtype': 'float64',
                    'n_unique': self.df[col_scaled].nunique(),
                    'missing': 0,
                    'source': 'standardisation',
                    'mean_avant': stats_avant[col_orig]['mean'],
                    'std_avant': stats_avant[col_orig]['std']
                })
        else:
            # Remplacer colonnes existantes
            self.df[colonnes] = scaled_df
            nouvelles_colonnes = colonnes

        # Statistiques post-scaling (pour vérification)
        stats_apres = {}
        for col in nouvelles_colonnes:
            stats_apres[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max()
            }

        # Histórico
        self.history.append({
            'operation': 'standardiser_features',
            'n_colonnes': len(colonnes),
            'colonnes': colonnes,
            'garder_originales': garder_originales,
            'suffix': suffix if garder_originales else None,
            'shape_avant': shape_avant,
            'shape_apres': self.df.shape,
            'scaler_params': {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
        })

        if self.verbose:
            print(f"\nColonnes standardisées : {len(colonnes)}")
            print("Méthode                 : StandardScaler (z-score)")
            print(f"Garder originales       : {garder_originales}")
            print(f"Forme DataFrame         : {self.df.shape}")

            # Montrer quelques exemples de transformation
            print(f"\n{'─'*70}")
            print("EXEMPLES DE TRANSFORMATION (5 premières colonnes)")
            print(f"{'─'*70}")

            for idx, col in enumerate(colonnes[:5], 1):
                col_scaled = nouvelles_colonnes[idx-1]
                print(f"\n{idx}. {col}")
                print(f"   Avant : μ={stats_avant[col]['mean']:.2f}, "
                      f"σ={stats_avant[col]['std']:.2f}, "
                      f"range=[{stats_avant[col]['min']:.2f}, {stats_avant[col]['max']:.2f}]")
                print(f"   Après : μ={stats_apres[col_scaled]['mean']:.2e}, "
                      f"σ={stats_apres[col_scaled]['std']:.2f}, "
                      f"range=[{stats_apres[col_scaled]['min']:.2f}, {stats_apres[col_scaled]['max']:.2f}]")

            print(f"\n{'='*70}\n")

        return self


    def supprimer_haute_cardinalite(
        self,
        seuil: float = 0.50,
        exclure: Optional[List[str]] = None
    ) -> 'DataCleaner':
        """
        Identifie et supprime les colonnes catégorielles avec un ratio de 
        cardinalité trop élevé (identifiants uniques).
        
        Args:
            seuil: Ratio (0.0 a 1.0) de valeurs uniques par rapport au total.
                   Par défaut 0.50 (plus de 50% de valeurs uniques).
            exclure: Liste de colonnes à protéger de la suppression.
            
        Returns:
            Self pour chaînage des méthodes
        """
        # 1. Identifier les colonnes "object" ou "category"
        cols_qualitatives = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        if exclure:
            cols_qualitatives = [c for c in cols_qualitatives if c not in exclure]

        cols_a_supprimer = []

        # 2. Calculer le ratio de cardinalité pour chaque colonne
        for col in cols_qualitatives:
            n_unique = self.df[col].nunique()
            total = len(self.df)
            ratio = n_unique / total

            if ratio > seuil:
                cols_a_supprimer.append(col)

        # 3. Réutiliser la logique de suppression existante
        if cols_a_supprimer:
            self.df = self.df.drop(columns=cols_a_supprimer)
            self.columns_suppressed.extend(cols_a_supprimer)

            self.history.append({
                'operation': 'suppression_haute_cardinalite',
                'colonnes': cols_a_supprimer,
                'seuil_utilise': seuil,
                'nb_colonnes': len(cols_a_supprimer)
            })

            if self.verbose:
                print(f"\n✓ Suppression de {len(cols_a_supprimer)} "
                      f"colonnes à haute cardinalité (Seuil: {seuil})")
                for col in cols_a_supprimer:
                    n_u = self.df_init[col].nunique() if hasattr(self, 'df_init') else "N/A"
                    print(f"  - {col} ({n_u} valeurs uniques)")
        else:
            if self.verbose:
                print("\nℹ Aucun identifiant à haute cardinalité détecté.")

        return self

    # =========================================================================
    # 8. Nettoyage Final (Élimination des Features Racine)
    # =========================================================================
    def ____8_Nettoyage_Final(self): pass

    def identifier_features_racine(
        self,
        suffixes_transformations: List[str] = None
    ) -> Dict[str, List[str]]:
        """
        Identifie les features "racine" qui ont été transformées.
        
        Paramètres :
        -----------
        suffixes_transformations : List[str], optional
            Liste des suffixes de transformation à chercher
            Par défaut : ['_log', '_wins', '_scaled', '_Encoded']
        
        Retourne :
        ---------
        Dict avec :
            - racines : Features originales ayant des transformations
            - transformees : Features transformées correspondantes
            - a_supprimer : Recommandations de suppression
        
        Example:
            >>> racines = engineer.identifier_features_racine()
            >>> print(f"Features racine : {racines['racines']}")
        """

        if suffixes_transformations is None:
            suffixes_transformations = ['_log', '_wins', '_scaled', '_Encoded']

        if self.verbose:
            print(f"\n{'='*70}")
            print("IDENTIFICATION DES FEATURES RACINE")
            print(f"{'='*70}")

        # Dictionnaire : racine -> [transformations]
        mapping = {}
        transformees = set()

        for col in self.df.columns:
            # Vérifier si la colonne a un suffixe de transformation
            for suffix in suffixes_transformations:
                if suffix in col:
                    # Extraire le nom racine
                    racine = col.replace(suffix, '')

                    # Si la racine existe dans le DataFrame
                    if racine in self.df.columns:
                        if racine not in mapping:
                            mapping[racine] = []
                        mapping[racine].append(col)
                        transformees.add(col)
                    break

        # Préparer résultats
        racines = list(mapping.keys())
        a_supprimer = []

        for racine, transformations in mapping.items():
            a_supprimer.append({
                'racine': racine,
                'transformations': transformations,
                'n_transformations': len(transformations),
                'raison': f'Remplacée par {len(transformations)} transformation(s)'
            })

        resultado = {
            'racines': racines,
            'transformees': list(transformees),
            'mapping': mapping,
            'a_supprimer': a_supprimer,
            'stats': {
                'n_racines': len(racines),
                'n_transformees': len(transformees),
                'suffixes_cherches': suffixes_transformations
            }
        }

        if self.verbose:
            print(f"\nSuffixes cherchés      : {', '.join(suffixes_transformations)}")
            print(f"Features racine trouvées : {len(racines)}")
            print(f"Features transformées    : {len(transformees)}")

            if racines:
                print(f"\n{'─'*70}")
                print("FEATURES RACINE ET LEURS TRANSFORMATIONS")
                print(f"{'─'*70}")

                for idx, (racine, transformations) in enumerate(mapping.items(), 1):
                    print(f"\n{idx}. {racine}")
                    print(f"   → Transformations ({len(transformations)}) :")
                    for transf in transformations:
                        print(f"      • {transf}")

            print(f"\n{'='*70}\n")

        return resultado

    def supprimer_features_racines(self, colonnes_a_eliminer: list[str]) -> 'DataCleaner':
        """
        Supprime les colonnes racines identifiées comme redondantes.
        
        Args:
            colonnes_a_eliminer: La liste 'existantes' que acabas de validar.
        """
        if not colonnes_a_eliminer:
            if self.verbose:
                print("\nℹ️ Aucune racine à supprimer.")
            return self

        # Reutilizamos tu lógica de eliminación
        self.df.drop(columns=colonnes_a_eliminer, inplace=True)
        self.columns_suppressed.extend(colonnes_a_eliminer)

        # Tracking para el reporte de la Etapa 9
        self.history.append({
            'operation': 'suppression_racines',
            'colonnes': colonnes_a_eliminer,
            'nb_colonnes': len(colonnes_a_eliminer)
        })

        if self.verbose:
            print("\n✅ ÉTAPE 9 : ÉLIMINATION DE LA REDONDANCE")
            print(f"   {len(colonnes_a_eliminer)} racines supprimées : {colonnes_a_eliminer}")

        return self

    # =========================================================================
    # GENERER RAPPORT
    # =========================================================================
    def ____GENERER_RAPPORT(self): pass

    """
    Método genérico para generar reportes de cualquier etapa del preprocessing.
    Añade este código al final de la clase DataCleaner, en la sección ____GENERER_RAPPORT.
    """

    def generer_rapport_etapes(
        self,
        etapes: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Genera un rapport consolidé para una o varias étapes du preprocessing.
        
        MÉTODO GENÉRICO Y EXTENSIBLE que reemplaza los métodos específicos.
        
        Args:
            etapes: Lista de números de étapes [1, 2, 3, ...]. Si None, genera todas.
            
        Returns:
            DataFrame con el résumé des opérations
            
        Examples:
            >>> # Une seule étape
            >>> rapport = cleaner.generer_rapport_etapes([2])
            
            >>> # Plusieurs étapes
            >>> rapport = cleaner.generer_rapport_etapes([1, 2, 3])
            
            >>> # Toutes les étapes
            >>> rapport = cleaner.generer_rapport_etapes()
        """

        # =========================================================================
        # MAPEO DE OPERACIONES POR ÉTAPE
        # =========================================================================
        # 🔧 EXTENSIBLE: Añadir aquí nuevas etapas (4, 5, 6...)
        operations_map = {
            1: [
                'suppression_constantes',
                'suppression_manquants',
                'suppression_specifique'
            ],
            2: [
                'creation_indicateurs_missing',
                'imputation_categoriques',
                'imputation_numeriques'
            ],
            3: [
                'transformation_logarithmique',
                'winsorisation'
            ],
            4: ['ajouter_feature' ],
            5: ['one_hot_encoding', 'target_encoding'] ,
            6: ['eliminer_correlees' ],
            7: ['suppression_haute_cardinalite' ],
            8: ['suppression_racines' ]

            # 4: ['encodage_categoriel', 'encodage_ordinal'],  # Futuro
            # 5: ['creation_interactions', 'binning'],         # Futuro
        }

        # Determinar qué étapes procesar
        if etapes is None:
            # Todas las étapes disponibles
            etapes_a_procesar = sorted(operations_map.keys())
        else:
            # Validar étapes solicitadas
            etapes_invalidas = [e for e in etapes if e not in operations_map]
            if etapes_invalidas:
                print(f"⚠️  Étapes no disponibles: {etapes_invalidas}")
            etapes_a_procesar = [e for e in etapes if e in operations_map]

        if not etapes_a_procesar:
            print("⚠️  Aucune étape valide spécifiée")
            return pd.DataFrame()

        # =========================================================================
        # EXTRACCIÓN Y FORMATEO DE DATOS
        # =========================================================================
        rapport_data = []

        for num_etape in etapes_a_procesar:
            operations_etape = operations_map[num_etape]

            print(f" num_etape: {num_etape}")

            # Filtrar operaciones de esta étape en el historial
            ops_historique = [
                op for op in self.history
                if op['operation'] in operations_etape
            ]

            # Procesar cada operación según su tipo
            for op in ops_historique:

                # =================================================================
                # ÉTAPE 1: NETTOYAGE INITIAL
                # =================================================================
                if num_etape == 1:
                    if op['operation'] == 'suppression_constantes':
                        rapport_data.append({
                            'Étape': '1.1 - Constantes',
                            'Operation': 'Suppression',
                            'Nb_Actions': op['nb_colonnes'],
                            'Details': ', '.join(op['colonnes']) if op['colonnes'] else 'Aucune'
                        })

                    elif op['operation'] == 'suppression_manquants':
                        rapport_data.append({
                            'Étape': f"1.2 - Manquants >{op['seuil']*100:.0f}%",
                            'Operation': 'Suppression',
                            'Nb_Actions': op['nb_colonnes'],
                            'Details': ', '.join(op['colonnes']) if op['colonnes'] else 'Aucune'
                        })

                    elif op['operation'] == 'suppression_specifique':
                        rapport_data.append({
                            'Étape': '1.3 - Spécifiques',
                            'Operation': 'Suppression',
                            'Nb_Actions': op['nb_colonnes'],
                            'Details': ', '.join(op['colonnes']) if op['colonnes'] else 'Aucune'
                        })

                # =================================================================
                # ÉTAPE 2: GESTION DES VALEURS MANQUANTES
                # =================================================================
                elif num_etape == 2:
                    if op['operation'] == 'creation_indicateurs_missing':
                        # Indicateurs numériques
                        rapport_data.append({
                            'Étape': '2.1 - Indicateurs',
                            'Operation': 'Création (Numériques)',
                            'Nb_Actions': op['nb_indicateurs_num'],
                            'Details': ', '.join(op['indicateurs_numeriques'])
                                       if op['indicateurs_numeriques'] else 'Aucun'
                        })

                        # Indicateurs catégoriels
                        rapport_data.append({
                            'Étape': '2.1 - Indicateurs',
                            'Operation': 'Création (Catégorielles)',
                            'Nb_Actions': op['nb_indicateurs_cat'],
                            'Details': ', '.join(op['indicateurs_categoriques'])
                                       if op['indicateurs_categoriques'] else 'Aucun'
                        })

                    elif op['operation'] == 'imputation_categoriques':
                        total_imputations = sum(
                            detail['nb_values_imputees']
                            for detail in op['details']
                        )
                        rapport_data.append({
                            'Étape': '2.2 - Imputation Cat.',
                            'Operation': f"Valeur: '{op['valeur_defaut']}'",
                            'Nb_Actions': f"{op['nb_colonnes_imputees']} cols, {total_imputations} vals",
                            'Details': ', '.join(op['colonnes'])
                        })

                    elif op['operation'] == 'imputation_numeriques':
                        total_imputations = sum(
                            detail['nb_values_imputees']
                            for detail in op['details']
                        )
                        rapport_data.append({
                            'Étape': '2.3 - Imputation Num.',
                            'Operation': f"Stratégie: {op['strategie']}",
                            'Nb_Actions': f"{op['nb_colonnes_imputees']} cols, {total_imputations} vals",
                            'Details': ', '.join(op['colonnes'])
                        })

                # =================================================================
                # ÉTAPE 3: TRAITEMENT ASYMÉTRIE ET OUTLIERS
                # =================================================================
                elif num_etape == 3:
                    if op['operation'] == 'transformation_logarithmique':
                        mode = 'Auto-détection' if op['auto_detect'] else 'Manuel'
                        threshold_info = f"(|skew| > {op['skew_threshold']})" if op['auto_detect'] else ''

                        rapport_data.append({
                            'Étape': f"3.1 - Log Transform {threshold_info}",
                            'Operation': mode,
                            'Nb_Actions': op['nb_transformations'],
                            'Details': ', '.join(op['colonnes_originales'])
                        })

                    elif op['operation'] == 'winsorisation':
                        percentiles = f"{op['percentile_bas']*100:.0f}%-{op['percentile_haut']*100:.0f}%"
                        mode = 'Inplace' if op['inplace'] else 'Nouvelles cols'

                        total_cappe = sum(
                            detail['nb_total_cappe']
                            for detail in op['details']
                        )

                        rapport_data.append({
                            'Étape': f"3.2 - Winsorisation ({percentiles})",
                            'Operation': mode,
                            'Nb_Actions': f"{op['nb_colonnes_traitees']} cols, {total_cappe} vals cappées",
                            'Details': ', '.join(op['colonnes'])
                        })


                # =================================================================
                # 4. Création de Nouvelles Features (Injection manuelle)
                # =================================================================
                elif num_etape == 4:
                    # --- Cas A: Variables spécifiques (Ratios, indicateurs) ---
                    if op['operation'] == 'ajouter_feature':
                        rapport_data.append({
                            'Étape': '4.3 - Feature Injection',
                            'Operation': f"Ajout: {op.get('feature')}",
                            'Nb_Actions': 1,
                            'Details': f"{op.get('description')} (Dtype: {op.get('dtype')})"
                        })

                # =================================================================
                # 5. Encodage Catégoriel (OHE & Target Encoding)
                # =================================================================
                elif num_etape == 5:
                    # Caso A: One-Hot Encoding
                    if op['operation'] == 'one_hot_encoding':
                        colonnes_source = ", ".join(op['colonnes'])
                        rapport_data.append({
                            'Étape': '5.1 - OHE',
                            'Operation': "One-Hot Encoding",
                            'Nb_Actions': op['n_colonnes_encodees'],
                            'Details': (f"Variables: {colonnes_source} | "
                                        f"Création de {op['n_colonnes_creees']} colonnes binarias "
                                        f"(drop_first={op['drop_first']})")
                        })

                    # Caso B: Target Encoding
                    elif op['operation'] == 'target_encoding':
                        rapport_data.append({
                            'Étape': '5.2 - Target Enc.',
                            'Operation': f"Encoding: {op['colonne']}",
                            'Nb_Actions': 1,
                            'Details': (f"Target: {op['target']} | Folds: {op['n_folds']} | "
                                        f"Smoothing: {op['smoothing']} | "
                                        f"Categories: {op['n_categories']}")
                        })

                # =================================================================
                # 6. Reduction_de_la_Redondance
                # =================================================================
                elif num_etape == 6:
                    if op['operation'] == 'eliminer_correlees':
                        liste_cols = ", ".join(op.get('colonnes', []))

                        rapport_data.append({
                            'Étape': '6.2. Élimination des Corréliées',
                            'Operation': "Réduction de Redondance",
                            'Nb_Actions': op.get('n_colonnes_supprimees', 0),
                            'Details': (f"Raison: {op.get('raison')} | "
                                        f"Colonnes: {liste_cols} | "
                                        f"Shape: {op.get('shape_avant')} ➔ {op.get('shape_apres')}")
                        })

                # =================================================================
                # 7. Standardization
                # =================================================================
                elif num_etape == 7:
                    # --- Étape 7 : Encodage et Nettoyage de Cardinalité ---
                    if op['operation'] == 'suppression_haute_cardinalite':
                        rapport_data.append({
                            'Étape': '7.1 - Data Purge',
                            'Operation': 'Suppression Identifiants',
                            'Nb_Actions': op.get('nb_colonnes', 0),
                            'Details': (f"Suppression des colonnes avec ratio de cardinalité > {op.get('seuil_utilise')*100}%. "
                                        f"Colonnes retirées : {', '.join(op.get('colonnes', []))}")
                        })
                # =================================================================
                # 8. Reduction_de_la_Redondance
                # =================================================================
                elif num_etape == 8:
                    if op['operation'] == 'suppression_racines':
                        rapport_data.append({
                            'Étape': '9.1 - Redundancy Purge',
                            'Operation': 'Suppression des Racines',
                            'Nb_Actions': op.get('nb_colonnes', 0),
                            'Details': (f"Élimination des variables originales après vérification de l'existence "
                                        f"de leurs versions transformées. Colonnes : {', '.join(op.get('colonnes', []))}")
                        })

                # =================================================================
                # 🔧 EXTENSIBLE: Añadir aquí nuevas etapas
                # =================================================================
                # elif num_etape == 4:
                #     if op['operation'] == 'encodage_categoriel':
                #         rapport_data.append({...})

        # =========================================================================
        # CONSTRUCCIÓN DEL DATAFRAME FINAL
        # =========================================================================
        if not rapport_data:
            print(f"⚠️  Aucune opération trouvée pour les étapes: {etapes_a_procesar}")
            return pd.DataFrame()

        rapport_df = pd.DataFrame(rapport_data)

        # Ordenar por étape
        rapport_df['Etape_Num'] = rapport_df['Étape'].str.extract(r'^(\d+)')[0].astype(int)
        rapport_df = rapport_df.sort_values(['Etape_Num', 'Étape']).drop(columns='Etape_Num')
        rapport_df = rapport_df.reset_index(drop=True)

        return rapport_df


    # =============================================================================
    # COMPATIBILIDAD CON MÉTODO ANTIGUO
    # =============================================================================
    def generer_rapport_etape2(self) -> pd.DataFrame:
        """
        [DEPRECATED] Usa generer_rapport_etapes([2]) en su lugar.
        
        Método mantenido para compatibilidad con código existente.
        Internamente llama al nuevo método genérico.
        """
        return self.generer_rapport_etapes([2])


    from datetime import datetime
    import os

    def save_stage(self, stage_name: str) -> str:
        """
        Sauvegarde l'état actuel du DataFrame avec métadonnées (heure et shape).
        
        Paramètres:
        -----------
        stage_name : str
            Nom de l'étape (ex: "etape_6")
            
        Retourne:
        ---------
        str : Le message de confirmation avec les détails de la sauvegarde.
        """
        # 1. Génération des métadonnées (Style Yann LeCun)
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M")
        rows, cols = self.df.shape

        # 2. Construction du nom de fichier intelligent
        file_name = f"dataset_{timestamp}_{stage_name}_{rows}x{cols}.csv"

        # 3. Sauvegarde physique
        self.df.to_csv(file_name, index=False)

        # 4. Création du message de log
        log_msg = f"✓ [SAVE] Stage: {stage_name} | Fichier: {file_name} | Shape: ({rows}, {cols})"

        # Optionnel: On peut aussi garder trace de cette sauvegarde dans l'historique
        self.history.append({
            'operation': 'sauvegarde_checkpoint',
            'stage': stage_name,
            'file': file_name,
            'shape': (rows, cols),
            'time': timestamp
        })

        return log_msg

    # --- Utilisation dans le Notebook ---
    # info_sauvegarde = cleaner.save_stage("etape_6")
    # print(info_sauvegarde)

    import pandas as pd
    import numpy as np

    # En data_preprocessing.py, dentro de la clase DataCleaner

    def detecter_colonnes_problematiques_v1(self, df, tolerance_mu=0.01, tolerance_sigma=0.01):
        """
        Identifie les colonnes qui ne respectent pas la standardisation.
        """
        rapport_erreurs = []

        # Correction : on utilise le 'df' passé en argument, pas 'self'
        cols_numeriques = df.select_dtypes(include=[np.number]).columns

        # On filtre pour ne garder que les colonnes non-binaires
        cols_a_verifier = [c for c in cols_numeriques if not df[c].isin([0, 1, np.nan]).all()]

        for col in cols_a_verifier:
            mu = df[col].mean()
            sigma = df[col].std()

            hors_mu = abs(mu) > tolerance_mu
            hors_sigma = abs(sigma - 1) > tolerance_sigma

            raison = 'Moyenne décalée' if hors_mu else 'Échelle incorrecte'
            if hors_mu or hors_sigma:
                rapport_erreurs.append({
                    'colonne': col,
                    'moyenne (μ)': round(mu, 4),
                    'ecart_type (σ)': round(sigma, 4),
                    'raison': raison
                })

        return pd.DataFrame(rapport_erreurs)

    def detecter_colonnes_problematiques(self, df, tolerance_mu=0.01, tolerance_sigma=0.01):
        """
        Identifie et explique les écarts de standardisation (μ=0, σ=1).
        """
        rapport_erreurs = []
        cols_numeriques = df.select_dtypes(include=[np.number]).columns

        # Filtre variables non-binaires
        cols_a_test    = [c for c in cols_numeriques
                          if not df[c].isin([0, 1, np.nan]).all()]

        for col in cols_a_test:
            mu         = df[col].mean()
            sigma      = df[col].std()

            hors_mu    = abs(mu) > tolerance_mu
            # Si sigma est quasi nul, c'est une variance nulle (Alerte Critique)
            variance_nulle = sigma < 1e-6
            hors_sigma     = abs(sigma - 1) > tolerance_sigma

            if hors_mu or hors_sigma or variance_nulle:
                raisons = []

                # Diagnostic de la Moyenne
                if hors_mu:
                    raisons.append(f"Moyenne décalée (μ={mu:.4f})")

                # Diagnostic de l'Échelle / Variance
                if variance_nulle:
                    raisons.append("CRITIQUE: Variance nulle (σ=0)")
                elif hors_sigma:
                    raisons.append(f"Échelle incorrecte (σ={sigma:.4f})")

                rapport_erreurs.append({
                    'colonne'       : col,
                    'moyenne (μ)'   : round(mu, 4),
                    'ecart_type (σ)': round(sigma, 4),
                    'diagnostic'    : " | ".join(raisons)
                })

        return pd.DataFrame(rapport_erreurs)

    # ============================================================================
    # EXEMPLE 8: INTÉGRATION DANS UN PIPELINE
    # ============================================================================
    def ____PIPELINE(self): pass

    def pipeline_preparation_initiale(self, df, target='', cols_to_drop=None):
        """
        Réalise le nettoyage structurel, élimine les valeurs cibles manquantes 
        et affiche chaque colonne supprimée.
        """
        df_work          = df.copy()

        # Initialisation de la liste si vide
        if cols_to_drop is None:
            cols_to_drop = []

        # 0.1 Élimination critique : Lignes sans valeur cible (Target)
        # --------------------------------------------------------------------------
        if target in df_work.columns:
            nb_avant     = len(df_work)
            df_work      = df_work.dropna(subset=[target])
            nb_apres     = len(df_work)

            if nb_avant != nb_apres:
                print(f"⚠️ Nettoyage Target...: {nb_avant - nb_apres} lignes supprimées (NaN dans {target})")
        else:
            print(f"❌ Erreur : La cible '{target}' est absente du DataFrame.")
            return None, None

        # 0.2 Suppression et Print des colonnes
        # --------------------------------------------------------------------------
        print("\n--- Nettoyage des colonnes ---")
        for col in cols_to_drop:
            if col in df_work.columns:
                print(f"🗑️ Suppression de : {col}")
                df_work  = df_work.drop(columns=[col])
            else:
                print(f"ℹ️ Colonne ignorée (déjà absente) : {col}")

        # 0.3 Séparation des caractéristiques et de la cible
        # --------------------------------------------------------------------------
        y                = df_work[target]
        X                = df_work.drop(columns=[target])

        # --- Vérification finale des axes ---
        print("-" * 60)
        print(f"🎯 CIBLE (y)      : {y.name}")
        print(f"🧬 FEATURES (X)   : {X.columns.tolist()}")
        print("-" * 60)

        print(f"\n✅ Préparation terminée : {X.shape[1]} features restantes.\n")

        return X, y

    def _fit_transformation_logarithmique(
        self,
        df,
        auto_detect: bool = True,
        skew_threshold: float = 1.0,
        colonnes: Optional[List[str]] = None
    ):
        """
        FIT: Identifier les colonnes à transformer et calculer les décalages.
        
        Ce qui est APPRIS et SAUVEGARDÉ:
        - self.COLS_TO_LOG : Liste des colonnes à transformer
        - self.log_decalages : Dict {colonne: valeur_decalage}
        """
        types_cols = self._identifier_colonnes_par_type()
        num_cols = [c for c in types_cols['numeriques'] if '_Manquant' not in c]

        # ──────────────────────────────────────────────────────────────────────
        # APPRENTISSAGE 1 : Quelles colonnes transformer ?
        # ──────────────────────────────────────────────────────────────────────
        if auto_detect and colonnes is None:
            # Auto-détection basée sur le skewness du TRAIN
            self.COLS_TO_LOG = []

            for col in num_cols:
                if col in df.columns:
                    skew_val = df[col].skew()
                    if abs(skew_val) > skew_threshold:
                        self.COLS_TO_LOG.append(col)

        elif colonnes is not None:
            # Utiliser les colonnes spécifiées
            self.COLS_TO_LOG = [c for c in colonnes if c in num_cols]

        else:
            self.COLS_TO_LOG = []

        # ──────────────────────────────────────────────────────────────────────
        # APPRENTISSAGE 2 : Calculer les décalages pour valeurs négatives
        # ──────────────────────────────────────────────────────────────────────
        self.log_decalages = {}  # NOUVEAU : Stocker les décalages

        for col in self.COLS_TO_LOG:
            if col in df.columns:
                min_val = df[col].min()

                if min_val < 0:
                    # Calculer le décalage sur le TRAIN uniquement
                    decalage = abs(min_val) + 1
                    self.log_decalages[col] = decalage
                else:
                    # Pas de décalage nécessaire
                    self.log_decalages[col] = 0

        if self.verbose:
            print("\n" + "-"*80)
            print(" FIT ÉTAPE 3.1 : TRANSFORMATION LOGARITHMIQUE")
            print("-"*80)
            print(f"Colonnes identifiées.....: {len(self.COLS_TO_LOG)}")

            for col in self.COLS_TO_LOG:
                decalage = self.log_decalages.get(col, 0)
                skew = df[col].skew()

                if decalage > 0:
                    print(f"  • {col:<40} (skew={skew:>6.2f}) → Décalage: +{decalage:.2f}")
                else:
                    print(f"  • {col:<40} (skew={skew:>6.2f})")
            print("="*80)

    # ──────────────────────────────────────────────────────────────────────────
    # FIT ÉTAPE 3.2 : CALCUL DES BORNES DE WINSORISATION
    # ──────────────────────────────────────────────────────────────────────────
    # ✅ APPRENDRE : Les percentiles 1% et 99% de chaque colonne (sur TRAIN)

        # ──────────────────────────────────────────────────────────────────────────
    # TRANSFORM ÉTAPE 3.1 : APPLICATION DE LA TRANSFORMATION LOGARITHMIQUE
    # ──────────────────────────────────────────────────────────────────────────
    # ✅ APPLIQUER : Les transformations log avec les décalages appris

    def _transform_logarithmique(self, df_clean):
        """
        TRANSFORM: Appliquer la transformation log avec les paramètres du fit.
        
        Utilise:
        - self.COLS_TO_LOG : Colonnes identifiées pendant fit()
        - self.log_decalages : Décalages calculés pendant fit()
        """
        if not hasattr(self, 'COLS_TO_LOG') or not self.COLS_TO_LOG:
            if self.verbose:
                print("\n⚠️  Aucune transformation logarithmique à appliquer")
            return df_clean

        for col in self.COLS_TO_LOG:
            if col in df_clean.columns:
                # Récupérer le décalage appris pendant fit()
                decalage = self.log_decalages.get(col, 0)

                if decalage > 0:
                    # Appliquer le MÊME décalage que sur le train
                    df_clean[f'log_{col}'] = np.log1p(df_clean[col] + decalage)

                    if self.verbose:
                        print(f"  log_{col:<38} (décalage: +{decalage:.2f})")
                else:
                    # Transformation directe
                    df_clean[f'log_{col}'] = np.log1p(df_clean[col])

                    if self.verbose:
                        print(f"  log_{col:<38}")

                # Sécurité : Gérer les NaN résiduels
                if df_clean[f'log_{col}'].isna().any():
                    # df_clean[f'log_{col}'].fillna(0, inplace=True) # deprecated !
                    df_clean[f'log_{col}'] = df_clean[f'log_{col}'].fillna(0)

        return df_clean



    def _fit_winsorisation(
        self,
        df,
        colonnes: Optional[List[str]] = None,
        percentile_bas: float = 0.01,
        percentile_haut: float = 0.99
    ):
        """
        FIT: Calculer les bornes de winsorisation depuis le dataset d'entraînement.
        
        Ce qui est APPRIS et SAUVEGARDÉ:
        - self.learned_winsor : Dict {colonne: (borne_min, borne_max)}
        """
        types_cols = self._identifier_colonnes_par_type()
        num_cols = types_cols['numeriques']

        # Déterminer quelles colonnes winsoriser
        if colonnes is None:
            # Par défaut : toutes les colonnes numériques
            cols_a_winsoriser = num_cols
        else:
            cols_a_winsoriser = [c for c in colonnes if c in num_cols]

        # ──────────────────────────────────────────────────────────────────────
        # APPRENTISSAGE : Calculer les bornes depuis le TRAIN uniquement
        # ──────────────────────────────────────────────────────────────────────
        self.learned_winsor = {}

        for col in cols_a_winsoriser:
            if col in df.columns:
                # Calculer les percentiles sur le TRAIN
                borne_min = df[col].quantile(percentile_bas)
                borne_max = df[col].quantile(percentile_haut)

                # Sauvegarder pour réutilisation dans transform()
                self.learned_winsor[col] = (borne_min, borne_max)

        if self.verbose:
            print("\n" + "-"*80)
            print(" FIT ÉTAPE 3.2 : WINSORISATION")
            print("-"*80)
            print(f"Percentiles..............: [{percentile_bas:.2%}, {percentile_haut:.2%}]")
            print(f"Colonnes traitées........: {len(self.learned_winsor)}")

            for col, (mini, maxi) in self.learned_winsor.items():
                print(f"  • {col:<40} → [{mini:>10.2f}, {maxi:>10.2f}]")
            print("="*80)


    # ──────────────────────────────────────────────────────────────────────────
    # TRANSFORM ÉTAPE 3.2 : APPLICATION DE LA WINSORISATION
    # ──────────────────────────────────────────────────────────────────────────
    # ✅ APPLIQUER : Capping avec les bornes apprises

    def _transform_winsorisation(self, df_clean):
        """
        TRANSFORM: Appliquer la winsorisation avec les bornes du fit.
        
        Utilise:
        - self.learned_winsor : Dict {colonne: (min, max)} calculé pendant fit()
        """
        if not hasattr(self, 'learned_winsor') or not self.learned_winsor:
            if self.verbose:
                print("\n⚠️  Aucune winsorisation à appliquer")
            return df_clean

        for col, (mini, maxi) in self.learned_winsor.items():
            if col in df_clean.columns:
                # Appliquer les MÊMES bornes que sur le train
                df_clean[col] = df_clean[col].clip(lower=mini, upper=maxi)

                if self.verbose:
                    n_clipped_low = (df_clean[col] == mini).sum()
                    n_clipped_high = (df_clean[col] == maxi).sum()

                    if n_clipped_low > 0 or n_clipped_high > 0:
                        print(f"  {col:<40} → {n_clipped_low:>4} bas, {n_clipped_high:>4} haut")

        return df_clean

    def pipeline_preprocessing_avec_rapport(df):
        """
        Pipeline complet avec génération automatique de rapports.

        1. Nettoyage des constantes et des identifiants (IDs) : Suppression des variables sans pouvoir prédictif.
        2. Imputation (Apprentissage des modes et médianes) : Stratégie de gestion des données manquantes pour garantir l'intégrité du dataset.
        3. Analyse de l'asymétrie (Skewness) pour les transformations logarithmiques : Identification des variables nécessitant une normalisation de leur distribution.
        4. Apprentissage géométrique (BallTree) et moyennes de référence : Extraction de caractéristiques spatiales et calcul des bases de comparaison pour l'ingénierie de variables (Feature Engineering).
        5. Codage des variables catégorielles (OHE et Target Encoding) : Transformation des données textuelles en vecteurs numériques exploitables par les algorithmes.
        6. Réduction de la redondance et de la colinéarité : Élimination des variables fortement corrélées pour stabiliser le modèle et éviter le surapprentissage (Overfitting).
        7. Mise à l'échelle (Apprentissage du Z-score) : Standardisation finale pour obtenir une moyenne de 0 et un écart-type de 1 sur l'ensemble des caractéristiques.

        """
        cleaner = DataCleaner(df, verbose=True)

        # Étape 1
        cleaner.nettoyage_initial_complet(
            missing_threshold=0.95,
            colonnes_specifiques=['Comments', 'Outlier']
        )
        rapport_1 = cleaner.generer_rapport_etapes([1])
        print("\n--- Rapport Étape 1 ---")
        print(rapport_1.to_string(index=False))

        # Étape 2
        cleaner.gestion_valeurs_manquantes_complete(
            threshold_indicateurs=0.50,
            valeur_cat_defaut='INCONNU',
            strategie_num='mediane'
        )
        rapport_2 = cleaner.generer_rapport_etapes([2])
        print("\n--- Rapport Étape 2 ---")
        print(rapport_2.to_string(index=False))

        # Étape 3
        cleaner.traitement_asymetrie_complet(
            log_auto_detect=True,
            log_skew_threshold=1.0,
            wins_percentiles=(0.01, 0.99)
        )
        rapport_3 = cleaner.generer_rapport_etapes([3])
        print("\n--- Rapport Étape 3 ---")
        print(rapport_3.to_string(index=False))

        # Rapport final
        rapport_complet = cleaner.generer_rapport_etapes()
        rapport_complet.to_csv('rapport_final.csv', index=False)

        return cleaner.get_dataframe(), rapport_complet



    # ==============================================================================
    # ESTRUCTURA DE MÉTODOS
    # ==============================================================================

    def fit_target_encode(self, df, columns, target, n_folds=5, smoothing=10.0) -> pd.DataFrame:
        """
        APRENDE los mapeos usando Cross-Validation para evitar Overfitting.
        """
        df_copy = df.copy()
        for col in columns:
            # 1. Calculamos la media global del target para el suavizado (smoothing)
            global_mean = df_copy[target].mean()
            self.global_means[col] = global_mean

            # 2. Lógica de Cross-Validation para el encoding (Evita que el dato 'i' se vea a sí mismo)
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            # Creamos una serie temporal para guardar los valores calculados por folds
            col_encoded = pd.Series(index=df_copy.index, dtype=float)

            for train_idx, val_idx in kf.split(df_copy):
                # Medias calculadas sobre el 'train_fold'
                fold_mean = df_copy.iloc[train_idx].groupby(col)[target].mean()
                # Aplicamos al 'val_fold'
                col_encoded.iloc[val_idx] = df_copy.iloc[val_idx][col].map(fold_mean)

            # 3. Guardamos el mapeo FINAL (usando todo el set de train) para el futuro (Transform)
            # Aplicamos suavizado: (n * mean + smoothing * global_mean) / (n + smoothing)
            agg = df_copy.groupby(col)[target].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)

            self.mappings[col] = smooth.to_dict()

        print(f"✅ [Fit] Mapeos aprendidos para: {list(self.mappings.keys())}")

        return df_copy

    def transform_target_encode(self, df):
        """
        APLICA los mapeos guardados a cualquier DataFrame (Train o Test).
        """
        df_res = df.copy()
        for col, mapping in self.mappings.items():
            if col in df_res.columns:
                # Aplicamos el mapeo y llenamos nulos (categorías nuevas) con la media global aprendida
                df_res[f"{col}_Encoded"] = df_res[col].map(mapping).fillna(self.global_means[col])
                # Eliminamos la original como querías
                df_res.drop(columns=[col], inplace=True)
        return df_res


    def fit(self, df):
        """
        MÉTHODE FIT : PHASE D'APPRENTISSAGE ET DÉTECTION AUTOMATIQUE
        Analyse le dataset d'entraînement pour configurer les paramètres du pipeline.

        1. **Nettoyage** (Identifiants et Constantes).
        2. **Imputation** (Statistiques du train).
        3. **Distribution** (Logs et Winsorisation).
        4. **Ingénierie** (F1 à F13).
        5. **Codification** (OHE, Target Encoding et Alignement).
        6. **Réduction** (Suppression des redondances).
        7. **Audition** (Cardinalité en mode lecture seule).
        8. **Scaling** (Normalisation finale sans risque de division par zéro).
        
        """
        if self.verbose:
            print("\n" + "="*80)
            print(" INITIATION DU FIT : ANALYSE STATISTIQUE DU DATASET D'ENTRAÎNEMENT")
            print("="*80)

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" FIT ÉTAPE 0 : NORMALISATION INITIALE (Cruciale pour la cohérence) ")
            print("--------------------------------------------------------------------------------")

        df_fit = self.normaliser_categories(df)


        if self.verbose:
             print("********************************************************************************")
             print(" FIT ÉTAPE 1 : IDENTIFICATION STRUCTURELLE (NETTOYAGE INITIAL)")
             print("********************************************************************************")

        # Constantes. Identification
        res_var            = self.constant_columns_analysis(df_fit)
        self.COLS_CONSTANT = res_var['constant_cols']

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" FIT ÉTAPE 1.1 : Constantes. Supresion ")
            print("--------------------------------------------------------------------------------")
            print(f" F 1.1 Colonnes Constantes............: {self.COLS_CONSTANT }")

        # Constantes. SUPPRESION
        df_fit = self.supprimer_colonnes_specifiques(df_fit, self.COLS_CONSTANT)

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" FIT ÉTAPE 1.2 : Élimination des Manquants Excessifs")
            print("--------------------------------------------------------------------------------")
            print(f" F 1.2 Colonnes Missing Excessifs.....: {self.COLS_MISSING_EXCESIF } ")

        df_fit = self.supprimer_colonnes_specifiques(df_fit, self.COLS_MISSING_EXCESIF)

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" FIT ÉTAPE 1.3. Suppression des Identifiants")
            print("--------------------------------------------------------------------------------")
            print(f" F 1.3 Colonnes des Identifiants......: {self.COLS_IDENTIFIANTS }")

        df_fit = self.supprimer_colonnes_specifiques(df_fit, self.COLS_IDENTIFIANTS)


        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" FIT ÉTAPE supprimer_colonnes_par_motif 'Electricity', 'Gas', 'Steam', 'SiteEnergyUseWN(kBtu)' ")
            print("--------------------------------------------------------------------------------")

        # Eliminar columnas con target leakage
        df_fit = self.supprimer_colonnes_par_motif(
            df_fit,
            ['Electricity', 'Gas', 'Steam', 'SiteEnergyUseWN(kBtu)', 'SiteEUI', 'SourceEUI', 'TotalGHGEmissions' ]
        )

        if self.verbose:
            print("********************************************************************************")
            print(" FIT ÉTAPE 2 : PRÉPARATION DE L'IMPUTATION (GESTION DES MANQUANTS)")
            print("********************************************************************************")

        # Séparation des types pour les calculs statistiques après nettoyage virtuel
        self.COLS_NUMERICAL   = df_fit.select_dtypes(include=[np.number]).columns.tolist()
        self.COLS_CATEGORICAL = df_fit.select_dtypes(exclude=[np.number]).columns.tolist()

        # Imputation Catégorielle: Remplacer les manquants par 'INCONNU' ou 'AUCUN'.
        # Apprentissage des Modes pour l'imputation catégorielle
        for col in self.COLS_CATEGORICAL:
            self.learned_modes[col] = df_fit[col].mode()[0] if not df_fit[col].mode().empty else 'INCONNU'

        print(f"Total de catégories apprises : {len(self.learned_modes)}")

        if self.verbose and len(self.learned_modes) > 0:
            # Mostramos un pequeño ejemplo de lo aprendido
            exemple_cols = list(self.learned_modes.keys())[:5]
            print("Aperçu (top 5) :")
            for c in exemple_cols:
                print(f"  • {c}: {self.learned_modes[c]}")

        # cleaner_global.imputer_categoriques(valeur_defaut='INCONNU')

        # Imputation Numérique: Imputer les valeurs restantes avec la Médiane.
        # Calcul des Médianes pour l'imputation numérique (Robuste aux outliers)
        for col in self.COLS_NUMERICAL:
            self.learned_medians[col] = df_fit[col].median()

        print(f"Total de variables numériques apprises : {len(self.learned_medians)}")

        if self.verbose and len(self.learned_medians) > 0:
            # Mostramos un pequeño ejemplo de las medianas calculadas
            exemple_cols = list(self.learned_medians.keys())[:5]
            print("Aperçu (top 5) :")
            for c in exemple_cols:
                # Formateo a 2 decimales para claridad visual
                print(f"  • {c}: {self.learned_medians[c]:.2f}")

        # cleaner_global.imputer_numeriques(strategie='mediane', colonnes_specifiques = self.COLS_SPECIFIQUES)


        # --- CRUCIAL: Aplicar la imputación a la copia de ensayo ---

        # 1. Aplicar modas aprendidas a df_fit
        for col, value in self.learned_modes.items():
            df_fit[col] = df_fit[col].fillna(value) # [cite: 83]

        # 2. Aplicar medianas aprendidas a df_fit
        for col, value in self.learned_medians.items():
            df_fit[col] = df_fit[col].fillna(value) # [cite: 82]

        if self.verbose:
            print("✅ df_fit imputé (plus de valeurs manquantes pour les étapes suivantes)")


        if self.verbose:
            print("********************************************************************************")
            print(" FIT ÉTAPE 3 : ANALYSE DE L'ASYMÉTRIE ET DES OUTLIERS ")
            print("********************************************************************************")

        # Appeler les méthodes de fit pour l'asymétrie
        self._fit_transformation_logarithmique(
            df_fit,
            auto_detect=True,
            skew_threshold=1.0
        )

        # Appeler les méthodes de fit pour la winsorisation
        self._fit_winsorisation(
            df_fit,
            colonnes=None,  # None = toutes les colonnes numériques
            percentile_bas=0.01,
            percentile_haut=0.99
        )

        if self.verbose:
            print("********************************************************************************")
            print(" FIT ÉTAPE 4. 👤 Création de Nouvelles Features ")
            print("********************************************************************************")

        # --------------------------------------------------------------------------
        # FIT : 4.1 Distance_Centre_m Distance_Port_m
        # --------------------------------------------------------------------------
        # 100% dans trasform. (Calcul mathématique pur sans apprentissage)

        # --------------------------------------------------------------------------
        # FIT : 4.2 Distance_Centre_m Distance_Port_m
        # --------------------------------------------------------------------------
        # 100% dans trasform. (Idem 4.1)

        # --------------------------------------------------------------------------
        # FIT : 4.3 Densite_Voisinage_{rayon}m
        # --------------------------------------------------------------------------
        # Guardamos las coordenadas de entrenamiento como referencia global
        # Usamos radianes para que el cálculo de Haversine sea exacto en metros
        df_coords = df_fit[['Latitude', 'Longitude']].dropna()
        if not df_coords.empty:
            coords = np.radians(df_coords.values)
            self.reference_tree = BallTree(coords, metric='haversine')

        # --------------------------------------------------------------------------
        # FIT : 4.4 Taille_Batiment_Ordinale
        # --------------------------------------------------------------------------
        # 100% dans trasform. (Règle métier fixe)

        # --------------------------------------------------------------------------
        # FIT : 4.9. Indice d'efficacité relative (NOUVEAU - plus utile que GHG_Density)
        # --------------------------------------------------------------------------
        # Versión optimizada para evitar fugas de datos y asegurar interpretación
        # Aprendemos la media solo del set que recibimos (Train)

        #if 'SiteEUI(kBtu/sf)' in df_fit.columns:
        #    self.media_referencia_SiteEUI = df_fit['SiteEUI(kBtu/sf)'].mean()


        #if hasattr(self, 'media_referencia_SiteEUI'):
        #    print(f"✅ [Fit] Media aprendida: {self.media_referencia_SiteEUI:.2f}")


        # CRUCIAL : On simule la création des features pour que le Scaler les voie !
        df_fit = self._generer_features_ingenierie(df_fit)

        if self.verbose:
            print("********************************************************************************")
            print(" FIT ÉTAPE 5 : APPRENTISSAGE DES ENCODAGES (CODIFICATION) ")
            print("********************************************************************************")

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" FIT ÉTAPE 5.1. One-Hot Encoding: Appliquer aux variables à faible cardinalité.")
            print("--------------------------------------------------------------------------------")

        # On utilise df_fit pour identifier les colonnes catégorielles stables
        self.ohe_columns = self.identifier_candidates_ohe(df_fit, max_cardinality=12)



        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" FIT ÉTAPE 5.2. TARGET ENCODING (Neighborhood ")
            print("--------------------------------------------------------------------------------")

        # Llamada al sub-método de Target Encoding
        # Este método llenará self.mappings y self.global_means
        df_fit = self.fit_target_encode(
            df=df_fit,
            columns=['Neighborhood', 'PrimaryPropertyType'],
            target='SiteEnergyUse(kBtu)', # O tu target específico
            n_folds=5,
            smoothing=10.0
        )


        # ==========================================================================
        # CRUCIAL : SIMULATION DE L'ENCODAGE SUR df_fit
        # ==========================================================================
        # Pour que le Scaler (Étape 7) puisse voir et apprendre les moyennes
        # des nouvelles colonnes encodées.

        # A. ONE-HOT ENCODING
        if self.verbose:
            print(f"🔥 [Fit] Aplicando OHE a: {self.ohe_columns}")

        # DEBUG
        print(f"DEBUG: Type of df_fit is {type(df_fit)}")
        print(f"DEBUG: Columns to OHE: {self.ohe_columns}")

        # ACTION (Toujours exécutée !) :
        df_fit = pd.get_dummies(
            df_fit,
            columns=self.ohe_columns,
            drop_first=True,
            prefix_sep='_',
            dtype=int  # <--- CRUCIAL: Esto asegura que sean 0 y 1 numéricos
        )
        # SAUVEGARDE DU CONTRAT (Toujours exécutée !) :
        self.columns_metadata = df_fit.columns.tolist()

        # B. TARGET ENCODING
        if self.verbose:
            print("🚀 [Fit] Application du Target Encoding sur df_fit pour le futur Scaling...")

        # ACTION (Toujours exécutée !) :
        df_fit = self.transform_target_encode(df_fit)


        if self.verbose:
            print("********************************************************************************")
            print(" FIT ÉTAPE 6. Réduction de la Redondance ")
            print("********************************************************************************")

        # Ejecutas el pipeline de redundancia para APRENDER qué columnas sobran
        # On utilise df_fit pour l'analyse de corrélation

        # A. Filtrage des colonnes à variance nulle pour éviter la division par zéro (Warnings)
        # On ne garde que les colonnes numériques qui ont des valeurs différentes
        variances    = df_fit[self.COLS_NUMERICAL].var()
        cols_valides = variances[variances > 0].index.tolist()

        # B. Préparation du DataFrame pour l'analyse (Données valides + Target)
        target_col = 'SiteEUI(kBtu/sf)'
        if target_col in df_fit.columns:
            df_pour_analyse = df_fit[cols_valides + [target_col]]
        else:
            df_pour_analyse = df_fit[cols_valides]

        # 1. Asegurar nombres únicos en el DataFrame de análisis
        df_pour_analyse = df_pour_analyse.loc[:, ~df_pour_analyse.columns.duplicated()].copy()

        # 2. Asegurar que la lista de columnas no tenga nombres repetidos
        cols_finales = list(dict.fromkeys(df_pour_analyse.columns))
        df_pour_analyse = df_pour_analyse[cols_finales]

        # C. LLAMADA CORREGIDA
        # On passe le DataFrame filtré pour éviter les RuntimeWarnings de corrélation
        resultados = self.pipeline_reduction_redondance(
            df=df_pour_analyse,
            target=target_col,
            auto_eliminer=False
        )

        # D. Stockage des résultats
        self.COLS_REDUNDANTS = [c['colonne'] for c in resultados.get('candidats_suppression', [])]

        # --- CRUCIAL : Suppression physique de df_fit ---
        if self.COLS_REDUNDANTS:
            df_fit = df_fit.drop(columns=self.COLS_REDUNDANTS)
            if self.verbose:
                print(f"✅ [Fit] {len(self.COLS_REDUNDANTS)} colonnes redondantes retirées de df_fit.")

        # E. Mise à jour du "Contrat" de colonnes pour le Transform
        self.columns_metadata = df_fit.columns.tolist()

        if self.verbose:
            print(f" F 6 Colonnes REDONDANTES.............: {self.COLS_REDUNDANTS } ")


        if self.verbose:
            print("********************************************************************************")
            print(" FIT ÉTAPE 7 : CALCUL DES PARAMÈTRES DE MISE À L'ÉCHELLE (SCALING) ")
            print("********************************************************************************")

        # --------------------------------------------------------------------------
        # FIT 7.1. STANDARDISATION (Z-Score)
        # --------------------------------------------------------------------------

        print(f"DEBUG FIT: ¿Está 'BuildingType_NONRESIDENTIAL' en df_fit? {'BuildingType_NONRESIDENTIAL' in df_fit.columns}")
        print(f"DEBUG FIT: Número total de columnas: {len(df_fit.columns)}")

        # CAMBIO CLAVE: Escalar TODO lo que sea numérico en el df_fit actual
        # Esto incluye las nuevas features F1-F13, OHE y Target Encoded.
        cols_finales_numericas = df_fit.select_dtypes(include=['number', 'bool']).columns.tolist()

        for col in cols_finales_numericas:
            # Calculamos estadísticas
            m = df_fit[col].mean()
            s = df_fit[col].std()

            # Si la desviación es 0 (columna constante), usamos 1 para evitar div/0
            if s == 0 or np.isnan(s):
                s = 1.0

            self.learned_scaler[col] = {
                'mean': m,
                'std':  s
            }

        if self.verbose:
            print(f"✅ [Fit] Parámetros de escalado aprendidos para {len(self.learned_scaler)} columnas.")
            print("🚀 [Fit] PROCESO COMPLETO. El objeto FeatureEngineer está listo.")

        if self.verbose:
            print("********************************************************************************")
            print(f"✅ FIT Analyse terminée : {len(self.COLS_CONSTANT) + len(self.COLS_REDUNDANTS)} colonnes à supprimer.")
            print(f"✅ Statistiques apprises pour {len(self.COLS_NUMERICAL)} variables numériques.")
            print(f"✅ {len(self.COLS_TO_LOG)} colonnes asymétriques détectées pour transformation log.")
            print("********************************************************************************")

        return self


    def transform(self, df):
        """
        MÉTHODE TRANSFORM : APPLICATION ET EXÉCUTION DU PIPELINE
        Applique les transformations apprises lors du fit à n'importe quel dataset.
        """
        if self.verbose:
            print("\n" + "="*80)
            print(" EXÉCUTION DU TRANSFORM : TRAITEMENT SÉQUENTIEL DES DONNÉES")
            print("="*80)

        # Création d'une copie pour éviter de modifier le DataFrame original (Immutabilité)
        df_res = df.copy()

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 0 : NORMALISATION INITIALE (Cruciale pour la cohérence) ")
            print("--------------------------------------------------------------------------------")

        df_clean = self.normaliser_categories(df_res)

        if self.verbose:
             print("********************************************************************************")
             print(" TRANSFORM ETAPE 1 : LIMPIEZA ESTRUCTURAL (ELIMINACIÓN) ")
             print("********************************************************************************")

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 1.1 : Constantes. Supresion ")
            print("--------------------------------------------------------------------------------")
            print(f" F 1.1 Colonnes Constantes............: {self.COLS_CONSTANT }")

        df_clean = self.supprimer_colonnes_specifiques(df_clean, self.COLS_CONSTANT)

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 1.2 : Élimination des Manquants Excessifs")
            print("--------------------------------------------------------------------------------")
            print(f" F 1.2 Colonnes Missing Excessifs.....: {self.COLS_MISSING_EXCESIF } ")

        df_clean = self.supprimer_colonnes_specifiques(df_clean, self.COLS_MISSING_EXCESIF)

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 1.3. Suppression des Identifiants")
            print("--------------------------------------------------------------------------------")
            print(f" F 1.3 Colonnes des Identifiants......: {self.COLS_IDENTIFIANTS }")

        df_clean = self.supprimer_colonnes_specifiques(df_clean, self.COLS_IDENTIFIANTS)



        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE supprimer_colonnes_par_motif 'Electricity', 'Gas', 'Steam', 'SiteEnergyUseWN(kBtu)' ")
            print("--------------------------------------------------------------------------------")

        # Eliminar columnas con target leakage
        df_clean = self.supprimer_colonnes_par_motif(
            df_clean,
            ['Electricity', 'Gas', 'Steam', 'SiteEnergyUseWN(kBtu)', 'SiteEUI', 'SourceEUI', 'TotalGHGEmissions' ]
        )

        # --------------------------------------------------------------------------
        # 1.4. 👤 Gestion de la Redondance:
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # ⚠️ NOTE SUR LA REDONDANCE (1.4)
        # --------------------------------------------------------------------------
        # ATTENTION : On ne supprime pas encore self.COLS_REDUNDANTS ici.
        # Ces colonnes pourraient être nécessaires pour le Feature Engineering (Étape 4).
        # Elles seront supprimées à l'étape 8, juste avant le Scaling.

        if self.verbose:
            print("********************************************************************************")
            print(" TRANSFORM ÉTAPE 2 : IMPUTACIÓN (GESTION DES MANQUANTS) ")
            print("********************************************************************************")

        # 2.3. Remplissage des variables numériques avec les MÉDIANES apprises
        for col, value in self.learned_medians.items():
            if col in df_clean.columns:
                num_nan = df_clean[col].isna().sum()
                if num_nan > 0:
                    df_clean[col] = df_clean[col].fillna(value)
                    if self.verbose:
                        print(f"   🔹 [Imputation Num] {col.ljust(30)} : {num_nan} valeurs remplacées par {value:.2f} (médiane)")

        # 2.4. Remplissage des variables catégorielles avec les MODES appris
        for col, value in self.learned_modes.items():
            if col in df_clean.columns:
                num_nan = df_clean[col].isna().sum()
                if num_nan > 0:
                    df_clean[col] = df_clean[col].fillna(value)
                    if self.verbose:
                        print(f"   🔸 [Imputation Cat] {col.ljust(30)} : {num_nan} valeurs remplacées par '{value}' (mode)")

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" ✅ [Transform] Phase d'imputation terminée avec succès.")
            print("--------------------------------------------------------------------------------")

        if self.verbose:
            print("********************************************************************************")
            print(" TRANSFORM ÉTAPE 3 : TRAITEMENT DE L'ASYMÉTRIE ET OUTLIERS")
            print("********************************************************************************")

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 3.1 Transformation logarithmique ")
            print("--------------------------------------------------------------------------------")

        df_clean = self._transform_logarithmique(df_clean)

        if self.verbose:
            nb_cols_log = len([c for c in df_clean.columns if 'log_' in c])
            print(f"\n✅ Étape 3.1 terminée : {nb_cols_log} colonnes log créées")

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 3.2 Winsorisation ")
            print("--------------------------------------------------------------------------------")

        df_clean = self._transform_winsorisation(df_clean)


        if self.verbose:
            print("********************************************************************************")
            print(" TRANSFORM ÉTAPE 4. 👤 FEATURE ENGINEERING (ENRICHISSEMENT) ")
            print("********************************************************************************")

        # ÉTAPE 4 : Application de l'ingénierie (Source unique)
        df_clean = self._generer_features_ingenierie(df_clean)

        if self.verbose:
            print("✅ Étape 4 terminée : Features F1 à F13 générées.")

        if self.verbose:
            print("********************************************************************************")
            print(" TRANSFORM ETAPE 5 : CODIFICATION CATÉGORIELLE (ENCODING) ")
            print("********************************************************************************")

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 5.1. One-Hot Encoding: Appliquer aux variables à faible cardinalité. ")
            print("--------------------------------------------------------------------------------")

        # Transformation
        df_clean = self.apply_one_hot_encoding(
            df_clean,
            columns=self.ohe_columns,
            drop_first=True,  # Recommandé pour éviter la multicolinéarité
            prefix_sep='_'
        )

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 5.2. TARGET ENCODING ")
            print("--------------------------------------------------------------------------------")

        # Nous appelons la sous-méthode qui utilise les dictionnaires appris
        df_clean = self.transform_target_encode(df_clean)


        # C. ALIGNEMENT (Le "Contrat" de colonnes)
        # Aquí usamos el self.columns_metadata que guardamos en el Fit
        # Esto asegura que si falta una categoría en Test, se cree la columna con 0s
        df_clean = df_clean.reindex(columns=self.columns_metadata, fill_value=0)

        if self.verbose:
            print(f"✅ Étape 5 terminée : Encodage appliqué et colonnes alignées ({len(self.columns_metadata)} cols).")

        if self.verbose:
            print("********************************************************************************")
            print(" TRANSFORM ÉTAPE 6. Réduction de la Redondance ")
            print("********************************************************************************")
            # Usamos el nombre consistente: REDUNDANTS
            print(f" Colonnes redondantes à supprimer : {self.COLS_REDUNDANTS}")

        # Verificamos si la lista existe y no está vacía
        if hasattr(self, 'COLS_REDUNDANTS') and self.COLS_REDUNDANTS:
            # Usamos tu método de clase para mantener la uniformidad
            df_clean = self.supprimer_colonnes_specifiques(df_clean, self.COLS_REDUNDANTS)

            if self.verbose:
                print(f"✅ Étape 6 terminée : {len(self.COLS_REDUNDANTS)} colonnes supprimées.")

        if self.verbose:
            print("********************************************************************************")
            print(" TRANSFORM ÉTAPE 8 : Analyser Cardinalite ")
            print("********************************************************************************")

        # 1. Analyser et obtenir la table
        df_analyse = self.analyser_cardinalite(df_clean)

        # 2. Afficher le rapport
        self.afficher_rapport(df_analyse)

        """
        # 3. Afficher la table complète
        #print("\n📋 Table complète:")
        #print(df_analyse.to_string(index=False))
        
        # 5. Aplicar los tratamientos
        cols_supprimer  = self.obtenir_colonnes_par_action(df_analyse, 'SUPPRIMER')
        cols_onehot     = self.obtenir_colonnes_par_action(df_analyse, 'ONE_HOT')
        cols_target     = self.obtenir_colonnes_par_action(df_analyse, 'TARGET_ENCODING')
        
        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 8.1. SUPPRIMER ")
            print("--------------------------------------------------------------------------------") 
            print(f" Colonnes SUPPRIMER...........: {cols_supprimer }")
        
        df_clean = self.supprimer_colonnes_specifiques(df_clean, cols_supprimer)

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 8.2. One-Hot Encoding ")
            print("--------------------------------------------------------------------------------") 
            print(f" Colonnes ONE_HOT.............: {cols_onehot }")
                    
        # Transformation
        df_clean = self.apply_one_hot_encoding(
            df_clean,
            columns= cols_onehot,
            drop_first=True,  # Recommandé pour éviter la multicolinéarité
            prefix_sep='_'
        )

        if self.verbose:
            print("--------------------------------------------------------------------------------")
            print(" TRANSFORM ÉTAPE 8.3. TARGET ENCODING  ")
            print("--------------------------------------------------------------------------------") 
            print(f" Colonnes ONE_HOT.............: {cols_target }")
                    
        # Nous appelons la sous-méthode qui utilise les dictionnaires appris
        # TE.  Target Encoding (Haute cardinalité)

        df_clean.info()
        
        for col in cols_target:
            # 1. Verificamos si la columna original existe en el DataFrame
            if col in df_clean.columns:
                print(f"🔄 Procesando {col}...")
                
                # 2. Limpieza de strings (Upper + Strip)
                df_clean[col] = df_clean[col].astype(str).str.upper().str.strip()
                
                # 3. Aplicación del Target Encoding (esto eliminará la columna al final)
                df_clean = self.apply_target_encode_cv(
                    df_clean,
                    column=col, 
                    target=self.FEATURE_TARGET, 
                    n_folds=5, 
                    smoothing=10.0
                )
            else:
                # 4. Si no existe, comprobamos si ya está codificada
                encoded_name = f"{col}_Encoded"
                if encoded_name in df_clean.columns:
                    print(f"✅ {col} ya ha sido transformada en {encoded_name}.")
                else:
                    print(f"⚠️ {col} no encontrada y no parece estar codificada.")
        """

        if self.verbose:
            print("********************************************************************************")
            print(" TRANSFORM ÉTAPE 9 : ESTANDARIZACIÓN (SCALING) ")
            print("********************************************************************************")

        # MISE À L'ÉCHELLE FINALE : Standardisation (Z-Score)
        # Action : (df - mean) / std
        # Aplicamos el escalado usando los parámetros exactos aprendidos en el Fit
        # self.learned_scaler contiene las medias y std de TODAS las columnas finales

        # Identificamos qué columnas son realmente numéricas en este momento
        cols_numeriques = df_clean.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols_numeriques:
            # Usamos los valores aprendidos en el FIT
            if hasattr(self, 'learned_scaler') and col in self.learned_scaler:
                mu    = self.learned_scaler[col]['mean']
                sigma = self.learned_scaler[col]['std']

                # Aplicamos la transformación
                df_clean[col] = (df_clean[col] - mu) / sigma
            else:
                # 🎓 Si es una columna nueva (como las F1-F13) que no estaba en el Fit
                # Esto no debería pasar si el Fit analizó el DF ya enriquecido.
                if self.verbose:
                    print(f"⚠️ Colonne {col} non trouvée dans le scaler du Fit.")


        if self.verbose:
            print(f"✅ [Transform] {len(self.learned_scaler)} variables standardisées avec succès.")
            print("="*80)
            print(" 🏁 TRANSFORM TERMINÉ : LE DATASET EST PRÊT POUR L'INFERENCE")
            print("="*80)

        return df_clean



        if self.verbose:
            print("********************************************************************************")
            print(f"✅ Transformation terminée : Dataset final prêt pour le modèle ({df_clean.shape[0]} lignes, {df_clean.shape[1]} colonnes).")
            print("********************************************************************************")

        return df_clean


    def analyser_cardinalite(self, df: pd.DataFrame, seuil_identifiant: float = 50.0, seuil_onehot: int = 10) -> pd.DataFrame:
        """
        Analyse les colonnes catégorielles et recommande des actions.
        
        Args:
            df: DataFrame à analyser
            seuil_identifiant: Ratio (%) au-dessus duquel considérer comme identifiant
            seuil_onehot: Nombre max de valeurs uniques pour One-Hot Encoding
            
        Returns:
            DataFrame avec colonnes: Variable, Type, Valeurs_Uniques, Ratio_%, Exemples, Action, Raison
        """
        colonnes_cat = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if not colonnes_cat:
            return pd.DataFrame()

        resultats = []

        for col in colonnes_cat:
            n_unique = df[col].nunique()
            total = len(df)
            ratio = (n_unique / total) * 100

            # Exemples de valeurs
            exemples = df[col].dropna().unique()[:4]
            exemples_str = ', '.join([str(x) for x in exemples])

            # Déterminer l'action
            if ratio > seuil_identifiant:
                action = "SUPPRIMER"
                raison = "Identifiant unique"
            elif n_unique <= seuil_onehot:
                action = "ONE_HOT"
                raison = "Basse cardinalité"
            else:
                action = "TARGET_ENCODING"
                raison = "Haute cardinalité"

            resultats.append({
                'Variable': col,
                'Type': str(df[col].dtype),
                'Valeurs_Uniques': n_unique,
                'Ratio_%': round(ratio, 1),
                'Exemples': exemples_str,
                'Action': action,
                'Raison': raison
            })

        return pd.DataFrame(resultats)


    # ═══════════════════════════════════════════════════════════════════
    # FONCTIONS UTILITAIRES
    # ═══════════════════════════════════════════════════════════════════

    def afficher_rapport(self, df_analyse: pd.DataFrame) -> None:
        """Affiche un rapport formaté"""
        if df_analyse.empty:
            print("✅ Aucune colonne catégorielle trouvée.")
            return

        print(f"\n{'─'*90}")
        print(f"{'ANALYSE DE CARDINALITÉ (' + str(len(df_analyse)) + ' colonnes)':^90}")
        print(f"{'─'*90}")

        for _, row in df_analyse.iterrows():
            icone = {"SUPPRIMER": "❌", "ONE_HOT": "🔄", "TARGET_ENCODING": "🎯"}[row['Action']]
            print(f"\n📊 Variable : {row['Variable']}")
            print(f"  • Type               : {row['Type']}")
            print(f"  • Valeurs uniques    : {row['Valeurs_Uniques']} ({row['Ratio_%']}%)")
            print(f"  • Exemples           : {row['Exemples']}")
            print(f"  {icone} ACTION : {row['Action']} - {row['Raison']}")

        print(f"\n{'─'*90}")


    def obtenir_colonnes_par_action(self, df_analyse: pd.DataFrame,
                                    action: str) -> list:
        """
        Filtre le rapport d'analyse pour isoler les colonnes par type de traitement.
        """
        # Validation de sécurité pour éviter le plantage du pipeline
        if 'Action' not in df_analyse.columns:
            raise KeyError("Le DataFrame fourni ne contient pas la colonne 'Action'.")

        mask   = (df_analyse['Action'] == action)     # Filtre booléen
        list_c = df_analyse[mask]['Variable'].tolist() # Extraction en liste

        return list_c
