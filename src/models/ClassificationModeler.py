"""
CLASSIFICATION MODELER : PIPELINE DE MACHINE LEARNING ORIENTÉ OBJET
============================================================================
Gestion complète d'expériences de classification binaire.
Conformité aux standards MLOps et Data Science industrielle.
"""

# BIBLIOTHÈQUES STANDARDS ET SYSTÈME
# ------------------------------------------------------------------------------
import time                                   # Gestion du temps d'exécution
import warnings                               # Filtrage des avertissements
import numpy             as np                # Calcul numérique et matrices
import pandas            as pd                # Manipulation de structures
import matplotlib.pyplot as plt               # Visualisation de base
import seaborn           as sns               # Graphiques statistiques

import joblib
from pathlib import Path

# TYPAGE ET STRUCTURES DE DONNÉES
# ------------------------------------------------------------------------------
from   typing            import (
       Dict, Any, Optional,
       List, Tuple
)

# SCIKIT-LEARN : MODÈLES ET VALIDATION
# ------------------------------------------------------------------------------
from   sklearn.model_selection import (
       cross_validate,                        # Validation croisée multiple
       StratifiedKFold                        # Découpage stratifié des plis
)

# SCIKIT-LEARN : MÉTRIQUES ET SCORES
# ------------------------------------------------------------------------------
from sklearn.metrics import (
    # --- 1. GÉNÉRATION DE SCORERS ---
    make_scorer,                    # Transforme une métrique en fonction utilisable par GridSearchCV

    # --- 2. MÉTRIQUES DE PERFORMANCE GLOBALE ---
    accuracy_score,                 # Justesse : proportion de prédictions correctes (attention au déséquilibre)
    log_loss,                       # Entropie croisée : mesure la qualité des probabilités prédites

    # --- 3. MÉTRIQUES DE DÉTECTION (ORIENTATION CLASSE POSITIVE) ---
    precision_score,                # Précision : capacité à ne pas avoir de faux positifs (fiabilité de l'alerte)
    recall_score,                   # Rappel : capacité à capturer tous les positifs (sensibilité)
    f1_score,                       # F1 : moyenne harmonique pondérant précision et rappel
    fbeta_score,                    #

    # --- 4. ANALYSE DE DISCRIMINATION ET SEUILS ---
    roc_auc_score,                  # Aire sous ROC : capacité globale de séparation des classes
    roc_curve,                      # Coordonnées pour tracer la courbe ROC (TPR vs FPR)
    auc,                            # Fonction générique pour calculer l'aire sous une courbe (ROC ou PR)
    precision_recall_curve,         # Données pour la courbe Précision-Rappel (idéal pour classes déséquilibrées)
    average_precision_score,        # AP : résumé de la courbe Précision-Rappel

    # --- 5. ROBUSTESSE ET FIABILITÉ STATISTIQUE ---
    matthews_corrcoef,              # MCC : métrique la plus robuste prenant en compte toute la matrice de confusion
    cohen_kappa_score,              # Kappa : mesure l'accord réel en neutralisant l'effet du hasard

    # --- 6. DIAGNOSTIC ET REPORTING ---
    confusion_matrix,               # Matrice brute : détail des VP, FP, VN, FN
    classification_report           # Rapport textuel complet : précision, rappel et F1 par classe
)

from sklearn.inspection import permutation_importance


# CONFIGURATION GLOBALE
# ------------------------------------------------------------------------------
warnings.filterwarnings('ignore')              # Désactivation des alertes

# DÉFINITION DES MÉTRIQUES PAR DÉFAUT
# ------------------------------------------------------------------------------
DEFAULT_METRICS = {
    'accuracy'  : 'accuracy',                 # Justesse
    'precision' : make_scorer(precision_score,
                               average='binary',
                               zero_division=0), # Précision binaire
    'recall'    : make_scorer(recall_score,
                               average='binary',
                               zero_division=0), # Rappel binaire
    'f1'        : make_scorer(f1_score,
                               average='binary',
                               zero_division=0), # F1-Score

    # ⚠️ MÉTRIQUE PRIORITAIRE : F-beta avec β=2
    # β=2 pénalise 2× plus les faux négatifs que les faux positifs
    'f2'       : make_scorer(fbeta_score     , average='binary', zero_division=0, beta=2  ),

    'roc_auc'   : 'roc_auc',                  # AUC-ROC
    'log_loss'  : 'neg_log_loss'              # Log-Loss négative
}

# ##############################################################################
# CLASSE PRINCIPALE : ClassificationModeler
# ##############################################################################

class ClassificationModeler:
    """
    Pipeline complet de Machine Learning pour la CLASSIFICATION BINAIRE.
    
    Caractéristiques :
    -----------------
    - Entraînement avec validation croisée stratifiée
    - Évaluation automatique sur les jeux d'entraînement (train) et de test
    - Détection du surapprentissage (overfitting)
    - Métriques de classification complètes
    - Analyse du seuil de décision (threshold)
    - Comparaison de modèles multiples
    - Visualisations automatiques (ROC, PR, Matrice de Confusion)
    - Historique complet des expériences
    - Sélection automatique du meilleur modèle
    
    Exemple d'utilisation :
    -----------------------
    >>> modeler = ClassificationModeler(X_train, y_train, X_test, y_test)
    >>> modeler.entrenar_modelo(DummyClassifier(), 'Dummy')
    >>> modeler.entrenar_modelo(LogisticRegression(), 'Logistic')
    >>> modeler.entrenar_modelo(RandomForestClassifier(), 'RandomForest')
    >>> modeler.comparar_modelos()
    >>> modeler.visualizar_mejor_modelo()
    >>> best_model = modeler.get_mejor_modelo()
    """

    # ##########################################################################
    # INITIALISATION DE LA CLASSE ET CONFIGURATION DU CONTEXTE
    # ##########################################################################

    def __init__(
        self,
        X_train : pd.DataFrame,               # Données d'entraînement (Features)
        y_train : pd.Series,                  # Cibles d'entraînement (Labels)
        X_test  : pd.DataFrame,               # Données de test (Features)
        y_test  : pd.Series,                  # Cibles de test (Labels)
        config  : Optional[Dict] = None       # Dictionnaire de config optionnel
    ):
        """
        Initialise le ClassificationModeler avec les données et les paramètres.
        """
        # Attribution des données de base
        self.X_train    = X_train             # Features pour l'apprentissage
        self.y_train    = y_train             # Cibles pour l'apprentissage
        self.X_test     = X_test              # Features pour la validation
        self.y_test     = y_test              # Cibles pour la validation

        # Définition de la configuration (Fusion des défauts et de l'utilisateur)
        default_config  = {
            'RANDOM_STATE' : 42,              # Graine pour la reproductibilité
            'CV_FOLDS'     : 5,               # Nombre de plis pour la val. croisée
            'METRICS'      : DEFAULT_METRICS, # Dictionnaire des scores à calculer
            'THRESHOLD'    : 0.5              # Seuil de décision par défaut
        }

        # Mise à jour de la configuration avec les paramètres utilisateur
        self.config     = {**default_config, **(config or {})}

        # Fixation de la graine aléatoire globale (Pratique recommandée)
        np.random.seed(self.config['RANDOM_STATE'])

        # ----------------------------------------------------------------------
        # VALIDATIONS ET ÉTAT INTERNE
        # ----------------------------------------------------------------------

        # Validation de la structure binaire de la cible
        self._validate_binary_classification()

        # Initialisation du stockage des résultats
        self.historique_resultats  = {}            # Registre de toutes les expériences
        self.compteur_experiences = 0              # Compteur incrémental d'essais

        # Variables de référence pour le meilleur modèle
        self._best_model_name = None          # Nom du modèle le plus performant

        # Affichage du rapport d'initialisation dans la console
        self._print_initialization_info()

    # --------------------------------------------------------------------------
    # MÉTHODES DE CONTRÔLE INTERNE (ALIGNÉES)
    # --------------------------------------------------------------------------

    def _validate_binary_classification(self):
        """ Valide que le problème est strictement binaire. """
        n_classes = len(np.unique(self.y_train))
        if n_classes != 2:
            raise ValueError(
                f"Ce pipeline supporte uniquement la classification binaire. "
                f"Détection de {n_classes} classes distinctes."
            )


   # ##########################################################################
    # AFFICHAGE DU RAPPORT D'INITIALISATION
    # ##########################################################################

    def _print_initialization_info(self):
        """
        Affiche le diagnostic complet de l'état initial du pipeline.
        """
        # Calcul de la distribution des classes
        train_dist      = self.y_train.value_counts(normalize=True)
        test_dist       = self.y_test.value_counts(normalize=True)

        # Calcul du ratio de déséquilibre (Imbalance)
        imbalance_ratio = train_dist.max() / train_dist.min()

        print("\n" + "="*80)
        print("🎯 CLASSIFICATION MODELER INITIALISÉ")
        print("="*80)

        # ----------------------------------------------------------------------
        # STATISTIQUES DES DONNÉES (VOLUMÉTRIE)
        # ----------------------------------------------------------------------
        print(f"  Samples Train.......: {self.X_train.shape[0]}")
        print(f"  Features Train......: {self.X_train.shape[1]}")
        print(f"  Samples Test........: {self.X_test.shape[0]}")

        # ----------------------------------------------------------------------
        # DISTRIBUTION ET DIAGNOSTIC DES CLASSES
        # ----------------------------------------------------------------------
        print("\n  DISTRIBUTION DES CLASSES (TRAIN) :")
        for clase, prop in train_dist.items():
            print(f"    Classe {clase}..........: {prop:.2%}")

        print("\n  DISTRIBUTION DES CLASSES (TEST) :")
        for clase, prop in test_dist.items():
            print(f"    Classe {clase}..........: {prop:.2%}")

        # Alerte de déséquilibre selon les principes de Yann LeCun
        if imbalance_ratio > 2:
            print(f"\n  ⚠️  DATASET DÉSÉQUILIBRÉ (Ratio {imbalance_ratio:.2f}:1)")
            print("     Conseil : Utiliser class_weight='balanced' dans vos modèles")

        # ----------------------------------------------------------------------
        # CONFIGURATION TECHNIQUE DU PIPELINE
        # ----------------------------------------------------------------------
        print(f"\n  Random State........: {self.config['RANDOM_STATE']}")
        print(f"  CV Folds (Stratified): {self.config['CV_FOLDS']}")
        print(f"  Seuil de Décision...: {self.config['THRESHOLD']}")

        # Extraction des noms de métriques pour affichage propre
        metrics_list    = list(self.config['METRICS'].keys())
        print(f"  Métriques actives...: {metrics_list}")

        print("="*80 + "\n")


    def set_train_data(self, X_new: pd.DataFrame, y_new: pd.Series):
        """
        Met à jour les données d'entraînement de l'instance.
        Utile pour injecter des données équilibrées (SMOTE, etc.).
        """
        if len(X_new) != len(y_new):
            raise ValueError("❌ Erreur : X et y doivent avoir la même longueur.")

        self.X_train = X_new
        self.y_train = y_new
        print(f"🔄 Données d'entraînement mises à jour : {len(self.X_train)} échantillons.")

    def ____TRAINING_METHODS(self): pass

    # ##########################################################################
    # MÉTHODES D'ENTRAÎNEMENT ET DE GESTION DES EXPÉRIENCES
    # ##########################################################################

    def entrainer_modele(
        self,
        modele,                               # Estimateur compatible sklearn
        nom_modele       : str,               # Identifiant du modèle
        verbeux          : bool = True        # Affichage du journal (logs)
    ) -> Dict[str, Any]:
        """
        Entraîne un modèle de classification et archive les résultats.
        
        Paramètres :
        -----------
        modele     : Estimateur sklearn compatible avec la classification.
        nom_modele : str, nom unique pour identifier l'expérience.
        verbeux    : bool, si True, affiche les rapports de performance.
        
        Retourne :
        ---------
        dict       : Dictionnaire complet des résultats de l'expérience.
        """
        self.compteur_experiences += 1        # Incrément du suivi interne

        if verbeux:
            print("\n" + "="*80)
            print(f"🎯 EXPÉRIENCE #{self.compteur_experiences} : {nom_modele}")
            print("="*80)

        # 1. Validation Croisée Stratifiée (Évaluation de la stabilité)
        # ----------------------------------------------------------------------
        resultats_cv    = self._valider_croisement(modele, verbeux)

        # 2. Entraînement sur l'intégralité du jeu d'apprentissage
        # ----------------------------------------------------------------------
        modele_entraine, temps_train = self._ajuster_modele(modele, verbeux)

        # 3. Génération des prédictions (Classes et Probabilités)
        # ----------------------------------------------------------------------
        predictions     = self._obtenir_predictions(modele_entraine, verbeux)

        # 4. Évaluation multi-métriques (Train et Test)
        # ----------------------------------------------------------------------
        scores_train    = self._evaluer(
            self.y_train,
            predictions['y_train_pred'],
            predictions['y_train_proba'],
            "Apprentissage (Train)",
            verbeux
        )

        scores_test     = self._evaluer(
            self.y_test,
            predictions['y_test_pred'],
            predictions['y_test_proba'],
            "Validation (Test)",
            verbeux
        )

        # 5. Détection automatique du surapprentissage (Overfitting)
        # ----------------------------------------------------------------------
        surappris, diagnostics = self._detecter_surapprentissage(
            scores_train,
            scores_test,
            verbeux
        )

        # 6. Rapport détaillé de classification (Précision/Rappel par classe)
        # ----------------------------------------------------------------------
        if verbeux:
            self._afficher_rapport_classification(
                self.y_test,
                predictions['y_test_pred'],
                nom_modele
            )

        # 7. Calcul des Matrices de Confusion
        # ----------------------------------------------------------------------
        matrice_train   = confusion_matrix(self.y_train,
                                           predictions['y_train_pred'])
        matrice_test    = confusion_matrix(self.y_test,
                                           predictions['y_test_pred'])

        # 8. Résumé synthétique des performances
        # ----------------------------------------------------------------------
        if verbeux:
            self._afficher_resume(scores_train, scores_test)

        # 9. Stockage structuré des résultats
        # ----------------------------------------------------------------------
        resultats       = {
            'id_experience'    : self.compteur_experiences, # Ordre de l'essai
            'nom_modele'       : nom_modele,                # Label modèle
            'modele'           : modele_entraine,           # Objet fité
            'scores_cv'        : resultats_cv,              # Scores K-Fold
            'scores_train'     : scores_train,              # Métriques Train
            'scores_test'      : scores_test,               # Métriques Test
            'temps_train'      : temps_train,               # Durée CPU fit
            'surapprentissage' : surappris,                 # Flag booléen
            'diagnostics'      : diagnostics,               # Texte analyse
            'predictions'      : predictions,               # Vecteurs y_hat
            'matrice_confusion': {
                'train'        : matrice_train,             # Erreurs Train
                'test'         : matrice_test               # Erreurs Test
            },
            'horodatage'       : pd.Timestamp.now()         # Date/Heure
        }

        self.historique_resultats[nom_modele] = resultats

        # Mise à jour du "champion" (meilleur modèle basé sur le score F1)
        self._mettre_a_jour_meilleur_modele()

        if verbeux:
            print("="*80 + "\n")

        return resultats

    # ##########################################################################
    # MÉTHODES TECHNIQUES DE VALIDATION
    # ##########################################################################

    def _valider_croisement(
        self,
        modele,
        verbeux : bool
    ) -> Dict[str, Dict]:
        """
        Exécute une validation croisée stratifiée pour évaluer la stabilité.
        """
        if verbeux:
            nb_plis     = self.config['CV_FOLDS']     # Nombre de plis (folds)
            print(f"\n🔄 Validation Croisée Stratifiée ({nb_plis} plis)...")

        # StratifiedKFold pour maintenir la proportion des classes
        # ----------------------------------------------------------------------
        cv              = StratifiedKFold(
            n_splits     = self.config['CV_FOLDS'],
            shuffle      = True,
            random_state = self.config['RANDOM_STATE']
        )                                             # Configuration du split

        resultats_cv    = cross_validate(
            estimator          = modele,
            X                  = self.X_train,
            y                  = self.y_train,
            cv                 = cv,
            scoring            = self.config['METRICS'],
            return_train_score = True,
            n_jobs             = -1                   # Utilisation multi-cœurs
        )                                             # Calcul des scores

        # Traitement et agrégation des scores
        # ----------------------------------------------------------------------
        scores_cv       = {}                          # Dictionnaire final

        for nom_metrique in self.config['METRICS'].keys():
            s_apprentissage = resultats_cv[f'train_{nom_metrique}']
            s_validation    = resultats_cv[f'test_{nom_metrique}']

            # Inversion de la log_loss (sklearn utilise des valeurs négatives)
            if nom_metrique == 'log_loss':
                s_apprentissage = -s_apprentissage    # Passage en positif
                s_validation    = -s_validation       # Passage en positif

            # Structuration des statistiques par métrique
            scores_cv[nom_metrique] = {
                'train_moyenne' : s_apprentissage.mean(),
                'train_ecart'   : s_apprentissage.std(),
                'cv_moyenne'    : s_validation.mean(),
                'cv_ecart'      : s_validation.std()
            }                                         # Moyenne et écart-type

        # Affichage didactique des résultats
        # ----------------------------------------------------------------------
        if verbeux:
            print("   Résultats :")
            for metrique, valeurs in scores_cv.items():
                m_nom   = metrique.upper()            # Nom en majuscules
                m_train = valeurs['train_moyenne']    # Moyenne apprentissage
                e_train = valeurs['train_ecart']      # Écart apprentissage
                m_cv    = valeurs['cv_moyenne']       # Moyenne validation
                e_cv    = valeurs['cv_ecart']         # Écart validation

                print(f"   {m_nom:10s} → "
                      f"Train: {m_train:.4f} (±{e_train:.4f}) | "
                      f"CV: {m_cv:.4f} (±{e_cv:.4f})")

        return scores_cv

# ##########################################################################
    # MÉTHODES TECHNIQUES DE SUPPORT (INTERNES)
    # ##########################################################################

    def _ajuster_modele(
        self,
        modele,
        verbeux : bool
    ) -> Tuple[Any, float]:
        """ Entraîne le modèle sur l'intégralité du jeu d'apprentissage. """
        if verbeux:
            nb_echantillons = self.X_train.shape[0]
            print(f"\n🏋️  Entraînement sur {nb_echantillons} échantillons...")

        temps_debut     = time.time()         # Chronomètre début
        modele.fit(self.X_train, self.y_train)# Apprentissage (Fit)
        temps_train     = time.time() - temps_debut

        if verbeux:
            print(f"   ✅ Complété en {temps_train:.2f}s")

        return modele, temps_train

    # --------------------------------------------------------------------------

    def _obtenir_predictions(
        self,
        modele,
        verbeux : bool
    ) -> Dict[str, np.ndarray]:
        """ Génère les prédictions de classe et les probabilités. """
        if verbeux:
            print("\n🔮 Génération des prédictions...")

        # Prédictions de classes (0 ou 1)
        y_train_pred    = modele.predict(self.X_train)
        y_test_pred     = modele.predict(self.X_test)

        # Gestion des probabilités selon les capacités du modèle
        if hasattr(modele, 'predict_proba'):
            y_train_prob = modele.predict_proba(self.X_train)[:, 1]
            y_test_prob  = modele.predict_proba(self.X_test)[:, 1]
        elif hasattr(modele, 'decision_function'):
            y_train_prob = modele.decision_function(self.X_train)
            y_test_prob  = modele.decision_function(self.X_test)
        else:
            y_train_prob = None               # Pas de probabilités disponibles
            y_test_prob  = None

        predictions     = {
            'y_train_pred'  : y_train_pred,   # Classes prédites apprentissage
            'y_test_pred'   : y_test_pred,    # Classes prédites validation
            'y_train_proba' : y_train_prob,   # Scores/Probas apprentissage
            'y_test_proba'  : y_test_prob     # Scores/Probas validation
        }

        if verbeux:
            ratio_train = (y_train_pred == 1).mean()
            ratio_test  = (y_test_pred == 1).mean()
            print(f"   Train - Classe positive: {ratio_train:.2%}")
            print(f"   Test  - Classe positive: {ratio_test:.2%}")

        return predictions

    def predire_probabilites(self, nom_modele: str):
        """
        Expose les probabilités de prédiction pour un modèle spécifique.
        Idéal pour l'utilisation directe en Notebook.
        """
        # 1. Recuperamos la instancia del modelo
        instance, res = self.obtenir_modele(nom_modele)

        # 2. Usamos tu método interno (el que ya tienes)
        preds = self._obtenir_predictions(instance, verbeux=False)

        # 3. Création du DataFrame avec les noms standards que tu utilises
        df_probas = pd.DataFrame({
            'proba': preds['y_test_proba'],  # Plus de noms à rallonge
            'pred' : preds['y_test_pred']    # Correspond à ton code de catégorisation
        }, index=self.X_test.index)

        return df_probas

    # --------------------------------------------------------------------------

    def _evaluer(
        self,
        y_reel          : pd.Series,          # Vérité terrain
        y_predit        : np.ndarray,         # Classes prédites
        y_probabilite   : Optional[np.ndarray],# Scores de confiance
        nom_dataset     : str,                # Étiquette du jeu de données
        verbeux         : bool                # Mode verbeux
    ) -> Dict[str, float]:
        """ Calcule l'ensemble des métriques de performance. """

        # Métriques basées sur les classes
        scores          = {
            'accuracy'    : accuracy_score(y_reel, y_predit),
            'precision'   : precision_score(y_reel, y_predit, zero_division=0),
            'recall'      : recall_score(y_reel, y_predit, zero_division=0),
            'f1'          : f1_score(y_reel, y_predit, zero_division=0),
            # Añadimos el F2-Score
            # beta=2 otorga el doble de peso al recall que a la precisión
            'f2'          : fbeta_score(y_reel, y_predit, beta=2, zero_division=0),
            'specificite' : self._calculer_specificite(y_reel, y_predit),
            'mcc'         : matthews_corrcoef(y_reel, y_predit),
            'cohen_kappa' : cohen_kappa_score(y_reel, y_predit)
        }

        # Métriques basées sur les probabilités (si disponibles)
        if y_probabilite is not None:
            scores['roc_auc']       = roc_auc_score(y_reel, y_probabilite)
            scores['avg_precision'] = average_precision_score(y_reel,
                                                              y_probabilite)
            scores['log_loss']      = log_loss(y_reel, y_probabilite)
        else:
            scores['roc_auc']       = np.nan  # Non applicable
            scores['avg_precision'] = np.nan
            scores['log_loss']      = np.nan

        return scores

    # --------------------------------------------------------------------------

    def _calculer_specificite(self, y_reel, y_predit) -> float:
        """ Calcule la Spécificité (Taux de Vrais Négatifs). """
        tn, fp, fn, tp  = confusion_matrix(y_reel, y_predit).ravel()
        # Formule : TN / (TN + FP)
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # --------------------------------------------------------------------------

    def _detecter_surapprentissage(
        self,
        scores_train    : Dict,
        scores_test     : Dict,
        verbeux         : bool
    ) -> Tuple[bool, Dict]:
        """ Analyse les écarts pour identifier un surapprentissage. """
        surapprentissage= False               # État initial
        diagnostics     = {}                  # Détail des écarts

        # Analyse de l'écart (Gap) du F1-Score
        ecart_f1        = scores_train['f1'] - scores_test['f1']
        diagnostics['ecart_f1'] = ecart_f1
        if ecart_f1 > 0.15:                   # Seuil de tolérance expert
            surapprentissage = True

        # Analyse de l'écart de Justesse (Accuracy)
        ecart_acc       = scores_train['accuracy'] - scores_test['accuracy']
        diagnostics['ecart_accuracy'] = ecart_acc
        if ecart_acc > 0.15:
            surapprentissage = True

        # Analyse de l'écart ROC-AUC (si calculable)
        if not np.isnan(scores_train['roc_auc']):
            ecart_roc   = scores_train['roc_auc'] - scores_test['roc_auc']
            diagnostics['ecart_roc_auc'] = ecart_roc
            if ecart_roc > 0.15:
                surapprentissage = True

        diagnostics['overfitting'] = surapprentissage

        if verbeux:
            if surapprentissage:
                print("\n⚠️  SURAPPRENTISSAGE (OVERFITTING) DÉTECTÉ :")
                if ecart_f1 > 0.15:  print(f"   • Écart F1.......: {ecart_f1:.3f}")
                if ecart_acc > 0.15: print(f"   • Écart Accuracy.: {ecart_acc:.3f}")
            else:
                print("\n✅ Généralisation optimale (pas d'overfitting)")

        return surapprentissage, diagnostics

    # --------------------------------------------------------------------------

    def _afficher_rapport_classification(self, y_reel, y_predit, nom_modele):
        """ Affiche le rapport détaillé (Précision, Rappel, F1 par classe). """
        print("\n" + "─"*70)
        print(f"📊 RAPPORT DE CLASSIFICATION ({nom_modele})")
        print("─"*70)
        print(classification_report(y_reel, y_predit, zero_division=0))

    # --------------------------------------------------------------------------

    def _afficher_resume(self, s_train: Dict, s_test: Dict):
        """ Affiche un tableau comparatif synthétique. """
        print("\n" + "─"*75)
        print("📊 RÉSUMÉ COMPARATIF DES PERFORMANCES")
        print("─"*75)
        print(f"{'Métrique':15s} | {'Train':>10s} | {'Test':>10s} | {'Δ':>10s}")
        print("-" * 75)

        metriques_clefs = ['accuracy', 'precision', 'recall', 'f1', 'f2',
                           'specificite', 'roc_auc', 'mcc']

        for m in metriques_clefs:
            v_train     = s_train.get(m, np.nan)
            v_test      = s_test.get(m, np.nan)

            if not np.isnan(v_train) and not np.isnan(v_test):
                diff    = v_train - v_test
                print(f"{m.upper():15s} | {v_train:10.4f} | "
                      f"{v_test:10.4f} | {diff:+10.4f}")

    def _afficher_resume_with_metrics(self, s_train: Dict, s_test: Dict, metrics_list: Optional[List[str]] = None):
        """
        Affiche un tableau comparatif synthétique.
        Si metrics_list est fourni (ex: CUSTOM_METRICS.keys()), il l'utilise pour remplir le tableau.
        """
        print("\n" + "─"*75)
        print("📊 RÉSUMÉ COMPARATIF DES PERFORMANCES")
        print("─"*75)
        print(f"{'Métrique':15s} | {'Train':>10s} | {'Test':>10s} | {'Δ':>10s}")
        print("-" * 75)

        # Si no se pasan métricas, usamos las estándar para no romper llamadas antiguas
        if metrics_list is None:
            metriques_clefs = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        else:
            # Extraemos las llaves del diccionario CUSTOM_METRICS si fuera necesario
            metriques_clefs = list(metrics_list)

        for m in metriques_clefs:
            # Usamos .get() para evitar errores si una métrica no existe en los diccionarios de entrada
            v_train = s_train.get(m, np.nan)
            v_test  = s_test.get(m, np.nan)

            # Solo imprimimos si al menos uno de los valores existe
            if not (np.isnan(v_train) and np.isnan(v_test)):
                diff = v_train - v_test
                print(f"{m.upper():15s} | {v_train:10.4f} | "
                      f"{v_test:10.4f} | {diff:+10.4f}")

    # --------------------------------------------------------------------------

    def _mettre_a_jour_meilleur_modele(self):
        """ Identifie le modèle leader basé sur le F1-Score en validation. """
        if not self.historique_resultats:
            return

        # Le critère de sélection est la moyenne harmonique F1 sur le jeu Test
        meilleur        = max(
            self.historique_resultats.items(),
            key         = lambda x: x[1]['scores_test']['f1']
        )
        self._nom_meilleur_modele = meilleur[0]

    def ____COMPARAISON_ANALYSIS_METHODS(self): pass

    # ##########################################################################
    # COMPARAISON AND ANALYSIS METHODS
    # ##########################################################################

    def comparer_modeles(self, trier_par: str = 'f1') -> pd.DataFrame:
        """
        Génère un tableau comparatif de tous les modèles entraînés.
        
        Paramètres :
        -----------
        trier_par : str, métrique pour le tri (f1, accuracy, precision, etc.)
        
        Retourne :
        ---------
        pd.DataFrame : Tableau comparatif trié.
        """
        if not self.historique_resultats:
            print("⚠️  Aucun modèle n'a encore été entraîné.")
            return pd.DataFrame()

        lignes          = []
        for nom_modele, resultats in self.historique_resultats.items():
            sc_test     = resultats['scores_test']
            sc_cv       = resultats['scores_cv']

            # Construction de la ligne de données
            ligne       = {
                'Modèle'      : nom_modele,
                'F1 (CV)'     : sc_cv['f1']['cv_moyenne'],
                'Justesse'    : sc_test['accuracy'],
                'Précision'   : sc_test['precision'],
                'Rappel'      : sc_test['recall'],
                'Score-F1'    : sc_test['f1'],
                'Spécificité' : sc_test['specificite'],
                'ROC-AUC'     : sc_test['roc_auc'],
                'MCC'         : sc_test['mcc'],
                'Temps (s)'   : resultats['temps_train'],
                'Surappris'   : '❗' if resultats['surapprentissage'] else '✅',
                'Expérience'  : resultats['id_experience']
            }
            lignes.append(ligne)

        df              = pd.DataFrame(lignes)

        # Mapping des colonnes pour le tri
        # ----------------------------------------------------------------------
        map_colonnes    = {
            'f1'        : 'Score-F1',
            'accuracy'  : 'Justesse',
            'precision' : 'Précision',
            'recall'    : 'Rappel',
            'roc_auc'   : 'ROC-AUC'
        }
        colonne_tri     = map_colonnes.get(trier_par, 'Score-F1')

        # Application du tri décroissant
        df              = df.sort_values(
                            colonne_tri,
                            ascending    = False,
                            na_position  = 'last'
                          ).reset_index(drop=True)

        # Identification visuelle du champion
        df['Rang']      = ''
        df.loc[0, 'Rang'] = '🏆'

        # Affichage du rapport final
        # ----------------------------------------------------------------------
        print("\n" + "="*110)
        print(f"📊 COMPARAISON DES MODÈLES (trié par {trier_par.upper()})")
        print("="*110 + "\n")
        print(df.to_string(index=False))
        print("\n" + "="*110)
        print(f"🏆 Meilleur modèle : {df.loc[0, 'Modèle']} "
              f"(F1 = {df.loc[0, 'Score-F1']:.4f})")
        print("="*110 + "\n")

        return df

    def comparer_modeles_with_custom_metrics(self, trier_par: str = 'f1', custom_metrics: Optional[Dict] = None) -> pd.DataFrame:
        """
        Génère un tableau comparatif dynamique de tous les modèles.
        """
        if not self.historique_resultats:
            print("⚠️ Aucun modèle n'a encore été entraîné.")
            return pd.DataFrame()

        # 1. Definir nombres amigables para las métricas estándar
        friendly_names = {
            'accuracy'    : 'Justesse',
            'precision'   : 'Précision',
            'recall'      : 'Rappel',
            'f1'          : 'Score-F1',
            'f2'          : 'Score-F2',
            'specificite' : 'Spécificité',
            'roc_auc'     : 'ROC-AUC',
            'mcc'         : 'MCC',
            'log_loss'    : 'Log-Loss'
        }

        # 2. Determinar qué métricas vamos a mostrar
        # Si viene custom_metrics, usamos sus llaves; si no, una lista base
        if custom_metrics:
            metriques_a_afficher = list(custom_metrics.keys())
        else:
            metriques_a_afficher = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']

        lignes = []
        for nom_modele, resultats in self.historique_resultats.items():
            sc_test = resultats['scores_test']
            sc_cv   = resultats['scores_cv']

            # Iniciamos la línea con datos básicos
            ligne = {
                'Modèle': nom_modele,
                'Temps (s)': resultats['temps_train'],
                'Surappris': '❗' if resultats['surapprentissage'] else '✅',
                'Expérience': resultats['id_experience']
            }

            # Añadimos dinámicamente las métricas solicitadas
            for m in metriques_a_afficher:
                col_name = friendly_names.get(m, m.upper()) # Nombre amigable o el código en mayúsculas
                ligne[col_name] = sc_test.get(m, np.nan)

                # Caso especial para mostrar el F1 de Cross-Validation si existe
                if m == 'f1' and 'f1' in sc_cv:
                    ligne['F1 (CV)'] = sc_cv['f1'].get('cv_moyenne', np.nan)

            lignes.append(ligne)

        df = pd.DataFrame(lignes)

        # 3. Gestión dinámica del tri (ordenamiento)
        colonne_tri = friendly_names.get(trier_par, trier_par.upper())

        # Verificamos si la columna de tri existe en el DF, si no, volvemos a Score-F1
        if colonne_tri not in df.columns:
            colonne_tri = 'Score-F1' if 'Score-F1' in df.columns else df.columns[1]

        # Aplicación del tri
        df = df.sort_values(by=colonne_tri, ascending=False, na_position='last').reset_index(drop=True)

        # Identificación del campeón
        # df.insert(0, 'Rang', '')
        df['Rang']        = ''
        df.loc[0, 'Rang'] = '🏆'

        # 4. Affichage
        print("\n" + "="*120)
        print(f"📊 COMPARAISON DES MODÈLES (trié par {colonne_tri})")
        print("="*120 + "\n")
        print(df.to_string(index=False))

        mejor_valor = df.loc[0, colonne_tri]
        print(f"\n🏆 Meilleur modèle : {df.loc[0, 'Modèle']} ({colonne_tri} = {mejor_valor:.4f})")
        print("="*120 + "\n")

        return df

    # --------------------------------------------------------------------------

    def obtenir_meilleur_modele(self) -> Tuple[str, Any, Dict]:
        """
        Récupère l'instance et les résultats du modèle champion.
        
        Retourne :
        ---------
        tuple : (nom, objet_modele, resultats_complets)
        """
        if not self._nom_meilleur_modele:
            raise ValueError("Aucun modèle n'est disponible dans l'historique.")

        resultats       = self.historique_resultats[self._nom_meilleur_modele]

        return (
            self._nom_meilleur_modele,
            resultats['modele'],
            resultats
        )

    # --------------------------------------------------------------------------

    def obtenir_modele(self, nom_modele: str) -> Tuple[Any, Dict]:
        """
        Récupère un modèle spécifique par son nom.
        """
        if nom_modele not in self.historique_resultats:
            modeles_dispo = list(self.historique_resultats.keys())
            raise KeyError(f"Modèle '{nom_modele}' introuvable. "
                           f"Disponibles : {modeles_dispo}")

        resultats       = self.historique_resultats[nom_modele]

        return resultats['modele'], resultats


    def analyser_importance_features_by_method(nom_modele=None, X=None, y=None, top_n=15, method='auto', scoring='f1'):
        """
        Analyse l'importance des variables avec priorité configurable.
        
        Arguments:
        ----------
        method : 'auto'       -> Utilise native si disponible, sinon permutation.
                 'permutation'-> FORCE le calcul par permutation (idéal pour modèles linéaires).
                 'native'     -> FORCE l'utilisation de coef_ ou feature_importances_.
        """
        # 1. Récupération du modèle (Logique inchangée)
        if nom_modele is None:
            nom, instance, res = self.obtenir_meilleur_modele()
        else:
            nom = nom_modele
            instance, res = self.obtenir_modele(nom)

        features = X.columns if X is not None else X_train.columns
        importances = None
        type_imp = ""

        # 2. Logique de décision (Modifiée pour donner la priorité à la méthode choisie)

        # CAS FORCE : PERMUTATION
        if method == 'permutation':
            if X is not None and y is not None:
                f2_scorer      = make_scorer(fbeta_score, beta=2, zero_division=0)
                actual_scoring = f2_scorer if scoring == 'f2' else scoring

                print(f"🔄 Calcul de la Permutation Importance ({scoring}) pour {nom}...")

                r = permutation_importance(
                    instance, X, y,
                    n_repeats     = 3,
                    scoring       = actual_scoring, # Aquí pasamos el objeto f2_scorer o el string 'f1'
                    random_state  = 42,
                    n_jobs        = -1
                )
                importances = r.importances_mean
                type_imp = "Permutation Importance (Forcée)"
            else:
                print("❌ Erreur: X et y sont requis pour la permutation.")
                return None

        # CAS NATIVE (ou AUTO si permutation n'est pas forcée)
        if importances is None:
            if hasattr(instance, 'feature_importances_') and method != 'permutation':
                importances = instance.feature_importances_
                type_imp = "Native (Gini/Gain)"
            elif hasattr(instance, 'coef_') and method != 'permutation':
                importances = np.abs(instance.coef_[0])
                type_imp = "Coefficients (Magnitude)"

        # 3. Finalisation (DataFrame et Plot)
        if importances is None:
            print(f"⚠️ Impossible d'extraire l'importance pour {nom} avec la méthode {method}")
            return None

        df_imp                  = pd.DataFrame({'Feature': features, 'Importance': importances})
        df_imp['Feature_Clean'] = df_imp['Feature'].str.replace(r'c_ohe__|n_std__|n_log__|__', ' ', regex=True).str.strip()
        df_imp                  = df_imp.sort_values(by='Importance', ascending=False).head(top_n)

        # Visualisation
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature_Clean', data=df_imp, palette='viridis')
        plt.title(f'Top {top_n} Variables - {nom}\nMéthode : {type_imp}', fontweight='bold')
        plt.show()

        return df_imp



    def ____VISUALISATION_METHODS(self): pass

    # ##########################################################################
    # MÉTHODES DE VISUALISATION
    # ##########################################################################

    def visualiser_modele(
        self,
        nom_modele       : Optional[str] = None,
        taille_figure    : Tuple[int, int] = (16, 10)
    ):
        """
        Analyse graphique complète : ROC, PR, Confusion et Distribution.
        """
        if nom_modele is None:
            nom_modele   = self._nom_meilleur_modele

        if nom_modele not in self.historique_resultats:
            raise KeyError(f"Modèle '{nom_modele}' introuvable.")

        resultats       = self.historique_resultats[nom_modele]

        fig             = plt.figure(figsize=taille_figure)
        grille          = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Courbe ROC (Receiver Operating Characteristic)
        ax1             = fig.add_subplot(grille[0, 0])
        self._tracer_courbe_roc(resultats, ax1)

        # 2. Courbe Précision-Rappel (Precision-Recall)
        ax2             = fig.add_subplot(grille[0, 1])
        self._tracer_courbe_pr(resultats, ax2)

        # 3. Tableau synthétique des métriques
        ax3             = fig.add_subplot(grille[0, 2])
        self._tracer_resume_metriques(resultats, ax3)

        # 4. Matrice de Confusion (Apprentissage / Train)
        ax4             = fig.add_subplot(grille[1, 0])
        self._tracer_matrice_confusion(resultats, 'train', ax4)

        # 5. Matrice de Confusion (Validation / Test)
        ax5             = fig.add_subplot(grille[1, 1])
        self._tracer_matrice_confusion(resultats, 'test', ax5)

        # 6. Distribution des densités de probabilités
        ax6             = fig.add_subplot(grille[1, 2])
        self._tracer_distribution_probabilites(resultats, ax6)


        plt.suptitle(f'Analyse Diagnostique Complète: {nom_modele}',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.show()

    # --------------------------------------------------------------------------

    def visualiser_meilleur_modele(self):
        """ Déclenche la visualisation pour le modèle champion. """
        self.visualiser_modele(nom_modele=None)

    # --------------------------------------------------------------------------

    def _tracer_courbe_roc(self, resultats, ax):
        """ Trace la courbe ROC et calcule l'aire sous la courbe (AUC). """
        y_reel          = self.y_test
        y_proba         = resultats['predictions']['y_test_proba']

        if y_proba is None:
            ax.text(0.5, 0.5, 'ROC non disponible\n(modèle sans probabilités)',
                    ha='center', va='center', fontsize=11)
            ax.set_title('Courbe ROC', fontweight='bold')
            return

        fpr, tpr, _     = roc_curve(y_reel, y_proba)
        aire_roc        = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2.5, color='darkorange',
                label=f'ROC (AUC = {aire_roc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire', alpha=0.5)
        ax.set_xlabel('Taux de Faux Positifs', fontweight='bold')
        ax.set_ylabel('Taux de Vrais Positifs', fontweight='bold')
        ax.set_title('Courbe ROC', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.2)

    # --------------------------------------------------------------------------

    def _tracer_courbe_pr(self, resultats, ax):
        """ Trace la courbe Précision-Rappel. """
        y_reel          = self.y_test
        y_proba         = resultats['predictions']['y_test_proba']

        if y_proba is None:
            ax.text(0.5, 0.5, 'PR non disponible', ha='center', va='center')
            return

        precision, rappel, _ = precision_recall_curve(y_reel, y_proba)
        pr_moyenne      = average_precision_score(y_reel, y_proba)
        reference       = y_reel.mean()

        ax.plot(rappel, precision, lw=2.5, color='navy',
                label=f'PR (AP = {pr_moyenne:.3f})')
        ax.axhline(y=reference, color='red', ls='--', lw=1,
                   label=f'Base ({reference:.2f})')
        ax.set_xlabel('Rappel (Recall)', fontweight='bold')
        ax.set_ylabel('Précision', fontweight='bold')
        ax.set_title('Courbe Précision-Rappel', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.2)

    # --------------------------------------------------------------------------

    def _tracer_matrice_confusion(self, resultats, jeu_donnees, ax):
        """ Affiche la matrice de confusion avec annotations. """
        mc              = resultats['matrice_confusion'][jeu_donnees]
        mc_norm         = mc.astype('float') / mc.sum(axis=1)[:, np.newaxis]

        annotations     = np.empty_like(mc, dtype=object)
        for i in range(mc.shape[0]):
            for j in range(mc.shape[1]):
                annotations[i, j] = f'{mc[i, j]}\n({mc_norm[i, j]:.1%})'

        sns.heatmap(mc, annot=annotations, fmt='', cmap='Blues',
                    cbar=False, ax=ax,
                    xticklabels=['Négatif', 'Positif'],
                    yticklabels=['Négatif', 'Positif'])

        ax.set_xlabel('Prédiction', fontweight='bold')
        ax.set_ylabel('Réel', fontweight='bold')
        ax.set_title(f'Matrice de Confusion ({jeu_donnees.upper()})',
                     fontweight='bold')

    # --------------------------------------------------------------------------

    def _tracer_distribution_probabilites(self, resultats, ax):
        """ Analyse la séparation des classes via les probabilités. """
        y_reel          = self.y_test
        y_proba         = resultats['predictions']['y_test_proba']

        if y_proba is None:
            ax.set_title('Distribution non disponible', fontweight='bold')
            return

        ax.hist(y_proba[y_reel == 0], bins=30, alpha=0.5, label='Classe 0',
                color='blue', edgecolor='black')
        ax.hist(y_proba[y_reel == 1], bins=30, alpha=0.5, label='Classe 1',
                color='red', edgecolor='black')

        # Priorité 1 : Le seuil enregistré spécifiquement pour ce modèle (0.72)
        # Priorité 2 : Le seuil global de la config (self.config)
        # Priorité 3 : 0.5 par défaut
        seuil = resultats.get('seuil_optimal', self.config.get('THRESHOLD', 0.5))

        ax.axvline(x=seuil, color='green', ls='--', lw=2, label=f'Seuil {seuil}')
        ax.set_xlabel('Probabilité prédite (Classe 1)', fontweight='bold')
        ax.set_ylabel('Fréquence', fontweight='bold')
        ax.set_title('Séparabilité des Classes', fontweight='bold')
        ax.legend()

    # --------------------------------------------------------------------------

    def _tracer_resume_metriques(self, resultats, ax):
        """ Affiche un bloc textuel avec les scores clés. """
        ax.axis('off')
        s_test          = resultats['scores_test']
        s_train         = resultats['scores_train']

        texte_metriques = (
            f"🏆 MODÈLE : {resultats['nom_modele']}\n\n"
            f"📊 PERFORMANCE TEST :\n"
            f"────────────────────────\n"
            f"Justesse (Acc) : {s_test['accuracy']:.4f}\n"
            f"Précision      : {s_test['precision']:.4f}\n"
            f"Rappel         : {s_test['recall']:.4f}\n"
            f"Score-F1       : {s_test['f1']:.4f}\n"
            f"ROC-AUC        : {s_test['roc_auc']:.4f}\n"
            f"MCC            : {s_test['mcc']:.4f}\n\n"
            f"⏱️ INFRASTRUCTURE :\n"
            f"────────────────────────\n"
            f"Temps Train    : {resultats['temps_train']:.2f}s\n"
            f"Surappris      : {'⚠️ OUI' if resultats['surapprentissage'] else '✅ NON'}"
        )

        couleur         = '#FFF9C4' if resultats['surapprentissage'] else '#C8E6C9'
        ax.text(0.5, 0.5, texte_metriques, transform=ax.transAxes,
                fontsize=9, ha='center', va='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor=couleur, alpha=0.9))

    # --------------------------------------------------------------------------

    def visualiser_comparaison(self, taille_fig: Tuple[int, int] = (16, 10)):
        """ Compare graphiquement tous les modèles de l'historique. """
        if len(self.historique_resultats) < 2:
            print("⚠️  Besoin d'au moins 2 modèles pour comparer.")
            return

        fig, axes       = plt.subplots(2, 2, figsize=taille_fig)
        noms            = list(self.historique_resultats.keys())
        x               = np.arange(len(noms))
        largeur         = 0.35

        # 1. Comparaison F1-Score : CV vs Test
        ax              = axes[0, 0]
        f1_cv           = [self.historique_resultats[m]['scores_cv']['f1']
                           ['cv_moyenne'] for m in noms]
        f1_test         = [self.historique_resultats[m]['scores_test']['f1']
                           for m in noms]

        ax.bar(x - largeur/2, f1_cv, largeur, label='F1 (Validation Croisée)')
        ax.bar(x + largeur/2, f1_test, largeur, label='F1 (Test Final)')
        ax.set_ylabel('Score F1', fontweight='bold')
        ax.set_title('Robustesse : CV vs Test', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(noms, rotation=45, ha='right')
        ax.legend()

        # 2. Comparaison ROC-AUC (Capacité de discrimination)
        # ----------------------------------------------------------------------
        ax = axes[0, 1]
        auc_scores = [self.historique_resultats[m]['scores_test'].get('roc_auc', 0) for m in noms]

        sns.barplot(x=noms, y=auc_scores, ax=ax, palette='viridis')
        ax.set_title('Capacité de Discrimination (ROC-AUC)', fontweight='bold')
        ax.set_ylabel('AUC Score')
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=45)
        # Ajout des valeurs au-dessus des barres
        for i, v in enumerate(auc_scores):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center')

        # 3. Comparaison Précision vs Rappel (Le compromis du classifieur)
        # ----------------------------------------------------------------------
        ax = axes[1, 0]
        precision = [self.historique_resultats[m]['scores_test']['precision'] for m in noms]
        rappel    = [self.historique_resultats[m]['scores_test']['recall'] for m in noms]

        ax.scatter(rappel, precision, s=100, color='red', edgecolors='black')
        ax.set_title('Compromis Précision / Rappel', fontweight='bold')
        ax.set_xlabel('Rappel (Recall)')
        ax.set_ylabel('Précision')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Étiquetage des points
        for i, nom in enumerate(noms):
            ax.annotate(nom, (rappel[i], precision[i]), xytext=(5, 5), textcoords='offset points')

        # 4. Efficacité Computationnelle (Temps d'entraînement)
        # ----------------------------------------------------------------------
        ax = axes[1, 1]
        temps = [self.historique_resultats[m]['temps_train'] for m in noms]

        sns.barplot(x=noms, y=temps, ax=ax, palette='magma')
        ax.set_title("Temps d'Exécution (Secondes)", fontweight='bold')
        ax.set_ylabel('Temps (s)')
        ax.tick_params(axis='x', rotation=45)

        # [Les autres graphiques suivent la même logique de traduction...]
        plt.tight_layout()
        plt.show()


    def visualiser_comparaison_avancee(self, taille_fig: Tuple[int, int] = (18, 12)):
        """ 
        Génère une analyse comparative de haut niveau avec superposition ROC 
        y métriques de robustesse (MCC).
        """
        if len(self.historique_resultats) < 2:
            print("⚠️ Besoin d'au moins 2 modèles pour une comparaison avancée.")
            return

        fig = plt.figure(figsize=taille_fig)
        grille = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
        noms = list(self.historique_resultats.keys())
        couleurs = plt.cm.tab10(np.linspace(0, 1, len(noms)))

        # --- GRAPH 1 : SUPERPOSITION DES COURBES ROC ---
        ax1 = fig.add_subplot(grille[0, 0])
        for i, nom in enumerate(noms):
            res = self.historique_resultats[nom]
            y_proba = res['predictions']['y_test_proba']
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                aire = res['scores_test']['roc_auc']
                ax1.plot(fpr, tpr, label=f'{nom} (AUC={aire:.3f})', color=couleurs[i], lw=2)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_title('Superposition des Courbes ROC', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Taux de Faux Positifs')
        ax1.set_ylabel('Taux de Vrais Positifs')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(alpha=0.2)

        # --- GRAPH 2 : COMPARAISON DU MCC (Robustesse réelle) ---
        ax2 = fig.add_subplot(grille[0, 1])
        mcc_scores = [self.historique_resultats[m]['scores_test']['mcc'] for m in noms]
        bars = ax2.bar(noms, mcc_scores, color=couleurs, alpha=0.8)
        ax2.set_title('Coefficient de Corrélation de Matthews (MCC)', fontweight='bold')
        ax2.set_ylabel('Score MCC (-1 à +1)')
        ax2.set_ylim(-0.1, 1.0) # On ajuste selon vos résultats
        ax2.tick_params(axis='x', rotation=45)
        # Ajout des étiquettes sur les barres
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', fontweight='bold')

        # --- GRAPH 3 : PRÉCISION VS RAPPEL (Cartographie Stratégique) ---
        ax3 = fig.add_subplot(grille[1, 0])
        precisions = [self.historique_resultats[m]['scores_test']['precision'] for m in noms]
        rappels = [self.historique_resultats[m]['scores_test']['recall'] for m in noms]

        for i, nom in enumerate(noms):
            ax3.scatter(rappels[i], precisions[i], color=couleurs[i], s=200, edgecolors='black', label=nom, zorder=3)

        # Ajout d'une zone cible (Haute performance)
        ax3.axvspan(0.5, 1.0, 0.5, 1.0, color='green', alpha=0.05, label='Zone Cible')
        ax3.set_title('Cartographie Précision / Rappel', fontweight='bold')
        ax3.set_xlabel('Rappel (Sensibilité)')
        ax3.set_ylabel('Précision (Fiabilité)')
        ax3.set_xlim(-0.05, 1.05)
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='lower right', fontsize=9)
        ax3.grid(True, linestyle='--', alpha=0.5)

        # --- GRAPH 4 : RATIO SURAPPRENTISSAGE (Ecart Train/Test) ---
        ax4 = fig.add_subplot(grille[1, 1])
        # On calcule l'écart de F1 pour visualiser l'instabilité
        ecarts = []
        for m in noms:
            f1_train = self.historique_resultats[m]['scores_train']['f1']
            f1_test = self.historique_resultats[m]['scores_test']['f1']
            ecarts.append(f1_train - f1_test)

        ax4.barh(noms, ecarts, color='salmon', alpha=0.8)
        ax4.set_title('Instabilité (F1_Train - F1_Test)', fontweight='bold')
        ax4.set_xlabel('Ecart de Score (Plus petit = plus stable)')

        plt.suptitle('BILAN COMPARATIF GLOBAL : STRATÉGIE & ROBUSTESSE', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def ____SAUVEGARDE_METHODS(self): pass

    # ##########################################################################
    # GESTION DE LA PERSISTENCE (SAUVEGARDE ET CHARGEMENT)
    # ##########################################################################

    def sauvegarder_modele(self, nom_modele: str, dossier: str = 'modeles_sauvegardes'):
        """
        Sauvegarde un modèle spécifique entraîné dans un fichier .joblib
        """
        # Création du dossier s'il n'existe pas
        Path(dossier).mkdir(parents=True, exist_ok=True)

        if nom_modele not in self.historique_resultats:
            print(f"❌ Erreur : Le modèle '{nom_modele}' n'existe pas dans l'historique.")
            return

        # Récupération de l'objet modèle
        modele = self.historique_resultats[nom_modele]['modele']
        chemin = Path(dossier) / f"{nom_modele}.joblib"

        # Sérialisation sur le disque
        joblib.dump(modele, chemin)
        largeur = 35
        print(f"💾 Modèle '{nom_modele:<{largeur}}' sauvegardé sous : {chemin}")

    # --------------------------------------------------------------------------

    def sauvegarder_tous_les_modeles(self, dossier: str = 'modeles_sauvegardes'):
        """
        Sauvegarde tous les modèles de l'historique dans le dossier spécifié.
        """
        if not self.historique_resultats:
            print("⚠️ Aucun modèle entraîné à sauvegarder.")
            return

        for nom in self.historique_resultats.keys():
            self.sauvegarder_modele(nom, dossier)

        print(f"✅ {len(self.historique_resultats)} modèles ont été sauvegardés avec succès.")

    def ____TOOLINGS_METHODS(self): pass

    # ##########################################################################
    # MÉTHODES UTILITAIRES
    # ##########################################################################

    def resume(self):
        """ Affiche un résumé synthétique de l'état actuel de l'instance. """
        print("\n" + "="*80)
        print("📋 RÉSUMÉ DU CLASSIFICATION MODELER")
        print("="*80)
        print(f"Modèles entraînés     : {len(self.historique_resultats)}")
        print(f"Expériences totales   : {self.compteur_experiences}")

        if self._nom_meilleur_modele:
            res_meilleur = self.historique_resultats[self._nom_meilleur_modele]
            f1_final     = res_meilleur['scores_test']['f1']
            auc_final    = res_meilleur['scores_test'].get('roc_auc', 0)

            print(f"Meilleur modèle       : {self._nom_meilleur_modele}")
            print(f"  └─ Score-F1         : {f1_final:.4f}")
            print(f"  └─ ROC-AUC          : {auc_final:.4f}")

        print("="*80 + "\n")

    # --------------------------------------------------------------------------

    def nettoyer_historique(self):
        """ Réinitialise complètement l'historique des expériences. """
        self.historique_resultats = {}
        self.compteur_experiences = 0
        self._nom_meilleur_modele = None
        print("✅ Historique nettoyé avec succès.")

    # --------------------------------------------------------------------------

    def exporter_resultats(
        self,
        chemin_fichier : str = 'resultats_classification.csv'
    ):
        """ 
        Exporte le tableau comparatif au format CSV pour archivage. 
        """
        df = self.comparar_modelos()           # Récupère le DataFrame trié
        if not df.empty:
            df.to_csv(chemin_fichier, index=False)
            print(f"✅ Résultats exportés vers : {chemin_fichier}")

    # --------------------------------------------------------------------------

    def __repr__(self):
        """ Représentation technique de l'objet (Dunder method). """
        return (f"ClassificationModeler("
                f"modeles={len(self.historique_resultats)}, "
                f"experiences={self.compteur_experiences}, "
                f"champion={self._nom_meilleur_modele})")
