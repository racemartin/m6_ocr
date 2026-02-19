# Étape 2 – Planification : Préparation des données et Feature Engineering

**Projet** : Scoring crédit (Prêt à dépenser / Home Credit)  
**Objectif** : Définir l’architecture professionnelle pour raw → clean → vues maîtresses, schéma métier/technique, et dérivation automatique des listes de transformation (OHE, log, standard, etc.) sans code dupliqué ni listes en dur.

---

## 1. Contexte et réutilisation (docs/old)

### 1.1 Ce qu’on réutilise

| Élément | Fichier / idée | Adaptation projet scoring |
|--------|----------------|----------------------------|
| **Classe base** | `DataCleaner` (colonnes par type, history, params appris) | Garder l’esprit : identifiants, constantes, catégorielles, numériques, imputation. Ne pas tout reprendre (fichier énorme) : extraire un noyau minimal ou une interface. |
| **Cleaner métier** | `AttritionCleaner(DataCleaner)` + engine SQL | **CreditScoringCleaner** : hérite d’une base légère, travaille avec `engine` (comme AttritionCleaner). |
| **Pattern DB** | Tables `raw_*` → vues `v_clean_*` → `v_master_*` → `v_features_*` | Même logique : une table raw par fichier source, une vue clean par source, une vue maîtresse (jointures), une vue feature engineering. |
| **Séquences d’étapes** | `phase1_preparation.py`, `phase2_feature_engineering.py` | Conserver des **pipelines séquentiels** (scripts ou modules) : Phase 1 = ingestion + clean + master + features SQL ; Phase 2 = chargement + preprocessing sklearn (schéma-driven). |
| **Config features** | `FeatureConfig` avec listes en dur (COLS_ONE_HOT, COLS_TO_LOG, etc.) | **À remplacer** par un **registre d’attributs** (schéma) dont on dérive ces listes automatiquement. |

### 1.2 Données brutes du projet (Home Credit)

- **Fichiers** : `application_train.csv`, `application_test.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv`, `HomeCredit_columns_description.csv`.
- **Clés** : `SK_ID_CURR` (dossier courant) dans application_* ; `SK_ID_BUREAU` lie bureau et bureau_balance ; `SK_ID_PREV` lie previous_application aux autres ; etc. La description des colonnes et tables est dans `HomeCredit_columns_description.csv`.
- **Cible** : `TARGET` (1 = défaut, 0 = remboursement OK) présente uniquement dans `application_train`.

Avant d’implémenter, il faut **documenter la structure de chaque fichier et les jointures** (qui est la table “pivot”, quelles clés pour chaque table). Ce document peut être généré ou maintenu à partir de `HomeCredit_columns_description.csv` + inspection des CSV (noms de colonnes, types).

---

## 2. Tables et vues (réception raw, nettoyage, master)

### 2.1 Principe

- **Tables physiques `raw_*`** : une par fichier source (ex. `raw_application_train`, `raw_bureau`, …). Schéma dérivé de la structure connue des CSV (noms, types SQL adaptés). Utilisées uniquement pour ingestion et rejeu.
- **Vues `v_clean_*`** : une par source. Nettoyage (TRIM, CAST, gestion des valeurs manquantes, normalisation des libellés métier → voir section 4). Pas de jointure.
- **Vue maîtresse** (ex. `v_master_application` ou `v_master_clean`) : jointures entre les vues clean (application train + agrégats bureau, previous_application, etc.) pour obtenir **un enregistrement par `SK_ID_CURR`** avec toutes les colonnes nécessaires au feature engineering et au modèle.
- **Vue feature engineering** (ex. `v_features_engineering`) : calculs SQL de features dérivées (ratios, agrégats temporels, etc.) à partir de la vue maîtresse. Optionnel selon ce qu’on préfère faire en SQL vs Python.

Les noms de colonnes dans les vues peuvent déjà suivre une convention “métier” (ex. noms lisibles, cohérents avec le schéma décrit plus bas).

### 2.2 Ordre de mise en place

1. **Documenter** : structure des CSV (colonnes, types, clés) + graphe de jointures (qui rejoint qui, sur quelle clé).
2. **Définir** le schéma SQL des tables `raw_*` (aligné sur les CSV ou sur le dictionnaire de données).
3. **Implémenter** dans un CreditScoringCleaner (ou équivalent) : `create_raw_*`, `clean_*` (vues), `create_master_view`, éventuellement `create_features_view`.
4. **Ingestion** : chargement des CSV vers les tables `raw_*` (avec harmonisation des clés si besoin, comme `emp_id` dans AttritionCleaner).

---

## 3. Schéma métier ↔ technique (registre d’attributs)

### 3.1 Objectif

- **Une seule source de vérité** pour : nom métier, nom technique (pour le modèle / API), type (catégoriel, numérique, binaire, date), encodage (one_hot, ordinal, target, none), transformation numérique (none, log, standard, robust), et pour les catégoriels : mapping des valeurs métier → valeurs techniques (codes ou libellés normalisés).
- **Dérivation automatique** des listes “colonnes à OHE”, “à log”, “à standard”, etc. : plus de listes en dur dans le code ; on filtre le registre par `encoding` / `transform`.
- **Traçabilité** : en préparation, entraînement et serving, les mêmes définitions sont utilisées (métier → technique pour les noms et les valeurs).

### 3.2 Où stocker le schéma (recommandation)

| Option | Avantages | Inconvénients |
|--------|-----------|----------------|
| **Fichier (YAML/JSON)** | Versionné, lisible, facile à éditer, pas de dépendance DB au démarrage | Pas de requêtes SQL directes ; il faut charger en mémoire. |
| **Code (Pydantic / dataclasses)** | Typage, validation, IDE, un seul langage | Moins lisible pour des non-développeurs ; évolutions = déploiement. |
| **Base de données** | Centralisation, possible UI, partage multi-outils | Dépendance DB, migrations, moins pratique pour versionner “diff” de schéma. |

**Recommandation** : **schéma en code (Pydantic recommandé)** comme source de vérité, optionnellement **export/sync vers une table ou un fichier** (DB ou YAML) pour l’outillage (ex. API, monitoring, documentation).

- **Pourquoi Pydantic** : validation des types et des contraintes, sérialisation JSON/YAML facile, réutilisation dans FastAPI plus tard. Un modèle par “attribut” ou un modèle “AttributeSpec” avec des enums pour encoding/transform.
- **Classe par variable catégorielle** : chaque catégoriel peut être décrit par un petit modèle (nom, valeurs autorisées, mapping métier → technique). En pratique, une **liste de spécifications d’attributs** (registre) suffit ; une classe par attribut n’est nécessaire que si on veut des comportements spécifiques (validation métier, règles métier). Pour rester simple et maintenable : **un seul type “AttributeSpec”** (ou “ColumnSpec”) avec un champ pour le mapping des valeurs (dict ou liste de paires).

### 3.3 Structure proposée du registre (conceptuel)

Pour **chaque colonne** (ou variable métier) :

- **Identité** : `name_metier`, `name_technique` (nom pour le modèle / pipeline).
- **Type** : `categorical` | `numerical` | `binary` | `datetime` | `id` (à exclure du modèle).
- **Rôle** : `feature` | `target` | `identifier` | `drop`.
- **Encodage** (si catégoriel) : `one_hot` | `ordinal` | `target_encoding` | `none`.
- **Transformation** (si numérique) : `none` | `log` | `standard` | `robust`.
- **Valeurs** (si catégoriel) : mapping `{ valeur_metier: valeur_technique }` ou liste ordonnée pour ordinal. Optionnel : valeurs par défaut pour inconnus.
- **Contraintes** : nullable, plage, etc. (optionnel).

À partir de ce registre :

- **Liste OHE** = attributs avec `type=categorical` et `encoding=one_hot`.
- **Liste log** = attributs avec `type=numerical` et `transform=log`.
- **Liste standard** = attributs avec `type=numerical` et `transform=standard`.
- **Liste drop** = `role in (identifier, drop)` ou colonnes constantes identifiées.

Cela remplace complètement les listes en dur de l’ancien `FeatureConfig`.

---

## 4. Couche métier ↔ couche technique (noms et valeurs)

### 4.1 Noms de colonnes

- **Métier** : libellés explicites (ex. “Type de contrat”, “Revenu total”) ou noms “business” (ex. `NAME_CONTRACT_TYPE`).
- **Technique** : noms stables pour le modèle (snake_case, sans espaces, cohérents avec l’API). Le registre définit `name_metier` → `name_technique` ; les vues SQL ou le preprocessing Python peuvent renommer selon ce mapping.

### 4.2 Valeurs catégorielles

- **Métier** : valeurs telles qu’elles arrivent (ex. “Cash loans”, “Revolving loans”, “M”, “F”).
- **Technique** : codes ou libellés normalisés pour le modèle (ex. 0/1, ou “cash” / “revolving”). Le mapping est dans le registre (par attribut). En Phase 1 (vues SQL) : on peut appliquer des `CASE WHEN` basés sur ce mapping. En Phase 2 (sklearn) : le pipeline peut utiliser un mapper qui s’appuie sur le même registre (ou sur des artefacts fittés qui en sont dérivés).

Pour **chaque attribut catégoriel**, avoir une définition (classe ou entrée de registre) avec le mapping garantit cohérence et maintenabilité ; inutile de dupliquer la logique en SQL et en Python si on lit la même source (le registre).

---

## 5. Feature engineering (Phase 2) : listes automatiques et pipeline

### 5.1 Dérivation automatique des listes

- **Entrée** : le **registre d’attributs** (Pydantic ou structure chargée depuis YAML/JSON) + éventuellement le **DataFrame** après chargement (pour ne garder que les colonnes présentes).
- **Règles** :
  - Colonnes à supprimer : `role in (identifier, drop)` + colonnes constantes (détectées sur les données ou marquées dans le registre).
  - `COLS_ONE_HOT` = colonnes du registre avec `encoding=one_hot` et présentes dans les données.
  - `COLS_TO_LOG` = colonnes avec `transform=log`, idem.
  - `COLS_STANDARD` = colonnes avec `transform=standard`.
  - `COLS_TO_ROBUST` = colonnes avec `transform=robust`.
- **Option** : détection automatique de type (object → catégoriel, number → numérique) avec **validation** contre le registre (alerte si une colonne réelle n’est pas dans le registre ou a un type incohérent).

Ainsi, plus de listes en dur : tout vient du schéma.

### 5.2 Pipeline sklearn (ColumnTransformer)

- Construction du `ColumnTransformer` à partir des listes dérivées du registre (comme dans l’ancien phase2, mais les listes viennent du registre).
- Pour les catégoriels : OneHotEncoder (ou ordinal/target) ; les mappings métier → technique peuvent être appliqués **avant** l’encodeur (étape de “normalisation” des valeurs depuis le registre), puis imputation + encodeur.
- Sauvegarde des artefacts : preprocessor, noms de colonnes entrée/sortie, et **optionnellement** une copie du registre (ou de la config utilisée) pour rejouer exactement le même preprocessing en production.

---

## 6. Où placer les fichiers (structure recommandée)

- **`src/data/`** ou **`src/preparation/`** :  
  - `schema.py` ou `registry.py` : modèles Pydantic + registre des attributs (définition des colonnes, encodages, transformations, mappings).  
  - `credit_scoring_cleaner.py` : classe type AttritionCleaner (tables raw, vues clean, vues master/features, ingestion CSV → DB).  
  - `make_dataset.py` : script d’orchestration Phase 1 (charger CSV, créer tables, vues, remplir raw, éventuellement exporter un CSV intermédiaire).
- **`src/features/`** ou **`src/preprocessing/`** :  
  - `config.py` ou `feature_config.py` : chargement du registre + construction des listes (OHE, log, standard, etc.) à partir du schéma.  
  - `build_features.py` ou `phase2_pipeline.py` : chargement des données (depuis DB ou CSV), nettoyage/sélection, construction du ColumnTransformer à partir du config dérivé du registre, fit_transform, sauvegarde des artefacts.
- **`config/`** ou **`schemas/`** (optionnel) :  
  - Fichier YAML/JSON du registre si on préfère le maintenir en données plutôt qu’en code ; sinon tout en Pydantic dans `src/data/schema.py`.
- **`docs/`** :  
  - Un document (ou script) qui décrit la structure des fichiers raw et les jointures (généré à partir de `HomeCredit_columns_description.csv` + métadonnées des CSV).

---

## 7. Ordre d’implémentation suggéré

1. **Documenter** la structure des CSV et les jointures (fichier + schéma des clés).
2. **Définir le registre d’attributs** (Pydantic) pour `application_train` au minimum (colonnes + types + encoding/transform + mappings pour les catégoriels principaux). Étendre ensuite aux autres tables si besoin pour la vue maîtresse.
3. **Implémenter CreditScoringCleaner** : création des tables `raw_*` (au moins application_train/test, bureau, previous_application), vues `v_clean_*`, vue maîtresse, éventuellement vue features. Ingestion des CSV existants.
4. **Implémenter le configurateur de features** : lecture du registre → listes OHE, log, standard, drop. Option : détection automatique avec validation par le registre.
5. **Adapter le pipeline Phase 2** : utiliser ce configurateur au lieu de `FeatureConfig` à listes en dur ; appliquer les mappings métier → technique (noms et valeurs) depuis le registre avant/après le ColumnTransformer selon le design choisi.
6. **Tests** : tests unitaires sur le registre (dérivation des listes), sur le cleaner (création de tables/vues, pas de régression sur les jointures), sur le pipeline Phase 2 (shape, pas de fuite de cible).

---

## 8. Points de vigilance (alignés avec la mission et les bonnes pratiques)

- **Déséquilibre des classes** : à traiter en Phase 3 (entraînement) ; en Phase 2, ne pas supprimer la cible ni la sur/sous-échantillonner, uniquement la séparer X/y.
- **Valeurs manquantes** : stratégie documentée (imputation médiane/mode, ou “Missing” pour catégoriels) et alignée avec le registre si on veut marquer des colonnes “impute_median” etc.
- **Reproductibilité** : versionner le registre (et/ou le fichier de config) avec le code ; sauvegarder avec les artefacts du modèle la config utilisée pour le preprocessing.
- **Coût métier (FN vs FP)** : hors scope Phase 2 ; à prendre en compte en Phase 3 (métrique, seuil de décision). Le registre peut prévoir un champ “target” et éventuellement “cost_weight” pour usage ultérieur.

---

## 9. Résumé des choix recommandés

| Sujet | Recommandation |
|--------|----------------|
| **Tables / vues** | Tables `raw_*` par fichier source ; vues `v_clean_*` ; une vue maîtresse (jointures) ; une vue feature engineering optionnelle. |
| **Schéma métier/technique** | Registre d’attributs en **code (Pydantic)** comme source de vérité ; optionnel : export YAML/DB pour outillage. |
| **Listes OHE / log / standard** | **Dérivation automatique** depuis le registre (filtrage par encoding/transform) ; plus de listes en dur. |
| **Classe par catégoriel** | Une **spécification par attribut** dans le registre (avec mapping valeurs) ; une classe Pydantic par type d’entité (ex. AttributeSpec) suffit, pas obligatoire une classe par variable. |
| **Stockage du schéma** | **Code (Pydantic)** versionné ; DB ou YAML en sortie optionnelle pour serving/documentation. |
| **Base commune** | Adapter **AttritionCleaner** → **CreditScoringCleaner** (engine, raw, clean, master, features) ; Phase 1 et Phase 2 en scripts/modules séquentiels comme dans docs/old. |

Cette planification permet d’avoir une base **maintenable, adaptable à d’autres projets** (changer le registre et les sources suffit) et **traçable** (métier ↔ technique documenté et utilisé partout). Dès que cette architecture est validée, l’implémentation peut suivre l’ordre de la section 7.
