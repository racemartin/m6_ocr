---
title: Pret a depenser App
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---


Initiez-vous au MLOps (partie 1/2) MLflow Gestion du cycle de vie des modèles
==============================

<img src="docs\images\pret_a_depenser_logo.png" alt="Pret a Depenser" width="200">

## Architecture de Scoring Binaire pour les Prêt à la Consommation

Construire et optimiser un modèle de classification binaire suivant le cycle de vie avec MLFlow


## 🛠️ Technologies (Complété)

* **Python 3.12+** : Langage de programmation principal choisi pour son écosystème mature en Data Science.
* **uv** : Gestionnaire de paquets ultra-rapide utilisé pour garantir la reproductibilité de l'environnement via un fichier de verrouillage (`uv.lock`).
* **FastAPI** : Framework web asynchrone utilisé pour l'API Gateway, offrant une documentation OpenAPI (Swagger) automatique.
* **Pydantic** : Validation des données et typage strict pour sécuriser les échanges entre l'utilisateur et le modèle.
* **BentoML / MLflow** : **MLflow** est utilisé ici pour le cycle de vie complet du modèle : tracking des hyperparamètres, versioning des artefacts et registre de modèles centralisé.
* **Docker** : Conteneurisation de l'ensemble des services (PostgreSQL, MLflow, API) pour assurer une portabilité totale entre le développement et la production.
* **PostgreSQL** : Base de données relationnelle servant de backend de persistance pour MLflow, garantissant l'intégrité des métadonnées des expériences.
* **Scikit-learn, XGBoost, CatBoost** : Bibliothèques de modélisation pour entraîner des algorithmes performants sur des données tabulaires.
* **SHAP** : Utilisé pour l'interprétabilité locale et globale, permettant d'expliquer chaque décision du modèle de crédit (obligatoire pour la conformité bancaire).
* **Pandas, Seaborn, ydata-profiling** : Analyse, visualisation et génération automatique de rapports d'exploration de données (EDA).

## 📦 Installation

```bash
# Cloner le dépôt
git clone https://github.com/racemartin/m6_ocr.git
cd m6_ocr

# Installer les dépendances et créer l'environnement virtuel avec uv
uv sync

# Vérifier l'installation des composants critiques
uv run python -c "import pandas, fastapi, mlflow; print('✅ Environnement prêt !')"
```

## 🔒 Sécurité
[Instructions d'authentification](docs/installation/security.md)

## ⚙️ Configuration

```bash
# -----------------------------------------------------------------------------
# Créer le fichier d'environnement à partir du template
cp .env.example .env

# -----------------------------------------------------------------------------
# 2. Redemarrer les services a travers de Docker
docker compose up -d postgres mlflow pgadmin

# verification des services 
docker ps
CONTAINER ID   IMAGE                           COMMAND                  CREATED       STATUS        PORTS                                         NAMES
e0cdeb4de8bd   dpage/pgadmin4                  "/entrypoint.sh"         4 hours ago   Up 2 hours   0.0.0.0:8088->80/tcp                          pgadmin_service
bece6dbc93ff   ghcr.io/mlflow/mlflow:v2.16.0   "/bin/sh -c ' pip in…"   4 hours ago   Up 2 hours   0.0.0.0:5001->5000/tcp, [::]:5001->5000/tcp   m6_ocr-mlflow-1
04030bb6618f   postgres:15                     "docker-entrypoint.s…"   4 hours ago   Up 2 hours   0.0.0.0:5433->5432/tcp                        postgres_db

# -----------------------------------------------------------------------------
# Note : Assurez-vous que DATABASE_URL pointe vers votre base SQLite/PostgreSQL
# 3. Initialiser l'infrastructure (Base de données). Une sole fois.
# C'est ici que l'on crée les tables 'vides'

# Avec l'aplication "postgres"
# psql -U postgres -f scripts/database/create_db_infrastructure_schema.sql
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -f scripts/database/create_db_infrastructure_schema.sql

# Avec un script Python
uv run python scripts/database/create_db_infrastructure_schema.py
```

## 🚀 Flux de Travail (Pipeline de Données)

**Important :** Vous devez impérativement exécuter ces phases dans l'ordre pour générer les artefacts nécessaires aux services.

```bash
# Phase 1 :  Analyse Exploratoire (Nettoyage et Préparation)
make phase1_1
make phase1_2

# Phase 2 : Feature Engineering (Preprocessing)
make phase2

# Phase 3 : Enregistrer Expérimentations dans MLFlow
# Options pour la phase 3 (edition de src/pipelines/phase2_feature_engineering.py): 
# * Aller plus vite: DEBUG_MODE = True, DEBUG_LIMIT = 10000 )
# * Utiliser touts les echantillons: DEBUG_MODE = False
make phase3

# Phase 4 :  Optimisation hyperparamètres
make phase4

# Lancer l'interface utilisateur MLflow pour suivre vos expérimentations et métriques
# (Accessible par défaut sur le port 5001)
http://localhost:5001/

# Démarrer le tableau de bord Optuna pour visualiser l'optimisation des hyperparamètres
# (Analyse en temps réel de la recherche bayésienne et du pruning)
uv run python srv_optuna_monitor.py

# Accéder à l'interface graphique d'Optuna pour comparer les performances des essais
# (Visualisation des courbes d'apprentissage et de l'importance des paramètres)
http://localhost:8082/

```


## 🧪 Tests & Santé du Système (Phase 3 Focus)

Cette suite de tests garantit que le passage du **Feature Engineering** à l'**Entraînement** est robuste, sans fuite de données (*Data Leakage*) et avec une traçabilité totale.

| Fichier de Test (.py) | Composant Cible | Type de Validation | Focus MLOps |
| --- | --- | --- | --- |
| `test_smoke.py` | **Environnement** | Test de "fumée" rapide | Vérification des imports et du venv |
| `test_03_database.py` | **Infrastructure DB** | Connexion et schémas SQL | Disponibilité de PostgreSQL |
| `test_phase3_database.py` | **Data Access Layer** | Lecture des vues agrégées | Intégrité des 52 colonnes calculées |
| `test_phase3_mlflow.py` | **Tracking Server** | Connexion et logging d'artefacts | Enregistrement du `preprocessor.pkl` |
| `test_phase3_pipeline.py` | **Workflow End-to-End** | Fit / Transform du Pipeline | Absence de `NaN` et alignement colonnes |
| `test_phase3_models.py` | **ML Logic** | Entraînement et prédictions | Calcul du F2-Score et Coût Métier |
| `conftest.py` | **Test Fixtures** | Configuration globale des tests | Mocking des données et moteurs SQL |

---

## 🚀 Exemples d'Exécution avec `uv`

Comme pour les phases de données, nous utilisons `uv` pour garantir que les tests s'exécutent dans l'environnement isolé du projet.

### 1. Tests de Santé Globale

```bash
# Lancer TOUS les tests pour vérifier la santé du projet
uv run pytest

# Lancer uniquement les tests de "fumée" (vérification rapide)
uv run pytest tests/test_smoke.py -v

```

### 2. Validation de l'Infrastructure et du Tracking

```bash
# Vérifier que le pipeline peut communiquer avec MLflow
uv run pytest tests/test_phase3_mlflow.py -s

# Valider l'accès aux données dans PostgreSQL
uv run pytest tests/test_phase3_database.py

```

### 3. Tests de Logique de Pipeline (Crucial)

```bash
# Vérifier la transformation des données et l'imputation (le fameux fix du log1p)
uv run pytest tests/test_phase3_pipeline.py -v

# Tester l'entraînement des modèles et le calcul des métriques métier
uv run pytest tests/test_phase3_models.py

```

### 4. Mode Débogage 

```bash
# Lancer un test spécifique avec affichage des logs et arrêt au premier échec
uv run pytest -x -s tests/test_phase3_pipeline.py

```


## 📝 Choix Techniques & Justifications (Complété)

* **`uv`** : Choisi pour la gestion ultra-rapide des dépendances et la reproductibilité garantie par le fichier lock, assurant la stabilité de l'environnement entre le développement local et le déploiement. Il remplace avantageusement `pip` par sa rapidité (10-100x) et sa gestion native des environnements virtuels.
* **`Docker`** : Utilisé pour conteneuriser l'infrastructure (PostgreSQL, MLflow) et l'application, garantissant une isolation totale et une parité stricte entre les environnements de test et de production. Cela permet d'éliminer le syndrome du "ça marche sur ma machine".
* **`FastAPI` + `Pydantic**` : Sélectionné pour sa performance asynchrone et la validation rigoureuse des données entrantes via des schémas de données stricts (Data Integrity), évitant les erreurs de modèle dues à des données corrompues ou mal formatées.
* **`MLflow` + `PostgreSQL**` : Implémenté pour assurer le tracking systématique des expérimentations et le versioning des modèles. L'utilisation de PostgreSQL comme backend garantit la persistance et l'intégrité des métadonnées de l'entraînement à long terme.
* **`ydata-profiling`** : Implémenté pour automatiser l'analyse exploratoire (EDA) de manière exhaustive, permettant de détecter instantanément les valeurs manquantes, les corrélations et les dérives de données (Data Drift) avant l'entraînement.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## 👤 Auteur

**Rafael Cerezo Martín**

* Email: [rafael.cerezo.martin@icloud.com](mailto:rafael.cerezo.martin@icloud.com)
* GitHub: [@racemartin](https://github.com/racemartin)

## 📄 Licence

MIT License - voir le fichier [LICENSE](https://www.google.com/search?q=LICENSE) pour plus de détails.


