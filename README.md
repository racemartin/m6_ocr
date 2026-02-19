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
# Créer le fichier d'environnement à partir du template
cp .env.example .env

# Note : Assurez-vous que DATABASE_URL pointe vers votre base SQLite/PostgreSQL
```


## 📝 Choix Techniques & Justifications (Complété)

* **`uv`** : Choisi pour la gestion ultra-rapide des dépendances et la reproductibilité garantie par le fichier lock, assurant la stabilité de l'environnement entre le développement local et le déploiement. Il remplace avantageusement `pip` par sa rapidité (10-100x) et sa gestion native des environnements virtuels.
* **`Docker`** : Utilisé pour conteneuriser l'infrastructure (PostgreSQL, MLflow) et l'application, garantissant une isolation totale et une parité stricte entre les environnements de test et de production. Cela permet d'éliminer le syndrome du "ça marche sur ma machine".
* **`FastAPI` + `Pydantic**` : Sélectionné pour sa performance asynchrone et la validation rigoureuse des données entrantes via des schémas de données stricts (Data Integrity), évitant les erreurs de modèle dues à des données corrompues ou mal formatées.
* **`MLflow` + `PostgreSQL**` : Implémenté pour assurer le tracking systématique des expérimentations et le versioning des modèles. L'utilisation de PostgreSQL comme backend garantit la persistance et l'intégrité des métadonnées de l'entraînement à long terme.
* **`ydata-profiling`** : Implémenté pour automatiser l'analyse exploratoire (EDA) de manière exhaustive, permettant de détecter instantanément les valeurs manquantes, les corrélations et les dérives de données (Data Drift) avant l'entraînement.



## 📚 Documentation

### Steps realized during Instalation

1. [Étape 1 : Initialisation et Structure Cookiecutter](docs/installation/01_Cookiecutter.md)
2. [Étape 2 : Configuration du pyproject.toml](docs/installation/02_pyproject.toml.md)
3. [Étape 3 : Gestion de l'Environnement Virtuel (uv)](docs/installation/02_pyproject.toml.md)
4. [Étape 4 : Database PostgreSQL & Configuration](docs/installation/03_PostgreSQL_Database.md)


### Prochainement
- [Guide d'utilisation](docs/usage.md)



## 🏗️ Architecture



### Notebooks Importants (Ordre d'implémentation)




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


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
