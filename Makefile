# =============================================================================
# Makefile — Projet M6/M7 MLOps : Scoring Crédit « Prêt à Dépenser »
# Fusionne les commandes de la Partie 1 (entraînement) et de la
# Partie 2 (API, déploiement, supervision).
#
# Utilisation :
#   make help           → liste toutes les commandes disponibles
#   make train_all      → pipeline complet phases 1 → 4
#   make api            → démarre l'API FastAPI (port 8001)
#   make tests          → lance tous les tests avec couverture ≥ 80%
# =============================================================================

# =============================================================================
# VARIABLES GLOBALES
# =============================================================================

# -- Identification du projet -------------------------------------------------
# PROJECT_DIR  := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIR    := $(CURDIR)
# Nom du projet courant
PROJECT_NAME  = m7_ocr
# Bucket S3 (optionnel)
BUCKET        = your-bucket-name
# Profil AWS (optionnel)
PROFILE       = default

# -- Interpréteur Python : priorité à uv si disponible -----------------------
PYTHON := uv run python
UV_PIP := uv pip install

# -- Ports des services -------------------------------------------------------
# API FastAPI (≠ 8000 pour éviter conflits)
PORT_API     = 8001
# Dashboard Streamlit
PORT_STREAM  = 8501
# Serveur MLflow (dev local)
PORT_MLFLOW  = 5001

MLFLOW_BACKEND_STORE=sqlite:///mlflow/mlflow.db
MLFLOW_ARTIFACT_PATH=./mlflow/artifacts

# -- Backend modèle -----------------------------------------------------------
# "onnx" (prod) | "mlflow" (dev)
BACKEND      = onnx

# -- Détection OS pour compatibilité ------------------------------------------
ifeq ($(OS),Windows_NT)
    SHELL := cmd.exe
endif

# -- Détection conda ----------------------------------------------------------
ifeq (,$(shell where conda 2>nul))
    HAS_CONDA = False
else
    HAS_CONDA = True
endif


# =============================================================================
# Cibles déclarées sans fichier cible (always run)
# =============================================================================
.PHONY: help requirements data clean lint sync_data_to_s3 sync_data_from_s3  \
        create_environment test_environment                                  \
        train_all phase1_1 phase1_2 phase2 phase3 phase4                     \
        mlflow export-model convert-onnx drift profile                       \
        api dashboard docker-build docker-up docker-dev                      \
        tests tests-unit tests-integration install


# =============================================================================
# Cible par défaut
# =============================================================================
.DEFAULT_GOAL := help


# ##############################################################################
# 0. AIDE
# ##############################################################################

# =============================================================================
help:
	@echo ""
	@echo "============================================================================"
	@echo "COMMANDES — M7 MLOps : Scoring Crédit (Parties 1 et 2)"
	@echo "============================================================================"
	@echo ""
	@echo "  ── ENVIRONNEMENT ────────────────────────────────────────────────────────"
	@echo "  make requirements       Installer les dépendances Python"
	@echo "  make create_environment Créer l'environnement conda ou virtualenv"
	@echo "  make test_environment   Vérifier que l'environnement est correct"
	@echo ""
	@echo "  ── DONNÉES & ENTRAÎNEMENT (Partie 1) ────────────────────────────────────"
	@echo "  make train_all          Pipeline complet : phases 1 → 4"
	@echo "  make phase1_1           Phase 1.1 : injection des données brutes"
	@echo "  make phase1_2           Phase 1.2 : vues SQL + FE + énumérations"
	@echo "  make phase2             Phase 2   : feature engineering sklearn"
	@echo "  make phase3             Phase 3   : entraînement + MLflow tracking"
	@echo "  make phase4             Phase 4   : optimisation bayésienne (Optuna)"
	@echo ""
	@echo "  ── MODÈLE & DÉPLOIEMENT (Partie 2) ──────────────────────────────────────"
	@echo "  make export-model       Exporter meilleur modèle MLflow → model_artifact/"
	@echo "  make convert-onnx       Convertir pipeline .joblib → ONNX"
	@echo "  make drift              Analyse de drift Evidently AI"
	@echo "  make profile            Profilage latences ONNX (500 requêtes)"
	@echo ""
	@echo "  ── SERVICES LOCAUX ───────────────────────────────────────────────────────"
	@echo "  make api                API FastAPI   → http://localhost:$(PORT_API)"
	@echo "  make dashboard          Streamlit     → http://localhost:$(PORT_STREAM)"
	@echo "  make docker-up          Tous les services Docker (prod)"
	@echo "  make docker-dev         Services Docker avec MLflow (dev)"
	@echo ""
	@echo "  ── QUALITÉ & NETTOYAGE ───────────────────────────────────────────────────"
	@echo "  make tests              Tests complets (couverture ≥ 80%)"
	@echo "  make tests-unit         Tests unitaires uniquement"
	@echo "  make tests-integration  Tests d'intégration API uniquement"
	@echo "  make lint               Vérification style (ruff)"
	@echo "  make clean              Supprimer fichiers temporaires"
	@echo "============================================================================"
	@echo ""


# ##############################################################################
# 1. ENVIRONNEMENT
# ##############################################################################

# =============================================================================
requirements:
	@echo "Instalando dependencias con uv..."
	uv pip install -U pip setuptools wheel
	uv pip install -r requirements.txt
	@echo "============================================================================"
	@echo "Dépendances installées avec succès."
	@echo "============================================================================"

# =============================================================================
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo "Conda détecté — création de l'environnement : $(PROJECT_NAME)"
	conda create --name $(PROJECT_NAME) python=3.12 -y
	@echo ">>> Activez avec : conda activate $(PROJECT_NAME)"
else
	@echo "Utilisation de uv pour créer l'environnement virtuel..."
	uv venv --python 3.12
	@echo ">>> Environnement créé. Pour l'activer en PowerShell :"
	@echo ">>> .\.venv\Scripts\Activate.ps1"
endif

# =============================================================================
test_environment:
	$(PYTHON) test_environment.py

# =============================================================================
install: requirements


# ##############################################################################
# 2. DONNÉES & ENTRAÎNEMENT (Partie 1)
# ##############################################################################

# =============================================================================
data: requirements
	$(PYTHON) src/data/make_dataset.py data/raw data/processed

# =============================================================================
train_all: phase1_1 phase1_2 phase2 phase3 phase4
	@echo "============================================================================"
	@echo "Pipeline d'entraînement complet (phases 1 à 4) terminé."
	@echo "============================================================================"

# =============================================================================
phase1_1:
	@echo "============================================================================"
	@echo "PHASE 1.1 — Injection des données brutes (CSV → PostgreSQL)"
	@echo "============================================================================"
ifdef UV
	uv run phase1_1
else
	$(PYTHON) -m src.pipelines.phase1_1_inject_raw
endif

# =============================================================================
phase1_2:
	@echo "============================================================================"
	@echo "PHASE 1.2 — Vues SQL + Feature Engineering + Énumérations"
	@echo "============================================================================"
ifdef UV
	uv run phase1_2
else
	$(PYTHON) -m src.pipelines.phase1_2_views_fe_enum
endif

# =============================================================================
phase2:
	@echo "============================================================================"
	@echo "PHASE 2 — Feature Engineering sklearn"
	@echo "============================================================================"
ifdef UV
	uv run phase2
else
	$(PYTHON) -m src.pipelines.phase2_feature_engineering
endif

# =============================================================================
phase3:
	@echo "============================================================================"
	@echo "PHASE 3 — Entraînement des modèles + MLflow tracking"
	@echo "============================================================================"
ifdef UV
	uv run phase3
else
	$(PYTHON) -m src.pipelines.phase3_model_training_mlflow
endif

# =============================================================================
phase4:
	@echo "============================================================================"
	@echo "PHASE 4 — Optimisation bayésienne (Optuna)"
	@echo "============================================================================"
ifdef UV
	uv run phase4
else
	$(PYTHON) -m src.pipelines.phase4_hyperparameter_tuning
endif


# ##############################################################################
# 3. MODÈLE & DÉPLOIEMENT (Partie 2)
# ##############################################################################

# =============================================================================
export-model:
	@echo "============================================================================"
	@echo "EXPORT DU MEILLEUR MODÈLE MLflow → model_artifact/"
	@echo "============================================================================"
	$(PYTHON) scripts/export_best_model.py

# =============================================================================
convert-onnx:
	@echo "============================================================================"
	@echo "CONVERSION ONNX — pipeline .joblib → best_model.onnx"
	@echo "============================================================================"
	$(PYTHON) scripts/convert_onnx.py

# =============================================================================
drift:
	@echo "============================================================================"
	@echo "ANALYSE DE DRIFT — Evidently AI"
	@echo "============================================================================"
	$(PYTHON) scripts/drift_analysis.py
	@echo "Rapport HTML généré : monitoring/drift_report.html"

# =============================================================================
profile:
	@echo "============================================================================"
	@echo "PROFILAGE PERFORMANCE ONNX (500 requêtes)"
	@echo "============================================================================"
	$(PYTHON) optimization/profile_model.py --nb-requetes 500


# ##############################################################################
# 4. SERVICES LOCAUX
# ##############################################################################

# =============================================================================
# LANZAR SERVIDOR MLFLOW (Tracking UI)
# =============================================================================
mlflow:
	@echo "Lanzando servidor de tracking MLflow..."
	@if not exist "mlflow" mkdir mlflow
	@if not exist "mlflow\artifacts" mkdir mlflow\artifacts
	mlflow server \
		--backend-store-uri $(MLFLOW_BACKEND_STORE) \
		--default-artifact-root $(MLFLOW_ARTIFACT_PATH)  \
		--host 0.0.0.0 \
		--port $(PORT_MLFLOW)

# =============================================================================
api:
	@echo "============================================================================"
	@echo "API FastAPI....: http://localhost:$(PORT_API)"
	@echo "Swagger UI.....: http://localhost:$(PORT_API)/docs"
	@echo "BACKEND........: $(BACKEND)"
ifeq ($(OS),Windows_NT)
	set MODEL_BACKEND=$(BACKEND)&& $(PYTHON) -m uvicorn src.api.main:application \
		--host 0.0.0.0 \
		--port $(PORT_API) \
		--reload
else
	MODEL_BACKEND=$(BACKEND) $(PYTHON) -m uvicorn src.api.main:application \
		--host 0.0.0.0 \
		--port $(PORT_API) \
		--reload
endif
	@echo "============================================================================"


# =============================================================================
dashboard:
	@echo "============================================================================"
	@echo "Dashboard Streamlit → http://localhost:$(PORT_STREAM)"
	@echo "============================================================================"
	$(PYTHON) -m streamlit run monitoring/dashboard.py \
		--server.port    $(PORT_STREAM)                \
		--server.address 0.0.0.0

# =============================================================================
# Interface de Simulation (Prise de décision en temps réel)
# =============================================================================
simulate:
	@echo "============================================================================"
	@echo "Simulateur de Crédit → http://localhost:8502"
	@echo "Assurez-vous que l'API est lancée (make api)"
	@echo "============================================================================"
	$(PYTHON) -m streamlit run monitoring/simulator.py \
	   --server.port    8502                      \
	   --server.address 0.0.0.0

# =============================================================================
docker-build:
	@echo "Construction de l'image Docker multi-stage..."
	docker build -t m7-scoring-credit:latest .
	@echo "Image construite : m7-scoring-credit:latest"

# =============================================================================
docker-up:
	@echo "Démarrage de tous les services Docker (production)..."
	docker-compose up --build

# =============================================================================
docker-dev:
	@echo "Démarrage avec profil dev (+ MLflow sur port $(PORT_MLFLOW))..."
	docker-compose --profile dev up --build


# ##############################################################################
# 5. QUALITÉ DU CODE & TESTS
# ##############################################################################

# =============================================================================
tests:
	@echo "============================================================================"
	@echo "TESTS PYTEST — couverture minimale : 80%"
	@echo "============================================================================"
	$(PYTHON) -m pytest tests/ -v            \
		--cov=src                            \
		--cov-report=term-missing            \
		--cov-report=html:reports/coverage   \
		--cov-fail-under=80

# =============================================================================
tests-unit:
	@echo "Tests unitaires uniquement (sans API ni modèle ONNX)..."
	$(PYTHON) -m pytest tests/unit/ -v -m unit

# =============================================================================
tests-integration:
	@echo "Tests d'intégration API uniquement..."
	$(PYTHON) -m pytest tests/integration/ -v -m integration

# =============================================================================
lint:
	@echo "Vérification du style de code avec ruff..."
	ruff check src/ tests/ scripts/ monitoring/ optimization/
	@echo "Style de code validé."


# ##############################################################################
# 6. SYNCHRONISATION S3 (optionnel)
# ##############################################################################

# =============================================================================
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

# =============================================================================
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif


# ##############################################################################
# 7. NETTOYAGE
# ##############################################################################

# =============================================================================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[co]"    -delete                        || true
	find . -type f -name "*.pyo"       -delete                        || true
	rm -rf .pytest_cache                                               || true
	rm -rf .coverage                                                   || true
	rm -rf reports/coverage                                            || true
	rm -rf optimization/rapports                                       || true
	rm -f  monitoring/drift_report.html                                || true
	@echo "Nettoyage terminé."
