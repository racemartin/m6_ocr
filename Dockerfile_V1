# =============================================================================
# Dockerfile — Image multi-stage pour HuggingFace Spaces
# Stage 1 (builder) : installation des dépendances dans un venv isolé
# Stage 2 (runtime) : image légère avec uniquement ce qui est nécessaire
#
# Build local :
#   docker build -t m7-scoring-credit .
#   docker run -p 7860:7860 m7-scoring-credit
#
# HuggingFace Spaces utilise le port 7860 par défaut.
# =============================================================================


# =============================================================================
# Stage 1 — builder : installation des dépendances Python
# =============================================================================
FROM python:3.11-slim AS builder

# -- Répertoire de travail dans le builder ------------------------------------
WORKDIR /app

# -- Installation des outils système nécessaires à la compilation -------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc             \
    && rm -rf /var/lib/apt/lists/*

# -- Création d'un environnement virtuel isolé --------------------------------
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# -- Copie du fichier de dépendances avant le code (cache Docker optimal) -----
COPY requirements.txt .

# -- Installation des dépendances Python --------------------------------------
RUN pip install --upgrade pip --quiet \
    && pip install --no-cache-dir -r requirements.txt


# =============================================================================
# Stage 2 — runtime : image de production légère
# =============================================================================
FROM python:3.11-slim AS runtime

# -- Métadonnées de l'image ---------------------------------------------------
LABEL maintainer="Prêt à Dépenser"
LABEL description="API de scoring crédit — M7 MLOps Partie 2/2"
LABEL version="2.0.0"

# -- Répertoire de travail ----------------------------------------------------
WORKDIR /app

# -- Utilisateur non-root pour la sécurité ------------------------------------
RUN useradd --create-home --shell /bin/bash appuser

# -- Copie du virtualenv depuis le builder ------------------------------------
COPY --from=builder /app/.venv /app/.venv

# -- Copie du code source et des artefacts modèle ----------------------------
COPY src/            ./src/
COPY config.py       ./config.py
COPY model_artifact/ ./model_artifact/

# -- Création des répertoires nécessaires à l'exécution ----------------------
RUN mkdir -p monitoring \
    && chown -R appuser:appuser /app

# -- Activation du venv dans le PATH -----------------------------------------
ENV PATH="/app/.venv/bin:$PATH"

# -- Variables d'environnement par défaut -------------------------------------
ENV MODEL_BACKEND=onnx
ENV SEUIL_DECISION=0.35
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# -- Passage à l'utilisateur non-root ----------------------------------------
USER appuser

# -- Port exposé (HuggingFace Spaces = 7860) ----------------------------------
EXPOSE 7860

# -- Healthcheck Docker -------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request; \
         urllib.request.urlopen('http://localhost:7860/health')"

# -- Commande de démarrage ----------------------------------------------------
CMD ["uvicorn", "src.api.main:application", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1"]
