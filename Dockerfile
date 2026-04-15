# Utilisation d'une image Python légère basée sur Debian
FROM python:3.12-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# ÉTAPE DE DÉBOGAGE : Confirmation du début de l'installation des outils de base
RUN echo "V1 Début de l'installation des outils de base (setuptools)..." && \
    pip install --no-cache-dir "setuptools<71.0.0" && \
    python -c "import pkg_resources; print('✅ pkg_resources est enfin prêt !')"

# Copie du fichier des dépendances
COPY requirements.txt .

# ÉTAPE DE DÉBOGAGE : Inspection du contenu avant l'installation massive
RUN echo "Contenu du répertoire /app avant l'installation des dépendances :" && ls -R

# Installation des bibliothèques listées dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie de l'intégralité du code source dans le conteneur
COPY . .

RUN chmod -R 777 /app/model_artifact
    
# Configuration du chemin Python pour que les modules internes soient trouvés
ENV PYTHONPATH=/app

# ÉTAPE DE DÉBOGAGE : Vérification de la variable d'environnement
RUN echo "Le PYTHONPATH configuré est : $PYTHONPATH"

# Exposition du port utilisé par l'application
EXPOSE 7860

# Commande de lancement de l'application
CMD ["python", "app.py"]