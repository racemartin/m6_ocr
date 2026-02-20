# Étape 1 — Guide d'Installation Complet
## Docker Desktop (Windows) · PostgreSQL · MLflow · Environnement local

---

## Table des matières

1. [Prérequis système Windows](#1-prérequis-système-windows)
2. [Installation Docker Desktop](#2-installation-docker-desktop)
3. [Configuration WSL 2](#3-configuration-wsl-2)
4. [Configuration Docker Desktop](#4-configuration-docker-desktop)
5. [Cloner le projet et configurer l'environnement](#5-cloner-le-projet-et-configurer-lenvironnement)
6. [Installer UV et les dépendances Python](#6-installer-uv-et-les-dépendances-python)
7. [Lancer PostgreSQL + MLflow avec Docker Compose](#7-lancer-postgresql--mlflow-avec-docker-compose)
8. [Vérifier que tout fonctionne](#8-vérifier-que-tout-fonctionne)
9. [Commandes de gestion quotidienne](#9-commandes-de-gestion-quotidienne)
10. [Résolution de problèmes](#10-résolution-de-problèmes)
11. [Quick Reference Card](#11-quick-reference-card)


---

## 1. Prérequis système Windows

### Configuration minimale

| Composant | Minimum | Recommandé |
|---|---|---|
| OS | Windows 10 version 2004 (build 19041) | Windows 11 |
| RAM | 8 Go | 16 Go |
| Disque libre | 20 Go | 50 Go (données Kaggle) |
| CPU | 64-bit avec virtualisation activée | — |

### Vérifier la version Windows

```powershell
# Dans PowerShell (touche Win + X → Windows PowerShell)
winver
# Doit afficher Windows 10 version 2004 ou supérieur
```

### Activer la virtualisation dans le BIOS (si nécessaire)

Si Docker indique "Virtualization not enabled" :
1. Redémarrer l'ordinateur
2. Appuyer sur `F2`, `F10`, `DEL` ou `ESC` selon le fabricant (au démarrage)
3. Chercher "Intel VT-x" ou "AMD-V" → **Activer**
4. Sauvegarder et redémarrer

### Vérifier que la virtualisation est activée (Windows)

```powershell
# Dans PowerShell
Get-ComputerInfo -property "HyperV*"
# Chercher : HyperVRequirementVirtualizationFirmwareEnabled : True
```

---

## 2. Installation Docker Desktop

### Étape 2.1 — Télécharger Docker Desktop

1. Aller sur **https://www.docker.com/products/docker-desktop/**
2. Cliquer sur **"Download for Windows"**
3. Le fichier `Docker Desktop Installer.exe` se télécharge (~600 Mo)

### Étape 2.2 — Installer Docker Desktop

1. **Double-cliquer** sur `Docker Desktop Installer.exe`
2. Dans l'écran d'installation :
   - ✅ **Cocher** "Use WSL 2 instead of Hyper-V" ← important !
   - ✅ **Cocher** "Add shortcut to desktop"
3. Cliquer **"OK"** → l'installation prend 3-5 minutes
4. Cliquer **"Close and restart"** → l'ordinateur redémarre

### Étape 2.3 — Premier démarrage

Après redémarrage :
1. Docker Desktop démarre automatiquement (icône baleine dans la barre des tâches)
2. Accepter les **Docker Subscription Service Agreement**
3. L'interface Docker Desktop s'ouvre
4. Attendre que le statut passe à **"Docker Desktop is running"** (point vert)

---

## 3. Configuration WSL 2

WSL 2 (Windows Subsystem for Linux) est nécessaire pour Docker sur Windows.
Il est installé automatiquement avec Docker Desktop, mais voici comment vérifier.

### Étape 3.1 — Vérifier WSL 2

```powershell
# Dans PowerShell (en tant qu'administrateur)
# Clic droit sur PowerShell → "Exécuter en tant qu'administrateur"
wsl --status
# Doit afficher : Default Version: 2
```

### Étape 3.2 — Mettre à jour le kernel WSL (si demandé)

Si Docker affiche "WSL 2 installation is incomplete" :

```powershell
# Dans PowerShell administrateur
wsl --update
wsl --set-default-version 2
```

Ou télécharger manuellement :
👉 https://aka.ms/wsl2kernel

### Étape 3.3 — Installer Ubuntu dans WSL (optionnel mais recommandé)

Avoir un terminal Linux natif facilite le travail :

```powershell
# Dans PowerShell
wsl --install -d Ubuntu
# Redémarrer si demandé, puis configurer username/password Ubuntu
```

---

## 4. Configuration Docker Desktop

### Étape 4.1 — Paramètres de ressources

1. Ouvrir Docker Desktop
2. Cliquer sur l'icône **⚙️ Settings** (en haut à droite)
3. Aller dans **Resources → WSL Integration**

```
Settings → Resources
├── CPUs : 4 (ou la moitié de vos cœurs)
├── Memory : 6144 MB (6 Go minimum pour MLflow + PostgreSQL)
├── Swap : 1024 MB
└── Disk image size : 60 GB
```

Subnet 192.168.65.0/24


4. Aller dans **Resources → WSL Integration**
   - ✅ Activer l'intégration avec Ubuntu (si installé)
5. Cliquer **"Apply & Restart"**

### Étape 4.2 — Activer Docker Compose V2

```
Settings → General
  ✅ "Use Docker Compose V2"
```

### Étape 4.3 — Vérifier l'installation

Ouvrir un terminal (**PowerShell** ou **Windows Terminal**) :

```powershell
# Vérifier Docker
docker --version
# Docker version 27.x.x, build xxxxxxx
# Docker version 19.03.12, build 0ed913b8-

# Vérifier Docker Compose
docker compose version
# Docker Compose version v2.x.x
# Docker Compose version v5.0.2

# Test rapide (télécharge et lance un conteneur Hello World)
docker run hello-world
# Doit afficher "Hello from Docker!"
```

---

## 5. Cloner le projet et configurer l'environnement

### Étape 5.1 — Installer Git (si pas déjà installé)

1. Télécharger sur **https://git-scm.com/download/win**
2. Installer avec les options par défaut
3. Vérifier :

```powershell
git --version
# git version 2.x.x.windows.x
```

### Étape 5.2 — Dézipper / Cloner le projet

**Option A — À partir du ZIP fourni :**
```powershell
# Extraire le ZIP dans votre dossier de projets
Expand-Archive -Path "credit-scoring-mlops-etape1.zip" -DestinationPath "C:\Projets\"
cd C:\Projets\credit-scoring-mlops
```

**Option B — Depuis GitHub (quand le dépôt est créé) :**
```powershell
cd C:\Projets
git clone https://github.com/votre-username/credit-scoring-mlops.git
cd credit-scoring-mlops
```

### Étape 5.3 — Créer le fichier `.env`

```powershell
# Copier le template
Copy-Item .env.example .env

# Ouvrir pour éditer (Notepad, VS Code, etc.)
notepad .env
```

Modifier uniquement `POSTGRES_PASSWORD` :

```bash
# .env — modifier cette ligne
POSTGRES_PASSWORD=MonMotDePasseLocal123

# Laisser tout le reste par défaut pour le développement local
MLFLOW_TRACKING_URI=http://localhost:5000
POSTGRES_DB=mlflow_db
POSTGRES_USER=mlflow_user
```

> ⚠️ **Important** : ne jamais committer le fichier `.env` dans Git.
> Il est déjà dans `.gitignore`.

---

## 6. Installer UV et les dépendances Python

UV est le gestionnaire de packages Python utilisé dans ce projet.
Il est 10-100× plus rapide que pip.

### Étape 6.1 — Installer UV sur Windows

```powershell
# Dans PowerShell (pas besoin de droits admin)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Fermer et rouvrir le terminal, puis vérifier
uv --version
# uv 0.x.x
```

### Étape 6.2 — Créer le venv et installer les dépendances

```powershell
# Dans le dossier du projet
cd C:\Projets\credit-scoring-mlops

# Créer le venv et installer toutes les dépendances (avec lockfile)
uv sync

# Vérifier
uv run python --version
# Python 3.12.x
```

### Étape 6.3 — Vérifier l'installation des packages

```powershell
uv run python -c "import mlflow, fastapi, lightgbm, shap; print('Tout OK')"
# Tout OK
```

---

## 7. Lancer PostgreSQL + MLflow avec Docker Compose

C'est l'étape principale — un seul fichier `docker-compose.yml` lance tout.

### Étape 7.1 — Lancer les services

```powershell
# Dans le dossier du projet (où se trouve docker-compose.yml)
cd C:\Projets\credit-scoring-mlops

# Lancer PostgreSQL et MLflow en arrière-plan
docker compose up -d postgres mlflow pgadmin

docker compose up -d pgadmin

```

**Ce qui se passe :**
1. Docker télécharge les images `postgres:15-alpine` et `mlflow:v2.16.2`
   (première fois : ~500 Mo à télécharger, 2-5 minutes selon connexion)
2. Lance le conteneur PostgreSQL sur le port `5432`
3. PostgreSQL crée la base de données `mlflow_db`
4. Lance le conteneur MLflow qui se connecte à PostgreSQL
5. MLflow démarre son serveur sur le port `5000`

### Étape 7.2 — Suivre le démarrage

```powershell
# Voir les logs en temps réel (Ctrl+C pour quitter)
docker compose logs -f

# Ou voir les logs d'un seul service
docker compose logs -f mlflow
docker compose logs -f postgres
```

**Logs attendus pour PostgreSQL :**
```
credit_scoring_postgres  | database system is ready to accept connections
```

**Logs attendus pour MLflow :**
```
credit_scoring_mlflow  | INFO:waitress:Serving on http://0.0.0.0:5000
```

### Étape 7.3 — Vérifier l'état des conteneurs

```powershell
# Voir les conteneurs qui tournent
docker compose ps

# Résultat attendu :
# NAME                        STATUS          PORTS
# credit_scoring_mlflow      Up (healthy)    0.0.0.0:5000->5000/tcp
# credit_scoring_postgres    Up (healthy)    0.0.0.0:5432->5432/tcp
```


### Étape 7.4 — Création de l'utilisateur 'credit_user'

# 1. Création de l'utilisateur 'credit_user'
docker exec -it postgres_db psql -U postgres -c "CREATE USER credit_user WITH PASSWORD 'admin';"

# 2. Création de la base de données 'credit_scoring' appartenant à cet utilisateur
docker exec -it postgres_db psql -U postgres -c "CREATE DATABASE credit_scoring OWNER credit_user;"

# 3. Octroi des droits à 'mlflow_user' (si tu veux qu'il puisse lire les données)
docker exec -it postgres_db psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE credit_scoring TO mlflow_user;"


# 1. Création de l'utilisateur 'credit_user' (via mlflow_user)
docker exec -it postgres_db psql -U mlflow_user -d mlflow_db -c "CREATE USER credit_user WITH PASSWORD 'admin';"

# 2. Création de la base 'credit_scoring'
docker exec -it postgres_db psql -U mlflow_user -d mlflow_db -c "CREATE DATABASE credit_scoring OWNER credit_user;"

# 3. Attribution des droits (pour que mlflow_user puisse aussi y accéder)
docker exec -it postgres_db psql -U mlflow_user -d mlflow_db -c "GRANT ALL PRIVILEGES ON DATABASE credit_scoring TO mlflow_user;"


---

## 8. Vérifier que tout fonctionne

### ✅ Test 1 — Interface MLflow

Ouvrir un navigateur et aller sur :

```
http://localhost:5000
```

**Ce que vous devez voir :**
- L'interface MLflow avec le menu "Experiments" à gauche
- Un experiment par défaut "Default" créé automatiquement
- Aucun run pour l'instant (normal)

### ✅ Test 2 — Connexion PostgreSQL depuis Python

```powershell
uv run python -c "
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST', 'localhost'),
    port=os.getenv('POSTGRES_PORT', 5432),
    database=os.getenv('POSTGRES_DB', 'mlflow_db'),
    user=os.getenv('POSTGRES_USER', 'mlflow_user'),
    password=os.getenv('POSTGRES_PASSWORD'),
)
print('PostgreSQL connecté :', conn.get_dsn_parameters())
conn.close()
"
```

**Résultat attendu :**
```
PostgreSQL connecté : {'dbname': 'mlflow_db', 'user': 'mlflow_user', ...}
```

### ✅ Test 3 — Enregistrer un premier run MLflow

```powershell
uv run python -c "
import mlflow, os
from dotenv import load_dotenv
load_dotenv()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
mlflow.set_experiment('test-installation')

with mlflow.start_run(run_name='test-run-1'):
    mlflow.log_param('test_param', 'hello')
    mlflow.log_metric('test_metric', 0.42)
    print('Run enregistré avec succès !')
"
```

**Vérifier dans l'UI MLflow** → http://localhost:5000
- Un experiment "test-installation" doit apparaître
- Un run "test-run-1" avec le param et la métrique

### ✅ Test 4 — Tests unitaires du projet

```powershell
uv run pytest tests/ -v
```

**Résultat attendu :**
```
tests/test_evaluate.py::TestScoreMetier::test_parfait_retourne_zero      PASSED
tests/test_evaluate.py::TestScoreMetier::test_fn_plus_cher_que_fp        PASSED
tests/test_evaluate.py::TestScoreMetier::test_seuil_bas_reduit_fn        PASSED
tests/test_evaluate.py::TestScoreMetier::test_sortie_entre_0_et_1        PASSED
tests/test_evaluate.py::TestOptimiserSeuil::test_seuil_dans_plage_valide PASSED
...
tests/test_api.py::test_healthcheck_retourne_200                         PASSED
...
================ X passed in X.XXs ================
```

### ✅ Test 5 — Linting du code

```powershell
uv run ruff check .
# Aucun output = aucune erreur
uv run ruff format --check .
# Aucun output = code bien formaté
```

---

## 9. Commandes de gestion quotidienne

### Démarrer l'environnement (chaque matin)

```powershell
cd C:\Projets\credit-scoring-mlops

# Démarrer PostgreSQL + MLflow
docker compose up -d postgres mlflow

# Vérifier que tout tourne
docker compose ps
```

### Arrêter l'environnement (fin de journée)

```powershell
# Arrêter les conteneurs (données conservées)
docker compose stop

# OU arrêter et supprimer les conteneurs (données conservées dans les volumes)
docker compose down
```

### Arrêter ET supprimer toutes les données (reset complet)

```powershell
# ⚠️ ATTENTION : supprime les données MLflow et PostgreSQL
docker compose down -v
```

### Voir les logs

```powershell
docker compose logs mlflow        # Logs MLflow
docker compose logs postgres      # Logs PostgreSQL
docker compose logs -f mlflow     # Logs en temps réel (suivre)
```

### Redémarrer un seul service

```powershell
docker compose restart mlflow
docker compose restart postgres
```

### Accéder à PostgreSQL en ligne de commande

```powershell
# Se connecter au conteneur PostgreSQL
docker exec -it credit_scoring_postgres psql -U mlflow_user -d mlflow_db

# Commandes SQL utiles
\l          -- Lister les bases de données
\dt         -- Lister les tables
\q          -- Quitter
```

### Voir les ressources utilisées

```powershell
docker stats
# Affiche CPU et mémoire de chaque conteneur en temps réel
```

---

## 10. Résolution de problèmes

### ❌ Problème : "Docker Desktop is starting..." ne se termine jamais

**Cause probable** : WSL 2 pas correctement installé.

```powershell
# Solution
wsl --update
wsl --shutdown
# Relancer Docker Desktop
```

### ❌ Problème : Port 5000 déjà utilisé

```
Error: bind: address already in use (port 5000)
```

**Solution** : Trouver et arrêter le processus qui utilise le port :

```powershell
# Trouver le processus
netstat -ano | findstr :5000

# Chercher le PID dans la dernière colonne
# Arrêter le processus (remplacer XXXX par le PID)
taskkill /PID XXXX /F

# OU changer le port MLflow dans docker-compose.yml
# ports: "5001:5000"  ← changer 5000 en 5001
# Et dans .env : MLFLOW_TRACKING_URI=http://localhost:5001
```

### ❌ Problème : Port 5432 déjà utilisé (PostgreSQL local installé)

```powershell
# Option 1 : Arrêter PostgreSQL local
net stop postgresql-x64-15

# Option 2 : Changer le port dans docker-compose.yml
# ports: "5433:5432"  ← utiliser 5433 au lieu de 5432
# Et dans .env : POSTGRES_PORT=5433
```

### ❌ Problème : "Cannot connect to the Docker daemon"

**Solution** : Docker Desktop n'est pas démarré.
1. Chercher Docker Desktop dans le menu Démarrer
2. Attendre que l'icône baleine soit verte dans la barre des tâches

### ❌ Problème : MLflow ne démarre pas (connection refused to PostgreSQL)

```powershell
# Vérifier que PostgreSQL est healthy
docker compose ps
# Si postgres n'est pas "healthy", attendre 30 secondes et réessayer

# Voir les logs PostgreSQL
docker compose logs postgres

# Redémarrer dans l'ordre
docker compose restart postgres
# Attendre 15 secondes
docker compose restart mlflow
```

### ❌ Problème : "uv: command not found"

```powershell
# Fermer et rouvrir le terminal PowerShell
# Si toujours absent, ajouter UV au PATH manuellement :
$env:PATH += ";$env:USERPROFILE\.cargo\bin"

# Ou réinstaller UV
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### ❌ Problème : Tests échouent avec erreur d'import

```powershell
# Vérifier que le package est installé en mode éditable
uv sync
uv run python -c "from credit_scoring.config import SEUIL_DECISION; print('OK')"
```

### ❌ Problème : MLflow UI vide après redémarrage

Les données sont dans le volume Docker `credit_scoring_postgres_data`.
Si vous avez fait `docker compose down -v`, les données ont été supprimées.
Avec `docker compose down` (sans `-v`), les données sont préservées.

---

## 11. Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════╗
║         CREDIT SCORING MLOPS — Quick Reference                  ║
╠══════════════════════════════════════════════════════════════════╣
║  DÉMARRER                                                        ║
║  docker compose up -d postgres mlflow                            ║
║                                                                  ║
║  VÉRIFIER                                                        ║
║  docker compose ps                                               ║
║  http://localhost:5001    ← MLflow UI                           ║
║                                                                  ║
║  ARRÊTER (données conservées)                                    ║
║  docker compose stop                                             ║
║                                                                  ║
║  RESET COMPLET (données supprimées)                              ║
║  docker compose down -v                                          ║
╠══════════════════════════════════════════════════════════════════╣
║  TESTS                                                           ║
║  uv run pytest tests/ -v                                         ║
║  uv run ruff check .                                             ║
╠══════════════════════════════════════════════════════════════════╣
║  PORTS                                                           ║
║  5001  → MLflow Tracking Server                                  ║
║  5433  → PostgreSQL                                              ║
║  8001  → API FastAPI (étape 8)                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  SERVICES                                                        ║
║  credit_scoring_postgres  → base de données MLflow              ║
║  credit_scoring_mlflow    → tracking server + registry          ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Récapitulatif — Étape 1 complétée ✅

Après avoir suivi ce guide, vous disposez de :

| Composant | État | Accès |
|---|---|---|
| Docker Desktop | ✅ Installé et configuré | — |
| WSL 2 | ✅ Activé | Terminal Linux natif |
| PostgreSQL | ✅ En cours d'exécution | `localhost:5432` |
| MLflow UI | ✅ En cours d'exécution | `http://localhost:5000` |
| UV + Python 3.12 | ✅ Installés | `uv run python` |
| Dépendances projet | ✅ Installées | `uv run ...` |
| Tests unitaires | ✅ Passent | `uv run pytest` |
| Structure projet | ✅ Prête | `credit-scoring-mlops/` |

**Prochaine étape →** Étape 2 : Télécharger les données Kaggle et commencer l'EDA

---

*Guide Étape 1 — Credit Scoring MLOps · Prêt à Dépenser*



```
----------------------------------------------------------------------------
DÉBUT DE L'INGESTION DES DONNÉES (CSV -> SQL)
----------------------------------------------------------------------------
  Traitement de..........: application_train.csv
     ↳ Mapping technique appliqué via Registry
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 307,511 lignes · 122 colonnes
     ⏱️  Performance.........: 269.06s (1,142 l/s)
  Statut table raw_application_train..: Renommé & Chargé ✅
  Traitement de..........: application_test.csv
     ↳ Mapping technique appliqué via Registry
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 48,744 lignes · 121 colonnes
     ⏱️  Performance.........: 46.44s (1,049 l/s)
  Statut table raw_application_test..: Renommé & Chargé ✅
  Traitement de..........: bureau.csv
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 1,716,428 lignes · 17 colonnes
     ⏱️  Performance.........: 233.26s (7,358 l/s)
  Statut table raw_bureau..: Chargé brut ✅
  Traitement de..........: bureau_balance.csv
```


```
PS C:\Users\Public\IAE_DELL\pra_dell\m6_ocr> uv run python -m src.pipelines.phase1_preparation
============================================================================
PHASE 1 — PRÉPARATION DES DONNÉES
============================================================================

[0] Connexion à la base de données ...

[1] Ingestion des CSV → tables raw_* ...

----------------------------------------------------------------------------
DÉBUT DE L'INGESTION DES DONNÉES (CSV -> SQL)
----------------------------------------------------------------------------
  Traitement de..........: application_train.csv
     ↳ Mapping technique appliqué via Registry
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 307,511 lignes · 122 colonnes
     ⏱️  Performance.........: 41.19s (7,466 l/s)
  Statut table raw_application_train..: Renommé & Chargé ✅
  Traitement de..........: application_test.csv
     ↳ Mapping technique appliqué via Registry
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 48,744 lignes · 121 colonnes
     ⏱️  Performance.........: 6.15s (7,928 l/s)
  Statut table raw_application_test..: Renommé & Chargé ✅
  Traitement de..........: bureau.csv
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 1,716,428 lignes · 17 colonnes
     ⏱️  Performance.........: 34.48s (49,780 l/s)
  Statut table raw_bureau..: Chargé brut ✅
  Traitement de..........: bureau_balance.csv
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 27,299,925 lignes · 3 colonnes
     ⏱️  Performance.........: 100.04s (272,878 l/s)
  Statut table raw_bureau_balance..: Chargé brut ✅
  Traitement de..........: previous_application.csv
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 1,670,214 lignes · 37 colonnes
     ⏱️  Performance.........: 88.71s (18,827 l/s)
  Statut table raw_previous_app..: Chargé brut ✅
  Traitement de..........: POS_CASH_balance.csv
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 10,001,358 lignes · 8 colonnes
     ⏱️  Performance.........: 91.90s (108,830 l/s)
  Statut table raw_pos_cash..: Chargé brut ✅
  Traitement de..........: credit_card_balance.csv
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 3,840,312 lignes · 23 colonnes
     ⏱️  Performance.........: 129.04s (29,759 l/s)
  Statut table raw_credit_card..: Chargé brut ✅
  Traitement de..........: installments_payments.csv
     📍 Destination.........: PostgreSQL (credit_scoring)
     📊 Volume..............: 13,605,401 lignes · 8 colonnes
     ⏱️  Performance.........: 236.12s (57,621 l/s)
  Statut table raw_installments..: Chargé brut ✅
----------------------------------------------------------------------------
INGESTION TERMINÉE AVEC SUCCÈS
----------------------------------------------------------------------------

    ⏱  Temps d'ingestion..: 730.6s
```


### 1. Le Concept de "Feature Engineering" (Ingénierie des caractéristiques)

La table `application_train` contient des informations statiques (âge, revenus, possession d'une voiture). C'est une "photo" du client à l'instant T. Mais le comportement financier est un **film**.

* **`bureau.csv`** : Comment ce client se comporte-t-il avec **les autres banques** ? Si `application_train` dit qu'il est solvable, mais que `bureau` montre qu'il a 5 crédits impayés ailleurs, ton modèle doit le savoir.
* **`previous_application.csv`** : A-t-il déjà demandé des crédits chez **nous** ? Lui a-t-on refusé ? Quelqu'un à qui on a refusé 3 crédits récemment présente un profil de risque très différent.

---

### 2. Capturer le comportement temporel et les habitudes

Les tables contenant le mot `balance` ou `installments` sont des séries temporelles. Elles ne sont pas utilisées directement, elles sont **agrégées**.

* **`installments_payments.csv`** : C'est ici que tu découvres si le client est un "retardataire". Tu calcules : *"Combien de fois a-t-il payé après la date limite ces 2 dernières années ?"*. Cette nouvelle colonne sera probablement la variable la plus prédictive de ton modèle.
* **`credit_card_balance.csv`** : Le client vit-il à la limite de son plafond de carte de crédit chaque mois ? Le taux d'utilisation de la carte est un indicateur de stress financier invisible dans la table principale.

---

### 3. La structure "Une ligne par client"

Pour entraîner le modèle, tu as besoin d'une table finale où **1 ligne = 1 client (`SK_ID_CURR`)**. Le rôle de ton pipeline (via les vues SQL) est de :

1. **Grouper (Group By)** les tables secondaires par identifiant client.
2. **Calculer des statistiques** (Moyenne des crédits, somme des retards, maximum de jours de retard).
3. **Joindre (Join)** ces résultats à la table `application_train`.

---

### Que se passe-t-il si tu ne les utilises pas ?

Si tu entraînes ton modèle uniquement avec `application_train` :

* Ton score **AUC (Area Under Curve)** sera faible (environ 0.68 - 0.70).
* En ajoutant les variables agrégées des autres tables, ton AUC montera à **0.75 - 0.80**. Dans le secteur bancaire, cette différence de 0.10 représente des **millions d'euros** d'économies en évitant des défauts de paiement.

### 💡 Résumé Didactique (Style Yann LeCun)

* **`application_train`** : C'est le sujet (l'individu).
* **Le reste des tables** : C'est le contexte, l'historique et les habitudes.

Un modèle sans contexte est un modèle "aveugle". C'est pour cela qu'en **Phase 2**, nous créerons la vue `v_master` : la fusion de toute cette intelligence distribuée.
 