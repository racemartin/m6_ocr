

**Phase 1** (8 tables → 14 vues) :
```
raw_application_train : 3 lignes · 122 colonnes
raw_bureau            : 3 lignes · 17 colonnes
... (7 tables total)

v_clean_application, v_clean_bureau, ..., v_agg_bureau,
v_agg_previous, v_master, v_features_engineering ✅
```

**Phase 2** (Feature Engineering SQL + preprocessing sklearn) :
```
Train: (3, 123) → X_train: (3, 91)   y_train: (3,)
Test:  (1, 123) → X_test:  (1, 91)   [aligné automatiquement]
```

Les 91 features comprennent : OHE × 6 colonnes, Ordinal × 1, Log+StandardScaler × 10, StandardScaler × 42, RobustScaler × 9, Binaires × 15, TargetEnc × 3.

---

### Fichiers livrés

| Fichier | Rôle |
|---------|------|
| `src/data/schema.py` | FeatureRegistry (dataclasses) — 92 AttributeSpec définis, listes dérivées automatiquement |
| `src/data/credit_scoring_cleaner.py` | 7 tables raw → 7 vues clean → 5 agrégations → v_master → v_features_engineering (FE SQL) |
| `src/features/registry.py` | FeatureConfigurator — fit/transform sans leakage, ColumnTransformer auto-construit |
| `src/features/generate_enums.py` | Génère 12 classes Enum Python depuis le registry |
| `src/features/enums/` | 12 classes auto-générées (ContractTypeEnum, GenderEnum, etc.) |
| `src/pipelines/phase1_preparation.py` | Script CLI Phase 1 (ingestion + vues) |
| `src/pipelines/phase2_feature_engineering.py` | Script CLI Phase 2 (preprocessing + export) |

---

### Innovations intégrées 

* Pydantic-like (dataclasses), `validate_dataframe()` avec alertes, ColumnTransformer construit depuis le registry, scripts modulaires phase1/phase2.
* Champs `learned_*` anti-leakage dans AttributeSpec, FE SQL dans les vues (ratios, scores composites), agrégations des 5 tables secondaires, Enums auto-générées, compatibilité SQLite natif + SQLAlchemy optionnel.




<img src="docs\images\secuence_recover_data_raw.svg" alt="secuence_recover_data_raw" width="900">



```
@startuml
skinparam Style strictuml
skinparam SequenceMessageAlignment center

actor "Data Engineer" as User
participant "Phase1Pipeline" as P1
participant "CreditScoringCleaner" as CSC
participant "Database (PostgreSQL)" as DB
participant "FeatureRegistry" as REG

== Étape 1: Ingestion & Nettoyage ==

User -> P1 : run_phase1()
activate P1

P1 -> CSC : ingest_all(data_dir)
activate CSC
CSC -> DB : CREATE TABLE raw_* (COPY FROM CSV)
CSC <-- DB : Success
deactivate CSC

P1 -> CSC : create_all_views()
activate CSC
Note over CSC : Utilise REGISTRY pour\nconnaître les types
CSC -> DB : CREATE VIEW v_clean_* (Casting & Renaming)
CSC -> DB : CREATE VIEW v_agg_* (Group By SK_ID_CURR)
CSC -> DB : CREATE VIEW v_master (JOIN v_clean + v_agg)
CSC <-- DB : Success
deactivate CSC

== Étape 2: Analyse & Qualité ==

P1 -> CSC : load_train_test()
activate CSC
CSC -> DB : SELECT * FROM v_features_engineering
CSC <-- DB : DataFrame (Pandas)
deactivate CSC

P1 -> P1 : Analyser déséquilibre (Target)
P1 -> P1 : Vérifier valeurs manquantes (Missingno)

== Étape 3: Préparation Finale ==

P1 -> REG : sync_to_db(engine)
activate REG
REG -> DB : UPDATE metadata_registry
deactivate REG

P1 -> User : Rapport de synthèse (Shapes, Timings)
deactivate P1

@enduml
```



Pour configurer **deux bases de données** (ou plus) dans un seul conteneur PostgreSQL avec Docker, il faut comprendre une limitation : par défaut, l'image PostgreSQL officielle ne crée **qu'une seule base** à l'aide de la variable `POSTGRES_DB`.

Voici comment procéder étape par étape pour avoir `mlflow_db` et `credit_scoring` au démarrage.

---

### 1. Créer le script d'initialisation SQL

À la racine de ton projet, crée un dossier nommé `scripts/` et à l'intérieur, un fichier nommé `init-db.sql`.

**Contenu de `scripts/init-db.sql` :**

```sql
-- Création de la deuxième base de données
CREATE DATABASE credit_scoring;

-- Optionnel : donner les droits à l'utilisateur mlflow_user sur cette base
GRANT ALL PRIVILEGES ON DATABASE credit_scoring TO mlflow_user;

```

### 2. Modifier le `docker-compose.yml`

Tu n'as pas besoin de modifier le `Dockerfile` de Postgres, mais seulement la section `volumes` de ton service `postgres`. L'image Postgres exécute automatiquement tout script situé dans `/docker-entrypoint-initdb.d/`.

```yaml
services:
  postgres:
    image: postgres:15
    container_name: postgres_db
    environment:
      POSTGRES_DB: mlflow_db        # La première base créée par défaut
      POSTGRES_USER: mlflow_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      # ON MONTE LE SCRIPT D'INITIALISATION ICI :
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql

```

---

### 3. Comment relancer pour que cela prenne effet ?

**Attention :** Le dossier `/docker-entrypoint-initdb.d/` n'est lu que si le dossier de données (`postgres_data`) est **vide**. Si tu as déjà des données, Postgres ignorera le script.

**Procédure pour réinitialiser proprement :**

```powershell
# 1. Arrête tout et supprime le volume de données (ATTENTION : PERTE DES DONNÉES ACTUELLES)
docker compose down -v 

# 2. Relance le tout
docker compose up -d

```

### 4. Vérification dans pgAdmin

Une fois relancé, quand tu te connecteras à pgAdmin :

1. Fais un clic droit sur **Databases** > **Refresh**.
2. Tu devrais voir apparaître deux bases : `mlflow_db` et `credit_scoring`.

