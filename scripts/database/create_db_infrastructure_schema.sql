-- ##############################################################################
-- FICHIER : infrastructure_schema.sql
-- PROJET  : Scoring Binaire pour les Prêt à la Consommation
-- RÔLE    : Définition de la structure PostgreSQL pour le monitoring et le log
-- USAGE   : psql -U postgres -f infrastructure_schema.sql
-- ##############################################################################

-- --- PRÉPARATION DE L'ENVIRONNEMENT ------------------------------------------
-- DROP DATABASE IF EXISTS credit_scoring;
-- CREATE DATABASE credit_scoring;

-- ##############################################################################

-- --- PRÉPARATION SÉCURISÉE ---------------------------------------------------

-- Au lieu de DROP DATABASE, on travaille dans la base existante
-- ou on la crée seulement si elle n'existe pas (logique shell/admin)

-- Note : Le DROP TABLE est plus granulaire et moins risqué que DROP DATABASE
-- car il permet de reconstruire la structure sans perdre les accès/privilèges.

DROP TABLE IF EXISTS drift_alerts         CASCADE;
DROP TABLE IF EXISTS monitoring_metrics   CASCADE;
DROP TABLE IF EXISTS prediction_results   CASCADE;
DROP TABLE IF EXISTS prediction_requests  CASCADE;
DROP TABLE IF EXISTS model_versions       CASCADE;
DROP TABLE IF EXISTS datasets             CASCADE;
DROP TABLE IF EXISTS feature_store        CASCADE;

-- Activation des extensions pour les identifiants uniques (UUID)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ##############################################################################
-- GESTION DES ARTEFACTS (DATASETS & MODÈLES)
-- ##############################################################################

-- Table: Datasets (Traçabilité des données d'entraînement)
DROP TABLE IF EXISTS datasets CASCADE;

CREATE TABLE datasets (
    id            SERIAL       PRIMARY KEY,
    file_path     VARCHAR(255) NOT NULL,
    description   TEXT,
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    version       VARCHAR(50),
    row_count     INTEGER,
    feature_count INTEGER,
    metadata      JSONB        -- Stockage flexible pour les stats EDA
);

-- Table: Feature Store (Vecteurs de données transformées)
DROP TABLE IF EXISTS feature_store CASCADE;

CREATE TABLE feature_store (
    id            SERIAL    PRIMARY KEY,
    dataset_id    INTEGER   REFERENCES datasets(id) ON DELETE CASCADE,
    file_path     TEXT      NOT NULL, -- ¡AQUÍ! El puntero directo al CSV estandarizado
    row_count     INTEGER,
    feature_count INTEGER,
    feature_names JSONB     NOT NULL, -- Lista de variables: ['Age', 'MonthlyIncome', ...]
    target_name   VARCHAR(100),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Índice para acelerar la recuperación de features de un dataset específico
CREATE INDEX idx_feature_store_dataset_id ON feature_store(dataset_id);

-- Suppression de l'ancienne table si nécessaire ou ALTER TABLE
DROP TABLE IF EXISTS model_versions;

CREATE TABLE model_versions (
    id              SERIAL       PRIMARY KEY,
    model_name      VARCHAR(255) NOT NULL,
    version         VARCHAR(50)  NOT NULL UNIQUE,
    mlflow_run_id   VARCHAR(255),
    algorithm       VARCHAR(100),
    hyperparameters JSONB,       
    metrics          JSONB,       
    
    -- Le seuil métier optimisé lors de la Phase 3/4
    optimal_threshold FLOAT         DEFAULT 0.5,
    
    -- 📂 TRAÇABILITÉ PHYSIQUE DISJOINTE
    model_path      TEXT         NOT NULL,  -- Le fichier .joblib (le muscle)
    metadata_path   TEXT         NOT NULL,  -- Le fichier .json (le cerveau/contexte)
    
    status          VARCHAR(50)  DEFAULT 'trained', -- trained, staging, production
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    deployed_at     TIMESTAMP,
    CONSTRAINT unique_model_version UNIQUE (model_name, version)
);

-- ##############################################################################
-- LOGS D'INFÉRENCE (TRAÇABILITÉ TEMPS RÉEL)
-- ##############################################################################

-- Table: Prediction Requests (Entrées envoyées à l'API)
CREATE TABLE prediction_requests (
    id               UUID      PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version_id INTEGER   REFERENCES model_versions(id),
    input_data       JSONB     NOT NULL, -- Caractéristiques de l'employé
    request_metadata JSONB,              -- IP, User-Agent, etc.
    user_id          VARCHAR(255),
    session_id       VARCHAR(255)
);

-- Table: Prediction Results (Sorties générées par le modèle)
CREATE TABLE prediction_results (
    id               UUID      PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id       UUID      REFERENCES prediction_requests(id) ON DELETE CASCADE,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_value NUMERIC,            -- Score numérique
    prediction_class VARCHAR(255),       -- Label (ex: "ÉLEVÉ")
    prediction_probs JSONB,              -- Probabilités par classe
    confidence_score NUMERIC,
    processing_time  INTEGER,            -- Latence en millisecondes
    success          BOOLEAN   DEFAULT TRUE,
    error_message    TEXT
);

-- ##############################################################################
-- OBSERVABILITÉ ET DRIFT (QUALITÉ IA)
-- ##############################################################################

-- Table: Monitoring Metrics (Performance système)
CREATE TABLE monitoring_metrics (
    id               SERIAL       PRIMARY KEY,
    recorded_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    metric_name      VARCHAR(255) NOT NULL,
    metric_value     NUMERIC      NOT NULL,
    metric_type      VARCHAR(100), -- drift, performance, latency
    model_version_id INTEGER      REFERENCES model_versions(id),
    metadata         JSONB
);

-- Table: Drift Alerts (Détection de décalage de données)
CREATE TABLE drift_alerts (
    id               SERIAL       PRIMARY KEY,
    detected_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    drift_type       VARCHAR(100), -- data_drift, concept_drift
    drift_score      NUMERIC      NOT NULL,
    threshold        NUMERIC      NOT NULL,
    features_affected JSONB,
    model_version_id INTEGER      REFERENCES model_versions(id),
    status           VARCHAR(50)  DEFAULT 'active',
    resolved_at      TIMESTAMP
);

-- ##############################################################################
-- OPTIMISATION (INDEXATION)
-- ##############################################################################

CREATE INDEX idx_pred_req_created ON prediction_requests(created_at);
CREATE INDEX idx_pred_res_request ON prediction_results(request_id);
CREATE INDEX idx_monitoring_model ON monitoring_metrics(model_version_id);
CREATE INDEX idx_drift_detected   ON drift_alerts(detected_at);

-- ##############################################################################
-- LOGIQUE AUTOMATISÉE (TRIGGERS & VUES)
-- ##############################################################################

-- Fonction pour la mise à jour automatique de l'horodatage
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER trg_datasets_updated BEFORE UPDATE ON datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Vue de Performance : Pour le Dashboard du projet
CREATE VIEW v_model_performance AS
SELECT
    mv.model_name,
    mv.version,
    COUNT(pr.id)                                  AS total_predictions,
    AVG(pres.confidence_score)                    AS avg_confidence,
    AVG(pres.processing_time)                     AS avg_latency_ms,
    SUM(CASE WHEN pres.success THEN 1 ELSE 0 END) AS success_count
FROM model_versions mv
LEFT JOIN prediction_requests pr ON mv.id = pr.model_version_id
LEFT JOIN prediction_results pres ON pr.id = pres.request_id
GROUP BY mv.id, mv.model_name, mv.version;

-- ##############################################################################
-- INITIALISATION (DONNÉES EXEMPLES)
-- ##############################################################################

/*
INSERT INTO datasets (name, description, version) 
VALUES ('training_hr_v1', 'Dataset initial Phase 1', '1.0.0');

INSERT INTO model_versions (model_name, version, algorithm, status) 
VALUES ('attrition_predictor', '1.0.0', 'RandomForest', 'production');
*/

-- Fin du script