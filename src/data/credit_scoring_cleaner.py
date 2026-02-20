"""
src/data/credit_scoring_cleaner.py
====================================
CreditScoringCleaner — Gestion complète du pipeline données Phase 1

Responsabilités :
    1. Création des tables raw_* (une par fichier source CSV)
    2. Création des vues v_clean_* (nettoyage SQL, mappings valeurs)
    3. Agrégations des tables secondaires → v_agg_*
    4. Vue maîtresse v_master (jointures, 1 ligne par SK_ID_CURR)
    5. Feature engineering SQL → v_features_engineering
    6. Ingestion CSV → tables raw_*

Architecture calquée sur AttritionCleaner(DataCleaner) :
    - engine SQLAlchemy fourni en constructor
    - méthodes create_*/clean_*/ingest_* séparées
    - cohérence avec le FeatureRegistry (schema.py)
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS - LIBRAIRIES STANDARD PYTHON
# ==============================================================================
import logging                              # Logs (mode verbose + debug)
import sqlite3                              # Backend local (sans SQLAlchemy)
from   pathlib import Path                  # Chemins robustes
from   typing  import Optional, Union       # Typage
import time                                 # Mesure des temps d'exécution
import io

# ==============================================================================
# IMPORTS - LIBRAIRIES TIERS (DATA)
# ==============================================================================
import numpy as np                          # Calculs numériques
import pandas as pd                         # DataFrames (ingestion / lecture SQL)

# ==============================================================================
# IMPORTS - MODULES INTERNES (REGISTRE)
# ==============================================================================
from src.data.schema import FeatureRegistry, REGISTRY  # Registre métier ↔ technique
from src.data.schema import ColumnType, ColumnRole 

# ==============================================================================
# SQLALCHEMY (OPTIONNEL) - COMPATIBILITÉ SQLITE / POSTGRES
# ==============================================================================
# Compatibilité : si SQLAlchemy est dispo, on l'accepte ; sinon sqlite3 natif
try:
    from sqlalchemy import Engine, text as sa_text
    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False
    Engine = None   # type: ignore


logger = logging.getLogger(__name__)


# ##############################################################################
# CREDIT SCORING CLEANER
# ##############################################################################


"""
-- 1. Borramos las vistas de nivel superior (Agregaciones y Master)
DROP VIEW IF EXISTS v_master_features CASCADE;
DROP VIEW IF EXISTS v_agg_bureau CASCADE;
DROP VIEW IF EXISTS v_agg_previous CASCADE;
DROP VIEW IF EXISTS v_agg_pos_cash CASCADE;
DROP VIEW IF EXISTS v_agg_credit_card CASCADE;
DROP VIEW IF EXISTS v_agg_installments CASCADE;

-- 2. Borramos las vistas de limpieza (Capa Silver)
DROP VIEW IF EXISTS v_clean_application CASCADE;
DROP VIEW IF EXISTS v_clean_bureau CASCADE;
DROP VIEW IF EXISTS v_clean_bureau_balance CASCADE;
DROP VIEW IF EXISTS v_clean_previous_app CASCADE;
DROP VIEW IF EXISTS v_clean_pos_cash CASCADE;
DROP VIEW IF EXISTS v_clean_credit_card CASCADE;
DROP VIEW IF EXISTS v_clean_installments CASCADE;

-- 3. Finalmente, borramos las tablas RAW (Capa Bronze)
DROP TABLE IF EXISTS raw_application_train CASCADE;
DROP TABLE IF EXISTS raw_application_test CASCADE;
DROP TABLE IF EXISTS raw_bureau CASCADE;
DROP TABLE IF EXISTS raw_bureau_balance CASCADE;
DROP TABLE IF EXISTS raw_previous_app CASCADE;
DROP TABLE IF EXISTS raw_pos_cash CASCADE;
DROP TABLE IF EXISTS raw_credit_card CASCADE;
DROP TABLE IF EXISTS raw_installments CASCADE;
"""

class CreditScoringCleaner:
    """
    Cleaner métier pour le projet Home Credit.

    Usage typique (Phase 1) :
        cleaner = CreditScoringCleaner(engine)
        cleaner.ingest_all(data_dir="data/raw")
        cleaner.create_all_views()
        df_master = cleaner.load_master(split="train")
    """

    # Valeur aberrante connue pour DAYS_EMPLOYED (retraité / sans emploi)
    DAYS_EMPLOYED_ANOMALY = 365243

    def __init__(
        self,
        engine,  # sqlalchemy Engine OU chemin str vers fichier .db
        registry: FeatureRegistry = REGISTRY,
        verbose: bool = True
    ):
        self.engine   = engine
        self.registry = registry
        self.verbose  = verbose

        # 1. Determinamos si estamos usando SQLite (vía path) o SQLAlchemy
        self._db_path = engine if isinstance(engine, str) else None
        
        # 2. Creamos el flag de PostgreSQL para el resto de métodos
        # Si engine no es un string y tiene una URL con 'postgresql', es Postgres.
        if self._db_path:
            self._is_postgres = False
        else:
            # Comprobamos el dialecto del engine de SQLAlchemy
            self._is_postgres = "postgresql" in str(engine.url)
            
        if self.verbose:
            motor = "PostgreSQL" if self._is_postgres else "SQLite"
            print(f"🔧 Cleaner inicializado para {motor}")

    # ##############################################################################
    # LOGGING HELPER
    # ##############################################################################
    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ##############################################################################
    # INGESTION CSV → tables raw_*
    # ##############################################################################

    def ingest_all(self, data_dir: str = "data/raw") -> None:
        """
        Charge tous les CSV du projet vers les tables raw_*.
        
        Cette étape assure l'ingestion brute. Le renommage technique
        est appliqué uniquement aux tables 'application' via le registre.
        """
        base  = Path(data_dir)                    # Chemin racine des données
        files = {                                 # Mapping Table SQL : Fichier
            "raw_application_train" : "application_train.csv",
            "raw_application_test"  : "application_test.csv",
            "raw_bureau"            : "bureau.csv",
            "raw_bureau_balance"    : "bureau_balance.csv",
            "raw_previous_app"      : "previous_application.csv",
            "raw_pos_cash"          : "POS_CASH_balance.csv",
            "raw_credit_card"       : "credit_card_balance.csv",
            "raw_installments"      : "installments_payments.csv",
        }

        print("\n----------------------------------------------------------------------------")
        print("DÉBUT DE L'INGESTION DES DONNÉES (CSV -> SQL)")
        print("----------------------------------------------------------------------------")

        for table_name, filename in files.items():
            fpath = base / filename               # Construction du chemin complet
            
            # Vérification de l'existence du fichier physique
            if not fpath.exists():
                print(f"  ⚠️  Fichier manquant....: {filename} (Ignoré)")
                continue

            # Information sur le traitement en cours
            print(f"  Traitement de..........: {filename}")
            
            # Détermination du besoin de renommage (uniquement pour application)
            should_rename = "application" in table_name
            
            # Appel de l'ingestion unitaire
            self._ingest_csv(
                file_path    = fpath,
                table_name   = table_name,
                apply_rename = should_rename
            )

            # Confirmation visuelle de réussite pour chaque table
            status = "Renommé & Chargé" if should_rename else "Chargé brut"
            print(f"  Statut table {table_name:<10}..: {status} ✅")

        print("----------------------------------------------------------------------------")
        print("INGESTION TERMINÉE AVEC SUCCÈS")
        print("----------------------------------------------------------------------------\n")

    def _conn(self):
        """Retourne une connexion sqlite3 (context manager compatible)."""
        if self._db_path:
            return sqlite3.connect(self._db_path)
        elif _HAS_SQLALCHEMY:
            return self.engine.begin()
        raise RuntimeError("Fournir un chemin .db (str) ou un sqlalchemy Engine")

    def _exec_sql(self, sql: str, view_name: str) -> None:
        """Exécute une requête SQL (compatible SQLite et PostgreSQL avec dépendances)."""
        
        # 1. Adaptación para SQLite (no soporta OR REPLACE)
        clean_sql = sql.replace("CREATE OR REPLACE VIEW", "CREATE VIEW")
        
        if self._db_path:
            # --- CASO SQLITE ---
            with sqlite3.connect(self._db_path) as conn:
                # SQLite no tiene CASCADE, borramos uno a uno
                if "DROP VIEW" not in sql.upper():
                    conn.execute(f"DROP VIEW IF EXISTS {view_name}")
                conn.execute(clean_sql)
                conn.commit()
        else:
            # --- CASO POSTGRESQL (engine) ---
            with self.engine.begin() as conn:
                if "DROP VIEW" in sql.upper():
                    # Si la instrucción ya es un DROP (viene de _drop_all_views),
                    # la ejecutamos TAL CUAL para que respete el CASCADE.
                    conn.execute(sa_text(sql))
                else:
                    # Si es un CREATE, primero limpiamos de forma segura.
                    # Usamos CASCADE por defecto en Postgres para evitar bloqueos.
                    conn.execute(sa_text(f"DROP VIEW IF EXISTS {view_name} CASCADE"))
                    conn.execute(sa_text(sql))
                    
        if view_name:
            self._print(f"     ✅ {view_name}")

    def _read_sql(self, sql: str, name: str = "") -> pd.DataFrame:
        """
        Lit un résultat SQL vers DataFrame (compatible SQLite et PostgreSQL).
        'name' permite debuguear qué consulta estamos ejecutando.
        """
        if self.verbose and name:
            self._print(f"    🔍 Lectura SQL: {name}...")

        if self._db_path:
            # --- RUTA SQLITE ---
            import sqlite3
            with sqlite3.connect(self._db_path) as conn:
                return pd.read_sql_query(sql, conn)
        else:
            # --- RUTA POSTGRESQL ---
            # pd.read_sql acepta tanto el SQL como el engine de SQLAlchemy
            return pd.read_sql(sql, self.engine)

    def _ingest_csv(self, file_path: Path, table_name: str, apply_rename: bool = False) -> None:
        """
        Charge un CSV vers une table raw_* avec optimisation COPY pour PostgreSQL.
        """
        start_time = time.time()
        
        # 1. Chargement en mémoire
        df = pd.read_csv(file_path, low_memory=False)
        
        # 2. Application du mapping métier (si demandé)
        if apply_rename:
            df = df.rename(columns=self.registry.rename_map)
            self._print(f"     ↳ Mapping technique appliqué via Registry")
    
        # 3. Normalisation SQL (lowercase, sans espaces)
        df.columns = [c.strip().lower() for c in df.columns]
    
        # 4. Détermination de la cible et Ingestion
        try:
            if self._db_path:
                # CAS SQLITE
                with sqlite3.connect(self._db_path) as conn:
                    df.to_sql(table_name, conn, if_exists="replace", index=False)
            else:
                # CAS POSTGRESQL (Docker)
                db_type = f"PostgreSQL ({self.engine.url.database})"
                
                # --- CORRECCIÓN CLAVE ---
                # Forzamos el borrado de las dependencias antes de 'replace'
                with self.engine.begin() as conn:
                    conn.execute(sa_text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                
                # Ahora Pandas puede crear la tabla de nuevo sin conflictos
                df.head(0).to_sql(table_name, self.engine, if_exists="replace", index=False)
                
                # Créer la structure de la table (vide) d'abord
                # Esto define las columnas basándose en el DataFrame sin insertar los datos aún
                
                # Preparar el buffer en memoria
                output = io.StringIO()
                df.to_csv(output, sep='\t', header=False, index=False)
                output.seek(0)
                
                # Conexión nativa para usar copy_from
                raw_conn = self.engine.raw_connection()
                try:
                    with raw_conn.cursor() as cursor:
                        cursor.copy_from(output, table_name, null="")
                    raw_conn.commit()
                finally:
                    raw_conn.close()
    
            duration = time.time() - start_time
            
            # Affichage du bilan détaillé
            self._print(f"     📍 Destination.........: {db_type}")
            self._print(f"     📊 Volume..............: {len(df):,} lignes · {len(df.columns)} colonnes")
            self._print(f"     ⏱️  Performance.........: {duration:.2f}s ({int(len(df)/duration):,} l/s)")
            
        except Exception as e:
            self._print(f"     ❌ ERREUR d'ingestion sur {table_name}: {str(e)}")
            raise e

    # =========================================================================
    # CRÉATION DE TOUTES LES VUES (entrée unique)
    # =========================================================================

    def _drop_all_views(self) -> None:
        """Limpia el lienzo antes de empezar."""
        self._print("  🧹 Limpiando vistas existentes...")
        # Al borrar las master y agg primero, facilitas el camino
        vistas = [
            # 1. Capa Final (Features y Master)
            "v_features_engineering", 
            "v_master_features",
            
            # 2. Capa de Agregación
            "v_agg_bureau", 
            "v_agg_previous", # Verifica si es v_agg_previous o v_agg_previous_app
            "v_agg_pos_cash", 
            "v_agg_credit_card", 
            "v_agg_installments",
            
            # 3. Capa de Limpieza (Clean)
            "v_clean_application",
            "v_clean_bureau",
            "v_clean_bureau_balance",
            "v_clean_previous_app",
            "v_clean_pos_cash",
            "v_clean_credit_card",
            "v_clean_installments"
        ]
        for v in vistas:
            self._exec_sql(f"DROP VIEW IF EXISTS {v} CASCADE;", v)
            
    def create_all_views(self) -> None:
        """Crée toutes les vues dans l'ordre correct (dépendances)."""
        self._print("\n🔧 Création des vues SQL ...\n")

        self._drop_all_views() # <--- Paso 1
        
        # 1. Vues clean (une par source)
        self._create_view_clean_application()
        self._create_view_clean_bureau()
        self._create_view_clean_bureau_balance()
        self._create_view_clean_previous()
        self._create_view_clean_pos_cash()
        self._create_view_clean_credit_card()
        self._create_view_clean_installments()

        # 2. Vues agrégées (tables secondaires → 1 ligne par SK_ID_CURR)
        self._create_view_agg_bureau()
        self._create_view_agg_previous()
        self._create_view_agg_pos_cash()
        self._create_view_agg_credit_card()
        self._create_view_agg_installments()

        # 3. Vue maîtresse (jointures)
        self._create_view_master()

        # 4. Feature engineering SQL
        self._create_view_features()

        self._print("\n✅ Toutes les vues créées avec succès.\n")


    def _build_select_fields(self, table_name: str, is_test: bool = False) -> list[str]:
        """
        Método genérico para construir los campos del SELECT.
        Sincronizado con AttributeSpec:
        - col_orig = name_raw
        - col_dest = name_technique
        """
        fields = []
        attributes = [a for a in self.registry.attributes if a.source_table == table_name]

        for attr in attributes:
            # --- AQUÍ ESTABA EL ERROR ---
            col_orig = attr.name_raw.lower()  # El nombre del CSV (ej: 'NAME_CONTRACT_TYPE')
            col_dest = attr.name_technique    # El nombre limpio (ej: 'contract_type')
            
            # 1. Identificadores
            if attr.col_type == ColumnType.IDENTIFIER:
                fields.append(f"CAST({col_orig} AS INTEGER) AS {col_dest}")
            
            # 2. Target
            elif attr.role == ColumnRole.TARGET:
                val = "NULL" if is_test else col_orig
                fields.append(f"{val} AS {col_dest}")
                
            # 3. Categorías con MAPEOS específicos (Agrupaciones/Jerarquías)
            elif attr.valeurs_possibles:
                cases = " ".join([
                    f"WHEN TRIM(UPPER(CAST({col_orig} AS TEXT))) = '{orig.upper()}' THEN '{dest}'"
                    for orig, dest in attr.valeurs_possibles.items()
                ])
                fields.append(f"CASE {cases} ELSE '{attr.valeur_inconnue}' END AS {col_dest}")

            # 4. Categorías Estándar (Limpieza automática sin CASE pesado)
            elif attr.col_type == ColumnType.CATEGORICAL:
                fields.append(f"LOWER(TRIM(CAST({col_orig} AS TEXT))) AS {col_dest}")

            # 5. Passthrough (Numéricos y otros)
            else:
                fields.append(f"{col_orig} AS {col_dest}")

        # print(fields)
        return fields
    
    # =========================================================================
    # VUES CLEAN
    # =========================================================================
    
    def _create_view_clean_application(self) -> None:
        """Ahora esta función es súper corta y elegante."""
        self._print("  → v_clean_application (vía _build_select_fields) ...")
        
        # Generamos los campos para cada parte del UNION
        train_cols = ",\n    ".join(self._build_select_fields("application", is_test=False))
        test_cols  = ",\n    ".join(self._build_select_fields("application", is_test=True))

        # El caso especial de DAYS_EMPLOYED lo podemos inyectar aquí 
        # o manejarlo dentro del generador si es común.
        
        sql = f"""
        CREATE OR REPLACE VIEW v_clean_application AS
        SELECT {train_cols}, 'train' AS split FROM raw_application_train
        UNION ALL
        SELECT {test_cols}, 'test' AS split FROM raw_application_test
        """
        
        self._exec_sql(sql, "v_clean_application")

    # #########################################################################
    def _create_view_clean_bureau(self) -> None:
        # 1. Asegúrate de que esta línea tenga 8 espacios (o 2 tabs de 4)
        self._print("  → v_clean_bureau (automatizada) ...")
        
        # 2. Generamos las columnas dinámicamente
        cols = ",\n    ".join(self._build_select_fields("bureau"))
        
        sql = f"""
        CREATE OR REPLACE VIEW v_clean_bureau AS
        SELECT {cols} FROM raw_bureau
        """
        self._exec_sql(sql, "v_clean_bureau")


    def _create_view_clean_bureau_bak(self) -> None:
        self._print("  → v_clean_bureau ...")
        sql = """
CREATE OR REPLACE VIEW v_clean_bureau AS
SELECT
    CAST(sk_id_curr AS INTEGER)   AS sk_id_curr,
    CAST(sk_id_bureau AS INTEGER) AS sk_id_bureau,
    TRIM(CAST(credit_active AS TEXT))   AS credit_active,
    TRIM(CAST(credit_currency AS TEXT)) AS credit_currency,
    days_credit,
    credit_day_overdue,
    days_credit_enddate,
    days_enddate_fact,
    amt_credit_max_overdue,
    cnt_credit_prolong,
    amt_credit_sum,
    amt_credit_sum_debt,
    amt_credit_sum_limit,
    amt_credit_sum_overdue,
    TRIM(CAST(credit_type AS TEXT))  AS credit_type,
    days_credit_update,
    amt_annuity
FROM raw_bureau

"""
        self._exec_sql(sql, "v_clean_bureau")

    # #########################################################################

    def _create_view_clean_bureau_balance(self) -> None:
        self._print("  → v_clean_bureau_balance (automatizada) ...")
        cols = ",\n    ".join(self._build_select_fields("bureau_balance"))
        
        sql = f"""
        CREATE OR REPLACE VIEW v_clean_bureau_balance AS
        SELECT {cols} FROM raw_bureau_balance
        """
        self._exec_sql(sql, "v_clean_bureau_balance")
    
    def _create_view_clean_bureau_balance_bak(self) -> None:
        self._print("  → v_clean_bureau_balance ...")
        sql = """
CREATE OR REPLACE VIEW v_clean_bureau_balance AS
SELECT
    CAST(sk_id_bureau AS INTEGER) AS sk_id_bureau,
    months_balance,
    TRIM(CAST(status AS TEXT)) AS status
FROM raw_bureau_balance
"""
        self._exec_sql(sql, "v_clean_bureau_balance")

    # #########################################################################

    def _create_view_clean_previous(self) -> None:
        self._print("  → v_clean_previous_app (automatizada vía Registry) ...")
        
        # Obtenemos los campos definidos para la tabla 'previous' en el Registry
        cols = ",\n    ".join(self._build_select_fields("previous_application"))
        
        sql = f"""
        CREATE OR REPLACE VIEW v_clean_previous_app AS
        SELECT {cols} FROM raw_previous_app
        """
        self._exec_sql(sql, "v_clean_previous_app")
        
    def _create_view_clean_previous_bak(self) -> None:
        self._print("  → v_clean_previous_app ...")
        sql = """
CREATE OR REPLACE VIEW v_clean_previous_app AS
SELECT
    CAST(sk_id_prev AS INTEGER) AS sk_id_prev,
    CAST(sk_id_curr AS INTEGER) AS sk_id_curr,
    TRIM(CAST(name_contract_type   AS TEXT)) AS contract_type,
    TRIM(CAST(name_contract_status AS TEXT)) AS contract_status,
    TRIM(CAST(code_reject_reason   AS TEXT)) AS reject_reason,
    amt_annuity,
    amt_application,
    amt_credit,
    amt_down_payment,
    amt_goods_price,
    days_decision,
    rate_down_payment,
    cnt_payment,
    days_first_drawing,
    days_first_due,
    days_last_due,
    days_termination
FROM raw_previous_app
"""
        self._exec_sql(sql, "v_clean_previous_app")

    # #########################################################################
    def _create_view_clean_pos_cash(self) -> None:
        self._print("  → v_clean_pos_cash (automatizada vía Registry) ...")
        
        # Obtenemos los campos definidos para la tabla 'pos_cash' en el Registry
        cols = ",\n    ".join(self._build_select_fields("pos_cash"))
        
        sql = f"""
        CREATE OR REPLACE VIEW v_clean_pos_cash AS
        SELECT {cols} FROM raw_pos_cash
        """
        self._exec_sql(sql, "v_clean_pos_cash")
    
    def _create_view_clean_pos_cash_bak(self) -> None:
        self._print("  → v_clean_pos_cash ...")
        sql = """
CREATE OR REPLACE VIEW v_clean_pos_cash AS
SELECT
    CAST(sk_id_prev AS INTEGER) AS sk_id_prev,
    CAST(sk_id_curr AS INTEGER) AS sk_id_curr,
    months_balance,
    cnt_instalment,
    cnt_instalment_future,
    TRIM(CAST(name_contract_status AS TEXT)) AS contract_status,
    sk_dpd,
    sk_dpd_def
FROM raw_pos_cash
"""
        self._exec_sql(sql, "v_clean_pos_cash")


    # #########################################################################
    def _create_view_clean_credit_card(self) -> None:
        self._print("  → v_clean_credit_card (automatizada) ...")
        # Extraemos campos del Registry (asegúrate que source_table sea "credit_card")
        cols = ",\n    ".join(self._build_select_fields("credit_card"))
        
        sql = f"""
        CREATE OR REPLACE VIEW v_clean_credit_card AS
        SELECT {cols} FROM raw_credit_card
        """
        self._exec_sql(sql, "v_clean_credit_card")
    
    def _create_view_clean_credit_card_bak(self) -> None:
        self._print("  → v_clean_credit_card ...")
        sql = """
CREATE OR REPLACE VIEW v_clean_credit_card AS
SELECT
    CAST(sk_id_prev AS INTEGER) AS sk_id_prev,
    CAST(sk_id_curr AS INTEGER) AS sk_id_curr,
    months_balance,
    amt_balance,
    amt_credit_limit_actual,
    amt_drawings_current,
    amt_drawings_atm_current,
    amt_payment_current,
    amt_payment_total_current,
    amt_receivable_principal,
    amt_total_receivable,
    cnt_drawings_current,
    cnt_instalment_mature_cum,
    TRIM(CAST(name_contract_status AS TEXT)) AS contract_status,
    sk_dpd,
    sk_dpd_def
FROM raw_credit_card
"""
        self._exec_sql(sql, "v_clean_credit_card")

    # #########################################################################

    def _create_view_clean_installments(self) -> None:
        self._print("  → v_clean_installments (híbrida) ...")
        # Automatizamos los campos base del Registry
        cols = ",\n    ".join(self._build_select_fields("installments"))
        
        # Inyectamos las columnas calculadas manualmente para no perder la lógica de negocio
        sql = f"""
        CREATE OR REPLACE VIEW v_clean_installments AS
        SELECT 
            {cols},
            -- Feature derivada: retard de paiement (jours)
            (days_entry_payment - days_instalment) AS payment_delay_days,
            -- Ratio payé / dû
            CASE WHEN amt_instalment > 0 
                 THEN amt_payment / amt_instalment 
                 ELSE NULL END AS payment_ratio
        FROM raw_installments
        """
        self._exec_sql(sql, "v_clean_installments")
    
    def _create_view_clean_installments_bak(self) -> None:
        self._print("  → v_clean_installments ...")
        sql = """
CREATE OR REPLACE VIEW v_clean_installments AS
SELECT
    CAST(sk_id_prev AS INTEGER) AS sk_id_prev,
    CAST(sk_id_curr AS INTEGER) AS sk_id_curr,
    num_instalment_version,
    num_instalment_number,
    days_instalment,
    days_entry_payment,
    amt_instalment,
    amt_payment,
    -- Feature dérivée : retard de paiement (jours)
    (days_entry_payment - days_instalment) AS payment_delay_days,
    -- Ratio payé / dû
    CASE WHEN amt_instalment > 0
         THEN amt_payment / amt_instalment
         ELSE NULL END AS payment_ratio
FROM raw_installments
"""
        self._exec_sql(sql, "v_clean_installments")

    # =========================================================================
    # VUES AGRÉGÉES — 1 ligne par SK_ID_CURR
    # =========================================================================

    def _create_view_agg_bureau(self) -> None:
        self._print("  → v_agg_bureau ...")
        sql = """
CREATE OR REPLACE VIEW v_agg_bureau AS
SELECT
    b.sk_id_curr,
    COUNT(*)                                    AS bureau_credit_count,
    AVG(b.amt_credit_sum)                       AS bureau_amt_credit_sum_mean,
    SUM(b.amt_credit_sum)                       AS bureau_amt_credit_sum_total,
    
    MAX(b.days_credit)                          AS bureau_days_credit_max,
    MIN(b.days_credit)                          AS bureau_days_credit_min,
    
    AVG(b.amt_credit_sum_debt)                  AS bureau_amt_credit_sum_debt_mean,
    SUM(b.amt_credit_sum_debt)                  AS bureau_amt_credit_sum_debt_total,
    
    SUM(CASE WHEN b.credit_active = 'Active' THEN 1 ELSE 0 END)
                                                AS bureau_active_credit_count,
    SUM(CASE WHEN b.credit_active = 'Closed' THEN 1 ELSE 0 END)
                                                AS bureau_closed_credit_count,
    SUM(CASE WHEN b.credit_day_overdue > 0 THEN 1 ELSE 0 END)
                                                AS bureau_overdue_count,
    AVG(b.credit_day_overdue)                   AS bureau_credit_day_overdue_mean,
    AVG(b.cnt_credit_prolong)                   AS bureau_cnt_prolong_mean,
    AVG(b.amt_credit_sum_limit)                 AS bureau_credit_limit_mean
FROM v_clean_bureau b
GROUP BY b.sk_id_curr
"""
        self._exec_sql(sql, "v_agg_bureau")

    def _create_view_agg_previous(self) -> None:
        self._print("  → v_agg_previous ...")
        sql = """
CREATE OR REPLACE VIEW v_agg_previous AS
SELECT
    sk_id_curr,
    COUNT(*)                                                AS prev_app_count,
    SUM(CASE WHEN name_contract_status = 'Approved'  THEN 1 ELSE 0 END)
                                                            AS prev_approved_count,
    SUM(CASE WHEN name_contract_status = 'Refused'   THEN 1 ELSE 0 END)
                                                            AS prev_refused_count,
    SUM(CASE WHEN name_contract_status = 'Canceled'  THEN 1 ELSE 0 END)
                                                            AS prev_canceled_count,
    AVG(amt_credit)                                         AS prev_amt_credit_mean,
    MAX(amt_credit)                                         AS prev_amt_credit_max,
    
    AVG(amt_annuity)                                        AS prev_amt_annuity_mean,
    MAX(days_decision)                                      AS prev_days_decision_max,
    AVG(rate_down_payment)                                  AS prev_rate_down_mean,
    AVG(cnt_payment)                                        AS prev_cnt_payment_mean,
    -- Taux de refus
    CASE WHEN COUNT(*) > 0
         THEN CAST(SUM(CASE WHEN name_contract_status = 'Refused' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)
         ELSE NULL END                                       AS prev_refusal_rate
FROM v_clean_previous_app
GROUP BY sk_id_curr
"""
        self._exec_sql(sql, "v_agg_previous")

    def _create_view_agg_pos_cash(self) -> None:
        self._print("  → v_agg_pos_cash ...")
        sql = """
CREATE OR REPLACE VIEW v_agg_pos_cash AS
SELECT
    sk_id_curr,
    COUNT(*)                          AS pos_records_count,
    AVG(months_balance)               AS pos_months_balance_mean,
    
    MAX(sk_dpd)                       AS pos_sk_dpd_max,
    AVG(sk_dpd)                       AS pos_sk_dpd_mean,
    
    MAX(sk_dpd_def)                   AS pos_sk_dpd_def_max,
    AVG(cnt_instalment)               AS pos_cnt_instalment_mean,
    AVG(cnt_instalment_future)        AS pos_cnt_instalment_future_mean,
    SUM(CASE WHEN sk_dpd > 0 THEN 1 ELSE 0 END) AS pos_dpd_count
FROM v_clean_pos_cash
GROUP BY sk_id_curr
"""
        self._exec_sql(sql, "v_agg_pos_cash")

    def _create_view_agg_credit_card(self) -> None:
        self._print("  → v_agg_credit_card ...")
        sql = """
CREATE OR REPLACE VIEW v_agg_credit_card AS
SELECT
    sk_id_curr,
    COUNT(*)                              AS cc_records_count,
    
    AVG(amt_balance)                      AS cc_amt_balance_mean,
    MAX(amt_balance)                      AS cc_amt_balance_max,
    
    AVG(amt_credit_limit_actual)          AS cc_credit_limit_mean,
    
    SUM(amt_drawings_current)             AS cc_amt_drawings_current_sum,
    AVG(amt_drawings_current)             AS cc_amt_drawings_current_mean,
    
    AVG(amt_payment_total_current)        AS cc_payment_total_mean,
    
    MAX(sk_dpd)                           AS cc_sk_dpd_max,
    AVG(sk_dpd)                           AS cc_sk_dpd_mean,
    
    AVG(cnt_drawings_current)             AS cc_cnt_drawings_mean,
    SUM(CASE WHEN sk_dpd > 0 THEN 1 ELSE 0 END) AS cc_dpd_count
FROM v_clean_credit_card
GROUP BY sk_id_curr
"""
        self._exec_sql(sql, "v_agg_credit_card")

    def _create_view_agg_installments(self) -> None:
        self._print("  → v_agg_installments ...")
        sql = """
CREATE OR REPLACE VIEW v_agg_installments AS
SELECT
    sk_id_curr,
    COUNT(*)                              AS install_records_count,
    
    AVG(payment_delay_days)               AS install_payment_delay_mean,
    MAX(payment_delay_days)               AS install_dpd_max,
    
    SUM(CASE WHEN payment_delay_days > 0 THEN 1 ELSE 0 END)
                                          AS install_late_count,
    AVG(payment_ratio)                    AS install_payment_ratio_mean,
    MIN(payment_ratio)                    AS install_payment_ratio_min,
    
    AVG(amt_payment)                      AS install_amt_payment_mean,
    AVG(amt_instalment)                   AS install_amt_instalment_mean,
    -- Ratio paiements en retard
    CASE WHEN COUNT(*) > 0
         THEN CAST(SUM(CASE WHEN payment_delay_days > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)
         ELSE NULL END                    AS install_late_rate
FROM v_clean_installments
GROUP BY sk_id_curr
"""
        self._exec_sql(sql, "v_agg_installments")

    # =========================================================================
    # VUE MAÎTRESSE — jointures LEFT JOIN sur SK_ID_CURR
    # =========================================================================

    def _create_view_master(self) -> None:
        self._print("  → v_master (jointures LEFT JOIN) ...")
        sql = """
CREATE OR REPLACE VIEW v_master AS
SELECT
    -- ── Application (base) ────────────────────────────────────────────────
    app.*,

    -- ── Bureau agrégé ─────────────────────────────────────────────────────
    bur.bureau_credit_count,
    bur.bureau_amt_credit_sum_mean,
    bur.bureau_amt_credit_sum_total,
    bur.bureau_days_credit_max,
    bur.bureau_days_credit_min,
    bur.bureau_amt_credit_sum_debt_mean,
    bur.bureau_amt_credit_sum_debt_total,
    bur.bureau_active_credit_count,
    bur.bureau_closed_credit_count,
    bur.bureau_overdue_count,
    bur.bureau_credit_day_overdue_mean,
    bur.bureau_cnt_prolong_mean,
    bur.bureau_credit_limit_mean,

    -- ── Demandes précédentes agrégées ──────────────────────────────────────
    prv.prev_app_count,
    prv.prev_approved_count,
    prv.prev_refused_count,
    prv.prev_canceled_count,
    prv.prev_amt_credit_mean,
    prv.prev_amt_credit_max,
    prv.prev_amt_annuity_mean,
    prv.prev_days_decision_max,
    prv.prev_rate_down_mean,
    prv.prev_cnt_payment_mean,
    prv.prev_refusal_rate,

    -- ── POS Cash agrégé ────────────────────────────────────────────────────
    pos.pos_records_count,
    pos.pos_months_balance_mean,
    pos.pos_sk_dpd_max,
    pos.pos_sk_dpd_mean,
    pos.pos_sk_dpd_def_max,
    pos.pos_cnt_instalment_mean,
    pos.pos_dpd_count,

    -- ── Carte de crédit agrégée ────────────────────────────────────────────
    cc.cc_records_count,
    cc.cc_amt_balance_mean,
    cc.cc_amt_balance_max,
    cc.cc_credit_limit_mean,
    cc.cc_amt_drawings_current_sum,
    cc.cc_amt_drawings_current_mean,
    cc.cc_payment_total_mean,
    cc.cc_sk_dpd_max,
    cc.cc_sk_dpd_mean,
    cc.cc_dpd_count,

    -- ── Remboursements agrégés ──────────────────────────────────────────────
    ins.install_records_count,
    ins.install_payment_delay_mean,
    ins.install_dpd_max,
    ins.install_late_count,
    ins.install_payment_ratio_mean,
    ins.install_payment_ratio_min,
    ins.install_amt_payment_mean,
    ins.install_amt_instalment_mean,
    ins.install_late_rate

FROM v_clean_application      app
LEFT JOIN v_agg_bureau         bur ON app.sk_id_curr = bur.sk_id_curr
LEFT JOIN v_agg_previous       prv ON app.sk_id_curr = prv.sk_id_curr
LEFT JOIN v_agg_pos_cash       pos ON app.sk_id_curr = pos.sk_id_curr
LEFT JOIN v_agg_credit_card    cc  ON app.sk_id_curr = cc.sk_id_curr
LEFT JOIN v_agg_installments   ins ON app.sk_id_curr = ins.sk_id_curr
"""
        self._exec_sql(sql, "v_master")

    # =========================================================================
    # VUE FEATURE ENGINEERING SQL
    # =========================================================================

    def _create_view_features(self) -> None:
        self._print("  → v_features_engineering (FE SQL) ...")
        sql = """
CREATE OR REPLACE VIEW v_features_engineering AS
SELECT
    m.*,

    -- ── 1. Ratios Financieros (Capacidad de Pago) ─────────────────────────
    
    -- fe1: Carga de la deuda total sobre ingresos
    CASE WHEN amt_income_total > 0 
         THEN amt_credit / amt_income_total ELSE NULL END 
         AS fe1_credit_income_ratio,

    -- fe2: Carga de la cuota anualizada sobre ingresos
    CASE WHEN amt_income_total > 0 
         THEN (amt_annuity * 12) / amt_income_total ELSE NULL END 
         AS fe2_annuity_income_ratio,

    -- fe3: Tasa de amortización (velocidad de pago)
    CASE WHEN amt_credit > 0 
         THEN amt_annuity / amt_credit ELSE NULL END 
         AS fe3_payment_rate,

    -- ── 2. Demografía y Empleo (Estabilidad) ──────────────────────────────
    
    -- fe4: Porcentaje de la vida transcurrido trabajando (limpiando outlier 365243)
    CASE WHEN days_birth <> 0 AND days_employed < 0 
         THEN CAST(days_employed AS FLOAT) / days_birth ELSE NULL END 
         AS fe4_days_employed_ratio,

    -- fe5: Edad cronológica exacta
    ABS(days_birth) / 365.25 
         AS fe5_age_years,

    -- fe6: Disponibilidad económica por miembro de familia
    CASE WHEN cnt_fam_members > 0 
         THEN amt_income_total / cnt_fam_members ELSE NULL END 
         AS fe6_income_per_person,

    -- ── 3. Calificación Externa (Riesgo Predictivo) ────────────────────────
    
    -- fe7: Media aritmética de los scores externos disponibles
    (COALESCE(ext_source_1, 0) + COALESCE(ext_source_2, 0) + COALESCE(ext_source_3, 0))
    / NULLIF(
        (CASE WHEN ext_source_1 IS NOT NULL THEN 1 ELSE 0 END
       + CASE WHEN ext_source_2 IS NOT NULL THEN 1 ELSE 0 END
       + CASE WHEN ext_source_3 IS NOT NULL THEN 1 ELSE 0 END), 0
    )    AS fe7_ext_sources_mean,

    -- fe8: El peor escenario de riesgo (mínimo de los scores)
    LEAST(ext_source_1, ext_source_2, ext_source_3) 
         AS fe8_ext_sources_min,

    -- ── 4. Historial Externo y Comportamiento (Bureau) ─────────────────────
    
    -- fe9: Tasa de morosidad en créditos pasados
    CASE WHEN COALESCE(bureau_credit_count, 0) > 0
         THEN CAST(COALESCE(bureau_overdue_count, 0) AS FLOAT) / bureau_credit_count
         ELSE 0.0 END 
         AS fe9_bureau_overdue_rate,

    -- fe10: Heurística de riesgo combinada (Bureau + External)
    (COALESCE(1.0 - ext_source_2, 0.5)
   + COALESCE(1.0 - ext_source_3, 0.5)
   + COALESCE(CAST(COALESCE(bureau_overdue_count, 0) AS FLOAT) /
              NULLIF(bureau_credit_count, 0), 0.0)) / 3.0 
         AS fe10_composite_risk_score

FROM v_master m;
"""
        self._exec_sql(sql, "v_features_engineering")

    # =========================================================================
    # CHARGEMENT
    # =========================================================================

    def load_master(
        self,
        split: str = "train",
        use_features_view: bool = True,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Charge la vue maîtresse ou feature engineering.

        Args:
            split:              "train", "test" ou "all"
            use_features_view:  True → v_features_engineering, False → v_master
            limit:              Nombre de lignes à charger (None = tout)

        Returns:
            DataFrame prêt pour Phase 2 (preprocessing sklearn)
        """
        view = "v_features_engineering" if use_features_view else "v_master"

        where = ""
        if split in ("train", "test"):
            where = f"WHERE split = '{split}'"

        limit_clause = f"LIMIT {limit}" if limit else ""

        sql = f"SELECT * FROM {view} {where} {limit_clause}"

        self._print(f"  📤 Chargement {view} (split={split}) ...")
        df = self._read_sql(sql)
        self._print(f"     ✅ {len(df):,} lignes · {len(df.columns)} colonnes")
        return df

    def load_train_test(
        self,
        use_features_view: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retourne (df_train, df_test) en un appel.
        df_train contient TARGET, df_test n'en a pas.
        """
        df_train = self.load_master("train", use_features_view)
        df_test  = self.load_master("test",  use_features_view)
        return df_train, df_test

    # =========================================================================
    # UTILS
    # =========================================================================

    def table_exists(self, table_name: str) -> bool:
        """Vérifie si une table ou vue existe dans la DB."""
        sql = f"SELECT name FROM sqlite_master WHERE name='{table_name}'"
        result = self._read_sql(sql)
        return len(result) > 0

    def get_db_summary_bak(self) -> dict:
        """Retourne un résumé des tables et vues présentes."""
        rows = self._read_sql(
            "SELECT type, name FROM sqlite_master "
            "WHERE type IN ('table', 'view') ORDER BY type, name"
        )
        tables = rows[rows["type"] == "table"]["name"].tolist()
        views  = rows[rows["type"] == "view"]["name"].tolist()

        self._print(f"\n📊 DB Summary — {len(tables)} tables · {len(views)} vues")
        self._print(f"  Tables : {tables}")
        self._print(f"  Vues   : {views}")
        return {"tables": tables, "views": views}

    def get_db_summary(self) -> dict:
        """
        Retorna un resumen de tablas y vistas compatible con SQLite y PostgreSQL.
        """
        if self._is_postgres:
            # --- CONSULTA PARA POSTGRESQL ---
            sql = """
                SELECT table_type AS type, table_name AS name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_type, table_name
            """
        else:
            # --- CONSULTA PARA SQLITE ---
            sql = "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type, name"

        df = self._read_sql(sql, "db_summary")
        
        # Traducir tipos para consistencia (Postgres usa 'BASE TABLE' y 'VIEW')
        summary = {"tables": [], "views": []}
        for _, row in df.iterrows():
            t_type = row['type'].lower()
            if 'table' in t_type:
                summary["tables"].append(row['name'])
            elif 'view' in t_type:
                summary["views"].append(row['name'])
                
        return summary